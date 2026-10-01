"""End-to-end dataset assembly (Step 1 deliverable).

Flow
----
    benchmark (Nifty 50)          -> NSE trading calendar (extended at the tail)
    index constituents / panel    -> tradable universe (+ liquidity screen)
    bhavcopy delivery panel       -> delivery %, turnover, traded quantity
    yfinance adjusted OHLCV       -> price / momentum features
    same-day bhavcopy overlay     -> the newest session's OHLC, same evening
    FeatureProvider (from config) -> causal feature matrix

The result is a single long frame keyed on ``(date, symbol)`` that Step 2
(triple-barrier labelling) consumes directly, plus the metadata needed to
reproduce exactly how the universe was chosen.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from swingml.config import AppConfig
from swingml.data.cache import DiskCache
from swingml.data.delivery import make_delivery_provider
from swingml.data.prices import make_price_provider
from swingml.data.universe import (
    Universe,
    apply_membership,
    build_universe,
    trading_days_from_benchmark,
)
from swingml.data.symbol_history import SymbolHistory, load_symbol_history
from swingml.features import make_feature_provider
from swingml.features.actions import detect_corporate_actions

logger = logging.getLogger(__name__)

#: How far past the benchmark's last session the calendar may be extended, in
#: business days. Bounded because a badly stale benchmark cache would otherwise
#: turn into hundreds of bhavcopy requests per build.
MAX_CALENDAR_TAIL_DAYS = 10


def inclusive_end(last_included: dt.date | dt.datetime | str | None) -> str:
    """The ``end`` to pass to a yfinance-backed fetch so the last session is *included*.

    yfinance treats ``end`` as **exclusive**, so passing a session date directly
    silently omits that session's bar. In a backtest the cost is one row at the
    tail and invisible; for a daily forward paper trade it is fatal, because the
    newest session is the entire point. Passing the calendar's last session
    directly would end the dataset one session before its own trading calendar,
    so every ``paper_trade`` freeze would have been permanently one session
    stale, i.e. every prediction would have been a backfill.

    ``None`` means "through today", which is also inclusive.
    """
    if last_included is None:
        last_included = dt.date.today()
    return (pd.Timestamp(last_included) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def extend_calendar_tail(
    trading_days: list[dt.date],
    end_date: dt.date | None,
    *,
    max_days: int = MAX_CALENDAR_TAIL_DAYS,
) -> list[dt.date]:
    """Append the business days that could follow the benchmark's last session.

    Why this exists: the benchmark supplies the calendar, but an index series can
    lag or simply lack its newest session while the exchange has *already*
    published that session's bhavcopy. Without the extension the newest session
    is never even requested, and the freeze looks like "no feature rows" rather
    than "one source is a day behind".

    These are *candidates* only: the bhavcopy provider's ``DATE1`` guard rejects
    every non-trading day before it reaches the panel, so a holiday costs one
    404 and adds nothing.
    """
    if not trading_days or end_date is None:
        return list(trading_days)
    last = pd.Timestamp(trading_days[-1]).normalize()
    stop = pd.Timestamp(end_date).normalize()
    if stop <= last:
        return list(trading_days)
    extra = [d.date() for d in pd.bdate_range(last + pd.Timedelta(days=1), stop)][:max_days]
    if extra:
        logger.info(
            "calendar: benchmark ends %s, extending %d candidate session(s) to %s",
            last.date(), len(extra), extra[-1],
        )
    return list(trading_days) + extra


def apply_symbol_aliases(
    fetched: dict[str, pd.DataFrame],
    history: SymbolHistory,
    universe_symbols: list[str],
    panel: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Recover renamed names by handing them their successor's history.

    A retired ticker has no price series of its own, but it is the *same security*
    as the ticker that replaced it, so the successor's (adjusted) history is the
    right series -- restricted to the sessions the retired symbol was actually
    listed for. The restriction is what keeps a rename from double-counting: a
    symbol's own successor is also in the universe, and the two spans do not
    overlap by construction, because ``symbol_history`` refuses an alias whose
    spans overlap.

    Returns the augmented price dict and ``{retired_symbol: successor}`` for the
    names that were filled, so the caller can report them rather than let a
    recovery happen invisibly.
    """
    spans: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    if panel is not None and not panel.empty:
        p = panel.copy()
        p["date"] = pd.to_datetime(p["date"]).dt.normalize()
        grouped = p.groupby("symbol")["date"]
        spans = {str(s): (g.min(), g.max()) for s, g in grouped}

    def _clip(frame: pd.DataFrame, span: tuple[pd.Timestamp, pd.Timestamp] | None) -> pd.DataFrame:
        if span is None:
            return frame
        lo, hi = span
        return frame.loc[(frame.index >= lo) & (frame.index <= hi)]

    clamped = 0

    # Recovery runs first, on the successor's FULL series: a successor's history
    # reaches back through the rename (Yahoo back-fills the retired name's bars
    # under the new ticker), and the retired symbol's slice lives in exactly that
    # pre-rename part.
    recovered: dict[str, str] = {}
    missing = [s for s in universe_symbols if s not in fetched]
    for sym in missing:
        if not history.is_alias(sym):
            continue
        successor = history.resolve(sym)
        frame = fetched.get(successor)
        if frame is None or frame.empty:
            continue
        span = spans.get(sym)
        # Never reach past the successor's own first observation. A clean rename
        # ends the retired symbol's span there anyway, but a demerger does not:
        # TATAMOTORS kept trading after TMPV inherited the ISIN, so without this
        # clamp the post-split TMPV (passenger-vehicle) series would be spliced
        # in as TATAMOTORS (commercial-vehicle) prices. The clamp is a hard
        # upper bound; the ISIN map's own sampling interval is the slack, so a
        # clamped span is reported rather than taken silently.
        first = history.first_seen.get(successor)
        if span is not None and first:
            limit = pd.Timestamp(first).normalize() - pd.Timedelta(days=1)
            if limit < span[1]:
                clamped += 1
                span = (span[0], limit)
        sliced = _clip(frame, span)
        if sliced.empty:
            continue
        fetched[sym] = sliced.copy()
        recovered[sym] = successor

    # Only then is the successor clipped to its own listing span. Without this,
    # when both names are in the universe the pre-rename sessions would sit under
    # two symbols and be counted twice -- and the invariant would depend on the
    # membership filter being present, which it is not for an index universe.
    targets = {v for v in history.aliases.values() if v in fetched and v in spans}
    for target in targets:
        fetched[target] = _clip(fetched[target], spans[target]).copy()

    if recovered:
        logger.info(
            "symbol history: recovered %d renamed name(s) from their successors: %s",
            len(recovered), sorted(recovered.items())[:8],
        )
    if clamped:
        logger.info(
            "symbol history: %d of them stopped at the successor's first session "
            "(the retired symbol kept trading -- demerger or spin-off, not a rename)",
            clamped,
        )
    return fetched, recovered


@dataclass
class TailSessionOverlay:
    """What :func:`overlay_tail_session` did, so the caller can log and record it."""

    session: pd.Timestamp | None = None
    filled: list[str] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)
    skipped_corporate_action: list[str] = field(default_factory=list)
    skipped_invalid_bar: list[str] = field(default_factory=list)
    absent_from_panel: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "session": str(self.session.date()) if self.session is not None else None,
            "n_filled": len(self.filled),
            "n_already_present": len(self.already_present),
            "n_skipped_corporate_action": len(self.skipped_corporate_action),
            "n_skipped_invalid_bar": len(self.skipped_invalid_bar),
            "n_absent_from_panel": len(self.absent_from_panel),
            "skipped_corporate_action": sorted(self.skipped_corporate_action)[:20],
            "absent_from_panel": sorted(self.absent_from_panel)[:20],
        }


def overlay_tail_session(
    prices: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    session: pd.Timestamp | dt.date | str | None = None,
    *,
    tolerance: float = 0.10,
) -> TailSessionOverlay:
    """Fill the newest session's OHLC from the same-evening bhavcopy, in place.

    The problem this solves: the index series updates promptly while its
    *constituent* bars trickle in for hours, so the newest session's price panel
    is truncated and size-biased -- which is exactly what the paper-trade
    thin-cross-section guard refuses. The exchange's own bhavcopy has every name
    the same evening and is already downloaded for the delivery features.

    Why the bhavcopy and not Breeze quotes, given both can supply the bar: the
    two agree on the close (both are raw/unadjusted). The bhavcopy is one request
    instead of one per symbol, is already a hard dependency of this build, and
    carries the ``PREV_CLOSE`` needed for the guard below.

    Two conventions worth stating:

    * **Only the tail session is touched.** A price bar the provider already has
      is never replaced, so a normal backfill is a no-op and the price family's
      single adjustment basis is preserved everywhere except the newest bar.
    * **A corporate action excludes the name for that session.** The bhavcopy is
      raw; the price history is split-adjusted. If the session is an ex-date the
      new bar is on a different basis from the history, and splicing it would
      fabricate a huge one-day move in every trailing window. The exclusion uses
      the same ``PREV_CLOSE`` restatement detector that cleaned the quantity
      features (1 ex-date per ~1,400 sessions), so it costs ~0 names per session.
      The name is reported, not silently dropped.

    ``volume`` is filled from the bhavcopy's traded quantity for completeness.
    No feature reads it (the volume block uses the delivery panel), and the raw
    quantity is the same basis as the close spliced in beside it.
    """
    report = TailSessionOverlay()
    if not prices or panel is None or panel.empty:
        return report

    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    target = pd.Timestamp(session).normalize() if session is not None else p["date"].max()
    report.session = target

    today = p[p["date"] == target]
    if today.empty:
        return report
    today = today.drop_duplicates(subset=["symbol"], keep="last").set_index("symbol")

    prior_close = (
        p[p["date"] < target]
        .sort_values(["symbol", "date"])
        .groupby("symbol")["close"]
        .last()
        .reindex(today.index)
    )
    # Same detector, same convention as the quantity repair -- one definition of
    # "ex-date" in the codebase.
    is_action, _ = detect_corporate_actions(
        today["prev_close"].astype(float), prior_close.astype(float), tolerance=tolerance
    )
    actions = set(today.index[is_action.fillna(False)])

    for sym, px in prices.items():
        if px is None or px.empty:
            report.absent_from_panel.append(sym)
            continue
        if target in px.index:
            report.already_present.append(sym)
            continue
        if sym not in today.index:
            report.absent_from_panel.append(sym)
            continue
        if sym in actions:
            report.skipped_corporate_action.append(sym)
            continue

        row = today.loc[sym]
        bar = {
            "open": row.get("open"),
            "high": row.get("high"),
            "low": row.get("low"),
            "close": row.get("close"),
            "volume": row.get("ttl_trd_qnty", np.nan),
        }
        def _num(value) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")

        values = {k: _num(v) for k, v in bar.items()}
        if not all(np.isfinite(values[c]) for c in ("open", "high", "low", "close")) \
                or values["close"] <= 0:
            report.skipped_invalid_bar.append(sym)
            continue

        frame = pd.DataFrame(
            {k: [v] for k, v in values.items()},
            index=pd.DatetimeIndex([target], name=px.index.name or "date"),
        )
        prices[sym] = pd.concat([px, frame]).sort_index()
        report.filled.append(sym)
    return report


@dataclass
class DatasetBundle:
    """A built dataset plus everything needed to reproduce it."""

    features: pd.DataFrame
    universe: Universe
    meta: dict = field(default_factory=dict)

    def save(self, dataset_dir: str | Path) -> Path:
        out = Path(dataset_dir)
        out.mkdir(parents=True, exist_ok=True)
        fpath = out / "features.parquet"
        self.features.to_parquet(fpath, index=False)

        (out / "universe.json").write_text(
            json.dumps(
                {"source": self.universe.source, "symbols": self.universe.symbols},
                indent=2,
            ),
            encoding="utf-8",
        )
        if self.universe.membership is not None and not self.universe.membership.empty:
            self.universe.membership.to_parquet(out / "membership.parquet", index=False)
        (out / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")
        logger.info("dataset written -> %s (%d rows, %d cols)", fpath, *self.features.shape)
        return fpath

    @staticmethod
    def load(dataset_dir: str | Path) -> tuple[pd.DataFrame, dict]:
        """Load a previously built feature matrix and its metadata."""
        d = Path(dataset_dir)
        features = pd.read_parquet(d / "features.parquet")
        meta_path = d / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return features, meta


def load_training_frame(cfg: AppConfig, regime_col: str) -> tuple[pd.DataFrame, list[str]]:
    """Features joined to labels on ``(date, symbol)``, plus the model features.

    Shared by every experiment script, so no two of them can disagree about what
    a training row is. Two conventions are load-bearing:

    * **Join, never concatenate.** Labels are absent for the last ``horizon``
      bars of each symbol by construction, so concatenating the two artifacts
      misaligns every row after the first missing one.
    * ``regime_col`` is validated here rather than trusted, because a model
      silently fitted without the regime block is a different experiment.
    """
    feats, meta = DatasetBundle.load(cfg.paths.dataset_dir)
    labels = pd.read_parquet(Path(cfg.paths.dataset_dir) / "labels.parquet")

    feature_cols = [c for c in (meta.get("feature_columns") or []) if c in feats.columns]
    if not feature_cols:
        raise RuntimeError("no model feature columns found in meta.json")
    if regime_col not in feature_cols:
        raise RuntimeError(f"{regime_col} missing from the feature set; cannot run this test")

    df = feats.merge(labels, on=["date", "symbol"], how="inner")
    # Binary buy-signal target: will the profit barrier be reached?
    df["target"] = (df["label"] == 1).astype(int)
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    return df, feature_cols


def build_dataset(
    cfg: AppConfig,
    force: bool = False,
    symbols: list[str] | None = None,
    max_days: int | None = None,
) -> DatasetBundle:
    """Build the feature matrix for the configured window.

    Parameters
    ----------
    force
        Ignore the parquet caches and re-download everything.
    symbols
        Explicit symbol override (skips universe resolution's index fetch).
    max_days
        Only use the most recent N trading sessions. Useful for a fast smoke
        run before committing to a full multi-year backfill.
    """
    cfg.ensure_dirs()
    logger.info(
        "building dataset | window=%s..%s | price=%s | delivery=%s | universe=%s",
        cfg.data.start, cfg.data.end or "today", cfg.data.price_provider,
        cfg.data.delivery_provider, cfg.universe.source,
    )

    price_cache = DiskCache(cfg.paths.cache_dir, "prices")
    deliv_cache = DiskCache(cfg.paths.cache_dir, "delivery")
    universe_cache = DiskCache(cfg.paths.cache_dir, "universe")

    prices = make_price_provider(cfg.data, price_cache)
    delivery = make_delivery_provider(cfg.data, deliv_cache)

    # -- 1. benchmark -> trading calendar ---------------------------------
    # ``cfg.data.end`` is inclusive in config terms, as is the last session of
    # the calendar passed further down; both are handed over through
    # ``inclusive_end`` so the provider's exclusive semantics cannot truncate.
    bench = prices.get_benchmark(cfg.data.start, inclusive_end(cfg.data.end))
    if bench is None or bench.empty:
        raise RuntimeError(
            f"benchmark '{cfg.data.bench_symbol}' returned no data; "
            "cannot derive the NSE trading calendar"
        )
    trading_days = trading_days_from_benchmark(bench)
    if max_days:
        trading_days = trading_days[-max_days:]
    logger.info("trading calendar: %d sessions (%s .. %s)", len(trading_days), trading_days[0], trading_days[-1])

    # -- 2. pre-resolve index symbols (needed to bound the delivery fetch) --
    # Mock mode stays strictly offline: the mock providers mint their own
    # coherent symbol set rather than reaching for the real index list.
    using_mock = cfg.data.price_provider == "mock" or cfg.data.delivery_provider == "mock"
    pre_symbols: list[str] | None = None
    if symbols:
        pre_symbols = [s.strip().upper() for s in symbols]
    elif not using_mock and cfg.universe.source in {"nifty200", "nifty500"}:
        idx_name = "NIFTY 500" if cfg.universe.source == "nifty500" else cfg.universe.index_name
        from swingml.data.delivery import fetch_index_constituents

        cons = fetch_index_constituents(idx_name, cache=universe_cache, force=force)
        if not cons.empty:
            pre_symbols = sorted({str(s).strip().upper() for s in cons["Symbol"]})

    # -- 3. delivery panel -------------------------------------------------
    # The benchmark is not the last word on which sessions traded: an index
    # series can lag the exchange, so the tail is offered as candidate days and
    # the bhavcopy's own DATE1 guard decides which are real. Those dates then
    # join the calendar below, because a session the exchange published is a
    # session whether or not the benchmark mentioned it.
    requested_days = extend_calendar_tail(trading_days, cfg.data.end_date)
    panel = delivery.get_delivery(
        requested_days[0], requested_days[-1],
        symbols=pre_symbols,
        trading_days=requested_days,
        force=force,
    )
    if panel.empty:
        raise RuntimeError("delivery panel is empty; check the NSE source or use mock providers")
    panel_dates = sorted({pd.Timestamp(d).date() for d in pd.to_datetime(panel["date"])})
    added = [d for d in panel_dates if d not in set(trading_days)]
    if added:
        logger.info("calendar: +%d session(s) present in the bhavcopy but not the benchmark", len(added))
        trading_days = sorted(set(trading_days).union(panel_dates))
    logger.info(
        "delivery panel: %d rows | %d symbols | %d sessions",
        len(panel), panel["symbol"].nunique(), panel["date"].nunique(),
    )

    # -- 4. universe -------------------------------------------------------
    from dataclasses import replace

    if symbols or cfg.universe.symbols:
        universe = build_universe(replace(cfg.universe, symbols=pre_symbols or cfg.universe.symbols), cache=universe_cache, panel=panel)
    elif using_mock:
        universe = Universe(symbols=delivery.default_symbols, source="mock")
        logger.info("universe: %d synthetic symbols (offline mock mode)", len(universe.symbols))
    else:
        # Resolve from the official list and apply the liquidity screen. The
        # network fetch is a cache hit here, so provenance stays accurate
        # ('nifty200') instead of being downgraded to 'explicit'.
        universe = build_universe(cfg.universe, cache=universe_cache, panel=panel)
    if not universe.symbols:
        raise RuntimeError("resolved universe is empty")

    # -- 5. prices ---------------------------------------------------------
    # A retired ticker has no series under its own name, so ask for the ticker
    # that replaced it and slice it back to the retired symbol's own listing
    # span afterwards. Without this the feature matrix silently drops the
    # universe's renamed names.
    history: SymbolHistory | None = None
    if cfg.data.symbol_history:
        try:
            history = load_symbol_history(
                cfg.data.symbol_history_path, days=trading_days
            )
        except Exception as exc:  # noqa: BLE001 - never fail a build over this
            logger.warning(
                "symbol history unavailable (%s); renamed names will be missing", exc,
            )

    want = sorted({history.resolve(s) if history else s for s in universe.symbols})
    fetched = prices.get_many(want, trading_days[0], inclusive_end(trading_days[-1]))
    logger.info("prices: %d/%d symbols returned data", len(fetched), len(want))
    if not fetched:
        raise RuntimeError("no price data returned for any universe symbol")

    recovered: dict[str, str] = {}
    if history is not None:
        fetched, recovered = apply_symbol_aliases(fetched, history, universe.symbols, panel)
        if recovered:
            logger.info("prices: %d/%d universe symbols now have a series",
                        sum(s in fetched for s in universe.symbols), len(universe.symbols))

    # -- 5b. same-day overlay ---------------------------------------------
    # An adjusted history plus the newest raw bar only mixes bases on an ex-date,
    # which is exactly what the overlay's corporate-action guard excludes.
    overlay = TailSessionOverlay()
    if cfg.data.sameday_overlay:
        overlay = overlay_tail_session(fetched, panel, session=trading_days[-1])
        logger.info(
            "same-day overlay (%s): %d filled, %d already present, %d ex-date, "
            "%d invalid, %d absent from the panel",
            overlay.session.date() if overlay.session is not None else "n/a",
            len(overlay.filled), len(overlay.already_present),
            len(overlay.skipped_corporate_action), len(overlay.skipped_invalid_bar),
            len(overlay.absent_from_panel),
        )
        if overlay.skipped_corporate_action:
            logger.warning(
                "same-day overlay: %d name(s) are ex-date on %s and were left out of that "
                "session (raw bhavcopy bar vs adjusted history): %s",
                len(overlay.skipped_corporate_action), overlay.session.date()
                if overlay.session is not None else "n/a",
                sorted(overlay.skipped_corporate_action)[:10],
            )

    # -- 6. features -------------------------------------------------------
    # The provider is resolved from config, so the pipeline is agnostic to which
    # feature set is in use. See swingml.features.registry.
    provider = make_feature_provider(cfg.features)
    features = provider.transform(fetched, panel, bench=bench)

    # Enforce point-in-time membership (liquidity universe only).
    if universe.membership is not None and not universe.membership.empty:
        before = len(features)
        features = apply_membership(features, universe.membership)
        logger.info("point-in-time filter: %d -> %d rows", before, len(features))

    # Post-split sanity: prices must be strictly positive and finite.
    features = features[features["close"] > 0].copy()
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()

    meta = {
        "built_at": dt.datetime.now().isoformat(timespec="seconds"),
        "start": str(trading_days[0]),
        "end": str(trading_days[-1]),
        "n_sessions": len(trading_days),
        "universe_source": universe.source,
        "n_symbols_requested": len(universe.symbols),
        "n_symbols_with_data": len(fetched),
        "n_rows": int(len(features)),
        "price_provider": cfg.data.price_provider,
        "delivery_provider": cfg.data.delivery_provider,
        "round_trip_cost_pct": cfg.costs.round_trip_cost_pct,
        "label": {
            "horizon_days": cfg.label.horizon_days,
            "pt_atr_mult": cfg.label.pt_atr_mult,
            "sl_atr_mult": cfg.label.sl_atr_mult,
            "atr_window": cfg.label.atr_window,
        },
        "sameday_overlay": overlay.as_dict(),
        "symbol_aliases": {
            "n_aliases_in_map": len(history.aliases) if history is not None else 0,
            "n_recovered": len(recovered),
            "recovered": dict(sorted(recovered.items())[:50]),
        },
        "n_features": len(provider.feature_columns),
        "feature_columns": provider.feature_columns,
        # Record both the short name and the fully-resolved spec, so the dataset
        # can be re-grouped later even if the config has since changed.
        "feature_provider": provider.name,
        "feature_provider_spec": cfg.features.provider,
    }
    return DatasetBundle(features=features, universe=universe, meta=meta)
