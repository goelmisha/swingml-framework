"""Tradable universe construction.

Two modes, and the difference matters for the integrity of the scorecard:

``nifty200`` / ``nifty500``
    The official *current* NSE constituent list. Convenient, but it applies
    today's membership to 2020 history -- textbook **survivorship bias**: names
    that were dropped from the index for being weak are excluded, so backtest
    results look better than reality. Fine for feature iteration; not for
    believing a Sharpe ratio.

``liquidity``
    Point-in-time membership: on each date, take the top-N cash equities by
    *trailing* turnover computed from the bhavcopy itself. Membership at time
    ``t`` uses only data up to ``t``, so it is survivorship-bias free and can be
    reproduced. This is the mode to trust for the final evaluation.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from swingml.config import UniverseConfig
from swingml.data.cache import DiskCache
from swingml.data.delivery import fetch_index_constituents

logger = logging.getLogger(__name__)


@dataclass
class Universe:
    """A resolved set of tradable symbols plus its provenance."""

    symbols: list[str]
    source: str
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Point-in-time membership (``liquidity`` mode only): columns
    #: ``[date, symbol]``. Downstream steps must filter rows to dates on which
    #: the symbol was actually a member.
    membership: pd.DataFrame | None = None

    def __len__(self) -> int:
        return len(self.symbols)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Universe(n={len(self.symbols)}, source={self.source!r})"


def trading_days_from_benchmark(bench: pd.DataFrame) -> list[dt.date]:
    """Derive the authoritative NSE trading calendar from the index series.

    Using the benchmark's own dates means we only ever request bhavcopy files
    for real sessions -- roughly 250 requests/year instead of 365.
    """
    if bench is None or bench.empty:
        return []
    idx = pd.to_datetime(bench.index)
    return sorted({ts.date() for ts in idx})


def _normalise_symbols(symbols) -> list[str]:
    return sorted({str(s).strip().upper() for s in symbols if str(s).strip()})


def liquidity_screen(
    panel: pd.DataFrame,
    min_avg_turnover_lacs: float,
    min_price: float,
    lookback_sessions: int = 60,
) -> list[str]:
    """Keep symbols passing a trailing liquidity and price floor.

    Both thresholds are measured on the bhavcopy's *own* trailing window, so the
    screen is causal (it never peeks at the full sample to decide membership).
    """
    if panel is None or panel.empty:
        return []
    df = panel.sort_values("date")
    tail = df.groupby("symbol", observed=True).tail(lookback_sessions)
    stats = tail.groupby("symbol", observed=True).agg(
        med_turnover=("turnover_lacs", "median"),
        med_close=("close", "median"),
        n=("close", "size"),
    )
    keep = stats[(stats["med_turnover"] >= min_avg_turnover_lacs) & (stats["med_close"] >= min_price)]
    out = _normalise_symbols(keep.index)
    logger.info(
        "liquidity screen kept %d/%d symbols (turnover>=%.0f lacs, price>=%.1f)",
        len(out), stats.shape[0], min_avg_turnover_lacs, min_price,
    )
    return out


def point_in_time_membership(
    panel: pd.DataFrame,
    top_n: int,
    lookback_sessions: int = 60,
) -> pd.DataFrame:
    """Per-date top-N membership by *trailing* turnover (survivorship-free).

    Returns a frame with columns ``[date, symbol, adv_lacs, rank]``. The rolling
    mean is computed per symbol on its own history and then right-aligned, so on
    date ``t`` only sessions ``<= t`` are used.
    """
    if panel is None or panel.empty:
        return pd.DataFrame(columns=["date", "symbol", "adv_lacs", "rank"])

    df = panel[["date", "symbol", "turnover_lacs"]].dropna().sort_values(["symbol", "date"]).copy()
    df["adv_lacs"] = (
        df.groupby("symbol", observed=True)["turnover_lacs"]
        .transform(lambda s: s.rolling(lookback_sessions, min_periods=max(5, lookback_sessions // 4)).mean())
    )
    df = df.dropna(subset=["adv_lacs"])
    df["rank"] = df.groupby("date")["adv_lacs"].rank(ascending=False, method="first")
    members = df[df["rank"] <= top_n].copy()
    logger.info(
        "point-in-time universe: %d sessions, %d distinct symbols, median members/session=%.0f",
        members["date"].nunique(), members["symbol"].nunique(),
        members.groupby("date")["symbol"].size().median(),
    )
    return members.reset_index(drop=True)


def build_universe(
    cfg: UniverseConfig,
    cache: DiskCache | None = None,
    panel: pd.DataFrame | None = None,
) -> Universe:
    """Resolve the configured universe.

    ``panel`` (the delivery/bhavcopy panel) is required for the ``liquidity``
    source and optional for the index sources, where it is used only to apply
    the price/liquidity floor.
    """
    if cfg.symbols:
        syms = _normalise_symbols(cfg.symbols)
        logger.info("universe: %d explicit symbols from config", len(syms))
        return Universe(symbols=syms, source="explicit")

    source = cfg.source.lower()

    if source in {"nifty200", "nifty500"}:
        idx_name = "NIFTY 500" if source == "nifty500" else cfg.index_name
        try:
            constituents = fetch_index_constituents(idx_name, cache=cache)
        except Exception as exc:
            logger.error("failed to fetch %s constituents: %s", idx_name, exc)
            constituents = pd.DataFrame()
        if constituents.empty:
            raise RuntimeError(
                f"could not obtain {idx_name} constituents and no explicit symbols configured"
            )
        meta = constituents.rename(columns={"Symbol": "symbol", "Company Name": "company", "Industry": "industry"})
        meta = meta.drop_duplicates(subset=["symbol"])
        # Only cash-market equities; funds/derivative series are not tradable here.
        if "Series" in meta.columns:
            meta = meta[meta["Series"].astype(str).str.upper().isin(["EQ", "BE", "BZ"])]
        syms = _normalise_symbols(meta["symbol"])

        if panel is not None and not panel.empty:
            passing = set(liquidity_screen(panel, cfg.min_avg_turnover_lacs, cfg.min_price))
            before = len(syms)
            syms = [s for s in syms if s in passing]
            logger.info("index '%s': %d -> %d symbols after liquidity screen", idx_name, before, len(syms))

        logger.info("universe: %d symbols from '%s' (survivorship-biased mode)", len(syms), idx_name)
        return Universe(symbols=syms, source=source, meta=meta.reset_index(drop=True))

    if source == "liquidity":
        if panel is None or panel.empty:
            raise ValueError("universe.source='liquidity' requires the delivery panel")
        members = point_in_time_membership(panel, cfg.universe_size)
        if members.empty:
            raise RuntimeError("point-in-time membership is empty; is the panel too short?")
        syms = _normalise_symbols(members["symbol"])
        logger.info("universe: %d symbols (point-in-time, survivorship-free)", len(syms))
        return Universe(symbols=syms, source=source, membership=members)

    raise ValueError(f"unknown universe.source: {cfg.source!r}")


def apply_membership(df: pd.DataFrame, membership: pd.DataFrame | None) -> pd.DataFrame:
    """Restrict a (date, symbol) dataset to point-in-time members.

    The frame must carry ``date`` and ``symbol`` as COLUMNS (the pipeline
    convention); the filtered frame keeps that shape. An earlier version
    reset the index and returned ``date`` as the index, which broke every
    downstream column access -- caught by test, never by a run, because the
    liquidity path had never been executed end-to-end.
    """
    if membership is None or membership.empty:
        return df
    if "date" not in df.columns or "symbol" not in df.columns:
        raise ValueError("apply_membership expects 'date' and 'symbol' columns")
    keys = membership[["date", "symbol"]].drop_duplicates().copy()
    keys["date"] = pd.to_datetime(keys["date"]).dt.normalize()
    left = df.copy()
    left["date"] = pd.to_datetime(left["date"]).dt.normalize()
    merged = left.merge(keys[["date", "symbol"]], on=["date", "symbol"], how="inner")
    return merged.sort_values(["date", "symbol"]).reset_index(drop=True)
