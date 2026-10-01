"""Price (OHLCV) providers.

Two hard requirements drive this module:

1. **Split/bonus correctness.** All return- and momentum-based features are
   computed on *adjusted* prices (``auto_adjust=True``). A raw NSE close series
   contains -40% "crashes" on bonus/split ex-dates that are pure artefacts and
   would poison every label.

2. **Offline determinism.** A ``MockPriceProvider`` backed by a seeded RNG lets
   the whole pipeline (and its tests) run with no network, reproducibly.

Volume note: yfinance volume is split-adjusted while the bhavcopy volume used
for delivery features is raw. The two are therefore NEVER divided by each
other -- see :mod:`swingml.data.delivery` and the split diagnostic in
:mod:`swingml.features`.
"""

from __future__ import annotations

import abc
import logging
import os
import time
import zlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from swingml.config import DataConfig
from swingml.data.breeze import (
    API_KEY_ENV,
    DEFAULT_SESSION_PATH,
    SECRET_KEY_ENV,
    load_env_file,
    load_session_token,
)
from swingml.data.breeze_master import DEFAULT_MASTER_PATH, BreezeSymbolMap, load_symbol_map
from swingml.data.cache import DiskCache

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
NSE_SUFFIX = ".NS"


def normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, order and validate a provider's OHLCV frame.

    Guarantees the contract every downstream consumer relies on:
    a tz-naive, sorted, duplicate-free ``DatetimeIndex`` named ``date`` with
    float ``open/high/low/close`` and float ``volume``.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    out = df.copy()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing column(s): {missing}")
    out = out[OHLCV_COLUMNS].astype("float64")

    # Normalise the index to tz-naive daily timestamps.
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "date"

    out = out[~out.index.duplicated(keep="last")].sort_index()
    # Bars with no traded volume are non-trading artefacts; keep price, drop
    # the row only when the whole bar is unusable.
    out = out.dropna(subset=["close"])
    return out


class PriceProvider(abc.ABC):
    """Abstract source of daily OHLCV bars."""

    def __init__(self, cfg: DataConfig, cache: DiskCache | None = None) -> None:
        self.cfg = cfg
        self.cache = cache

    @abc.abstractmethod
    def get_ohlcv(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Daily bars for ``symbol`` (bare NSE symbol, no exchange suffix)."""

    def get_many(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch several symbols. Overridden by providers that can batch."""
        out: dict[str, pd.DataFrame] = {}
        for i, sym in enumerate(symbols, 1):
            try:
                df = self.get_ohlcv(sym, start, end)
                if not df.empty:
                    out[sym] = df
            except Exception as exc:
                logger.warning("[%d/%d] %s failed: %s", i, len(symbols), sym, exc)
        return out

    def get_benchmark(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Market-regime series (Nifty 50 by default)."""
        symbol = getattr(self, "bench_symbol", "^NSEI")
        return self.get_ohlcv(symbol, start, end)

    # -- helpers -----------------------------------------------------------
    def _window(self, start: str | None, end: str | None) -> tuple[str, str]:
        s = start or self.cfg.start
        e = end or self.cfg.end or pd.Timestamp.today().strftime("%Y-%m-%d")
        return str(s), str(e)


class YFinancePriceProvider(PriceProvider):
    """Primary source: Yahoo Finance daily bars for NSE symbols."""

    bench_symbol = "^NSEI"

    def __init__(self, cfg: DataConfig, cache: DiskCache | None = None) -> None:
        super().__init__(cfg, cache)
        self.bench_symbol = cfg.bench_symbol

    @staticmethod
    def to_yahoo_symbol(symbol: str) -> str:
        """Map a bare NSE symbol to its Yahoo ticker (``RELIANCE`` -> ``RELIANCE.NS``)."""
        s = symbol.strip().upper()
        if s.startswith("^") or "." in s:
            return s  # already an index or fully-qualified ticker
        return f"{s}{NSE_SUFFIX}"

    def get_ohlcv(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        start, end = self._window(start, end)
        ysym = self.to_yahoo_symbol(symbol)

        def _build() -> pd.DataFrame | None:
            raw = self._download([ysym], start, end)
            df = self._extract(raw, ysym)
            return normalise_ohlcv(df)

        if self.cache is None:
            return normalise_ohlcv(self._extract(self._download([ysym], start, end), ysym))
        key = f"px_{ysym}_{start}_{end}"
        res = self.cache.get_or_build(key, _build)
        return res if res is not None else pd.DataFrame(columns=OHLCV_COLUMNS)

    def get_many(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        chunk_size: int = 40,
    ) -> dict[str, pd.DataFrame]:
        """Batched download -- ~40x fewer requests than one-per-symbol."""
        start, end = self._window(start, end)
        out: dict[str, pd.DataFrame] = {}

        # Serve whatever is already cached; only fetch the remainder.
        pending: list[str] = []
        for sym in symbols:
            ysym = self.to_yahoo_symbol(sym)
            cached = self.cache.read(f"px_{ysym}_{start}_{end}") if self.cache else None
            if cached is not None and not cached.empty:
                out[sym] = cached
            else:
                pending.append(sym)
        if out:
            logger.info("price cache hit for %d/%d symbols", len(out), len(symbols))
        if not pending:
            return out

        for i in range(0, len(pending), chunk_size):
            chunk = pending[i : i + chunk_size]
            ytickers = [self.to_yahoo_symbol(s) for s in chunk]
            logger.info("downloading %d symbols (%d/%d)", len(chunk), i + len(chunk), len(pending))
            try:
                raw = self._download(ytickers, start, end)
            except Exception as exc:
                logger.error("batch download failed: %s", exc)
                continue

            for sym, ysym in zip(chunk, ytickers):
                try:
                    df = normalise_ohlcv(self._extract(raw, ysym))
                except Exception as exc:
                    logger.warning("%s: normalise failed (%s)", sym, exc)
                    continue
                if df.empty:
                    logger.warning("%s: no data returned", sym)
                    continue
                if self.cache:
                    self.cache.write(f"px_{ysym}_{start}_{end}", df)
                out[sym] = df
        return out

    # -- internals ---------------------------------------------------------
    def _download(self, ytickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Download with bounded retries; yfinance raises on throttling."""
        import yfinance as yf

        last_exc: Exception | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                df = yf.download(
                    ytickers if len(ytickers) > 1 else ytickers[0],
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=True,   # adjusted OHLC -- mandatory for returns
                    actions=False,
                    progress=False,
                    threads=len(ytickers) > 1,
                    group_by="ticker",
                    timeout=self.cfg.request_timeout_sec,
                )
                if df is not None and not df.empty:
                    return df
                raise RuntimeError("empty frame")
            except Exception as exc:
                last_exc = exc
                backoff = min(2.0 ** attempt, 15.0)
                logger.warning("download attempt %d/%d failed (%s); retrying in %.1fs",
                               attempt, self.cfg.max_retries, exc, backoff)
                time.sleep(backoff)
        raise RuntimeError(f"yfinance download failed after {self.cfg.max_retries} attempts: {last_exc}")

    @staticmethod
    def _extract(raw: pd.DataFrame, ysym: str) -> pd.DataFrame:
        """Pull one ticker's frame out of yfinance's (field, ticker) layout."""
        if raw is None or raw.empty:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        if not isinstance(raw.columns, pd.MultiIndex):
            return raw  # single-ticker download: plain columns

        lvl0 = set(raw.columns.get_level_values(0))
        if ysym in lvl0:
            return raw[ysym]
        lvl1 = set(raw.columns.get_level_values(1))
        if ysym in lvl1:
            return raw.xs(ysym, axis=1, level=1)
        raise KeyError(f"{ysym} absent from downloaded frame")


class MockPriceProvider(PriceProvider):
    """Deterministic synthetic OHLCV -- no network, reproducible.

    Prices follow a geometric random walk with a mild regime/trend component so
    that momentum features and barrier labels have something real to bite on.
    """

    bench_symbol = "NIFTY50"

    def __init__(self, cfg: DataConfig, cache: DiskCache | None = None) -> None:
        super().__init__(cfg, cache)
        self.n_symbols = cfg.mock.n_symbols
        self.n_days = cfg.mock.n_days
        self.seed = cfg.mock.seed

    def _rng(self, symbol: str) -> np.random.Generator:
        """Per-symbol seed -> same series every run, independent of call order."""
        return np.random.default_rng(self.seed + zlib.crc32(symbol.encode()) % 100_000)

    def get_ohlcv(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        rng = self._rng(symbol)
        s = pd.Timestamp(start or self.cfg.start)
        # Extend far enough that any requested window is covered.
        e = pd.Timestamp(end) if end else s + pd.Timedelta(days=int(self.n_days * 1.45) + 400)
        dates = pd.bdate_range(s, e)
        if len(dates) > self.n_days:
            dates = dates[-self.n_days :]
        n = len(dates)

        # Trend regime: piecewise drift so the regime filter sees real swings.
        regime = np.repeat(rng.choice([0.0009, -0.0006, 0.0], size=3, p=[0.4, 0.3, 0.3]), int(np.ceil(n / 3)))[:n]
        vol = rng.uniform(0.010, 0.028)
        rets = rng.normal(regime, vol, size=n)
        close = 100.0 * np.exp(np.cumsum(rets))

        # Build OHLC consistent with the close path.
        prev_close = np.concatenate([[close[0]], close[:-1]])
        span = np.abs(rng.normal(0, vol, size=n)) * close
        high = np.maximum(close, prev_close) + span
        low = np.minimum(close, prev_close) - span
        open_ = prev_close * (1.0 + rng.normal(0, vol / 3, size=n))
        low = np.minimum(low, np.minimum(open_, close))
        high = np.maximum(high, np.maximum(open_, close))

        base_vol = rng.uniform(2e5, 5e6)
        volume = base_vol * np.exp(rng.normal(0, 0.45, size=n)) * (1.0 + 6.0 * np.abs(rets))

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
        df.index.name = "date"
        return normalise_ohlcv(df)

    def get_many(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        return {s: self.get_ohlcv(s, start, end) for s in symbols}

    def get_benchmark(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        return self.get_ohlcv("NIFTY50", start, end)


class BreezeApiError(RuntimeError):
    """Breeze answered definitively with an error -- retrying cannot help."""


def _breeze_frame(raw, symbol: str) -> pd.DataFrame:
    """Coerce whatever the Breeze SDK hands back into a plain OHLCV frame.

    The v2 SDK returns the raw REST dict, not a DataFrame. Two shapes matter:

    * ``{"Success": [records], "Status": 200, "Error": None}`` -- the good path;
      an EMPTY list is legitimate (a weekend, a holiday, a range with no
      sessions) and must NOT be retried.
    * ``{"Success": "", "Status": 500, "Error": "<reason>"}`` -- how the SDK
      reports bad parameters, a rejected symbol and auth failures. **It does not
      raise**; the error arrives as data. Left unhandled this becomes an empty
      frame, and a rejected symbol silently looks like a delisted one.
    """
    if raw is None:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    if isinstance(raw, pd.DataFrame):
        df = raw
    elif isinstance(raw, dict):
        err = raw.get("Error") or raw.get("error")
        if err:
            raise BreezeApiError(f"Breeze rejected {symbol}: {err}")
        rows = next((raw[k] for k in ("Success", "success", "data") if k in raw), None)
        if rows is None:
            raise BreezeApiError(
                f"Breeze returned no 'Success' payload for {symbol}: {sorted(raw)[:6]}"
            )
        if isinstance(rows, str):
            raise BreezeApiError(
                f"Breeze returned a string 'Success' payload for {symbol}: {raw}"
            )
        df = pd.DataFrame(rows if rows else [])
    elif isinstance(raw, (list, tuple)):
        df = pd.DataFrame(list(raw))
    else:
        raise BreezeApiError(
            f"unrecognised Breeze payload for {symbol}: {type(raw).__name__}"
        )
    if df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    stamp = next((c for c in ("datetime", "date", "time") if c in df.columns), None)
    if stamp is None:
        raise BreezeApiError(
            f"Breeze payload for {symbol} has no timestamp column: {sorted(df.columns)}"
        )
    return df.set_index(pd.to_datetime(df[stamp]))


class BreezePriceProvider(PriceProvider):
    """Same-day-complete OHLCV from the broker feed, replacing yfinance.

    Why it exists: yfinance's newest session is still filling in hours after the
    close, so a same-day freeze ranks a truncated, size-biased universe and is
    correctly refused. Breeze is the broker's own feed, so the session is
    complete when the evening freeze runs.

    Three design points that matter operationally:

    * **The ``stock_code`` is not the NSE symbol.** Breeze answers
      ``Success: []`` -- with HTTP 200 and no error -- for a symbol it does not
      recognise under that exact code, so ``RELIANCE`` reads as a delisted
      series while ``RELIND`` returns its full history. The NSE symbol is
      translated through ICICI's security master (:mod:`swingml.data.breeze_master`)
      before every call, because most liquid names carry a different code.
    * **Incremental, not range-keyed.** Prices are cached per symbol and only the
      missing tail is fetched. The yfinance path keys its cache on the date
      range, so each new session forces a full refetch -- fine for ~30 seconds
      of Yahoo, not fine for ~460 rate-limited Breeze calls every evening.
      The cache key is the *NSE symbol*, so it survives a mapping change.
    * **Adjusted prices are NOT assumed.** Breeze daily bars are documented as
      raw; a large close-to-close jump is logged loudly, because every feature and
      label here assumes adjusted prices. Verify against the bhavcopy
      corporate-action detector before trusting a backfill.

    Credentials: ``BREEZE_API_KEY`` / ``BREEZE_SECRET_KEY`` from the environment
    (never stored), the session token from the gitignored token file. The App Key
    is static; only the token rotates daily.
    """

    def __init__(
        self,
        cfg: DataConfig,
        cache: DiskCache | None = None,
        client=None,
        symbol_map: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(cfg, cache)
        self.bench_symbol = getattr(cfg, "breeze_bench_symbol", "NIFTY")
        self.session_path = getattr(cfg, "breeze_session_path", DEFAULT_SESSION_PATH)
        self.master_path = getattr(cfg, "breeze_master_path", DEFAULT_MASTER_PATH)
        self._client = client
        #: Injected in tests; otherwise loaded lazily from the cached master.
        self._symbol_map: Mapping[str, str] | None = symbol_map
        self._unmapped: set[str] = set()
        self._warned_unadjusted: set[str] = set()

    # -- symbol mapping ----------------------------------------------------
    def symbol_map(self) -> Mapping[str, str]:
        """The NSE-symbol -> Breeze-code map, loaded once on first use.

        A failure here degrades to the NSE symbol plus a warning rather than
        killing the build: some codes are identical and an unknown one must not
        abort a 460-symbol refresh. It must never be *silent*, though -- a
        symbol the master does not know returns an empty series that otherwise
        reads as delisted.
        """
        if self._symbol_map is None:
            try:
                loaded: BreezeSymbolMap = load_symbol_map(self.master_path)
                self._symbol_map = loaded.codes
                logger.info(
                    "breeze symbol master: %d codes (snapshot %s)",
                    len(loaded), loaded.fetched_at or "unknown",
                )
            except Exception as exc:  # noqa: BLE001 - network/file access
                logger.warning(
                    "breeze symbol master unavailable (%s); using NSE symbols as-is -- "
                    "expect empty series for names whose Breeze code differs", exc,
                )
                self._symbol_map = {}
        return self._symbol_map

    def _breeze_code(self, symbol: str) -> str:
        """Translate an NSE symbol to the ``stock_code`` Breeze actually serves."""
        sym = str(symbol).strip().upper()
        if sym == str(self.bench_symbol or "").strip().upper():
            # The benchmark is an index code (``NIFTY``), not a cash equity, so
            # the security master has nothing to say about it.
            return sym
        smap = self.symbol_map()
        code = smap.get(sym)
        if code is None:
            if smap:  # only worth saying when we do have a master to compare against
                self._unmapped.add(sym)
                logger.warning(
                    "%s is not in the Breeze security master; requesting it as-is", sym,
                )
            return sym
        return code

    # -- client ------------------------------------------------------------
    def _breeze(self):
        """The authenticated SDK client, built once from env + token file."""
        if self._client is not None:
            return self._client
        try:
            from breeze_connect import BreezeConnect
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "breeze-connect is not installed; run: uv sync --extra breeze"
            ) from exc
        # A gitignored .env is the normal home for the two static keys; a shell
        # export still wins. Without this the CLI would need them exported by
        # hand in every shell and every cron line.
        load_env_file()
        api_key = os.environ.get(API_KEY_ENV, "").strip()
        secret = os.environ.get(SECRET_KEY_ENV, "").strip()
        if not api_key or not secret:
            raise RuntimeError(
                f"set {API_KEY_ENV} and {SECRET_KEY_ENV} in the environment or in a "
                "gitignored .env at the repo root (they are static and must not be "
                "committed)"
            )
        token = load_session_token(self.session_path)
        client = BreezeConnect(api_key=api_key)
        client.generate_session(api_secret=secret, session_token=token)
        logger.info("breeze session established (token ...%s)", token[-6:])
        self._client = client
        return client

    # -- fetching ----------------------------------------------------------
    def get_many(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """One request per symbol (Breeze has no batch endpoint), throttled.

        Fails fast on a missing/stale token instead of swallowing the same auth
        error for every name and reporting an empty panel.
        """
        self._breeze()
        self._unmapped = set()
        out: dict[str, pd.DataFrame] = {}
        for i, sym in enumerate(symbols, 1):
            try:
                df = self.get_ohlcv(sym, start, end)
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the build
                logger.warning("[%d/%d] %s failed: %s", i, len(symbols), sym, exc)
                continue
            if not df.empty:
                out[sym] = df
            if self.cfg.request_delay_sec:
                time.sleep(float(self.cfg.request_delay_sec))
        logger.info("breeze prices: %d/%d symbols returned data", len(out), len(symbols))
        if self._unmapped:
            logger.warning(
                "breeze prices: %d symbol(s) absent from the security master ("
                "empty series expected): %s",
                len(self._unmapped), sorted(self._unmapped)[:12],
            )
        return out

    def get_ohlcv(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        start, end = self._window(start, end)
        sym = str(symbol).strip().upper()
        # yfinance-style ``end`` is exclusive; the caller already widens via
        # inclusive_end, so compare against min(end, today) to avoid refetching
        # the empty tail on every run.
        target_end = min(pd.Timestamp(end).normalize(), pd.Timestamp.today().normalize())
        key = f"breeze_{sym}"

        cached = self.cache.read(key) if self.cache else None
        if cached is not None and not cached.empty:
            cached = normalise_ohlcv(cached)
            if cached.index.max() >= target_end:
                return self._slice(cached, start, end)
            fetch_from_ts = cached.index.max() + pd.Timedelta(days=1)
            # No point asking for a gap that contains no weekday: a Saturday or
            # Sunday freeze would otherwise issue one empty call per symbol.
            if fetch_from_ts.normalize() > target_end or len(pd.bdate_range(fetch_from_ts, target_end)) == 0:
                return self._slice(cached, start, end)
            fresh = self._fetch(sym, fetch_from_ts.strftime("%Y-%m-%d"), end)
            if fresh.empty:
                return self._slice(cached, start, end)
            merged = normalise_ohlcv(pd.concat([cached, fresh]))
            if self.cache:
                self.cache.write(key, merged)
            return self._slice(merged, start, end)

        fresh = self._fetch(sym, start, end)
        if self.cache and not fresh.empty:
            self.cache.write(key, fresh)
        return self._slice(fresh, start, end)

    def get_benchmark(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Index series, falling back to yfinance if Breeze has no such code.

        The benchmark only supplies the trading calendar and the regime filter.
        The Breeze index code varies, so a failure here degrades to the previous
        source with a warning rather than failing the whole build.
        """
        sym = str(self.bench_symbol or "").strip()
        if sym:
            try:
                df = self.get_ohlcv(sym, start, end)
                if not df.empty:
                    return df
                logger.warning("breeze benchmark %r returned no rows; using yfinance", sym)
            except Exception as exc:  # noqa: BLE001
                logger.warning("breeze benchmark %r failed (%s); using yfinance", sym, exc)
        return YFinancePriceProvider(self.cfg, None).get_benchmark(start, end)

    def _fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """One symbol's daily bars in [start, end], with bounded retries."""
        if pd.Timestamp(start) > pd.Timestamp(end):
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        client = self._breeze()
        code = self._breeze_code(symbol)
        last: Exception | None = None
        for attempt in range(1, int(self.cfg.max_retries) + 1):
            try:
                raw = client.get_historical_data_v2(
                    interval="1day",
                    from_date=f"{pd.Timestamp(start).strftime('%Y-%m-%d')}T00:00:00.000Z",
                    to_date=f"{pd.Timestamp(end).strftime('%Y-%m-%d')}T23:59:59.000Z",
                    stock_code=code,
                    exchange_code="NSE",
                    product_type="cash",
                )
                df = normalise_ohlcv(_breeze_frame(raw, code))
                self._warn_if_unadjusted(symbol, df)
                return df
            except BreezeApiError:
                # A definitive answer from the API: retrying only wastes the
                # rate limit and buries the message under four identical logs.
                raise
            except Exception as exc:  # noqa: BLE001
                last = exc
                backoff = min(2.0 ** attempt, 15.0)
                logger.warning(
                    "breeze %s (code %s) attempt %d/%d failed (%s); retrying in %.1fs",
                    symbol, code, attempt, self.cfg.max_retries, exc, backoff,
                )
                time.sleep(backoff)
        raise RuntimeError(f"breeze fetch failed for {symbol} (code {code}): {last}")

    #: A close-to-close move beyond this is treated as an unadjusted corporate
    #: action rather than a real move. Set below 50% on purpose: a 1:2 split is
    #: exactly -50%, and a 60% threshold would silently miss every one of them.
    UNADJUSTED_JUMP = 0.35

    def _warn_if_unadjusted(self, symbol: str, df: pd.DataFrame) -> None:
        """A large close-to-close move is likely a corporate action the feed did not adjust."""
        if df.empty or symbol in self._warned_unadjusted:
            return
        jumps = int((df["close"].pct_change().abs() > self.UNADJUSTED_JUMP).sum())
        if jumps:
            self._warned_unadjusted.add(symbol)
            logger.warning(
                "%s: %d close-to-close jump(s) >%.0f%% -- Breeze prices may be UNADJUSTED "
                "for splits/bonuses; reconcile against the bhavcopy corporate-action "
                "detector before trusting this series",
                symbol, jumps, self.UNADJUSTED_JUMP * 100,
            )

    @staticmethod
    def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        if df.empty:
            return df
        return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


def make_price_provider(cfg: DataConfig, cache: DiskCache | None = None) -> PriceProvider:
    """Factory: resolve ``data.price_provider`` to a concrete provider."""
    name = (cfg.price_provider or "yfinance").lower()
    if name in {"yfinance", "yahoo"}:
        return YFinancePriceProvider(cfg, cache)
    if name in {"breeze", "icici"}:
        return BreezePriceProvider(cfg, cache)
    if name == "mock":
        return MockPriceProvider(cfg, cache)
    raise ValueError(f"unknown price_provider: {cfg.price_provider!r}")


# Re-exported for callers that want a bare session for NSE archive requests.
def nse_session() -> requests.Session:
    """A requests session carrying the browser headers NSE's CDN requires."""
    sess = requests.Session()
    sess.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return sess
