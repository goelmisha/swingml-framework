"""NSE delivery-quantity ingestion (the heavily weighted feature block).

Source: NSE's full security-wise bhavcopy, published daily at
``.../products/content/sec_bhavdata_full_<DDMMYYYY>.csv``. It is the only free
authoritative source that carries **delivery** data, which Yahoo Finance does
not expose at all.

Schema (measured against the live archive):
    SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE,
    LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS,
    NO_OF_TRADES, DELIV_QTY, DELIV_PER

Three measured hazards this module defends against
--------------------------------------------------
1. **Silent date aliasing.** NSE serves the *previous trading day's* file for
   non-trading dates. Requesting Sunday 2026-09-27 returned HTTP **200** with a
   payload whose ``DATE1`` was Friday 2026-09-25. Trusting the HTTP status alone
   writes duplicate Friday rows keyed to Sunday and corrupts every label, so we
   parse ``DATE1`` out of the payload and reject any mismatch.
2. **Archive floor.** Files exist from **2020-01-01** onward; earlier dates 404.
3. **Series pollution.** Each file mixes equities with mutual-fund and
   derivative series (``MF``, ``N1``..``YZ``, ...). Only ``EQ`` (+ optionally
   ``BE``/``BZ``) are tradable cash-market equities and must be kept.
"""

from __future__ import annotations

import abc
import datetime as dt
import io
import logging
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from swingml.config import DataConfig
from swingml.data.cache import DiskCache

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
INDEX_LIST_URL = "https://nsearchives.nseindia.com/content/indices/{name}.csv"
ARCHIVE_START = dt.date(2020, 1, 1)  # measured: nothing exists before this

#: Tradable cash-market series. Everything else is a fund/derivative artefact.
DEFAULT_SERIES = ("EQ",)

OUT_COLUMNS = [
    "symbol", "date", "series", "prev_close", "open", "high", "low", "close",
    "ttl_trd_qnty", "turnover_lacs", "no_of_trades", "deliv_qty", "deliv_per",
]


def _session() -> requests.Session:
    """Session carrying the browser headers NSE's CDN requires (bare UA -> 403)."""
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/csv,application/csv,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
    )
    return s


class _RateLimiter:
    """Global throttle so NSE does not start returning 403s mid-backfill."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = max(0.0, min_interval)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            delta = time.monotonic() - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


class DeliveryProvider(abc.ABC):
    """Abstract source of per-symbol daily delivery statistics."""

    def __init__(self, cfg: DataConfig, cache: DiskCache | None = None) -> None:
        self.cfg = cfg
        self.cache = cache

    @abc.abstractmethod
    def get_delivery(
        self,
        start: str | dt.date,
        end: str | dt.date,
        symbols: list[str] | None = None,
        trading_days: list[dt.date] | None = None,
    ) -> pd.DataFrame:
        """Long panel: one row per (date, symbol) with delivery columns."""


class NseBhavcopyDeliveryProvider(DeliveryProvider):
    """Real NSE full-bhavcopy ingestion, cached one parquet per trading day."""

    def __init__(self, cfg: DataConfig, cache: DiskCache | None = None, series: tuple[str, ...] = DEFAULT_SERIES) -> None:
        super().__init__(cfg, cache)
        self.series = tuple(s.upper() for s in series)
        self._limiter = _RateLimiter(cfg.request_delay_sec)
        self._local = threading.local()

    # -- http --------------------------------------------------------------
    @property
    def http(self) -> requests.Session:
        """One session per thread (requests.Session is not thread-safe)."""
        if not hasattr(self._local, "sess"):
            self._local.sess = _session()
        return self._local.sess

    @staticmethod
    def _to_str_date(d: str | dt.date | pd.Timestamp) -> dt.date:
        if isinstance(d, dt.date) and not isinstance(d, dt.datetime):
            return d
        return pd.Timestamp(d).date()

    @staticmethod
    def _ddmmyyyy(d: dt.date) -> str:
        return d.strftime("%d%m%Y")

    def _download_one(self, day: dt.date) -> pd.DataFrame | None:
        """Fetch and validate a single day. Returns ``None`` when not a session."""
        if day < ARCHIVE_START:
            return None
        url = ARCHIVE_URL.format(ddmmyyyy=self._ddmmyyyy(day))
        last_exc: Exception | None = None

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                self._limiter.wait()
                resp = self.http.get(url, timeout=self.cfg.request_timeout_sec)
                if resp.status_code == 404:
                    return None  # pre-archive date
                resp.raise_for_status()
                if not resp.text.strip():
                    return None
                df = pd.read_csv(io.StringIO(resp.text), skipinitialspace=True)
                break
            except Exception as exc:
                last_exc = exc
                backoff = min(2.0 ** attempt, 20.0)
                logger.warning("%s: attempt %d/%d failed (%s); retry in %.1fs",
                               day, attempt, self.cfg.max_retries, exc, backoff)
                time.sleep(backoff)
        else:
            logger.error("%s: giving up after %d attempts: %s", day, self.cfg.max_retries, last_exc)
            return None

        return self._validate_and_shape(df, day)

    def _validate_and_shape(self, df: pd.DataFrame, requested: dt.date) -> pd.DataFrame | None:
        """**The date-aliasing guard.** Reject payloads whose DATE1 != requested."""
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        if "DATE1" not in df.columns or "SYMBOL" not in df.columns:
            logger.warning("%s: unexpected bhavcopy schema %s", requested, list(df.columns)[:8])
            return None

        payload_dates = pd.to_datetime(df["DATE1"].astype(str).str.strip(), format="%d-%b-%Y", errors="coerce").dropna()
        if payload_dates.empty:
            return None
        payload_date = payload_dates.dt.date.mode().iloc[0]

        if payload_date != requested:
            # Non-trading day: NSE served the previous session. Dropping this is
            # what prevents duplicate rows on weekends/holidays.
            logger.debug("%s: payload contains %s (non-trading day) -- skipped", requested, payload_date)
            return None

        for col in ("PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE",
                    "TTL_TRD_QNTY", "TURNOVER_LACS", "NO_OF_TRADES", "DELIV_QTY", "DELIV_PER"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].replace("-", np.nan), errors="coerce")

        df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
        df["SERIES"] = df["SERIES"].astype(str).str.strip().str.upper()
        df = df[df["SERIES"].isin(self.series)]

        out = pd.DataFrame(
            {
                "symbol": df["SYMBOL"],
                "date": payload_date,
                "series": df["SERIES"],
                "prev_close": df.get("PREV_CLOSE"),
                "open": df.get("OPEN_PRICE"),
                "high": df.get("HIGH_PRICE"),
                "low": df.get("LOW_PRICE"),
                "close": df.get("CLOSE_PRICE"),
                "ttl_trd_qnty": df.get("TTL_TRD_QNTY"),
                "turnover_lacs": df.get("TURNOVER_LACS"),
                "no_of_trades": df.get("NO_OF_TRADES"),
                "deliv_qty": df.get("DELIV_QTY"),
                "deliv_per": df.get("DELIV_PER"),
            }
        )
        return out.dropna(subset=["close"]).reset_index(drop=True)

    # -- public ------------------------------------------------------------
    def get_delivery(
        self,
        start: str | dt.date,
        end: str | dt.date,
        symbols: list[str] | None = None,
        trading_days: list[dt.date] | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Assemble the delivery panel for [start, end].

        ``trading_days`` should come from the exchange calendar (we derive it
        from the benchmark index). When omitted, weekdays are probed and
        non-trading days are silently filtered by the DATE1 guard.
        """
        start_d = max(self._to_str_date(start), ARCHIVE_START)
        end_d = self._to_str_date(end)

        if trading_days is None:
            days = [d.date() for d in pd.bdate_range(start_d, end_d)]
        else:
            days = sorted({self._to_str_date(d) for d in trading_days})
            days = [d for d in days if start_d <= d <= end_d]
        if not days:
            return pd.DataFrame(columns=OUT_COLUMNS)

        frames: list[pd.DataFrame] = []
        fetched = hits = 0
        workers = max(1, int(self.cfg.max_workers))

        def task(day: dt.date) -> tuple[dt.date, pd.DataFrame | None]:
            key = f"deliv_{self._ddmmyyyy(day)}"
            if not force and self.cache and self.cache.exists(key):
                return day, self.cache.read(key)
            df = self._download_one(day)
            if df is not None and not df.empty and self.cache:
                self.cache.write(key, df)
            return day, df

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(task, d): d for d in days}
            for i, fut in enumerate(as_completed(futures), 1):
                day, df = fut.result()
                if df is None or df.empty:
                    continue
                hits += 1
                frames.append(df)
                fetched += 1
                if i % 100 == 0 or i == len(days):
                    logger.info("delivery: %d/%d days processed (%d sessions)", i, len(days), hits)

        if not frames:
            logger.warning("no delivery data assembled for %s..%s", start_d, end_d)
            return pd.DataFrame(columns=OUT_COLUMNS)

        panel = pd.concat(frames, ignore_index=True)
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

        # Belt-and-braces: the guard already rejects aliases, but assert the
        # invariant that would silently duplicate rows if it ever regressed.
        dupes = panel.duplicated(subset=["date", "symbol"], keep=False)
        if dupes.any():
            n = int(dupes.sum())
            logger.warning("dropping %d duplicate (date, symbol) delivery rows", n)
            panel = panel.drop_duplicates(subset=["date", "symbol"], keep="last")

        if symbols:
            wanted = {s.strip().upper() for s in symbols}
            panel = panel[panel["symbol"].isin(wanted)]

        return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


class MockDeliveryProvider(DeliveryProvider):
    """Deterministic synthetic delivery data for offline runs and tests.

    Delivery % is generated with genuine persistence and volume correlation, so
    the delivery features carry plausible signal without any network access.
    """

    def __init__(self, cfg: DataConfig, cache: DiskCache | None = None) -> None:
        super().__init__(cfg, cache)

    @property
    def default_symbols(self) -> list[str]:
        """The synthetic symbol set used when none is supplied."""
        return [f"MOCK{i:03d}" for i in range(self.cfg.mock.n_symbols)]

    @staticmethod
    def _rng(symbol: str, seed: int) -> np.random.Generator:
        return np.random.default_rng(seed + zlib.crc32(symbol.encode()) % 100_000 + 991)

    def get_delivery(
        self,
        start: str | dt.date,
        end: str | dt.date,
        symbols: list[str] | None = None,
        trading_days: list[dt.date] | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        syms = [s.strip().upper() for s in symbols] if symbols else self.default_symbols

        if trading_days is None:
            days = [d.date() for d in pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))]
        else:
            days = [pd.Timestamp(d).date() for d in trading_days]
        if not days:
            return pd.DataFrame(columns=OUT_COLUMNS)

        rows = []
        for sym in syms:
            rng = self._rng(sym, self.cfg.mock.seed)
            n = len(days)

            # A coherent price path, so turnover and quantity stay consistent.
            # Drawing a fresh divisor per day (as opposed to per symbol) would
            # make traded quantity jump ~40x at random and spuriously trip the
            # split detector downstream.
            base_price = rng.uniform(50.0, 2000.0)
            close = base_price * np.exp(np.cumsum(rng.normal(0.0004, 0.014, size=n)))
            prev_close = np.concatenate([[close[0]], close[:-1]])
            day_span = close * np.abs(rng.normal(0.0, 0.012, size=n))

            # AR(1) delivery-% with genuine persistence.
            base_per = rng.uniform(25.0, 55.0)
            shocks = rng.normal(0.0, 4.0, size=n)
            ar = np.zeros(n)
            for i in range(1, n):
                ar[i] = 0.72 * ar[i - 1] + shocks[i]
            deliver_per = np.clip(base_per + ar, 5.0, 95.0)

            base_turnover = rng.uniform(400.0, 6000.0)  # lacs; clears the liquidity floor
            turnover = np.exp(rng.normal(np.log(base_turnover), 0.45, size=n))
            ttl_qty = np.maximum(1.0, turnover * 1e5 / close)

            rows.append(
                pd.DataFrame(
                    {
                        "symbol": sym,
                        "date": pd.to_datetime(days),
                        "series": "EQ",
                        "prev_close": prev_close,
                        "open": prev_close * (1.0 + rng.normal(0.0, 0.004, size=n)),
                        "high": np.maximum(close, prev_close) + day_span,
                        "low": np.minimum(close, prev_close) - day_span,
                        "close": close,
                        "ttl_trd_qnty": ttl_qty,
                        "turnover_lacs": turnover,
                        "no_of_trades": np.maximum(1, (ttl_qty / rng.uniform(80.0, 300.0, size=n))).astype(int),
                        "deliv_qty": ttl_qty * deliver_per / 100.0,
                        "deliv_per": deliver_per,
                    }
                )
            )
        panel = pd.concat(rows, ignore_index=True)
        if symbols:
            wanted = {s.strip().upper() for s in symbols}
            panel = panel[panel["symbol"].isin(wanted)]
        return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


def make_delivery_provider(cfg: DataConfig, cache: DiskCache | None = None) -> DeliveryProvider:
    """Factory: resolve ``data.delivery_provider`` to a concrete provider."""
    name = (cfg.delivery_provider or "nse_bhavcopy").lower()
    if name in {"nse_bhavcopy", "nse", "bhavcopy"}:
        return NseBhavcopyDeliveryProvider(cfg, cache)
    if name == "mock":
        return MockDeliveryProvider(cfg, cache)
    raise ValueError(f"unknown delivery_provider: {cfg.delivery_provider!r}")


def fetch_index_constituents(
    index_name: str,
    cache: DiskCache | None = None,
    cfg: DataConfig | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Download an official NSE index constituent list.

    ``index_name`` must be passed explicitly; the index you trade is a policy
    decision, and a default here would silently apply someone else's.
    """
    slug = index_name.strip().lower().replace(" ", "")
    name = f"ind_{slug}list"
    url = INDEX_LIST_URL.format(name=name)
    key = f"index_{slug}"

    def _build() -> pd.DataFrame:
        sess = _session()
        resp = sess.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), skipinitialspace=True)
        df.columns = [str(c).strip() for c in df.columns]
        if "Symbol" not in df.columns:
            raise ValueError(f"unexpected constituent schema: {list(df.columns)}")
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        df["Series"] = df.get("Series", pd.Series("EQ", index=df.index)).astype(str).str.strip().str.upper()
        return df.drop_duplicates(subset=["Symbol"]).reset_index(drop=True)

    if cache is None:
        return _build()
    out = cache.get_or_build(key, _build, force=force, allow_empty=True)
    return out if out is not None else pd.DataFrame()
