"""Point-in-time NSE symbol history -- resolving renamed and delisted tickers.

The problem this solves
-----------------------
The trading universe is built from the bhavcopy, so it is *point-in-time*: it
contains a name on the dates it was actually listed and liquid. The **price**
panel is not. Yahoo keys history on the *current* ticker, so every name that has
since been renamed comes back empty under its historical symbol -- the provider
reports it as "possibly delisted" -- and those names silently vanish from the
feature matrix.

The largest misses are renames, not data failures: a renamed large cap can
account for over a thousand missing member-sessions. A survivorship claim that
silently drops a meaningful slice of the universe's member-sessions is not
survivorship-free, so this is a correctness bug, not a nice-to-have.

How it works
------------
NSE's bhavcopy carries an **ISIN** column, in both of its formats, so a
point-in-time ``symbol -> ISIN`` map is buildable from the archive this project
already reads:

* pre-2024-07: ``content/historical/EQUITIES/<YYYY>/<MON>/cm<DD><MON><YYYY>bhav.csv.zip``
  (``SYMBOL`` ... ``ISIN``);
* 2024-07 onward: ``content/cm/BhavCopy_NSE_CM_0_0_0_<YYYYMMDD>_F_0000.csv.zip``
  (``ISIN`` + ``TckrSymb``).

ISIN is the stable identity; the ticker is not. Two symbols sharing one ISIN are
the same security under two names, so the *latest* symbol observed for an ISIN is
the priceable one and every earlier one is an alias of it. Sampling the archive
monthly (not daily) is enough: a name is present for its whole listing period, so
one sample inside that period fixes its ISIN. Renames survive that sampling
because the ISIN is the same on both sides of the change.

Why the map is validated, not trusted
-------------------------------------
A wrong alias **fabricates price history**, which is the worst failure mode in
this project. So an alias is admitted only when both halves are observed and the
listing spans do not overlap: if the old and the new symbol traded on the same
session they are two live securities, not one renamed one, and claiming otherwise
would double-count a name. ``tests/test_symbol_history.py`` pins both guards.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: Old (pre-UDiFF) CM bhavcopy, which carried an ``ISIN`` column.
OLD_BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/content/historical/EQUITIES/"
    "{yyyy}/{mon}/cm{dd}{mon3}{yyyy}bhav.csv.zip"
)

#: UDiFF CM bhavcopy, in force from July 2024; carries ``ISIN`` and ``TckrSymb``.
UDIFF_BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)

#: UDiFF replaced the old format in early July 2024.
UDIFF_START = dt.date(2024, 7, 1)

#: Tradable cash series, the same set the bhavcopy delivery provider keeps.
CASH_SERIES = ("EQ", "BE", "BZ")

DEFAULT_HISTORY_PATH = "data/symbol_history.json"

#: One sample per ~20 sessions. A name is in the bhavcopy for its entire listing
#: period, so a monthly sample cannot miss a listing that lasted a month or more.
DEFAULT_SAMPLE_EVERY = 20


def _session() -> "object":
    """A requests session with the browser headers NSE's CDN requires."""
    import requests

    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/csv,application/csv,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return s


def _urls_for(day: dt.date) -> list[str]:
    if day >= UDIFF_START:
        return [
            UDIFF_BHAVCOPY_URL.format(yyyymmdd=day.strftime("%Y%m%d")),
            OLD_BHAVCOPY_URL.format(yyyy=day.strftime("%Y"), mon=day.strftime("%b").upper(),
                                    dd=day.strftime("%d"), mon3=day.strftime("%b").upper()),
        ]
    return [
        OLD_BHAVCOPY_URL.format(yyyy=day.strftime("%Y"), mon=day.strftime("%b").upper(),
                                dd=day.strftime("%d"), mon3=day.strftime("%b").upper()),
        UDIFF_BHAVCOPY_URL.format(yyyymmdd=day.strftime("%Y%m%d")),
    ]


def parse_isin_frame(text: str) -> pd.DataFrame:
    """``{symbol, isin, series}`` from one bhavcopy payload, in either format.

    Only the two columns that matter are read, and only cash series are kept --
    the same convention the delivery provider uses, so the two cannot disagree
    about what a tradable symbol is.
    """
    head = text[:4096]
    if "TckrSymb" in head:  # UDiFF
        sym_col, isin_col, series_col = "TckrSymb", "ISIN", "SctySrs"
    elif "SYMBOL" in head:
        sym_col, isin_col, series_col = "SYMBOL", "ISIN", "SERIES"
    else:
        raise ValueError("unrecognised bhavcopy schema; cannot locate SYMBOL/ISIN")

    df = pd.read_csv(io.StringIO(text), skipinitialspace=True, usecols=lambda c: c.strip() in {sym_col, isin_col, series_col})
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={sym_col: "symbol", isin_col: "isin", series_col: "series"})
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    df["series"] = df["series"].astype(str).str.strip().str.upper()
    df = df[df["series"].isin(CASH_SERIES)]
    df = df[(df["symbol"] != "") & (df["isin"] != "") & (df["isin"] != "NAN")]
    return df[["symbol", "isin"]].drop_duplicates()


def fetch_isin_frame(day: dt.date, timeout_sec: int = 45) -> pd.DataFrame:
    """One session's ``{symbol, isin}``, trying both archive formats."""
    sess = _session()
    last: Exception | None = None
    for url in _urls_for(day):
        try:
            resp = sess.get(url, timeout=timeout_sec)
            if resp.status_code != 200 or not resp.content:
                continue
            raw = resp.content
            if raw[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    raw = archive.read(archive.namelist()[0])
            frame = parse_isin_frame(raw.decode("utf-8", errors="replace"))
            if not frame.empty:
                return frame
        except Exception as exc:  # noqa: BLE001 - try the other format
            last = exc
    if last is not None:
        logger.debug("%s: no usable bhavcopy for the ISIN map (%s)", day, last)
    return pd.DataFrame(columns=["symbol", "isin"])


@dataclass
class SymbolHistory:
    """Point-in-time ``symbol -> isin`` plus the aliases derived from it.

    ``observed`` maps each symbol to the last session it was seen on (for the
    span guard), ``isin_of`` to its ISIN, and ``aliases`` maps a superseded
    symbol to the current one carrying the same ISIN.
    """

    isin_of: dict[str, str] = field(default_factory=dict)
    first_seen: dict[str, str] = field(default_factory=dict)
    last_seen: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    fetched_at: str = ""
    samples: list[str] = field(default_factory=list)

    def resolve(self, symbol: str) -> str:
        """The priceable ticker for a symbol: itself, or its successor."""
        s = str(symbol).strip().upper()
        return self.aliases.get(s, s)

    def is_alias(self, symbol: str) -> bool:
        return str(symbol).strip().upper() in self.aliases

    @property
    def n_symbols(self) -> int:
        return len(self.isin_of)

    def to_json(self) -> str:
        return json.dumps(
            {
                "fetched_at": self.fetched_at,
                "samples": self.samples,
                "n_symbols": len(self.isin_of),
                "n_aliases": len(self.aliases),
                "isin_of": self.isin_of,
                "first_seen": self.first_seen,
                "last_seen": self.last_seen,
                "aliases": self.aliases,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> "SymbolHistory":
        p = json.loads(text)
        return cls(
            isin_of={str(k).upper(): str(v) for k, v in (p.get("isin_of") or {}).items()},
            first_seen=dict(p.get("first_seen") or {}),
            last_seen=dict(p.get("last_seen") or {}),
            aliases={str(k).upper(): str(v) for k, v in (p.get("aliases") or {}).items()},
            fetched_at=str(p.get("fetched_at") or ""),
            samples=list(p.get("samples") or []),
        )

    def save(self, path: str | Path = DEFAULT_HISTORY_PATH) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        tmp.replace(out)
        logger.info("symbol history saved -> %s (%d symbols, %d aliases)",
                    out, len(self.isin_of), len(self.aliases))
        return out


def build_symbol_history(
    days: list[dt.date],
    *,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    timeout_sec: int = 45,
) -> SymbolHistory:
    """Sample the archive and derive ``symbol -> isin`` and the rename aliases.

    The alias rule is deliberately strict, in both directions:

    * two symbols share an ISIN;
    * the retired symbol's **last** observation precedes the replacement's
      **first** -- if they overlap on any sampled session they were both listed,
      which makes them different securities (a demerger, a dual listing) and not
      a rename.
    """
    days = sorted({pd.Timestamp(d).date() for d in days})
    sampled = days[:: max(1, sample_every)]
    if sampled and days and sampled[-1] != days[-1]:
        sampled.append(days[-1])  # always include the newest session

    first_seen: dict[str, dt.date] = {}
    last_seen: dict[str, dt.date] = {}
    isin_of: dict[str, str] = {}
    got: list[dt.date] = []
    for day in sampled:
        frame = fetch_isin_frame(day, timeout_sec=timeout_sec)
        if frame.empty:
            continue
        got.append(day)
        for sym, isin in zip(frame["symbol"], frame["isin"]):
            isin_of.setdefault(sym, isin)
            last_seen[sym] = day  # sampled in order, so this ends up the newest
            first_seen.setdefault(sym, day)
            isin_of[sym] = isin_of.get(sym) or isin
    logger.info("symbol history: %d symbols over %d sampled sessions", len(isin_of), len(got))

    # The successor is the symbol with the latest last_seen for an ISIN.
    latest_for_isin: dict[str, str] = {}
    for sym, isin in isin_of.items():
        incumbent = latest_for_isin.get(isin)
        if incumbent is None or last_seen.get(sym) and last_seen.get(sym) >= last_seen.get(incumbent, dt.date.min):
            latest_for_isin[isin] = sym

    aliases: dict[str, str] = {}
    for sym, isin in isin_of.items():
        current = latest_for_isin.get(isin)
        if current is None or current == sym:
            continue
        if last_seen.get(sym) is None or first_seen.get(current) is None:
            continue
        if last_seen[sym] < first_seen[current]:
            aliases[sym] = current

    return SymbolHistory(
        isin_of=isin_of,
        first_seen={k: v.isoformat() for k, v in first_seen.items()},
        last_seen={k: v.isoformat() for k, v in last_seen.items()},
        aliases=aliases,
        fetched_at=dt.date.today().isoformat(),
        samples=[d.isoformat() for d in got],
    )


def load_symbol_history(
    path: str | Path = DEFAULT_HISTORY_PATH,
    *,
    days: list[dt.date] | None = None,
    refresh: bool = False,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    timeout_sec: int = 45,
) -> SymbolHistory:
    """Return a cached history, building it from ``days`` when stale or absent."""
    p = Path(path)
    if p.exists() and not refresh:
        try:
            cached = SymbolHistory.from_json(p.read_text(encoding="utf-8"))
            if cached.n_symbols:
                return cached
        except (OSError, ValueError) as exc:
            logger.warning("%s is unreadable (%s); rebuilding", p, exc)
    if not days:
        raise RuntimeError(
            f"no symbol history at {p} and no trading days supplied to build one"
        )
    fresh = build_symbol_history(days, sample_every=sample_every, timeout_sec=timeout_sec)
    fresh.save(p)
    return fresh


__all__ = [
    "CASH_SERIES",
    "DEFAULT_HISTORY_PATH",
    "DEFAULT_SAMPLE_EVERY",
    "OLD_BHAVCOPY_URL",
    "UDIFF_BHAVCOPY_URL",
    "UDIFF_START",
    "SymbolHistory",
    "build_symbol_history",
    "fetch_isin_frame",
    "load_symbol_history",
    "parse_isin_frame",
]
