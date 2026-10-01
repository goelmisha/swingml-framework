"""ICICI's security master -- the NSE symbol -> Breeze ``stock_code`` mapping.

Why this module exists
----------------------
Breeze's historical endpoint takes a ``stock_code`` that is **not** the NSE
trading symbol. Handing it the NSE symbol (``RELIANCE``) returns HTTP 200 with
``Success: []`` and ``Error: null`` -- indistinguishable from a delisted series.
The codes come from ICICI's security master, not from the NSE symbol:

===========  ===============  =========
NSE symbol   Breeze code      NSE token
===========  ===============  =========
RELIANCE     ``RELIND``       2885
HDFCBANK     ``HDFBAN``       1333
ICICIBANK    ``ICIBAN``       4963
INFY         ``INFTEC``       1594
SBIN         ``STABAN``       3045
LT           ``LARTOU``       11483
===========  ===============  =========

Most liquid names carry a code different from their NSE symbol, so a symbol left
unmapped looks like an empty series rather than an error. The resolution here is
a client-side code mapping, not a data outage at the broker.

The join key
------------
``NSEScripMaster.txt`` has both halves of the mapping on one row, which makes
the lookup exact rather than a name guess:

* ``ExchangeCode`` -- the **NSE trading symbol** (``RELIANCE``);
* ``ShortName``    -- the **Breeze ``stock_code``** (``RELIND``);
* ``Token``        -- the **NSE token** (``2885``), which independently confirms
  the pairing: it matches the NSE token for the same security.

Only cash series (``EQ``/``BE``/``BZ``) are kept; every other series in the file
is a fund, bond or derivative whose ``ExchangeCode`` is the *underlying*.

Coverage caveat: the master is **today's** file. Symbols that no longer trade
under their old name are absent and cannot be resolved from it. That is exactly
the survivorship-relevant set, so a historical backfill must not read "absent
from the master" as "no data".
"""

from __future__ import annotations

import datetime as dt
import io
import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: ICICI's master archive. The Breeze SDK downloads this at import time; it is
#: public and needs no credentials, which is why this module can be used (and
#: tested) without a session.
SECURITY_MASTER_URL = "https://directlink.icicidirect.com/MotherAppMaster/SecurityMaster.zip"

#: The NSE member holding cash equities. The archive also carries BSE, currency
#: and F&O masters, which this project never reads.
NSE_MASTER_MEMBER = "NSEScripMaster.txt"

#: Tradable cash series, the same set the bhavcopy provider keeps. Every other
#: series' ``ExchangeCode`` is an underlying, not a tradable symbol.
CASH_SERIES = ("EQ", "BE", "BZ")

DEFAULT_MASTER_PATH = "data/breeze_master.json"

#: The master is refreshed daily upstream. A week is short enough that new
#: listings appear before they matter and long enough to make it one download
#: per work-week rather than one per run.
MASTER_MAX_AGE_DAYS = 7


def _unquote(value: str) -> str:
    return (value or "").strip().strip('"').strip()


def parse_security_master(text: str) -> dict[str, str]:
    """Parse ``NSEScripMaster.txt`` into ``{nse_symbol: breeze_stock_code}``.

    The header is quoted and space-padded upstream (``"ShortName"``), so every
    row is mapped positionally from a normalised header rather than by name.
    Rows whose series is not cash are skipped, and the first cash row wins when
    a symbol appears more than once.
    """
    import csv

    reader = csv.reader(io.StringIO(text))
    try:
        header = [_unquote(h) for h in next(reader)]
    except StopIteration:
        return {}
    required = {"ExchangeCode", "ShortName", "Series"}
    if not required.issubset(header):
        raise ValueError(
            f"unexpected NSEScripMaster schema; missing {sorted(required - set(header))}"
        )
    i_exch = header.index("ExchangeCode")
    i_short = header.index("ShortName")
    i_series = header.index("Series")

    out: dict[str, str] = {}
    for row in reader:
        if len(row) <= max(i_exch, i_short, i_series):
            continue
        if _unquote(row[i_series]) not in CASH_SERIES:
            continue
        nse = _unquote(row[i_exch]).upper()
        code = _unquote(row[i_short]).upper()
        if nse and code:
            out.setdefault(nse, code)
    return out


def download_security_master(timeout_sec: int = 60) -> str:
    """Fetch and decode the NSE member of ``SecurityMaster.zip``."""
    import requests

    logger.info("downloading Breeze security master from %s", SECURITY_MASTER_URL)
    resp = requests.get(
        SECURITY_MASTER_URL,
        timeout=timeout_sec,
        headers={"User-Agent": "Mozilla/5.0 (compatible; swingml)"},
    )
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        names = archive.namelist()
        if NSE_MASTER_MEMBER not in names:
            raise ValueError(f"{NSE_MASTER_MEMBER} absent from the archive: {names}")
        raw = archive.read(NSE_MASTER_MEMBER)
    return raw.decode("utf-8", errors="replace")


@dataclass
class BreezeSymbolMap:
    """``nse_symbol -> breeze_stock_code`` plus the provenance of the snapshot.

    ``resolve`` never raises: an unknown symbol is returned unchanged so the
    provider still tries the NSE symbol (56 of 688 codes are identical, and a
    brand-new listing is not in today's master yet). The caller logs the miss --
    silently issuing a request that returns nothing is the failure this module
    exists to remove.
    """

    codes: dict[str, str] = field(default_factory=dict)
    fetched_at: str = ""
    source: str = SECURITY_MASTER_URL
    conflicts: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.codes)

    def __contains__(self, symbol: str) -> bool:
        return str(symbol).strip().upper() in self.codes

    def resolve(self, symbol: str) -> str:
        """The Breeze ``stock_code`` for an NSE symbol (identity when unknown)."""
        s = str(symbol).strip().upper()
        return self.codes.get(s, s)

    def has_every(self, symbols: list[str]) -> bool:
        return all(str(s).strip().upper() in self.codes for s in symbols)

    @property
    def age_days(self) -> int | None:
        if not self.fetched_at:
            return None
        try:
            stamp = dt.date.fromisoformat(self.fetched_at)
        except ValueError:
            return None
        return (dt.date.today() - stamp).days

    def to_json(self) -> str:
        return json.dumps(
            {
                "fetched_at": self.fetched_at,
                "source": self.source,
                "n_symbols": len(self.codes),
                "conflicts": self.conflicts,
                "codes": self.codes,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> "BreezeSymbolMap":
        payload = json.loads(text)
        return cls(
            codes={str(k).upper(): str(v) for k, v in (payload.get("codes") or {}).items()},
            fetched_at=str(payload.get("fetched_at") or ""),
            source=str(payload.get("source") or SECURITY_MASTER_URL),
            conflicts=list(payload.get("conflicts") or []),
        )

    def save(self, path: str | Path = DEFAULT_MASTER_PATH) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        tmp.replace(out)
        logger.info("breeze symbol map saved -> %s (%d codes)", out, len(self.codes))
        return out


def build_symbol_map(timeout_sec: int = 60, now: dt.date | None = None) -> BreezeSymbolMap:
    """Download and parse the master into a fresh :class:`BreezeSymbolMap`."""
    codes = parse_security_master(download_security_master(timeout_sec=timeout_sec))
    return BreezeSymbolMap(
        codes=codes,
        fetched_at=(now or dt.date.today()).isoformat(),
        conflicts=[],
    )


def load_symbol_map(
    path: str | Path = DEFAULT_MASTER_PATH,
    *,
    max_age_days: int = MASTER_MAX_AGE_DAYS,
    refresh: bool = False,
    timeout_sec: int = 60,
) -> BreezeSymbolMap:
    """Return a snapshot, reusing the cached file while it is fresh.

    A stale or missing file triggers one download and one write. Failure to
    download with a usable file on disk degrades to the cached copy; failure
    with nothing on disk propagates, because an empty map would turn every
    symbol into a silent miss -- the exact bug this module removes.
    """
    p = Path(path)
    cached: BreezeSymbolMap | None = None
    if p.exists() and not refresh:
        try:
            cached = BreezeSymbolMap.from_json(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("%s is unreadable (%s); refetching", p, exc)
            cached = None
        if cached is not None and cached.age_days is not None and cached.age_days <= max_age_days:
            return cached

    try:
        fresh = build_symbol_map(timeout_sec=timeout_sec)
    except Exception as exc:  # noqa: BLE001 - network is allowed to be down
        if cached is not None and len(cached):
            logger.warning("security master refresh failed (%s); using the cached snapshot", exc)
            return cached
        raise RuntimeError(
            f"could not load the Breeze security master from {p} or {SECURITY_MASTER_URL}: {exc}"
        ) from exc
    fresh.save(p)
    return fresh


__all__ = [
    "CASH_SERIES",
    "DEFAULT_MASTER_PATH",
    "MASTER_MAX_AGE_DAYS",
    "NSE_MASTER_MEMBER",
    "SECURITY_MASTER_URL",
    "BreezeSymbolMap",
    "build_symbol_map",
    "download_security_master",
    "load_symbol_map",
    "parse_security_master",
]
