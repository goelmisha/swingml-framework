"""Probe Breeze Connect with the smallest possible number of read-only calls.

Purpose: answer six questions before trusting the provider with a build --

1. does the session token authenticate (and is the static IP accepted)?
2. what shape does ``get_historical_data_v2`` actually return, and does
   :func:`swingml.data.prices._breeze_frame` parse it?
3. does ICICI's security master resolve the NSE symbol to a Breeze
   ``stock_code``, and does the NSE symbol itself return nothing? (This is the
   difference between a working feed and the "~80% of symbols are empty" bug.)
4. does the mapped code return data through the provider's own parser?
5. can the Nifty index be fetched for the benchmark, and under which code?
6. what does a call cost (rate-limit sanity check)?

It places no orders, reads no positions, and prints no secrets -- only key names,
shapes and prices (prices are public).

Credentials
-----------
``BREEZE_API_KEY`` / ``BREEZE_SECRET_KEY`` are read from the environment, or from
a gitignored ``.env`` in the repo root (so they never have to be pasted into a
chat). The session token is read from the gitignored token file written by
``scripts/breeze_token.py``.

Usage
-----
    .venv/bin/python scripts/breeze_probe.py
    .venv/bin/python scripts/breeze_probe.py --symbol RELIANCE --days 20
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time

from swingml.config import configure_logging
from swingml.data.breeze import (
    API_KEY_ENV,
    DEFAULT_SESSION_PATH,
    SECRET_KEY_ENV,
    BreezeSessionError,
    load_env_file,
    load_session_token,
    session_status,
)
from swingml.data.breeze_master import DEFAULT_MASTER_PATH, load_symbol_map
from swingml.data.prices import _breeze_frame, normalise_ohlcv

#: Index codes worth trying for the benchmark. Breeze's index naming is not the
#: NSE trading symbol, so this is a lookup rather than an assumption.
BENCH_CANDIDATES = ("NIFTY", "NIFTY50", "NIFTY 50", "NIFTY 50 INDEX")


def _describe_payload(raw) -> list[str]:
    """Report a payload's shape without dumping data."""
    lines = []
    if raw is None:
        return ["  type: None"]
    lines.append(f"  python type : {type(raw).__name__}")
    if isinstance(raw, dict):
        lines.append(f"  dict keys   : {sorted(raw)[:8]}")
        payload = next((raw[k] for k in ("Success", "success", "data") if k in raw), None)
        if isinstance(payload, list):
            lines.append(f"  Success     : list of {len(payload)} record(s)")
            if payload:
                lines.append(f"  record keys : {sorted(payload[0])}")
                lines.append(f"  first record: {payload[0]}")
    elif hasattr(raw, "shape"):
        lines.append(f"  shape       : {raw.shape}")
        lines.append(f"  columns     : {list(getattr(raw, 'columns', []))}")
        lines.append(f"  index       : {raw.index.min()} .. {raw.index.max()}" if len(raw) else "  index: empty")
    return lines


def probe_symbol_master(symbol: str) -> str:
    """Resolve the NSE symbol through ICICI's security master.

    Breeze's ``stock_code`` is the master's ``ShortName``, not the NSE trading
    symbol: RELIANCE is ``RELIND``. Requesting the NSE symbol returns HTTP 200
    with no error and no rows, which is what made this look like a data outage.
    """
    print("\n[3] security master (NSE symbol -> Breeze stock_code)")
    try:
        smap = load_symbol_map(DEFAULT_MASTER_PATH)
    except Exception as exc:  # noqa: BLE001 - exactly what this is testing
        print(f"  master unavailable: {type(exc).__name__}: {exc}")
        print("  -> requests will carry the NSE symbol and most names will be empty")
        return symbol
    print(f"  {len(smap)} codes | snapshot {smap.fetched_at} | source {smap.source}")
    code = smap.resolve(symbol)
    print(f"  {symbol} -> {code}" + ("  (identical)" if code == symbol else ""))
    if symbol not in smap:
        print("  NOTE: absent from the master -- delisted under this name, or a new listing")
    return code


def probe_historical(client, symbol: str, days: int, code: str | None = None) -> dict:
    """One historical call through the provider's own parser."""
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    code = code or symbol
    t0 = time.time()
    raw = client.get_historical_data_v2(
        interval="1day",
        from_date=f"{start.isoformat()}T00:00:00.000Z",
        to_date=f"{end.isoformat()}T23:59:59.000Z",
        stock_code=code,
        exchange_code="NSE",
        product_type="cash",
    )
    elapsed = time.time() - t0

    print(f"\n[4] historical v2  {symbol}" + (f"  (stock_code={code})" if code != symbol else "")
          + f"  {start} .. {end}   ({elapsed:.2f}s)")
    for line in _describe_payload(raw):
        print(line)

    out: dict = {"elapsed": elapsed, "rows": 0, "adjusted": None}
    try:
        df = normalise_ohlcv(_breeze_frame(raw, symbol))
    except Exception as exc:  # noqa: BLE001 - this is exactly what we are testing
        print(f"  provider parser: FAILED -- {type(exc).__name__}: {exc}")
        return out
    out["rows"] = len(df)
    print(f"  provider parser: OK -> {len(df)} rows, columns {list(df.columns)}")
    if len(df):
        print(f"  date range  : {df.index.min().date()} .. {df.index.max().date()}")
        print(f"  last close  : {float(df['close'].iloc[-1]):.2f}")
        rets = df["close"].pct_change().abs()
        worst = float(rets.max()) if len(rets.dropna()) else float("nan")
        out["adjusted"] = worst < 0.35
        print(f"  max |1d move|: {worst:.1%}  -> {'looks ADJUSTED' if out['adjusted'] else 'LOOKS UNADJUSTED (corporate action?)'}")
    return out


def probe_raw_vs_mapped(client, symbol: str, code: str, days: int) -> int:
    """Show the failure the mapping removes: the NSE symbol returns zero rows."""
    if code == symbol:
        return -1
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    try:
        raw = client.get_historical_data_v2(
            interval="1day",
            from_date=f"{start.isoformat()}T00:00:00.000Z",
            to_date=f"{end.isoformat()}T23:59:59.000Z",
            stock_code=symbol, exchange_code="NSE", product_type="cash",
        )
        rows = len(normalise_ohlcv(_breeze_frame(raw, symbol)))
    except Exception as exc:  # noqa: BLE001
        print(f"  raw NSE symbol {symbol!r}: raised {type(exc).__name__}")
        return -1
    note = "the empty-series failure the mapping fixes" if rows == 0 else "also works"
    print(f"  raw NSE symbol {symbol!r}: {rows} row(s) ({note})")
    return rows


def run(args: argparse.Namespace) -> int:
    print("=" * 88)
    print("BREEZE CONNECT PROBE  --  read-only, no orders")
    print("=" * 88)

    dotenv_n = load_env_file()
    if dotenv_n:
        print(f"[0] loaded {dotenv_n} variable(s) from .env")

    api_key = os.environ.get(API_KEY_ENV, "").strip()
    secret = os.environ.get(SECRET_KEY_ENV, "").strip()
    st = session_status(args.token_path)
    print(f"[0] api key: {'set' if api_key else 'MISSING'} | "
          f"secret: {'set' if secret else 'MISSING'} | "
          f"token: {'present' if st['present'] else 'MISSING'}"
          + (f" (dated {st['date']}, current={st['current']})" if st["present"] else ""))

    if not (api_key and secret):
        print(f"\nBLOCKED: {API_KEY_ENV} / {SECRET_KEY_ENV} are not set.")
        print("  Either export them, or put them in a gitignored .env at the repo root:")
        print(f"    {API_KEY_ENV}=your_app_key")
        print(f"    {SECRET_KEY_ENV}=your_secret_key")
        print("  Then mint today's token:  .venv/bin/python scripts/breeze_token.py")
        return 2
    try:
        token = load_session_token(args.token_path)
    except BreezeSessionError as exc:
        print(f"\nBLOCKED: {exc}")
        return 2

    from breeze_connect import BreezeConnect

    client = BreezeConnect(api_key=api_key)

    print("\n[1] generate_session (auth + static-IP check)")
    try:
        client.generate_session(api_secret=secret, session_token=token)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED -- {type(exc).__name__}: {exc}")
        msg = str(exc).lower()
        if "ip" in msg:
            print("  This looks like the STATIC-IP gate: register this machine's IP on the API key.")
        return 1
    print("  OK -- session established")

    print("\n[2] get_customer_details (proves the token is accepted for data)")
    try:
        details = client.get_customer_details(api_session=token)
        payload = details.get("Success") if isinstance(details, dict) else details
        if isinstance(payload, dict):
            print(f"  OK -- keys: {sorted(payload)[:10]}")
        else:
            print(f"  OK -- payload: {type(payload).__name__}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED -- {type(exc).__name__}: {exc}")
        return 1

    code = probe_symbol_master(args.symbol)
    result = probe_historical(client, args.symbol, args.days, code=code)
    probe_raw_vs_mapped(client, args.symbol, code, args.days)

    print("\n[5] benchmark index lookup (for the trading calendar / regime filter)")
    found = None
    for code in BENCH_CANDIDATES:
        try:
            raw = client.get_historical_data_v2(
                interval="1day",
                from_date=f"{(dt.date.today() - dt.timedelta(days=args.days)).isoformat()}T00:00:00.000Z",
                to_date=f"{dt.date.today().isoformat()}T23:59:59.000Z",
                stock_code=code, exchange_code="NSE", product_type="cash",
            )
            df = normalise_ohlcv(_breeze_frame(raw, code))
        except Exception as exc:  # noqa: BLE001
            print(f"  {code!r}: failed ({type(exc).__name__})")
            continue
        print(f"  {code!r}: {len(df)} rows")
        if len(df) and found is None:
            found = code
    print(f"  -> use breeze_bench_symbol: {found!r}" if found else
          "  -> no index code worked; the benchmark will fall back to yfinance")

    print("\n[6] throughput (3 sequential calls, to sanity-check rate limits)")
    t0 = time.time()
    for _ in range(3):
        try:
            client.get_historical_data_v2(
                interval="1day",
                from_date=f"{(dt.date.today() - dt.timedelta(days=10)).isoformat()}T00:00:00.000Z",
                to_date=f"{dt.date.today().isoformat()}T23:59:59.000Z",
                stock_code=code, exchange_code="NSE", product_type="cash",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  call failed: {type(exc).__name__}: {exc}")
            break
    per = (time.time() - t0) / 3
    print(f"  ~{per:.2f}s per call -> ~{460 * per / 60:.1f} min for 460 symbols, one call each")

    print("\n" + "=" * 88)
    verdict = ["auth OK", "customer_details OK"]
    verdict.append(f"historical {'OK' if result['rows'] else 'EMPTY'}")
    verdict.append("benchmark " + (f"found {found!r}" if found else "NOT FOUND"))
    print("SUMMARY: " + " | ".join(verdict))
    if result["adjusted"] is False:
        print("WARNING: prices look UNADJUSTED -- reconcile against the bhavcopy "
              "corporate-action detector before a backfill.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="read-only Breeze Connect connectivity probe")
    ap.add_argument("--symbol", default="RELIANCE", help="NSE symbol to test (default RELIANCE)")
    ap.add_argument("--days", type=int, default=20, help="lookback window in calendar days")
    ap.add_argument("--token-path", default=DEFAULT_SESSION_PATH)
    args = ap.parse_args()
    configure_logging()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
