"""Grade the frozen predictions, once their holding periods have closed.

The reader half of the forward test. It takes the append-only log written by
``scripts/paper_trade.py``, checks the hash chain, fetches the prices that have
appeared since, and reports the forward Sharpe next to the two benchmarks the
kill test uses -- the same trade on every candidate, and the Nifty over the same
windows, both charged the same friction.

Two things it refuses to do, on purpose:

* **It does not touch the trials ledger.** The forward run searches nothing, so
  no new multiple-testing penalty applies; the configuration was charged once by
  the backtest that chose it. Logging one trial per day would inflate the DSR's
  search penalty for a decision that involved no search.
* **It does not report a DSR.** With a handful of periods a deflated Sharpe is a
  number with no power, and quoting one would dress up the wait. It reports the
  Sharpe with a t-statistic and says plainly how many independent periods exist.

Only ``is_grid`` records are scored, so the periods do not overlap. If two
scored signal dates turn out to be closer than the hold length, that is reported
as a problem rather than averaged away.

Usage
-----
    .venv/bin/python scripts/paper_score.py --dataset-dir data/datasets_liquidity_fh
    .venv/bin/python scripts/paper_score.py --log paper/predictions.jsonl --out paper/scored.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from swingml.config import configure_logging, load_config
from swingml.data.cache import DiskCache
from swingml.data.prices import make_price_provider
from swingml.paper import (
    DEFAULT_LOG_PATH,
    STATUS_CLOSED,
    STATUS_NO_SESSION,
    format_report,
    grid_problems,
    latest_per_signal_date,
    load_records,
    score_records,
    summarise,
    verify_chain,
)

#: Days of price history fetched before the earliest scored signal date, so the
#: entry session is always inside the window.
_PRICE_LOOKBACK_DAYS = 10


def _symbols_to_fetch(records, calendar: pd.DatetimeIndex) -> list[str]:
    """Candidate names for the grid records whose period has closed.

    Scoped deliberately: fetching all ~460 names for every record ever written
    would be wasteful, and only closed grid periods contribute to the series.
    """
    if len(calendar) == 0:
        return []
    position = {d: i for i, d in enumerate(calendar)}
    wanted: set[str] = set()
    for rec in records:
        if not rec.is_grid:
            continue
        i = position.get(pd.Timestamp(rec.signal_date))
        if i is None or i + rec.horizon_days >= len(calendar):
            continue
        wanted.update(rec.scores)
    return sorted(wanted)


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.dataset_dir:
        cfg.paths.dataset_dir = args.dataset_dir
    horizon = int(cfg.label.horizon_days)
    friction = float(cfg.costs.round_trip_cost_pct)
    log_path = Path(args.log)

    all_records = load_records(log_path)
    print("=" * 100)
    print("PAPER SCORE  --  forward results of the frozen predictions")
    print("=" * 100)
    print(f"log {log_path}   |   horizon {horizon} sessions   |   friction {friction:.2%} per period")
    if not all_records:
        # Normal on day one, not an error: the writer has not run yet.
        print("\n  the log is empty; run scripts/paper_trade.py first.")
        return 0

    problems = verify_chain(all_records)
    print(f"records {len(all_records)} | chain: {'intact' if not problems else 'PROBLEMS'}")
    for p in problems:
        print(f"  !! {p}")
    if problems:
        print("\n  refusing to score a log whose chain does not verify.")
        return 1

    records = list(latest_per_signal_date(all_records).values())
    n_superseded = len(all_records) - len(records)
    if n_superseded:
        print(f"  ({n_superseded} superseded record(s) excluded; the log keeps them for the chain)")

    # The benchmark supplies the session calendar, so it is fetched first and the
    # symbol set is scoped from it rather than guessed.
    cache = DiskCache(cfg.paths.cache_dir, "prices")
    provider = make_price_provider(cfg.data, cache)
    start = (pd.Timestamp(min(r.signal_date for r in records))
             - pd.Timedelta(days=_PRICE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    try:
        benchmark = provider.get_benchmark(start, end)
    except Exception as exc:  # a cron job must report, not traceback
        print(f"\n  failed to fetch the benchmark for {start}..{end}: {exc}")
        print("  check the network, then re-run; nothing has been written.")
        return 1
    if benchmark is None or benchmark.empty:
        print(f"\n  no benchmark data in {start}..{end}; cannot build the session calendar.")
        return 1
    calendar = pd.DatetimeIndex(pd.to_datetime(benchmark.index).normalize()).sort_values()

    symbols = _symbols_to_fetch(records, calendar)
    print(f"prices {start}..{end} | calendar {len(calendar)} sessions | "
          f"{len(symbols)} candidate symbols to fetch")
    prices = provider.get_many(symbols, start, end) if symbols else {}
    if not prices:
        # No closed grid period yet, so there is nothing to fetch. A daily cron
        # must not report failure for the ordinary state of waiting.
        print("\n  no closed grid period yet; nothing to score. Re-run after the first "
              f"{horizon}-session hold finishes.")
        return 0

    scored = score_records(records, prices, benchmark=benchmark,
                           friction=friction, horizon=horizon)
    overlaps = grid_problems(scored, calendar, horizon)
    summary = summarise(scored, horizon)

    print()
    print(format_report(scored, summary, horizon, friction, problems=overlaps))

    if "status" in scored.columns:
        tally = scored["status"].value_counts().to_dict()
        print()
        print("  record status: " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items())))

    out = Path(args.out) if args.out else log_path.parent / "scored.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)
    print(f"\n  per-record grades -> {out}")

    closed = scored[scored["status"] == STATUS_CLOSED] if not scored.empty else scored
    if closed is not None and not closed.empty:
        print()
        print("  forward periods so far (signal -> entry -> exit, strategy vs universe):")
        for _, row in closed.iterrows():
            print(f"    {row['signal_date']}  entry {row['entry_date']}  exit {row['exit_date']}  "
                  f"strategy {row['strategy']:+.3%}  universe {row['universe']:+.3%}  "
                  f"market {row.get('market', float('nan')):+.3%}  "
                  f"n {int(row['n_priced_selected'])}/{int(row['n_selected'])}")
    elif not scored.empty and (scored["status"] == STATUS_OPEN).any():
        n_open = int((scored["status"] == STATUS_OPEN).sum())
        print(f"\n  {n_open} period(s) still open -- re-run after their exit sessions close.")
    else:
        n_nosession = int((scored["status"] == STATUS_NO_SESSION).sum())
        if n_nosession:
            print(f"\n  {n_nosession} record(s) have no session in the fetched window; widen the "
                  f"price range or rebuild the cache.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="grade the frozen forward predictions (read-only; places no orders)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset-dir", default=None,
                    help="only used for the price cache location; scoring reads prices directly")
    ap.add_argument("--log", default=DEFAULT_LOG_PATH,
                    help=f"append-only prediction log (default {DEFAULT_LOG_PATH})")
    ap.add_argument("--out", default=None,
                    help="per-record grades CSV (default: alongside the log)")
    args = ap.parse_args()
    configure_logging()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
