"""Freeze today's prediction before tomorrow's open.

Why this script exists
----------------------
Gate 1a (STATUS section 9) is forward paper trading on the ``fixed_hold``
target. This is the writer half: it takes the dataset as it stands this evening,
retrains on every label that is already resolved, ranks the latest session, and
writes the result to an append-only hash-chained log. It never places an order
and never talks to a broker.

What "already resolved" means
-----------------------------
The label for a signal date ``t0`` is only known once its 10-session hold has
finished, so the newest usable training rows sit ``horizon`` sessions behind the
prediction date. The cut is ``max(purge_gap_days, horizon_days)`` -- the same
floor ``ValidationConfig.effective_purge_days`` applies to the walk-forward
folds, read from config rather than hard-coded, so a geometry edit moves both.
That keeps the live cut exactly as strict as the backtest's fold boundary: the
last training label resolves on the prediction date's own close, and the
prediction itself is for the *next* session's open.

Why the grid
------------
One prediction is written for every session (cheap, and it keeps the option of a
tranched variant), but only sessions on a pre-registered non-overlapping grid
are marked ``is_grid``, and only those are scored. A 10-session hold sampled
every session would count the same price path ten times, inflating the Sharpe.
The grid is every ``horizon``-th session counted from the dataset's first
session, so the anchor comes from the config's start date, not from anyone's
judgement about which periods looked good.

Usage
-----
    # prerequisites: the dataset and its labels are current
    .venv/bin/python -m swingml.cli build-dataset --universe liquidity \\
        --start 2020-01-01 --dataset-dir data/datasets_liquidity_fh
    .venv/bin/python -m swingml.cli label-dataset --dataset data/datasets_liquidity_fh \\
        --mode fixed_hold

    .venv/bin/python scripts/paper_trade.py --dataset-dir data/datasets_liquidity_fh
    .venv/bin/python scripts/paper_trade.py --dataset-dir data/datasets_liquidity_fh --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from swingml.config import configure_logging, load_config
from swingml.dataset import DatasetBundle, load_training_frame
from swingml.metalabel import fit_meta_predict
from swingml.models import ENGINES, fit_predict
from swingml.paper import (
    CROSS_SECTION_LOOKBACK,
    DEFAULT_LOG_PATH,
    append_record,
    code_fingerprint,
    config_fingerprint,
    cross_section_ok,
    entry_session_open,
    is_forward_record,
    load_records,
    new_record,
    select_top,
    verify_chain,
)

#: How many calendar days the dataset may lag today before the run warns. A
#: weekend plus a holiday makes three normal; more usually means the rebuild did
#: not happen and the "latest session" is not the latest session.
DEFAULT_MAX_STALENESS_DAYS = 4


def dataset_label_mode(dataset_dir: str | Path) -> str | None:
    """The target mode the labels on disk were built with.

    The config carries a ``label.mode`` too, but the labels on disk are what the
    model trains on and they may predate a config edit. Reading
    ``label_diagnostics.json`` makes the run report the target it actually used.
    """
    diag = Path(dataset_dir) / "label_diagnostics.json"
    if not diag.exists():
        return None
    try:
        with open(diag, "r", encoding="utf-8") as fh:
            return json.load(fh).get("mode")
    except (OSError, json.JSONDecodeError):
        return None


def training_cutoff(
    sessions: pd.Index,
    signal_date: pd.Timestamp,
    purge_sessions: int,
) -> pd.Timestamp:
    """The newest signal date whose label is fully resolved by ``signal_date``.

    ``purge_sessions`` sessions back from the prediction date. Refuses to guess:
    if the dataset is too short to place the cut, that is an error rather than a
    silently weakened backtest.
    """
    ordered = pd.DatetimeIndex(sorted(pd.to_datetime(sessions).unique()))
    position = ordered.get_indexer([pd.Timestamp(signal_date)])[0]
    if position < 0:
        raise ValueError(f"signal date {signal_date} is not in the dataset calendar")
    if position < purge_sessions:
        raise ValueError(
            f"only {position} sessions precede {signal_date}; need at least "
            f"{purge_sessions} to place the purge"
        )
    return ordered[position - purge_sessions]


def grid_flag(session_index: int, horizon: int) -> bool:
    """Pre-registered non-overlapping grid membership for one session.

    Counting from the dataset's first session (pinned by ``data.start`` in
    config), every ``horizon``-th session is a period start. Decided here, at
    freeze time, and stored in the record -- never recomputed after the results
    are visible.
    """
    return horizon > 0 and session_index % horizon == 0


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.dataset_dir:
        cfg.paths.dataset_dir = args.dataset_dir
    horizon = int(cfg.label.horizon_days)

    regime_col = cfg.experiment.require_regime_col()
    df, feature_cols = load_training_frame(cfg, regime_col)
    feats, meta = DatasetBundle.load(cfg.paths.dataset_dir)

    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(feats["date"]).unique()))
    signal_date = pd.Timestamp(args.signal_date) if args.signal_date else sessions[-1]
    if signal_date not in sessions:
        raise SystemExit(f"signal date {signal_date.date()} is not a session in the dataset "
                         f"(last session: {sessions[-1].date()})")
    session_index = int(sessions.get_indexer([signal_date])[0])

    on_disk = dataset_label_mode(cfg.paths.dataset_dir)
    target = on_disk or cfg.label.mode
    purge = cfg.validation.effective_purge_days(horizon)
    cutoff = training_cutoff(sessions, signal_date, purge)
    train = df[df["date"] <= cutoff]
    if train.empty:
        raise SystemExit("no training rows before the purge cutoff; rebuild the dataset")

    candidates = feats[feats["date"] == signal_date]
    if candidates.empty:
        raise SystemExit(f"no feature rows for {signal_date.date()}; rebuild the dataset")
    missing = [c for c in feature_cols if c not in candidates.columns]
    if missing:
        raise SystemExit(f"feature columns missing from the prediction frame: {missing[:5]}")

    # Per-symbol prices for the newest session keep arriving after the close, so
    # a cross-section that is thinner than usual means the universe is still
    # filling in -- not that fewer names qualified. Ranking it would trade a
    # different, size-biased universe than the one the backtest measured.
    per_session = feats.groupby("date").size().sort_index()
    prior = per_session.iloc[-(CROSS_SECTION_LOOKBACK + 1):-1]
    section_ok, recent_median = cross_section_ok(len(candidates), prior)

    staleness = (pd.Timestamp.today().normalize() - signal_date).days
    # A record is forward evidence only while the entry session has not opened.
    # The test is about the entry PRICE, so the evening of the signal session
    # and the next morning before 09:15 IST both qualify; later does not.
    is_forward = is_forward_record(signal_date)
    print("=" * 100)
    print("PAPER TRADE  --  freeze the prediction before the open (no orders, no broker)")
    print("=" * 100)
    print(f"dataset      {cfg.paths.dataset_dir}")
    print(f"target       {target} (labels on disk)"
          + (f"   WARNING: config says {cfg.label.mode!r}" if on_disk and on_disk != cfg.label.mode else ""))
    print(f"signal date  {signal_date.date()}  (session {session_index} of {len(sessions)}; "
          f"{staleness} calendar days old)")
    if staleness > args.max_staleness_days:
        print(f"  !! the dataset is {staleness} days behind today; the newest session may not "
              f"be\nthe newest session. Re-run build-dataset + label-dataset first "
              f"(threshold {args.max_staleness_days}d).")
    if not is_forward:
        print(f"  !! BACKFILL: the entry session for {signal_date.date()} opened "
              f"{entry_session_open(signal_date).date()} at 09:15 IST, which has passed.\n"
              f"     The record would be chain-linked but would NOT count as forward "
              f"evidence.\n     Pass --allow-backfill to accept this knowingly.")
    else:
        print(f"  forward window: entry opens {entry_session_open(signal_date).date()} 09:15 IST "
              f"-- not yet traded.")
    print(f"train cut    <= {cutoff.date()}  ({purge} sessions back = "
          f"max(purge_gap_days, horizon_days); the last training label resolves on "
          f"{signal_date.date()})")
    print(f"training     {len(train):,} rows, {train['symbol'].nunique()} symbols")
    print(f"candidates   {len(candidates):,} names on the signal date | engine {args.engine} | "
          f"selector {args.selector} | decile {args.decile} (top {100 // args.decile}%)")
    if args.selector == "meta":
        print(f"meta         primary_frac {cfg.meta.primary_frac:.2f} recall pool -> "
              f"secondary filter, OOF groups {cfg.meta.oof_groups}")
    print(f"cross-section {len(candidates)} vs {CROSS_SECTION_LOOKBACK}-session median "
          f"{recent_median or 'n/a'}  =>  "
          f"{'complete enough' if section_ok else 'THIN'}")
    if not section_ok:
        print(f"  !! THIN CROSS-SECTION: the newest session's prices are still arriving, so this "
              f"\n     would rank a truncated and size-biased universe. Wait for the panel to "
              f"fill in.\n     Pass --allow-incomplete to record it anyway (it will be marked and "
              f"excluded\n     from the forward series).")
    print()

    ytr = train["target"].to_numpy()
    if ytr.min() == ytr.max():
        raise SystemExit("training block has only one target class; cannot fit")
    if args.selector == "meta":
        # Primary (recall-first) -> secondary (does the call pay, net of
        # friction) -> composite score. Same selection rule as the backtest.
        meta = fit_meta_predict(
            args.engine, train, candidates, feature_cols,
            meta_cfg=cfg.meta, params=cfg.models, validation_cfg=cfg.validation,
            horizon=horizon, decile=args.decile, seed=args.seed,
        )
        scores_arr = meta.final_scores
        if meta.fallback:
            print("  !! meta secondary fell back to the primary ranking for this block")
    else:
        scores_arr = fit_predict(
            args.engine, train[feature_cols], ytr, train["uniqueness"].to_numpy(),
            candidates[feature_cols], cfg.models, seed=args.seed,
        )
    scores = {str(s): float(p) for s, p in zip(candidates["symbol"], scores_arr)}
    selected = select_top(scores, decile=args.decile)
    is_grid = grid_flag(session_index, horizon)

    base = float((train["target"] == 1).mean())
    print(f"  ranked {len(scores)} names | base rate {base:.1%} | selected {len(selected)} "
          f"| is_grid {is_grid}")
    print(f"  top 10 by score: {', '.join(s[:14] for s in selected[:10])}")
    if not is_grid:
        print(f"  NOTE: session {session_index} is off-grid, so this record is frozen but "
              f"will NOT\n        enter the headline series (overlapping periods would "
              f"double-count the hold).")
    print()

    if args.dry_run:
        print("--dry-run: nothing written to the log.")
        return 0
    if not is_forward and not args.allow_backfill:
        print("refusing to write a backfilled record; re-run with --allow-backfill to override.")
        return 1
    if not section_ok and not args.allow_incomplete:
        print("refusing to freeze a thin cross-section; re-run with --allow-incomplete to override.")
        return 1

    config_hash, config_payload = config_fingerprint(
        cfg,
        args.engine,
        args.decile,
        selector=args.selector,
        # The dataset's own facts, not the config's defaults: the labels were
        # built by `label-dataset --mode ...` and the panel by an explicit
        # `--universe`, and neither is necessarily the YAML default.
        dataset_facts={
            "label_mode": target,
            "universe_source": meta.get("universe_source"),
        },
    )
    log_path = Path(args.log)
    previous = load_records(log_path)
    supersedes = None
    if any(r.signal_date == str(signal_date.date()) for r in previous):
        if not args.refreeze:
            print(f"  !! a prediction for {signal_date.date()} is already frozen; "
                  f"re-run with --refreeze to supersede it (the old record stays in the log).")
            return 1
        supersedes = next(r.record_hash for r in reversed(previous)
                          if r.signal_date == str(signal_date.date()))

    rec = new_record(
        signal_date=str(signal_date.date()),
        session_index=session_index,
        is_grid=is_grid,
        is_forward=is_forward,
        horizon_days=horizon,
        decile=int(args.decile),
        engine=args.engine,
        selector=args.selector,
        label_mode=target,
        dataset_dir=str(cfg.paths.dataset_dir),
        n_candidates=len(scores),
        selected=selected,
        scores=scores,
        config_hash=config_hash,
        config=config_payload,
        code_hash=code_fingerprint(),
        n_train_rows=int(len(train)),
        train_cutoff=str(cutoff.date()),
        dataset_built_at=meta.get("built_at"),
        n_candidates_recent=int(recent_median),
        cross_section_ok=bool(section_ok),
        supersedes=supersedes,
    )
    written = append_record(log_path, rec)
    problems = verify_chain(load_records(log_path))
    print(f"frozen -> {log_path}")
    print(f"  record_hash {written.record_hash[:16]}... | prev "
          f"{written.prev_hash[:16] or '(genesis)'}... | code {written.code_hash[:12]}")
    print(f"  chain: {'intact' if not problems else 'PROBLEMS: ' + '; '.join(problems)}")
    print()
    print("  Nothing was traded. Grade it once the holding period closes:")
    print(f"    .venv/bin/python scripts/paper_score.py --log {log_path}"
          f" --dataset-dir {cfg.paths.dataset_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="freeze the walk-forward prediction for the latest session (no orders)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset-dir", default=None,
                    help="dataset directory (e.g. data/datasets_liquidity_fh)")
    ap.add_argument("--log", default=DEFAULT_LOG_PATH,
                    help=f"append-only prediction log (default {DEFAULT_LOG_PATH})")
    ap.add_argument("--engine", default="sklearn-hgb", choices=ENGINES,
                    help="engine to freeze (the backtest headline is sklearn-hgb)")
    ap.add_argument("--selector", default="decile", choices=["decile", "meta"],
                    help="decile: single model top 1/decile (default) | "
                         "meta: primary + meta-labelled filter")
    ap.add_argument("--decile", type=int, default=10, help="selection fraction 1/decile")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--signal-date", default=None,
                    help="freeze a specific session instead of the latest (backfill/testing)")
    ap.add_argument("--max-staleness-days", type=int, default=DEFAULT_MAX_STALENESS_DAYS)
    ap.add_argument("--refreeze", action="store_true",
                    help="supersede an existing record for the same signal date")
    ap.add_argument("--allow-backfill", action="store_true",
                    help="write a record for a session whose entry has already traded "
                         "(marked, and excluded from the forward series)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="record a prediction ranked on a cross-section thinner than its "
                         "recent median (marked, and excluded from the forward series)")
    ap.add_argument("--dry-run", action="store_true", help="rank but do not write")
    args = ap.parse_args()
    configure_logging()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
