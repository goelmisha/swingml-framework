"""Experiment: does meta-labelling beat the single model at matched selection rate?

The gate, stated before the measurement: filtered precision must beat the
primary's, paired per fold. This script fits, on identical purged walk-forward
folds and identical features:

* **baseline** -- the current single model, top 1/``--decile`` per date;
* **meta** -- a recall-first primary (``meta.primary_frac`` of the cross-section,
  scored out-of-fold), then a secondary trained on the primary's positive calls
  whose label is definition B (``ret_net > 0``).

Both arms select the **same number of names** (the same 1/``--decile``), so the
comparison is precision at matched n, not "trade less and look better". The
secondary's probability is used as a within-pool reranker via
:func:`swingml.metalabel.fit_meta_predict`, and both arms are scored with the one
implementation in :mod:`swingml.evaluation` (definitions A and B together).

Pre-registered prediction: a lift of a few points of precision-A at matched n.
If the lift is marginal, meta-labelling is dropped -- not tuned.

Usage
-----
    .venv/bin/python scripts/meta_experiment.py --dataset-dir data/datasets_liquidity_fh
    .venv/bin/python scripts/meta_experiment.py --dataset-dir data/datasets_liquidity_fh \\
        --engine lightgbm --max-folds 3 --skip-trials
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from swingml.config import configure_logging, load_config
from swingml.dataset import load_training_frame
from swingml.evaluation import DEFINITIONS_NOTE, evaluate_selections, format_stats
from swingml.metalabel import fit_meta_predict
from swingml.models import ENGINES
from swingml.trials import count_trials, record_trial
from swingml.validation import assert_no_overlap, walk_forward_splits

EXPORT_FILENAME = "meta_folds.json"


def dataset_label_mode(dataset_dir: str | Path) -> str | None:
    """The target mode the labels on disk were actually built with.

    The config carries a ``label.mode`` too, but a config edit after labelling
    would make the report describe a target the model never saw.
    """
    diag = Path(dataset_dir) / "label_diagnostics.json"
    if not diag.exists():
        return None
    try:
        return json.loads(diag.read_text(encoding="utf-8")).get("mode")
    except (OSError, json.JSONDecodeError):
        return None


def _fit_baseline(engine, train, test, feature_cols, params, seed):
    """The current single model, fitted exactly as the live pipeline does."""
    from swingml.models import fit_predict

    return fit_predict(
        engine, train[feature_cols], train["target"].to_numpy(),
        train["uniqueness"].to_numpy(), test[feature_cols], params, seed=seed,
    )


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.dataset_dir:
        cfg.paths.dataset_dir = args.dataset_dir
    regime_col = cfg.experiment.require_regime_col()
    horizon = int(cfg.label.horizon_days)
    seed = int(args.seed)

    df, feature_cols = load_training_frame(cfg, regime_col)
    folds = walk_forward_splits(df, cfg.validation, horizon)
    assert_no_overlap(folds, df["date"], horizon)
    if args.max_folds:
        folds = folds[: args.max_folds]

    sampling = cfg.label.sampling
    on_disk = dataset_label_mode(cfg.paths.dataset_dir) or cfg.label.mode
    print("=" * 108)
    print("EXPERIMENT: meta-labelling (L1)  vs  single model, at matched selection rate")
    print("=" * 108)
    print(f"rows {len(df):,} | features {len(feature_cols)} | folds {len(folds)} | "
          f"label mode {on_disk} (on disk) | sampling {sampling}")
    print(f"primary_frac {cfg.meta.primary_frac:.2f} (recall pool) -> trade top "
          f"{100 // args.decile}% | OOF groups {cfg.meta.oof_groups} | engine {args.engine}")
    print(f"meta-label: ret_net > 0 (definition B). target base {df['target'].mean():.1%}, "
          f"net-positive base {(df['ret_net'] > 0).mean():.1%}")
    print(DEFINITIONS_NOTE)

    rows: list[dict] = []
    for fold in folds:
        train = df.iloc[fold.train_idx]
        test = df.iloc[fold.test_idx]
        if train["target"].nunique() < 2:
            print(f"  fold {fold.index}: single-class training target; skipped")
            continue
        base_scores = _fit_baseline(args.engine, train, test, feature_cols, cfg.models, seed)
        meta = fit_meta_predict(
            args.engine, train, test, feature_cols,
            meta_cfg=cfg.meta, params=cfg.models, validation_cfg=cfg.validation,
            horizon=horizon, decile=args.decile, seed=seed,
        )
        sb = evaluate_selections(test, base_scores, decile=args.decile)
        sm = evaluate_selections(test, meta.final_scores, decile=args.decile)
        rows.append({
            "fold": fold.index,
            "test_start": str(pd.Timestamp(test["date"].min()).date()),
            "test_end": str(pd.Timestamp(test["date"].max()).date()),
            "n_test": int(len(test)),
            "n_pool": int(meta.n_train_pool),
            "fallback": bool(meta.fallback),
            "base": sb, "meta": sm,
        })
        print(f"  fold {fold.index}: test {rows[-1]['test_start']}..{rows[-1]['test_end']} "
              f"| pool {meta.n_train_pool:,}{' (FALLBACK)' if meta.fallback else ''} "
              f"| precA base {sb.precision_a:.1%} -> meta {sm.precision_a:.1%} "
              f"| precB {sb.precision_b:.1%} -> {sm.precision_b:.1%}")

    if not rows:
        print("no usable folds")
        return 1

    print("\n" + "-" * 108)
    print("PER-FOLD RESULT  (matched top "
          f"{100 // args.decile}% per date)")
    print("-" * 108)
    print(f"{'fold':>4s} {'period':>24s} {'base precA':>11s} {'meta precA':>11s} "
          f"{'base precB':>11s} {'meta precB':>11s} {'base net':>10s} {'meta net':>10s}")
    for r in rows:
        b, m = r["base"], r["meta"]
        print(f"{r['fold']:>4d} {r['test_start'] + '..' + r['test_end']:>24s} "
              f"{b.precision_a:>11.1%} {m.precision_a:>11.1%} "
              f"{b.precision_b:>11.1%} {m.precision_b:>11.1%} "
              f"{b.avg_net:>10.2%} {m.avg_net:>10.2%}")

    diff_a = np.array([r["meta"].precision_a - r["base"].precision_a for r in rows], dtype=float)
    diff_b = np.array([r["meta"].precision_b - r["base"].precision_b for r in rows], dtype=float)
    diff_net = np.array([r["meta"].avg_net - r["base"].avg_net for r in rows], dtype=float)

    def _sign(d: np.ndarray, name: str) -> None:
        wins = int((d > 0).sum())
        n = int(np.isfinite(d).sum())
        sd = float(d.std(ddof=1)) if n > 1 else float("nan")
        t = float(d.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else float("nan")
        print(f"  {name:12s} mean {d.mean():+.3%} | median {np.median(d):+.3%} | "
              f"wins {wins}/{n} | sd {sd:.3%} | t {t:+.2f}")

    print("\n" + "=" * 108)
    print("PAIRED VERDICT  (meta - baseline, per fold)")
    print("=" * 108)
    _sign(diff_a, "precision-A")
    _sign(diff_b, "precision-B")
    _sign(diff_net, "net/period")

    gate = bool(diff_a.mean() > 0 and (diff_a > 0).sum() > len(diff_a) / 2)
    print(f"\n  L1 gate (mean > 0 AND majority of folds): "
          f"{'PASS' if gate else 'FAIL'}   |   pre-registered prediction +2-4 pts precision-A; "
          f"drop if < +1 pt")
    print(f"  mean precision-A lift {diff_a.mean():+.2%}")

    # Pooled, selection-free base rates for reference.
    all_test = pd.concat([df.iloc[f.test_idx] for f in folds], ignore_index=True)
    print(f"  base rates (all test rows): A {all_test['label'].eq(1).mean():.1%} | "
          f"B {(all_test['ret_net'] > 0).mean():.1%} | net {all_test['ret_net'].mean():+.3%}")

    out = Path(cfg.paths.dataset_dir) / EXPORT_FILENAME
    payload = {
        "generated_by": "scripts/meta_experiment.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": args.engine,
        "decile": args.decile,
        "label_mode": on_disk,
        "sampling": sampling,
        "primary_frac": cfg.meta.primary_frac,
        "folds": [
            {
                "fold": r["fold"], "test_start": r["test_start"], "test_end": r["test_end"],
                "n_test": r["n_test"], "n_pool": r["n_pool"], "fallback": r["fallback"],
                "base": r["base"].as_row(), "meta": r["meta"].as_row(),
            }
            for r in rows
        ],
        "paired": {
            "precision_a_lift_mean": float(diff_a.mean()),
            "precision_b_lift_mean": float(diff_b.mean()),
            "net_lift_mean": float(diff_net.mean()),
            "gate_passed": gate,
        },
    }
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nper-fold metrics -> {out}")

    if not args.skip_trials:
        record_trial(
            script="meta_experiment",
            dataset=str(cfg.paths.dataset_dir),
            engine=args.engine,
            question="meta-labelling vs single model at matched selection rate (L1)",
            config={
                "decile": args.decile, "primary_frac": cfg.meta.primary_frac,
                "oof_groups": cfg.meta.oof_groups, "label_mode": on_disk,
                "sampling": sampling,
            },
            metrics={
                "precision_a_lift": float(diff_a.mean()),
                "precision_b_lift": float(diff_b.mean()),
                "net_lift": float(diff_net.mean()),
                "gate_passed": gate,
            },
            n_folds=len(rows),
        )
        print(f"ledger: {count_trials(cfg.paths.dataset_dir)['total']} trials")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="measure the L1 meta-labelling gate")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--engine", default="sklearn-hgb", choices=ENGINES)
    ap.add_argument("--decile", type=int, default=10, help="trading selection fraction 1/decile")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--skip-trials", action="store_true")
    args = ap.parse_args()
    configure_logging()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
