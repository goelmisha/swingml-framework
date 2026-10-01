"""CSCV Probability of Backtest Overfitting across model configurations.

What this measures
------------------
Walk-forward gives ONE chronological path, so one experiment is one sample.
This script instead refits every configuration on all C(N, N/2) CPCV paths of
the timeline and computes the Probability of Backtest Overfitting (Bailey et
al. 2015): pick the best configuration on a path's test groups (IN), then
check whether it still beats the median configuration on the complementary
sessions (OUT). The share of paths where the "best" finishes below-median OUT
is the PBO -- 0.5 means the backtest ranking carried no out-of-sample
information.

Configurations are (engine, arm) pairs -- arm A = one model with the regime
feature, arm B = per-regime models -- so the same controlled A/B measured by
``regime_experiment.py`` is re-asked as a fragility question: does the
preferred arm keep winning when the data it won on is excluded?

CSCV requires symmetric geometry (k = N/2): a path's OUT region must itself
be a path. N=6 groups of ~244 sessions each on the full sample, with k=3, gives
C(6,3) = 20 paths, each session in test 10 times -- about 40 model fits per
configuration. On the full dataset this is the heavy job; run it on the EC2
box (see EC2.md).

Usage
-----
    .venv/bin/python scripts/pbo_experiment.py --groups 4 --test-groups 2   # smoke
    .venv/bin/python scripts/pbo_experiment.py                              # N=6, k=3
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from swingml.config import configure_logging, load_config
from swingml.dataset import load_training_frame
from swingml.evaluation import evaluate_selections
from swingml.models import ENGINES
from swingml.pbo import cscv_from_path_metrics
from swingml.trials import count_trials, record_trial
from swingml.validation import assert_no_overlap_cpcv, cpcv_splits

ARMS = ("A", "B")


def fit_predict(engine: str, Xtr, ytr, wtr, Xte, params, seed: int = 0) -> np.ndarray:
    from swingml.models import fit_classifier, make_classifier, predict_positive

    model = make_classifier(engine, params, seed=seed)
    fit_classifier(engine, model, Xtr, ytr, wtr)
    return predict_positive(engine, model, Xte)


def arm_predictions(engine, train, test, feature_cols, regime_col, params, min_regime_rows) -> dict[str, np.ndarray]:
    """Arm A (single model) and arm B (per-regime, fallback to A) scores."""
    Xtr, ytr, wtr = train[feature_cols], train["target"].to_numpy(), train["uniqueness"].to_numpy()
    Xte = test[feature_cols]
    pred_a = fit_predict(engine, Xtr, ytr, wtr, Xte, params)
    pred_b = np.full(len(test), np.nan)
    for regime in (0, 1):
        tr_mask = train[regime_col].to_numpy() == regime
        te_mask = test[regime_col].to_numpy() == regime
        if te_mask.sum() == 0:
            continue
        n_tr = int(tr_mask.sum())
        n_pos = int(ytr[tr_mask].sum()) if n_tr else 0
        if n_tr < min_regime_rows or ytr[tr_mask].min() == ytr[tr_mask].max() or n_pos < 20:
            pred_b[te_mask] = pred_a[te_mask]
            continue
        pred_b[te_mask] = fit_predict(engine, Xtr[tr_mask], ytr[tr_mask], wtr[tr_mask], Xte[te_mask], params)
    if np.isnan(pred_b).any():
        pred_b = np.where(np.isnan(pred_b), pred_a, pred_b)
    return {"A": pred_a, "B": pred_b}


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if getattr(args, "dataset_dir", None):
        cfg.paths.dataset_dir = args.dataset_dir
    regime_col = cfg.experiment.require_regime_col()
    params = cfg.models  # hyperparameters come from YAML, not from the code
    df, feature_cols = load_training_frame(cfg, regime_col)
    horizon = cfg.label.horizon_days

    if args.n_test_groups != args.n_groups // 2:
        print(f"error: CSCV needs symmetric geometry (k == N/2); got N={args.n_groups}, k={args.n_test_groups}")
        return 1

    paths = cpcv_splits(df, cfg.validation, horizon,
                        n_groups=args.n_groups, n_test_groups=args.n_test_groups)
    assert_no_overlap_cpcv(paths, df["date"], horizon)  # fail loudly on any leak
    engines = args.engine
    configs = [(e, a) for e in engines for a in ARMS]
    print(f"cpcv: {len(paths)} paths (N={args.n_groups}, k={args.n_test_groups}) x "
          f"{len(configs)} configs ({', '.join(engines)}) x 2 arms")
    print(f"rows {len(df):,} | features {len(feature_cols)} | horizon {horizon}")
    print(f"total model fits: {len(paths) * len(engines) * 2}")

    path_metrics: dict[tuple, dict] = {}
    for path in paths:
        train = df.iloc[path.train_idx]
        test = df.iloc[path.test_idx]
        for engine in engines:
            preds = arm_predictions(engine, train, test, feature_cols, regime_col, params,
                                    args.min_regime_rows)
            for arm, scores in preds.items():
                st = evaluate_selections(test, scores, decile=args.decile, auc=False)
                path_metrics[(path.index, (engine, arm))] = {
                    "avg_net": st.avg_net,
                    "precision_a": st.precision_a,
                    "precision_b": st.precision_b,
                    "test_groups": path.test_groups,
                }
        print(f"  path {path.index} ({path.test_groups}): done "
              f"({path.n_train_sessions} train / {path.n_test_sessions} test sessions)")

    # Rank configurations on avg net return of the top-decile selection.
    pbo = cscv_from_path_metrics(path_metrics, configs,
                                 n_groups=args.n_groups, n_test_groups=args.n_test_groups)
    print("\n" + "=" * 104)
    print(f"PBO (CSCV over {pbo.n_paths} paths, {pbo.n_configs} configurations, ranked on avg net)")
    print("=" * 104)
    print(f"  {pbo.summary()}")
    ok = "GOOD" if pbo.pbo < 0.25 else ("MARGINAL" if pbo.pbo < 0.5 else "BAD")
    print(f"  reading: <25% low overfitting risk | 25-50% marginal | >=50% the ranking is a coin flip -> {ok}")
    print("\n  per-path logits (positive = IS winner underperformed OUT):")
    print("  " + "  ".join(f"{v:+.2f}" if np.isfinite(v) else "  n/a" for v in pbo.logits))

    if not args.skip_trials:
        n_fits = len(paths) * len(engines) * len(ARMS)
        record_trial(
            script="pbo_experiment.py",
            dataset=str(cfg.paths.dataset_dir),
            engine="multi" if len(engines) > 1 else engines[0],
            question=f"CSCV PBO across engine x arm (N={args.n_groups}, k={args.n_test_groups})",
            config={
                "engines": list(engines), "n_groups": args.n_groups,
                "n_test_groups": args.n_test_groups, "decile": args.decile,
                "n_paths": len(paths),
            },
            metrics={"pbo": pbo.pbo},
            n_folds=len(paths),
            # Each (engine, arm) evaluated across all paths is one configuration
            # trial; the paths are resamples, not independent configurations.
            trials_added=len(configs),
        )
        c = count_trials()
        print(f"\nledger: +{len(configs)} trials ({n_fits} model fits) -> {c['total']} total (data/trials.jsonl)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="CSCV Probability of Backtest Overfitting")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset-dir", default=None,
                    help="dataset directory override (e.g. data/datasets_liquidity)")
    ap.add_argument("--engine", nargs="+", default=["sklearn-hgb"], choices=ENGINES)
    ap.add_argument("--groups", type=int, default=6, dest="n_groups",
                    help="CPCV groups N (default 6; must be even)")
    ap.add_argument("--test-groups", type=int, default=3, dest="n_test_groups",
                    help="test groups per path k (default 3 = N/2, required for CSCV)")
    ap.add_argument("--decile", type=int, default=10)
    ap.add_argument("--min-regime-rows", type=int, default=300)
    ap.add_argument("--skip-trials", action="store_true")
    args = ap.parse_args()
    configure_logging()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
