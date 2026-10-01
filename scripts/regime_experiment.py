"""Experiment: do per-regime models beat one model with the regime feature?

Motivation
----------
Momentum's information coefficient **flips sign** between bull and bear regimes
while delivery's holds. If a single model must learn one function across both
regimes, it may be averaging two contradictory relationships. Splitting the
model per regime would let each learn its own.

But splitting has an obvious cost: **the bear regime is only ~20% of sessions**,
so a regime-specific training set is small, and a 504-session training window may
contain almost no bear data at all. This experiment measures whether the benefit
outweighs the data loss, rather than assuming either way.

Controlled comparison
---------------------
Identical in every respect except regime conditioning:

* same folds (purged walk-forward, no shuffle),
* same feature set (regime features INCLUDED in both, so the only difference is
  specialisation, not information availability),
* same sample weights (label uniqueness),
* same hyperparameters and seed per engine (see swingml/models.py),
* same decision rule (top decile of predicted profit probability **per date**).

Metrics
-------
Every arm is scored with :mod:`swingml.evaluation`, which reports precision
definition A (barrier profit) and B (net money) together, plus AUC and net
return. Definition C is never printed here. Each run appends its trials to the
ledger (``data/trials.jsonl``); pass ``--skip-trials`` for throwaway smoke runs.

Per-fold per-config metrics are also written to JSON (default
``<dataset_dir>/regime_folds.json``, override with ``--export-json``) so the
**config-averaged existence test** -- arm A's engine-averaged net per fold against
that fold's base rate, paired over folds -- can be computed without re-fitting.
That verdict is printed at the end of the run and stored in the same file.

Engines
-------
``--engine sklearn-hgb lightgbm xgboost`` runs the arms on each engine with
matched hyperparameters, measuring engine-invariance instead of asserting it.
With two or more engines and ``--pbo``, a CSCV Probability-of-Backtest-
Overfitting is computed across the engine x arm configurations.

Usage
-----
    .venv/bin/python scripts/regime_experiment.py
    .venv/bin/python scripts/regime_experiment.py --engine lightgbm
    .venv/bin/python scripts/regime_experiment.py --engine sklearn-hgb lightgbm xgboost --pbo
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
from swingml.evaluation import (
    DEFINITIONS_NOTE,
    SelectionStats,
    evaluate_selections,
    format_stats,
)
from swingml.models import ENGINES, fit_classifier, make_classifier, predict_positive
from swingml.trials import count_trials, record_trial
from swingml.validation import assert_no_overlap, walk_forward_splits

ARMS = ("A", "B")

#: Selection thresholds evaluated for the trials ledger. The primary table uses
#: ``--decile``; the sweep exists because threshold choice is itself a trial.
SWEEP_DECILES = (10, 5)

#: Per-fold per-config metrics, next to the dataset's other artifacts.
EXPORT_FILENAME = "regime_folds.json"


def default_export_path(dataset_dir: str | Path) -> Path:
    """Where per-fold metrics land unless ``--export-json`` overrides it."""
    return Path(dataset_dir) / EXPORT_FILENAME


def config_averaged_existence(records: list[dict], arm: str = "A") -> dict:
    """Arm's engine-averaged net vs base, paired per fold (STATUS section 8 item 1a).

    Averaging the engines before comparing removes engine-selection noise, which
    is what made the single-config headline unreliable (PBO 55%). The verdict is
    deliberately crude and pre-declared: the average must beat that fold's base
    rate in a majority of folds AND on the mean. It is not a significance test;
    the DSR correction still owns that question.
    """
    per_fold: list[dict] = []
    for rec in records:
        nets = [
            c["avg_net"] for c in rec["configs"]
            if c["arm"] == arm and np.isfinite(c["avg_net"])
        ]
        if not nets:
            continue
        mean_net = float(np.mean(nets))
        base_net = float(rec["base"]["avg_net"])
        per_fold.append({
            "fold": rec["fold"], "n_configs": len(nets), "mean_net": mean_net,
            "base_net": base_net, "diff": mean_net - base_net,
        })

    if not per_fold:
        return {"arm": arm, "n_folds": 0, "mean_diff": float("nan"),
                "sd_diff": float("nan"), "folds_improved": 0, "passed": False,
                "per_fold": []}

    diffs = np.asarray([r["diff"] for r in per_fold], dtype=float)
    improved = int((diffs > 0).sum())
    return {
        "arm": arm,
        "n_folds": len(per_fold),
        "mean_diff": float(diffs.mean()),
        "sd_diff": float(diffs.std(ddof=1)) if len(diffs) > 1 else float("nan"),
        "folds_improved": improved,
        "passed": bool(improved > len(diffs) / 2 and diffs.mean() > 0),
        "per_fold": per_fold,
    }


def export_fold_metrics(path: str | Path, records: list[dict], *, meta: dict,
                        existence: dict) -> Path:
    """Write per-fold per-config metrics + the gate-0 verdict as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "scripts/regime_experiment.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meta": meta,
        "folds": records,
        "config_averaged_existence": existence,
    }
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def fit_predict(engine: str, Xtr, ytr, wtr, Xte, params, seed: int = 0) -> np.ndarray:
    """One model on one engine; identical protocol for every arm."""
    model = make_classifier(engine, params, seed=seed)
    fit_classifier(engine, model, Xtr, ytr, wtr)
    return predict_positive(engine, model, Xte)


def arm_b_predictions(engine, train, test, Xtr, ytr, wtr, Xte, pred_a, regime_col, params, min_regime_rows, seed: int = 0) -> tuple[np.ndarray, dict, int]:
    """Per-regime models with fallback to arm A when data-starved."""
    pred_b = np.full(len(test), np.nan)
    fallbacks = 0
    per_regime = {}
    for regime in (0, 1):
        tr_mask = train[regime_col].to_numpy() == regime
        te_mask = test[regime_col].to_numpy() == regime
        if te_mask.sum() == 0:
            continue
        n_tr = int(tr_mask.sum())
        n_pos = int(ytr[tr_mask].sum()) if n_tr else 0
        if n_tr < min_regime_rows or ytr[tr_mask].min() == ytr[tr_mask].max() or n_pos < 20:
            # Not enough history in this regime to specialise honestly.
            pred_b[te_mask] = pred_a[te_mask]
            fallbacks += 1
            per_regime[int(regime)] = f"FALLBACK (n_train={n_tr}, n_pos={n_pos})"
            continue
        pred_b[te_mask] = fit_predict(engine, Xtr[tr_mask], ytr[tr_mask], wtr[tr_mask], Xte[te_mask], params, seed=seed)
        per_regime[int(regime)] = f"trained (n_train={n_tr}, n_pos={n_pos})"
    if np.isnan(pred_b).any():  # any uncovered rows inherit arm A
        pred_b = np.where(np.isnan(pred_b), pred_a, pred_b)
    return pred_b, per_regime, fallbacks


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if getattr(args, "dataset_dir", None):
        cfg.paths.dataset_dir = args.dataset_dir
    regime_col = cfg.experiment.require_regime_col()
    params = cfg.models  # hyperparameters come from YAML, not from the code
    seed = int(getattr(args, "seed", 0) or 0)
    df, feature_cols = load_training_frame(cfg, regime_col)
    horizon = cfg.label.horizon_days

    folds = walk_forward_splits(df, cfg.validation, horizon)
    assert_no_overlap(folds, df["date"], horizon)  # fail loudly on any leak
    if args.max_folds:
        folds = folds[: args.max_folds]
    for f in folds:
        print("  " + f.describe())

    engines = args.engine
    print("\n" + "=" * 104)
    print("EXPERIMENT: per-regime models  vs  one model with regime features")
    print("=" * 104)
    print(f"rows {len(df):,} | features {len(feature_cols)} | folds {len(folds)} | "
          f"target = profit-barrier reached (base {df['target'].mean():.1%})")
    print(f"engines: {', '.join(engines)}")
    print(DEFINITIONS_NOTE)

    # fold_store[fold] = {"fold": Fold, "test": DataFrame, "preds": {engine: {arm: scores}}}
    fold_store: list[dict] = []
    for fold in folds:
        train = df.iloc[fold.train_idx]
        test = df.iloc[fold.test_idx]
        Xtr, ytr, wtr = train[feature_cols], train["target"].to_numpy(), train["uniqueness"].to_numpy()
        Xte = test[feature_cols]
        if ytr.min() == ytr.max():
            print(f"  fold {fold.index}: training target is single-class; skipped")
            continue

        preds: dict[str, dict[str, np.ndarray]] = {}
        for engine in engines:
            pred_a = fit_predict(engine, Xtr, ytr, wtr, Xte, params, seed=seed)
            pred_b, _, _ = arm_b_predictions(
                engine, train, test, Xtr, ytr, wtr, Xte, pred_a, regime_col, params,
                args.min_regime_rows, seed=seed,
            )
            preds[engine] = {"A": pred_a, "B": pred_b}
        fold_store.append({"fold": fold, "test": test, "preds": preds})
        print(f"  fold {fold.index}: fitted {len(engines)} engine(s) x 2 arms")

    primary = engines[0]
    rows: list[dict] = []
    for store in fold_store:
        test, preds = store["test"], store["preds"][primary]
        sa = evaluate_selections(test, preds["A"], decile=args.decile)
        sb = evaluate_selections(test, preds["B"], decile=args.decile)
        rows.append({
            "fold": store["fold"].index,
            "test_start": str(pd.Timestamp(test["date"].min()).date()),
            "test_end": str(pd.Timestamp(test["date"].max()).date()),
            "a": sa, "b": sb,
        })

    res = pd.DataFrame(rows)
    if res.empty:
        print("no usable folds")
        return 1

    print("\n" + "-" * 104)
    print(f"PER-FOLD RESULT  ({primary}, top {100 // args.decile}% of predicted profit probability, per date)")
    print("-" * 104)
    print(f"{'fold':>4s} {'test period':>25s} {'A precA':>8s} {'B precA':>8s} "
          f"{'A precB':>8s} {'B precB':>8s} {'A net':>8s} {'B net':>8s} {'A auc':>7s} {'B auc':>7s}")
    for _, r in res.iterrows():
        sa, sb = r["a"], r["b"]
        print(f"{int(r['fold']):>4d} {r['test_start'] + '..' + r['test_end']:>25s} "
              f"{sa.precision_a:>8.1%} {sb.precision_a:>8.1%} "
              f"{sa.precision_b:>8.1%} {sb.precision_b:>8.1%} "
              f"{sa.avg_net:>8.2%} {sb.avg_net:>8.2%} {sa.auc:>7.4f} {sb.auc:>7.4f}")

    # Per-fold per-config records: every engine x arm, for the pooled-verdict
    # brackets below AND the JSON export / existence test at the end. Built here
    # (not per arm) so no engine's bracket is silently the primary engine's.
    fold_records: list[dict] = []
    for store in fold_store:
        fold, test = store["fold"], store["test"]
        record = {
            "fold": fold.index,
            "test_start": str(pd.Timestamp(test["date"].min()).date()),
            "test_end": str(pd.Timestamp(test["date"].max()).date()),
            "n_train": int(len(fold.train_idx)),
            "n_test": int(len(test)),
            "base": {
                "precision_a": float(test["label"].eq(1).mean()),
                "precision_b": float((test["ret_net"] > 0).mean()),
                "avg_net": float(test["ret_net"].mean()),
            },
            "configs": [],
        }
        for engine in engines:
            for arm in ARMS:
                st = evaluate_selections(test, store["preds"][engine][arm], decile=args.decile)
                record["configs"].append({
                    "engine": engine, "arm": arm,
                    "precision_a": st.precision_a, "precision_b": st.precision_b,
                    "avg_net": st.avg_net, "auc": st.auc,
                    "n_selected": st.n_selected,
                    "base_precision_a": st.base_precision_a,
                    "base_precision_b": st.base_precision_b,
                    "base_avg_net": st.base_avg_net,
                })
        fold_records.append(record)

    def per_fold_mean_net(engine: str, arm: str) -> float:
        """That engine's own per-fold mean net -- not the primary engine's."""
        vals = [
            c["avg_net"]
            for rec in fold_records for c in rec["configs"]
            if c["engine"] == engine and c["arm"] == arm
        ]
        return float(np.nanmean(vals))

    print("\n" + "=" * 104)
    print("POOLED VERDICT  (per-fold mean; pooled-over-rows in brackets)")
    print("=" * 104)
    # Stack every fold's test rows for a single pooled evaluation per engine/arm.
    pooled_stats: dict[tuple[str, str], SelectionStats] = {}
    for engine in engines:
        for arm in ARMS:
            tests, scores = [], []
            for store in fold_store:
                tests.append(store["test"])
                scores.append(store["preds"][engine][arm])
            pooled_stats[(engine, arm)] = evaluate_selections(
                pd.concat(tests, ignore_index=True), np.concatenate(scores), decile=args.decile
            )

    for engine in engines:
        for arm in ARMS:
            st = pooled_stats[(engine, arm)]
            print(format_stats(st, label=f"{engine} arm {arm}") +
                  f"   [per-fold mean net {per_fold_mean_net(engine, arm):+.3%}]")

    # Base rates come from the whole test block (selection-free).
    all_test = pd.concat([s["test"] for s in fold_store], ignore_index=True)
    print(f"  {'base rate (all test rows)':36s} "
          f"A {all_test['label'].eq(1).mean():6.1%}   "
          f"B {(all_test['ret_net'] > 0).mean():6.1%}   "
          f"net {all_test['ret_net'].mean():+.3%}")

    dprec = res["b"].map(lambda s: s.precision_a) - res["a"].map(lambda s: s.precision_a)
    dnet = res["b"].map(lambda s: s.avg_net) - res["a"].map(lambda s: s.avg_net)
    dauc = res["b"].map(lambda s: s.auc) - res["a"].map(lambda s: s.auc)
    print(f"\n  B - A ({primary}): precision-A {dprec.mean():+.2%} (sd {dprec.std():.2%}), "
          f"net {dnet.mean():+.3%} (sd {dnet.std():.2%}), AUC {dauc.mean():+.4f}")
    print(f"  folds where B > A: precision {int((dprec > 0).sum())}/{len(res)}, "
          f"AUC {int((dauc > 0).sum())}/{len(res)}")

    # -- threshold sweep (every threshold examined is a trial) -----------------
    print("\n" + "-" * 104)
    print("THRESHOLD SWEEP  (pooled over all folds; each cell is one trial for the DSR ledger)")
    print("-" * 104)
    sweep_rows = []
    for d in SWEEP_DECILES:
        for engine in engines:
            for arm in ARMS:
                tests, scores = [], []
                for store in fold_store:
                    tests.append(store["test"])
                    scores.append(store["preds"][engine][arm])
                st = evaluate_selections(pd.concat(tests, ignore_index=True),
                                         np.concatenate(scores), decile=d, auc=False)
                sweep_rows.append({"decile": d, "engine": engine, "arm": arm, "st": st})
                print(f"  top {100 // d:>2d}%  {engine:12s} arm {arm}  "
                      f"A {st.precision_a:6.1%}  B {st.precision_b:6.1%}  net {st.avg_net:+.3%}  n {st.n_selected:,d}")

    # -- regime breakdown (primary engine) -------------------------------------
    print("\n" + "-" * 104)
    print(f"TEST-BLOCK BREAKDOWN BY REGIME  (1 = {regime_col}, primary engine)")
    print("-" * 104)
    for regime, name in ((1, "bull"), (0, "bear")):
        pa, pb, pb0 = [], [], []
        for store, (_, r) in zip(fold_store, res.iterrows()):
            m = store["test"][regime_col].to_numpy() == regime
            if m.sum() < 100:
                continue
            fa = evaluate_selections(store["test"][m], store["preds"][primary]["A"][m], decile=args.decile, auc=False)
            fb = evaluate_selections(store["test"][m], store["preds"][primary]["B"][m], decile=args.decile, auc=False)
            if np.isfinite(fa.precision_a) and np.isfinite(fb.precision_a):
                pa.append(fa.precision_a); pb.append(fb.precision_a); pb0.append(fa.base_precision_a)
        if pa:
            print(f"  {name} test sessions: folds {len(pa)}  base {np.mean(pb0):.1%}  "
                  f"A {np.mean(pa):.1%}  B {np.mean(pb):.1%}   B-A {np.mean(pb) - np.mean(pa):+.2%}")
        else:
            print(f"  {name} test sessions: not enough independent folds to report")

    # -- config-averaged existence test + per-fold export -----------------------
    existence = config_averaged_existence(fold_records, arm="A")
    print("\n" + "-" * 104)
    print("CONFIG-AVERAGED EXISTENCE TEST  (gate 0 -- STATUS section 8 item 1a; arm A, engine-averaged per fold)")
    print("-" * 104)
    if existence["n_folds"] == 0:
        print("  no folds with a finite engine-averaged net; gate 0 cannot be evaluated")
    else:
        print(f"  folds better than base: {existence['folds_improved']}/{existence['n_folds']}   "
              f"mean net diff {existence['mean_diff']:+.3%} (sd {existence['sd_diff']:.3%})")
        print(f"  => GATE 0 {'PASS' if existence['passed'] else 'FAIL'} "
              "(needs a majority of folds AND a positive mean; not a significance test)")

    export_path = None if args.no_export_json else (
        args.export_json or default_export_path(cfg.paths.dataset_dir)
    )
    if export_path:
        written = export_fold_metrics(
            export_path, fold_records,
            meta={
                "dataset_dir": str(cfg.paths.dataset_dir),
                "engines": list(engines),
                "primary_engine": primary,
                "decile": args.decile,
                "n_folds": len(fold_records),
                "n_features": len(feature_cols),
                "n_rows": int(len(df)),
                "regime_col": regime_col,
                "min_regime_rows": args.min_regime_rows,
                "seed": seed,
                "base_target_rate": float(df["target"].mean()),
            },
            existence=existence,
        )
        print(f"\n  per-fold metrics written to {written}")

    # PBO is computed by scripts/pbo_experiment.py, which refits on CPCV
    # paths with mirror pairing (a walk-forward fold has no complement, so
    # the OUT ranking cannot be built from this experiment's folds).

    verdict_arm = "single model + regime feature" if dprec.mean() <= 0.002 or (dprec > 0).sum() <= len(res) / 2 \
        else "per-regime models"
    print(f"\n  => CONCLUSION: no reliable advantage for splitting. Prefer: {verdict_arm}")

    # -- trials ledger ----------------------------------------------------------
    if not args.skip_trials:
        n_trials = len(engines) * len(ARMS) * len(SWEEP_DECILES)
        head = pooled_stats[(primary, "A")]
        entry = record_trial(
            script="regime_experiment.py",
            dataset=str(cfg.paths.dataset_dir),
            engine="multi" if len(engines) > 1 else primary,
            question="per-regime models vs single model with regime feature (definitions A and B)",
            config={
                "engines": list(engines), "decile": args.decile, "sweep_deciles": list(SWEEP_DECILES),
                "min_regime_rows": args.min_regime_rows, "n_features": len(feature_cols),
                "train_days": cfg.validation.train_days, "test_days": cfg.validation.test_days,
            },
            metrics={
                "precision_a": head.precision_a, "precision_b": head.precision_b,
                "avg_net": head.avg_net, "auc": head.auc,
                "base_precision_a": head.base_precision_a, "base_precision_b": head.base_precision_b,
                "b_minus_a_precision_a": float(dprec.mean()), "b_minus_a_net": float(dnet.mean()),
                "config_avg_arm_a_mean_net_diff": existence["mean_diff"],
                "config_avg_arm_a_folds_improved": existence["folds_improved"],
                "config_avg_arm_a_passed": existence["passed"],
            },
            n_folds=len(fold_store),
            trials_added=n_trials,
        )
        c = count_trials()
        print(f"\nledger: +{entry['trials_added']} trials -> {c['total']} total "
              f"(data/trials.jsonl); DSR correction will divide by this count")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-regime vs single-model A/B test")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset-dir", default=None,
                    help="dataset directory override (e.g. data/datasets_liquidity)")
    ap.add_argument("--engine", nargs="+", default=["sklearn-hgb"], choices=ENGINES,
                    help="one or more engines; first is the primary for the per-fold table")
    ap.add_argument("--decile", type=int, default=10, help="selection fraction 1/decile")
    ap.add_argument("--min-regime-rows", type=int, default=300,
                    help="minimum training rows in a regime before specialising")
    ap.add_argument("--max-folds", type=int, default=None, help="limit folds (smoke runs)")
    ap.add_argument("--seed", type=int, default=0, help="model seed (default 0; vary it to test seed robustness)")
    ap.add_argument("--skip-trials", action="store_true", help="do not append to the trials ledger")
    ap.add_argument("--export-json", default=None,
                    help=f"per-fold metrics path (default <dataset-dir>/{EXPORT_FILENAME})")
    ap.add_argument("--no-export-json", action="store_true",
                    help="do not write per-fold metrics")
    args = ap.parse_args()
    configure_logging()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
