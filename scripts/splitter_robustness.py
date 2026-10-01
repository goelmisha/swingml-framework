"""Splitter-geometry robustness: does the retargeted edge depend on the fold shape?

Why this script exists
----------------------
STATUS section 8 item 8 asked for seed/configuration robustness and was marked
**VOID**: the three configured engines have no stochastic component (no
subsampling, no column sampling, no early stopping), so a different ``--seed``
reproduces the identical predictions -- verified as
``max|p(seed=0) - p(seed=7)| = 0.000e+00`` for sklearn-hgb, lightgbm and
xgboost. A robustness claim that cannot fail is not evidence.

What a single seed *does* actually fix is the walk-forward geometry: train 504 /
gap 10 / test 126 over seven folds. That is the one measurement-design choice in
the headline result that nobody has varied, and it is not innocuous -- the
retargeted label holds for exactly 10 sessions, so purge and embargo are
horizon-floored at 10, and every fold's train block is ~2 years against a 6
month test. If the +1.9%/period edge only exists for that particular cut, it is
a property of the split, not of the market.

What it measures
----------------
Each geometry rebuilds the folds, refits the same engine(s) on the same
``fixed_hold`` features, and reports the strategy's non-overlapping period
series next to the two benchmarks the kill test uses:

* **same-rule pool** -- every name under the labels' own trade, which isolates
  selection from trade construction;
* **hold everything** -- the fixed-horizon hold of the whole universe, gross
  minus one round trip, i.e. the opportunity cost that could actually be bought.

Verdict per geometry: the strategy must beat **both** on Sharpe and on
cumulative return over the periods the two series share. The headline claim
survives only if that holds for every geometry, so a single FAIL is reported
loudly rather than averaged away.

Geometry moves ONE axis at a time off the configured base (longer/shorter train,
shorter/longer test, wider purge+embargo). The purge/embargo variants are the
leak-sensitivity check: if the edge is overlap leakage, widening the purge is
what makes it disappear.

Usage
-----
    .venv/bin/python scripts/splitter_robustness.py --dataset-dir data/datasets_liquidity_fh
    .venv/bin/python scripts/splitter_robustness.py --dataset-dir data/datasets_liquidity_fh --skip-trials
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from swingml.config import ValidationConfig, configure_logging, load_config
from swingml.dataset import load_training_frame
from swingml.models import ENGINES, fit_predict
from swingml.evaluation import add_forward_returns
from swingml.scorecard import SESSIONS_PER_YEAR, performance_metrics, period_returns
from swingml.trials import count_trials, record_trial
from swingml.validation import assert_no_overlap, walk_forward_splits


@dataclass(frozen=True)
class Geometry:
    """One walk-forward cut. Every field maps 1:1 onto ``ValidationConfig``."""

    name: str
    train_days: int
    test_days: int
    purge_gap_days: int
    embargo_days: int

    def to_config(self, base: ValidationConfig) -> ValidationConfig:
        return dataclasses.replace(
            base,
            train_days=self.train_days,
            test_days=self.test_days,
            purge_gap_days=self.purge_gap_days,
            embargo_days=self.embargo_days,
        )


def dataset_label_mode(dataset_dir: str | Path) -> str | None:
    """The target mode the **dataset's labels** were built with.

    The config carries a ``label.mode`` too, but the labels on disk are what the
    model actually trains on: they may predate a config edit. Reading the mode
    from ``label_diagnostics.json`` (written by ``label-dataset``) makes the
    header state the target that was measured rather than the one that was
    configured, and a disagreement is worth a warning.
    """
    diag = Path(dataset_dir) / "label_diagnostics.json"
    if not diag.exists():
        return None
    try:
        with open(diag, "r", encoding="utf-8") as fh:
            return json.load(fh).get("mode")
    except (OSError, json.JSONDecodeError):
        return None


def geometry_variants(base: ValidationConfig) -> list[Geometry]:
    """The configured geometry plus five variants that move one axis each.

    Derived from ``base`` rather than hard-coded, so editing
    ``validation:`` in ``config/config.yaml`` moves the whole sweep with it.
    """
    train = int(base.train_days)
    test = int(base.test_days)
    return [
        Geometry("configured (base)", train, test, base.purge_gap_days, base.embargo_days),
        Geometry("longer train (1.5x)", round(train * 1.5), test,
                 base.purge_gap_days, base.embargo_days),
        Geometry("shorter train (0.75x)", max(1, round(train * 0.75)), test,
                 base.purge_gap_days, base.embargo_days),
        Geometry("shorter test (0.5x)", train, max(1, test // 2),
                 base.purge_gap_days, base.embargo_days),
        Geometry("longer test (1.5x)", train, round(test * 1.5),
                 base.purge_gap_days, base.embargo_days),
        Geometry("wide purge+embargo (20/20)", train, test, 20, 20),
    ]


def _cum(series: pd.Series) -> float:
    arr = series.dropna().to_numpy(dtype=float)
    return float(np.prod(1.0 + arr) - 1.0) if arr.size else float("nan")


def score_geometry(
    df: pd.DataFrame,
    feature_cols: list[str],
    geo: Geometry,
    cfg,
    engines: list[str],
    decile: int,
    seed: int,
    max_folds: int | None,
) -> dict:
    """Fit one engine over one geometry and score strategy vs both benchmarks."""
    horizon = cfg.label.horizon_days
    vcfg = geo.to_config(cfg.validation)
    folds = walk_forward_splits(df, vcfg, horizon)
    if max_folds:
        folds = folds[:max_folds]
    assert_no_overlap(folds, df["date"], horizon)

    fwd_col = f"fwd_ret_{horizon}"
    fold_store: list[dict] = []
    for fold in folds:
        train, test = df.iloc[fold.train_idx], df.iloc[fold.test_idx]
        ytr = train["target"].to_numpy()
        if ytr.min() == ytr.max():
            continue
        test = add_forward_returns(test, [horizon])
        preds = {
            e: fit_predict(e, train[feature_cols], ytr, train["uniqueness"].to_numpy(),
                           test[feature_cols], cfg.models, seed=seed)
            for e in engines
        }
        fold_store.append({"fold": fold, "test": test, "preds": preds})

    if not fold_store:
        raise RuntimeError(f"geometry {geo.name!r}: no usable folds")

    periods_per_year = SESSIONS_PER_YEAR / horizon
    cost = cfg.costs.round_trip_cost_pct
    primary = engines[0]

    def concat(parts: list[pd.Series]) -> pd.Series:
        return pd.concat(parts).sort_index()

    strategy = concat([
        period_returns(s["test"], s["preds"][primary], decile=decile, stride=horizon)
        for s in fold_store
    ])
    # Same-rule pool: every name, labels' own trade (ret_net already carries the
    # round trip, so no extra charge). Hold-everything: gross fwd return, so the
    # round trip is charged here -- exactly once per period.
    same_rules = concat([
        period_returns(s["test"], np.zeros(len(s["test"])), decile=1, stride=horizon)
        for s in fold_store
    ])
    hold_all = concat([
        period_returns(s["test"], np.zeros(len(s["test"])), decile=1, stride=horizon,
                       return_col=fwd_col, extra_friction=cost)
        for s in fold_store
    ])

    out = {
        "geometry": geo,
        "n_folds": len(fold_store),
        "effective_gap": vcfg.effective_purge_days(horizon),
        "strategy": performance_metrics(strategy, periods_per_year=periods_per_year,
                                        period_sessions=horizon),
        "strategy_cum": _cum(strategy),
        "benchmarks": {},
        "verdict": "PASS",
    }
    benchmarks = {"same-rule pool": same_rules, "hold everything": hold_all}
    if len(engines) > 1:
        out["other_engines"] = {
            e: performance_metrics(
                concat([period_returns(s["test"], s["preds"][e], decile=decile, stride=horizon)
                        for s in fold_store]),
                periods_per_year=periods_per_year, period_sessions=horizon)
            for e in engines[1:]
        }
    for name, ser in benchmarks.items():
        perf = performance_metrics(ser, periods_per_year=periods_per_year, period_sessions=horizon)
        # Like-for-like: only the periods the two series actually share.
        common = strategy.dropna().index.intersection(ser.dropna().index)
        mine = _cum(strategy.reindex(common))
        theirs = _cum(ser.reindex(common))
        beats = (out["strategy"].sharpe > perf.sharpe) and (mine > theirs)
        out["benchmarks"][name] = {"perf": perf, "cum": theirs, "n_common": len(common),
                                   "strategy_cum": mine, "beats": bool(beats)}
        if not beats:
            out["verdict"] = "FAIL"
    return out


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.dataset_dir:
        cfg.paths.dataset_dir = args.dataset_dir
    regime_col = cfg.experiment.require_regime_col()
    df, feature_cols = load_training_frame(cfg, regime_col)
    horizon = cfg.label.horizon_days

    geometries = geometry_variants(cfg.validation)
    engines = args.engine
    print("=" * 104)
    print("SPLITTER-GEOMETRY ROBUSTNESS  --  does the edge survive the fold shape?")
    print("=" * 104)
    on_disk = dataset_label_mode(cfg.paths.dataset_dir)
    target = on_disk or cfg.label.mode
    if on_disk and on_disk != cfg.label.mode:
        print(f"WARNING: dataset labels were built with mode={on_disk!r} but the config says "
              f"{cfg.label.mode!r}; measuring the labels on disk.")
    print(f"dataset {cfg.paths.dataset_dir}   |   target (labels on disk) {target}   |   "
          f"horizon {horizon} sessions")
    print(f"rows {len(df):,} | features {len(feature_cols)} | engines {', '.join(engines)} | "
          f"decile {args.decile} (top {100 // args.decile}%) | {len(geometries)} geometries")
    print("NOTE: predictions are identical across seeds in these engines (no stochastic "
          "component),\n      so geometry, not the seed, is the untested axis. Every series "
          "is charged the round\ntrip once per period: ret_net carries it, the gross "
          "hold-everything series is charged here.\n")

    results: list[dict] = []
    for geo in geometries:
        try:
            res = score_geometry(df, feature_cols, geo, cfg, engines, args.decile,
                                 args.seed, args.max_folds)
        except RuntimeError as exc:
            print(f"  {geo.name:28s} SKIPPED -- {exc}")
            continue
        results.append(res)
        p = res["strategy"]
        print(f"  {geo.name:28s} folds {res['n_folds']:2d} | periods {p.n_periods:3d} | "
              f"mean {p.mean_return:+.3%}/period | Sharpe {p.sharpe_annualised:5.2f} | "
              f"cum {res['strategy_cum']:+9.2%} | gap {res['effective_gap']:2d}")
        for name, b in res["benchmarks"].items():
            print(f"      vs {name:20s} Sharpe {b['perf'].sharpe_annualised:5.2f} | "
                  f"cum {b['cum']:+9.2%} over {b['n_common']:3d} shared periods  =>  "
                  f"{'BEATS' if b['beats'] else 'LOSES TO'}")
        if "other_engines" in res:
            for engine, perf in res["other_engines"].items():
                print(f"      {engine:12s} Sharpe {perf.sharpe_annualised:5.2f}")

    print("\n" + "-" * 104)
    print("VERDICT ACROSS GEOMETRIES")
    print("-" * 104)
    if not results:
        print("  no geometry produced folds; nothing to conclude")
        return 1
    sharpes = [r["strategy"].sharpe_annualised for r in results]
    cums = [r["strategy_cum"] for r in results]
    pool = [r["benchmarks"]["same-rule pool"]["perf"].sharpe_annualised for r in results]
    hold = [r["benchmarks"]["hold everything"]["perf"].sharpe_annualised for r in results]
    for r in results:
        p = r["strategy"]
        print(f"  {r['geometry'].name:28s} Sharpe {p.sharpe_annualised:5.2f} vs pool "
              f"{r['benchmarks']['same-rule pool']['perf'].sharpe_annualised:5.2f} vs hold "
              f"{r['benchmarks']['hold everything']['perf'].sharpe_annualised:5.2f} | "
              f"cum {r['strategy_cum']:+9.2%}  =>  {r['verdict']}")
    n_pass = sum(1 for r in results if r["verdict"] == "PASS")
    print(f"\n  strategy Sharpe  min {min(sharpes):.2f} / median {float(np.median(sharpes)):.2f} "
          f"/ max {max(sharpes):.2f};  cumulative {min(cums):+.2%} ... {max(cums):+.2%}")
    print(f"  beat-both verdict: {n_pass}/{len(results)} geometries PASS")
    print("  A single FAIL is a real finding, not noise to be averaged away: it means the "
          "headline\ndepends on a fold shape nobody chose on principle."
          if n_pass < len(results) else
          "  The edge is not a property of any one fold shape; it survives every "
          "geometry tested.")

    if not args.skip_trials:
        record_trial(
            script="splitter_robustness.py",
            dataset=str(cfg.paths.dataset_dir),
            engine="multi" if len(engines) > 1 else engines[0],
            question="does the walk-forward edge survive the splitter geometry?",
            config={"geometries": {r["geometry"].name: {
                        "train_days": r["geometry"].train_days,
                        "test_days": r["geometry"].test_days,
                        "purge_gap_days": r["geometry"].purge_gap_days,
                        "embargo_days": r["geometry"].embargo_days} for r in results},
                    "decile": args.decile, "seed": args.seed,
                    "label_mode": target},
            metrics={"sharpe_min": min(sharpes), "sharpe_median": float(np.median(sharpes)),
                     "sharpe_max": max(sharpes), "n_pass": n_pass, "n_geometries": len(results)},
            n_folds=sum(r["n_folds"] for r in results),
            trials_added=len(results),
        )
        print(f"\nledger: +{len(results)} trials -> {count_trials()['total']} total "
              f"(data/trials.jsonl)")
    return 0 if n_pass == len(results) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="splitter-geometry robustness of the walk-forward edge")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset-dir", default=None,
                    help="dataset directory override (e.g. data/datasets_liquidity_fh)")
    ap.add_argument("--engine", nargs="+", default=list(ENGINES), choices=ENGINES,
                    help="one or more engines; first is the headline")
    ap.add_argument("--decile", type=int, default=10, help="selection fraction 1/decile")
    ap.add_argument("--seed", type=int, default=0, help="model seed")
    ap.add_argument("--max-folds", type=int, default=None, help="limit folds (smoke runs)")
    ap.add_argument("--skip-trials", action="store_true", help="do not append to the trials ledger")
    args = ap.parse_args()
    configure_logging()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
