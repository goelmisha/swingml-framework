"""Step 4 scorecard: Sharpe, Sortino, drawdown, profit factor -- and the DSR.

Motivation
----------
The existence test asked whether the engine-averaged arm-A selection beat the
same-date universe mean. That is a ranking result. It is **not** a profitability
result, and it says nothing about how many attempts it took to find.

This script produces the missing half:

* the walk-forward selection turned into an equal-weight portfolio series,
  scored for Sharpe, Sortino, max drawdown and profit factor -- money, not
  precision;
* the Deflated Sharpe Ratio, which asks whether that Sharpe survives the search
  recorded in the trials ledger (`data/trials.jsonl`). With hundreds of trials
  the best Sharpe from pure noise is large, and this is the number that decides
  whether the headline is an edge or a search artefact;
* beta control -- the same periods scored against the index buy-and-hold, plus
  the Sharpe of the beta-hedged residual. A long-only strategy in a rising
  sample can post a Sharpe that is mostly market exposure. This section costs no
  trials, which matters once the DSR bar is the binding constraint.

Friction is already inside ``ret_net``; it is never charged twice. Arm A
(single model + regime feature) is the configuration the docs settled on --
regime splitting was measured and rejected, so only the accepted arm is scored
here.

Overlap discipline
------------------
Every ``ret_net`` is a ``horizon``-session holding return, so the series samples
one period every ``horizon`` sessions. Sampling every session instead counts each
trade ten times, inflating both the Sharpe and the cumulative return.
Annualisation follows the period length (``252 / horizon``), never a bare 252.

Caveats this script prints rather than hides
--------------------------------------------
* The trial-Sharpe variance is estimated from the configurations *this run*
  evaluates. The ledger's trials span features and horizons this script does
  not re-run, so the variance is a **lower bound** and the DSR is therefore an
  optimistic (upper) bound on significance.
* Folds tile seven contiguous test blocks over one strongly rising regime, and
  four of the seven fold base rates are negative. Excess over the same-date
  universe mean is the only benchmark here; factor beta is not neutralised.

Usage
-----
    .venv/bin/python scripts/scorecard.py --dataset-dir data/datasets_liquidity
    .venv/bin/python scripts/scorecard.py --dataset-dir data/datasets_liquidity --skip-trials
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd

from swingml.config import configure_logging, load_config
from swingml.costs import ImpactModel, describe as describe_impact, impact_cost_per_trade, positions_by_date
from swingml.dataset import load_training_frame
from swingml.evaluation import DEFINITIONS_NOTE, add_forward_returns, top_fraction_mask
from swingml.models import ENGINES, fit_predict
from swingml.scorecard import (
    SESSIONS_PER_YEAR,
    benchmark_period_returns,
    beta_analysis,
    deflated_sharpe_ratio,
    format_beta_control,
    format_scorecard,
    performance_metrics,
    period_returns,
)
from swingml.trials import count_trials, record_trial
from swingml.validation import assert_no_overlap, walk_forward_splits

#: Selection thresholds whose portfolio series feed the trial-Sharpe variance.
SWEEP_DECILES = (10, 5)

#: Friction stress levels for the kill test, in multiples of the configured
#: round-trip cost. 0.25% flat is optimistic for less-liquid names, so the
#: conclusion has to hold at several times that before it means anything.
KILL_FRICTION_MULTIPLIERS = (1.0, 2.0, 3.0)

#: Capacity grid: book size in crore x square-root impact coefficient. Spans
#: small retail in liquid names to an aggressive institutional book.
IMPACT_BOOK_SIZES_CR = (1.0, 10.0, 100.0)
IMPACT_K_VALUES = (0.05, 0.10, 0.20)


def trade_level_stats(block: pd.DataFrame, scores: np.ndarray, decile: int) -> dict:
    """Profit factor and hit rate per *trade* (selected row), not per session.

    STATUS section 7 sets the Step-4 target as "Profit Factor > 1.5 after the
    0.25%", which is a per-trade reading. Both readings are reported because
    the portfolio-level one is what a drawn-down account actually experiences.
    """
    mask = top_fraction_mask(pd.Series(np.asarray(scores, dtype=float)), block["date"], decile)
    r = block["ret_net"].to_numpy(dtype=float)[mask]
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    return {
        "n_trades": int(r.size),
        "mean_net": float(r.mean()) if r.size else float("nan"),
        "hit_rate": float((r > 0).mean()) if r.size else float("nan"),
        "profit_factor": (gains / losses if losses > 0 else float("inf") if gains > 0 else float("nan")),
    }


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if getattr(args, "dataset_dir", None):
        cfg.paths.dataset_dir = args.dataset_dir
    regime_col = cfg.experiment.require_regime_col()
    df, feature_cols = load_training_frame(cfg, regime_col)
    horizon = cfg.label.horizon_days

    folds = walk_forward_splits(df, cfg.validation, horizon)
    assert_no_overlap(folds, df["date"], horizon)
    if args.max_folds:
        folds = folds[: args.max_folds]

    engines = args.engine
    print("=" * 104)
    print("STEP 4 SCORECARD  --  walk-forward portfolio metrics + deflated Sharpe")
    print("=" * 104)
    print(f"dataset {cfg.paths.dataset_dir}")
    print(f"rows {len(df):,} | features {len(feature_cols)} | folds {len(folds)} | "
          f"engines {', '.join(engines)} | decile {args.decile} (top {100 // args.decile}%)")
    print(DEFINITIONS_NOTE)
    print("  NOTE: this script scores a portfolio series. Definition C never appears here.\n")

    # Fold predictions, arm A only (the accepted configuration).
    fold_store: list[dict] = []
    for fold in folds:
        train, test = df.iloc[fold.train_idx], df.iloc[fold.test_idx]
        Xtr, ytr, wtr = train[feature_cols], train["target"].to_numpy(), train["uniqueness"].to_numpy()
        if ytr.min() == ytr.max():
            print(f"  fold {fold.index}: training target is single-class; skipped")
            continue
        preds = {e: fit_predict(e, Xtr, ytr, wtr, test[feature_cols], cfg.models, seed=args.seed)
                 for e in engines}
        fold_store.append({"fold": fold, "test": test, "preds": preds})
        print(f"  fold {fold.index}: fitted {len(engines)} engine(s)")

    if not fold_store:
        print("no usable folds")
        return 1

    # -- portfolio series per engine, folds concatenated in chronological order --
    # stride = horizon: one entry per holding period, never one per session.
    periods_per_year = SESSIONS_PER_YEAR / horizon
    extra_friction = (args.friction_multiplier - 1.0) * cfg.costs.round_trip_cost_pct
    # Fixed-hold (definition C) trade per fold: the no-barrier hold, used for the
    # buy-and-hold benchmark AND the kill test.
    fwd_col = f"fwd_ret_{horizon}"
    for s in fold_store:
        s["test"] = add_forward_returns(s["test"], [horizon])

    series: dict[str, pd.Series] = {}
    for engine in engines:
        parts = [period_returns(s["test"], s["preds"][engine], decile=args.decile, stride=horizon,
                                extra_friction=extra_friction)
                 for s in fold_store]
        series[engine] = pd.concat(parts).sort_index()

    primary = engines[0]
    perf = performance_metrics(series[primary], periods_per_year=periods_per_year,
                               period_sessions=horizon)
    trade = trade_level_stats(
        pd.concat([s["test"] for s in fold_store], ignore_index=True),
        np.concatenate([s["preds"][primary] for s in fold_store]),
        decile=args.decile,
    )

    # -- trial-Sharpe variance (lower bound: only this run's configs) ----------
    trial_sharpes = []
    for engine in engines:
        for d in SWEEP_DECILES:
            parts = [period_returns(s["test"], s["preds"][engine], decile=d, stride=horizon)
                     for s in fold_store]
            trial_sharpes.append(performance_metrics(pd.concat(parts)).sharpe)
    trial_sharpes = np.asarray([s for s in trial_sharpes if np.isfinite(s)], dtype=float)
    trial_var = float(np.var(trial_sharpes, ddof=1)) if trial_sharpes.size > 1 else 0.0

    ledger = count_trials()
    n_trials = int(ledger["total"]) if ledger["total"] else len(trial_sharpes)
    if not ledger["total"]:
        print("  WARNING: no trials ledger found; DSR falls back to this run's config count")

    dsr = deflated_sharpe_ratio(
        sharpe=perf.sharpe, n_obs=perf.n_periods, n_trials=max(n_trials, 2),
        trial_sharpe_variance=trial_var, skew=perf.skew, kurtosis=perf.kurtosis,
    )

    print("\n" + "-" * 104)
    print(f"OUT-OF-SAMPLE PORTFOLIO  (arm A, {primary}, top {100 // args.decile}%, equal weight; "
          f"one {horizon}-session period per selection, non-overlapping)")
    print("-" * 104)
    print(format_scorecard(perf, dsr, label=f"{primary} arm A"))
    cumulative = float(np.prod(1.0 + series[primary].to_numpy()) - 1.0)
    print(f"  cumulative net return over the test blocks: {cumulative:+.2%}")
    print(f"  per-trade: n {trade['n_trades']:,} | mean net {trade['mean_net']:+.3%} "
          f"| hit rate {trade['hit_rate']:.1%} | profit factor {trade['profit_factor']:.2f}")

    print("\n" + "-" * 104)
    print("SAME SERIES, OTHER ENGINES  (engine-invariance of the scorecard)")
    print("-" * 104)
    for engine in engines[1:]:
        p = performance_metrics(series[engine], periods_per_year=periods_per_year,
                                period_sessions=horizon)
        print(f"  {engine:12s} Sharpe {p.sharpe_annualised:6.2f} | Sortino {p.sortino_annualised:6.2f} "
              f"| MDD {p.max_drawdown:6.2%} | PF {p.profit_factor:5.2f} | "
              f"cum net {float(np.prod(1.0 + series[engine].to_numpy()) - 1.0):+.2%}")

    # -- beta control (STATUS section 8 item 7) --------------------------------
    # Costs no trials: it is an analysis of series already produced above.
    print("\n" + "-" * 104)
    print("BETA CONTROL  (how much of that Sharpe is the market?)")
    print("-" * 104)
    def friction_charge(col: str, mult: float) -> float:
        """Charge the round trip once per period.

        ``ret_net`` already carries the 0.25%, so stressing it means adding
        ``(mult - 1) x cost``. ``fwd_ret`` is **gross** -- charging it
        ``(mult - 1)`` silently measured a friction-free benchmark, which is how
        the first bench-mark table managed to look like it charged friction at
        1x while charging nothing.
        """
        cost = cfg.costs.round_trip_cost_pct
        return (mult - 1.0) * cost if col == "ret_net" else mult * cost

    def charge_for(block: pd.DataFrame, col: str, mult: float, decile: int,
                   model: ImpactModel | None) -> float | np.ndarray:
        """Total per-row charge: the flat toll (if the column is gross) + impact.

        Charge must be split by column: ``ret_net`` already carries the flat
        round trip, so adding it again would double-charge.
        """
        flat = friction_charge(col, mult)
        if model is None:
            return flat
        impact = impact_cost_per_trade(block, positions_by_date(block, decile), model)
        return flat + impact

    def pooled(col: str, mult: float, decile: int, scores=None,
               impact: ImpactModel | None = None) -> pd.Series:
        parts = [
            period_returns(s["test"],
                           np.zeros(len(s["test"])) if scores is None else s["preds"][primary],
                           decile=decile, stride=horizon, return_col=col,
                           extra_friction=charge_for(s["test"], col, mult, decile, impact))
            for s in fold_store
        ]
        return pd.concat(parts).sort_index()

    def summarise(ser: pd.Series) -> dict:
        p = performance_metrics(ser.dropna(), periods_per_year=periods_per_year,
                                period_sessions=horizon)
        return {"perf": p,
                "cum": float(np.prod(1.0 + ser.dropna().to_numpy(dtype=float)) - 1.0)}

    # Two benchmarks, because they answer different questions:
    #   same rules  -- identical exit rule and entry convention, so it isolates
    #                  *selection* rather than trade construction;
    #   hold all    -- the opportunity cost you could actually buy (no barriers).
    benchmarks = {
        "pool, same exit rule as the labels": ("ret_net", 1.0),
        f"hold everything, no barriers (fixed {horizon}-session hold)": (fwd_col, 1.0),
    }
    bench_scores = {}
    for name, (col, mult) in benchmarks.items():
        ser = pooled(col, mult, decile=1)
        s = summarise(ser)
        bench_scores[name] = s
        print(f"  BENCHMARK  {name}: {s['perf'].n_periods} periods, mean "
              f"{s['perf'].mean_return:+.3%}/period, Sharpe "
              f"{s['perf'].sharpe_annualised:.2f}, cumulative {s['cum']:+.2%}")
    buy_hold = pooled(fwd_col, 1.0, decile=1).dropna()

    strat_scores = summarise(series[primary])
    print(f"  STRATEGY   {primary} arm A, its own trade: {strat_scores['perf'].n_periods} "
          f"periods, mean {strat_scores['perf'].mean_return:+.3%}/period, Sharpe "
          f"{strat_scores['perf'].sharpe_annualised:.2f}, cumulative {strat_scores['cum']:+.2%}")
    # Like-for-like cumulative: restrict both sides to the periods they share.
    for name, (col, _) in benchmarks.items():
        other = pooled(col, 1.0, decile=1)
        common = series[primary].dropna().index.intersection(other.dropna().index)
        mine = float(np.prod(1.0 + series[primary].reindex(common).dropna().to_numpy()) - 1.0)
        theirs = float(np.prod(1.0 + other.reindex(common).dropna().to_numpy()) - 1.0)
        verdict_ = "BEATS" if mine > theirs else "loses to"
        print(f"    same {len(common)} periods vs "
              f"{name.split(',')[0]}: strategy {mine:+.2%} vs {theirs:+.2%}  =>  "
              f"strategy {verdict_} it")
    if args.beta_vs_pool:
        bh_beta = beta_analysis(series[primary], buy_hold, periods_per_year=periods_per_year)
        print("  strategy vs the no-barrier basket:")
        print(format_beta_control(bh_beta, label=f"{primary} arm A"))

    # -- KILL TEST: does CHOOSING beat HOLDING, on the same trade? -------------
    print("\n" + "-" * 104)
    print("KILL TEST  (does choosing beat holding, on the same trade, at stress costs?)")
    print("-" * 104)
    print("  Verdict requires beating BOTH benchmarks on Sharpe and cumulative return at 1x "
          "friction:\n  selection must add something over the same exit rule, and over an "
          "opportunity cost you\n  could actually buy. Every series is charged the round trip "
          "per period, ``ret_net``'s\n  copy included only once.")
    kill_rows = []
    for mult in KILL_FRICTION_MULTIPLIERS:
        strat = pooled("ret_net", mult, decile=args.decile, scores=True)
        same_rules = pooled("ret_net", mult, decile=1)
        hold_all = pooled(fwd_col, mult, decile=1)
        both = pd.concat({"s": strat, "b": same_rules, "h": hold_all}, axis=1).dropna()
        s_perf = performance_metrics(both["s"], periods_per_year=periods_per_year,
                                     period_sessions=horizon)
        b_perf = performance_metrics(both["b"], periods_per_year=periods_per_year,
                                     period_sessions=horizon)
        h_perf = performance_metrics(both["h"], periods_per_year=periods_per_year,
                                     period_sessions=horizon)
        s_cum = float(np.prod(1.0 + both["s"].to_numpy()) - 1.0)
        b_cum = float(np.prod(1.0 + both["b"].to_numpy()) - 1.0)
        h_cum = float(np.prod(1.0 + both["h"].to_numpy()) - 1.0)
        kill_rows.append({"mult": mult, "n": s_perf.n_periods, "s": s_perf, "b": b_perf,
                          "h": h_perf, "s_cum": s_cum, "b_cum": b_cum, "h_cum": h_cum})
        print(f"  friction x{mult:<4.1f} ({s_perf.n_periods} periods)  strategy "
              f"{s_perf.mean_return:+.3%}/period, Sharpe {s_perf.sharpe_annualised:+.2f}, "
              f"cum {s_cum:+.2%}")
        print(f"                    same rules  {b_perf.mean_return:+.3%}/period, Sharpe "
              f"{b_perf.sharpe_annualised:+.2f}, cum {b_cum:+.2%}   |   hold all  "
              f"{h_perf.mean_return:+.3%}/period, Sharpe {h_perf.sharpe_annualised:+.2f}, "
              f"cum {h_cum:+.2%}")
    verdict = kill_rows[0]
    beats_same = (verdict["s"].sharpe_annualised > verdict["b"].sharpe_annualised
                  and verdict["s_cum"] > verdict["b_cum"])
    beats_hold = (verdict["s"].sharpe_annualised > verdict["h"].sharpe_annualised
                  and verdict["s_cum"] > verdict["h_cum"])
    kill_passed = beats_same and beats_hold
    print(f"  => KILL TEST {'PASSED' if kill_passed else 'FAILED'} at 1x friction "
          f"(vs same-rule pool: {'yes' if beats_same else 'NO'}; "
          f"vs hold-everything: {'yes' if beats_hold else 'NO'})")

    # -- capacity / impact sensitivity ----------------------------------------
    print("\n" + "-" * 104)
    print("CAPACITY STRESS  (flat 0.25% is a toll, not a cost model)")
    print("-" * 104)
    if args.no_capacity:
        print("  capacity grid skipped (--no-capacity)")
    elif "adv_20" not in df.columns:
        print("  adv_20 missing from the feature set; impact cannot be sized here.")
    else:
        print("  Round trip = flat + 2*k*sqrt(position/ADV), position = book / names held.")
        print("  k has no measurement behind it, so this is a grid, not a number: the "
              "conclusion\n  has to hold across cells, not at one hand-picked point.")
        grid_ok = True
        for book in IMPACT_BOOK_SIZES_CR:
            for k in IMPACT_K_VALUES:
                mdl = ImpactModel(book_size_cr=book, impact_k=k)
                st = pooled("ret_net", 1.0, decile=args.decile, scores=True, impact=mdl)
                same = pooled("ret_net", 1.0, decile=1, impact=mdl)
                hold = pooled(fwd_col, 1.0, decile=1, impact=mdl)
                both = pd.concat({"s": st, "b": same, "h": hold}, axis=1).dropna()
                sp = performance_metrics(both["s"], periods_per_year=periods_per_year,
                                         period_sessions=horizon)
                hp = performance_metrics(both["h"], periods_per_year=periods_per_year,
                                         period_sessions=horizon)
                s_cum = float(np.prod(1.0 + both["s"].to_numpy()) - 1.0)
                h_cum = float(np.prod(1.0 + both["h"].to_numpy()) - 1.0)
                wins = sp.sharpe_annualised > hp.sharpe_annualised and s_cum > h_cum
                grid_ok = grid_ok and wins
                print(f"  book Rs {book:>6g}cr  k={k:<4.2f}  strategy "
                      f"{sp.mean_return:+.3%}/period, Sharpe {sp.sharpe_annualised:+.2f}, "
                      f"cum {s_cum:+.2%}   |   hold all {hp.mean_return:+.3%}/period, "
                      f"Sharpe {hp.sharpe_annualised:+.2f}, cum {h_cum:+.2%}   "
                      f"{'strategy wins' if wins else 'STRATEGY LOSES'}")
        print(f"  => capacity verdict at {args.decile}-decile concentration: "
              f"{'holds across the whole grid' if grid_ok else 'DOES NOT hold across the grid'}")
        print("  (still unmodelled: partial fills, borrow, gapping, ADV shrinkage in "
              "stress; treat\n   this as a floor on what impact would cost, not a "
              "capacity measurement.)")

    # Exit alternative: the SAME selection held to the vertical barrier instead of
    # exited at a horizontal one. Only meaningful when the labels are barriers --
    # on a fixed_hold dataset this IS the strategy series above.
    if cfg.label.mode == "barrier":
        alt = pooled(fwd_col, 1.0, decile=args.decile, scores=True).dropna()
        alt_perf = performance_metrics(alt, periods_per_year=periods_per_year,
                                       period_sessions=horizon)
        alt_common = alt.index.intersection(buy_hold.index)
        alt_cum = float(np.prod(1.0 + alt.reindex(alt_common).dropna().to_numpy()) - 1.0)
        bh_cum = float(np.prod(1.0 + buy_hold.reindex(alt_common).dropna().to_numpy()) - 1.0)
        alt_trial = []
        for engine in engines:
            for d in SWEEP_DECILES:
                ser = pd.concat([
                    period_returns(s["test"], s["preds"][engine], decile=d, stride=horizon,
                                   return_col=fwd_col, extra_friction=friction_charge(fwd_col, 1.0))
                    for s in fold_store
                ]).sort_index().dropna()
                alt_trial.append(performance_metrics(ser).sharpe)
        alt_trial = np.asarray([x for x in alt_trial if np.isfinite(x)], dtype=float)
        alt_var = float(np.var(alt_trial, ddof=1)) if alt_trial.size > 1 else 0.0
        alt_dsr = deflated_sharpe_ratio(
            sharpe=alt_perf.sharpe, n_obs=alt_perf.n_periods, n_trials=max(n_trials, 2),
            trial_sharpe_variance=alt_var, skew=alt_perf.skew, kurtosis=alt_perf.kurtosis,
        )
        print("\n  EXIT ALTERNATIVE -- same selection, exit at the vertical barrier instead "
              f"(labels are '{cfg.label.mode}')")
        print(format_scorecard(alt_perf, alt_dsr,
                               label=f"{primary} arm A, fixed {horizon}-session hold"))
        print(f"    vs hold-everything on the same {len(alt_common)} periods: "
              f"{alt_cum:+.2%} vs {bh_cum:+.2%} (friction charged on both)")
        print("    This is a CHANGE OF EXIT RULE, not a re-fit; it is the diagnostic "
              "that motivated")
        print("    `label.mode = fixed_hold`.")

    if args.no_beta:
        print("\n  market beta skipped (--no-beta)")
    else:
        try:
            from swingml.data.cache import DiskCache
            from swingml.data.prices import make_price_provider

            provider = make_price_provider(cfg.data, DiskCache(cfg.paths.cache_dir, "prices"))
            bench = provider.get_benchmark(cfg.data.start, cfg.data.end)
        except Exception as exc:  # noqa: BLE001 - benchmark is optional context
            bench = None
            print(f"  benchmark unavailable ({exc}); beta control skipped")
        if bench is not None and not bench.empty:
            bench_series = benchmark_period_returns(
                bench, series[primary].index, horizon,
                friction=args.friction_multiplier * cfg.costs.round_trip_cost_pct,
            )
            analysis = beta_analysis(series[primary], bench_series, periods_per_year=periods_per_year)
            print(f"  benchmark {cfg.data.bench_symbol}, buy-and-hold over the same windows, "
                  f"friction charged to match")
            print(format_beta_control(analysis, label=f"{primary} arm A"))

    # -- is the beta-hedged residual significant? ------------------------------
    # Same construction as the headline DSR, applied to the market-neutral series
    # instead of the long-only one.
    bh_beta = beta_analysis(series[primary], buy_hold, periods_per_year=periods_per_year)
    residual = bh_beta.get("residual_series")
    if residual is not None and len(residual) > 2:
        resid_perf = performance_metrics(residual, periods_per_year=periods_per_year,
                                         period_sessions=horizon)
        resid_sharpes = []
        for engine in engines:
            for d in SWEEP_DECILES:
                s_series = pd.concat([
                    period_returns(st["test"], st["preds"][engine], decile=d, stride=horizon)
                    for st in fold_store
                ]).sort_index()
                an = beta_analysis(s_series, buy_hold, periods_per_year=periods_per_year)
                r = an.get("residual_series")
                if r is not None and np.isfinite(an["residual_sharpe_annualised"]):
                    resid_sharpes.append(performance_metrics(
                        r, periods_per_year=periods_per_year).sharpe)
        resid_sharpes = np.asarray([s for s in resid_sharpes if np.isfinite(s)], dtype=float)
        resid_var = float(np.var(resid_sharpes, ddof=1)) if resid_sharpes.size > 1 else 0.0
        resid_dsr = deflated_sharpe_ratio(
            sharpe=resid_perf.sharpe, n_obs=resid_perf.n_periods, n_trials=max(n_trials, 2),
            trial_sharpe_variance=resid_var, skew=resid_perf.skew, kurtosis=resid_perf.kurtosis,
        )
        print("\n" + "-" * 104)
        print("MARKET-NEUTRAL VARIANT  (beta-hedged against the pool buy-and-hold)")
        print("-" * 104)
        print(format_scorecard(resid_perf, resid_dsr, label=f"{primary} arm A, beta-hedged"))
        print(f"  residual trial-Sharpe sd {math.sqrt(resid_var) * math.sqrt(periods_per_year):.2f} "
              f"annualised over {len(resid_sharpes)} configs; hedge beta "
              f"{bh_beta['beta']:.2f}, R-squared {bh_beta['r_squared']:.2f}")

    print("\n" + "-" * 104)
    print("DEFLATION INPUTS  (read these before believing the Sharpe)")
    print("-" * 104)
    print(f"  trials in ledger            : {ledger['total']} "
          f"({ledger['n_runs']} runs; scripts {', '.join(sorted(ledger['by_script'])) or '-'})")
    print(f"  trial-Sharpe sd this run    : {math.sqrt(trial_var) * math.sqrt(periods_per_year):.2f} annualised "
          f"({len(trial_sharpes)} configs: {len(engines)} engines x {list(SWEEP_DECILES)})")
    print(f"  observations                : {perf.n_periods} non-overlapping {horizon}-session periods")
    print(f"  skew / kurtosis             : {perf.skew:+.2f} / {perf.kurtosis:.2f}")
    print("  trial-Sharpe sd is a LOWER bound (features/horizons in the ledger are not re-run "
          "here),\n  so the DSR below is an OPTIMISTIC bound on significance. Friction is inside "
          "ret_net.")

    if not args.skip_trials:
        n_added = len(engines) * len(SWEEP_DECILES)
        record_trial(
            script="scorecard.py",
            dataset=str(cfg.paths.dataset_dir),
            engine="multi" if len(engines) > 1 else primary,
            question="walk-forward portfolio Sharpe (arm A) and its deflated Sharpe",
            config={"engines": list(engines), "decile": args.decile,
                    "sweep_deciles": list(SWEEP_DECILES), "seed": args.seed,
                    "train_days": cfg.validation.train_days, "test_days": cfg.validation.test_days},
            metrics={"sharpe_annualised": perf.sharpe_annualised,
                     "sortino_annualised": perf.sortino_annualised,
                     "max_drawdown": perf.max_drawdown, "profit_factor": perf.profit_factor,
                     "cumulative_net": cumulative, "dsr": dsr["dsr"],
                     "expected_max_sharpe": dsr["expected_max_sharpe"]},
            n_folds=len(fold_store),
            trials_added=n_added,
        )
        print(f"\nledger: +{n_added} trials -> {count_trials()['total']} total "
              f"(data/trials.jsonl)")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Step-4 scorecard + deflated Sharpe")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset-dir", default=None,
                    help="dataset directory override (e.g. data/datasets_liquidity)")
    ap.add_argument("--engine", nargs="+", default=list(ENGINES), choices=ENGINES,
                    help="one or more engines; first is the headline")
    ap.add_argument("--decile", type=int, default=10, help="selection fraction 1/decile")
    ap.add_argument("--seed", type=int, default=0, help="model seed")
    ap.add_argument("--max-folds", type=int, default=None, help="limit folds (smoke runs)")
    ap.add_argument("--skip-trials", action="store_true", help="do not append to the trials ledger")
    ap.add_argument("--no-beta", action="store_true", help="skip the market beta control")
    ap.add_argument("--beta-vs-pool", action="store_true",
                    help="also regress on the universe buy-and-hold basket, not just the index")
    ap.add_argument("--friction-multiplier", type=float, default=1.0,
                    help="scale the round-trip cost for the headline series (1.0 = as configured)")
    ap.add_argument("--no-capacity", action="store_true", help="skip the impact sensitivity grid")
    args = ap.parse_args()
    configure_logging()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
