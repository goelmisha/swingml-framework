"""Feature signal check: do the delivery features actually separate forward returns?

This is a *measurement* script, not part of the pipeline. It exists to answer one
question before the modelling step is committed to:

    Does the delivery block carry information about forward returns, or is it a
    repackaged version of momentum that will just add correlated noise?

Method
------
Everything is measured **cross-sectionally**, per date, which is the only honest
way to evaluate a panel of stocks. Pooling all (date, symbol) rows into one
regression conflates time-series regime drift with cross-sectional signal: if the
whole market rose during the sample, every feature "works".

1. **Information Coefficient (IC)** -- Spearman rank correlation between the
   feature at ``t`` and the forward return, computed *within each date*, then
   averaged over dates. Reported with an IC information ratio and a share of
   positive dates.

2. **Quantile ladder** -- per date, sort symbols into quintiles by the feature
   and average the forward return in each. Monotonicity across Q1..Q5 is the
   signal; a single lucky bucket is not.

3. **Overlap correction (critical).** A 10-day forward return on daily data means
   consecutive dates share 9 of 10 days. The IC series is therefore heavily
   autocorrelated and a naive t-stat is inflated by roughly sqrt(10). Two defences
   are reported:
     - ``t (overlap-adj)``: effective sample size ``n_dates / horizon``
     - ``t (non-overlap)``: IC recomputed on every ``horizon``-th date only

4. **Incremental value** -- delivery features are compared against plain momentum,
   and correlated against it, because a feature that is 0.9 correlated with
   ``roc_20`` adds a column but not information.

Known limits of this measurement (do not over-read the result)
-------------------------------------------------------------
- Only ~400 sessions, so ~40 non-overlapping 10-day periods. Power is low.
- The sample sits in a single regime (Nifty stayed on one side of its 200-EMA
  throughout), so regime-conditional conclusions are impossible here.
- Universe is current Nifty 200 membership (survivorship-biased).

Usage
-----
    .venv/bin/python scripts/signal_check.py
    .venv/bin/python scripts/signal_check.py --feature xs_rank_roc_20 --horizon 10
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from swingml.config import PROJECT_ROOT, configure_logging, load_config
from swingml.dataset import DatasetBundle
# add_forward_returns lives in swingml.evaluation, beside definitions A/B/C, so
# the scorecard's buy-and-hold benchmark and this script cannot diverge.
from swingml.evaluation import add_forward_returns
from swingml.trials import count_trials, record_trial

def resolve_feature_lists(cfg, df) -> tuple[list[str], list[str]]:
    """``(signal, control)`` columns to sweep, from ``experiment.*`` in config.

    Neither list is hard-coded here: a column name that only one feature
    provider emits has no business inside a script. When a list is unset, the
    default is every cross-sectional rank column, because ranks are comparable
    across symbols while raw levels are not.
    """
    present = [c for c in df.columns if c not in ("date", "symbol")]
    default = sorted(c for c in present if c.startswith("xs_rank_"))

    signal = [c for c in (cfg.experiment.signal_features or default) if c in present]
    control = [c for c in (cfg.experiment.control_features or default) if c in present]
    if not signal:
        raise RuntimeError(
            "no signal features to score: set experiment.signal_features in the "
            "config, or build a dataset that has cross-sectional rank columns"
        )
    return signal, control


def daily_ic(df: pd.DataFrame, feature: str, target: str, min_names: int = 10) -> pd.Series:
    """Spearman IC per date, computed from within-date ranks (no pooling).

    Fully vectorised: ranks are taken within each date, then Pearson on ranks is
    evaluated per date via grouped sums. Avoids ``groupby.apply`` entirely.
    """
    d = df[["date", feature, target]].dropna(subset=[feature, target])
    if d.empty:
        return pd.Series(dtype=float)

    g = d.groupby("date", sort=True)
    d["_x"] = g[feature].rank()
    d["_y"] = g[target].rank()
    g = d.groupby("date", sort=True)

    dx = d["_x"] - g["_x"].transform("mean")
    dy = d["_y"] - g["_y"].transform("mean")
    d["_num"] = dx * dy
    d["_dx2"] = dx * dx
    d["_dy2"] = dy * dy

    agg = d.groupby("date", sort=True).agg(
        num=("_num", "sum"), dx2=("_dx2", "sum"), dy2=("_dy2", "sum"), n=("_x", "size")
    )
    agg = agg[agg["n"] >= min_names]
    denom = np.sqrt(agg["dx2"] * agg["dy2"]).replace(0.0, np.nan)
    return (agg["num"] / denom).dropna()


def ic_stats(ic: pd.Series, horizon: int) -> dict:
    """Summarise an IC series with an explicit overlap correction."""
    n = len(ic)
    if n < 5:
        return {}
    mean, std = ic.mean(), ic.std()
    n_eff = max(1.0, n / horizon)                       # overlapping observations
    t_overlap = mean / (std / np.sqrt(n_eff)) if std > 0 else np.nan
    non_overlap = ic.iloc[::horizon]                    # ~independent periods
    t_nonoverlap = (
        non_overlap.mean() / (non_overlap.std() / np.sqrt(len(non_overlap)))
        if len(non_overlap) > 2 and non_overlap.std() > 0 else np.nan
    )
    return {
        "n_dates": n,
        "mean_ic": mean,
        "ic_ir": mean / std if std > 0 else np.nan,
        "pct_pos": (ic > 0).mean(),
        "t_overlap_adj": t_overlap,
        "t_non_overlap": t_nonoverlap,
        "n_non_overlap": len(non_overlap),
    }


def pooled_ic(df: pd.DataFrame, feature: str, target: str) -> float:
    """IC pooled across every row, ignoring the date structure.

    Reported only to show the *wrong* number for a panel. Pooling lets a feature
    earn credit from time-series co-movement -- e.g. delivery% drifting down while
    the market drifts up produces a spurious correlation that has nothing to do
    with picking stocks. That spurious component is exactly what a within-date
    percentile transform removes, which is the entire justification for the
    ``xs_rank_*`` block.
    """
    d = df[[feature, target]].dropna()
    if len(d) < 10:
        return np.nan
    return float(d.corr(method="spearman").iloc[0, 1])


def precision_at_quantile(
    df: pd.DataFrame,
    feature: str,
    target: str,
    q: int = 10,
    friction: float = 0.0,
    side: str = "high",
) -> dict:
    """Tier-1 style metric: if we BUY the names this feature selects each date,
    how often do we win *after* friction, and what is the average net return?

    This is the operationally relevant question -- a positive IC is not the same
    thing as a profit, because the mandatory round trip has to be paid on every
    selection.
    """
    d = df[["date", feature, target]].dropna(subset=[feature, target]).copy()
    if d.empty:
        return {}

    def _bucket(s: pd.Series) -> np.ndarray:
        r = s.rank(method="first")
        return np.minimum(np.floor((r - 1) / len(s) * q), q - 1).astype(int).to_numpy()

    d["_b"] = d.groupby("date", sort=True)[feature].transform(_bucket)
    sel = d[d["_b"] == (q - 1 if side == "high" else 0)]
    net = sel[target] - friction
    base = d[target] - friction
    return {
        "n": len(sel),
        "precision_net": float((net > 0).mean()),
        "base_precision_net": float((base > 0).mean()),
        "avg_net": float(net.mean()),
        "base_avg_net": float(base.mean()),
        "hit_lift": float((net > 0).mean() - (base > 0).mean()),
    }


def quantile_returns(df: pd.DataFrame, feature: str, target: str, q: int = 5) -> pd.Series | None:
    """Mean forward return per within-date quantile bucket."""
    d = df[["date", feature, target]].dropna(subset=[feature, target])
    if d.empty:
        return None

    def _bucket(s: pd.Series) -> np.ndarray:
        r = s.rank(method="first")
        return np.minimum(np.floor((r - 1) / len(s) * q), q - 1).astype(int).to_numpy()

    d["_bucket"] = d.groupby("date", sort=True)[feature].transform(_bucket)
    g = d.groupby("_bucket")[target]
    out = pd.DataFrame({"mean": g.mean(), "median": g.median(), "n": g.size()})
    return out


def report_percentiles(df: pd.DataFrame, horizon: int, target: str, friction: float, horizons: list[int]) -> None:
    """Deep dive on the cross-sectional percentile (``xs_rank_*``) features."""
    rank_feats = [c for c in df.columns if c.startswith("xs_rank_")]
    if not rank_feats:
        return

    print("\n" + "=" * 104)
    print("PERCENTILE FEATURES (xs_rank_*) -- do they separate forward returns?")
    print("=" * 104)

    # 1. A within-date percentile transform is monotone, and Spearman IC only
    #    depends on within-date ordering -- so the IC MUST be identical. Any
    #    difference would indicate a bug in the rank or IC computation.
    print("\n[1] Rank-invariance check: within-date percentile vs raw feature, same IC?")
    prefix = "xs_rank_"
    pairs = [(c[len(prefix):], c) for c in rank_feats if c[len(prefix):] in df.columns]
    for raw, rk in pairs:
        a, b = daily_ic(df, raw, target), daily_ic(df, rk, target)
        both = pd.concat([a, b], axis=1).dropna()
        diff = (both.iloc[:, 0] - both.iloc[:, 1]).abs().max() if len(both) else np.nan
        print(f"  {rk:38s} max|IC_raw - IC_rank| = {diff:.2e}  -> {'identical (expected)' if diff < 1e-12 else 'DIFFERS'}")

    # 2. Pooled vs per-date IC. The percentile transform should strip the
    #    spurious time-series component out of the pooled number.
    print("\n[2] Pooled IC (WRONG for a panel) vs per-date IC (correct) -- what the transform fixes")
    print(f"  {'feature':38s} {'pooled_IC':>10s} {'per_date_IC':>12s} {'retained':>9s}")
    for raw, rk in pairs:
        p_raw, p_rk = pooled_ic(df, raw, target), pooled_ic(df, rk, target)
        d_rk = daily_ic(df, rk, target).mean()
        print(f"  {rk:38s} {p_raw:>10.4f} {d_rk:>12.4f} {(d_rk / p_raw if p_raw else np.nan):>9.2f}")

    # 3. The operationally relevant Tier-1 metric, net of the 0.25% round trip.
    print(f"\n[3] BUY the top decile each date: precision AFTER {friction:.2%} friction")
    print(f"  {'feature':38s} {'precision':>10s} {'base':>8s} {'lift':>8s} {'avg_net':>9s} {'base_net':>9s} {'n':>7s}")
    rows = []
    for f in rank_feats:
        st = precision_at_quantile(df, f, target, q=10, friction=friction)
        if st:
            rows.append((f, st))
    for f, st in sorted(rows, key=lambda kv: -kv[1]["precision_net"]):
        print(f"  {f:38s} {st['precision_net']:>10.1%} {st['base_precision_net']:>8.1%} "
              f"{st['hit_lift']:>+8.1%} {st['avg_net']:>9.2%} {st['base_avg_net']:>9.2%} {st['n']:>7d}")

    # 4. Does the delivery percentile hold up across horizons?
    print("\n[4] Delivery-percentile IC across horizons (is the signal horizon-stable?)")
    sel = [c for c in rank_feats if "deliv" in c]
    print(f"  {'feature':38s}" + "".join(f"{str(h) + 'd':>16s}" for h in horizons))
    for f in sel:
        cells = []
        for h in horizons:
            st = ic_stats(daily_ic(df, f, f"fwd_ret_{h}"), h)
            cells.append(f"{st.get('mean_ic', np.nan):+.4f}({st.get('t_non_overlap', np.nan):+.2f})")
        print(f"  {f:38s}" + "".join(f"{c:>16s}" for c in cells))


def barrier_trade_check(
    df: pd.DataFrame, features: list[str], decile: int = 10,
    labels_dir: str | None = None,
) -> None:
    """Definition-B check: the SAME top-decile selection, scored as a barrier trade.

    Definition C above (fixed 10-session hold) is NOT the trade the labeller
    produces: the barrier trade exits at whichever of +2xATR / -1xATR / the
    vertical barrier is hit first. Here the top-decile selection per date is
    joined to that trade's outcome, so precision is definition B --
    P(ret_net > 0) -- directly comparable with the model's B, never with C.
    """
    from pathlib import Path

    base = Path(labels_dir) if labels_dir else PROJECT_ROOT / "data" / "datasets"
    cand = base / "labels.parquet"
    if not cand.exists():
        print("\n(labels.parquet not found -- definition-B check skipped; run `swingml label-dataset`)")
        return
    labels = pd.read_parquet(cand)
    j = df[["date", "symbol"]].merge(
        labels[["date", "symbol", "label", "ret_net"]], on=["date", "symbol"], how="inner"
    )
    if j.empty:
        print("\n(definition-B check skipped: features and labels share no (date, symbol) keys)")
        return
    # Feature values come from the feature matrix; attach them to the joined keys.
    feat_avail = [f for f in features if f in df.columns]
    j = j.merge(df[["date", "symbol"] + feat_avail], on=["date", "symbol"], how="left")

    print("\n" + "=" * 104)
    print(f"DEFINITION B -- same top-{100 // decile}% selections scored as BARRIER trades  "
          f"(P(ret_net > 0); base B = {(j['ret_net'] > 0).mean():.1%} over {len(j):,} labelled rows)")
    print("=" * 104)
    print("  definition B is the model's metric: exit at first barrier, friction included.")
    print("  NOT comparable with definition C above (fixed-hold), which rides the full 10-session drift.")
    print(f"  {'feature':38s} {'prec_B':>8s} {'base_B':>8s} {'lift':>8s} {'avg_net':>9s} {'n':>8s}")
    rows = []
    for f in feat_avail:
        d = j[["date", f, "ret_net"]].dropna()
        if d.empty:
            continue
        pct = d.groupby("date")[f].rank(pct=True, method="first")
        sel = d[pct > 1.0 - 1.0 / decile]
        if sel.empty:
            continue
        rows.append((
            f,
            float((sel["ret_net"] > 0).mean()),
            float((d["ret_net"] > 0).mean()),
            float(sel["ret_net"].mean()),
            len(sel),
        ))
    for f, pb, bb, an, n in sorted(rows, key=lambda r: -r[1]):
        print(f"  {f:38s} {pb:>8.1%} {bb:>8.1%} {pb - bb:>+8.1%} {an:>9.2%} {n:>8,d}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-sectional feature signal check")
    ap.add_argument("--horizon", type=int, default=10, help="forward horizon in sessions")
    ap.add_argument("--horizons", type=int, nargs="+", default=[3, 5, 10, 15])
    ap.add_argument("--feature", default=None, help="show the quantile ladder for one feature")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset-dir", default=None,
                    help="dataset directory override (e.g. data/datasets_liquidity)")
    ap.add_argument("--skip-def-b", action="store_true", help="skip the barrier-trade (definition B) check")
    ap.add_argument("--skip-trials", action="store_true", help="do not append to the trials ledger")
    args = ap.parse_args()
    configure_logging()

    cfg = load_config(args.config)
    if getattr(args, "dataset_dir", None):
        cfg.paths.dataset_dir = args.dataset_dir
    df, meta = DatasetBundle.load(cfg.paths.dataset_dir)
    df = add_forward_returns(df, args.horizons)

    h = args.horizon
    target = f"fwd_ret_{h}"
    friction = cfg.costs.round_trip_cost_pct

    print("=" * 104)
    print(f"FEATURE SIGNAL CHECK  |  target = {h}-session forward return  |  friction = {friction:.2%} round trip")
    print("=" * 104)
    print(f"rows={len(df):,}  sessions={df['date'].nunique():,}  symbols={df['symbol'].nunique()}  "
          f"window={df['date'].min().date()}..{df['date'].max().date()}")
    print(f"non-overlapping {h}d periods available: ~{df['date'].nunique() // h}")
    print(f"usable target rows: {df[target].notna().sum():,} (tail {h} bars per symbol have no future)")

    signal_feats, control_feats = resolve_feature_lists(cfg, df)

    rows = []
    for feat in signal_feats + control_feats:
        if feat not in df.columns:
            continue
        ic = daily_ic(df, feat, target)
        st = ic_stats(ic, h)
        if not st:
            continue
        st["feature"] = feat
        st["block"] = "signal" if feat in signal_feats else "control"
        rows.append(st)

    tbl = pd.DataFrame(rows).set_index("feature")
    tbl = tbl.sort_values("mean_ic", key=lambda s: s.abs(), ascending=False)

    print("\n" + "-" * 104)
    print(f"INFORMATION COEFFICIENT vs {h}-day forward return   (Spearman, within-date, averaged over dates)")
    print("-" * 104)
    print(f"{'feature':34s} {'block':9s} {'mean_IC':>8s} {'IC_IR':>7s} {'%pos':>6s} {'t_ovl':>7s} {'t_nonovl':>9s} {'n':>5s}")
    for name, r in tbl.iterrows():
        print(f"{name:34s} {r['block']:9s} {r['mean_ic']:>8.4f} {r['ic_ir']:>7.2f} "
              f"{r['pct_pos']:>6.1%} {r['t_overlap_adj']:>7.2f} {r['t_non_overlap']:>9.2f} {int(r['n_dates']):>5d}")

    print("\n" + "-" * 104)
    print(f"QUANTILE LADDER (Q1=lowest feature, Q5=highest) -- mean {h}-day forward return")
    print("-" * 104)
    print(f"{'feature':34s} {'Q1':>8s} {'Q2':>8s} {'Q3':>8s} {'Q4':>8s} {'Q5':>8s} {'Q5-Q1':>8s} {'net_Q5Q1':>9s} {'mono':>5s}")
    ladders = {}
    for name in tbl.index:
        qr = quantile_returns(df, name, target)
        if qr is None or len(qr) < 5:
            continue
        ladders[name] = qr
        m = qr["mean"]
        spread = m.iloc[-1] - m.iloc[0]
        mono = np.sign(np.diff(m.to_numpy()))
        mono_ok = "yes" if np.all(mono == mono[0]) else "no"
        print(f"{name:34s} {m.iloc[0]:>8.2%} {m.iloc[1]:>8.2%} {m.iloc[2]:>8.2%} {m.iloc[3]:>8.2%} "
              f"{m.iloc[4]:>8.2%} {spread:>8.2%} {spread - friction:>9.2%} {mono_ok:>5s}")

    # -- is the signal block independent of the control block? -----------------
    print("\n" + "-" * 104)
    print("IS THE SIGNAL BLOCK INDEPENDENT OF THE CONTROL BLOCK?  max |corr| per signal feature")
    print("-" * 104)
    for feat in signal_feats:
        cors = df[[feat] + control_feats].corr(method="spearman")[feat].drop(feat)
        if cors.empty:
            continue
        worst = cors.abs().idxmax()
        print(f"  {feat:34s} max |rho| {cors[worst]:+.3f} vs {worst}")

    # -- stability across the sample ------------------------------------------
    print("\n" + "-" * 104)
    print("STABILITY: mean IC in the first vs second half of the sample")
    print("-" * 104)
    dates = np.sort(df["date"].unique())
    mid = dates[len(dates) // 2]
    print(f"  split at {pd.Timestamp(mid).date()}   (single-regime sample: read as stability, not regime effect)")
    print(f"  {'feature':34s} {'IC_1st_half':>12s} {'IC_2nd_half':>12s} {'same_sign':>10s}")
    for name in tbl.index[:12]:
        a = daily_ic(df[df["date"] < mid], name, target).mean()
        b = daily_ic(df[df["date"] >= mid], name, target).mean()
        same = "yes" if np.sign(a) == np.sign(b) else "NO"
        print(f"  {name:34s} {a:>12.4f} {b:>12.4f} {same:>10s}")

    # -- percentile-specific deep dive ----------------------------------------
    report_percentiles(df, h, target, friction, args.horizons)

    # -- detail for one feature ------------------------------------------------
    feat = args.feature or tbl.index[0]
    print("\n" + "=" * 104)
    print(f"DETAIL: {feat}")
    print("=" * 104)
    qr = ladders.get(feat)
    if qr is not None:
        print(qr.to_string())
    print("\nforward return by feature decile (Q1..Q10), pooled across dates:")
    dec = quantile_returns(df, feat, target, q=10)
    if dec is not None:
        print("  " + "  ".join(f"Q{i+1}:{v:+.2%}" for i, v in enumerate(dec["mean"])))

    # -- the model's trade, definition B ---------------------------------------
    if not args.skip_def_b:
        barrier_trade_check(df, signal_feats + control_feats,
                            labels_dir=cfg.paths.dataset_dir)

    # -- trials ledger ----------------------------------------------------------
    if not args.skip_trials:
        n_feats = int(tbl.shape[0])
        n_trials = n_feats * len(args.horizons)
        entry = record_trial(
            script="signal_check.py",
            dataset=str(cfg.paths.dataset_dir),
            engine="none",
            question=f"feature screen: {n_feats} features x {len(args.horizons)} horizons "
                    f"(IC, ladders, top-decile precision under definitions C and B)",
            config={"horizons": list(args.horizons), "friction": friction,
                    "n_features": n_feats},
            metrics={},
            n_folds=None,
            trials_added=n_trials,
        )
        c = count_trials()
        print(f"\nledger: +{entry['trials_added']} trials -> {c['total']} total (data/trials.jsonl)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
