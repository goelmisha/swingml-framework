"""Step 2 -- Triple-barrier labelling (López de Prado, *Advances in Financial ML*, ch. 3).

Why not a naive "up/down tomorrow" label
----------------------------------------
A fixed-horizon binary label ignores the path. A stock that falls 8% before
recovering 2% by expiry is labelled the same as one that drifts quietly up 2%,
yet the first would have stopped out of any real trade. The triple barrier fixes
this by asking *which barrier is reached first*:

    +upper barrier (take profit, +2 x ATR)  -> label  1
    -lower barrier (stop loss,     -1 x ATR) -> label -1
     vertical barrier (time stop, horizon)   -> label  0

Barriers are ATR-scaled, so they adapt to each name's volatility instead of
imposing one arbitrary percentage on a calm large-cap and a jumpy mid-cap alike.

Two exit rules (``label.mode``)
-------------------------------
The exit rule *is* the learning target, so it is a config choice, not a detail:

    barrier     exit at whichever horizontal barrier is reached first (above).
    fixed_hold  exit at the vertical barrier only; label 1 if that hold was
                profitable **net of friction**, else 0 (definition C as a target).

Same features, same folds, same names -- only the exit differs.

Rows are dropped on non-finite ``atr_20`` in **both** modes, so the two rule sets
label an identical row set and the A/B comparison is not confounded by sample.
In ``fixed_hold`` the label is a net-P&L sign, so ``LABEL_STOP`` never occurs and
``LABEL_EXPIRY`` (0) means "held to expiry and lost money net", not "no barrier".

Conventions (all configurable, both choices are deliberate)
-----------------------------------------------------------
**Entry is the NEXT session's open**, not the close of the signal bar. Entering
at the close you just used to compute the signal assumes you could trade at a
price you had only just observed; next-open entry is the honest version and it
costs nothing to implement.

**Same-bar ambiguity resolves pessimistically.** If one session's range spans
both barriers we cannot know from daily bars which came first, so the stop loss
wins. The optimistic reading would flatter every result; with a 2x ATR target
and 1x ATR stop, this case only arises on a >3x ATR day, but it must be resolved
explicitly rather than silently.

**Gaps fill beyond the barrier.** If a session opens through a barrier, the fill
is the open, not the barrier level -- better than target for a take profit,
worse than stop for a stop loss.

Causality
---------
The label is *supposed* to look forward; it is the target. What must not happen
is a label at ``t`` depending on anything after its own vertical barrier. The
labelling window is closed at ``t + horizon``, and ``tests/test_labeling.py``
proves it by truncating the series exactly at ``t1`` and confirming the label is
unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from swingml.config import CostsConfig, LabelConfig
from swingml.events import cusum_events

logger = logging.getLogger(__name__)

#: Columns the labeller requires from the feature matrix.
REQUIRED_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "atr_20")

#: Barrier outcomes.
BARRIER_PROFIT = "pt"
BARRIER_STOP = "sl"
BARRIER_VERTICAL = "vertical"

#: Label codes.
LABEL_PROFIT = 1
LABEL_EXPIRY = 0
LABEL_STOP = -1


@dataclass
class LabelResult:
    """Labels plus the diagnostics needed to sanity-check them."""

    labels: pd.DataFrame
    diagnostics: dict


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"labelling needs columns {missing}; rebuild the dataset so that "
            "open/high/low are carried through as context"
        )
    if not np.isfinite(df["atr_20"].dropna()).all():
        raise ValueError("atr_20 contains non-finite values")


def triple_barrier_labels(
    df: pd.DataFrame,
    label_cfg: LabelConfig,
    costs_cfg: CostsConfig,
    horizon_days: int | None = None,
    pt_mult: float | None = None,
    sl_mult: float | None = None,
) -> LabelResult:
    """Apply the triple-barrier method to a feature matrix.

    Parameters
    ----------
    df
        Feature matrix keyed on ``(date, symbol)`` carrying OHLC, ``atr_20`` and
        the label context columns.
    label_cfg, costs_cfg
        Barrier geometry and the mandatory friction.
    horizon_days, pt_mult, sl_mult
        Optional overrides for the vertical horizon and the barrier multiples,
        used for sensitivity work without editing the config.

    Returns
    -------
    :class:`LabelResult`
        ``labels`` has one row per labelled signal; ``diagnostics`` summarises
        class balance, holding periods and effective sample size.
    """
    _validate_input(df)

    horizon = int(horizon_days if horizon_days is not None else label_cfg.horizon_days)
    pt_k = float(pt_mult if pt_mult is not None else label_cfg.pt_atr_mult)
    sl_k = float(sl_mult if sl_mult is not None else label_cfg.sl_atr_mult)
    entry_mode = str(getattr(label_cfg, "entry_price", "next_open")).lower()
    same_bar = str(getattr(label_cfg, "same_bar_resolution", "pessimistic")).lower()
    mode = str(getattr(label_cfg, "mode", "barrier")).lower()
    sampling = str(getattr(label_cfg, "sampling", "every_bar")).lower()
    cusum_k = float(getattr(label_cfg, "cusum_h_mult", 1.5))
    cusum_span = int(getattr(label_cfg, "cusum_vol_span", 20))
    friction = float(costs_cfg.round_trip_cost_pct)

    if horizon < 1:
        raise ValueError("horizon_days must be >= 1")
    if entry_mode not in {"next_open", "close"}:
        raise ValueError(f"entry_price must be 'next_open' or 'close', got {entry_mode!r}")
    if same_bar not in {"pessimistic", "optimistic"}:
        raise ValueError(f"same_bar_resolution must be 'pessimistic' or 'optimistic', got {same_bar!r}")
    if mode not in {"barrier", "fixed_hold"}:
        raise ValueError(f"mode must be 'barrier' or 'fixed_hold', got {mode!r}")
    if sampling not in {"every_bar", "cusum"}:
        raise ValueError(f"sampling must be 'every_bar' or 'cusum', got {sampling!r}")

    need_next_open = entry_mode == "next_open"
    records: list[dict] = []
    dropped_tail = dropped_atr = dropped_bad_entry = dropped_not_event = 0

    for symbol, g in df.groupby("symbol", observed=True, sort=False):
        g = g.sort_values("date")
        n = len(g)
        # A signal at position i needs bars up to i + horizon to resolve.
        last_signal = n - 1 - horizon
        if last_signal < 0:
            dropped_tail += n
            continue
        dropped_tail += n - 1 - last_signal

        dates = g["date"].to_numpy()
        # Event sampling is decided per symbol: two names moving together are
        # two independent events, not one. The mask is causal (see events.py).
        if sampling == "cusum":
            ev = cusum_events(
                pd.Series(g["close"].to_numpy(dtype=float), index=pd.Index(dates)),
                h_mult=cusum_k, vol_span=cusum_span,
            ).to_numpy()
        else:
            ev = None
        o = g["open"].to_numpy(dtype=float)
        h = g["high"].to_numpy(dtype=float)
        low = g["low"].to_numpy(dtype=float)
        c = g["close"].to_numpy(dtype=float)
        atr = g["atr_20"].to_numpy(dtype=float)

        for i in range(last_signal + 1):
            if ev is not None and not ev[i]:
                dropped_not_event += 1
                continue
            a = atr[i]
            if not np.isfinite(a) or a <= 0:
                dropped_atr += 1
                continue

            entry_idx = i + 1 if need_next_open else i
            entry_price = o[entry_idx] if need_next_open else c[entry_idx]
            if not np.isfinite(entry_price) or entry_price <= 0:
                dropped_bad_entry += 1
                continue

            pt_price = entry_price + pt_k * a
            sl_price = entry_price - sl_k * a
            vertical_idx = i + horizon

            barrier_hit = None
            exit_idx = vertical_idx
            exit_price = c[vertical_idx]

            for j in (() if mode == "fixed_hold" else range(entry_idx, vertical_idx + 1)):
                up = h[j] >= pt_price
                down = low[j] <= sl_price

                if up and down:
                    # Daily bars cannot resolve intrabar order. Be conservative.
                    barrier_hit = BARRIER_PROFIT if same_bar == "optimistic" else BARRIER_STOP
                elif down:
                    barrier_hit = BARRIER_STOP
                elif up:
                    barrier_hit = BARRIER_PROFIT
                else:
                    continue

                exit_idx = j
                if barrier_hit == BARRIER_PROFIT:
                    # Gap up through the target fills at the open (favourable).
                    exit_price = max(pt_price, o[j])
                else:
                    # Gap down through the stop fills at the open (unfavourable).
                    exit_price = min(sl_price, o[j])
                break

            if barrier_hit is None:
                barrier_hit = BARRIER_VERTICAL

            ret_gross = exit_price / entry_price - 1.0
            ret_net = ret_gross - friction
            if mode == "fixed_hold":
                # Definition C as a target: profitable after friction, or not.
                label = LABEL_PROFIT if ret_net > 0 else LABEL_EXPIRY
            else:
                label = (
                    LABEL_PROFIT if barrier_hit == BARRIER_PROFIT
                    else LABEL_STOP if barrier_hit == BARRIER_STOP
                    else LABEL_EXPIRY
                )

            records.append(
                {
                    "date": dates[i],              # signal bar
                    "symbol": symbol,
                    "entry_date": dates[entry_idx],
                    "entry_price": entry_price,
                    "t1": dates[vertical_idx],     # vertical barrier date
                    "exit_date": dates[exit_idx],
                    "exit_price": float(exit_price),
                    "barrier_hit": barrier_hit,
                    "label": label,
                    "ret_gross": float(ret_gross),
                    # The mandatory 0.25% round trip, applied to every trade.
                    "ret_net": float(ret_net),
                    "bars_held": int(exit_idx - entry_idx + (1 if need_next_open else 0)),
                    "t0_pos": int(i),
                    "t1_pos": int(vertical_idx),
                }
            )

    if not records:
        raise RuntimeError("no labels produced -- is the history shorter than the horizon?")

    labels = pd.DataFrame.from_records(records)
    labels["date"] = pd.to_datetime(labels["date"])

    weights = sample_uniqueness_weights(labels)
    labels["uniqueness"] = weights.to_numpy()

    diagnostics = _diagnose(
        labels, total_rows=len(df), dropped_tail=dropped_tail, dropped_atr=dropped_atr,
        dropped_bad_entry=dropped_bad_entry, dropped_not_event=dropped_not_event,
        horizon=horizon, pt_k=pt_k, sl_k=sl_k,
        entry_mode=entry_mode, same_bar=same_bar, friction=friction, mode=mode,
        sampling=sampling, cusum_k=cusum_k, cusum_span=cusum_span,
    )
    return LabelResult(labels=labels, diagnostics=diagnostics)


def sample_uniqueness_weights(labels: pd.DataFrame) -> pd.Series:
    """Average uniqueness per label, after López de Prado.

    A 10-day label at ``t`` and one at ``t+1`` share 9 of their 10 days, so the
    training set contains far fewer *independent* observations than rows. Weight
    each label by the inverse of how many other labels were open alongside it:

        w_k = mean over its span of 1 / concurrency

    Concurrency is counted **within each symbol**. Overlapping trades in
    different names do not share a return path, so summing concurrency across
    the panel would understate every label's uniqueness.

    ``sum(w)`` is the effective number of independent samples -- report it rather
    than the raw row count.
    """
    lab = labels.reset_index(drop=True)
    out = np.ones(len(lab), dtype=float)

    for _, g in lab.groupby("symbol", observed=True, sort=False):
        starts = g["t0_pos"].to_numpy(dtype=int)
        ends = g["t1_pos"].to_numpy(dtype=int)
        n = int(ends.max()) + 1

        diff = np.zeros(n + 1, dtype=float)
        np.add.at(diff, starts, 1.0)
        np.add.at(diff, ends + 1, -1.0)
        concurrency = np.cumsum(diff)[:n]

        inv = 1.0 / np.maximum(concurrency, 1.0)
        cum_inv = np.concatenate([[0.0], np.cumsum(inv)])
        uniq = (cum_inv[ends + 1] - cum_inv[starts]) / (ends - starts + 1)
        out[g.index.to_numpy()] = uniq

    return pd.Series(out, index=labels.index, name="uniqueness")


def _diagnose(
    labels: pd.DataFrame, total_rows: int, dropped_tail: int, dropped_atr: int,
    dropped_bad_entry: int, dropped_not_event: int, horizon: int, pt_k: float, sl_k: float,
    entry_mode: str, same_bar: str, friction: float, mode: str = "barrier",
    sampling: str = "every_bar", cusum_k: float = 1.5, cusum_span: int = 20,
) -> dict:
    """Build the measurement report that Step 3 will be judged against."""
    counts = labels["label"].value_counts().to_dict()
    n = len(labels)

    def _stat(series: pd.Series) -> dict:
        s = series.dropna()
        if s.empty:
            return {}
        return {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
        }

    by_label = {}
    for lab in (LABEL_PROFIT, LABEL_EXPIRY, LABEL_STOP):
        sub = labels[labels["label"] == lab]
        if sub.empty:
            continue
        by_label[str(lab)] = {
            "n": int(len(sub)),
            "pct": float(len(sub) / n),
            "bars_held_median": float(sub["bars_held"].median()),
            "ret_net_mean": float(sub["ret_net"].mean()),
            "ret_net_median": float(sub["ret_net"].median()),
        }

    expiry = labels[labels["label"] == LABEL_EXPIRY]
    fixed_hold = mode == "fixed_hold"
    return {
        "mode": mode,
        "sampling": sampling,
        "cusum_h_mult": cusum_k,
        "cusum_vol_span": cusum_span,
        "horizon_days": horizon,
        "pt_atr_mult": pt_k,
        "sl_atr_mult": sl_k,
        "entry_price": entry_mode,
        "same_bar_resolution": same_bar,
        "round_trip_cost_pct": friction,
        "skew_note": (
            (
                f"fixed_hold: the trade runs to the vertical barrier ({horizon} sessions) "
                "and the label is its net-P&L sign, so there is no stop-loss class and "
                f"label 0 means 'held and lost money net' -- base rate = {float((labels['ret_net'] > 0).mean()):.1%} profitable"
            )
            if fixed_hold else
            (
                f"upper barrier {pt_k}xATR vs lower barrier {sl_k}xATR: an efficient "
                "random walk crosses the nearer barrier more often, so no-profit"
                + (" and stop-loss" if sl_k < pt_k else "")
                + " labels can legitimately outnumber profit labels"
            )
        ),
        "n_rows_in": int(total_rows),
        "n_labelled": int(n),
        "n_dropped_tail": int(dropped_tail),
        "n_dropped_bad_atr": int(dropped_atr),
        "n_dropped_bad_entry": int(dropped_bad_entry),
        "n_dropped_not_event": int(dropped_not_event),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "class_pct": {str(k): float(v / n) for k, v in counts.items()},
        "by_label": by_label,
        "barrier_counts": {k: int(v) for k, v in labels["barrier_hit"].value_counts().to_dict().items()},
        "bars_held": _stat(labels["bars_held"]),
        "ret_gross": _stat(labels["ret_gross"]),
        "ret_net": _stat(labels["ret_net"]),
        # Label 0 means "no barrier reached", NOT "flat in P&L terms": after
        # friction a small drift can still be a net loss, which matters for
        # how Step 3 interprets a 3-class prediction. Under fixed_hold that
        # identity is exact instead (label 0 IS a net loss), so reporting it
        # would be circular -- say so rather than print a structural 100%.
        "expiry_frac_net_negative": (
            float("nan") if fixed_hold
            else float((expiry["ret_net"] <= 0).mean()) if len(expiry) else float("nan")
        ),
        "frac_net_positive": float((labels["ret_net"] > 0).mean()),
        "mean_uniqueness": float(labels["uniqueness"].mean()),
        "effective_n": float(labels["uniqueness"].sum()),
        "sessions": int(labels["date"].nunique()),
        "symbols": int(labels["symbol"].nunique()),
    }


def format_diagnostics(diag: dict) -> str:
    """Human-readable diagnostics block for the CLI."""
    fixed_hold = diag.get("mode", "barrier") == "fixed_hold"
    lines = []
    lines.append("=" * 78)
    lines.append("LABEL DIAGNOSTICS" + ("  (fixed_hold)" if fixed_hold else "  (triple barrier)"))
    lines.append("=" * 78)
    barrier_desc = (
        "none -- exit at the vertical barrier only"
        if fixed_hold else
        f"barriers: +{diag['pt_atr_mult']}xATR / -{diag['sl_atr_mult']}xATR (ATR-20)"
    )
    lines.append(f"horizon            : {diag['horizon_days']} sessions   {barrier_desc}")
    lines.append(
        f"entry              : {diag['entry_price']}    "
        f"same-bar: {diag['same_bar_resolution']}    friction: {diag['round_trip_cost_pct']:.2%}"
    )
    if diag.get("sampling") == "cusum":
        lines.append(
            f"sampling           : cusum events (h = {diag['cusum_h_mult']} x "
            f"EWM std, span {diag['cusum_vol_span']})"
        )
    else:
        lines.append("sampling           : every bar")
    lines.append(
        f"rows in / labelled : {diag['n_rows_in']:,} / {diag['n_labelled']:,}"
        f"   (dropped: {diag['n_dropped_tail']:,} insufficient future, "
        f"{diag['n_dropped_bad_atr']:,} bad ATR, {diag['n_dropped_bad_entry']:,} bad entry"
        + (f", {diag['n_dropped_not_event']:,} no CUSUM event" if diag.get('n_dropped_not_event') else "")
        + ")"
    )
    lines.append(f"sessions / symbols : {diag['sessions']:,} / {diag['symbols']:,}")

    lines.append("\nclass balance:")
    names = ({1: "net win (+)", 0: "net loss ( )"} if fixed_hold
             else {1: "profit (+)", 0: "expiry ( )", -1: "stop  (-)"})
    for lab in (1, 0, -1):
        b = diag["by_label"].get(str(lab))
        if not b:
            continue
        lines.append(
            f"  {names[lab]:10s} {b['n']:>7,}  {b['pct']:>6.1%}   "
            f"median hold {b['bars_held_median']:.0f}d   net mean {b['ret_net_mean']:+.2%}"
        )

    lines.append("\nexit:" if fixed_hold else "\nbarrier reached first:")
    for k, v in sorted(diag["barrier_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:10s} {v:>7,}  {v / diag['n_labelled']:>6.1%}")

    bh, rn = diag["bars_held"], diag["ret_net"]
    lines.append(
        f"\nbars held          : median {bh['median']:.0f}  p25 {bh['p25']:.0f}  p75 {bh['p75']:.0f}"
    )
    lines.append(
        f"net return         : mean {rn['mean']:+.2%}  median {rn['median']:+.2%}  "
        f"(gross mean {diag['ret_gross']['mean']:+.2%})"
    )
    tail = (
        f"   expiry rows that are net-negative: {diag['expiry_frac_net_negative']:.1%}"
        if np.isfinite(diag["expiry_frac_net_negative"]) else
        "   (label 0 IS a net loss in this mode; the ratio would be structural)"
    )
    lines.append(f"frac net > 0       : {diag['frac_net_positive']:.1%}" + tail)
    lines.append(
        f"\nsample overlap     : mean uniqueness {diag['mean_uniqueness']:.3f} over "
        f"{diag['n_labelled']:,} rows -> EFFECTIVE n = {diag['effective_n']:,.0f}"
    )
    lines.append(f"  (consecutive {diag['horizon_days']}-day labels share most of their span;")
    lines.append("   effective n, not row count, is what limits statistical power)")
    lines.append("\nnote: " + diag["skew_note"])
    return "\n".join(lines)
