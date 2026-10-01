"""Triple-barrier labelling tests.

The label is the target, so it is *meant* to look forward -- but it must not see
past its own vertical barrier. These tests pin the barrier precedence rules, the
gap-fill convention, the mandatory friction, and the causality property.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingml.config import CostsConfig, LabelConfig
from swingml.labeling import (
    LABEL_EXPIRY,
    LABEL_PROFIT,
    LABEL_STOP,
    triple_barrier_labels,
)

FRICTION = 0.0025


def _frame(symbol: str, closes, highs=None, lows=None, opens=None, atr=1.0, start="2024-01-01"):
    """Build a minimal single-symbol feature frame with a flat ATR."""
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    opens = np.asarray(opens, dtype=float) if opens is not None else closes.copy()
    highs = np.asarray(highs, dtype=float) if highs is not None else np.maximum(opens, closes)
    lows = np.asarray(lows, dtype=float) if lows is not None else np.minimum(opens, closes)
    atrs = np.full(n, float(atr)) if np.isscalar(atr) else np.asarray(atr, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=n),
            "symbol": symbol,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "atr_20": atrs,
        }
    )


def _run(df, horizon=10, pt=2.0, sl=1.0, entry="next_open", same_bar="pessimistic"):
    cfg = LabelConfig(
        horizon_days=horizon, pt_atr_mult=pt, sl_atr_mult=sl,
        entry_price=entry, same_bar_resolution=same_bar,
    )
    return triple_barrier_labels(df, cfg, CostsConfig(round_trip_cost_pct=FRICTION))


def test_upper_barrier_hit_gives_profit_label():
    # Entry 100 (atr 1, +2 => 102). Price ramps straight up through 102.
    df = _frame("UP", closes=[100, 100, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112])
    res = _run(df)
    row = res.labels.iloc[0]
    assert row["label"] == LABEL_PROFIT
    assert row["barrier_hit"] == "pt"
    assert row["exit_price"] >= row["entry_price"] + 2.0 - 1e-9


def test_lower_barrier_hit_gives_stop_label():
    df = _frame("DN", closes=[100, 100, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88])
    res = _run(df)
    row = res.labels.iloc[0]
    assert row["label"] == LABEL_STOP
    assert row["barrier_hit"] == "sl"


def test_no_barrier_hit_gives_expiry_label():
    # Dead flat: neither +2 nor -1 is ever touched inside the window.
    df = _frame("FLAT", closes=[100] * 15)
    res = _run(df)
    row = res.labels.iloc[0]
    assert row["label"] == LABEL_EXPIRY
    assert row["barrier_hit"] == "vertical"
    assert row["bars_held"] == 10  # exactly the horizon


# ---------------------------------------------------------------------------
# label.mode = fixed_hold (definition C as a training target)
# ---------------------------------------------------------------------------

def _run_mode(df, mode="fixed_hold", horizon=10, entry="next_open"):
    cfg = LabelConfig(mode=mode, horizon_days=horizon, pt_atr_mult=2.0, sl_atr_mult=1.0,
                      entry_price=entry)
    return triple_barrier_labels(df, cfg, CostsConfig(round_trip_cost_pct=FRICTION))


def test_fixed_hold_ignores_both_horizontal_barriers():
    """The whole point of the mode: a -1xATR dip that recovers must NOT stop out."""
    closes = [100, 100, 98.5, 99, 100.5, 101, 102, 103, 104, 105, 106, 107]
    barrier = _run(_frame("V", closes=closes))
    fixed = _run_mode(_frame("V", closes=closes))
    assert barrier.labels.iloc[0]["label"] == LABEL_STOP          # barrier stops out
    assert fixed.labels.iloc[0]["label"] == LABEL_PROFIT          # fixed hold rides it
    assert fixed.labels.iloc[0]["barrier_hit"] == "vertical"
    assert fixed.labels.iloc[0]["bars_held"] == 10
    assert LABEL_STOP not in set(fixed.labels["label"])


def test_fixed_hold_label_is_the_net_pnl_sign():
    """A hold that is up gross but not enough to pay the toll is a loss label."""
    flat = _run_mode(_frame("F", closes=[100] * 15))
    assert flat.labels.iloc[0]["label"] == LABEL_EXPIRY
    assert flat.labels.iloc[0]["ret_net"] == pytest.approx(-FRICTION)

    up = _run_mode(_frame("U", closes=[100, 100] + [110] * 13))
    assert up.labels.iloc[0]["label"] == LABEL_PROFIT
    assert up.labels.iloc[0]["ret_net"] > 0


def test_both_modes_label_the_same_row_set():
    """Comparability: the A/B must not be confounded by a different sample."""
    df = _frame("S", closes=[100, 101, 99, 103, 98, 104, 97, 105, 96, 106, 95, 107])
    a = _run(df)
    b = _run_mode(df)
    assert len(a.labels) == len(b.labels)
    assert list(a.labels["date"]) == list(b.labels["date"])


def test_fixed_hold_diagnostics_say_which_mode_and_do_not_print_a_structural_ratio():
    from swingml.labeling import format_diagnostics

    diag = _run_mode(_frame("F", closes=[100] * 15)).diagnostics
    assert diag["mode"] == "fixed_hold"
    assert np.isnan(diag["expiry_frac_net_negative"])   # would be a structural 100%
    text = format_diagnostics(diag)
    assert "fixed_hold" in text
    assert "none -- exit at the vertical barrier only" in text


def test_unknown_mode_is_rejected():
    cfg = LabelConfig(mode="nonsense")
    with pytest.raises(ValueError):
        triple_barrier_labels(_frame("X", closes=[100] * 15), cfg,
                              CostsConfig(round_trip_cost_pct=FRICTION))


def test_pessimistic_resolves_same_bar_as_stop_loss():
    """A single bar spanning both barriers must book the stop loss by default."""
    # Bar 2 has low 96 (below the 99 stop) and high 104 (above the 102 target).
    df = _frame(
        "BOTH",
        closes=[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        highs=[100, 100, 104, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        lows=[100, 100, 96, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    )
    res = _run(df, same_bar="pessimistic")
    assert res.labels.iloc[0]["label"] == LABEL_STOP

    res_opt = _run(df, same_bar="optimistic")
    assert res_opt.labels.iloc[0]["label"] == LABEL_PROFIT


def test_friction_is_applied_to_every_trade():
    df = _frame("UP", closes=[100, 100, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112])
    res = _run(df)
    row = res.labels.iloc[0]
    assert row["ret_net"] == pytest.approx(row["ret_gross"] - FRICTION)
    assert res.diagnostics["round_trip_cost_pct"] == pytest.approx(FRICTION)


def test_gap_through_target_fills_at_open_not_the_barrier():
    """Gapping above a take profit fills at the open, which is better than target."""
    opens = [100, 100, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110]
    closes = [100, 100, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110]
    highs = [100, 100, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111]
    lows = [100, 100, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109]
    df = _frame("GAPUP", closes=closes, highs=highs, lows=lows, opens=opens)

    row = _run(df).labels.iloc[0]
    assert row["barrier_hit"] == "pt"
    # Entry is bar-1 open = 100; target = 102; the gap fills at 110.
    assert row["entry_price"] == pytest.approx(100.0)
    assert row["exit_price"] == pytest.approx(110.0)
    assert row["ret_gross"] > 0.02


def test_gap_through_stop_fills_at_open_and_is_worse_than_the_stop():
    opens = [100, 100, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]
    closes = [100, 100, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]
    highs = [100, 100, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91]
    lows = [100, 100, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89]
    df = _frame("GAPDN", closes=closes, highs=highs, lows=lows, opens=opens)

    row = _run(df).labels.iloc[0]
    assert row["barrier_hit"] == "sl"
    # Stop sits at 99; the gap down fills at 90, which is worse.
    assert row["exit_price"] == pytest.approx(90.0)
    assert row["ret_gross"] < -0.09


def test_label_does_not_depend_on_data_after_its_vertical_barrier():
    """Truncating the series exactly at t1 must not change any earlier label."""
    rng = np.random.default_rng(11)
    n = 60
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    df = _frame("CAUSAL", closes=closes, highs=closes * 1.02, lows=closes * 0.98, atr=1.5)

    full = _run(df, horizon=10).labels

    # Cut the series mid-way, then compare only the labels whose ENTIRE window
    # (t0..t1) lies inside what remains -- those must be untouched.
    cut = full.iloc[len(full) // 2]["t1"]
    df_trunc = df[df["date"] <= cut].copy()
    trunc = _run(df_trunc, horizon=10).labels

    a = full[full["t1"] <= cut].reset_index(drop=True)
    b = trunc[trunc["t1"] <= cut].reset_index(drop=True)
    assert len(a) > 10, "test needs a meaningful number of fully-contained labels"
    assert len(a) == len(b)
    for col in ("date", "label", "barrier_hit", "ret_gross", "entry_price", "exit_price", "bars_held"):
        pd.testing.assert_series_equal(a[col], b[col], check_names=False)


def test_last_horizon_bars_are_dropped():
    """Signals without a full forward window must not be labelled."""
    df = _frame("TAIL", closes=[100] * 40)
    res = _run(df, horizon=10)
    # 40 bars -> signals 0..29 are resolvable (need i+10 <= 39).
    assert len(res.labels) == 30
    assert res.diagnostics["n_dropped_tail"] == 10


def test_notes_are_recorded_and_effective_n_is_smaller_than_rows():
    df = _frame("OVER", closes=[100] * 40)
    res = _run(df, horizon=10)
    diag = res.diagnostics
    assert diag["n_labelled"] == len(res.labels)
    # Overlapping 10-day labels share their span, so effective n << row count.
    assert 0 < diag["effective_n"] < diag["n_labelled"]
    assert diag["mean_uniqueness"] < 1.0
    assert set(diag["class_counts"]) <= {"1", "0", "-1"}


def test_expiry_rows_are_reported_as_net_negative_share():
    """A '0' label means no barrier reached, not 'flat in P&L terms'."""
    df = _frame("FLAT", closes=[100] * 20)
    diag = _run(df, horizon=10).diagnostics
    # Zero gross move minus friction is a net loss on every expiry row.
    assert diag["expiry_frac_net_negative"] == pytest.approx(1.0)


def test_asymmetric_barriers_produce_more_stops_than_profits_on_a_random_walk():
    rng = np.random.default_rng(5)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 800)))
    df = _frame("RW", closes=closes, highs=closes * 1.015, lows=closes * 0.985, atr=2.0)
    diag = _run(df, horizon=10).diagnostics
    # -1xATR is twice as close as +2xATR, so the stop must dominate.
    assert diag["class_pct"].get("-1", 0.0) > diag["class_pct"].get("1", 0.0)


def test_missing_context_columns_raise_a_clear_error():
    df = _frame("X", closes=[100] * 20).drop(columns=["high"])
    with pytest.raises(ValueError, match="rebuild the dataset"):
        _run(df)


def test_bad_atr_rows_are_dropped_not_labelled():
    df = _frame("NAN", closes=[100] * 30, atr=1.0)
    df.loc[0:4, "atr_20"] = np.nan
    diag = _run(df).diagnostics
    assert diag["n_dropped_bad_atr"] == 5
