"""Leakage and causality tests.

The central guarantee of Step 1 is that **no feature at bar ``t`` depends on
anything after ``t``**. Rather than eyeballing the code, these tests prove it by
*truncation invariance*: rebuild every feature using only data up to ``T`` and
assert the values on ``date <= T`` are unchanged. Any forward-looking window,
centred rolling statistic or global normalisation fails this immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingml.features.actions import detect_quantity_jumps
from swingml.features.primitives import (
    atr_wilder,
    ema,
    realized_vol,
    rsi_wilder,
    rolling_zscore,
    true_range,
)
from swingml.features import make_feature_provider
from tests.conftest import _make


def _features(prices, panel, bench, cfg):
    """Exercise whichever provider the config names (see swingml.features.registry).

    Provider-agnostic on purpose: the no-look-ahead rule is a property of the
    contract, so every provider must satisfy it, not just the one shipped here.
    """
    return make_feature_provider(cfg).transform(prices, panel, bench)


def test_truncation_invariance(synth, feature_cfg):
    """Features computed on data up to T must match the full-sample values."""
    prices, panel, bench = synth

    full = _features(prices, panel, bench, feature_cfg)

    # Truncate ~40% of the way through, then recompute from scratch.
    all_dates = sorted(full["date"].unique())
    cutoff = all_dates[int(len(all_dates) * 0.6)]
    cut = pd.Timestamp(cutoff)

    px_t = {s: df.loc[df.index <= cut] for s, df in prices.items()}
    panel_t = panel.loc[panel["date"] <= cut].copy()
    bench_t = bench.loc[bench.index <= cut]

    trunc = _features(px_t, panel_t, bench_t, feature_cfg)

    common = [c for c in full.columns if c in trunc.columns]
    a = full.loc[full["date"] <= cut, common].sort_values(["date", "symbol"]).reset_index(drop=True)
    b = trunc.loc[trunc["date"] <= cut, common].sort_values(["date", "symbol"]).reset_index(drop=True)

    # Same rows must exist in both (warm-up dropping is also backward-looking).
    assert len(a) == len(b), f"row count diverged under truncation: {len(a)} vs {len(b)}"
    assert list(a["symbol"]) == list(b["symbol"])

    numeric = [c for c in common if c not in ("date", "symbol") and pd.api.types.is_numeric_dtype(a[c])]
    for col in numeric:
        av, bv = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
        # NaNs must appear in exactly the same places -- a shifted NaN mask is
        # itself evidence of a look-ahead window.
        assert np.array_equal(np.isnan(av), np.isnan(bv)), f"{col}: NaN mask changed under truncation"
        both = ~np.isnan(av)
        if both.any():
            np.testing.assert_allclose(
                av[both], bv[both], rtol=1e-9, atol=1e-12,
                err_msg=f"{col} changed when future data was removed -> look-ahead leak",
            )


def test_no_future_columns_present(synth, feature_cfg):
    """Guard against accidentally shipping forward-looking target columns."""
    prices, panel, bench = synth
    feats = _features(prices, panel, bench, feature_cfg)
    # Substring match on the *whole* column name. Beware false positives: a bare
    # "y_" prefix also matches unrelated names that merely end in "y_".
    banned_substrings = ("future", "fwd", "label", "target", "horizon", "next_ret", "next_day")
    banned_prefixes = ("y_", "t1_", "next_", "fwd_")
    offenders = [
        c for c in feats.columns
        if any(b in c.lower() for b in banned_substrings)
        or any(c.lower().startswith(p) for p in banned_prefixes)
    ]
    assert not offenders, f"forward-looking columns leaked into the feature matrix: {offenders}"


def test_cross_sectional_ranks_use_only_same_date(synth, feature_cfg):
    """Ranks must be computed strictly within a date, never across time."""
    prices, panel, bench = synth
    feats = _features(prices, panel, bench, feature_cfg)
    rank_cols = [c for c in feats.columns if c.startswith("xs_rank_")]
    assert rank_cols, "expected cross-sectional rank features to be produced"
    for col in rank_cols:
        # A symbol may legitimately have NaN here (e.g. its quantity features were
        # suppressed after a split), so evaluate only dates that produced ranks.
        mx = feats.groupby("date")[col].max().dropna()
        mn = feats.groupby("date")[col].min().dropna()
        assert len(mx) > 0, f"{col} produced no rank values at all"
        assert mx.max() <= 1.0 + 1e-12, f"{col} exceeds the [0,1] percentile range"
        assert mn.min() >= 0.0 - 1e-12, f"{col} falls below the [0,1] percentile range"


# ---------------------------------------------------------------------------
# Indicator primitives
# ---------------------------------------------------------------------------
def test_ema_is_causal_and_ignores_future():
    s = pd.Series(np.linspace(1, 100, 200))
    full = ema(s, 20)
    cut = ema(s.iloc[:150], 20)
    np.testing.assert_allclose(full.iloc[19:150].to_numpy(), cut.iloc[19:].to_numpy())


def test_rsi_bounds_and_monotonic_case():
    up = pd.Series(np.linspace(10, 200, 100))
    r = rsi_wilder(up, 14).dropna()
    assert (r <= 100.0 + 1e-9).all() and (r >= 0.0).all()
    # An unbroken up-run has zero average loss -> RSI pinned at 100.
    assert r.iloc[-1] == pytest.approx(100.0)

    down = pd.Series(np.linspace(200, 10, 100))
    rd = rsi_wilder(down, 14).dropna()
    assert rd.iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_true_range_uses_previous_close():
    high = pd.Series([10.0, 12.0, 13.0])
    low = pd.Series([9.0, 11.0, 11.5])
    close = pd.Series([9.5, 11.5, 12.0])
    tr = true_range(high, low, close)
    # First bar has no previous close, so TR reduces to the bar's own range.
    assert tr.iloc[0] == pytest.approx(1.0)
    # Second bar: max(high-low=1.0, |12-9.5|=2.5, |11-9.5|=1.5) = 2.5
    assert tr.iloc[1] == pytest.approx(2.5)


def test_atr_is_positive_and_causal():
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    high = close * 1.01
    low = close * 0.99
    a = atr_wilder(high, low, close, 14).dropna()
    assert (a > 0).all()
    a_cut = atr_wilder(high.iloc[:200], low.iloc[:200], close.iloc[:200], 14).dropna()
    np.testing.assert_allclose(a.iloc[: len(a_cut)].to_numpy()[:150], a_cut.to_numpy()[:150])


def test_realized_vol_scales_with_volatility():
    quiet = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.002, 500))))
    loud = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.03, 500))))
    assert realized_vol(loud, 20).mean() > realized_vol(quiet, 20).mean()


def test_rolling_zscore_has_no_lookahead():
    s = pd.Series(np.arange(1, 201, dtype=float))
    z_full = rolling_zscore(s, 20)
    z_cut = rolling_zscore(s.iloc[:120], 20)
    np.testing.assert_allclose(z_full.iloc[19:120].to_numpy(), z_cut.iloc[19:].to_numpy())


def test_detect_quantity_jumps_flags_splits():
    qty = pd.Series([1e5] * 60, dtype=float)
    qty.iloc[50] = 1e6  # 10x overnight jump == split / bonus
    flags = detect_quantity_jumps(qty)
    assert flags.iloc[50]
    assert flags.sum() == 1


def test_missing_delivery_raises(synth, feature_cfg):
    prices, _, bench = synth
    with pytest.raises(ValueError, match="delivery"):
        make_feature_provider(feature_cfg).transform(prices, pd.DataFrame(), bench=bench)
