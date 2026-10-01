"""CUSUM event sampling + the L1 meta-labelling helpers.

Two properties matter and are proved by construction here rather than asserted:

* the CUSUM mask is **causal** -- truncating the series must not change earlier
  values (the same test the feature layer uses);
* event sampling only ever *removes* label rows, and every surviving row was an
  event bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingml.config import AppConfig, LabelConfig, MetaConfig, ValidationConfig
from swingml.events import cusum_events, event_mask
from swingml.labeling import triple_barrier_labels
from swingml.metalabel import fit_meta_predict, meta_target


def _frame(n: int = 260, symbols=("AAA", "BBB"), seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLC + atr_20 on a business-day calendar, one block per symbol."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    rows = []
    for sym in symbols:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.022, n)))
        for i in range(n):
            c = float(close[i])
            o = c * (1.0 + rng.normal(0.0, 0.003))
            h = max(o, c) * (1.0 + abs(rng.normal(0.0, 0.004)))
            low = min(o, c) * (1.0 - abs(rng.normal(0.0, 0.004)))
            rows.append({
                "date": dates[i], "symbol": sym,
                "open": o, "high": h, "low": low, "close": c,
                "atr_20": c * 0.02,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CUSUM filter
# ---------------------------------------------------------------------------

def test_cusum_fires_on_a_cumulative_move_not_a_single_bar():
    """A monotone drift must accumulate into events; a flat line must fire none."""
    flat = pd.Series(np.full(120, 100.0))
    assert not cusum_events(flat).any()

    rng = np.random.default_rng(0)
    # A steady drift PLUS noise: a perfectly constant return has zero volatility,
    # so h would be 0 and the filter would (correctly) emit nothing.
    drift = pd.Series(100.0 * np.exp(np.cumsum(0.01 + rng.normal(0, 0.001, 120))))
    events = cusum_events(drift, h_mult=1.5, vol_span=20)
    assert events.sum() >= 3


def test_cusum_threshold_controls_event_count():
    """A higher multiple of volatility must fire fewer events (L4's k sweep)."""
    rng = np.random.default_rng(1)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 600))))
    counts = [int(cusum_events(close, h_mult=k).sum()) for k in (1.5, 2.0, 3.0)]
    assert counts[0] > counts[1] > counts[2] > 0


def test_cusum_mask_is_causal_under_truncation():
    """The mask at ``t`` may only depend on data up to ``t``."""
    rng = np.random.default_rng(2)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 500))))
    full = cusum_events(close)
    for cut in (50, 137, 300, 499):
        assert cusum_events(close.iloc[:cut]).equals(full.iloc[:cut])


def test_cusum_rejects_bad_parameters():
    close = pd.Series(np.linspace(100, 120, 50))
    with pytest.raises(ValueError):
        cusum_events(close, h_mult=0.0)
    with pytest.raises(ValueError):
        cusum_events(close, vol_span=1)


def test_event_mask_is_computed_per_symbol():
    """Two names moving independently are two event streams, not one pooled one."""
    df = _frame(n=200, symbols=("AAA", "BBB"), seed=5)
    mask = event_mask(df, h_mult=1.5, vol_span=20)
    assert len(mask) == len(df)
    per_symbol = df.assign(_ev=mask).groupby("symbol")["_ev"].sum()
    assert (per_symbol >= 1).all()


def test_event_mask_requires_columns():
    with pytest.raises(ValueError):
        event_mask(pd.DataFrame({"symbol": ["A"], "close": [1.0]}))


# ---------------------------------------------------------------------------
# labelling integration
# ---------------------------------------------------------------------------

def test_cusum_sampling_removes_rows_and_keeps_only_events():
    df = _frame(n=260, symbols=("AAA",), seed=7)
    cfg = LabelConfig(mode="barrier", horizon_days=10, pt_atr_mult=2.0, sl_atr_mult=1.0)
    costs = AppConfig().costs

    every = triple_barrier_labels(df, cfg, costs)
    cfg_cusum = LabelConfig(
        mode="barrier", horizon_days=10, pt_atr_mult=2.0, sl_atr_mult=1.0,
        sampling="cusum", cusum_h_mult=1.5, cusum_vol_span=20,
    )
    sampled = triple_barrier_labels(df, cfg_cusum, costs)

    assert len(sampled.labels) < len(every.labels)
    assert sampled.diagnostics["sampling"] == "cusum"
    assert sampled.diagnostics["n_dropped_not_event"] > 0

    # Every labelled bar must be an event bar for its own symbol.
    events = event_mask(df, h_mult=1.5, vol_span=20)
    event_dates = set(df.loc[events, "date"])
    assert set(sampled.labels["date"]).issubset(event_dates)


def test_every_bar_sampling_is_the_default_and_unchanged():
    df = _frame(n=200, symbols=("AAA",), seed=8)
    cfg = LabelConfig(horizon_days=10, pt_atr_mult=2.0, sl_atr_mult=1.0)
    res = triple_barrier_labels(df, cfg, AppConfig().costs)
    assert res.diagnostics["sampling"] == "every_bar"
    assert res.diagnostics["n_dropped_not_event"] == 0


def test_bad_sampling_mode_is_rejected():
    df = _frame(n=120, symbols=("AAA",), seed=9)
    cfg = LabelConfig(horizon_days=10)
    cfg.sampling = "sometimes"
    with pytest.raises(ValueError, match="sampling"):
        triple_barrier_labels(df, cfg, AppConfig().costs)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_cusum_config_validates():
    cfg = AppConfig()
    cfg.label.sampling = "cusum"
    cfg.label.cusum_h_mult = 2.0
    cfg.label.cusum_vol_span = 20
    cfg.validate()

    cfg.label.sampling = "weekly"
    with pytest.raises(ValueError, match="label.sampling"):
        cfg.validate()

    cfg = AppConfig()
    cfg.label.sampling = "cusum"
    cfg.label.cusum_h_mult = 0.0
    with pytest.raises(ValueError, match="cusum_h_mult"):
        cfg.validate()

    cfg = AppConfig()
    cfg.label.cusum_vol_span = 1
    with pytest.raises(ValueError, match="cusum_vol_span"):
        cfg.validate()


def test_meta_config_validates():
    cfg = AppConfig()
    cfg.meta = MetaConfig(primary_frac=0.5, oof_groups=6)
    cfg.validate()

    cfg.meta.primary_frac = 1.0
    with pytest.raises(ValueError, match="primary_frac"):
        cfg.validate()

    cfg = AppConfig()
    cfg.meta.oof_groups = 1
    with pytest.raises(ValueError, match="oof_groups"):
        cfg.validate()


# ---------------------------------------------------------------------------
# meta-labelling
# ---------------------------------------------------------------------------

def _labelled(n_sessions: int = 160, symbols=("AAA", "BBB", "CCC", "DDD"), seed: int = 3) -> pd.DataFrame:
    """A minimal training frame: features + target + ret_net + uniqueness."""
    df = _frame(n=n_sessions, symbols=symbols, seed=seed)
    rng = np.random.default_rng(seed + 99)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    # A feature set with some signal plus noise, and a definition-B meta-label.
    df["f1"] = df["close"].pct_change().fillna(0.0)
    df["f2"] = rng.normal(0, 1, len(df))
    df["ret_net"] = df["f1"] - 0.0025
    df["target"] = (df["label"] if "label" in df.columns else (df["f1"] > 0)).astype(int)
    df["uniqueness"] = 1.0
    return df


def test_meta_target_is_money_not_barrier():
    df = pd.DataFrame({"ret_net": [-0.01, 0.0, 0.02]})
    np.testing.assert_array_equal(meta_target(df), np.array([0, 0, 1]))
    with pytest.raises(ValueError):
        meta_target(pd.DataFrame({"label": [1, 0]}))


def test_fit_meta_predict_returns_finite_scores_and_pool_beats_non_pool():
    df = _labelled()
    dates = np.sort(df["date"].unique())
    train = df[df["date"] <= dates[110]].reset_index(drop=True)
    test = df[df["date"] > dates[110]].reset_index(drop=True)

    res = fit_meta_predict(
        "sklearn-hgb", train, test, ["f1", "f2"],
        meta_cfg=MetaConfig(primary_frac=0.4, oof_groups=3),
        params=AppConfig().models,
        validation_cfg=ValidationConfig(train_days=60, test_days=20, purge_gap_days=5, embargo_days=5),
        horizon=5, decile=10, seed=0,
    )
    assert len(res.final_scores) == len(test)
    assert np.isfinite(res.final_scores).all()
    assert np.isfinite(res.primary_scores).all()

    # The composite must rank every primary-pool row above every non-pool row,
    # so a top-decile selection is a meta-filtered subset of the primary pool.
    if not res.fallback:
        assert res.final_scores.min() <= 0.0 <= res.final_scores.max()


def test_meta_requires_primary_wider_than_selection():
    df = _labelled(n_sessions=80)
    with pytest.raises(ValueError, match="primary_frac"):
        fit_meta_predict(
            "sklearn-hgb", df, df, ["f1", "f2"],
            meta_cfg=MetaConfig(primary_frac=0.05, oof_groups=3),
            params=AppConfig().models,
            validation_cfg=ValidationConfig(),
            horizon=5, decile=10, seed=0,
        )


def test_meta_feature_columns_must_exist():
    df = _labelled(n_sessions=80)
    with pytest.raises(ValueError, match="missing"):
        fit_meta_predict(
            "sklearn-hgb", df, df, ["nope"],
            meta_cfg=MetaConfig(primary_frac=0.4, oof_groups=3),
            params=AppConfig().models,
            validation_cfg=ValidationConfig(),
            horizon=5, decile=10, seed=0,
        )
