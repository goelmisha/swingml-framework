"""Tests for the Step-4 scorecard: portfolio metrics and the deflated Sharpe."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from swingml.scorecard import (
    SESSIONS_PER_YEAR,
    benchmark_period_returns,
    beta_analysis,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    format_beta_control,
    format_scorecard,
    performance_metrics,
    period_returns,
)


def _block(n_dates: int = 3, n_names: int = 20, decile: int = 10) -> pd.DataFrame:
    """Deterministic block: the top slice by `score` wins, the rest loses."""
    keep = max(1, n_names // decile)
    rows = []
    for d in range(n_dates):
        for s in range(n_names):
            rows.append({
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                "symbol": f"S{s:02d}",
                "score": float(s),
                "ret_net": 0.04 if s >= n_names - keep else -0.005,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Portfolio series
# ---------------------------------------------------------------------------

def test_period_returns_is_the_equal_weight_top_slice():
    block = _block()
    r = period_returns(block, block["score"].to_numpy(), decile=10)
    assert len(r) == 3
    assert np.allclose(r.to_numpy(), 0.04)


def test_period_returns_excludes_non_selected_rows():
    block = _block()
    block.loc[block["symbol"] == "S00", "ret_net"] = -0.90  # worst name, never selected
    r = period_returns(block, block["score"].to_numpy(), decile=10)
    assert np.allclose(r.to_numpy(), 0.04)


def test_period_returns_stride_drops_overlapping_periods():
    """With a 10-session label, one period in ten -- not one per session."""
    block = _block(n_dates=30)
    scores = block["score"].to_numpy()
    every = period_returns(block, scores, decile=10, stride=1)
    strided = period_returns(block, scores, decile=10, stride=10)
    assert len(every) == 30
    assert len(strided) == 3


def test_overlap_inflates_the_count_but_not_the_mean():
    """The signature of the overlap bug, pinned so it cannot come back.

    Overlapping sampling leaves each period's return identical and multiplies
    the number of them, so the cumulative return explodes while the mean is
    unchanged.
    """
    block = _block(n_dates=30)
    scores = block["score"].to_numpy()
    overlapping = period_returns(block, scores, decile=10, stride=1)
    non_overlapping = period_returns(block, scores, decile=10, stride=10)
    assert overlapping.mean() == pytest.approx(non_overlapping.mean())
    cumulative = {
        "overlapping": float(np.prod(1.0 + overlapping.to_numpy()) - 1.0),
        "non_overlapping": float(np.prod(1.0 + non_overlapping.to_numpy()) - 1.0),
    }
    assert cumulative["overlapping"] > cumulative["non_overlapping"]


def test_period_returns_can_score_the_fixed_hold_trade():
    """Same selection, different trade: barrier (ret_net) vs fixed hold (fwd_ret)."""
    block = _block()
    block["fwd_ret_10"] = np.where(block["symbol"].str[-2:].astype(int) >= 10, 0.09, -0.02)
    barrier = period_returns(block, block["score"].to_numpy(), decile=10)
    fixed = period_returns(block, block["score"].to_numpy(), decile=10, return_col="fwd_ret_10")
    assert np.allclose(barrier.to_numpy(), 0.04)
    assert np.allclose(fixed.to_numpy(), 0.09)


def test_extra_friction_is_charged_once_per_period_not_per_name():
    block = _block()
    r = period_returns(block, block["score"].to_numpy(), decile=10, extra_friction=0.0025)
    assert np.allclose(r.to_numpy(), 0.04 - 0.0025)


def test_period_returns_rejects_a_missing_return_column():
    block = _block()
    with pytest.raises(ValueError):
        period_returns(block, block["score"].to_numpy(), return_col="fwd_ret_10")


def test_beta_analysis_exposes_the_residual_series_for_scoring():
    rng = np.random.default_rng(3)
    bench = pd.Series(rng.normal(0.004, 0.02, 120))
    strat = 0.5 * bench + rng.normal(0.002, 0.01, 120)
    an = beta_analysis(strat, bench, periods_per_year=25.2)
    residual = an["residual_series"]
    assert len(residual) == an["n_periods"]
    assert residual.mean() == pytest.approx(an["residual_mean"])
    perf = performance_metrics(residual, periods_per_year=25.2)
    assert perf.sharpe_annualised == pytest.approx(an["residual_sharpe_annualised"])


def test_stride_must_be_positive_and_lengths_must_match():
    block = _block()
    with pytest.raises(ValueError):
        period_returns(block, block["score"].to_numpy(), stride=0)
    with pytest.raises(ValueError):
        period_returns(block, block["score"].to_numpy()[:-1])


def test_period_returns_survives_a_non_rangeindex_block():
    """A fold's test block keeps its frame index (df.iloc), not 0..n-1.

    top_fraction_mask aligns scores and dates by index, so a fresh-index score
    vector against an offset-index block silently matches nothing and yields an
    empty series -- which is exactly how the first scorecard run came back with
    0 sessions.
    """
    block = _block().set_index(pd.Index(range(100, 100 + 3 * 20)))
    assert block.index[0] == 100
    r = period_returns(block, block["score"].to_numpy(), decile=10)
    assert len(r) == 3
    assert np.allclose(r.to_numpy(), 0.04)


def test_period_returns_respects_the_decile_size():
    # Distinct returns so the slice size is unambiguous: score s returns 0.01*(s+1).
    rows = [
        {"date": pd.Timestamp("2024-01-01"), "symbol": f"S{s:02d}",
         "score": float(s), "ret_net": 0.01 * (s + 1)}
        for s in range(20)
    ]
    block = pd.DataFrame(rows)
    scores = block["score"].to_numpy()
    assert period_returns(block, scores, decile=10).iloc[0] == pytest.approx((0.19 + 0.20) / 2)
    assert period_returns(block, scores, decile=2).iloc[0] == pytest.approx((0.11 + 0.20) / 2)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def test_max_drawdown_matches_the_equity_curve():
    perf = performance_metrics(pd.Series([0.10, -0.20]))
    # equity 1.10 -> 0.88 ; drawdown = 1 - 0.88/1.10
    assert perf.max_drawdown == pytest.approx(1.0 - 0.88 / 1.10)


def test_profit_factor_and_hit_rate():
    perf = performance_metrics(pd.Series([0.02, -0.01, -0.01]))
    assert perf.profit_factor == pytest.approx(1.0)
    assert perf.hit_rate == pytest.approx(1 / 3)


def test_sharpe_is_per_period_and_annualises_by_sqrt_periods_per_year():
    r = pd.Series([0.01, -0.005, 0.02, -0.004, 0.01])
    perf = performance_metrics(r)
    assert perf.sharpe == pytest.approx(r.mean() / r.std(ddof=1))
    assert perf.sharpe_annualised == pytest.approx(perf.sharpe * math.sqrt(SESSIONS_PER_YEAR))


def test_annualisation_follows_the_period_length_not_252():
    """10-session periods: ~25 periods a year, not 252."""
    r = pd.Series([0.01, -0.005, 0.02, -0.004, 0.01])
    perf = performance_metrics(r, periods_per_year=SESSIONS_PER_YEAR / 10, period_sessions=10)
    assert perf.sharpe_annualised == pytest.approx(perf.sharpe * math.sqrt(SESSIONS_PER_YEAR / 10))
    assert perf.sharpe_annualised < performance_metrics(r).sharpe_annualised


def test_zero_dispersion_series_has_no_sharpe():
    perf = performance_metrics(pd.Series([0.01, 0.01, 0.01]))
    assert np.isnan(perf.sharpe)
    assert np.isnan(perf.sharpe_annualised)


def test_empty_series_returns_empty_performance():
    perf = performance_metrics(pd.Series(dtype=float))
    assert perf.n_periods == 0
    assert np.isnan(perf.mean_return)


# ---------------------------------------------------------------------------
# Deflated Sharpe
# ---------------------------------------------------------------------------

def test_expected_max_sharpe_grows_with_trials_and_with_trial_spread():
    one = expected_max_sharpe(0.0004, 2)
    many = expected_max_sharpe(0.0004, 1000)
    assert many > one > 0
    assert expected_max_sharpe(0.0016, 156) == pytest.approx(2 * expected_max_sharpe(0.0004, 156))


def test_expected_max_sharpe_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        expected_max_sharpe(0.0004, 1)
    with pytest.raises(ValueError):
        expected_max_sharpe(-0.1, 10)


def test_dsr_falls_as_the_search_widens():
    kwargs = dict(sharpe=0.05, n_obs=500, trial_sharpe_variance=0.0004, skew=0.0, kurtosis=3.0)
    few = deflated_sharpe_ratio(n_trials=2, **kwargs)
    many = deflated_sharpe_ratio(n_trials=1000, **kwargs)
    assert few["dsr"] > many["dsr"]


def test_a_sharpe_below_the_expected_maximum_is_not_significant():
    # trial sd 0.02/session (0.32 annualised) over 156 trials puts the noise bar
    # around 0.054/session -- an observed 0.05 must not survive it.
    out = deflated_sharpe_ratio(
        sharpe=0.05, n_obs=500, n_trials=156, trial_sharpe_variance=0.0004,
        skew=0.0, kurtosis=3.0,
    )
    assert out["expected_max_sharpe"] > 0.05
    assert out["significant"] is False
    assert out["dsr"] < 0.5


def test_a_large_sharpe_clears_the_noise_bar():
    out = deflated_sharpe_ratio(
        sharpe=0.30, n_obs=500, n_trials=156, trial_sharpe_variance=0.0004,
        skew=0.0, kurtosis=3.0,
    )
    assert out["significant"] is True
    assert out["dsr"] >= 0.95


def test_dsr_rejects_bad_geometry():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(sharpe=0.05, n_obs=1, n_trials=10, trial_sharpe_variance=0.0004)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(sharpe=0.05, n_obs=100, n_trials=1, trial_sharpe_variance=0.0004)


def test_nan_sharpe_yields_a_nan_dsr_rather_than_an_exception():
    out = deflated_sharpe_ratio(
        sharpe=float("nan"), n_obs=100, n_trials=10, trial_sharpe_variance=0.0004
    )
    assert np.isnan(out["dsr"])
    assert out["significant"] is False


# ---------------------------------------------------------------------------
# Beta control
# ---------------------------------------------------------------------------

def _bench(n_sessions: int = 30, step: float = 0.005) -> pd.DataFrame:
    """Benchmark whose open is 100 and close rises by `step` each session."""
    idx = pd.bdate_range("2024-01-01", periods=n_sessions)
    close = 100.0 * (1.0 + step) ** np.arange(1, n_sessions + 1)
    return pd.DataFrame({"open": np.full(n_sessions, 100.0), "close": close}, index=idx)


def test_benchmark_periods_use_next_open_and_the_label_horizon():
    bench = _bench(n_sessions=30, step=0.005)
    starts = pd.Index(bench.index[::10][:2])          # two period starts, 10 apart
    r = benchmark_period_returns(bench, starts, horizon=10)
    # enter at the NEXT open (100) and exit 10 sessions later
    assert len(r) == 2
    expected_first = bench["close"].iloc[10] / 100.0 - 1.0
    assert r.iloc[0] == pytest.approx(expected_first)


def test_benchmark_periods_charge_friction_and_skip_runs_off_the_end():
    bench = _bench(n_sessions=12)
    starts = pd.Index([bench.index[0], bench.index[-1]])   # second start has no room
    with_friction = benchmark_period_returns(bench, starts, horizon=10, friction=0.0025)
    without = benchmark_period_returns(bench, starts, horizon=10)
    assert len(with_friction) == len(without) == 1
    assert without.iloc[0] - with_friction.iloc[0] == pytest.approx(0.0025)


def test_forward_returns_are_per_symbol_and_never_cross_a_symbol_boundary():
    """Definition C's return, now built in the library so it cannot drift."""
    from swingml.evaluation import add_forward_returns

    rows = []
    for sym, closes in (("AAA", [10.0, 11.0, 12.0]), ("BBB", [50.0, 40.0, 45.0])):
        for d, c in enumerate(closes):
            rows.append({"symbol": sym, "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                         "close": c})
    out = add_forward_returns(pd.DataFrame(rows), [1])
    aaa = out[out["symbol"] == "AAA"]["fwd_ret_1"].to_list()
    bbb = out[out["symbol"] == "BBB"]["fwd_ret_1"].to_list()
    assert aaa[0] == pytest.approx(0.10)
    assert bbb[0] == pytest.approx(-0.20)
    assert np.isnan(aaa[-1]) and np.isnan(bbb[-1])   # last bar per symbol has no future


def test_forward_returns_preserve_the_callers_row_order():
    """Date-major frames are the normal case, and the shift must not reorder them.

    Returning the (symbol, date)-sorted frame misaligns the forward return
    against a parallel score vector, so a selection gets scored on other rows'
    returns. That is what produced a wrong kill-test run before this was fixed.
    """
    from swingml.evaluation import add_forward_returns

    rows = []
    for d in range(2):
        for sym, closes in (("AAA", [10.0, 20.0]), ("BBB", [50.0, 25.0])):
            rows.append({"symbol": sym, "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                         "close": closes[d]})
    src = pd.DataFrame(rows)
    out = add_forward_returns(src, [1])
    assert list(zip(out["symbol"], out["close"])) == list(zip(src["symbol"], src["close"]))
    assert list(out.index) == list(src.index)


def test_benchmark_of_an_empty_frame_is_empty_not_an_exception():
    assert benchmark_period_returns(pd.DataFrame(), pd.Index([]), horizon=10).empty


def test_beta_is_recovered_when_the_strategy_is_a_scaled_market():
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0.004, 0.02, 200))
    strat = 0.5 * bench + rng.normal(0.001, 0.001, 200)     # beta 0.5 + small alpha
    out = beta_analysis(strat, bench, periods_per_year=25.2)
    assert out["beta"] == pytest.approx(0.5, abs=0.02)
    assert out["r_squared"] > 0.98
    assert out["alpha_per_period"] == pytest.approx(0.001, abs=0.0005)


def test_pure_market_exposure_has_no_meaningful_residual_sharpe():
    """Strategy == market: the hedge removes everything but float noise.

    That residual's Sharpe is an arbitrary signed number (it came out -0.53 from
    1e-16 dispersion), so it must be reported as nan rather than printed as if
    it were a market-neutral result.
    """
    rng = np.random.default_rng(1)
    bench = pd.Series(rng.normal(0.004, 0.02, 200))
    out = beta_analysis(bench.copy(), bench, periods_per_year=25.2)   # strategy IS the market
    assert out["beta"] == pytest.approx(1.0, abs=1e-9)
    assert abs(out["residual_mean"]) < 1e-15
    assert np.isnan(out["residual_sharpe_annualised"])


def test_beta_analysis_degrades_gracefully():
    out = beta_analysis(pd.Series([0.01, 0.02]), pd.Series([0.01, 0.02]), periods_per_year=25.2)
    assert out["n_periods"] == 2
    assert np.isnan(out["beta"])
    assert "unavailable" in format_beta_control(out)


def test_format_beta_control_reports_alpha_and_residual():
    rng = np.random.default_rng(2)
    bench = pd.Series(rng.normal(0.004, 0.02, 100))
    strat = 0.5 * bench + rng.normal(0.002, 0.01, 100)
    out = beta_analysis(strat, bench, periods_per_year=25.2)
    text = format_beta_control(out, label="hgb arm A")
    assert "hgb arm A vs market" in text
    assert "beta-hedged residual" in text


def test_format_scorecard_annualises_the_noise_bar_with_the_period_length():
    """A 10-session series must not annualise its noise bar by sqrt(252)."""
    r = pd.Series([0.01, -0.005, 0.02, -0.004, 0.01, 0.015, -0.002])
    perf = performance_metrics(r, periods_per_year=SESSIONS_PER_YEAR / 10, period_sessions=10)
    dsr = deflated_sharpe_ratio(
        sharpe=perf.sharpe, n_obs=perf.n_periods, n_trials=162,
        trial_sharpe_variance=0.0004, skew=perf.skew, kurtosis=perf.kurtosis,
    )
    text = format_scorecard(perf, dsr, label="t")
    expected = dsr["expected_max_sharpe"] * math.sqrt(SESSIONS_PER_YEAR / 10)
    assert f"{expected:.2f}" in text
    assert "162 trials" in text


def test_format_scorecard_states_the_verdict_and_the_trial_count():
    perf = performance_metrics(pd.Series([0.01, -0.005, 0.02, -0.004, 0.01]))
    dsr = deflated_sharpe_ratio(
        sharpe=perf.sharpe, n_obs=perf.n_periods, n_trials=156,
        trial_sharpe_variance=0.0004, skew=perf.skew, kurtosis=perf.kurtosis,
    )
    text = format_scorecard(perf, dsr, label="test")
    assert "test:" in text
    assert "156 trials" in text
    assert ("SIGNIFICANT" in text) or ("NOT significant" in text)
