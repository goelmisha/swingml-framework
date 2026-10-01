"""Tests for the trials ledger and the PBO/CSCV implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingml.pbo import cscv_from_path_metrics, probability_of_backtest_overfitting
from swingml.trials import count_trials, load_trials, record_trial, trials_path
from swingml.validation import ValidationConfig, assert_no_overlap_cpcv, cpcv_splits


# ---------------------------------------------------------------------------
# Trials ledger
# ---------------------------------------------------------------------------

def test_record_and_count_trials(tmp_path):
    record_trial("s.py", "ds", "sklearn-hgb", "q1", dataset_dir=tmp_path)
    record_trial("s.py", "ds", "lightgbm", "q2", trials_added=4, dataset_dir=tmp_path)
    c = count_trials(tmp_path)
    assert c["total"] == 5
    assert c["n_runs"] == 2
    assert c["by_script"] == {"s.py": 5}
    entries = load_trials(tmp_path)
    assert entries[1]["trials_added"] == 4


def test_ledger_file_is_jsonl(tmp_path):
    record_trial("a.py", "d", "e", "q", dataset_dir=tmp_path)
    p = trials_path(tmp_path)
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 1
    assert '"script": "a.py"' in lines[0]


# ---------------------------------------------------------------------------
# PBO / CSCV core
# ---------------------------------------------------------------------------

def _paired(agree: bool) -> tuple[np.ndarray, np.ndarray]:
    """Two configs x six path rows with controllable IN/OUT agreement."""
    # Config 0 strong on even paths, weak on odd; config 1 the reverse.
    perf_in = np.array([
        [5.0, 1.0],
        [1.0, 5.0],
        [6.0, 2.0],
        [2.0, 6.0],
        [4.0, 3.0],
        [3.0, 4.0],
    ])
    if agree:
        perf_out = perf_in.copy()
    else:
        # Flip the ordering on every path so the IN winner is the OUT loser.
        perf_out = perf_in[:, ::-1].copy()
    return perf_in, perf_out


def test_pbo_is_zero_when_in_and_out_rankings_agree():
    perf_in, perf_out = _paired(agree=True)
    res = probability_of_backtest_overfitting(perf_in, perf_out)
    assert res.pbo == 0.0
    assert res.n_paths == 6 and res.n_configs == 2


def test_pbo_is_one_when_out_rankings_flip():
    perf_in, perf_out = _paired(agree=False)
    res = probability_of_backtest_overfitting(perf_in, perf_out)
    assert res.pbo == 1.0
    assert (res.logits > 0).all()


def test_pbo_rejects_mismatched_or_degenerate_input():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.ones((3, 2)), np.ones((2, 2)))  # shape
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.ones((1, 2)), np.ones((1, 2)))  # 1 path
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.ones((2, 1)), np.ones((2, 1)))  # 1 config


def test_pbo_nan_configs_are_excluded_per_row():
    perf_in = np.array([
        [np.nan, 2.0, 1.0],
        [3.0, 2.0, 1.0],
    ])
    perf_out = np.array([
        [np.nan, 1.0, 2.0],
        [1.0, 2.0, 3.0],
    ])
    res = probability_of_backtest_overfitting(perf_in, perf_out)
    assert res.detail["n_valid_paths"] == 2  # row 0 ranks its 2 finite configs


# ---------------------------------------------------------------------------
# CSCV assembly (mirror-path pairing)
# ---------------------------------------------------------------------------

def _four_group_metrics() -> tuple[dict, list]:
    """N=4, k=2 -> C(4,2)=6 paths, each the mirror of another. Config 'a'
    wins on paths touching group 0/1 rows, config 'b' elsewhere."""
    combos = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    pm = {}
    for i, combo in enumerate(combos):
        pm[(i, "a")] = {"avg_net": 0.02 if 0 in combo else 0.005, "test_groups": combo}
        pm[(i, "b")] = {"avg_net": 0.005 if 0 in combo else 0.02, "test_groups": combo}
    return pm, combos


def test_cscv_assembles_mirror_pairs_and_runs():
    pm, combos = _four_group_metrics()
    res = cscv_from_path_metrics(pm, ["a", "b"], n_groups=4, n_test_groups=2)
    assert res.n_paths == len(combos)
    assert res.n_configs == 2
    assert 0.0 <= res.pbo <= 1.0


def test_cscv_rejects_asymmetric_geometry():
    pm, _ = _four_group_metrics()
    with pytest.raises(ValueError, match="symmetric"):
        cscv_from_path_metrics(pm, ["a", "b"], n_groups=4, n_test_groups=1)


def test_cscv_requires_test_groups_metadata():
    pm, _ = _four_group_metrics()
    for key in pm:
        pm[key] = {"avg_net": pm[key]["avg_net"]}  # strip test_groups
    with pytest.raises(ValueError, match="test_groups"):
        cscv_from_path_metrics(pm, ["a", "b"], n_groups=4, n_test_groups=2)


# ---------------------------------------------------------------------------
# CPCV geometry
# ---------------------------------------------------------------------------

@pytest.fixture
def session_frame():
    """One row per session for 120 consecutive sessions."""
    dates = pd.bdate_range("2021-01-04", periods=120)
    return pd.DataFrame({"date": dates, "x": 1.0})


def test_cpcv_path_count_and_coverage(session_frame):
    paths = cpcv_splits(session_frame, ValidationConfig(), label_horizon=5, n_groups=6, n_test_groups=2)
    # C(6, 2) = 15 paths; each session is in test C(5, 1) = 5 times.
    assert len(paths) == 15
    test_counts = pd.Series(0, index=range(120))
    for p in paths:
        test_counts.iloc[p.test_idx] += 1
    assert (test_counts == 5).all()


def test_cpcv_purge_removes_gap_around_test_groups(session_frame):
    cfg = ValidationConfig(purge_gap_days=2, embargo_days=2)
    horizon = 5  # effective purge/embargo = max(2, 5) = 5
    paths = cpcv_splits(session_frame, cfg, label_horizon=horizon, n_groups=4, n_test_groups=1)
    p = paths[0]  # test group 0 = sessions 0..29
    assert p.test_idx.min() == 0 and p.test_idx.max() == 29
    # Train must exclude sessions 0..34 (test band + gap on the right edge).
    dates = session_frame["date"]
    unique_dates = np.sort(dates.unique())
    pos_map = pd.Series(np.arange(len(unique_dates)), index=unique_dates)
    train_sessions = {int(pos_map[d]) for d in dates.iloc[p.train_idx]}
    assert max(train_sessions) >= 35  # later groups are untouched
    assert train_sessions & set(range(30, 35)) == set()  # gap band purged


def test_cpcv_leak_detector_fires(session_frame):
    paths = cpcv_splits(session_frame, ValidationConfig(), label_horizon=10, n_groups=4, n_test_groups=1)
    # Claim a horizon of 25 > the purge actually used (10): training labels
    # then resolve inside the adjacent test group, which must be caught.
    with pytest.raises(AssertionError, match="leak"):
        assert_no_overlap_cpcv(paths, session_frame["date"], label_horizon=25)


def test_cpcv_leak_detector_passes_on_its_own_splits(session_frame):
    paths = cpcv_splits(session_frame, ValidationConfig(), label_horizon=10, n_groups=4, n_test_groups=1)
    assert_no_overlap_cpcv(paths, session_frame["date"], label_horizon=10)  # no raise
