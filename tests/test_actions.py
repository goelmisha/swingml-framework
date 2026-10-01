"""Corporate-action handling for raw quantity series.

The failure mode being guarded against: a 1:2 split doubles the reported share
count overnight, so every trailing quantity mean that straddles the ex-date is
wrong for a full lookback. Detection keys on the exchange's restated previous
close rather than a volume spike, because genuine volume spikes are common and a
spike-based detector would misread them as corporate actions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingml.features.actions import (
    adjust_quantities_for_corporate_actions,
    detect_corporate_actions,
)


def test_detect_corporate_actions_ignores_normal_sessions():
    prior = pd.Series([100.0, 200.0, 300.0, 400.0])
    prev = pd.Series([100.0, 200.0, 300.0, 400.0])  # restated == prior close
    is_action, ratio = detect_corporate_actions(prev, prior)
    assert not is_action.any()
    np.testing.assert_allclose(ratio.to_numpy(), 1.0)


def test_detect_corporate_actions_flags_split_and_bonus():
    prior = pd.Series([1000.0, 1000.0, 1000.0])
    prev = pd.Series([1000.0, 500.0, 333.333333])  # 1:2 split, then 1:2 bonus
    is_action, ratio = detect_corporate_actions(prev, prior)
    assert list(is_action) == [False, True, True]
    assert ratio.iloc[1] == pytest.approx(0.5)
    assert ratio.iloc[2] == pytest.approx(1 / 3)


def test_detect_corporate_actions_handles_zero_and_nan():
    prior = pd.Series([0.0, np.nan, 100.0])
    prev = pd.Series([0.0, 50.0, 100.0])
    is_action, ratio = detect_corporate_actions(prev, prior)
    assert not is_action.any(), "degenerate inputs must not raise or flag"


def test_adjust_quantities_repairs_a_split():
    """A constant-quantity series with a split appended must come back constant.

    Note the ex-date is the restatement day *itself*: the sessions that follow are
    already on the new basis and are not separate events.
    """
    n, split_at = 60, 40
    qty = pd.Series(np.concatenate([np.full(split_at, 1e5), np.full(n - split_at, 2e5)]))
    prior = pd.Series(np.full(n, 100.0))
    prev = prior.copy()
    prev.iloc[split_at] = 50.0  # ex-date restatement

    is_action, ratio = detect_corporate_actions(prev, prior)
    assert list(is_action) == [False] * split_at + [True] + [False] * (n - split_at - 1)

    adjusted = adjust_quantities_for_corporate_actions(qty, ratio, is_action)
    # Everything should now sit on the post-split basis.
    np.testing.assert_allclose(adjusted.to_numpy(), 2e5)
