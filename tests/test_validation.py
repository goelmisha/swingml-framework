"""Walk-forward splitter tests.

This is the mechanism that stops the pipeline from lying to itself, so it gets
tested directly: folds must move forward in time, test blocks must not overlap,
and the purge gap must actually cover the label horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingml.config import ValidationConfig
from swingml.validation import Fold, assert_no_overlap, walk_forward_splits

HORIZON = 10


def _frame(n_sessions: int, rows_per: int = 5) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_sessions)
    return pd.DataFrame(
        [
            {"date": d, "symbol": f"S{k}"}
            for d in dates
            for k in range(rows_per)
        ]
    )


def _cfg(train=100, test=20, purge=3, embargo=3):
    return ValidationConfig(train_days=train, test_days=test, purge_gap_days=purge, embargo_days=embargo)


def test_folds_move_forward_and_have_the_requested_purge():
    df = _frame(400)
    folds = walk_forward_splits(df, _cfg(), HORIZON)

    assert folds, "expected folds to be produced"
    dates = pd.to_datetime(df["date"])
    sessions = np.sort(dates.unique())
    pos = pd.Series(np.arange(len(sessions)), index=sessions).reindex(dates).to_numpy()

    for fold in folds:
        train_pos = pos[fold.train_idx]
        test_pos = pos[fold.test_idx]
        # Strictly forward in time. A label at the last training session spans
        # that session plus HORIZON more, so the test block must start one
        # session beyond that window -- hence HORIZON + 1 of separation.
        assert train_pos.max() < test_pos.min()
        assert train_pos.max() + HORIZON < test_pos.min()
        assert test_pos.min() - train_pos.max() == HORIZON + 1
        assert fold.gap_sessions == HORIZON


def test_purge_and_embargo_are_floored_at_the_label_horizon():
    """A 3-day purge with a 10-day label would leave overlapping outcomes."""
    df = _frame(400)
    fold = walk_forward_splits(df, _cfg(purge=3, embargo=3), HORIZON)[0]
    assert fold.gap_sessions == HORIZON

    # A generous explicit purge is respected rather than shrunk.
    fold_big = walk_forward_splits(df, _cfg(purge=25, embargo=25), HORIZON)[0]
    assert fold_big.gap_sessions == 25


def test_test_blocks_do_not_overlap_between_folds():
    df = _frame(400)
    folds = walk_forward_splits(df, _cfg(), HORIZON)
    seen: set[int] = set()
    for fold in folds:
        rows = set(fold.test_idx.tolist())
        assert not (rows & seen), f"fold {fold.index} test block overlaps an earlier one"
        seen |= rows


def test_no_training_label_reaches_into_its_test_block():
    """The core leak assertion, on the real config geometry."""
    df = _frame(600)
    folds = walk_forward_splits(df, _cfg(train=150, test=40, purge=5, embargo=5), HORIZON)
    assert_no_overlap(folds, df["date"], HORIZON)  # must not raise


def test_leak_detector_actually_fires():
    """The checker must reject a fold whose purge is too small."""
    df = _frame(200)
    dates = pd.to_datetime(df["date"])
    sessions = np.sort(dates.unique())
    pos = pd.Series(np.arange(len(sessions)), index=sessions).reindex(dates).to_numpy()

    # Hand-built bad fold: the last training session sits only 2 sessions before
    # the test block, but labels need 10.
    bad = Fold(
        index=0,
        train_idx=np.flatnonzero(pos == 50),
        test_idx=np.flatnonzero(pos == 52),
        n_train_sessions=1,
        n_test_sessions=1,
        gap_sessions=2,
    )
    with pytest.raises(AssertionError, match="overlap leak"):
        assert_no_overlap([bad], df["date"], HORIZON)


def test_raises_when_history_is_too_short():
    df = _frame(50)
    with pytest.raises(RuntimeError, match="no walk-forward folds fit"):
        walk_forward_splits(df, _cfg(train=100, test=20), HORIZON)


def test_rolls_and_expands_when_asked():
    df = _frame(400)
    rolling = walk_forward_splits(df, _cfg(), HORIZON)
    stepped = walk_forward_splits(df, _cfg(), HORIZON, step_sessions=5)
    assert len(stepped) > len(rolling), "a smaller step must produce more folds"
    # Every test block still starts after its training block ends.
    for fold in stepped:
        assert fold.n_train_sessions == 100 and fold.n_test_sessions == 20
