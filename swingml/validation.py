"""Walk-forward (purged) validation.

Random K-Fold is banned in this project: shuffling rows puts labels from the
future into the training set, and because a 10-day label spans 10 sessions, it
also puts *overlapping* labels on both sides of every split. Both effects
inflate scores by large margins and are invisible in the output.

The split geometry here is strictly forward in time:

    |<---- train (T) ---->|<- gap ->|<---- test (S) ---->|<-- next fold ...
                           purge+embargo

Two distinct leakage channels are handled by that one gap:

* **label overlap** -- a training label at the end of the train block is only
  resolved ``horizon`` sessions later, which reaches into the test block. Drop
  the last ``horizon`` training sessions.
* **feature-window overlap** -- training rows adjacent to the test block share
  input windows with it. Same remedy.

Purge and embargo are the two names for these channels; both are floored at the
label horizon (see :meth:`ValidationConfig.effective_purge_days`), and the gap
used is the larger of the two.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from swingml.config import ValidationConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """One walk-forward split, expressed as positional row indices."""

    index: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    n_train_sessions: int
    n_test_sessions: int
    gap_sessions: int

    def describe(self) -> str:
        return (
            f"fold {self.index}: train {self.n_train_sessions} sessions -> "
            f"gap {self.gap_sessions} -> test {self.n_test_sessions} sessions"
        )


def walk_forward_splits(
    data: pd.DataFrame,
    cfg: ValidationConfig,
    label_horizon: int,
    date_col: str = "date",
    step_sessions: int | None = None,
) -> list[Fold]:
    """Build purged walk-forward folds over the sorted unique sessions.

    Parameters
    ----------
    data
        Any frame carrying the ``date`` column; only the calendar is used.
    cfg
        Walk-forward geometry (train/test sizes, purge, embargo).
    label_horizon
        Label horizon in sessions; floors the purge/embargo.
    step_sessions
        How far to advance between folds. Defaults to the test-block length, so
        test blocks tile the timeline without overlapping.

    Returns
    -------
    A list of :class:`Fold`, each holding positional indices into ``data``.
    """
    dates = pd.to_datetime(data[date_col])
    unique_dates = np.sort(dates.unique())
    n_sessions = len(unique_dates)

    train_sessions = int(cfg.train_days)
    test_sessions = int(cfg.test_days)
    purge = cfg.effective_purge_days(label_horizon)
    embargo = cfg.effective_embargo_days(label_horizon)
    gap = max(purge, embargo)

    if train_sessions < 1 or test_sessions < 1:
        raise ValueError("train_days and test_days must be >= 1")
    step = int(step_sessions) if step_sessions else test_sessions
    if step < 1:
        raise ValueError("step_sessions must be >= 1")

    # Map each row to its session position, so folds can be selected by index.
    session_pos = pd.Series(np.arange(n_sessions), index=unique_dates)
    row_pos = dates.map(session_pos).to_numpy()

    folds: list[Fold] = []
    fold_id = 0
    start = 0
    while start + train_sessions + gap + test_sessions <= n_sessions:
        train_start = start
        train_end = train_start + train_sessions          # exclusive
        test_start = train_end + gap
        test_end = test_start + test_sessions

        train_sess = np.arange(train_start, train_end)
        test_sess = np.arange(test_start, test_end)

        train_mask = np.isin(row_pos, train_sess)
        test_mask = np.isin(row_pos, test_sess)

        folds.append(
            Fold(
                index=fold_id,
                train_idx=np.flatnonzero(train_mask),
                test_idx=np.flatnonzero(test_mask),
                n_train_sessions=len(train_sess),
                n_test_sessions=len(test_sess),
                gap_sessions=gap,
            )
        )
        fold_id += 1
        start += step

    if not folds:
        raise RuntimeError(
            f"no walk-forward folds fit: {n_sessions} sessions available but "
            f"train {train_sessions} + gap {gap} + test {test_sessions} "
            f"= {train_sessions + gap + test_sessions} required"
        )

    logger.info(
        "walk-forward: %d folds over %d sessions (train %d, gap %d [purge %d/embargo %d], test %d)",
        len(folds), n_sessions, train_sessions, gap, purge, embargo, test_sessions,
    )
    return folds


def assert_no_overlap(folds: list[Fold], row_dates: pd.Series, label_horizon: int) -> None:
    """Verify no training label window reaches into its test block.

    A training row at session ``t`` carries a label resolved at ``t + horizon``.
    That must stay strictly before the test block's first session.
    """
    dates = pd.to_datetime(row_dates)
    unique_dates = np.sort(dates.unique())
    session_pos = pd.Series(np.arange(len(unique_dates)), index=unique_dates)
    pos = dates.map(session_pos).to_numpy()

    for fold in folds:
        if len(fold.train_idx) == 0 or len(fold.test_idx) == 0:
            continue
        last_train_pos = pos[fold.train_idx].max()
        first_test_pos = pos[fold.test_idx].min()
        resolved_at = last_train_pos + label_horizon
        if resolved_at >= first_test_pos:
            raise AssertionError(
                f"fold {fold.index}: last training label resolves at session {resolved_at} "
                f"but the test block starts at {first_test_pos} -- label overlap leak"
            )


# ---------------------------------------------------------------------------
# Combinatorial Purged Cross-Validation (CPCV)
# ---------------------------------------------------------------------------
# Walk-forward produces ONE chronological path, so one experiment is one
# sample of the strategy's behaviour. CPCV (Lopez de Prado, AFML ch. 12)
# splits the timeline into N contiguous groups, then forms every combination
# of k groups as the test set -- the remaining groups are train, purged around
# each test group. That yields C(N, k) overlapping train/test paths, i.e. a
# DISTRIBUTION of results, which is what the Probability of Backtest
# Overfitting (PBO) needs as input. Walk-forward stays the primary splitter;
# CPCV exists to measure how fragile the walk-forward number is.


@dataclass(frozen=True)
class CpcvPath:
    """One CPCV combination, expressed as positional row indices."""

    index: int
    test_groups: tuple[int, ...]
    train_idx: np.ndarray
    test_idx: np.ndarray
    n_train_sessions: int
    n_test_sessions: int
    gap_sessions: int

    def describe(self) -> str:
        return (
            f"path {self.index}: test groups {self.test_groups} "
            f"({self.n_test_sessions} sessions), train {self.n_train_sessions}, "
            f"gap {self.gap_sessions}"
        )


def cpcv_splits(
    data: pd.DataFrame,
    cfg: ValidationConfig,
    label_horizon: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    date_col: str = "date",
) -> list[CpcvPath]:
    """Build all C(N, k) purged combinatorial splits over the sorted sessions.

    Each test group is a contiguous block of sessions; a combination is a set of
    ``n_test_groups`` groups. Train is everything else, minus a purge band of
    ``gap = max(purge, embargo)`` sessions around EVERY test group (the band
    covers both the label-overlap and feature-window channels, exactly as in
    :func:`walk_forward_splits`).

    Every session lands in test exactly ``C(N-1, k-1)`` times, so no period is
    privileged. Groups are equal-sized contiguous blocks, so with ``N`` groups
    each block is ``n_sessions // N`` sessions.
    """
    dates = pd.to_datetime(data[date_col])
    unique_dates = np.sort(dates.unique())
    n_sessions = len(unique_dates)

    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if not 1 <= n_test_groups < n_groups:
        raise ValueError("n_test_groups must be in [1, n_groups - 1]")
    if n_sessions < n_groups:
        raise RuntimeError(
            f"{n_sessions} sessions cannot form {n_groups} groups"
        )

    purge = cfg.effective_purge_days(label_horizon)
    embargo = cfg.effective_embargo_days(label_horizon)
    gap = max(purge, embargo)

    # Contiguous, near-even session groups.
    bounds = np.linspace(0, n_sessions, n_groups + 1).astype(int)
    group_ranges = [(int(bounds[g]), int(bounds[g + 1])) for g in range(n_groups)]

    session_pos = pd.Series(np.arange(n_sessions), index=unique_dates)
    row_pos = dates.map(session_pos).to_numpy()
    all_sessions = np.arange(n_sessions)

    paths: list[CpcvPath] = []
    for path_id, combo in enumerate(
        itertools.combinations(range(n_groups), n_test_groups)
    ):
        test_sess = np.concatenate(
            [np.arange(group_ranges[g][0], group_ranges[g][1]) for g in combo]
        )
        # Purge a band around each test group out of train.
        banned = np.zeros(n_sessions, dtype=bool)
        banned[test_sess] = True
        for g in combo:
            lo = max(0, group_ranges[g][0] - gap)
            hi = min(n_sessions, group_ranges[g][1] + gap)
            banned[lo:hi] = True
        train_sess = all_sessions[~banned]

        paths.append(
            CpcvPath(
                index=path_id,
                test_groups=combo,
                train_idx=np.flatnonzero(np.isin(row_pos, train_sess)),
                test_idx=np.flatnonzero(np.isin(row_pos, test_sess)),
                n_train_sessions=len(train_sess),
                n_test_sessions=len(test_sess),
                gap_sessions=gap,
            )
        )

    if not paths:  # pragma: no cover - combinations() always yields
        raise RuntimeError("no CPCV paths were formed")

    logger.info(
        "cpcv: %d paths over %d sessions (N=%d groups, k=%d test groups, gap %d "
        "[purge %d/embargo %d]); each session is in test %d times",
        len(paths), n_sessions, n_groups, n_test_groups, gap, purge, embargo,
        _test_coverage_count(n_groups, n_test_groups),
    )
    return paths


def _test_coverage_count(n_groups: int, n_test_groups: int) -> int:
    """How many paths place a given session in test: C(N-1, k-1)."""
    from math import comb

    return comb(n_groups - 1, n_test_groups - 1)


def assert_no_overlap_cpcv(
    paths: list[CpcvPath], row_dates: pd.Series, label_horizon: int
) -> None:
    """Verify no training label window reaches into any test group of its path.

    Stronger than the walk-forward check: a path has MULTIPLE test blocks, so
    every training label must resolve strictly outside every one of them.
    """
    dates = pd.to_datetime(row_dates)
    unique_dates = np.sort(dates.unique())
    session_pos = pd.Series(np.arange(len(unique_dates)), index=unique_dates)
    pos = dates.map(session_pos).to_numpy()

    for path in paths:
        if len(path.train_idx) == 0 or len(path.test_idx) == 0:
            continue
        test_positions = pos[path.test_idx]
        # A training label at p resolves at p + horizon; it must never land on
        # a session that this path's test set owns.
        resolved = pos[path.train_idx] + label_horizon
        clash = np.intersect1d(resolved, np.unique(test_positions))
        if clash.size:
            raise AssertionError(
                f"cpcv path {path.index} (groups {path.test_groups}): "
                f"{clash.size} training labels resolve inside the test block "
                "-- label overlap leak"
            )
        if np.intersect1d(pos[path.train_idx], pos[path.test_idx]).size:
            raise AssertionError(
                f"cpcv path {path.index}: train and test share rows"
            )
