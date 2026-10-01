"""Feature-provider interface.

The modelling layers (labelling, validation, model fitting, evaluation) never
import a concrete feature set. They resolve a :class:`FeatureProvider` from
configuration and talk to it through this contract, so the same leak-free
plumbing runs unchanged against any feature set that honours it.

A provider owns three things:

``transform``
    Build the causal feature matrix from price + bhavcopy panels.
``groups``
    Named feature blocks, so whole blocks can be pruned on importance instead
    of tuning thousands of individual columns.
``context_columns``
    Pass-through columns that are *not* model inputs but must survive into the
    labelled dataset (labelling needs the forward intrabar path and the
    barrier-width ATR).

Causality is the provider's responsibility and is testable from the outside: a
feature computed on data up to ``t`` must be identical whether or not data after
``t`` exists. See ``tests/test_leakage.py``.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a config import cycle
    from swingml.config import FeatureConfig

#: Context columns every provider must carry. Deliberately minimal: the labeller
#: needs open/high/low/close to resolve barriers and ``atr_20`` to size them.
#: Providers may add to this tuple but must not remove from it.
REQUIRED_CONTEXT_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "atr_20")


class FeatureProvider(abc.ABC):
    """Turns raw price + delivery panels into a causal, model-ready matrix."""

    #: Registry name used in ``features.provider``.
    name: str = "unnamed"

    #: Feature blocks keyed by block name. Populate in the subclass.
    groups: dict[str, tuple[str, ...]] = {}

    def __init__(self, cfg: "FeatureConfig") -> None:
        self.cfg = cfg
        #: Populated by :meth:`transform`; the dataset records it in meta.json.
        self.feature_columns: list[str] = []
        #: Split/bonus events detected and repaired during the last transform.
        #: Surfaced because a silently rising count is how a data problem shows
        #: up as a modelling result.
        self.corporate_actions_detected: int = 0

    @property
    def context_columns(self) -> tuple[str, ...]:
        """Columns carried through but excluded from the model input set."""
        return REQUIRED_CONTEXT_COLUMNS

    @abc.abstractmethod
    def transform(
        self,
        prices: dict[str, pd.DataFrame],
        delivery: pd.DataFrame,
        bench: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return a long frame with columns ``[date, symbol, <features>, <context>]``.

        Implementations must be causal: no centred windows, no backward fills
        from the future, and no statistic computed over the full sample.
        """

    # -- shared helpers ----------------------------------------------------
    def group_of(self, column: str) -> str:
        """Map a feature name back to its block (cross-sectional ranks included)."""
        name = column[len("xs_rank_"):] if column.startswith("xs_rank_") else column
        for group, cols in self.groups.items():
            if name in cols:
                return group
        return "other"

    def check_output(self, full: pd.DataFrame) -> None:
        """Fail loudly if a provider breaks the contract.

        Called at the end of :meth:`transform` by well-behaved implementations.
        Catches the two mistakes that silently poison everything downstream: a
        missing context column (the labeller would KeyError, or worse, quietly
        pick up the wrong barrier width) and an empty feature set.
        """
        missing = [c for c in self.context_columns if c not in full.columns]
        if missing:
            raise ValueError(
                f"{type(self).__name__} did not emit required context column(s): {missing}"
            )
        if not self.feature_columns:
            raise ValueError(f"{type(self).__name__} produced no feature columns")
