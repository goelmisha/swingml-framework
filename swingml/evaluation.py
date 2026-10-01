"""Shared evaluation -- ONE implementation of the Tier-1 metrics.

Why this module exists
----------------------
Three different "precision" definitions circulated in this repo and were
compared against each other, which produced numbers that looked comparable but
measured different trades:

    A: P(label == 1)              -- buy, exit at whichever barrier hits first
    B: P(ret_net > 0)             -- same barrier trade, scored on money made
    C: P(fwd_ret > friction)      -- buy and hold exactly ``horizon`` sessions

Every experiment must now report A and B together from THIS module, so a
feature number and a model number can never again be produced by two different
implementations. Definition C lives only in ``scripts/signal_check.py`` and is
always printed with its definition attached.

Base rates belong to the dataset, not to a definition, so every block reports
its own and each selection is scored against it. A model that reports only A
will understate; one that reports only B will flatter. Report both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

#: Temporary ordering column used by :func:`add_forward_returns` to restore the
#: caller's row order after the shift requires a sorted frame.
_ORDER_COL = "__swingml_row_order__"


def add_forward_returns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Close-to-close forward returns per symbol. The target, never a feature.

    This is the return that defines **definition C** (``P(fwd_ret > friction)``),
    so it lives beside the other definitions rather than inside the one script
    that prints it. It is deliberately the last thing that may be built from a
    feature matrix: every column here looks forward.

    The shift needs ``(symbol, date)`` order, but the result is returned in the
    **caller's original row order**. Returning the sorted frame instead silently
    misaligns a value column against a parallel score vector -- exactly how a
    selection can end up scored against other rows' returns.
    """
    work = df.copy()
    work[_ORDER_COL] = np.arange(len(work))
    work = work.sort_values(["symbol", "date"])
    for h in horizons:
        work[f"fwd_ret_{h}"] = (
            work.groupby("symbol", observed=True)["close"].shift(-h) / work["close"] - 1.0
        )
    return work.sort_values(_ORDER_COL).drop(columns=_ORDER_COL)


@dataclass
class SelectionStats:
    """Tier-1 metrics for one selection rule on one evaluation block.

    "Selection" is whatever the experiment ranked to the top (a model's
    predicted probability, a feature's within-date rank, ...). The base rates
    are computed over the whole block, not the selection.
    """

    n_rows: int = 0
    n_selected: int = 0
    #: Definition A: P(label == 1 | selected) and its base rate.
    precision_a: float = float("nan")
    base_precision_a: float = float("nan")
    #: Definition B: P(ret_net > 0 | selected) and its base rate.
    precision_b: float = float("nan")
    base_precision_b: float = float("nan")
    #: Money made by the selections after friction (definition-B average).
    avg_net: float = float("nan")
    base_avg_net: float = float("nan")
    #: AUC against ``target`` when the block has both classes.
    auc: float = float("nan")

    def lifts(self) -> dict[str, float]:
        """Selected minus base -- the number an edge claim rests on."""
        return {
            "precision_a_lift": self.precision_a - self.base_precision_a,
            "precision_b_lift": self.precision_b - self.base_precision_b,
            "net_lift": self.avg_net - self.base_avg_net,
        }

    def as_row(self, **extra) -> dict:
        out = {
            "n_rows": self.n_rows,
            "n_selected": self.n_selected,
            "precision_a": self.precision_a,
            "precision_b": self.precision_b,
            "avg_net": self.avg_net,
            "base_precision_a": self.base_precision_a,
            "base_precision_b": self.base_precision_b,
            "base_avg_net": self.base_avg_net,
            "auc": self.auc,
        }
        out.update(extra)
        return out


def evaluate_selections(
    test: pd.DataFrame,
    scores: np.ndarray,
    target_col: str = "target",
    decile: int = 10,
    auc: bool = True,
) -> SelectionStats:
    """Score a model score vector on a test block under definitions A and B.

    Parameters
    ----------
    test
        Test rows; must carry ``label`` (A), ``ret_net`` (B) and ``target_col``.
    scores
        Higher = more buy-worthy. Same length and order as ``test``.
    decile
        Selection fraction is 1/decile of the within-date rank (10 = top decile).
    """
    if len(test) != len(scores):
        raise ValueError(f"test rows ({len(test)}) != scores ({len(scores)})")
    d = test[["date", "label", "ret_net", target_col]].copy()
    d["score"] = np.asarray(scores, dtype=float)

    sel_mask = top_fraction_mask(d["score"], d["date"], decile)
    sel = d[sel_mask]

    st = SelectionStats(n_rows=len(d), n_selected=int(sel_mask.sum()))
    if len(sel) == 0:
        return st

    st.precision_a = float((sel["label"] == 1).mean())
    st.base_precision_a = float((d["label"] == 1).mean())
    st.precision_b = float((sel["ret_net"] > 0).mean())
    st.base_precision_b = float((d["ret_net"] > 0).mean())
    st.avg_net = float(sel["ret_net"].mean())
    st.base_avg_net = float(d["ret_net"].mean())
    y = d[target_col].to_numpy()
    if auc and np.unique(y).size > 1:
        st.auc = float(roc_auc_score(y, d["score"]))
    return st


def top_fraction_mask(scores: pd.Series, dates: pd.Series, decile: int) -> np.ndarray:
    """Within-date top-1/decile mask. Ties broken by first occurrence, which
    makes the mask deterministic for a fixed score vector."""
    if decile < 1:
        raise ValueError("decile must be >= 1")
    pct = scores.groupby(dates).rank(pct=True, method="first")
    return (pct > 1.0 - 1.0 / decile).to_numpy()


def format_stats(st: SelectionStats, label: str = "model") -> str:
    lift = st.lifts()
    return (
        f"{label:36s} A {st.precision_a:6.1%} (base {st.base_precision_a:5.1%}, "
        f"lift {lift['precision_a_lift']:+.2%})   B {st.precision_b:6.1%} "
        f"(base {st.base_precision_b:5.1%}, lift {lift['precision_b_lift']:+.2%})   "
        f"net {st.avg_net:+.3%} (base {st.base_avg_net:+.3%}, lift {lift['net_lift']:+.3%})"
        + (f"   AUC {st.auc:.4f}" if np.isfinite(st.auc) else "")
    )


DEFINITIONS_NOTE = (
    "precision definitions: A = P(label==1), the barrier trade | B = P(ret_net>0),\n"
    "the SAME barrier trade scored on money after friction. A and B are always\n"
    "reported together and are NOT comparable to definition C = P(fwd_ret>friction),\n"
    "a fixed-hold trade that signal_check.py prints separately. Base rates are\n"
    "per-block, so read each precision against the base printed beside it."
)
