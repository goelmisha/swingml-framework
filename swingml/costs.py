"""Size-aware trading costs -- what the flat 0.25% round trip leaves out.

Why this module exists
----------------------
Every headline number in this repo charges a flat ``costs.round_trip_cost_pct``
(0.25%). That is a *toll*, not a cost model: it does not care whether the name
trades Rs 2 crore or Rs 500 crore a day, or whether the book is Rs 1 crore or
Rs 100 crore. The strategy selects the top decile of a point-in-time liquidity
universe, which tilts it toward the *less* liquid end by construction, so the
flat toll is optimistic exactly where it matters.

What it models
--------------
Participation in each name's average daily traded value, and a square-root
impact law on it -- the standard empirical shape (impact grows with the square
root of participation, not linearly):

    participation_i = position_value / ADV_i
    impact_per_side = k * sqrt(participation_i)
    round_trip_cost = flat + 2 * impact_per_side

The factor of two is because a round trip pays impact on both legs. ``k`` is a
free parameter with no measurement behind it here, so this module is used in a
**sensitivity panel**, never as a single number to quote: book sizes of 1/10/100
crore against k of 0.05/0.10/0.20 spans "small retail book in liquid names" to
"aggressive institutional book", and the conclusion has to hold across the grid
rather than at one hand-picked cell.

What it still does not model: partial fills, borrow, corporate-action timing,
gapping, the difference between the open and the decision price, and the fact
that ADV itself shrinks when everyone else is selling. It is a floor, and it is
closer to reality than a constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: One crore = 100 lakh; the feature matrix carries turnover in lakhs.
LAKHS_PER_CRORE = 100.0


@dataclass(frozen=True)
class ImpactModel:
    """Book size and impact shape for the size-aware cost."""

    book_size_cr: float = 10.0
    #: Impact coefficient per side, applied to sqrt(participation).
    impact_k: float = 0.10
    #: Column of average daily traded value, in lakhs.
    adv_col: str = "adv_20"

    def __post_init__(self) -> None:
        if self.book_size_cr <= 0:
            raise ValueError("book_size_cr must be > 0")
        if self.impact_k < 0:
            raise ValueError("impact_k must be >= 0")


def positions_by_date(block: pd.DataFrame, decile: int) -> pd.Series:
    """How many names a decile rule holds per session.

    Mirrors :func:`swingml.evaluation.top_fraction_mask`: the top ``1/decile``
    slice, floor, which is the same count the selection rule itself uses.
    """
    if decile < 1:
        raise ValueError("decile must be >= 1")
    counts = block.groupby("date")["symbol"].size() if "symbol" in block.columns \
        else block.groupby("date").size()
    return np.maximum(1, np.floor(counts / decile).astype(int))


def impact_cost_per_trade(
    block: pd.DataFrame,
    positions: pd.Series,
    model: ImpactModel,
) -> np.ndarray:
    """Round-trip impact cost for each row, as a fraction of position value.

    ``positions`` is the number of names held per session (see
    :func:`positions_by_date`) -- it differs between the strategy and the
    benchmark, and that difference is the point: selecting a tenth of the
    universe concentrates ten times the capital into each name and therefore
    pays roughly sqrt(10) more impact on it.
    """
    if model.adv_col not in block.columns:
        raise ValueError(f"{model.adv_col!r} not in the block; cannot size the cost")

    adv_lakhs = block[model.adv_col].astype(float)
    # Warm-up rows have no ADV; fall back to that session's median rather than
    # dropping the row (which would quietly shrink the selection) or zeroing the
    # cost (which would quietly flatter it).
    if adv_lakhs.isna().any():
        adv_lakhs = adv_lakhs.fillna(
            adv_lakhs.groupby(block["date"]).transform("median")
        )
    adv_cr = adv_lakhs.to_numpy(dtype=float) / LAKHS_PER_CRORE

    n = block["date"].map(positions).to_numpy(dtype=float)
    position_cr = model.book_size_cr / np.maximum(n, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        participation = np.where(adv_cr > 0, position_cr / adv_cr, np.nan)

    cost = 2.0 * model.impact_k * np.sqrt(np.clip(participation, 0.0, None))
    return np.nan_to_num(cost, nan=0.0)


def describe(model: ImpactModel) -> str:
    """One-line description for experiment output."""
    return (f"book Rs {model.book_size_cr:g} crore, k={model.impact_k:g}, "
            f"round trip = flat + 2*k*sqrt(position/ADV), ADV = {model.adv_col}")
