"""Corporate-action handling for raw quantity series.

A 1:2 split doubles the reported share count overnight. Any trailing mean of a
raw quantity that straddles the ex-date is therefore wrong for a full lookback,
and a model trained on it learns the split rather than the market.

The repair here has two halves:

1. **Detection** from the exchange's own restated previous close, which is
   published with every session and is an unambiguous signal -- on real
   bhavcopy history the overwhelming majority of sessions sit at exactly 1.0.
2. **Rescaling** pre-action quantities onto today's basis, rather than nulling
   the affected windows. Nulling is the obvious alternative and is much worse:
   it punches holes in the history, and because quantity features are trailing
   ratios, those holes propagate across every window that touches them.

Detection deliberately avoids volume-spike heuristics. Real liquid names spike
6x on results and block deals all the time, so a spike-based detector fires
constantly and destroys the volume features rather than cleaning them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_corporate_actions(
    prev_close: pd.Series,
    prior_close: pd.Series,
    tolerance: float = 0.10,
) -> tuple[pd.Series, pd.Series]:
    """Detect splits/bonuses from the bhavcopy's own ``PREV_CLOSE`` restatement.

    The exchange restates ``PREV_CLOSE`` on an ex-date to reflect the corporate
    action, so ``PREV_CLOSE_t / CLOSE_{t-1}`` is ~1.0 on a normal session and the
    split ratio on an ex-date.

    Returns
    -------
    (is_action, ratio)
        ``is_action`` marks ex-dates; ``ratio`` is ``prev_close/prior_close``
        (< 1 for a split or bonus issue).
    """
    ratio = prev_close / prior_close.replace(0.0, np.nan)
    is_action = ((ratio - 1.0).abs() > tolerance).fillna(False)
    return is_action, ratio


def adjust_quantities_for_corporate_actions(
    qty: pd.Series,
    ratio: pd.Series,
    is_action: pd.Series,
) -> pd.Series:
    """Rescale share counts onto a single, split-consistent basis.

    Pre-action quantities are multiplied by ``1/ratio`` so the whole history
    shares today's basis.

    Causality is preserved: quantity features are all ratios against a trailing
    mean, and every window member carries the same cumulative factor, so it
    cancels. Only actions at or before the observation date can influence the
    *relative* scaling inside that window.
    """
    mult = pd.Series(1.0, index=qty.index, dtype=float)
    valid = is_action.fillna(False) & ratio.notna() & np.isfinite(ratio) & (ratio > 0)
    mult[valid] = 1.0 / ratio[valid]
    # factor[i] = product of multipliers for actions strictly AFTER day i.
    inclusive = mult[::-1].cumprod()[::-1]
    factor = inclusive / mult
    return (qty.astype(float) * factor).rename(qty.name)


def detect_quantity_jumps(ttl_qty: pd.Series, threshold: float = 6.0, window: int = 45) -> pd.Series:
    """Data-quality diagnostic: sessions where traded quantity spikes vs its own median.

    Used only to warn. On real exchange data large spikes are usually legitimate
    (results, block deals, index inclusion), not splits -- so this must never
    gate feature construction.
    """
    med = ttl_qty.rolling(window, min_periods=5).median()
    ratio = ttl_qty / med.replace(0.0, np.nan)
    return (ratio > threshold) & med.notna()
