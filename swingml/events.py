"""Event sampling -- the symmetric CUSUM filter (Lopez de Prado, AFML ch. 2).

Why this module exists
----------------------
By default this pipeline labels **every bar**. That is fixed-interval sampling,
and it is why 258,906 ``fixed_hold`` rows carry only ~24,000 effective
independent samples: consecutive 10-session labels overlap almost completely.

A CUSUM filter asks a different question -- *when is a bet even considered?* --
and only emits a label when the cumulative signed log-return since the last
event exceeds a dynamic threshold ``h``::

    s_pos = max(0, s_pos + r_t)
    s_neg = min(0, s_neg + r_t)
    event when s_pos > h or s_neg < -h   (reset the triggered side)

with ``h = h_mult * EWM(span=vol_span).std(r_t)``, so a quiet name needs a
smaller move than a volatile one to register. Events land at meaningful moves
instead of uniformly in time, which raises the information per label and cuts
the overlap problem that caps effective sample size.

Two conventions, both deliberate
--------------------------------
**Only the triggered side resets.** Zeroing both accumulators on every event --
a common shorthand -- discards a live excursion on the other side and fires
fewer, lumpier events.

**Causality.** ``h`` at bar ``t`` is an EWMA of returns up to and including
``t``, and the accumulators only ever consume ``r_t``, so the mask at ``t`` is a
function of data up to ``t``. ``tests/test_events.py`` proves it by truncation.

The ``h_mult`` choice is a trial and belongs in
``data/trials.jsonl``; the config default is a starting value, not a tuning
result.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cusum_events(
    close: pd.Series,
    h_mult: float = 1.5,
    vol_span: int = 20,
) -> pd.Series:
    """Boolean mask: ``True`` at bars where a symmetric CUSUM event fires.

    Parameters
    ----------
    close
        Price series for **one** symbol, ordered oldest-first.
    h_mult
        Threshold multiple of the (EWM) dynamic volatility.
    vol_span
        EWM span for the volatility estimate, in bars.

    Returns
    -------
    A boolean ``Series`` aligned to ``close``. The first bar never fires (no
    return yet), and bars with a non-finite or non-positive threshold are
    skipped rather than treated as a zero move.
    """
    if h_mult <= 0:
        raise ValueError("h_mult must be > 0")
    if vol_span < 2:
        raise ValueError("vol_span must be >= 2")

    log_ret = np.log(close.astype(float)).diff()
    vol = log_ret.ewm(span=vol_span, adjust=False).std()

    mask = pd.Series(False, index=close.index, dtype=bool)
    s_pos = 0.0
    s_neg = 0.0
    for t in close.index:
        r = log_ret.at[t]
        v = vol.at[t]
        if not np.isfinite(r) or not np.isfinite(v) or v <= 0:
            continue
        h = h_mult * float(v)
        s_pos = max(0.0, s_pos + float(r))
        s_neg = min(0.0, s_neg + float(r))
        if s_pos > h:
            mask.at[t] = True
            s_pos = 0.0
        elif s_neg < -h:
            mask.at[t] = True
            s_neg = 0.0
    return mask


def event_mask(
    df: pd.DataFrame,
    h_mult: float = 1.5,
    vol_span: int = 20,
    date_col: str = "date",
    symbol_col: str = "symbol",
    price_col: str = "close",
) -> pd.Series:
    """Per-symbol CUSUM events across the whole panel.

    Events are counted **within each symbol**: two names having a big move on
    the same session are two independent events, not one, because their return
    paths are independent. Returns a boolean Series aligned to ``df``.
    """
    for col in (date_col, symbol_col, price_col):
        if col not in df.columns:
            raise ValueError(f"event_mask needs column {col!r}")

    out = pd.Series(False, index=df.index, dtype=bool)
    for _, g in df.groupby(symbol_col, observed=True, sort=False):
        g = g.sort_values(date_col)
        mask = cusum_events(g[price_col], h_mult=h_mult, vol_span=vol_span)
        out.loc[g.index] = mask.to_numpy()
    return out


__all__ = ["cusum_events", "event_mask"]
