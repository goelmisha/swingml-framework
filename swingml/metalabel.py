"""Meta-labelling -- a primary selects, a secondary filters.

The problem this fixes
----------------------
The single-model pipeline already ranks names well enough to beat the base rate,
but its top decile is still only ~52-53% right under ``fixed_hold``, so roughly
half the names it buys lose money net. That is a **false-positive** problem, and
it is what meta-labelling is designed for (Lopez de Prado, AFML ch. 3):

1. a **primary** model (or rule) finds candidates at a *recall-first* rate --
   ``primary_frac`` of the cross-section, deliberately wider than what is traded;
2. a **secondary** model learns one narrow thing -- *given the primary's call,
   was it right?* -- on the trades the primary actually took;
3. the secondary's probability filters (and later sizes) the primary's bets.

The ML only decides size, never side, which is why the architecture limits
overfitting.

Two correctness details this implementation is strict about
-----------------------------------------------------------
**The meta-label is money, not barrier touch.** ``meta_target`` is
``ret_net > 0`` (definition B). Labelling a bet by "did it reach the profit
barrier first" would mark a trade that ran to the vertical barrier *profitably*
as a failure -- and would train the
filter on something other than the P&L it is used to size.

**The secondary never sees in-sample primary scores.** The primary's ranking of
its own training rows is optimistic, so a secondary trained on those predictions
would learn from overfit signal. The primary's scores on the training block are
therefore generated **out-of-fold** with combinatorial purged CV (each session
lands in test exactly once), which is the same purge/embargo discipline the
walk-forward folds use.

Selection stays matched
-----------------------
The final score is a **composite**: every primary-pool row ranks above every
non-pool row, and inside the pool the secondary's probability orders the names.
Selection is then the ordinary within-date top 1/decile (``top_fraction_mask``),
so the number of names traded is unchanged -- the gate is "filtered precision at
matched selection rate", not "trade less and look better".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from swingml.config import MetaConfig, ModelsConfig, ValidationConfig
from swingml.models import fit_classifier, make_classifier, predict_positive
from swingml.validation import cpcv_splits

logger = logging.getLogger(__name__)

#: Column the secondary sees in addition to the ordinary feature set: what the
#: primary believed about this row. Named so it cannot collide with a feature.
PRIMARY_SCORE_COL = "__primary_score__"


@dataclass
class MetaResult:
    """Scores from one meta-labelled fit.

    ``final_scores`` is the composite used for selection and is finite for every
    row (pool rows first, then the rest), so it can be handed straight to
    ``evaluation.top_fraction_mask`` / ``paper.select_top``.
    """

    final_scores: np.ndarray
    primary_scores: np.ndarray
    n_train_pool: int = 0
    fallback: bool = False


def meta_target(df: pd.DataFrame) -> np.ndarray:
    """The meta-label: did the primary's bet make money **net of friction**?

    Definition B. Deliberately NOT "did the profit barrier hit first" -- a
    profitable run to the vertical barrier is a win here and a 0 under that
    reading.
    """
    if "ret_net" not in df.columns:
        raise ValueError("meta_target needs the labels' ret_net column (definition B)")
    return (df["ret_net"].to_numpy(dtype=float) > 0).astype(int)


def _top_fraction(scores: np.ndarray, dates, frac: float) -> np.ndarray:
    """Within-date boolean mask for the top ``frac`` of rows by score.

    Mirrors :func:`swingml.evaluation.top_fraction_mask` but takes an explicit
    fraction, because the primary's pool is a recall-first rate rather than the
    1/decile trading selection.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError("frac must be in (0, 1)")
    pct = pd.Series(np.asarray(scores, dtype=float)).groupby(np.asarray(dates)).rank(
        pct=True, method="first"
    )
    return (pct > 1.0 - frac).to_numpy()


def primary_oof_scores(
    engine: str,
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    params: ModelsConfig,
    validation_cfg: ValidationConfig,
    horizon: int,
    n_groups: int,
    seed: int = 0,
) -> np.ndarray:
    """Out-of-fold primary scores for the whole training block.

    Combinatorial purged CV with one test group per path places every session in
    test exactly once, so the union of the paths' test predictions covers the
    training block with no in-sample leakage. Rows whose path could not be fitted
    (single-class train block) stay NaN and are excluded from the pool.
    """
    oof = np.full(len(train), np.nan)
    paths = cpcv_splits(train, validation_cfg, horizon, n_groups=n_groups, n_test_groups=1)
    n_fit = 0
    for path in paths:
        tr = train.iloc[path.train_idx]
        te = train.iloc[path.test_idx]
        if len(tr) == 0 or len(te) == 0 or tr[target_col].nunique() < 2:
            continue
        model = make_classifier(engine, params, seed=seed)
        fit_classifier(
            engine, model, tr[feature_cols],
            tr[target_col].to_numpy(), tr["uniqueness"].to_numpy(),
        )
        oof[path.test_idx] = predict_positive(engine, model, te[feature_cols])
        n_fit += 1
    logger.info("primary OOF: %d/%d CPCV paths fitted over %d rows", n_fit, len(paths), len(train))
    return oof


def fit_meta_predict(
    engine: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    *,
    meta_cfg: MetaConfig,
    params: ModelsConfig,
    validation_cfg: ValidationConfig,
    horizon: int,
    decile: int = 10,
    target_col: str = "target",
    seed: int = 0,
) -> MetaResult:
    """Primary (recall-first) -> secondary (filter) -> composite score vector.

    Parameters
    ----------
    train, test
        Row blocks already carrying ``target``, ``ret_net``, ``uniqueness`` and
        the feature columns. ``train`` must be purged from ``test`` by the
        caller (the walk-forward fold boundary does this).
    meta_cfg
        ``primary_frac`` and the OOF group count.
    decile
        The trading selection fraction (1/decile). ``primary_frac`` must be
        strictly wider, or the secondary has nothing to filter.
    """
    if not 0.0 < meta_cfg.primary_frac < 1.0:
        raise ValueError("meta.primary_frac must be in (0, 1)")
    if meta_cfg.primary_frac <= 1.0 / decile:
        raise ValueError(
            f"meta.primary_frac ({meta_cfg.primary_frac}) must exceed the trading "
            f"selection 1/decile ({1.0 / decile:.3f}); otherwise the secondary "
            "cannot filter anything"
        )
    missing = [c for c in feature_cols if c not in train.columns or c not in test.columns]
    if missing:
        raise ValueError(f"feature columns missing from the meta blocks: {missing[:5]}")

    oof = primary_oof_scores(
        engine, train, feature_cols, target_col, params, validation_cfg,
        horizon, meta_cfg.oof_groups, seed=seed,
    )
    finite = np.isfinite(oof)
    if finite.sum() == 0:
        raise RuntimeError(
            "no out-of-fold primary scores: the training block is too short for "
            "CPCV OOF. Widen the history or reduce meta.oof_groups."
        )
    pool = _top_fraction(oof, train["date"].to_numpy(), meta_cfg.primary_frac) & finite
    pool_rows = train.loc[pool]

    # Primary fitted on the whole training block; this is what ranks the test date.
    primary = make_classifier(engine, params, seed=seed)
    fit_classifier(
        engine, primary, train[feature_cols],
        train[target_col].to_numpy(), train["uniqueness"].to_numpy(),
    )
    primary_scores = predict_positive(engine, primary, test[feature_cols])
    primary_pool = _top_fraction(primary_scores, test["date"].to_numpy(), meta_cfg.primary_frac)

    y_meta = meta_target(pool_rows)
    fallback = False
    if y_meta.min() == y_meta.max():
        logger.warning(
            "meta pool has a single class (%d rows, label %d); falling back to the "
            "primary ranking for this block", len(pool_rows), int(y_meta.min()),
        )
        meta_scores = primary_scores
        fallback = True
    else:
        Xtr = pool_rows[feature_cols].copy()
        Xtr[PRIMARY_SCORE_COL] = oof[pool]
        secondary = make_classifier(engine, params, seed=seed)
        fit_classifier(engine, secondary, Xtr, y_meta, pool_rows["uniqueness"].to_numpy())
        Xte = test[feature_cols].copy()
        Xte[PRIMARY_SCORE_COL] = primary_scores
        meta_scores = predict_positive(engine, secondary, Xte)

    # Composite: pool beats non-pool, secondary probability orders the pool.
    final = np.where(primary_pool, meta_scores, meta_scores - 1.0)
    return MetaResult(
        final_scores=final,
        primary_scores=primary_scores,
        n_train_pool=int(pool.sum()),
        fallback=fallback,
    )


__all__ = [
    "PRIMARY_SCORE_COL",
    "MetaResult",
    "fit_meta_predict",
    "meta_target",
    "primary_oof_scores",
]
