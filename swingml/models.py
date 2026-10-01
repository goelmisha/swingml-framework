"""Gradient-boosting engine factory -- one hyperparameter set, three engines.

Every experiment must be able to swap engine with a flag, because a conclusion
that holds for only one implementation is not a conclusion about the data. The
three engines are configured to be as close as their APIs allow: identical
boosting rounds, learning rate, depth, leaf floor and L2 regularisation, no
early stopping, and a fixed seed.

They are the same *family* (histogram GBDT) but NOT identical implementations:
LightGBM adds GOSS/EFB sampling and XGBoost its own histogram refinements, so
scores come out close rather than equal. Measure how close before drawing any
conclusion from the last decimal of AUC.

The hyperparameters are read from :class:`swingml.config.ModelsConfig`, so they
live in YAML alongside every other number that affects a score, and nothing here
hard-codes a value a reader could mistake for a recommendation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from swingml.config import ModelsConfig

logger = logging.getLogger(__name__)

ENGINES = ("sklearn-hgb", "lightgbm", "xgboost")


def make_classifier(engine: str, params: ModelsConfig, seed: int = 0):
    """Instantiate the requested engine with one shared hyperparameter set."""
    if engine == "sklearn-hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=params.n_estimators,
            learning_rate=params.learning_rate,
            max_depth=params.max_depth,
            min_samples_leaf=params.min_samples_leaf,
            l2_regularization=params.l2,
            early_stopping=False,
            random_state=seed,
        )
    if engine == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=params.n_estimators,
            learning_rate=params.learning_rate,
            max_depth=params.max_depth,
            min_child_samples=params.min_samples_leaf,
            reg_lambda=params.l2,
            random_state=seed,
            verbosity=-1,
            n_jobs=-1,
        )
    if engine == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=params.n_estimators,
            learning_rate=params.learning_rate,
            max_depth=params.max_depth,
            min_child_weight=params.min_samples_leaf,
            reg_lambda=params.l2,
            random_state=seed,
            tree_method="hist",
            verbosity=0,
            n_jobs=-1,
        )
    raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")


def fit_predict(engine: str, Xtr, ytr, wtr, Xte, params: ModelsConfig, seed: int = 0) -> np.ndarray:
    """One model, one engine, one protocol: fit on train, score P(y=1) on test.

    Every experiment must instantiate, fit and predict the same way, otherwise
    an engine comparison measures the calling code rather than the engine.
    """
    model = make_classifier(engine, params, seed=seed)
    fit_classifier(engine, model, Xtr, ytr, wtr)
    return predict_positive(engine, model, Xte)


def fit_classifier(engine: str, model, X: pd.DataFrame, y: np.ndarray, w: np.ndarray):
    """Fit with sample weights; the weight semantics are identical everywhere.

    The weights are the label-uniqueness values, so overlapping labels stop
    counting as independent evidence.
    """
    if engine == "sklearn-hgb":
        model.fit(X, y, sample_weight=w)
    elif engine == "lightgbm":
        model.fit(X, y, sample_weight=w)
    elif engine == "xgboost":
        model.fit(X, y, sample_weight=w)
    else:  # pragma: no cover - make_classifier validates first
        raise ValueError(f"unknown engine {engine!r}")
    return model


def predict_positive(engine: str, model, X: pd.DataFrame) -> np.ndarray:
    """P(y == 1) for every row, from whichever engine was fitted."""
    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", None)
    if classes is None:  # pragma: no cover - all three expose classes_
        return proba[:, 1]
    pos = int(np.flatnonzero(classes == 1)[0])
    return proba[:, pos]
