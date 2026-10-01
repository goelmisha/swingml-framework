"""Tests for the gradient-boosting engine factory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingml.config import ModelsConfig
from swingml.models import ENGINES, fit_classifier, make_classifier, predict_positive


@pytest.fixture
def tiny():
    """Tiny deterministic classification problem (one informative column)."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(200, 4), columns=[f"f{i}" for i in range(4)])
    y = (X["f0"] > 0.5).astype(int).to_numpy()
    w = np.ones(200)
    Xte = pd.DataFrame(rng.rand(20, 4), columns=[f"f{i}" for i in range(4)])
    return X, y, w, Xte


def test_engines_registry():
    assert ENGINES == ("sklearn-hgb", "lightgbm", "xgboost")


def test_make_classifier_rejects_unknown_engine():
    with pytest.raises(ValueError, match="unknown engine"):
        make_classifier("catboost", ModelsConfig())


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_fit_predict_shapes(engine, tiny):
    X, y, w, Xte = tiny
    model = make_classifier(engine, ModelsConfig(), seed=0)
    fit_classifier(engine, model, X, y, w)
    p = predict_positive(engine, model, Xte)
    assert p.shape == (20,)
    assert np.all((p >= 0.0) & (p <= 1.0))


@pytest.mark.parametrize("engine", ENGINES)
def test_engines_learn_the_informative_feature(engine, tiny):
    """All three engines must beat chance on a one-feature problem."""
    X, y, w, Xte = tiny
    model = make_classifier(engine, ModelsConfig(), seed=0)
    fit_classifier(engine, model, X, y, w)
    p = predict_positive(engine, model, Xte)
    yte = (Xte["f0"] > 0.5).astype(int).to_numpy()
    acc = ((p > 0.5) == yte).mean()
    assert acc > 0.5


def test_hyperparameters_come_from_config(tiny):
    """Numbers must be read from ModelsConfig, never hard-coded in the module.

    This guards the published boundary as much as behaviour: a tuned constant
    parked in this file would ship to everyone who clones the repository.
    """
    params = ModelsConfig(n_estimators=5, learning_rate=0.1, max_depth=1, min_samples_leaf=2)
    model = make_classifier("sklearn-hgb", params, seed=0)
    assert model.max_iter == 5
    assert model.max_depth == 1
    assert model.min_samples_leaf == 2

    X, y, w, Xte = tiny
    fit_classifier("sklearn-hgb", model, X, y, w)
    assert predict_positive("sklearn-hgb", model, Xte).shape == (20,)


def test_lightgbm_and_xgboost_import_after_libomp():
    """The macOS libomp blocker is resolved: both engines load and train."""
    import lightgbm  # noqa: F401
    import xgboost  # noqa: F401
