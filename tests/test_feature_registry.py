"""Feature-provider registry and the public reference provider.

The registry is what lets the framework stay agnostic about which feature set is
in use, so its failure modes matter: an unknown name must say what *is* known,
and a provider that breaks the contract must fail loudly rather than silently
producing a matrix the labeller cannot use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingml.config import FeatureConfig
from swingml.features import (
    DemoFeatureProvider,
    FeatureProvider,
    available_providers,
    make_feature_provider,
    register_provider,
    resolve_feature_provider,
)
from swingml.features.base import REQUIRED_CONTEXT_COLUMNS


class _DummyProvider(FeatureProvider):
    name = "dummy"

    def transform(self, prices, delivery, bench=None):  # pragma: no cover - unused
        return pd.DataFrame(columns=["date", "symbol", *REQUIRED_CONTEXT_COLUMNS])


class _NotAProvider:
    """Deliberately not a FeatureProvider."""


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def test_builtin_demo_provider_resolves():
    assert resolve_feature_provider("demo") is DemoFeatureProvider


def test_make_feature_provider_honours_config_field():
    assert isinstance(make_feature_provider(FeatureConfig(provider="demo")), DemoFeatureProvider)


def test_available_providers_lists_demo():
    assert "demo" in available_providers()


def test_unknown_provider_error_names_the_alternatives():
    with pytest.raises(LookupError) as exc:
        resolve_feature_provider("does-not-exist")
    message = str(exc.value)
    assert "does-not-exist" in message
    assert "demo" in message, "the error must tell the user what is available"


def test_dotted_path_resolution():
    resolved = resolve_feature_provider("tests.test_feature_registry:_DummyProvider")
    # Compare by name rather than identity: import order decides whether this is
    # the same module object, but it is always the same class.
    assert issubclass(resolved, FeatureProvider)
    assert resolved.__name__ == "_DummyProvider"


def test_dotted_path_must_be_a_provider_subclass():
    with pytest.raises(TypeError):
        resolve_feature_provider("tests.test_feature_registry:_NotAProvider")


def test_dotted_path_without_a_colon_explains_the_typo():
    """A dotted path missing its colon must not read as a plain unknown name."""
    with pytest.raises(LookupError) as exc:
        resolve_feature_provider("tests.test_feature_registry._DummyProvider")
    message = str(exc.value)
    assert "module:Class" in message
    assert "_DummyProvider" in message, "the hint should show the corrected form"


def test_missing_module_raises_lookup_error():
    with pytest.raises(LookupError):
        resolve_feature_provider("no_such_module_here:Provider")


def test_register_provider_by_alias():
    registered = register_provider(_DummyProvider, name="dummy-alias")
    assert resolve_feature_provider(registered) is _DummyProvider


def test_empty_spec_rejected():
    with pytest.raises(ValueError):
        resolve_feature_provider("")


# ---------------------------------------------------------------------------
# The reference provider's contract
# ---------------------------------------------------------------------------
def test_demo_provider_emits_contract_columns(synth, feature_cfg):
    prices, panel, bench = synth
    provider = DemoFeatureProvider(feature_cfg)
    out = provider.transform(prices, panel, bench)

    assert {"date", "symbol"} <= set(out.columns)
    for col in REQUIRED_CONTEXT_COLUMNS:
        assert col in out.columns, f"missing required context column {col}"

    # Context columns are carried through but must not be model inputs.
    assert not set(provider.feature_columns) & set(provider.context_columns)
    assert "open" not in provider.feature_columns and "atr_20" not in provider.feature_columns
    assert provider.feature_columns, "provider produced no features"
    # Cross-sectional ranks are appended last.
    assert any(c.startswith("xs_rank_") for c in provider.feature_columns)


def test_every_demo_feature_belongs_to_a_block(synth, feature_cfg):
    """Block-level pruning is only safe if no feature is orphaned."""
    prices, panel, bench = synth
    provider = DemoFeatureProvider(feature_cfg)
    provider.transform(prices, panel, bench)
    orphans = [c for c in provider.feature_columns if provider.group_of(c) == "other"]
    assert not orphans, f"features not assigned to any group: {orphans}"


def test_group_of_ignores_the_rank_prefix():
    provider = DemoFeatureProvider(FeatureConfig())
    assert provider.group_of("xs_rank_roc_20") == "trend_momentum"
    assert provider.group_of("roc_20") == "trend_momentum"
    assert provider.group_of("not_a_feature") == "other"


def test_check_output_rejects_missing_context():
    class Bad(FeatureProvider):
        name = "bad"

        def transform(self, prices, delivery, bench=None):  # pragma: no cover - unused
            return pd.DataFrame(columns=["date", "symbol"])

    provider = Bad(FeatureConfig())
    with pytest.raises(ValueError, match="context column"):
        provider.check_output(pd.DataFrame(columns=["date", "symbol"]))


def test_check_output_rejects_empty_feature_set():
    provider = DemoFeatureProvider(FeatureConfig())
    frame = pd.DataFrame({c: [1.0] for c in REQUIRED_CONTEXT_COLUMNS})
    with pytest.raises(ValueError, match="no feature columns"):
        provider.check_output(frame)


def test_demo_provider_is_causal(synth, feature_cfg):
    """The public reference provider must obey the same no-look-ahead rule."""
    prices, panel, bench = synth

    full = DemoFeatureProvider(feature_cfg).transform(prices, panel, bench)

    all_dates = sorted(full["date"].unique())
    cut = pd.Timestamp(all_dates[int(len(all_dates) * 0.6)])

    px_t = {s: df.loc[df.index <= cut] for s, df in prices.items()}
    panel_t = panel.loc[panel["date"] <= cut].copy()
    bench_t = bench.loc[bench.index <= cut]
    trunc = DemoFeatureProvider(feature_cfg).transform(px_t, panel_t, bench_t)

    common = [c for c in full.columns if c in trunc.columns]
    a = full.loc[full["date"] <= cut, common].sort_values(["date", "symbol"]).reset_index(drop=True)
    b = trunc.loc[trunc["date"] <= cut, common].sort_values(["date", "symbol"]).reset_index(drop=True)

    assert len(a) == len(b), f"row count diverged under truncation: {len(a)} vs {len(b)}"
    assert list(a["symbol"]) == list(b["symbol"])

    numeric = [c for c in common if c not in ("date", "symbol") and pd.api.types.is_numeric_dtype(a[c])]
    for col in numeric:
        av, bv = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
        assert np.array_equal(np.isnan(av), np.isnan(bv)), f"{col}: NaN mask changed under truncation"
        both = ~np.isnan(av)
        if both.any():
            np.testing.assert_allclose(
                av[both], bv[both], rtol=1e-9, atol=1e-12,
                err_msg=f"{col} changed when future data was removed -> look-ahead leak",
            )
