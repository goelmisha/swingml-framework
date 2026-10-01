"""Feature layer: provider contract, registry and reusable primitives.

Nothing outside this package needs to know which feature set is in use. Resolve
one via :func:`swingml.features.make_feature_provider` and the rest of the
pipeline is unchanged.
"""

from swingml.features.base import REQUIRED_CONTEXT_COLUMNS, FeatureProvider
from swingml.features.demo import DemoFeatureProvider
from swingml.features.registry import (
    BUILTIN_PROVIDERS,
    ENTRY_POINT_GROUP,
    available_providers,
    make_feature_provider,
    register_provider,
    resolve_feature_provider,
)

__all__ = [
    "BUILTIN_PROVIDERS",
    "ENTRY_POINT_GROUP",
    "REQUIRED_CONTEXT_COLUMNS",
    "DemoFeatureProvider",
    "FeatureProvider",
    "available_providers",
    "make_feature_provider",
    "register_provider",
    "resolve_feature_provider",
]
