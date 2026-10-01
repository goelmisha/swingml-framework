"""Feature-provider registry.

Providers are resolved by name at runtime so no part of the pipeline hard-codes
a feature set. Resolution order:

1. built-in providers (:data:`BUILTIN_PROVIDERS`)
2. third-party entry points in the ``swingml.features`` group
3. an explicit ``"module:Class"`` dotted path

Option 3 is the escape hatch for a feature set you do not want to publish: keep
the provider in your own package and point ``features.provider`` at it. The rest
of the pipeline -- labelling, purged validation, CPCV, PBO, evaluation -- stays
exactly the same and stays leak-free.

Example
-------
    features:
      provider: "my_private_pkg.signals:MyFeatureProvider"

or, for an installed distribution declaring::

    [project.entry-points."swingml.features"]
    mine = "my_private_pkg.signals:MyFeatureProvider"

    features:
      provider: mine
"""

from __future__ import annotations

import importlib
import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from swingml.features.base import FeatureProvider
from swingml.features.demo import DemoFeatureProvider

if TYPE_CHECKING:  # pragma: no cover - typing only
    from swingml.config import FeatureConfig

logger = logging.getLogger(__name__)

#: Entry-point group third-party providers register under.
ENTRY_POINT_GROUP = "swingml.features"

#: Providers shipped in-tree.
BUILTIN_PROVIDERS: dict[str, type[FeatureProvider]] = {
    DemoFeatureProvider.name: DemoFeatureProvider,
}


def register_provider(cls: type[FeatureProvider], *, name: str | None = None) -> str:
    """Register ``cls`` in-process. Returns the name it was registered under."""
    key = name or cls.name
    if not key or key == "unnamed":
        raise ValueError("a provider must set a non-empty `name` or be registered with one")
    BUILTIN_PROVIDERS[key] = cls
    return key


def _from_entry_points() -> dict[str, type[FeatureProvider]]:
    """Discover installed providers, tolerating a broken third-party plugin."""
    found: dict[str, type[FeatureProvider]] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            found[ep.name] = ep.load()
        except Exception as exc:  # a broken plugin must not break the pipeline
            logger.warning("ignoring feature provider %r (%s): %s", ep.name, ep.value, exc)
    return found


def _from_dotted_path(spec: str) -> type[FeatureProvider]:
    module_path, _, cls_name = spec.partition(":")
    if not cls_name:
        raise ValueError(
            f"dotted provider spec must look like 'module:Class', got {spec!r}"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise LookupError(f"cannot import feature provider module {module_path!r}: {exc}") from exc
    try:
        cls = getattr(module, cls_name)
    except AttributeError as exc:
        raise LookupError(f"{module_path!r} has no attribute {cls_name!r}") from exc
    if not (isinstance(cls, type) and issubclass(cls, FeatureProvider)):
        raise TypeError(f"{spec} is not a FeatureProvider subclass")
    return cls


def available_providers() -> dict[str, str]:
    """Known providers mapped to a human-readable origin, for errors and listing."""
    out = {name: "built-in" for name in BUILTIN_PROVIDERS}
    for name, cls in _from_entry_points().items():
        out.setdefault(name, f"entry point -> {cls.__module__}.{cls.__name__}")
    return out


def resolve_feature_provider(spec: str) -> type[FeatureProvider]:
    """Resolve ``spec`` to a provider class.

    ``spec`` is a built-in/entry-point name, or ``"module:Class"`` for a provider
    that is not installed as an entry point.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("features.provider must be a non-empty string")
    spec = spec.strip()

    if ":" in spec:
        cls = _from_dotted_path(spec)
        logger.info("feature provider %s -> %s.%s", spec, cls.__module__, cls.__name__)
        return cls

    if spec in BUILTIN_PROVIDERS:
        return BUILTIN_PROVIDERS[spec]

    discovered = _from_entry_points()
    if spec in discovered:
        return discovered[spec]

    known = ", ".join(sorted({*BUILTIN_PROVIDERS, *discovered})) or "<none>"
    # A dotted module path is a common typo for the 'module:Class' form; say so
    # explicitly rather than reporting it as an unknown name.
    hint = ""
    if "." in spec:
        hint = (
            " This looks like a module path: use the 'module:Class' form with a "
            f"colon, e.g. {spec}:MyFeatureProvider."
        )
    raise LookupError(
        f"unknown feature provider {spec!r}. Known providers: {known}. "
        "For a provider outside the installed distributions, use a "
        "'module:Class' dotted path." + hint
    )


def make_feature_provider(cfg: "FeatureConfig") -> FeatureProvider:
    """Instantiate the provider named by ``cfg.provider``."""
    return resolve_feature_provider(cfg.provider)(cfg)


__all__ = [
    "BUILTIN_PROVIDERS",
    "ENTRY_POINT_GROUP",
    "available_providers",
    "make_feature_provider",
    "register_provider",
    "resolve_feature_provider",
]
