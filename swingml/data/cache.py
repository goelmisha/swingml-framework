"""Parquet-backed disk cache.

NSE archives are rate-limited and occasionally flaky, so every raw download is
persisted and reused. Downloads are cached per (symbol, date); a rerun costs
zero network calls.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


class DiskCache:
    """Namespaced parquet cache rooted at ``base_dir``."""

    def __init__(self, base_dir: str | Path, namespace: str) -> None:
        self.base_dir = Path(base_dir)
        self.namespace = namespace

    # -- paths -------------------------------------------------------------
    def path_for(self, key: str) -> Path:
        """Stable, filesystem-safe path for a cache key."""
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(key))
        # Long/hashed keys still collide-free: prefix a short digest.
        digest = hashlib.sha1(str(key).encode()).hexdigest()[:8]
        return self.base_dir / self.namespace / f"{safe[:120]}.{digest}.parquet"

    def exists(self, key: str) -> bool:
        return self.path_for(key).exists()

    # -- io ----------------------------------------------------------------
    def read(self, key: str) -> pd.DataFrame | None:
        p = self.path_for(key)
        if not p.exists():
            return None
        try:
            return pd.read_parquet(p)
        except Exception as exc:  # corrupt/partial write -> treat as a miss
            logger.warning("cache read failed for %s (%s); refetching", key, exc)
            p.unlink(missing_ok=True)
            return None

    def write(self, key: str, df: pd.DataFrame) -> None:
        p = self.path_for(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish write: write then rename to avoid half-written parquet.
        tmp = p.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=True)
        tmp.replace(p)

    def get_or_build(
        self,
        key: str,
        builder: Callable[[], pd.DataFrame | None],
        force: bool = False,
        allow_empty: bool = False,
    ) -> pd.DataFrame | None:
        """Return the cached frame, else build it, else persist and return it.

        ``None`` results (e.g. an NSE 404 on a trading holiday) are NOT cached,
        so transient failures can be retried on the next run.
        """
        if not force:
            cached = self.read(key)
            if cached is not None:
                return cached
        built = builder()
        if built is None:
            return None
        if built.empty and not allow_empty:
            return built
        self.write(key, built)
        return built

    def clear(self) -> int:
        """Delete every file in this namespace. Returns the count removed."""
        d = self.base_dir / self.namespace
        if not d.exists():
            return 0
        n = 0
        for p in d.glob("*.parquet"):
            p.unlink()
            n += 1
        return n
