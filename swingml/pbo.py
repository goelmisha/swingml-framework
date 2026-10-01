"""Probability of Backtest Overfitting via CSCV (Bailey, Borwein, Lopez de Prado
and Zhu, 2015).

The problem PBO answers
-----------------------
If you try N strategy configurations and pick the one with the best backtest,
the winner's backtest is an optimistically biased estimate of its true skill --
you selected the max of N noisy numbers. PBO measures how often that selection
procedure betrays you: pick the best configuration on one region of the data
(IN), then check whether it still beats the *median* configuration on the
complementary region (OUT). The fraction of regions where the "best" turns out
below-median out-of-sample is the PBO. 0.5 means the selection is a coin flip --
the backtest ranking carried NO out-of-sample information.

Method (CSCV -- Combinatorially Symmetric Cross-Validation)
-----------------------------------------------------------
1. Split the timeline into N chronological groups; form all C(N, k) paths whose
   test set is a k-group combination (:func:`swingml.validation.cpcv_splits`).
2. For each path p: IN = its test groups, OUT = the complement. Measure every
   configuration's performance on BOTH. For the ranks to be well formed the
   complement must itself be a path, which requires **k = N/2** -- enforce that
   in :func:`cscv_from_path_metrics`.
3. Per path: rank the configurations by IN performance; take the winner; find
   its rank among the OUT performances. l = that rank.
4. logit(l) = ln(l / (n + 1 - l)) is positive when l > (n + 1)/2, i.e. the
   in-sample winner finished below-median out-of-sample.
5. PBO = share of paths with logit(l) > 0.

A path where some configuration has non-finite performance (degenerate fold)
drops that configuration from the ranking on that path only.

Reference: Bailey et al. (2015), "The Probability of Backtest Overfitting".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PboResult:
    """Outcome of one CSCV run."""

    #: Share of paths where the in-sample winner finished below-median out.
    pbo: float
    #: logit(l) per path; negative = the IS winner beat the median OUT.
    logits: np.ndarray
    #: OUT-rank of the IS winner per path (1 = best).
    l_ranks: np.ndarray
    n_paths: int
    n_configs: int
    #: Per-path detail for debugging.
    detail: dict = field(default_factory=dict)

    def summary(self) -> str:
        pos = int((self.logits > 0).sum())
        return (
            f"PBO = {self.pbo:.1%}  ({pos}/{self.detail.get('n_valid_paths', self.n_paths)} "
            f"valid paths where the in-sample winner finished below-median "
            f"out-of-sample; {self.n_configs} configurations)"
        )


def probability_of_backtest_overfitting(
    perf_in: np.ndarray,
    perf_out: np.ndarray,
) -> PboResult:
    """Compute PBO from paired (n_paths x n_configs) performance matrices.

    Parameters
    ----------
    perf_in
        Row p = configuration performance on path p's IN region (its test
        groups). Higher is better.
    perf_out
        Row p = the SAME configurations on path p's OUT region (the
        complement of its test groups). Rows of the two matrices must
        correspond path-for-path; :func:`cscv_from_path_metrics` builds this
        pairing from raw path metrics.
    """
    a = np.asarray(perf_in, dtype=float)
    b = np.asarray(perf_out, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"perf_in {a.shape} and perf_out {b.shape} must have the same shape")
    if a.ndim != 2:
        raise ValueError("performance matrices must be 2-D (paths x configs)")
    n_paths, n_configs = a.shape
    if n_paths < 2 or n_configs < 2:
        raise ValueError("need at least 2 paths and 2 configurations")

    logits = np.full(n_paths, np.nan)
    l_ranks = np.full(n_paths, np.nan)

    for p in range(n_paths):
        finite = np.isfinite(a[p]) & np.isfinite(b[p])
        n_f = int(finite.sum())
        if n_f < 2:
            continue  # cannot rank a single configuration; path contributes nothing

        idx = np.flatnonzero(finite)
        # Winner on IN: highest metric; ties resolved by first column (deterministic).
        winner = idx[int(np.argmax(a[p][idx]))]
        # Position of the winner among the finite columns.
        pos_in_idx = int(np.argmax(idx == winner))
        # Winner's rank on OUT (rank 1 = best; ties resolved by first column).
        order = np.argsort(-b[p][idx], kind="stable")
        l = int(np.where(order == pos_in_idx)[0][0]) + 1

        l_ranks[p] = l
        logits[p] = np.log(l / (n_f + 1.0 - l))

    valid = np.isfinite(logits)
    if not valid.any():
        raise RuntimeError("no path had at least 2 rankable configurations")
    # Overfitting = the IS winner landed ABOVE the median OUT (large l, large
    # logit). Exactly-median outcomes (logit 0) do not count.
    pbo = float((logits[valid] > 0).mean())

    return PboResult(
        pbo=pbo,
        logits=logits,
        l_ranks=l_ranks,
        n_paths=n_paths,
        n_configs=n_configs,
        detail={"n_valid_paths": int(valid.sum())},
    )


def cscv_from_path_metrics(
    path_metrics: dict[tuple, dict],
    configs: list,
    n_groups: int,
    n_test_groups: int,
) -> PboResult:
    """Assemble the paired IN/OUT matrices from per-(path, config) metrics.

    Parameters
    ----------
    path_metrics
        ``{(path_index, config_id): {"avg_net": float, "test_groups": (...)}}``
        where ``avg_net`` was measured on that path's TEST groups (its IN
        region) and ``test_groups`` names the combo so the mirror (complement)
        path can be found. Every path must carry the same config ids.
    n_groups, n_test_groups
        The CPCV geometry the paths came from. The OUT region of a path is the
        complement of its test groups; for that complement to be another path,
        ``n_test_groups`` must equal ``n_groups // 2`` -- enforced here with a
        clear error rather than silently mis-pairing regions.

    Raises
    ------
    ValueError
        If the geometry is not symmetric (k != N/2) or a path's mirror is
        missing from ``path_metrics``.
    """
    if not path_metrics:
        raise ValueError("path_metrics is empty")
    if n_test_groups != n_groups - n_test_groups:
        raise ValueError(
            f"CSCV needs symmetric geometry (n_test_groups == n_groups / 2); got "
            f"N={n_groups}, k={n_test_groups}. The OUT region of a path must be "
            "another path in the set, otherwise no honest OUT ranking exists."
        )

    path_combo: dict[int, tuple[int, ...]] = {}
    for (path, cfg_id), m in path_metrics.items():
        if cfg_id == configs[0]:
            if "test_groups" not in m:
                raise ValueError(
                    f"path {path}: metrics must carry 'test_groups' so the "
                    "complement (OUT) path can be identified"
                )
            path_combo[path] = tuple(m["test_groups"])
    if not path_combo:
        raise ValueError(f"no metrics found for any config in {configs!r}")
    combo_path = {combo: path for path, combo in path_combo.items()}

    paths = sorted(path_combo)
    perf_in = np.full((len(paths), len(configs)), np.nan)
    perf_out = np.full((len(paths), len(configs)), np.nan)
    for i, path in enumerate(paths):
        mirror_combo = tuple(sorted(set(range(n_groups)) - set(path_combo[path])))
        mirror = combo_path.get(mirror_combo)
        if mirror is None:
            raise ValueError(
                f"path {path} (groups {path_combo[path]}) has no mirror path for "
                f"{mirror_combo}; CSCV requires the full C(N, N/2) path set"
            )
        for j, cfg_id in enumerate(configs):
            m = path_metrics.get((path, cfg_id))
            if m is not None and np.isfinite(m.get("avg_net", np.nan)):
                perf_in[i, j] = m["avg_net"]
            mo = path_metrics.get((mirror, cfg_id))
            if mo is not None and np.isfinite(mo.get("avg_net", np.nan)):
                perf_out[i, j] = mo["avg_net"]

    return probability_of_backtest_overfitting(perf_in, perf_out)
