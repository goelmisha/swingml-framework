"""Trial ledger -- the raw input to any multiple-testing correction.

Why
---
The Deflated Sharpe Ratio needs the number of *trials* (independent
configurations whose scores were examined) behind a headline result. That
number cannot be reconstructed after the fact: it must be counted while the
search happens. Every experiment script appends one line per run here, and
``count_trials`` is what Step 4 will divide by.

What counts as a trial is a judgement call and is recorded explicitly:
a walk-forward experiment over 7 folds that compares 2 arms and reports one
selection threshold is ONE trial per arm per threshold -- the folds are
resamples of one configuration, not independent configurations. Sweeping the
selection threshold (top-10% and top-5%) or the feature set multiplies the
count. Over-counting is safe (DSR grows more conservative); under-counting is
the error that produced fake Sharpe ratios across the industry.

The ledger is append-only JSONL under ``data/`` and is gitignored with the
rest of the data directory. Losing it only costs honesty about history; the
correction itself is recomputed from the surviving entries.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from swingml.config import PROJECT_ROOT

TRIALS_FILENAME = "trials.jsonl"


def trials_path(dataset_dir: str | Path | None = None) -> Path:
    base = Path(dataset_dir) if dataset_dir else PROJECT_ROOT / "data"
    return base / TRIALS_FILENAME


def record_trial(
    script: str,
    dataset: str,
    engine: str,
    question: str,
    config: dict | None = None,
    metrics: dict | None = None,
    n_folds: int | None = None,
    trials_added: int = 1,
    dataset_dir: str | Path | None = None,
) -> dict:
    """Append one trial record; returns what was written (for scripts to echo)."""
    entry = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "script": script,
        "dataset": dataset,
        "engine": engine,
        "question": question,
        "config": config or {},
        "metrics": metrics or {},
        "n_folds": n_folds,
        # Number of independent configurations this entry contributes to the
        # DSR denominator. One config evaluated on many folds = 1, not n_folds.
        "trials_added": int(trials_added),
    }
    p = trials_path(dataset_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    return entry


def load_trials(dataset_dir: str | Path | None = None) -> list[dict]:
    p = trials_path(dataset_dir)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def count_trials(dataset_dir: str | Path | None = None) -> dict:
    """Total trials and the breakdown the DSR correction needs."""
    trials = load_trials(dataset_dir)
    by_script: dict[str, int] = {}
    for t in trials:
        by_script[t["script"]] = by_script.get(t["script"], 0) + int(t.get("trials_added", 1))
    return {
        "total": sum(by_script.values()),
        "by_script": by_script,
        "n_runs": len(trials),
    }


def format_ledger(dataset_dir: str | Path | None = None) -> str:
    """Human-readable ledger for ``--print-ledger``."""
    trials = load_trials(dataset_dir)
    if not trials:
        return "trial ledger is empty"
    c = count_trials(dataset_dir)
    lines = [f"trial ledger: {c['total']} trials across {c['n_runs']} runs"]
    for script, n in sorted(c["by_script"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {script:28s} {n:>5d} trials")
    lines.append("")
    lines.append(f"{'when':>17s} {'script':26s} {'engine':12s} question")
    for t in trials[-25:]:
        q = t["question"]
        lines.append(f"{t['timestamp']:>17s} {t['script']:26s} {t['engine']:12s} {q[:70]}")
    if len(trials) > 25:
        lines.append(f"  ... {len(trials) - 25} earlier runs elided")
    return "\n".join(lines)
