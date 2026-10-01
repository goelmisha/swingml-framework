"""Command-line entry points.

    swingml build-dataset --max-days 400
    swingml build-dataset --universe liquidity --start 2020-01-01
    swingml inspect
    swingml clear-cache --namespace prices
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from swingml.config import AppConfig, configure_logging, load_config
from swingml.data.cache import DiskCache
from swingml.dataset import DatasetBundle, build_dataset
from swingml.features import make_feature_provider, resolve_feature_provider
from swingml.labeling import format_diagnostics, triple_barrier_labels

logger = logging.getLogger("swingml.cli")


def _apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    """CLI flags beat the YAML file (flags are for experiments, YAML is the baseline)."""
    changed = False
    for attr in ("start", "end", "price_provider", "delivery_provider"):
        val = getattr(args, attr, None)
        if val is not None:
            setattr(cfg.data, attr, val)
            changed = True
    if getattr(args, "universe", None):
        cfg.universe.source = args.universe
        changed = True
    if getattr(args, "dataset_dir", None):
        # Keep alternative universes (e.g. the point-in-time liquidity build)
        # in their own directory instead of clobbering the current artifacts.
        cfg.paths.dataset_dir = args.dataset_dir
        changed = True
    if getattr(args, "universe_size", None):
        cfg.universe.universe_size = args.universe_size
        changed = True
    if getattr(args, "seed", None) is not None:
        cfg.data.mock.seed = args.seed
        changed = True
    if changed:
        # Re-validate only when a flag actually altered the config, so the
        # purge/horizon advisory is not logged twice for every run.
        cfg.validate()
    return cfg


def cmd_build_dataset(args: argparse.Namespace) -> int:
    cfg = _apply_overrides(load_config(args.config), args)
    bundle = build_dataset(cfg, force=args.force, max_days=args.max_days)
    path = bundle.save(cfg.paths.dataset_dir)
    _print_summary(bundle)
    print(f"\nDataset saved -> {path}")
    print("Next: Step 2 triple-barrier labelling consumes this frame.")
    return 0


def cmd_label_dataset(args: argparse.Namespace) -> int:
    """Step 2: apply the triple-barrier method and write the label file."""
    import json

    cfg = _apply_overrides(load_config(args.config), args)
    if getattr(args, "mode", None):
        cfg.label.mode = args.mode
    label_changed = False
    if getattr(args, "sampling", None):
        cfg.label.sampling = args.sampling
        label_changed = True
    if getattr(args, "cusum_h_mult", None) is not None:
        cfg.label.cusum_h_mult = args.cusum_h_mult
        label_changed = True
    if getattr(args, "cusum_vol_span", None) is not None:
        cfg.label.cusum_vol_span = args.cusum_vol_span
        label_changed = True
    if label_changed:
        cfg.validate()
    # Read and write the SAME directory: labelling an alternative dataset
    # (e.g. the point-in-time liquidity build) must not clobber the artifacts
    # of the default one.
    dataset_dir = str(Path(args.dataset)) if args.dataset else cfg.paths.dataset_dir
    features, meta = DatasetBundle.load(dataset_dir)

    result = triple_barrier_labels(
        features, cfg.label, cfg.costs,
        horizon_days=args.horizon, pt_mult=args.pt_mult, sl_mult=args.sl_mult,
    )
    print(format_diagnostics(result.diagnostics))

    out = Path(dataset_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels_path = out / "labels.parquet"
    result.labels.to_parquet(labels_path, index=False)
    (out / "label_diagnostics.json").write_text(
        json.dumps(result.diagnostics, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nLabels written -> {labels_path}  ({len(result.labels):,} rows)")
    print("Step 3 joins these on (date, symbol) with features.parquet.")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    features, meta = DatasetBundle.load(args.dataset or cfg.paths.dataset_dir)
    print("=" * 78)
    print("DATASET SUMMARY")
    print("=" * 78)
    print(f"rows x cols        : {features.shape[0]:,} x {features.shape[1]}")
    print(f"sessions           : {features['date'].nunique():,}")
    print(f"symbols            : {features['symbol'].nunique():,}")
    print(f"date range         : {features['date'].min().date()} .. {features['date'].max().date()}")
    print(f"friction (roundtrip): {meta.get('round_trip_cost_pct', cfg.costs.round_trip_cost_pct)}")
    print(f"label horizon      : {meta.get('label', {}).get('horizon_days', cfg.label.horizon_days)} bars")
    labels_path = Path(args.dataset or cfg.paths.dataset_dir) / "labels.parquet"
    if labels_path.exists():
        lab = pd.read_parquet(labels_path)
        dist = lab["label"].value_counts(normalize=True).to_dict()
        print(f"labels present     : {len(lab):,} rows  "
              f"(profit {dist.get(1, 0.0):.1%} / expiry {dist.get(0, 0.0):.1%} / stop {dist.get(-1, 0.0):.1%})")
        print(f"effective n        : {lab['uniqueness'].sum():,.0f} independent samples")
    else:
        print("labels present     : none -- run `swingml label-dataset`")

    print("\nrows per session (median / min / max):")
    per = features.groupby("date").size()
    print(f"  {per.median():.0f} / {per.min()} / {per.max()}")

    # Grouping comes from the provider that built the dataset, not from a
    # hard-coded list and not necessarily from the current config.
    spec = meta.get("feature_provider_spec") or cfg.features.provider
    try:
        provider = resolve_feature_provider(spec)(cfg.features)
    except (LookupError, TypeError, ValueError) as exc:
        print(f"\nnote: cannot resolve recorded provider {spec!r} ({exc}); "
              f"grouping by {cfg.features.provider!r} instead")
        provider = make_feature_provider(cfg.features)
    else:
        if spec != cfg.features.provider:
            print(f"\nnote: grouping by the provider recorded in meta.json ({spec!r}), "
                  f"not the one in the current config ({cfg.features.provider!r})")

    print("\nfeature blocks:")
    for group in list(provider.groups) + ["other"]:
        cols = [
            c for c in features.columns
            if provider.group_of(c) == group
            and c not in provider.context_columns
            and c not in ("date", "symbol")
        ]
        if not cols:
            continue
        nan_rate = features[cols].isna().mean().mean()
        print(f"  {group:18s} {len(cols):3d} cols   mean NaN {nan_rate:6.2%}")
    return 0


def _print_summary(bundle: DatasetBundle) -> None:
    f = bundle.features
    print("\n" + "=" * 78)
    print("BUILD SUMMARY")
    print("=" * 78)
    print(f"universe source : {bundle.universe.source}")
    print(f"symbols         : {f['symbol'].nunique():,} with data of {len(bundle.universe.symbols):,} requested")
    print(f"sessions        : {f['date'].nunique():,}  ({f['date'].min().date()} .. {f['date'].max().date()})")
    print(f"rows            : {len(f):,}")
    print(f"columns         : {f.shape[1]}")
    # Context columns are carried for labelling rather than modelled, so report
    # their completeness generically: a NaN barrier-width ATR silently drops rows
    # further down the pipeline, which is worth seeing at build time.
    feature_cols = set(bundle.meta.get("feature_columns", []))
    context = [c for c in f.columns if c not in feature_cols and c not in ("date", "symbol")]
    if context:
        worst = f[context].isna().mean().sort_values(ascending=False)
        shown = ", ".join(f"{c} {r:.2%}" for c, r in worst.head(3).items())
        print(f"context NaN     : {shown}")


def cmd_clear_cache(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    namespaces = [args.namespace] if args.namespace else ["prices", "delivery", "universe"]
    total = 0
    for ns in namespaces:
        # Cache keys are date-stamped, so the namespace itself is the directory.
        d = Path(cfg.paths.cache_dir) / ns
        if not d.exists():
            print(f"{ns:10s}: (no cache)")
            continue
        n = sum(1 for _ in d.glob("*.parquet"))
        for p in d.glob("*.parquet"):
            p.unlink()
        total += n
        print(f"{ns:10s}: removed {n} file(s)")
    print(f"total removed: {total}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swingml", description="NSE swing-trading ML pipeline")
    p.add_argument("--config", default=None, help="path to config YAML")
    p.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-dataset", help="build the Step 1 feature matrix")
    b.add_argument("--start", help="ISO start date (default: config)")
    b.add_argument("--end", help="ISO end date (default: today)")
    b.add_argument("--universe", choices=["nifty200", "nifty500", "liquidity"])
    b.add_argument("--dataset-dir", help="output directory (default: config paths.dataset_dir)")
    b.add_argument("--universe-size", type=int, help="top-N size for --universe liquidity")
    b.add_argument("--price-provider", choices=["yfinance", "mock", "breeze"])
    b.add_argument("--delivery-provider", choices=["nse_bhavcopy", "mock"])
    b.add_argument("--max-days", type=int, help="use only the most recent N sessions (fast smoke run)")
    b.add_argument("--seed", type=int, help="mock provider seed")
    b.add_argument("--force", action="store_true", help="ignore caches and re-download")
    b.set_defaults(func=cmd_build_dataset)

    lb = sub.add_parser("label-dataset", help="Step 2: apply triple-barrier labelling")
    lb.add_argument("--dataset", help="dataset directory (default: config paths.dataset_dir)")
    lb.add_argument("--horizon", type=int, help="vertical barrier in sessions (default: config)")
    lb.add_argument("--pt-mult", type=float, dest="pt_mult", help="take-profit ATR multiple")
    lb.add_argument("--sl-mult", type=float, dest="sl_mult", help="stop-loss ATR multiple")
    lb.add_argument("--mode", choices=["barrier", "fixed_hold"], default=None,
                    help="exit rule: barrier (default) or fixed_hold (definition C as a target)")
    lb.add_argument("--sampling", choices=["every_bar", "cusum"], default=None,
                    help="when a label is emitted: every bar (default) or CUSUM events")
    lb.add_argument("--cusum-h-mult", type=float, dest="cusum_h_mult", default=None,
                    help="CUSUM threshold multiple of EWM volatility (trials: 1.5 / 2 / 3)")
    lb.add_argument("--cusum-vol-span", type=int, dest="cusum_vol_span", default=None,
                    help="EWM span (bars) for the CUSUM volatility estimate")
    lb.set_defaults(func=cmd_label_dataset)

    i = sub.add_parser("inspect", help="summarise a built dataset")
    i.add_argument("--dataset", help="dataset directory (default: config paths.dataset_dir)")
    i.set_defaults(func=cmd_inspect)

    c = sub.add_parser("clear-cache", help="delete cached downloads")
    c.add_argument("--namespace", choices=["prices", "delivery", "universe"])
    c.set_defaults(func=cmd_clear_cache)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        logger.error("interrupted")
        return 130
    except Exception as exc:
        logger.error("%s: %s", type(exc).__name__, exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
