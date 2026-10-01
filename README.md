# SwingML — a leak-free ML pipeline for swing trading

A research pipeline for cross-sectional equity swing trading on daily bars, built
around one conviction: **most backtests fail before any modelling happens**, and
almost always in a way that makes the results look *better*, not worse.

So this repository is mostly machinery for not fooling yourself. Labelling,
split geometry, friction and evaluation are each implemented once, in one place,
with the failure mode they exist to prevent written down next to them and a test
that fails if it regresses.

```
pipelines that look good          pipelines that hold up
────────────────────────          ──────────────────────
random K-Fold CV                  purged walk-forward + CPCV
label overlap ignored             purge >= label horizon, enforced
no costs                          0.25% round trip, mandatory
one "precision" number            two definitions, each vs its base rate
"the model scores 55%"            "55% against a 50% base, +0.05% net/trade"
```

## What is in the box

| Module | Responsibility |
|---|---|
| `swingml/data/` | Point-in-time liquidity universe, exchange bhavcopy parsing, price + delivery caching, pluggable providers |
| `swingml/features/` | Feature-provider contract, registry, indicator primitives, corporate-action repair, a reference provider |
| `swingml/labeling.py` | Triple-barrier labels with sample uniqueness, from the forward intrabar path |
| `swingml/validation.py` | Purged walk-forward splits and combinatorial purged cross-validation (CPCV) |
| `swingml/pbo.py` | CSCV probability of backtest overfitting, with paired in/out mirror paths |
| `swingml/evaluation.py` | The single Tier-1 scorer: precision A and B, each against its base rate |
| `swingml/trials.py` | Append-only trial ledger, so the number of things you tried is a fact, not a memory |
| `swingml/models.py` | One protocol across gradient-boosting engines (scikit-learn, LightGBM, XGBoost) |
| `swingml/config.py` | Typed schema; every number that affects a score lives in YAML, never in code |

## The four disciplines

**1. Causality, proven rather than asserted.** `tests/test_leakage.py` truncates
the sample and recomputes: a feature built on data up to `t` must be bit-identical
when the data after `t` is deleted, including its NaN mask. A shifted NaN mask is
itself evidence of a look-ahead window, so the test checks that too.

**2. Splits that respect overlapping labels.** A `horizon`-day label at bar `t`
resolves `horizon` bars later, so a purge shorter than the horizon still lets
training outcomes bleed into the test block. The requested purge is treated as a
floor and widened automatically, and `assert_no_overlap` fails the run if any
geometry slips through. CPCV adds the mirror-paired paths that CSCV needs.

**3. Friction on every trade.** Not a sensitivity knob — applied to every trade
or the run is invalid. A 0.2% gross move is a *loss* after a 0.25% round trip,
which is the whole reason naive up/down labels look profitable.

**4. Two precisions, never one.** A model that reports only the barrier-hit rate
understates; one that reports only the money-made rate flatters. Both are
reported together, each against the base rate of its own evaluation block:

- **A** — `P(label == 1)`: the barrier trade
- **B** — `P(ret_net > 0)`: the same trade, scored after friction

## Quickstart

Runs entirely offline on synthetic data — no network, no API keys:

```bash
uv sync --extra dev --extra ml     # Python 3.12 + the committed lockfile

# build a dataset with the deterministic mock providers
python -m swingml.cli build-dataset --price-provider mock --delivery-provider mock \
    --max-days 400 --dataset-dir data/demo
python -m swingml.cli label-dataset --dataset data/demo
python -m swingml.cli inspect --dataset data/demo

python -m pytest -q                # the leak-free properties are the test suite
```

To run against real data, edit `config/config.yaml` and drop `--price-provider`:
prices come from `yfinance` (split-adjusted) and delivery/volume from the
exchange's daily full bhavcopy.

## Bring your own features

Nothing downstream knows which feature set is in use. A provider owns its
`transform`, its feature `groups`, and the `context_columns` the labeller needs;
the framework resolves it from config at runtime:

```yaml
features:
  provider: demo                                   # shipped reference provider
  # provider: "my_package.signals:MyFeatureProvider"   # your own, by dotted path
  # provider: mine                                # or via an entry point
```

A provider is ~100 lines; `swingml/features/demo.py` is a worked example, and
`swingml/features/actions.py` shows the corporate-action handling you will
probably want to reuse. Registering one as an entry point is how a private
feature set stays private while the pipeline stays shared:

```toml
[project.entry-points."swingml.features"]
mine = "my_private_package.signals:MyFeatureProvider"
```

`tests/test_public_boundary.py` enforces the direction of that seam: no module
outside the private overlay may import from it, because the published release has
to be runnable on its own.

## Layout

```
swingml/
├── config.py         typed schema; YAML is the single source of truth
├── data/             universe selection, bhavcopy parsing, caching, providers
├── features/         provider contract, registry, primitives, reference provider
├── labeling.py       triple-barrier labels + sample uniqueness
├── validation.py     purged walk-forward, CPCV, leak assertions
├── pbo.py            CSCV probability of backtest overfitting
├── evaluation.py     the shared Tier-1 scorer (definitions A and B)
├── trials.py         append-only experiment ledger
├── models.py         engine-agnostic classifier protocol
└── cli.py            build-dataset / label-dataset / inspect / clear-cache
scripts/              standalone experiments (signal check, regime A/B, PBO)
tests/                the leak-free invariants, plus mock-provider fixtures
```

## What is deliberately not here

The production feature set, its hyperparameters, the traded universe and every
measured result are absent by design. The interesting contribution is the
validation and evaluation machinery, and it is more useful to you as a framework
you can point at your own hypothesis than as somebody else's signals. The plugin
seam above is exactly where the missing piece plugs in.

## License

MIT — see [LICENSE](LICENSE).

## Status

Research code, published to be read and reused. No performance claim is made or
implied, and nothing here is investment advice. It is built for one market's
microstructure (delivery-based daily data), so the data layer in particular will
need adapting elsewhere.
