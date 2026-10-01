# Running the heavy jobs off your laptop

Every job in this repository is a pure-CPU batch, so a spot instance is the
cheap way to run the full experiment matrix. Two habits make it painless:

- **Sync the cache, never scrape from the instance.** Exchange archives throttle
  and frequently refuse cloud IPs. Populate `data/cache/` locally first (it is
  gitignored, so it never travels with the repo) and rsync it up.
- **Run inside `tmux`.** A dropped SSH session should not kill a multi-hour fit.

## 1. Instance

Ubuntu 24.04 LTS, a compute-optimised spot instance with 4+ vCPUs, 30 GB gp3.
Python 3.12 — 3.13+ may not have wheels for the gradient-boosting libraries.

```bash
sudo apt update && sudo apt install -y git rsync curl
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"          # puts uv on PATH for this shell

git clone <this-repo-url> swingml && cd swingml
uv sync --extra dev --extra ml --python 3.12   # reads the committed lockfile
.venv/bin/python -m pytest -q                  # green before any experiment
```

## 2. Data

```bash
# from your laptop, from the project root
rsync -avP data/cache/ ubuntu@<HOST>:~/swingml/data/cache/
rsync -avP data/raw/   ubuntu@<HOST>:~/swingml/data/raw/     # if non-empty
```

The delivery cache is one file per session covering every listed symbol, so it is
reused across every universe choice — including a point-in-time liquidity
universe, which needs no additional exchange downloads. Only the price source
runs on the instance.

## 3. The jobs

```bash
tmux new -s swingml
cd ~/swingml && source .venv/bin/activate && mkdir -p ~/logs

# a. per-regime vs single-model A/B, walk-forward, all engines
time python scripts/regime_experiment.py --engine sklearn-hgb lightgbm xgboost \
    2>&1 | tee ~/logs/regime.log

# b. probability of backtest overfitting over CPCV paths (the heaviest job)
time python scripts/pbo_experiment.py --engine sklearn-hgb lightgbm xgboost \
    2>&1 | tee ~/logs/pbo.log

# c. a survivorship-free dataset, in its own directory
python -m swingml.cli build-dataset --universe liquidity --start 2020-01-01 \
    --dataset-dir data/datasets_liquidity 2>&1 | tee ~/logs/build.log
python -m swingml.cli label-dataset --dataset data/datasets_liquidity
python -m swingml.cli inspect --dataset data/datasets_liquidity

# d. re-measure on the bias-free universe
python scripts/signal_check.py --dataset-dir data/datasets_liquidity \
    2>&1 | tee ~/logs/signal.log
python scripts/regime_experiment.py --engine sklearn-hgb lightgbm xgboost \
    --dataset-dir data/datasets_liquidity 2>&1 | tee ~/logs/regime_liquidity.log
```

Detach with `Ctrl-b d`, reattach with `tmux a -t swingml`.

Run (a) first: it is the cheapest and it tells you the real per-fold cost before
you commit to (b). Before (d), sanity-check (c): a liquidity universe should
report more symbols than an index-list build, while the label mix stays in the
same broad range. A drastically different mix means the screen changed *what you
are measuring*, not merely how much of it.

## 4. Results and cost

```bash
# from your laptop
rsync -avP ubuntu@<HOST>:~/swingml/data/datasets_liquidity/ data/datasets_liquidity/
rsync -avP ubuntu@<HOST>:~/swingml/data/trials.jsonl data/trials.jsonl
rsync -avP ubuntu@<HOST>:~/logs/ logs/
```

`data/trials.jsonl` is the experiment ledger: keep it, because the count of
trials you have run is the denominator any deflated performance estimate needs.

Then **terminate** the instance rather than stopping it — a stopped instance
still bills for its disk. Use spot, set a modest total-cost guard, and expect a
couple of dollars for an afternoon's compute.
