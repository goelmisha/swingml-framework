"""Step-4 scorecard: what the selected trades earned, and whether the Sharpe
survived the search that produced it.

Why this module exists
----------------------
The existence test asked whether the model's selections beat the same-date
universe mean. That is a question about *ranking*, and it cannot answer the two
questions a trading decision needs:

1. Does the selection make money as a **portfolio** -- Sharpe, Sortino, max
   drawdown, profit factor -- rather than merely a higher average on the rows
   it picked?
2. Is that Sharpe distinguishable from the best Sharpe a search over the
   trials ledger would have produced from pure noise? That is the **Deflated
   Sharpe Ratio** (Bailey and Lopez de Prado, 2014), and until it exists every
   headline number in this repo is undeflated.

Three rules inherited from the rest of the project:

* Returns passed in must already carry the 0.25% round-trip friction
  (``labels.parquet["ret_net"]``). This module never re-charges it.
* **Periods must not overlap.** Every ``ret_net`` here is a ``horizon``-session
  holding return, so taking one per *session* counts the same trade ten times,
  compounds it ten times, and inflates the Sharpe by roughly sqrt(10). A first
  version of this module did exactly that and printed wildly inflated
  performance. ``stride`` exists to make the non-overlapping sampling explicit:
  pass the label horizon and annualise with
  ``periods_per_year = 252 / horizon``.
* A series that overlapping-samples is not merely optimistic, it is
  meaningless -- the same trap overlapping labels create for t-statistics.
* Selection is done by :func:`swingml.evaluation.top_fraction_mask`, the same
  implementation every experiment uses, so a portfolio series can never be
  built by a different definition of "top decile" than the precision tables.

Definitions A/B live in :mod:`swingml.evaluation`. This module converts a
selection into a return *series* and scores the series; it does not restate
precision, and it never touches definition C.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from swingml.evaluation import top_fraction_mask

#: Sessions per year for annualising per-session statistics. NSE trades ~250
#: sessions a year; 252 is the convention and the difference is cosmetic.
SESSIONS_PER_YEAR = 252

#: Euler-Mascheroni constant, as used in the expected-maximum-Sharpe term.
EULER_GAMMA = 0.5772156649015329

#: Bailey and Lopez de Prado's significance bar for the DSR (a probability).
DSR_SIGNIFICANCE = 0.95


def period_returns(
    block: pd.DataFrame,
    scores: np.ndarray,
    decile: int = 10,
    stride: int = 1,
    return_col: str = "ret_net",
    extra_friction: float | np.ndarray = 0.0,
) -> pd.Series:
    """Equal-weight net return of the top ``1/decile`` selection, per period.

    One number per **non-overlapping period**: the mean ``ret_net`` of the names
    that a period's ranking put in the top slice, sampled every ``stride``
    sessions. Equal weight because nothing in this repo has yet justified a
    sizing rule -- that is what the meta-model's probability is earmarked for.

    ``stride`` defaults to 1 only for single-session labels. With a 10-session
    label horizon, pass ``stride=horizon``: consecutive sessions' labels share
    nine sessions of the same price path, so sampling them all as separate
    periods multiplies one trade into ten.

    ``return_col`` selects which trade the series scores: ``ret_net`` is the
    barrier trade (definitions A/B), ``fwd_ret_<h>`` is the fixed-hold trade
    (definition C). Scoring the *same* selection both ways is what separates a
    ranking that survives a path from one that only survives drift.

    ``extra_friction`` is charged once per period, on top of whatever the column
    already carries -- the friction stress test. Equal weighting means charging
    it per selected row and charging it once to the period mean are identical,
    so it may also be a **per-row array** (a size-aware impact cost), in which
    case the period charge is the mean over the selected names.
    """
    scores_arr = np.asarray(scores, dtype=float)
    if len(block) != len(scores_arr):
        raise ValueError(f"block rows ({len(block)}) != scores ({len(scores_arr)})")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if return_col not in block.columns:
        raise ValueError(f"{return_col!r} not in the block; cannot score this trade")

    # Build the frame FIRST: top_fraction_mask aligns scores and dates on their
    # index, and a caller's block carries whatever index the split produced
    # (df.iloc), not a RangeIndex.
    charge = np.asarray(extra_friction, dtype=float)
    if charge.ndim == 0:
        charge = np.full(len(block), float(charge))
    elif len(charge) != len(block):
        raise ValueError(f"extra_friction ({len(charge)}) != block rows ({len(block)})")

    frame = pd.DataFrame({
        "date": pd.to_datetime(block["date"]).to_numpy(),
        "trade_ret": block[return_col].to_numpy(dtype=float) - charge,
        "score": scores_arr,
    })
    if stride > 1:
        sessions = np.sort(frame["date"].unique())
        kept = set(sessions[::stride])
        frame = frame[frame["date"].isin(kept)].reset_index(drop=True)

    mask = top_fraction_mask(frame["score"], frame["date"], decile)
    selected = frame.loc[mask]
    if selected.empty:
        return pd.Series(dtype=float)
    return selected.groupby("date")["trade_ret"].mean().sort_index()


@dataclass
class Performance:
    """Portfolio statistics for one return series, per-session units."""

    n_periods: int = 0
    mean_return: float = float("nan")
    #: Per-period Sharpe (NOT annualised) -- the units the DSR requires.
    sharpe: float = float("nan")
    sortino: float = float("nan")
    max_drawdown: float = float("nan")
    profit_factor: float = float("nan")
    hit_rate: float = float("nan")
    skew: float = float("nan")
    kurtosis: float = float("nan")
    #: Sessions each period spans (the label horizon), and the matching count
    #: of periods per year. Annualising with the wrong one silently rescales
    #: every Sharpe by a constant, which is exactly how an overlap bug hides.
    period_sessions: int = 1
    periods_per_year: float = float(SESSIONS_PER_YEAR)

    @property
    def sharpe_annualised(self) -> float:
        return self.sharpe * math.sqrt(self.periods_per_year)

    @property
    def sortino_annualised(self) -> float:
        return self.sortino * math.sqrt(self.periods_per_year)

    def as_row(self) -> dict:
        return {
            "n_periods": self.n_periods,
            "period_sessions": self.period_sessions,
            "periods_per_year": self.periods_per_year,
            "mean_return": self.mean_return,
            "sharpe_per_period": self.sharpe,
            "sharpe_annualised": self.sharpe_annualised,
            "sortino_annualised": self.sortino_annualised,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
            "hit_rate": self.hit_rate,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
        }


def performance_metrics(
    returns: pd.Series | np.ndarray,
    periods_per_year: float = float(SESSIONS_PER_YEAR),
    period_sessions: int = 1,
) -> Performance:
    """Sharpe / Sortino / MDD / profit factor for a per-period return series.

    ``periods_per_year`` must match the sampling: for a series of 10-session
    holding returns sampled every 10th session, pass ``252 / 10``. Pass the
    wrong factor and the Sharpe is rescaled by a constant -- plausible-looking
    and wrong.

    Degenerate series return ``nan`` rather than a flattering number: a series
    with zero dispersion has no defined Sharpe, and silently printing
    ``inf`` would make every downstream annualisation meaningless.
    """
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if r.size == 0:
        return Performance(period_sessions=period_sessions, periods_per_year=periods_per_year)

    mean = float(r.mean())
    sd = float(r.std(ddof=1)) if r.size > 1 else 0.0
    sharpe = mean / sd if sd > 0 else float("nan")

    downside = r[r < 0]
    dsd = float(np.sqrt((downside ** 2).mean())) if downside.size else 0.0
    sortino = mean / dsd if dsd > 0 else float("nan")

    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max(1.0 - equity / peak)) if equity.size else float("nan")

    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    if losses > 0:
        pf = gains / losses
    else:
        pf = float("inf") if gains > 0 else float("nan")

    n = r.size
    skew = float(((r - mean) ** 3).mean() / sd ** 3) if sd > 0 else float("nan")
    kurt = float(((r - mean) ** 4).mean() / sd ** 4) if sd > 0 else float("nan")

    return Performance(
        n_periods=int(n),
        period_sessions=period_sessions,
        periods_per_year=periods_per_year,
        mean_return=mean,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        profit_factor=pf,
        hit_rate=float((r > 0).mean()),
        skew=skew,
        kurtosis=kurt,
    )


def benchmark_period_returns(
    bench: pd.DataFrame,
    period_starts: pd.Index,
    horizon: int,
    friction: float = 0.0,
) -> pd.Series:
    """Buy-and-hold market return over the SAME windows the strategy trades.

    Entry follows the strategy's own convention -- the next session's **open**
    after the signal -- and the exit is the close ``horizon`` sessions later, so
    both series cover identical windows and a beta between them is a
    like-for-like exposure. ``friction`` is charged to the benchmark as well,
    because every strategy period pays the round trip.
    """
    if bench is None or bench.empty:
        return pd.Series(dtype=float)

    px = bench.sort_index()
    px.index = pd.to_datetime(px.index).normalize()
    opens = px["open"].to_numpy(dtype=float)
    closes = px["close"].to_numpy(dtype=float)
    position = {d: i for i, d in enumerate(px.index)}

    out: dict[pd.Timestamp, float] = {}
    for start in pd.to_datetime(pd.Index(period_starts)).normalize():
        i = position.get(start)
        if i is None or i + horizon >= len(px):
            continue
        entry = opens[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        out[start] = float(closes[i + horizon] / entry - 1.0) - friction
    return pd.Series(out).sort_index()


def beta_analysis(
    strategy: pd.Series,
    benchmark: pd.Series,
    periods_per_year: float = float(SESSIONS_PER_YEAR),
) -> dict:
    """Market exposure of the strategy's period returns, and what is left of it.

    Reports beta, per-period alpha, R-squared and the Sharpe of the
    **beta-hedged** residual ``r_strategy - beta * r_benchmark`` -- the closest
    thing here to the market-neutral variant STATUS section 8 item 7 asks for.
    A long-only strategy in a rising sample can post a Sharpe that is mostly
    this beta; the residual is the part the market does not explain.
    """
    pair = pd.concat({"s": strategy, "b": benchmark}, axis=1).dropna()
    empty = {
        "n_periods": int(len(pair)), "beta": float("nan"), "alpha_per_period": float("nan"),
        "alpha_annualised": float("nan"), "r_squared": float("nan"),
        "residual_sharpe_annualised": float("nan"), "residual_mean": float("nan"),
        "bench_sharpe_annualised": float("nan"),
    }
    if len(pair) < 3 or pair["b"].std(ddof=1) == 0:
        return empty

    var_b = float(pair["b"].var(ddof=1))
    beta = float(pair["s"].cov(pair["b"]) / var_b)
    alpha = float(pair["s"].mean() - beta * pair["b"].mean())
    corr = float(pair["s"].corr(pair["b"]))
    residual = pair["s"] - beta * pair["b"]
    resid_perf = performance_metrics(residual, periods_per_year=periods_per_year)
    bench_perf = performance_metrics(pair["b"], periods_per_year=periods_per_year)

    # If the strategy is (numerically) the market, the residual is float noise
    # and its Sharpe is an arbitrary signed number. Report nan, not a verdict.
    strat_sd = float(pair["s"].std(ddof=1))
    resid_sd = float(residual.std(ddof=1))
    resid_sharpe = resid_perf.sharpe_annualised
    if strat_sd > 0 and resid_sd < 1e-9 * strat_sd:
        resid_sharpe = float("nan")

    return {
        "n_periods": int(len(pair)),
        "beta": beta,
        "alpha_per_period": alpha,
        "alpha_annualised": alpha * periods_per_year,
        "r_squared": corr ** 2,
        "residual_sharpe_annualised": resid_sharpe,
        "residual_mean": float(residual.mean()),
        #: The beta-hedged series itself, for scoring (skew/kurtosis/DSR) rather
        #: than re-deriving it per caller.
        "residual_series": residual,
        "bench_sharpe_annualised": bench_perf.sharpe_annualised,
        "bench_mean": float(pair["b"].mean()),
    }


def expected_max_sharpe(trial_sharpe_variance: float, n_trials: int) -> float:
    """E[max Sharpe] under the null that every trial's true Sharpe is zero.

    Bailey and Lopez de Prado's order-statistic approximation. This is the
    number that makes a search expensive: with many trials, noise alone
    produces a large best Sharpe, and the DSR has to clear it.
    """
    if n_trials < 2:
        raise ValueError("n_trials must be >= 2 for a non-degenerate expected maximum")
    if trial_sharpe_variance < 0:
        raise ValueError("trial_sharpe_variance must be non-negative")
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(trial_sharpe_variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    trial_sharpe_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> dict:
    """Probability that the observed Sharpe is not an artefact of the search.

    All arguments are in **per-observation** units: pass the Sharpe of the
    per-session (or per-fold) series and a trial-Sharpe variance measured on
    the same frequency. Annualising one but not the other is the classic way to
    make this number meaningless, so it is asserted by the caller, not here.

    ``kurtosis`` is the raw (non-excess) fourth moment; 3.0 is the normal value.

    A result at or above :data:`DSR_SIGNIFICANCE` is conventionally the bar for
    "not explained by the number of attempts".
    """
    if not np.isfinite(sharpe):
        return {"dsr": float("nan"), "expected_max_sharpe": float("nan"),
                "significant": False, "n_trials": n_trials, "n_obs": n_obs}
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")

    e_max = expected_max_sharpe(trial_sharpe_variance, n_trials)
    denominator = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe ** 2
    if denominator <= 0:
        return {"dsr": float("nan"), "expected_max_sharpe": e_max,
                "significant": False, "n_trials": n_trials, "n_obs": n_obs}

    z = (sharpe - e_max) * math.sqrt(n_obs - 1) / math.sqrt(denominator)
    dsr = float(norm.cdf(z))
    return {
        "dsr": dsr,
        "expected_max_sharpe": e_max,
        "significant": bool(dsr >= DSR_SIGNIFICANCE),
        "n_trials": n_trials,
        "n_obs": n_obs,
    }


def format_beta_control(analysis: dict, label: str = "strategy") -> str:
    """Beta-control block: how much of the Sharpe the market explains."""
    if analysis["n_periods"] < 3 or not np.isfinite(analysis["beta"]):
        return "  beta control unavailable (not enough paired periods)"
    return "\n".join([
        f"  {label} vs market, {analysis['n_periods']} paired periods",
        f"  market: mean {analysis['bench_mean']:+.3%}/period, Sharpe "
        f"{analysis['bench_sharpe_annualised']:.2f} (annualised)",
        f"  beta {analysis['beta']:.2f} | R\u00b2 {analysis['r_squared']:.2f} | "
        f"alpha {analysis['alpha_per_period']:+.3%}/period "
        f"(arithmetic {analysis['alpha_annualised']:+.1%}/yr)",
        f"  beta-hedged residual: mean {analysis['residual_mean']:+.3%}/period, Sharpe "
        f"{analysis['residual_sharpe_annualised']:.2f} (annualised) -- the part the "
        f"market does not explain",
    ])


def format_scorecard(perf: Performance, dsr: dict | None, label: str = "strategy") -> str:
    """One-line-per-metric block, always with the trial count beside the DSR."""
    span = ("session" if perf.period_sessions == 1
            else f"{perf.period_sessions}-session period")
    lines = [
        f"{label}: {perf.n_periods} non-overlapping {span}s, mean net {perf.mean_return:+.3%}",
        f"  Sharpe {perf.sharpe_annualised:6.2f} (annualised) | Sortino {perf.sortino_annualised:6.2f} "
        f"| max drawdown {perf.max_drawdown:.2%} | profit factor {perf.profit_factor:.2f} "
        f"| hit rate {perf.hit_rate:.1%}",
    ]
    if dsr is None:
        return "\n".join(lines)
    verdict = "SIGNIFICANT" if dsr["significant"] else "NOT significant"
    # Annualise the noise bar with the SAME period length as the observed Sharpe.
    lines.append(
        f"  deflated Sharpe {dsr['dsr']:.3f} ({verdict} at {DSR_SIGNIFICANCE:.2f}) | "
        f"expected max Sharpe under the null {dsr['expected_max_sharpe']:.3f}/period "
        f"(annualised {dsr['expected_max_sharpe'] * math.sqrt(perf.periods_per_year):.2f} vs "
        f"observed {perf.sharpe_annualised:.2f}) over {dsr['n_trials']} trials, "
        f"{dsr['n_obs']} observations"
    )
    return "\n".join(lines)
