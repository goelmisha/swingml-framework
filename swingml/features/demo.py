"""Reference feature provider.

A deliberately plain, textbook feature set -- trend/momentum, volatility,
liquidity and market-regime blocks -- that exists to demonstrate the pipeline
end to end: build a dataset, label it with triple barriers, validate it
walk-forward, fit a model and score it. It is a working example, not a
recommendation; the interesting part of this repository is the validation and
evaluation machinery around it.

Swap in your own provider by pointing ``features.provider`` at it (see
:mod:`swingml.features.registry`). Nothing downstream needs to change.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from swingml.features.actions import (
    adjust_quantities_for_corporate_actions,
    detect_corporate_actions,
)
from swingml.features.base import FeatureProvider
from swingml.features.primitives import (
    atr_wilder,
    ema,
    realized_vol,
    roc,
    rsi_wilder,
)

logger = logging.getLogger(__name__)


class DemoFeatureProvider(FeatureProvider):
    """Trend / volatility / liquidity / regime blocks on adjusted prices."""

    name = "demo"

    groups: dict[str, tuple[str, ...]] = {
        "trend_momentum": (
            "ret_1d", "roc_5", "roc_10", "roc_20", "roc_60",
            "close_to_ema20", "close_to_ema50", "close_to_ema200",
            "ema50_to_ema200", "rsi_14",
        ),
        "volatility": (
            "atr_pct", "realized_vol_10", "realized_vol_20",
            "dist_52w_high", "dist_52w_low", "range_pct", "gap_pct",
            "close_location",
        ),
        "liquidity": (
            "log_turnover", "turnover_lacs", "adv_20", "turnover_trend",
            "vol_ratio_5", "vol_ratio_20", "vol_ratio_60",
            "volume_expansion", "trades_ratio",
        ),
        "market_regime": (
            "bench_close_to_ema200", "bench_above_ema200", "bench_above_ema50",
            "bench_roc_20", "bench_roc_60", "bench_atr_pct",
            "bench_realized_vol_20", "rel_strength_20", "rel_strength_60",
        ),
    }

    # -- blocks -------------------------------------------------------------
    def _price_features(self, px: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=px.index)
        close, high, low, open_ = px["close"], px["high"], px["low"], px["open"]

        f["ret_1d"] = close.pct_change(fill_method=None)
        for w in self.cfg.roc_windows:
            f[f"roc_{w}"] = roc(close, w)

        emas = {w: ema(close, w) for w in self.cfg.ema_windows}
        for w in self.cfg.ema_windows:
            f[f"close_to_ema{w}"] = close / emas[w] - 1.0
        if 50 in emas and 200 in emas:
            f["ema50_to_ema200"] = emas[50] / emas[200] - 1.0

        f["rsi_14"] = rsi_wilder(close, self.cfg.rsi_window)

        f["atr_pct"] = atr_wilder(high, low, close, self.cfg.atr_window) / close
        # Barrier-width ATR. Context for labelling, never a model input.
        f["atr_20"] = atr_wilder(high, low, close, 20)

        for w in self.cfg.realized_vol_windows:
            f[f"realized_vol_{w}"] = realized_vol(close, w)

        win = self.cfg.range_window
        warm = max(2, self.cfg.min_history_days // 2)
        hi = high.rolling(win, min_periods=warm).max()
        lo = low.rolling(win, min_periods=warm).min()
        f["dist_52w_high"] = close / hi - 1.0
        f["dist_52w_low"] = close / lo - 1.0

        f["range_pct"] = (high - low) / close
        f["gap_pct"] = open_ / close.shift(1) - 1.0
        rng = (high - low).replace(0.0, np.nan)
        f["close_location"] = (close - low) / rng

        f["open"], f["high"], f["low"], f["close"] = open_, high, low, close
        return f

    def _volume_features(self, deliv: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=deliv.index)
        qty = deliv["ttl_trd_qnty"].astype(float)
        turnover = deliv["turnover_lacs"].astype(float)

        f["log_turnover"] = np.log1p(turnover)
        f["turnover_lacs"] = turnover
        f["adv_20"] = turnover.rolling(20, min_periods=5).mean()
        f["turnover_trend"] = (
            turnover.rolling(5, min_periods=3).mean()
            / turnover.rolling(20, min_periods=5).mean()
            - 1.0
        )

        # Ratios only. A raw quantity level is not comparable across a split;
        # a ratio against its own trailing mean is.
        for w in self.cfg.vol_windows:
            f[f"vol_ratio_{w}"] = qty / qty.rolling(w, min_periods=max(2, w // 2)).mean()
        f["volume_expansion"] = f["vol_ratio_5"] / f["vol_ratio_20"].replace(0.0, np.nan)

        trades = deliv["no_of_trades"].astype(float)
        f["trades_ratio"] = trades / trades.rolling(20, min_periods=5).mean()
        return f

    def _regime_features(self, bench: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
        """Benchmark block, treated strictly as a filter and reindexed forward."""
        b = bench.sort_index()
        close = b["close"]
        f = pd.DataFrame(index=index)

        ema200, ema50 = ema(close, 200), ema(close, 50)
        f["bench_close_to_ema200"] = close / ema200 - 1.0
        f["bench_above_ema200"] = (close > ema200).astype(float)
        f["bench_above_ema50"] = (close > ema50).astype(float)
        f["bench_roc_20"] = roc(close, 20)
        f["bench_roc_60"] = roc(close, 60)
        f["bench_atr_pct"] = atr_wilder(b["high"], b["low"], close, 14) / close
        f["bench_realized_vol_20"] = realized_vol(close, 20)

        # ffill propagates the last known benchmark state forward, never backward.
        return f.reindex(index).ffill()

    def _cross_sectional(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-date percentile ranks. Uses only same-date information."""
        if not self.cfg.cross_sectional_ranks:
            return df
        cols = [c for c in ("roc_20", "close_to_ema200", "vol_ratio_20",
                            "turnover_lacs", "rsi_14") if c in df.columns]
        for c in cols:
            df[f"xs_rank_{c}"] = df.groupby("date")[c].rank(pct=True)
        return df

    # -- orchestration ------------------------------------------------------
    def transform(
        self,
        prices: dict[str, pd.DataFrame],
        delivery: pd.DataFrame,
        bench: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if not prices:
            raise ValueError("no price data supplied")
        if delivery is None or delivery.empty:
            raise ValueError("no delivery data supplied")

        delivery = delivery.copy()
        delivery["date"] = pd.to_datetime(delivery["date"]).dt.normalize()
        by_symbol = {
            sym: g.set_index("date").sort_index()
            for sym, g in delivery.groupby("symbol", observed=True)
        }

        regime_cache: dict[tuple, pd.DataFrame] = {}
        frames: list[pd.DataFrame] = []

        for sym, px in prices.items():
            if px is None or px.empty:
                continue
            deliv = by_symbol.get(sym)
            if deliv is None or deliv.empty:
                continue

            px = px.sort_index()
            # The exchange panel defines the real trading calendar.
            common = px.index.intersection(deliv.index)
            if len(common) < self.cfg.min_history_days:
                logger.debug("%s: only %d common sessions (< warm-up); skipped", sym, len(common))
                continue
            px, deliv = px.loc[common], deliv.loc[common]

            # Repair share counts BEFORE any quantity ratio is taken, so trailing
            # means never straddle a split.
            if "prev_close" in deliv.columns:
                is_action, ratio = detect_corporate_actions(
                    deliv["prev_close"], deliv["close"].shift(1)
                )
                if bool(is_action.any()):
                    self.corporate_actions_detected += int(is_action.sum())
                    for col in ("ttl_trd_qnty", "deliv_qty", "no_of_trades"):
                        if col in deliv.columns:
                            deliv[col] = adjust_quantities_for_corporate_actions(
                                deliv[col], ratio, is_action
                            )

            out = self._price_features(px).join(self._volume_features(deliv), how="left")

            if bench is not None and not bench.empty:
                key = (out.index[0], out.index[-1])
                if key not in regime_cache:
                    regime_cache[key] = self._regime_features(bench, out.index)
                rf = regime_cache[key]
                out = out.join(rf, how="left")
                if "roc_20" in out and "bench_roc_20" in rf:
                    out["rel_strength_20"] = out["roc_20"] - rf["bench_roc_20"]
                if "roc_60" in out and "bench_roc_60" in rf:
                    out["rel_strength_60"] = out["roc_60"] - rf["bench_roc_60"]

            out["symbol"] = sym
            frames.append(out.reset_index().rename(columns={"index": "date"}))

        if not frames:
            raise RuntimeError(
                "feature matrix is empty -- no symbol had usable price+delivery overlap"
            )

        full = pd.concat(frames, ignore_index=True)

        # Drop the warm-up window: rows where the long windows are still NaN.
        required = [c for c in ("close_to_ema200", "dist_52w_high", "rsi_14", "atr_20",
                                "open", "high", "low") if c in full.columns]
        before = len(full)
        full = full.dropna(subset=required)
        logger.info("dropped %d/%d warm-up rows", before - len(full), before)

        full = self._cross_sectional(full.sort_values(["date", "symbol"]))

        self.feature_columns = [
            c for c in full.columns
            if c not in ("date", "symbol")
            and not c.startswith("xs_rank_")
            and c not in self.context_columns
        ] + [c for c in full.columns if c.startswith("xs_rank_")]

        self.check_output(full)
        return full.reset_index(drop=True)


__all__ = ["DemoFeatureProvider"]
