"""Causal backtest engine (Eq. 7).

Positions decided at close of day t are applied to returns of day t+1, net of
transaction costs and slippage. No look-ahead.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

from metrics import performance_summary


def run_backtest(positions: pd.Series, asset_returns: pd.Series,
                 cost_bps: float, slippage_bps: float) -> pd.Series:
    """Eq. 7: strategy return net of costs.

    r_strat_t = pos_{t-1} * r_t - (cost+slip) * |pos_{t-1} - pos_{t-2}|
    """
    lagged = positions.shift(1).fillna(0.0)         # causal: decided yesterday
    turnover = lagged.diff().abs().fillna(0.0)
    cost = (cost_bps + slippage_bps) / 1e4
    gross = lagged * asset_returns
    net = gross - cost * turnover
    return net.rename("strategy_return")


def run(positions: pd.Series, asset_returns: pd.Series, cfg: dict) -> dict:
    b = cfg["backtest"]
    net = run_backtest(positions, asset_returns, b["transaction_cost_bps"],
                       b["slippage_bps"])
    return performance_summary(net)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/metrics_fast.json")
    args = ap.parse_args()
    print(f"[backtest] loaded config {args.config}")
    print(f"[backtest] write summary to {args.out} after supplying data")
