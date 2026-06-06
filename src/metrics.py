"""Performance metrics: Sharpe, Sortino, Calmar, max drawdown, turnover."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    ex = returns - rf / TRADING_DAYS
    sd = ex.std(ddof=1)
    return float(np.sqrt(TRADING_DAYS) * ex.mean() / sd) if sd > 0 else 0.0


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    ex = returns - rf / TRADING_DAYS
    downside = ex[ex < 0]
    dd = downside.std(ddof=1)
    return float(np.sqrt(TRADING_DAYS) * ex.mean() / dd) if dd > 0 else 0.0


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def calmar(returns: pd.Series) -> float:
    ann_ret = (1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1
    mdd = abs(max_drawdown(returns))
    return float(ann_ret / mdd) if mdd > 0 else 0.0


def performance_summary(returns: pd.Series) -> dict:
    returns = returns.dropna()
    return {
        "n_days": int(len(returns)),
        "ann_return": float((1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "calmar": calmar(returns),
        "max_drawdown": max_drawdown(returns),
    }
