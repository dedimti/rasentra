"""Decision module: position sizing and risk control (Eq. 6)."""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def volatility_target_size(signal: pd.Series, realized_vol: pd.Series,
                           vol_target: float, max_position: float) -> pd.Series:
    """Eq. 6: position scaled to a target volatility, capped at max_position.

    pos_t = clip( sign(S_t) * vol_target / realized_vol_t * |S_t|,
                  -max_position, +max_position )
    """
    raw = np.sign(signal) * (vol_target / realized_vol.replace(0, np.nan)) * signal.abs()
    raw = raw.fillna(0.0)
    return raw.clip(-max_position, max_position)


def apply_stop_loss(positions: pd.Series, returns: pd.Series, atr: pd.Series,
                    atr_mult: float) -> pd.Series:
    """Flatten a position the day after an adverse move beyond atr_mult * ATR."""
    out = positions.copy()
    adverse = (np.sign(positions.shift(1)) * returns) < -(atr_mult * atr)
    out[adverse.fillna(False)] = 0.0
    return out


def enforce_gross_exposure(positions: pd.Series, max_gross: float) -> pd.Series:
    gross = positions.abs()
    scale = np.where(gross > max_gross, max_gross / gross, 1.0)
    return positions * scale


def decide(signal: pd.Series, realized_vol: pd.Series, atr: pd.Series,
           returns: pd.Series, cfg: dict) -> pd.Series:
    d = cfg["decision"]
    pos = volatility_target_size(signal, realized_vol, d["vol_target"], d["max_position"])
    pos = apply_stop_loss(pos, returns, atr, d["stop_loss_atr_mult"])
    pos = enforce_gross_exposure(pos, d["max_gross_exposure"])
    return pos.rename("position")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    print(f"[decision] loaded config {args.config}")
