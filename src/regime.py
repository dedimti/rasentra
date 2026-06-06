"""Market-regime detection (Eq. 5).

Fits a Gaussian HMM to market features and reweights the sentiment signal by the
trust assigned to the inferred regime.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover
    GaussianHMM = None


def fit_regimes(features: np.ndarray, n_states: int, covariance_type: str,
                seed: int = 42) -> "GaussianHMM":
    if GaussianHMM is None:
        raise ImportError("hmmlearn is required: pip install hmmlearn")
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=200,
        random_state=seed,
    )
    model.fit(features)
    return model


def order_states_by_vol(model: "GaussianHMM", vol_idx: int) -> np.ndarray:
    """Return a permutation that orders states from low to high volatility,
    so regime indices map consistently to weight vectors."""
    means_vol = model.means_[:, vol_idx]
    return np.argsort(means_vol)


def modulate_signal(signal: pd.Series, states: np.ndarray,
                    regime_weights: list[float]) -> pd.Series:
    """Eq. 5:  S_mod_t = w(regime_t) * S_t."""
    weights = np.asarray(regime_weights, dtype=float)[states]
    return signal * pd.Series(weights, index=signal.index)


def run(features: np.ndarray, signal: pd.Series, cfg: dict) -> pd.Series:
    rcfg = cfg["regime"]
    model = fit_regimes(features, rcfg["n_states"], rcfg["covariance_type"],
                        cfg["backtest"]["seed"])
    raw_states = model.predict(features)
    order = order_states_by_vol(model, vol_idx=1)
    remap = {old: new for new, old in enumerate(order)}
    states = np.array([remap[s] for s in raw_states])
    return modulate_signal(signal, states, rcfg["regime_weights"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    print(f"[regime] loaded config {args.config}")
