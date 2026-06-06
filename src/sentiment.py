"""Context-aware sentiment aggregation (Eqs. 1-4).

Derives per-article sentiment with FinBERT and aggregates daily signals using
confidence weighting, temporal decay, and a dispersion penalty.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def article_sentiment(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map FinBERT class probabilities to a signed score and a confidence.

    probs columns are ordered [positive, negative, neutral].
    Eq. 1:  s = p_pos - p_neg
    Eq. 2:  c = max_k p_k   (confidence)
    """
    s = probs[:, 0] - probs[:, 1]
    c = probs.max(axis=1)
    return s, c


def decay_weights(ages_days: np.ndarray, halflife: float) -> np.ndarray:
    """Temporal-decay weights (Eq. 3): w = 2 ** (-age / halflife)."""
    return np.power(2.0, -np.asarray(ages_days, dtype=float) / halflife)


def aggregate_daily(
    scores: np.ndarray,
    confidences: np.ndarray,
    ages_days: np.ndarray,
    halflife: float,
    dispersion_penalty: float,
) -> float:
    """Aggregate one day's article scores into a single signal (Eq. 4).

    S = (sum_i w_i c_i s_i / sum_i w_i c_i) * (1 - lambda * dispersion)
    where dispersion is the confidence-weighted std of article scores.
    """
    w = decay_weights(ages_days, halflife) * confidences
    if w.sum() <= 0:
        return 0.0
    mean = np.average(scores, weights=w)
    var = np.average((scores - mean) ** 2, weights=w)
    dispersion = np.sqrt(var)
    return float(mean * (1.0 - dispersion_penalty * dispersion))


def build_signal(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """df columns: date, age_days, p_pos, p_neg, p_neu. Returns daily signal."""
    probs = df[["p_pos", "p_neg", "p_neu"]].to_numpy()
    s, c = article_sentiment(probs)
    df = df.assign(_s=s, _c=c)
    hl = cfg["sentiment"]["decay_halflife_days"]
    lam = cfg["sentiment"]["dispersion_penalty"]
    out = df.groupby("date").apply(
        lambda g: aggregate_daily(
            g["_s"].to_numpy(), g["_c"].to_numpy(),
            g["age_days"].to_numpy(), hl, lam,
        )
    )
    return out.rename("sentiment_signal")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    print(f"[sentiment] loaded config {args.config}")
    print("[sentiment] run build_signal() with your FNSPID-derived frame")
