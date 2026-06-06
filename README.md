# RaSenTra: Regime-Aware Sentiment Trading Decision Support System

This repository contains the source code and experimental results for the paper
*"RaSenTra: A Regime-Aware Sentiment-Driven Decision Support System for
Risk-Controlled Equity Trading"*, submitted to **Expert Systems With Applications**.

> **Note on anonymity.** This repository is shared anonymously for double-blind
> peer review. Author names, affiliations, and identifying information have been
> removed and will be added after acceptance.

## Overview

RaSenTra converts financial news sentiment into risk-controlled equity trading
decisions through three integrated modules:

1. **Context-aware sentiment module** (`src/sentiment.py`) — derives sentiment
   from financial news with FinBERT and aggregates it using confidence,
   dispersion, and temporal-decay weighting (Eqs. 1–4).
2. **Market-regime detection module** (`src/regime.py`) — fits a Gaussian HMM to
   market features and adaptively reweights the sentiment signal per regime (Eq. 5).
3. **Decision module** (`src/decision.py`) — maps the modulated signal to position
   sizing, adaptive stop-loss, and exposure limits (Eq. 6).

A causal backtest (`src/backtest.py`) evaluates the strategy net of transaction
costs and slippage, reporting Sharpe, Sortino, Calmar, maximum drawdown, and
turnover (Eq. 7).

## Repository structure

```
rasentra/
├── README.md
├── requirements.txt
├── LICENSE
├── configs/
│   └── default.yaml         # experiment hyperparameters
├── src/
│   ├── sentiment.py         # context-aware sentiment aggregation (Eqs. 1–4)
│   ├── regime.py            # HMM-based regime detection (Eq. 5)
│   ├── decision.py          # position sizing & risk control (Eq. 6)
│   ├── backtest.py          # causal backtest engine (Eq. 7)
│   ├── metrics.py           # performance metrics
│   └── make_figures.py      # reproduce paper figures from results
├── figures/                 # generated figures
└── results/                 # metrics_fast.json, ablation outputs
```

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10+.

## Data

This study uses the **FNSPID** dataset (publicly available; Dong et al., 2024),
which couples S&P 500 financial news with aligned price data. Price OHLCV data
are obtained via `yfinance`. The dataset is not redistributed here; see the paper
for retrieval instructions.

## Usage

```bash
# 1. Compute aggregated sentiment signals
python src/sentiment.py --config configs/default.yaml

# 2. Detect market regimes
python src/regime.py --config configs/default.yaml

# 3. Run the decision module + causal backtest
python src/backtest.py --config configs/default.yaml

# 4. Reproduce paper figures from stored results
python src/make_figures.py --results results/metrics_fast.json --out figures/
```

## Reproducing paper results

The numerical results reported in the paper are stored in
`results/metrics_fast.json`, which contains performance metrics for RaSenTra and
three baselines (BuyHold, PriceOnly, SentimentOnly) over the 501-day evaluation
window, plus a paired significance test of RaSenTra against the sentiment-only
ablation. Running `src/make_figures.py` regenerates all figures directly from
this file without re-executing the full backtest:

```bash
python src/make_figures.py --results results/metrics_fast.json --out figures/
```

## License

Released under the MIT License (see `LICENSE`).
