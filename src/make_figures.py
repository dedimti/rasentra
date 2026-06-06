"""Reproduce paper figures from stored results (results/metrics_fast.json).

The results file holds one entry per strategy (RaSenTra and baselines) plus an
optional significance-test block prefixed with an underscore.
"""
from __future__ import annotations

import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_strategies(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def grouped_metrics(strategies: dict, out_dir: str) -> None:
    """Sharpe / Sortino / Calmar per strategy."""
    metrics = ["sharpe", "sortino", "calmar"]
    names = list(strategies.keys())
    x = range(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(metrics):
        vals = [strategies[n].get(m, 0.0) for n in names]
        ax.bar([p + i * width for p in x], vals, width, label=m.capitalize())
    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels(names, rotation=15)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Value")
    ax.set_title("Risk-adjusted performance across strategies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_risk_adjusted.png"), dpi=200)
    plt.close(fig)


def return_vs_risk(strategies: dict, out_dir: str) -> None:
    """Annual return vs max drawdown scatter."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for n, v in strategies.items():
        ax.scatter(v.get("max_drawdown", 0.0) * 100,
                   v.get("annual_return", 0.0) * 100, s=80)
        ax.annotate(n, (v.get("max_drawdown", 0.0) * 100,
                        v.get("annual_return", 0.0) * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Max drawdown (%)")
    ax.set_ylabel("Annual return (%)")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_title("Return vs. downside risk")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_return_vs_risk.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    strategies = load_strategies(args.results)
    grouped_metrics(strategies, args.out)
    return_vs_risk(strategies, args.out)
    print(f"[make_figures] {len(strategies)} strategies plotted -> {args.out}/")


if __name__ == "__main__":
    main()
