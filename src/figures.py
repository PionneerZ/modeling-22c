from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Polygon

from src.backtest import run_backtest


@dataclass
class FigureContext:
    prices: pd.DataFrame
    results: pd.DataFrame
    trades: pd.DataFrame
    config: Dict
    tables_config: Dict | None = None


def _t_axis(df: pd.DataFrame) -> np.ndarray:
    if "t" in df.columns:
        return df["t"].to_numpy()
    return np.arange(len(df), dtype=float)


def plot_fig1(context: FigureContext, out_path: str | Path) -> None:
    prices = context.prices
    t = _t_axis(prices)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(t, prices["price_btc"], color="black", linewidth=1)
    axes[0].set_title("Bitcoin")
    axes[0].set_xlabel("Days")
    axes[0].set_ylabel("USD")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(t, prices["price_gold"], color="red", linewidth=1)
    axes[1].set_title("Gold")
    axes[1].set_xlabel("Days")
    axes[1].set_ylabel("USD")
    axes[1].grid(True, alpha=0.2)

    fig.suptitle("Figure 1: Plots of bitcoin and gold", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig2(context: FigureContext, out_path: str | Path) -> None:
    prices = context.prices
    t = _t_axis(prices)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, prices["price_btc"], color="black", linewidth=1, label="Bitcoin")
    ax.plot(t, prices["price_gold"], color="red", linewidth=1, label="Gold")
    ax.set_title("Figure 2: Bitcoin Price from 9/11/2016 to 9/10/2021")
    ax.set_xlabel("Days")
    ax.set_ylabel("USD")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _add_box(ax, xy, width, height, text) -> None:
    x, y = xy
    rect = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=9)


def _add_diamond(ax, xy, width, height, text) -> None:
    x, y = xy
    verts = [
        (x, y + height / 2),
        (x + width / 2, y),
        (x, y - height / 2),
        (x - width / 2, y),
    ]
    poly = Polygon(verts, closed=True, edgecolor="black", facecolor="white", linewidth=1)
    ax.add_patch(poly)
    ax.text(x, y, text, ha="center", va="center", fontsize=9)


def plot_fig3(context: FigureContext, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _add_diamond(ax, (0.2, 0.7), 0.22, 0.18, "Above Moving\nAverage?")
    _add_diamond(ax, (0.6, 0.7), 0.22, 0.18, "Positive\nSlope?")
    _add_box(ax, (0.2, 0.35), 0.18, 0.12, "Buy")
    _add_box(ax, (0.85, 0.75), 0.2, 0.12, "Don't Buy")
    _add_box(ax, (0.85, 0.55), 0.2, 0.12, "Buy")

    ax.annotate("", xy=(0.31, 0.7), xytext=(0.49, 0.7), arrowprops={"arrowstyle": "-|>"})
    ax.annotate("", xy=(0.2, 0.61), xytext=(0.2, 0.41), arrowprops={"arrowstyle": "-|>"})
    ax.annotate("", xy=(0.71, 0.7), xytext=(0.75, 0.75), arrowprops={"arrowstyle": "-|>"})
    ax.annotate("", xy=(0.71, 0.7), xytext=(0.75, 0.55), arrowprops={"arrowstyle": "-|>"})

    ax.text(0.4, 0.73, "Yes", fontsize=8)
    ax.text(0.17, 0.5, "No", fontsize=8)
    ax.text(0.73, 0.78, "No", fontsize=8)
    ax.text(0.73, 0.61, "Yes", fontsize=8)

    fig.suptitle("Figure 3: Flow Chart Describing our buy logic", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig4(context: FigureContext, out_path: str | Path) -> None:
    results = context.results
    t_vals = _t_axis(results)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_vals, results["nav_total"], label="Total", color="blue", linewidth=1)
    ax.plot(t_vals, results["nav_btc"], label="Bitcoin", color="black", linewidth=1)
    ax.plot(t_vals, results["nav_gold"], label="Gold", color="red", linewidth=1)
    ax.plot(t_vals, results["nav_cash"], label="Cash", color="green", linewidth=1)
    ax.set_title("Figure 4: Our results")
    ax.set_xlabel("Days")
    ax.set_ylabel("USD")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _apply_weight_to_scores(config: Dict, t_values: np.ndarray, scores: np.ndarray, asset: str) -> np.ndarray:
    weight_cfg = config.get("weight_factor", {})
    apply_mode = str(weight_cfg.get("apply_mode", "none")).lower()
    if apply_mode == "none":
        return scores
    w_use = [str(v).lower() for v in weight_cfg.get("W_use", [])]
    if "all" not in w_use and asset not in w_use and f"profit_{asset}" not in w_use:
        return scores
    w_c = float(weight_cfg.get("W_C", 0.0))
    use_t_plus_one = bool(weight_cfg.get("use_t_plus_one", True))
    t_for_weight = t_values + 1.0 if use_t_plus_one else np.maximum(t_values, 1.0)
    weight = w_c / (t_for_weight**2)
    return scores * weight


def plot_fig5(context: FigureContext, out_path: str | Path) -> None:
    results = context.results
    config = context.config
    t_values = _t_axis(results).astype(float)

    gold_scores = results["profit_gold"].to_numpy(dtype=float)
    btc_scores = results["profit_btc"].to_numpy(dtype=float)

    gold_weighted = _apply_weight_to_scores(config, t_values, gold_scores, "gold")
    btc_weighted = _apply_weight_to_scores(config, t_values, btc_scores, "btc")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_values, gold_weighted, color="blue", linewidth=1, label="Gold signal")
    ax.plot(t_values, btc_weighted, color="orange", linewidth=1, label="Bitcoin signal")
    ax.set_title("Figure 5: The weight factor normalizing the magnitude of the signals")
    ax.set_xlabel("Days")
    ax.set_ylabel("Profitability (unitless)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _config_with_params(base_cfg: Dict, t_val: int, n_val: float, e_val: float) -> Dict:
    run_cfg = copy.deepcopy(base_cfg)
    run_cfg["paper_params"] = {
        **base_cfg.get("paper_params", {}),
        "hold_T": t_val,
        "reentry_N": n_val,
        "extreme_E": e_val,
    }
    run_cfg["holding"] = {**base_cfg.get("holding", {}), "min_days_between_sells": t_val}
    run_cfg["extreme"] = {**base_cfg.get("extreme", {}), "extreme_sell_pct_E": e_val}
    return run_cfg


def compute_param_grid_runs(prices: pd.DataFrame, base_cfg: Dict, grid: Dict) -> list[dict]:
    runs = []
    idx = 1
    for t_val in grid["T"]:
        for n_val in grid["N"]:
            for e_val in grid["E"]:
                run_cfg = _config_with_params(base_cfg, int(t_val), float(n_val), float(e_val))
                run_cfg["fees"] = {
                    **base_cfg.get("fees", {}),
                    "fee_gold": float(grid["fee_gold"]),
                    "fee_btc": float(grid["fee_btc"]),
                }
                results_df, _trades_df, _ = run_backtest(prices, run_cfg)
                final_nav = float(results_df["nav_total"].iloc[-1]) if len(results_df) else 0.0
                runs.append(
                    {
                        "index": idx,
                        "params": [t_val, n_val, e_val],
                        "t": _t_axis(results_df),
                        "nav_total": results_df["nav_total"].to_numpy(),
                        "final_nav": final_nav,
                    }
                )
                idx += 1
    return runs


def plot_fig6(context: FigureContext, out_path: str | Path) -> list[dict]:
    tables_cfg = context.tables_config or {}
    grid = tables_cfg.get("param_grid", {})
    if not grid:
        raise ValueError("tables_config.param_grid is required for Figure 6")

    base_cfg = copy.deepcopy(context.config)
    base_cfg["strategy"]["mode"] = "full"

    runs = compute_param_grid_runs(context.prices, base_cfg, grid)

    fig, ax = plt.subplots(figsize=(7, 4))
    for run in runs:
        t_vals = run["t"]
        ax.plot(
            t_vals,
            run["nav_total"],
            linewidth=0.8,
            label=f"{run['index']}: {run['params']}",
        )

    ax.set_title("Figure 6: Results under different combinations")
    ax.set_xlabel("Days")
    ax.set_ylabel("USD")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=6, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return runs


def compute_fee_sensitivity_runs(prices: pd.DataFrame, base_cfg: Dict, fee_cfg: Dict) -> list[dict]:
    runs = []
    for idx, (fee_gold, fee_btc) in enumerate(fee_cfg["fee_pairs"], start=1):
        run_cfg = copy.deepcopy(base_cfg)
        run_cfg["fees"] = {**base_cfg.get("fees", {}), "fee_gold": fee_gold, "fee_btc": fee_btc}
        results_df, _trades_df, _ = run_backtest(prices, run_cfg)
        final_nav = float(results_df["nav_total"].iloc[-1]) if len(results_df) else 0.0
        runs.append(
            {
                "index": idx,
                "fees": [fee_gold, fee_btc],
                "t": _t_axis(results_df),
                "nav_total": results_df["nav_total"].to_numpy(),
                "final_nav": final_nav,
            }
        )
    return runs


def plot_fig7(context: FigureContext, out_path: str | Path) -> list[dict]:
    tables_cfg = context.tables_config or {}
    fee_cfg = tables_cfg.get("fee_sensitivity", {})
    if not fee_cfg:
        raise ValueError("tables_config.fee_sensitivity is required for Figure 7")

    params = fee_cfg["params"]
    base_cfg = copy.deepcopy(context.config)
    base_cfg["strategy"]["mode"] = "full"
    base_cfg = _config_with_params(base_cfg, int(params["T"]), float(params["N"]), float(params["E"]))

    runs = compute_fee_sensitivity_runs(context.prices, base_cfg, fee_cfg)

    fig, ax = plt.subplots(figsize=(7, 4))
    for run in runs:
        ax.plot(
            run["t"],
            run["nav_total"],
            linewidth=1.0,
            label=f"{run['index']}: {run['fees']}",
        )

    ax.set_title("Figure 7: Results under different combinations")
    ax.set_xlabel("Days")
    ax.set_ylabel("USD")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return runs


FIGURE_REGISTRY = {
    1: plot_fig1,
    2: plot_fig2,
    3: plot_fig3,
    4: plot_fig4,
    5: plot_fig5,
    6: plot_fig6,
    7: plot_fig7,
}


def supported_figures() -> Iterable[int]:
    return sorted(FIGURE_REGISTRY.keys())
