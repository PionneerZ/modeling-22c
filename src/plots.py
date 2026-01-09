from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_nav(results_df, out_path: str | Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(results_df["date"], results_df["nav_total"], label="total")
    plt.plot(results_df["date"], results_df["nav_cash"], label="cash")
    plt.plot(results_df["date"], results_df["nav_gold"], label="gold")
    plt.plot(results_df["date"], results_df["nav_btc"], label="btc")
    plt.legend()
    plt.title("NAV: total / cash / gold / btc")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_drawdown(results_df, out_path: str | Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(results_df["date"], results_df["dd"], label="drawdown")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Drawdown")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_positions(results_df, out_path: str | Path) -> None:
    total = results_df["nav_total"].replace(0, 1e-9)
    cash_pct = results_df["nav_cash"] / total
    gold_pct = results_df["nav_gold"] / total
    btc_pct = results_df["nav_btc"] / total

    plt.figure(figsize=(10, 4))
    plt.plot(results_df["date"], cash_pct, label="cash")
    plt.plot(results_df["date"], gold_pct, label="gold")
    plt.plot(results_df["date"], btc_pct, label="btc")
    plt.legend()
    plt.title("Position Weights")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_trade_window(results_df, trades_df, config: dict, out_path: str | Path) -> None:
    plot_cfg = config.get("plots", {})
    btc_window = plot_cfg.get("window_btc_t")
    gold_window = plot_cfg.get("window_gold_t")
    if not btc_window and not gold_window:
        return
    if "t" not in results_df.columns:
        return
    if trades_df is None or trades_df.empty:
        return
    if "t" not in trades_df.columns:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    def _plot_asset(ax, asset: str, window):
        if not window:
            ax.set_visible(False)
            return
        t_start, t_end = window
        df_window = results_df[(results_df["t"] >= t_start) & (results_df["t"] <= t_end)]
        if df_window.empty:
            ax.set_title(f"{asset} window empty")
            return
        ax.plot(df_window["t"], df_window[f"price_{asset}"], color="black", linewidth=1)
        trades_asset = trades_df[(trades_df["asset"] == asset) & (trades_df["qty"] > 0)]
        trades_asset = trades_asset[(trades_asset["t"] >= t_start) & (trades_asset["t"] <= t_end)]
        buys = trades_asset[trades_asset["side"] == "buy"]
        sells = trades_asset[trades_asset["side"] == "sell"]
        ax.scatter(buys["t"], buys["price"], color="red", s=18, label="buy")
        ax.scatter(sells["t"], sells["price"], color="blue", s=18, label="sell")
        ax.set_title(f"{asset.upper()} trades window t={t_start}..{t_end}")
        ax.legend()

    _plot_asset(axes[0], "btc", btc_window)
    _plot_asset(axes[1], "gold", gold_window)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
