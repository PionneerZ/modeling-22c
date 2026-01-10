from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
    btc_window_t = plot_cfg.get("window_btc_t")
    gold_window_t = plot_cfg.get("window_gold_t")
    btc_window_date = plot_cfg.get("window_btc_date")
    gold_window_date = plot_cfg.get("window_gold_date")
    if not any([btc_window_t, gold_window_t, btc_window_date, gold_window_date]):
        return
    if trades_df is None or trades_df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    def _normalize_dates(series):
        return pd.to_datetime(series, errors="coerce")

    def _plot_asset(ax, asset: str, window_t, window_date):
        if not window_t and not window_date:
            ax.set_visible(False)
            return

        df = results_df.copy()
        if "date" in df.columns:
            df["date"] = _normalize_dates(df["date"])
        trades = trades_df.copy()
        if "date" in trades.columns:
            trades["date"] = _normalize_dates(trades["date"])

        x_col = "t" if "t" in df.columns else "date"
        mode = "t"
        if window_t:
            t_start, t_end = window_t
            df_window = df[(df["t"] >= t_start) & (df["t"] <= t_end)] if "t" in df.columns else df
            trades_asset = trades[(trades["asset"] == asset) & (trades["qty"] > 0)]
            if "t" in trades.columns:
                trades_asset = trades_asset[
                    (trades_asset["t"] >= t_start) & (trades_asset["t"] <= t_end)
                ]
        else:
            mode = "date"
            start, end = pd.to_datetime(window_date[0]), pd.to_datetime(window_date[1])
            df_window = df[(df["date"] >= start) & (df["date"] <= end)]
            trades_asset = trades[(trades["asset"] == asset) & (trades["qty"] > 0)]
            trades_asset = trades_asset[(trades_asset["date"] >= start) & (trades_asset["date"] <= end)]
            if "t" in df_window.columns and not df_window.empty:
                t_start = int(df_window["t"].iloc[0])
                t_end = int(df_window["t"].iloc[-1])
            else:
                t_start = t_end = None

        if df_window.empty:
            ax.set_title(f"{asset} window empty")
            return

        ax.plot(df_window[x_col], df_window[f"price_{asset}"], color="black", linewidth=1)
        buys = trades_asset[trades_asset["side"] == "buy"]
        sells = trades_asset[trades_asset["side"] == "sell"]
        ax.scatter(buys[x_col], buys["price"], color="red", s=18, label="buy")
        ax.scatter(sells[x_col], sells["price"], color="blue", s=18, label="sell")
        if mode == "date":
            title_range = f"{window_date[0]}..{window_date[1]}"
        else:
            title_range = f"t={t_start}..{t_end}"
        ax.set_title(f"{asset.upper()} trades window {title_range}")
        ax.legend()

    _plot_asset(axes[0], "btc", btc_window_t, btc_window_date)
    _plot_asset(axes[1], "gold", gold_window_t, gold_window_date)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
