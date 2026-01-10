from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import dataio
from src.backtest import run_backtest
from src.utils import ensure_dir, load_yaml, make_run_id


WINDOWS = {
    "fig8_btc": {"asset": "btc", "t_start": 1610, "t_end": 1760, "price_col": "price_btc"},
    "fig9_gold": {"asset": "gold", "t_start": 460, "t_end": 560, "price_col": "price_gold"},
}


def _build_trade_tag(reason: str, side: str, rebuy: bool) -> str:
    if reason == "extreme_sell":
        tag = "extreme sell"
    elif reason == "sell_signal":
        tag = "normal sell"
    elif reason in {"buy_momentum", "buy_reversion"}:
        tag = "normal buy"
    else:
        tag = reason or "trade"
    if side == "buy" and rebuy:
        tag = "rebuy/no-buy release"
    return tag


def _plot_window(
    window_name: str,
    price_series: pd.DataFrame,
    trades: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(price_series["t"], price_series["price"], color="black", linewidth=1.2, label="price")

    for _, trade in trades.iterrows():
        side = trade["side"]
        tag = trade["trigger_tag"]
        is_blocked = bool(trade.get("is_blocked", False))
        color = "red" if side == "buy" else "blue"
        marker = "^" if side == "buy" else "v"
        if "extreme sell" in tag:
            marker = "*"
            color = "blue"
        if "rebuy" in tag:
            marker = "^"
            color = "green"
        if is_blocked:
            marker = "x"
            color = "gray"

        ax.scatter(trade["t"], trade["price"], color=color, marker=marker, s=60, zorder=3)
        ax.text(
            trade["t"],
            trade["price"],
            trade["tag_abbrev"],
            color=color,
            fontsize=8,
            ha="center",
            va="bottom",
        )

    ax.set_title(window_name)
    ax.set_xlabel("t")
    ax.set_ylabel("price")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config yaml")
    parser.add_argument("--run-id", default=None, help="existing run id under outputs/")
    parser.add_argument("--asset", default=None, choices=["gold", "btc"], help="asset filter")
    parser.add_argument("--t0", type=int, default=None, help="window start t")
    parser.add_argument("--t1", type=int, default=None, help="window end t")
    parser.add_argument("--date0", default=None, help="window start date (YYYY-MM-DD)")
    parser.add_argument("--date1", default=None, help="window end date (YYYY-MM-DD)")
    args = parser.parse_args()

    config = load_yaml(args.config)
    config["_config_path"] = args.config

    prices = dataio.load_price_data(config)
    results_df, trades_df, _w_stats, debug_df, state_events_df = run_backtest(
        prices, config, return_debug=True
    )

    run_id = args.run_id or config["run"].get("run_id") or make_run_id()
    out_dir = ensure_dir(Path("outputs") / run_id / "debug")

    index_mapping_path = out_dir / "index_mapping.csv"
    results_df[["t", "date"]].drop_duplicates().sort_values("t").to_csv(
        index_mapping_path, index=False
    )

    if not state_events_df.empty:
        state_events_path = out_dir / "state_events.csv"
        state_events_df.to_csv(state_events_path, index=False)

        btc_no_buy = state_events_df[
            (state_events_df["state_type"] == "no_buy")
            & (state_events_df["asset"] == "btc")
            & (state_events_df["now"] == True)
        ].sort_values("t")
        if not btc_no_buy.empty:
            first_row = btc_no_buy.iloc[0]
            summary = (
                f"first_no_buy_on_btc = (t={int(first_row['t'])}, "
                f"date={first_row['date']}, reason={first_row['reason']})"
            )
        else:
            summary = "first_no_buy_on_btc = None"
    else:
        summary = "first_no_buy_on_btc = None"

    summary_path = out_dir / "first_no_buy_on_btc.txt"
    summary_path.write_text(summary, encoding="utf-8")

    windows = []
    if args.date0 and args.date1:
        date0 = pd.to_datetime(args.date0)
        date1 = pd.to_datetime(args.date1)
        assets = [args.asset] if args.asset else ["gold", "btc"]
        for asset in assets:
            price_col = "price_gold" if asset == "gold" else "price_btc"
            window_df = results_df[(results_df["date"] >= date0) & (results_df["date"] <= date1)]
            if window_df.empty:
                continue
            t_start = int(window_df["t"].min())
            t_end = int(window_df["t"].max())
            date0_str = date0.date().isoformat()
            date1_str = date1.date().isoformat()
            window_name = f"{asset}_{date0_str}_{date1_str}"
            windows.append(
                {
                    "name": window_name,
                    "asset": asset,
                    "t_start": t_start,
                    "t_end": t_end,
                    "price_col": price_col,
                }
            )
    elif args.t0 is not None and args.t1 is not None:
        assets = [args.asset] if args.asset else ["gold", "btc"]
        for asset in assets:
            price_col = "price_gold" if asset == "gold" else "price_btc"
            window_name = f"{asset}_t{args.t0}_{args.t1}"
            windows.append(
                {
                    "name": window_name,
                    "asset": asset,
                    "t_start": args.t0,
                    "t_end": args.t1,
                    "price_col": price_col,
                }
            )
    else:
        for window_name, info in WINDOWS.items():
            windows.append({"name": window_name, **info})

    for window in windows:
        asset = window["asset"]
        t_start = window["t_start"]
        t_end = window["t_end"]
        price_col = window["price_col"]
        window_name = window["name"]

        debug_window = debug_df[
            (debug_df["asset"] == asset)
            & (debug_df["t"] >= t_start)
            & (debug_df["t"] <= t_end)
        ].copy()
        debug_window = debug_window[
            [
                "t",
                "date",
                "price",
                "E",
                "threshold",
                "no_buy_state",
                "extreme_state",
                "cooldown_remaining",
                "skipped_extreme_due_to_history",
                "target_position",
                "action",
                "action_reason",
            ]
        ]
        debug_window_path = out_dir / f"{window_name}_window_debug.csv"
        debug_window.to_csv(debug_window_path, index=False)

        rebuy_days = set(
            debug_df[
                (debug_df["asset"] == asset)
                & (debug_df["no_buy_released"])
                & (debug_df["t"] >= t_start)
                & (debug_df["t"] <= t_end)
            ]["t"].tolist()
        )

        trades_window = trades_df[
            (trades_df["asset"] == asset)
            & (trades_df["t"] >= t_start)
            & (trades_df["t"] <= t_end)
        ].copy()
        if not trades_window.empty:
            trades_window["is_blocked"] = trades_window["qty"] <= 0
        else:
            trades_window["is_blocked"] = []

        if not trades_window.empty:
            trades_window["trigger_tag"] = trades_window.apply(
                lambda row: _build_trade_tag(
                    str(row.get("reason", "")),
                    str(row.get("side", "")),
                    row.get("t") in rebuy_days,
                ),
                axis=1,
            )
            trades_window.loc[trades_window["is_blocked"], "trigger_tag"] = trades_window.loc[
                trades_window["is_blocked"], "side"
            ].map({"buy": "blocked buy", "sell": "blocked sell"})
            trades_window["tag_abbrev"] = trades_window["trigger_tag"].map(
                {
                    "normal buy": "B",
                    "normal sell": "S",
                    "extreme sell": "E",
                    "rebuy/no-buy release": "R",
                    "blocked buy": "Bb",
                    "blocked sell": "Sb",
                }
            ).fillna("T")
        else:
            trades_window["trigger_tag"] = []
            trades_window["tag_abbrev"] = []

        trades_out = trades_window[
            ["t", "date", "asset", "side", "qty", "price", "reason", "trigger_tag", "is_blocked"]
        ].rename(columns={"qty": "size"})
        trades_window_path = out_dir / f"{window_name}_window_trades.csv"
        trades_out.to_csv(trades_window_path, index=False)

        price_window = results_df[
            (results_df["t"] >= t_start) & (results_df["t"] <= t_end)
        ][["t", price_col]].rename(columns={price_col: "price"})
        plot_path = out_dir / f"{window_name}_window_plot.png"
        _plot_window(window_name, price_window, trades_window, plot_path)

    print(summary)
    print(f"debug outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
