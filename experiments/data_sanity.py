"""Diagnostics: print data coverage and anchor points for a config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import dataio
from src.utils import load_yaml


ANCHOR_T = [0, 100, 460, 560, 1610, 1760]


def _print_series_summary(df: pd.DataFrame, name: str, col: str) -> None:
    series = df[col]
    print(f"{name} start_date={df['date'].iloc[0].date()} end_date={df['date'].iloc[-1].date()} n_rows={len(df)}")
    print(f"{name} min={series.min(skipna=True)} max={series.max(skipna=True)}")


def _print_anchor_points(df: pd.DataFrame, name: str, col: str) -> None:
    print(f"{name} anchor points ({col}):")
    for t in ANCHOR_T:
        row = df[df["t"] == t]
        if row.empty:
            print(f"  t={t}: missing")
            continue
        rec = row.iloc[0]
        date = rec["date"].date()
        price = rec[col]
        print(f"  t={t} date={date} price={price}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    config["_config_path"] = args.config

    df = dataio.load_price_data(config)
    data_info = df.attrs.get("data_info", {})
    source = data_info.get("source", {})

    print("source_info:", source)
    print("range:", data_info.get("raw_min"), "to", data_info.get("raw_max"))
    print("start_date:", data_info.get("start_date"), "end_date:", data_info.get("end_date"))
    print("trade_end:", data_info.get("trade_end"))
    print("gold_missing_trading:", data_info.get("gold_missing_trading"))
    print()

    _print_series_summary(df, "btc", "price_btc")
    _print_series_summary(df, "gold", "price_gold")
    if "price_gold_trade" in df.columns:
        _print_series_summary(df, "gold_trade", "price_gold_trade")
    print()

    _print_anchor_points(df, "btc", "price_btc")
    _print_anchor_points(df, "gold", "price_gold")
    if "price_gold_trade" in df.columns:
        _print_anchor_points(df, "gold_trade", "price_gold_trade")


if __name__ == "__main__":
    main()
