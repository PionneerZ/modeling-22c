from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _load_run(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(run_dir / "results_table.csv")
    trades = pd.read_csv(run_dir / "trades.csv")
    events_path = run_dir / "debug" / "state_events.csv"
    if events_path.exists():
        events = pd.read_csv(events_path)
    else:
        events = pd.DataFrame()
    return results, trades, events


def _normalize_results(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("t").reset_index(drop=True)


def _safe_log(series: pd.Series) -> pd.Series:
    values = series.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logged = np.where(values > 0, np.log(values), np.nan)
    return pd.Series(logged, index=series.index)


def _nav_diff_timeseries(
    res_a: pd.DataFrame, res_b: pd.DataFrame
) -> pd.DataFrame:
    cols = [
        "t",
        "date",
        "nav_total",
        "nav_cash",
        "nav_gold",
        "nav_btc",
        "gold_qty",
        "btc_qty",
        "price_gold",
        "price_btc",
    ]
    a = res_a[cols].copy()
    b = res_b[cols].copy()
    merged = pd.merge(a, b, on=["t", "date"], how="outer", suffixes=("_a", "_b"))
    merged = merged.sort_values("t").reset_index(drop=True)

    merged["log_nav_a"] = _safe_log(merged["nav_total_a"])
    merged["log_nav_b"] = _safe_log(merged["nav_total_b"])
    merged["delta_log"] = merged["log_nav_a"] - merged["log_nav_b"]
    merged["delta_nav"] = merged["nav_total_a"] - merged["nav_total_b"]
    return merged


def _top_diverge_days(nav_diff: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    df = nav_diff.copy()
    df["nav_a_change"] = df["nav_total_a"] - df["nav_total_a"].shift(1)
    df["nav_b_change"] = df["nav_total_b"] - df["nav_total_b"].shift(1)
    df["delta_nav_change"] = df["nav_a_change"] - df["nav_b_change"]
    df["delta_log_abs"] = df["delta_log"].abs()
    top = df.sort_values("delta_log_abs", ascending=False).head(top_n)
    return top[
        [
            "t",
            "date",
            "nav_total_a",
            "nav_total_b",
            "delta_nav",
            "log_nav_a",
            "log_nav_b",
            "delta_log",
            "nav_a_change",
            "nav_b_change",
            "delta_nav_change",
        ]
    ]


def _pnl_decompose(results: pd.DataFrame) -> pd.DataFrame:
    res = results.sort_values("t").reset_index(drop=True)
    res["nav_change"] = res["nav_total"] - res["nav_total"].shift(1)
    res["cash_change"] = res["cash"] - res["cash"].shift(1)
    res["hold_pnl_btc"] = res["btc_qty"].shift(1) * (
        res["price_btc"] - res["price_btc"].shift(1)
    )
    res["hold_pnl_gold"] = res["gold_qty"].shift(1) * (
        res["price_gold"] - res["price_gold"].shift(1)
    )
    res["trade_pnl"] = res["nav_change"] - res["hold_pnl_btc"] - res["hold_pnl_gold"]
    return res[
        [
            "t",
            "date",
            "nav_total",
            "nav_change",
            "cash_change",
            "hold_pnl_btc",
            "hold_pnl_gold",
            "trade_pnl",
        ]
    ]


def _summarize_events(trades: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    trades["date"] = pd.to_datetime(trades["date"])
    live = trades[trades["qty"] > 0]

    def _count_by(df: pd.DataFrame, name: str) -> pd.DataFrame:
        counts = df.groupby(["t", "date"]).size().reset_index(name=name)
        return counts

    buys = _count_by(live[live["side"] == "buy"], "buy_count")
    sells = _count_by(live[live["side"] == "sell"], "sell_count")
    extreme_sells = _count_by(live[live["reason"] == "extreme_sell"], "extreme_sell_count")

    if not events.empty and {"state_type", "now"}.issubset(events.columns):
        events = events.copy()
        events["date"] = pd.to_datetime(events["date"])
        no_buy_on = events[(events["state_type"] == "no_buy") & (events["now"] == True)]
        no_buy_off = events[(events["state_type"] == "no_buy") & (events["now"] == False)]
        no_buy_on = _count_by(no_buy_on, "no_buy_on")
        no_buy_off = _count_by(no_buy_off, "no_buy_off")
    else:
        no_buy_on = pd.DataFrame(columns=["t", "date", "no_buy_on"])
        no_buy_off = pd.DataFrame(columns=["t", "date", "no_buy_off"])

    merged = buys
    for df in [sells, extreme_sells, no_buy_on, no_buy_off]:
        merged = pd.merge(merged, df, on=["t", "date"], how="outer")

    merged = merged.fillna(0)
    merged["rebuy_on_release"] = (
        (merged.get("no_buy_off", 0) > 0) & (merged.get("buy_count", 0) > 0)
    ).astype(int)
    return merged


def _merge_event_overlay(
    base_index: pd.DataFrame,
    events_a: pd.DataFrame,
    events_b: pd.DataFrame,
) -> pd.DataFrame:
    base = base_index.copy()
    base = pd.merge(base, events_a, on=["t", "date"], how="left", suffixes=("", "_a"))
    base = base.rename(
        columns={
            "buy_count": "buy_count_a",
            "sell_count": "sell_count_a",
            "extreme_sell_count": "extreme_sell_count_a",
            "no_buy_on": "no_buy_on_a",
            "no_buy_off": "no_buy_off_a",
            "rebuy_on_release": "rebuy_on_release_a",
        }
    )
    base = pd.merge(base, events_b, on=["t", "date"], how="left", suffixes=("", "_b"))
    base = base.rename(
        columns={
            "buy_count": "buy_count_b",
            "sell_count": "sell_count_b",
            "extreme_sell_count": "extreme_sell_count_b",
            "no_buy_on": "no_buy_on_b",
            "no_buy_off": "no_buy_off_b",
            "rebuy_on_release": "rebuy_on_release_b",
        }
    )
    for col in [
        "buy_count_a",
        "sell_count_a",
        "extreme_sell_count_a",
        "no_buy_on_a",
        "no_buy_off_a",
        "rebuy_on_release_a",
        "buy_count_b",
        "sell_count_b",
        "extreme_sell_count_b",
        "no_buy_on_b",
        "no_buy_off_b",
        "rebuy_on_release_b",
    ]:
        if col in base.columns:
            base[col] = base[col].fillna(0).astype(int)
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True, help="path to run A dir under outputs/")
    parser.add_argument("--run-b", required=True, help="path to run B dir under outputs/")
    parser.add_argument(
        "--outdir",
        default="outputs/run_fixpack",
        help="output run dir (default: outputs/run_fixpack)",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="optional debug output directory (overrides outdir/debug)",
    )
    args = parser.parse_args()

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
    else:
        out_dir = Path(args.outdir)
        debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    res_a, trades_a, events_a = _load_run(run_a)
    res_b, trades_b, events_b = _load_run(run_b)
    res_a = _normalize_results(res_a)
    res_b = _normalize_results(res_b)

    nav_diff = _nav_diff_timeseries(res_a, res_b)
    nav_diff.to_csv(debug_dir / "nav_diff_timeseries.csv", index=False)

    top_diverge = _top_diverge_days(nav_diff)
    top_diverge.to_csv(debug_dir / "top_diverge_days.csv", index=False)

    pnl_a = _pnl_decompose(res_a).rename(columns=lambda c: f"{c}_a" if c not in {"t", "date"} else c)
    pnl_b = _pnl_decompose(res_b).rename(columns=lambda c: f"{c}_b" if c not in {"t", "date"} else c)
    pnl = pd.merge(pnl_a, pnl_b, on=["t", "date"], how="outer").sort_values("t")
    pnl["delta_nav_change"] = pnl["nav_change_a"] - pnl["nav_change_b"]
    pnl["delta_hold_pnl_btc"] = pnl["hold_pnl_btc_a"] - pnl["hold_pnl_btc_b"]
    pnl["delta_hold_pnl_gold"] = pnl["hold_pnl_gold_a"] - pnl["hold_pnl_gold_b"]
    pnl["delta_trade_pnl"] = pnl["trade_pnl_a"] - pnl["trade_pnl_b"]
    pnl.to_csv(debug_dir / "pnl_decompose.csv", index=False)

    events_a_summary = _summarize_events(trades_a, events_a)
    events_b_summary = _summarize_events(trades_b, events_b)

    base_index = pd.concat([res_a[["t", "date"]], res_b[["t", "date"]]]).drop_duplicates()
    base_index = base_index.sort_values("t").reset_index(drop=True)
    overlay = _merge_event_overlay(base_index, events_a_summary, events_b_summary)
    overlay.to_csv(debug_dir / "event_overlay.csv", index=False)

    print(f"diagnostics written to: {debug_dir}")


if __name__ == "__main__":
    main()
