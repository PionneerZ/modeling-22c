from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import dataio
from src.backtest import run_backtest
from src.utils import ensure_dir, load_yaml, make_run_id


TARGET_NAV = 220486.0
BTC_WINDOW = (1610, 1760)
PAPER_BTC_EXTREME_T = 1628


def _locate_windows(df: pd.DataFrame, asset: str, window_len: int, topk: int = 2) -> list[dict]:
    price_col = "price_gold" if asset == "gold" else "price_btc"
    min_max = 1900.0 if asset == "gold" else None
    min_min = 1450.0 if asset == "gold" else None
    min_range = 400.0 if asset == "gold" else None

    candidates = []
    for start in range(0, len(df) - window_len + 1):
        end = start + window_len - 1
        window = df.iloc[start : end + 1]
        prices = window[price_col]
        if prices.isna().all():
            continue
        w_min = float(prices.min())
        w_max = float(prices.max())
        w_range = w_max - w_min

        cond1_active = (min_min is not None) or (min_max is not None)
        cond2_active = min_range is not None

        cond1 = True
        if min_min is not None:
            cond1 = cond1 and (w_min >= min_min)
        if min_max is not None:
            cond1 = cond1 and (w_max >= min_max)

        cond2 = True
        if min_range is not None:
            cond2 = cond2 and (w_range >= min_range)
        if min_max is not None:
            cond2 = cond2 and (w_max >= min_max)

        if cond1_active and cond2_active:
            if not (cond1 or cond2):
                continue
        elif cond1_active:
            if not cond1:
                continue
        elif cond2_active:
            if not cond2:
                continue
        score = w_max + 0.5 * w_range
        candidates.append(
            {
                "t_start": int(window["t"].iloc[0]),
                "t_end": int(window["t"].iloc[-1]),
                "date_start": window["date"].iloc[0].date().isoformat(),
                "date_end": window["date"].iloc[-1].date().isoformat(),
                "min_price": w_min,
                "max_price": w_max,
                "end_price": float(window[price_col].iloc[-1]),
                "range": w_range,
                "score": score,
            }
        )

    if not candidates:
        return []
    cand_df = pd.DataFrame(candidates).sort_values("score", ascending=False)
    return cand_df.head(topk).to_dict(orient="records")


def _btc_metrics(trades: pd.DataFrame, events: pd.DataFrame) -> dict:
    t_start, t_end = BTC_WINDOW
    live_trades = trades[trades["qty"] > 0]
    btc_trades = live_trades[
        (live_trades["asset"] == "btc") & (live_trades["t"].between(t_start, t_end))
    ]
    extreme_sell = btc_trades[btc_trades["reason"] == "extreme_sell"]
    no_buy_on = events[
        (events["asset"] == "btc")
        & (events["state_type"] == "no_buy")
        & (events["now"] == True)
        & (events["t"].between(t_start, t_end))
    ]
    no_buy_off = events[
        (events["asset"] == "btc")
        & (events["state_type"] == "no_buy")
        & (events["now"] == False)
        & (events["t"].between(t_start, t_end))
    ]
    release_t = int(no_buy_off.iloc[0]["t"]) if not no_buy_off.empty else None
    rebuy_after_release = False
    if release_t is not None:
        rebuy_after_release = not btc_trades[
            (btc_trades["side"] == "buy") & (btc_trades["t"] >= release_t)
        ].empty

    return {
        "btc_trades_window": int(btc_trades.shape[0]),
        "btc_extreme_sell_window": not extreme_sell.empty,
        "btc_no_buy_on_window": not no_buy_on.empty,
        "btc_no_buy_release_window": not no_buy_off.empty,
        "btc_rebuy_after_release": rebuy_after_release,
        "btc_first_extreme_t": int(extreme_sell.iloc[0]["t"]) if not extreme_sell.empty else None,
    }


def _gold_metrics(results: pd.DataFrame, trades: pd.DataFrame, window: dict) -> dict:
    t_start = window["t_start"]
    t_end = window["t_end"]
    price_window = results[results["t"].between(t_start, t_end)]["price_gold"]
    max_price = float(price_window.max()) if not price_window.empty else float("nan")
    high_threshold = max_price * 0.95 if pd.notna(max_price) else float("nan")

    gold_trades = trades[
        (trades["asset"] == "gold") & (trades["t"].between(t_start, t_end))
    ]
    buys = gold_trades[(gold_trades["side"] == "buy") & (gold_trades["qty"] > 0)]
    sell_signals = gold_trades[
        (gold_trades["side"] == "sell")
        & (
            (gold_trades["qty"] > 0)
            | (gold_trades["reason"].fillna("").str.startswith("blocked_by_hold"))
        )
    ]
    high_sells = (
        sell_signals[sell_signals["price"] >= high_threshold]
        if pd.notna(high_threshold)
        else sell_signals
    )

    buy_count = int(buys.shape[0])
    sell_count = int(sell_signals.shape[0])
    high_sell_count = int(high_sells.shape[0])
    continuous_buy = buy_count >= 5
    high_sell_ok = sell_count >= 2 and high_sell_count >= 2
    score = int(continuous_buy) + int(high_sell_ok)

    return {
        "gold_buy_count": buy_count,
        "gold_sell_signal_count": sell_count,
        "gold_high_sell_count": high_sell_count,
        "gold_max_price": max_price if pd.notna(max_price) else None,
        "gold_high_threshold": high_threshold if pd.notna(high_threshold) else None,
        "gold_continuous_buy": continuous_buy,
        "gold_high_sell_ok": high_sell_ok,
        "gold_shape_score": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to base config yaml")
    parser.add_argument("--run-id", default=None, help="output run id under outputs/")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    base_cfg["_config_path"] = args.config

    prices = dataio.load_price_data(base_cfg)
    run_id = args.run_id or make_run_id()
    out_dir = ensure_dir(Path("outputs") / run_id / "calibration")

    gold_windows = _locate_windows(prices, "gold", window_len=101, topk=2)
    if len(gold_windows) < 2:
        gold_windows = gold_windows + gold_windows

    release_modes = ["days", "price"]
    cooldown_days_list = [30, 45, 60, 90, 120]
    release_fracs = [0.75, 0.85, 0.89, 0.92]

    rows = []
    for release_mode in release_modes:
        for cooldown_days in cooldown_days_list:
            for release_frac in release_fracs:
                cfg = copy.deepcopy(base_cfg)
                cfg["no_buy"]["release_mode"] = release_mode
                cfg["no_buy"]["cooldown_days"] = cooldown_days
                cfg["no_buy"]["release_frac"] = release_frac

                results, trades, _w, _debug, events = run_backtest(
                    prices, cfg, return_debug=True
                )
                final_nav = float(results["nav_total"].iloc[-1])
                nav_diff = abs(final_nav - TARGET_NAV)

                btc = _btc_metrics(trades, events)
                gold1 = _gold_metrics(results, trades, gold_windows[0]) if gold_windows else {}
                gold2 = _gold_metrics(results, trades, gold_windows[1]) if len(gold_windows) > 1 else {}

                gold_best_score = max(gold1.get("gold_shape_score", 0), gold2.get("gold_shape_score", 0))
                btc_score = sum(
                    [
                        int(btc["btc_extreme_sell_window"]),
                        int(btc["btc_no_buy_on_window"]),
                        int(btc["btc_no_buy_release_window"]),
                        int(btc["btc_rebuy_after_release"]),
                    ]
                )
                shape_score = btc_score + gold_best_score

                rows.append(
                    {
                        "release_mode": release_mode,
                        "cooldown_days": cooldown_days,
                        "release_frac": release_frac,
                        "final_nav": final_nav,
                        "nav_diff": nav_diff,
                        "shape_score": shape_score,
                        **btc,
                        "gold1_t_start": gold_windows[0]["t_start"] if gold_windows else None,
                        "gold1_t_end": gold_windows[0]["t_end"] if gold_windows else None,
                        "gold1_date_start": gold_windows[0]["date_start"] if gold_windows else None,
                        "gold1_date_end": gold_windows[0]["date_end"] if gold_windows else None,
                        "gold1_buy_count": gold1.get("gold_buy_count"),
                        "gold1_sell_signal_count": gold1.get("gold_sell_signal_count"),
                        "gold1_high_sell_count": gold1.get("gold_high_sell_count"),
                        "gold1_shape_score": gold1.get("gold_shape_score"),
                        "gold2_t_start": gold_windows[1]["t_start"] if len(gold_windows) > 1 else None,
                        "gold2_t_end": gold_windows[1]["t_end"] if len(gold_windows) > 1 else None,
                        "gold2_date_start": gold_windows[1]["date_start"] if len(gold_windows) > 1 else None,
                        "gold2_date_end": gold_windows[1]["date_end"] if len(gold_windows) > 1 else None,
                        "gold2_buy_count": gold2.get("gold_buy_count"),
                        "gold2_sell_signal_count": gold2.get("gold_sell_signal_count"),
                        "gold2_high_sell_count": gold2.get("gold_high_sell_count"),
                        "gold2_shape_score": gold2.get("gold_shape_score"),
                    }
                )

    report_df = pd.DataFrame(rows).sort_values(
        ["shape_score", "nav_diff"], ascending=[False, True]
    )
    report_csv = out_dir / "report.csv"
    report_df.to_csv(report_csv, index=False)

    best = report_df.iloc[0]
    best_cfg = copy.deepcopy(base_cfg)
    best_cfg["no_buy"]["release_mode"] = str(best["release_mode"])
    best_cfg["no_buy"]["cooldown_days"] = int(best["cooldown_days"])
    best_cfg["no_buy"]["release_frac"] = float(best["release_frac"])
    best_cfg["no_buy"]["scope"] = "global"
    best_cfg["run"]["change_note"] = "calibrate_to_paper: best shape score"

    recommended_path = Path("config") / "recommended.yaml"
    recommended_path.write_text(yaml.safe_dump(best_cfg, sort_keys=False), encoding="utf-8")

    btc_score = sum(
        [
            int(best["btc_extreme_sell_window"]),
            int(best["btc_no_buy_on_window"]),
            int(best["btc_no_buy_release_window"]),
            int(best["btc_rebuy_after_release"]),
        ]
    )
    gold_best_score = max(
        int(best.get("gold1_shape_score") or 0),
        int(best.get("gold2_shape_score") or 0),
    )
    report_lines = [
        "calibration_report",
        f"target_nav={TARGET_NAV:.2f}",
        f"candidates={len(report_df)}",
        "",
        "best_summary:",
        f"- shape_score={int(best['shape_score'])} (btc={btc_score}, gold={gold_best_score})",
        f"- final_nav={best['final_nav']:.2f}",
        f"- nav_diff={best['nav_diff']:.2f}",
        f"- btc_first_extreme_t={best['btc_first_extreme_t']}",
        f"- btc_trades_window={best['btc_trades_window']}",
        "",
        "best_config (only no-buy release params changed):",
        f"- no_buy.release_mode: {best['release_mode']}",
        f"- no_buy.cooldown_days: {best['cooldown_days']}",
        f"- no_buy.release_frac: {best['release_frac']}",
        "",
        "gold_windows:",
        f"- gold1: t={best['gold1_t_start']}..{best['gold1_t_end']} "
        f"({best['gold1_date_start']} to {best['gold1_date_end']})",
        f"- gold2: t={best['gold2_t_start']}..{best['gold2_t_end']} "
        f"({best['gold2_date_start']} to {best['gold2_date_end']})",
        "",
        "rationale:",
        "- selected by max shape_score then min nav_diff within the no-buy release grid.",
        "- minimal deviation because buy/sell logic and core params are unchanged.",
        "",
        f"recommended_config: {recommended_path}",
    ]
    report_md = out_dir / "report.md"

    # Optional 1D E scan if nav diff is still large
    minimal_delta_path = None
    if best["nav_diff"] > 15000:
        e_values = [0.83, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93]
        scan_rows = []
        for e_val in e_values:
            cfg = copy.deepcopy(best_cfg)
            cfg["extreme"]["extreme_sell_pct_E"] = float(e_val)
            cfg["paper_params"]["extreme_E"] = float(e_val)
            results, trades, _w, _debug, events = run_backtest(
                prices, cfg, return_debug=True
            )
            final_nav = float(results["nav_total"].iloc[-1])
            nav_diff = abs(final_nav - TARGET_NAV)
            btc = _btc_metrics(trades, events)
            extreme_t = btc.get("btc_first_extreme_t")
            t_diff = abs(extreme_t - PAPER_BTC_EXTREME_T) if extreme_t is not None else 999
            shape_score = sum(
                [
                    int(btc["btc_extreme_sell_window"]),
                    int(btc["btc_no_buy_on_window"]),
                    int(btc["btc_no_buy_release_window"]),
                    int(btc["btc_rebuy_after_release"]),
                ]
            )
            scan_rows.append(
                {
                    "E": e_val,
                    "final_nav": final_nav,
                    "nav_diff": nav_diff,
                    "btc_extreme_t": extreme_t,
                    "btc_extreme_t_diff": t_diff,
                    "shape_score": shape_score,
                }
            )
        scan_df = pd.DataFrame(scan_rows).sort_values(
            ["shape_score", "btc_extreme_t_diff", "nav_diff"], ascending=[False, True, True]
        )
        scan_path = out_dir / "e_scan.csv"
        scan_df.to_csv(scan_path, index=False)

        best_e = scan_df.iloc[0]
        best_cfg["extreme"]["extreme_sell_pct_E"] = float(best_e["E"])
        best_cfg["paper_params"]["extreme_E"] = float(best_e["E"])
        minimal_delta_path = Path("config") / "recommended_minimal_delta.yaml"
        minimal_delta_path.write_text(
            yaml.safe_dump(best_cfg, sort_keys=False), encoding="utf-8"
        )

        report_lines += [
            "",
            "e_scan:",
            f"- e_values={e_values}",
            f"- best_E={best_e['E']}",
            f"- best_E_nav={best_e['final_nav']:.2f}",
            f"- best_E_nav_diff={best_e['nav_diff']:.2f}",
            f"- best_E_extreme_t={best_e['btc_extreme_t']}",
            f"- best_E_extreme_t_diff={best_e['btc_extreme_t_diff']}",
            f"- recommended_minimal_delta: {minimal_delta_path}",
        ]

    report_md.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"report: {report_md}")
    print(f"report_csv: {report_csv}")
    print(f"recommended: {recommended_path}")
    if minimal_delta_path:
        print(f"recommended_minimal_delta: {minimal_delta_path}")


if __name__ == "__main__":
    main()
