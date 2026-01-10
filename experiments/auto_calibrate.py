from __future__ import annotations

import argparse
import copy
import itertools
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
GOLD_WINDOW = (460, 560)


def _btc_shape(trades: pd.DataFrame, events: pd.DataFrame) -> dict:
    t_start, t_end = BTC_WINDOW
    live_trades = trades[trades["qty"] > 0]
    btc_trades = live_trades[
        (live_trades["asset"] == "btc") & (live_trades["t"].between(t_start, t_end))
    ]
    btc_events = events[(events["asset"] == "btc")]

    extreme_sell_in_window = (
        not btc_trades[btc_trades["reason"] == "extreme_sell"].empty
    )
    no_buy_on = btc_events[
        (btc_events["state_type"] == "no_buy")
        & (btc_events["now"] == True)
        & (btc_events["t"].between(t_start, t_end))
    ]
    no_buy_release = btc_events[
        (btc_events["state_type"] == "no_buy")
        & (btc_events["now"] == False)
        & (btc_events["t"].between(t_start, t_end))
    ]

    release_t = None
    if not no_buy_release.empty:
        release_t = int(no_buy_release.iloc[0]["t"])
    buy_after_release = False
    if release_t is not None:
        buy_after_release = not btc_trades[
            (btc_trades["side"] == "buy") & (btc_trades["t"] >= release_t)
        ].empty

    return {
        "btc_extreme_sell_window": extreme_sell_in_window,
        "btc_no_buy_on_window": not no_buy_on.empty,
        "btc_no_buy_release_window": not no_buy_release.empty,
        "btc_buy_after_release_window": buy_after_release,
    }


def _gold_shape(results: pd.DataFrame, trades: pd.DataFrame) -> dict:
    t_start, t_end = GOLD_WINDOW
    gold_trades = trades[
        (trades["asset"] == "gold") & (trades["t"].between(t_start, t_end))
    ]
    buys = gold_trades[(gold_trades["side"] == "buy") & (gold_trades["qty"] > 0)]
    sells_actual = gold_trades[(gold_trades["side"] == "sell") & (gold_trades["qty"] > 0)]
    sells_signal = gold_trades[
        (gold_trades["side"] == "sell")
        & (
            (gold_trades["qty"] > 0)
            | (gold_trades["reason"].fillna("").str.startswith("blocked_by_hold"))
        )
    ]

    window_prices = results[results["t"].between(t_start, t_end)]["price_gold"]
    max_price = window_prices.max() if not window_prices.empty else float("nan")
    high_sell_threshold = max_price * 0.95 if pd.notna(max_price) else float("nan")
    high_sells = (
        sells_signal[sells_signal["price"] >= high_sell_threshold]
        if pd.notna(max_price)
        else sells_signal
    )

    return {
        "gold_buy_count": int(buys.shape[0]),
        "gold_sell_count": int(sells_signal.shape[0]),
        "gold_sell_actual_count": int(sells_actual.shape[0]),
        "gold_high_sell_count": int(high_sells.shape[0]),
        "gold_max_price_window": float(max_price) if pd.notna(max_price) else None,
        "gold_high_sell_threshold": float(high_sell_threshold) if pd.notna(max_price) else None,
    }


def _count_delta(base: dict, cand: dict, keys: list[str]) -> int:
    return sum(1 for k in keys if base.get(k) != cand.get(k))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to base config yaml")
    parser.add_argument("--run-id", default=None, help="output run id under outputs/")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    base_cfg["_config_path"] = args.config

    prices = dataio.load_price_data(base_cfg)
    run_id = args.run_id or base_cfg["run"].get("run_id") or make_run_id()
    out_dir = ensure_dir(Path("outputs") / run_id / "debug")

    base_results, base_trades, _w, _debug, base_events = run_backtest(
        prices, base_cfg, return_debug=True
    )
    base_nav = float(base_results["nav_total"].iloc[-1])

    base_flat = {
        "no_buy.release_frac": base_cfg["no_buy"].get("release_frac"),
        "no_buy.scope": base_cfg["no_buy"].get("scope"),
        "buy_logic.mode": base_cfg.get("buy_logic", {}).get("mode"),
        "weight_factor.apply_mode": base_cfg.get("weight_factor", {}).get("apply_mode"),
        "weight_factor.W_use": tuple(base_cfg.get("weight_factor", {}).get("W_use", [])),
        "extreme.extreme_sell_pct_E": base_cfg.get("extreme", {}).get("extreme_sell_pct_E"),
        "buy_sizing.mode": base_cfg.get("buy_sizing", {}).get("mode"),
        "buy_sizing.fixed_fraction": base_cfg.get("buy_sizing", {}).get("fixed_fraction"),
    }

    release_fracs = [0.75, 0.8, 0.85]
    scopes = ["asset"]
    buy_modes = ["single", "multi"]
    apply_modes = ["compare", "score"]
    w_use_options = [("btc",), ("btc", "gold")]
    extreme_es = [0.89, 0.86]
    buy_modes_sizing = ["fixed_fraction", "score"]
    fixed_fracs = [0.6, 0.8, 1.0]

    candidates = []
    for release_frac, scope, buy_mode, apply_mode, w_use, extreme_e, sizing_mode in itertools.product(
        release_fracs, scopes, buy_modes, apply_modes, w_use_options, extreme_es, buy_modes_sizing
    ):
        frac_options = fixed_fracs if sizing_mode == "fixed_fraction" else [None]
        for fixed_fraction in frac_options:
            cfg = copy.deepcopy(base_cfg)
            cfg["no_buy"]["release_frac"] = release_frac
            cfg["no_buy"]["scope"] = scope
            cfg["buy_logic"]["mode"] = buy_mode
            cfg["weight_factor"]["apply_mode"] = apply_mode
            cfg["weight_factor"]["W_use"] = list(w_use)
            cfg["extreme"]["extreme_sell_pct_E"] = extreme_e
            cfg["paper_params"]["extreme_E"] = extreme_e
            cfg["buy_sizing"]["mode"] = sizing_mode
            if fixed_fraction is not None:
                cfg["buy_sizing"]["fixed_fraction"] = fixed_fraction

            results, trades, _w_stats, _debug, events = run_backtest(
                prices, cfg, return_debug=True
            )
            final_nav = float(results["nav_total"].iloc[-1])
            nav_diff = abs(final_nav - TARGET_NAV)

            btc_shape = _btc_shape(trades, events)
            gold_shape = _gold_shape(results, trades)
            shape_ok = (
                btc_shape["btc_extreme_sell_window"]
                and btc_shape["btc_no_buy_on_window"]
                and btc_shape["btc_no_buy_release_window"]
                and btc_shape["btc_buy_after_release_window"]
                and gold_shape["gold_buy_count"] >= 3
                and gold_shape["gold_high_sell_count"] >= 2
            )

            cand_flat = {
                "no_buy.release_frac": release_frac,
                "no_buy.scope": scope,
                "buy_logic.mode": buy_mode,
                "weight_factor.apply_mode": apply_mode,
                "weight_factor.W_use": tuple(w_use),
                "extreme.extreme_sell_pct_E": extreme_e,
                "buy_sizing.mode": sizing_mode,
                "buy_sizing.fixed_fraction": fixed_fraction
                if fixed_fraction is not None
                else base_cfg["buy_sizing"].get("fixed_fraction"),
            }
            delta_count = _count_delta(base_flat, cand_flat, list(base_flat.keys()))

            candidates.append(
                {
                    **cand_flat,
                    "final_nav": final_nav,
                    "nav_diff": nav_diff,
                    "shape_ok": shape_ok,
                    "delta_count": delta_count,
                    **btc_shape,
                    **gold_shape,
                }
            )

    cand_df = pd.DataFrame(candidates)
    cand_path = out_dir / "auto_calibrate_candidates.csv"
    cand_df.to_csv(cand_path, index=False)

    ok_df = cand_df[cand_df["shape_ok"]].sort_values(
        ["nav_diff", "delta_count"], ascending=[True, True]
    )
    best_row = ok_df.iloc[0] if not ok_df.empty else cand_df.sort_values("nav_diff").iloc[0]

    report_lines = [
        "auto_calibrate_report",
        f"base_final_nav={base_nav:.2f}",
        f"target_nav={TARGET_NAV:.2f}",
        f"candidates_total={cand_df.shape[0]}",
        f"candidates_shape_ok={ok_df.shape[0]}",
        "",
        "best_candidate:",
    ]
    for col in [
        "no_buy.release_frac",
        "no_buy.scope",
        "buy_logic.mode",
        "weight_factor.apply_mode",
        "weight_factor.W_use",
        "extreme.extreme_sell_pct_E",
        "buy_sizing.mode",
        "buy_sizing.fixed_fraction",
        "final_nav",
        "nav_diff",
        "delta_count",
        "btc_extreme_sell_window",
        "btc_no_buy_on_window",
        "btc_no_buy_release_window",
        "btc_buy_after_release_window",
        "gold_buy_count",
        "gold_sell_count",
        "gold_sell_actual_count",
        "gold_high_sell_count",
    ]:
        report_lines.append(f"- {col}: {best_row[col]}")

    report_path = out_dir / "auto_calibrate_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # Build recommended config
    rec_cfg = copy.deepcopy(base_cfg)
    rec_cfg["no_buy"]["release_frac"] = float(best_row["no_buy.release_frac"])
    rec_cfg["no_buy"]["scope"] = str(best_row["no_buy.scope"])
    rec_cfg["buy_logic"]["mode"] = str(best_row["buy_logic.mode"])
    rec_cfg["weight_factor"]["apply_mode"] = str(best_row["weight_factor.apply_mode"])
    rec_cfg["weight_factor"]["W_use"] = list(best_row["weight_factor.W_use"])
    rec_cfg["extreme"]["extreme_sell_pct_E"] = float(best_row["extreme.extreme_sell_pct_E"])
    rec_cfg["paper_params"]["extreme_E"] = float(best_row["extreme.extreme_sell_pct_E"])
    rec_cfg["buy_sizing"]["mode"] = str(best_row["buy_sizing.mode"])
    if pd.notna(best_row["buy_sizing.fixed_fraction"]):
        rec_cfg["buy_sizing"]["fixed_fraction"] = float(best_row["buy_sizing.fixed_fraction"])
    rec_cfg["run"]["change_note"] = "auto_calibrate: window-shape aligned with minimal delta"

    rec_path = Path("config") / "recommended.yaml"
    rec_path.write_text(yaml.safe_dump(rec_cfg, sort_keys=False), encoding="utf-8")

    print(f"report: {report_path}")
    print(f"candidates: {cand_path}")
    print(f"recommended: {rec_path}")


if __name__ == "__main__":
    main()
