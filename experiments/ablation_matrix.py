from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import dataio
from src.backtest import run_backtest
from src.params import resolve_paper_params
from src.utils import calc_drawdown, load_yaml


BTC_WINDOW = (1610, 1760)


def _get_nested(cfg: Dict, path: str):
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _set_nested(cfg: Dict, path: str, value) -> None:
    cur = cfg
    keys = path.split(".")
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = copy.deepcopy(value)


def _apply_group(base: Dict, override: Dict, keys: List[str]) -> None:
    for path in keys:
        value = _get_nested(override, path)
        _set_nested(base, path, value)


def _btc_window_metrics(trades: pd.DataFrame, events: pd.DataFrame) -> dict:
    t_start, t_end = BTC_WINDOW
    live_trades = trades[trades["qty"] > 0] if not trades.empty else trades
    btc_trades = live_trades[
        (live_trades["asset"] == "btc") & (live_trades["t"].between(t_start, t_end))
    ]
    extreme_sell = btc_trades[btc_trades["reason"] == "extreme_sell"]

    if events.empty or "state_type" not in events.columns:
        return {
            "btc_first_extreme_t": None,
            "btc_rebuy_t": None,
        }

    no_buy_off = events[
        (events["asset"] == "btc")
        & (events["state_type"] == "no_buy")
        & (events["now"] == False)
        & (events["t"].between(t_start, t_end))
    ]
    extreme_t = int(extreme_sell.iloc[0]["t"]) if not extreme_sell.empty else None
    release_t = int(no_buy_off.iloc[0]["t"]) if not no_buy_off.empty else None
    rebuy_t = None
    if release_t is not None:
        after_release = btc_trades[(btc_trades["side"] == "buy") & (btc_trades["t"] >= release_t)]
        if not after_release.empty:
            rebuy_t = int(after_release.iloc[0]["t"])
    return {
        "btc_first_extreme_t": extreme_t,
        "btc_rebuy_t": rebuy_t,
    }


def _run_once(cfg: Dict, prices: pd.DataFrame) -> dict:
    results, trades, _w, _debug, events = run_backtest(prices, cfg, return_debug=True)
    final_nav = float(results["nav_total"].iloc[-1]) if len(results) else cfg["run"]["initial_cash"]
    max_dd = float(min(calc_drawdown(results["nav_total"]))) if len(results) else 0.0
    exposure_ratio = (results["nav_gold"] + results["nav_btc"]) / results["nav_total"]
    avg_exposure = float(exposure_ratio.mean()) if not exposure_ratio.empty else 0.0

    live_trades = trades[trades["qty"] > 0] if not trades.empty else trades
    first_trade_date = None
    if not live_trades.empty:
        first_trade_date = pd.to_datetime(live_trades.iloc[0]["date"]).date().isoformat()

    btc_metrics = _btc_window_metrics(trades, events)
    return {
        "final_nav": final_nav,
        "maxDD": max_dd,
        "avg_exposure_ratio": avg_exposure,
        "trades_count": int(live_trades.shape[0]) if not live_trades.empty else 0,
        "first_trade_date": first_trade_date,
        **btc_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-highnav", required=True, help="path to high-nav config yaml")
    parser.add_argument("--config-paper", required=True, help="path to paper-aligned config yaml")
    parser.add_argument(
        "--outdir",
        default="outputs/run_fixpack",
        help="output run dir (default: outputs/run_fixpack)",
    )
    args = parser.parse_args()

    cfg_high = load_yaml(args.config_highnav)
    cfg_high["_config_path"] = args.config_highnav
    cfg_paper = load_yaml(args.config_paper)
    cfg_paper["_config_path"] = args.config_paper

    prices = dataio.load_price_data(cfg_high)

    groups = {
        "G1_buy_sizing": [
            "buy_sizing.mode",
            "buy_sizing.fixed_fraction",
            "buy_sizing.buy_scale",
            "buy_sizing.buy_cap",
            "buy_sizing.buy_min_cash_reserve",
        ],
        "G2_no_buy": [
            "no_buy.scope",
            "no_buy.release_mode",
            "no_buy.release_direction",
            "no_buy.cooldown_days",
            "no_buy.release_frac",
            "no_buy.rebuy_on_release",
            "no_buy.rebuy_fraction",
            "no_buy.max_price_ref",
        ],
        "G3_extreme_holding": [
            "extreme.assets",
            "extreme.extreme_C",
            "extreme.extreme_sell_pct_E",
            "extreme.extreme_deltaP",
            "extreme.min_history_days",
            "extreme.avg_mode",
            "extreme.avg_window_days",
            "holding.min_days_between_buys",
            "holding.min_days_between_sells",
            "holding.sell_hold_ref",
        ],
        "G4_signals": [
            "signals.ma_include_current",
            "signals.require_full_window",
            "signals.reversion_mode",
            "signals.reversion_pct",
            "signals.score_mode",
            "signals.score_threshold",
            "thresholds.thr_grad",
            "thresholds.thr_ma_diff",
            "momentum.w_scheme",
        ],
        "G5_execution": [
            "selling.avg_cost_method",
            "selling.margin_L",
            "selling.skip_if_buy_signal",
            "execution.sell_then_buy",
        ],
    }

    out_dir = Path(args.outdir)
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    base_metrics = _run_once(cfg_high, prices)
    paper_metrics = _run_once(cfg_paper, prices)

    rows = [
        {"scenario": "base_highnav", "group": "baseline", **base_metrics},
        {"scenario": "base_paper", "group": "baseline", **paper_metrics},
    ]

    for name, keys in groups.items():
        cfg = copy.deepcopy(cfg_high)
        _apply_group(cfg, cfg_paper, keys)
        metrics = _run_once(cfg, prices)
        rows.append({"scenario": "highnav_with_paper_group", "group": name, **metrics})

        cfg = copy.deepcopy(cfg_paper)
        _apply_group(cfg, cfg_high, keys)
        metrics = _run_once(cfg, prices)
        rows.append({"scenario": "paper_with_highnav_group", "group": name, **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(debug_dir / "ablation_results.csv", index=False)

    # summary: largest drop from base_highnav when swapping in paper groups
    base_nav = base_metrics["final_nav"]
    swapped = df[df["scenario"] == "highnav_with_paper_group"].copy()
    swapped["nav_drop"] = base_nav - swapped["final_nav"]
    worst = swapped.sort_values("nav_drop", ascending=False).head(1)

    summary_lines = [
        "# ablation_summary",
        f"- base_highnav_final_nav: {base_metrics['final_nav']:.2f}",
        f"- base_paper_final_nav: {paper_metrics['final_nav']:.2f}",
    ]
    if not worst.empty:
        row = worst.iloc[0]
        summary_lines.append(
            f"- largest_drop_group: {row['group']} (drop {row['nav_drop']:.2f})"
        )
    else:
        summary_lines.append("- largest_drop_group: none")

    (debug_dir / "ablation_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"ablation results written to: {debug_dir}")


if __name__ == "__main__":
    main()
