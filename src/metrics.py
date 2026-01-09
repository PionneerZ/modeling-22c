from __future__ import annotations

from typing import Dict, Optional


def build_metrics(
    results_df,
    trades_df,
    config: Dict,
    run_id: str,
    data_range: str,
    w_stats: Optional[Dict] = None,
) -> Dict:
    initial_cash = config["run"]["initial_cash"]
    final_nav = float(results_df["nav_total"].iloc[-1]) if len(results_df) else initial_cash
    roi = final_nav / initial_cash - 1.0
    max_dd = float(results_df["dd"].min(skipna=True)) if len(results_df) else 0.0

    params = {
        "hold_days_T": config["holding"]["hold_days_T"],
        "rebuy_pct_N": config["no_buy"]["rebuy_pct_N"],
        "extreme_sell_pct_E": config["extreme"]["extreme_sell_pct_E"],
        "extreme_C": config["extreme"]["extreme_C"],
        "margin_L": config["selling"]["margin_L"],
        "n_window": config["momentum"]["n_window"],
        "w_scheme": config["momentum"]["w_scheme"],
        "thr_grad": config["thresholds"]["thr_grad"],
        "thr_ma_diff": config["thresholds"]["thr_ma_diff"],
        "buy_scale": config["buy_sizing"]["buy_scale"],
        "buy_cap": config["buy_sizing"]["buy_cap"],
    }

    payload = {
        "final_nav": final_nav,
        "ROI": roi,
        "maxDD": max_dd,
        "max_drawdown": max_dd,
        "trades_count": int(len(trades_df)),
        "params": params,
        "fee_pair": {
            "fee_gold": config["fees"]["fee_gold"],
            "fee_btc": config["fees"]["fee_btc"],
        },
        "run_id": run_id,
        "data_range": data_range,
        "calendar_mode": config["data"].get("calendar_anchor", "btc"),
        "end_date_mode": config["data"].get("end_date_mode", "trade_end"),
        "seed": config["run"].get("seed"),
    }
    if w_stats:
        payload["W"] = w_stats
    return payload
