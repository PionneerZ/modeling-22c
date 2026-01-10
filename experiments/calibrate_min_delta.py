from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import dataio
from src.backtest import run_backtest
from src.utils import ensure_dir, load_yaml


TARGET_NAV = 220486.0
BTC_WINDOW = (1610, 1760)


def _btc_window_metrics(trades: pd.DataFrame, events: pd.DataFrame) -> dict:
    t_start, t_end = BTC_WINDOW
    live_trades = trades[trades["qty"] > 0] if not trades.empty else trades
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

    extreme_t = int(extreme_sell.iloc[0]["t"]) if not extreme_sell.empty else None
    release_t = int(no_buy_off.iloc[0]["t"]) if not no_buy_off.empty else None
    rebuy_t = None
    if release_t is not None:
        after_release = btc_trades[(btc_trades["side"] == "buy") & (btc_trades["t"] >= release_t)]
        if not after_release.empty:
            rebuy_t = int(after_release.iloc[0]["t"])

    shape_score = sum(
        [
            int(extreme_t is not None),
            int(not no_buy_on.empty),
            int(not no_buy_off.empty),
            int(rebuy_t is not None),
        ]
    )
    return {
        "btc_extreme_t": extreme_t,
        "btc_rebuy_t": rebuy_t,
        "shape_score": shape_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to base config yaml")
    parser.add_argument(
        "--outdir",
        default="outputs/run_fixpack",
        help="output run dir (default: outputs/run_fixpack)",
    )
    parser.add_argument(
        "--fixed-fracs",
        default="0.5,0.55,0.6",
        help="comma-separated fixed_fraction candidates",
    )
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    base_cfg["_config_path"] = args.config
    prices = dataio.load_price_data(base_cfg)

    fixed_fracs = [float(v.strip()) for v in args.fixed_fracs.split(",") if v.strip()]
    base_fraction = float(base_cfg["buy_sizing"].get("fixed_fraction", 0.5))

    rows = []
    for frac in fixed_fracs:
        cfg = copy.deepcopy(base_cfg)
        cfg["buy_sizing"]["fixed_fraction"] = float(frac)
        results, trades, _w, _debug, events = run_backtest(prices, cfg, return_debug=True)
        final_nav = float(results["nav_total"].iloc[-1]) if len(results) else base_cfg["run"]["initial_cash"]
        btc = _btc_window_metrics(trades, events)
        rows.append(
            {
                "fixed_fraction": frac,
                "delta_fixed_fraction": frac - base_fraction,
                "final_nav": final_nav,
                "nav_diff": abs(final_nav - TARGET_NAV),
                "btc_extreme_t": btc["btc_extreme_t"],
                "btc_rebuy_t": btc["btc_rebuy_t"],
                "shape_score": btc["shape_score"],
            }
        )

    out_dir = ensure_dir(Path(args.outdir))
    debug_dir = ensure_dir(out_dir / "debug")
    df = pd.DataFrame(rows).sort_values(["nav_diff", "shape_score"], ascending=[True, False])
    df.to_csv(debug_dir / "calibrate_min_delta.csv", index=False)

    print(f"calibration written to: {debug_dir / 'calibrate_min_delta.csv'}")


if __name__ == "__main__":
    main()
