from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import dataio
from src.backtest import run_backtest
from src.utils import ensure_dir, load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/reproduce_tables.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base_cfg = load_yaml(cfg["base_config"])
    base_cfg["strategy"]["mode"] = "full"

    prices = dataio.load_price_data(base_cfg)

    params = cfg["fee_sensitivity"]["params"]
    base_cfg["holding"]["hold_days_T"] = params["T"]
    base_cfg["no_buy"]["rebuy_pct_N"] = params["N"]
    base_cfg["extreme"]["extreme_sell_pct_E"] = params["E"]

    results = []
    for fee_gold, fee_btc in cfg["fee_sensitivity"]["fee_pairs"]:
        run_cfg = copy.deepcopy(base_cfg)
        run_cfg["fees"] = {**base_cfg["fees"], "fee_gold": fee_gold, "fee_btc": fee_btc}
        results_df, trades_df, _ = run_backtest(prices, run_cfg)
        final_nav = float(results_df["nav_total"].iloc[-1]) if len(results_df) else base_cfg["run"]["initial_cash"]
        roi = final_nav / base_cfg["run"]["initial_cash"] - 1.0
        max_dd = float(results_df["dd"].min()) if len(results_df) else 0.0
        trades_count = int(len(trades_df))
        results.append(
            {
                "fee_gold": fee_gold,
                "fee_btc": fee_btc,
                "final_nav": final_nav,
                "ROI": roi,
                "maxDD": max_dd,
                "trades_count": trades_count,
            }
        )

    out_dir = ensure_dir(Path("outputs") / "fee_sweep")
    out_path = out_dir / "fee_sensitivity.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)

    notes = [
        "# fee_sweep notes",
        "- params: [T=12, N=0.6, E=0.89] (paper 4.3)",
        "- reference_fees: [0.01,0.02], [0.02,0.03], [0.03,0.05], [0.1,0.1] (paper 4.4)",
        "",
        "## rows",
    ]
    for row in results:
        notes.append(f"- [{row['fee_gold']},{row['fee_btc']}] final_nav={row['final_nav']}")
    (out_dir / "notes.md").write_text("\n".join(notes), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
