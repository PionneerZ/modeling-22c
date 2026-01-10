from __future__ import annotations

import argparse
import copy
import sys
from itertools import product
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import dataio
from src.backtest import run_backtest
from src.utils import ensure_dir, load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/reproduce_tables.yaml")
    parser.add_argument("--outdir", default="outputs/param_grid")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base_cfg = load_yaml(cfg["base_config"])
    base_cfg["strategy"]["mode"] = "full"

    prices = dataio.load_price_data(base_cfg)

    grid = cfg["param_grid"]
    results = []
    for T, N, E in product(grid["T"], grid["N"], grid["E"]):
        run_cfg = copy.deepcopy(base_cfg)
        run_cfg["paper_params"] = {
            **base_cfg["paper_params"],
            "hold_T": T,
            "reentry_N": N,
            "extreme_E": E,
        }
        run_cfg["holding"] = {**base_cfg["holding"], "min_days_between_sells": T}
        run_cfg["extreme"] = {**base_cfg["extreme"], "extreme_sell_pct_E": E}
        run_cfg["fees"] = {**base_cfg["fees"], "fee_gold": grid["fee_gold"], "fee_btc": grid["fee_btc"]}

        results_df, trades_df, _ = run_backtest(prices, run_cfg)
        final_nav = float(results_df["nav_total"].iloc[-1]) if len(results_df) else base_cfg["run"]["initial_cash"]
        roi = final_nav / base_cfg["run"]["initial_cash"] - 1.0
        max_dd = float(results_df["dd"].min()) if len(results_df) else 0.0
        trades_count = int(len(trades_df))
        results.append(
            {
                "T": T,
                "N": N,
                "E": E,
                "final_nav": final_nav,
                "ROI": roi,
                "maxDD": max_dd,
                "trades_count": trades_count,
            }
        )

    out_dir = ensure_dir(Path(args.outdir))
    out_path = out_dir / "param_sweep.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)

    results_sorted = sorted(results, key=lambda x: x["final_nav"], reverse=True)
    best = results_sorted[0]
    print(f"best combo: {[best['T'], best['N'], best['E']]} final_nav={best['final_nav']}")

    notes = [
        "# param_grid notes",
        f"- best_combo: {[best['T'], best['N'], best['E']]}",
        f"- best_final_nav: {best['final_nav']}",
        "- reference_best: [12, 0.6, 0.89] (paper 4.3)",
        "",
        "## Top5",
    ]
    for row in results_sorted[:5]:
        notes.append(f"- {[row['T'], row['N'], row['E']]} final_nav={row['final_nav']}")
    (out_dir / "notes.md").write_text("\n".join(notes), encoding="utf-8")


if __name__ == "__main__":
    main()
