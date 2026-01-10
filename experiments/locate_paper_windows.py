"""Utility: locate paper window indices for Figure 8/9 alignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import dataio
from src.utils import ensure_dir, load_yaml, make_run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config yaml")
    parser.add_argument("--asset", required=True, choices=["gold", "btc"], help="asset name")
    parser.add_argument("--window-len", type=int, default=101, help="window length in days")
    parser.add_argument("--topk", type=int, default=5, help="top windows to output")
    parser.add_argument("--min-max", type=float, default=None, help="min required max price")
    parser.add_argument("--min-min", type=float, default=None, help="min required min price")
    parser.add_argument("--min-range", type=float, default=None, help="min required price range")
    parser.add_argument("--run-id", default=None, help="output run id under outputs/")
    args = parser.parse_args()

    config = load_yaml(args.config)
    config["_config_path"] = args.config

    df = dataio.load_price_data(config)
    price_col = "price_gold" if args.asset == "gold" else "price_btc"

    # Default constraints for gold if not specified
    min_max = args.min_max
    min_min = args.min_min
    min_range = args.min_range
    if args.asset == "gold":
        if min_max is None:
            min_max = 1900.0
        if min_min is None:
            min_min = 1450.0
        if min_range is None:
            min_range = 400.0

    window_len = args.window_len
    if window_len <= 0:
        raise ValueError("window_len must be positive")

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

        date_start = window["date"].iloc[0].date().isoformat()
        date_end = window["date"].iloc[-1].date().isoformat()
        score = w_max + 0.5 * w_range
        candidates.append(
            {
                "t_start": int(window["t"].iloc[0]),
                "t_end": int(window["t"].iloc[-1]),
                "date_start": date_start,
                "date_end": date_end,
                "min_price": w_min,
                "max_price": w_max,
                "end_price": float(window[price_col].iloc[-1]),
                "range": w_range,
                "score": score,
            }
        )

    if not candidates:
        print("no candidates matched constraints")
        return

    cand_df = pd.DataFrame(candidates).sort_values("score", ascending=False)
    topk = cand_df.head(args.topk)

    run_id = args.run_id or make_run_id()
    out_dir = ensure_dir(Path("outputs") / run_id / "debug")
    out_path = out_dir / f"paper_window_candidates_{args.asset}.csv"
    topk.to_csv(out_path, index=False)

    print(topk.to_string(index=False))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
