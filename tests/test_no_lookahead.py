import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.backtest import run_backtest


def make_config():
    return {
        "run": {"initial_cash": 1000.0, "seed": 42},
        "fees": {"fee_gold": 0.01, "fee_btc": 0.02},
        "momentum": {"n_window": 3, "w_scheme": "equal"},
        "thresholds": {"thr_grad": 1.0, "thr_ma_diff": 1.0},
        "weight_factor": {"W_C": 1.0, "W_use": []},
        "selling": {"margin_L": 0.0, "avg_cost_method": "weighted_avg"},
        "extreme": {"extreme_C": 5.0, "extreme_sell_pct_E": 0.89, "extreme_deltaP": None},
        "no_buy": {
            "cooldown_days": 0,
            "release_mode": "days",
            "scope": "asset",
            "rebuy_on_release": False,
            "max_price_ref": "extreme_window",
            "rebuy_fraction": 1.0,
        },
        "holding": {"min_days_between_buys": 0, "min_days_between_sells": 2},
        "paper_params": {"hold_T": 2, "reentry_N": 0.6, "extreme_E": 0.89, "lookback_M": 3},
        "buy_sizing": {"buy_scale": 1.0, "buy_cap": 0.5, "buy_min_cash_reserve": 0.0},
        "execution": {"calendar_mode": "union", "gold_ffill": True, "sell_then_buy": True},
        "strategy": {"mode": "full"},
    }


def make_price_df(values_gold, values_btc):
    start = datetime(2021, 1, 1)
    dates = [start + timedelta(days=i) for i in range(len(values_gold))]
    df = pd.DataFrame(
        {
            "date": dates,
            "price_gold": values_gold,
            "price_btc": values_btc,
            "is_trading_gold": [True] * len(values_gold),
            "is_trading_btc": [True] * len(values_btc),
        }
    )
    return df


class TestNoLookahead(unittest.TestCase):
    def test_no_lookahead(self):
        cfg = make_config()

        gold = [100, 101, 102, 103, 104, 105, 106, 107]
        btc = [200, 201, 202, 203, 204, 205, 206, 207]
        df1 = make_price_df(gold, btc)

        gold_future = gold[:4] + [1000, 1000, 1000, 1000]
        btc_future = btc[:4] + [5000, 5000, 5000, 5000]
        df2 = make_price_df(gold_future, btc_future)

        res1, trades1, _ = run_backtest(df1, cfg)
        res2, trades2, _ = run_backtest(df2, cfg)

        # first 4 days should be identical if no lookahead is used
        cols = ["nav_total", "cash", "gold_qty", "btc_qty"]
        np.testing.assert_allclose(res1.loc[:3, cols], res2.loc[:3, cols], rtol=1e-8)

        trades1_early = trades1[trades1["date"] <= res1.loc[3, "date"]]
        trades2_early = trades2[trades2["date"] <= res2.loc[3, "date"]]
        self.assertEqual(len(trades1_early), len(trades2_early))


if __name__ == "__main__":
    unittest.main()
