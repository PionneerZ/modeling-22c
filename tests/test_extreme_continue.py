import unittest
from datetime import datetime, timedelta

import pandas as pd

from src.backtest import run_backtest


def make_config():
    return {
        "run": {"initial_cash": 1000.0, "seed": 42},
        "fees": {"fee_gold": 0.01, "fee_btc": 0.02},
        "momentum": {"n_window": 2, "w_scheme": "equal"},
        "thresholds": {"thr_grad": 0.1, "thr_ma_diff": 0.1},
        "weight_factor": {"W_C": 1.0, "W_use": []},
        "selling": {"margin_L": 0.0, "avg_cost_method": "weighted_avg"},
        "extreme": {"extreme_C": 5.0, "extreme_sell_pct_E": 0.5, "extreme_deltaP": None},
        "no_buy": {
            "cooldown_days": 0,
            "release_mode": "days",
            "scope": "asset",
            "rebuy_on_release": False,
            "max_price_ref": "extreme_window",
            "rebuy_fraction": 1.0,
        },
        "holding": {"min_days_between_buys": 0, "min_days_between_sells": 0},
        "paper_params": {"hold_T": 0, "reentry_N": 0.6, "extreme_E": 0.5, "lookback_M": 2},
        "buy_sizing": {"buy_scale": 1.0, "buy_cap": 0.5, "buy_min_cash_reserve": 0.0},
        "execution": {"sell_then_buy": True},
        "strategy": {"mode": "full"},
    }


class TestExtremeContinue(unittest.TestCase):
    def test_extreme_skips_normal_logic(self):
        start = datetime(2021, 1, 1)
        dates = [start + timedelta(days=i) for i in range(3)]
        df = pd.DataFrame(
            {
                "date": dates,
                "price_gold": [100.0, 100.0, 100.0],
                "price_btc": [10.0, 12.0, 100.0],
                "is_trading_gold": [True, True, True],
                "is_trading_btc": [True, True, True],
            }
        )

        cfg = make_config()
        results, trades, _ = run_backtest(df, cfg)

        day3 = dates[2]
        trades_day3 = trades[trades["date"] == day3]
        self.assertTrue((~trades_day3["reason"].str.startswith("buy_")).all())
        self.assertTrue((trades_day3["reason"] != "sell_signal").all())


if __name__ == "__main__":
    unittest.main()
