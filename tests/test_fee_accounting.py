import unittest

from src.portfolio import Portfolio


class TestFeeAccounting(unittest.TestCase):
    def test_buy_sell_fee_cash_flow(self):
        p = Portfolio(cash=1000.0, avg_cost_method="weighted_avg")
        buy = p.buy("gold", qty=2, price=100.0, fee_rate=0.01)
        self.assertAlmostEqual(buy["fee_paid"], 2.0)
        self.assertAlmostEqual(p.cash, 1000.0 - 202.0)
        self.assertAlmostEqual(p.avg_cost["gold"], 101.0)

        sell = p.sell("gold", qty=2, price=120.0, fee_rate=0.01)
        self.assertAlmostEqual(sell["fee_paid"], 2.4)
        self.assertAlmostEqual(p.cash, 1000.0 - 202.0 + 237.6)


if __name__ == "__main__":
    unittest.main()
