import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.dataio import load_price_data


class TestWeekendHandling(unittest.TestCase):
    def test_gold_weekend_ffill(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prices.csv"
            start = datetime(2021, 1, 1)  # Friday
            rows = []
            for i in range(5):
                d = start + timedelta(days=i)
                gold = "" if d.weekday() >= 5 else f"{1800 + i:.2f}"
                btc = f"{30000 + i:.2f}"
                rows.append([d.strftime("%Y-%m-%d"), gold, btc])

            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["date", "gold", "btc"])
                w.writerows(rows)

            cfg = {
                "data": {
                    "path": str(path),
                    "date_col": "date",
                    "gold_col": "gold",
                    "btc_col": "btc",
                    "date_format": "%Y-%m-%d",
                    "start_date": "2021-01-01",
                    "end_date": "2021-01-05",
                    "end_date_mode": "trade_end",
                    "calendar_anchor": "btc",
                    "gold_ffill_for_valuation": True,
                    "gold_trade_weekdays_only": True,
                }
            }

            df = load_price_data(cfg)
            df["weekday"] = pd.to_datetime(df["date"]).dt.weekday

            weekend = df[df["weekday"] >= 5]
            self.assertTrue((~weekend["is_trading_gold"]).all())
            self.assertTrue((weekend["price_gold"].notna()).all())
            self.assertTrue((weekend["is_trading_btc"]).all())
            self.assertTrue(df["is_trading_btc"].all())
            self.assertEqual(df["t"].iloc[-1], len(df) - 1)


if __name__ == "__main__":
    unittest.main()
