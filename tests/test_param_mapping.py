import unittest

from src.params import resolve_paper_params
from src.utils import load_yaml


class TestParamMapping(unittest.TestCase):
    def test_base_config_mapping(self):
        cfg = load_yaml("config/base.yaml")
        cfg["_config_path"] = "config/base.yaml"
        params = resolve_paper_params(cfg)
        self.assertEqual(params["hold_T"], 12)
        self.assertAlmostEqual(params["reentry_N"], 0.6)
        self.assertAlmostEqual(params["extreme_E"], 0.89)
        self.assertEqual(params["lookback_M"], 5)


if __name__ == "__main__":
    unittest.main()
