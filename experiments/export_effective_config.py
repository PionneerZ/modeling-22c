"""Utility: dump an expanded config with resolved defaults."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.params import resolve_paper_params
from src.utils import load_yaml


def _ensure_section(cfg: dict, key: str, default: dict) -> None:
    if key not in cfg or cfg[key] is None:
        cfg[key] = copy.deepcopy(default)


def _set_default(d: dict, key: str, value) -> None:
    if key not in d or d[key] is None:
        d[key] = value


def _expand_effective_config(cfg: dict) -> dict:
    expanded = copy.deepcopy(cfg)
    paper = resolve_paper_params(expanded)

    _ensure_section(expanded, "signals", {})
    _ensure_section(expanded, "buy_logic", {})
    _ensure_section(expanded, "weight_factor", {})
    _ensure_section(expanded, "no_buy", {})
    _ensure_section(expanded, "holding", {})
    _ensure_section(expanded, "buy_sizing", {})
    _ensure_section(expanded, "selling", {})
    _ensure_section(expanded, "execution", {})
    _ensure_section(expanded, "extreme", {})

    signals = expanded["signals"]
    _set_default(signals, "reversion_mode", "below_ma")
    _set_default(signals, "score_mode", "max")
    _set_default(signals, "score_threshold", 0.0)
    _set_default(signals, "ma_include_current", True)
    _set_default(signals, "require_full_window", False)
    if signals.get("reversion_mode") == "below_ma_pct" and signals.get("reversion_pct") is None:
        signals["reversion_pct"] = float(paper["reentry_N"])

    buy_logic = expanded["buy_logic"]
    _set_default(buy_logic, "mode", "single")
    _set_default(buy_logic, "rebalance_on_buy", False)
    _set_default(buy_logic, "rebalance_ignore_hold", True)
    if buy_logic.get("rebalance_threshold") is None:
        buy_logic["rebalance_threshold"] = float(signals.get("score_threshold", 0.0) or 0.0)

    weight = expanded["weight_factor"]
    _set_default(weight, "apply_mode", "compare")
    _set_default(weight, "W_use", [])
    _set_default(weight, "use_t_plus_one", True)

    no_buy = expanded["no_buy"]
    _set_default(no_buy, "scope", "asset")
    _set_default(no_buy, "rebuy_on_release", False)
    _set_default(no_buy, "max_price_ref", "extreme_window")
    _set_default(no_buy, "release_mode", "hybrid")
    _set_default(no_buy, "release_direction", "drop")
    _set_default(no_buy, "cooldown_days", 0)
    if no_buy.get("release_frac") is None:
        no_buy["release_frac"] = float(paper["reentry_N"])

    holding = expanded["holding"]
    _set_default(holding, "min_days_between_buys", 0)
    if holding.get("min_days_between_sells") is None:
        holding["min_days_between_sells"] = int(paper["hold_T"])
    _set_default(holding, "sell_hold_ref", "entry")

    buy_sizing = expanded["buy_sizing"]
    _set_default(buy_sizing, "mode", "score")
    _set_default(buy_sizing, "fixed_fraction", 0.5)
    _set_default(buy_sizing, "buy_scale", 10.0)
    _set_default(buy_sizing, "buy_cap", 1.0)
    _set_default(buy_sizing, "buy_min_cash_reserve", 0.0)

    selling = expanded["selling"]
    _set_default(selling, "avg_cost_method", "weighted_avg")
    _set_default(selling, "skip_if_buy_signal", True)

    execution = expanded["execution"]
    _set_default(execution, "sell_then_buy", True)

    extreme = expanded["extreme"]
    _set_default(extreme, "min_history_days", 0)
    _set_default(extreme, "avg_mode", "positive_history")
    avg_window = extreme.get("avg_window_days")
    if not avg_window:
        extreme["avg_window_days"] = int(paper["lookback_M"])

    expanded["paper_params"] = {
        "hold_T": int(paper["hold_T"]),
        "reentry_N": float(paper["reentry_N"]),
        "extreme_E": float(paper["extreme_E"]),
        "lookback_M": int(paper["lookback_M"]),
    }

    return expanded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config yaml")
    parser.add_argument("--out", required=True, help="output json path")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg["_config_path"] = args.config
    effective = _expand_effective_config(cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(effective, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
