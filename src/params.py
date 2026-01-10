from __future__ import annotations

from typing import Dict


def resolve_paper_params(config: Dict) -> Dict[str, float]:
    paper = config.get("paper_params") or {}
    missing = [k for k in ("hold_T", "reentry_N", "extreme_E", "lookback_M") if k not in paper]
    if missing:
        raise ValueError(f"paper_params missing keys: {missing}")

    hold_T = int(paper["hold_T"])
    reentry_N = float(paper["reentry_N"])
    extreme_E = float(paper["extreme_E"])
    lookback_M = int(paper["lookback_M"])

    legacy_hold = config.get("holding", {}).get("hold_days_T")
    if legacy_hold is not None and int(legacy_hold) != hold_T:
        raise ValueError("holding.hold_days_T conflicts with paper_params.hold_T")

    legacy_reentry = config.get("no_buy", {}).get("rebuy_pct_N")
    if legacy_reentry is not None and float(legacy_reentry) != reentry_N:
        raise ValueError("no_buy.rebuy_pct_N conflicts with paper_params.reentry_N")

    legacy_extreme = config.get("extreme", {}).get("extreme_sell_pct_E")
    if legacy_extreme is not None and float(legacy_extreme) != extreme_E:
        raise ValueError("extreme.extreme_sell_pct_E conflicts with paper_params.extreme_E")

    legacy_window = config.get("momentum", {}).get("n_window")
    if legacy_window is not None and int(legacy_window) != lookback_M:
        raise ValueError("momentum.n_window conflicts with paper_params.lookback_M")

    return {
        "hold_T": hold_T,
        "reentry_N": reentry_N,
        "extreme_E": extreme_E,
        "lookback_M": lookback_M,
    }
