from __future__ import annotations

from typing import Dict, Optional


def should_sell(net_sell_price: float, avg_cost: float, margin_L: float) -> bool:
    return (net_sell_price - avg_cost - margin_L) > 0


def extreme_trigger(gradient: float, avg_grad: float, extreme_C: float) -> bool:
    if avg_grad <= 0:
        return False
    return gradient > (extreme_C * avg_grad)


def extreme_sell(price: float, max_price: float, sell_pct_E: float, delta_p: Optional[float]):
    if max_price <= 0:
        return False
    if price <= sell_pct_E * max_price:
        return True
    if delta_p is not None and price <= max_price - delta_p:
        return True
    return False


def size_buy(
    cash: float,
    price: float,
    score: float,
    buy_scale: float,
    buy_cap: float,
    min_cash_reserve: float,
    mode: str = "score",
    fixed_fraction: float = 0.5,
) -> float:
    if cash <= min_cash_reserve or price <= 0:
        return 0.0
    available_cash = max(0.0, cash - min_cash_reserve)
    if mode == "fixed_fraction":
        fraction = max(0.0, min(1.0, fixed_fraction))
        spend = available_cash * fraction
    elif mode == "score":
        if score <= 0:
            return 0.0
        score_norm = score / max(price, 1e-9)
        fraction = max(0.0, min(buy_cap, buy_scale * score_norm))
        spend = available_cash * fraction
    else:
        raise ValueError(f"unsupported buy sizing mode: {mode}")
    return spend / price


def choose_buy_asset(
    buy_gold: bool,
    buy_btc: bool,
    score_gold: float,
    score_btc: float,
) -> Optional[str]:
    if buy_gold and buy_btc:
        return "btc" if score_btc >= score_gold else "gold"
    if buy_btc:
        return "btc"
    if buy_gold:
        return "gold"
    return None
