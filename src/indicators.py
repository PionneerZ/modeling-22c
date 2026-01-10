from __future__ import annotations

from typing import Dict

import numpy as np


def _make_weights(n: int, scheme: str) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    scheme = scheme.lower()
    if scheme == "equal":
        w = np.ones(n, dtype=float)
    elif scheme == "linear":
        # Higher weight for more recent prices
        w = np.arange(n, 0, -1, dtype=float)
    elif scheme == "exp":
        w = np.exp(-np.arange(n, dtype=float))
    else:
        raise ValueError(f"unsupported weight scheme: {scheme}")
    return w


def weighted_moving_average(prices: np.ndarray, window: int, scheme: str) -> float:
    if prices.size == 0:
        return float("nan")
    n = min(window, prices.size)
    w = _make_weights(n, scheme)
    window_prices = prices[-n:]
    return float(np.dot(window_prices, w) / np.sum(w))


def price_gradient(prices: np.ndarray, window: int) -> float:
    if prices.size == 0:
        return 0.0
    if window <= 1:
        return 0.0
    start_idx = max(0, prices.size - window + 1)
    return float(prices[-1] - prices[start_idx])


def compute_asset_signals(
    prices: np.ndarray,
    window: int,
    w_scheme: str,
    thr_grad: float,
    thr_ma_diff: float,
    reversion_mode: str = "below_ma",
    reversion_pct: float | None = None,
    score_mode: str = "max",
    ma_include_current: bool = True,
    require_full_window: bool = False,
) -> Dict[str, float | bool]:
    if prices.size == 0:
        return {
            "gradient": 0.0,
            "ma": float("nan"),
            "price": float("nan"),
            "momentum_buy": False,
            "reversion_buy": False,
            "buy_signal": False,
            "momentum_profit": 0.0,
            "reversion_profit": 0.0,
            "profit_score": 0.0,
        }

    price_now = float(prices[-1])
    ma_prices = prices if ma_include_current else prices[:-1]
    if require_full_window and ma_prices.size < window:
        ma = float("nan")
    else:
        ma = weighted_moving_average(ma_prices, window, w_scheme) if ma_prices.size else float("nan")

    if require_full_window and prices.size < window:
        gradient = 0.0
    else:
        gradient = price_gradient(prices, window)

    momentum_buy = False
    if not np.isnan(ma):
        momentum_buy = (gradient > thr_grad) and ((price_now - ma) > thr_ma_diff)
    momentum_profit = gradient if momentum_buy and gradient > 0 else 0.0

    reversion_buy = False
    if not np.isnan(ma):
        if reversion_mode == "below_ma":
            reversion_buy = price_now < ma
        elif reversion_mode == "below_ma_pct":
            if reversion_pct is not None:
                reversion_buy = price_now < (ma * reversion_pct)
        else:
            raise ValueError(f"unsupported reversion_mode: {reversion_mode}")
    reversion_profit = (ma - price_now) ** 2 if reversion_buy else 0.0

    buy_signal = momentum_buy or reversion_buy
    if score_mode == "sum":
        profit_score = momentum_profit + reversion_profit
    elif score_mode == "max":
        profit_score = max(momentum_profit, reversion_profit)
    else:
        raise ValueError(f"unsupported score_mode: {score_mode}")

    return {
        "gradient": gradient,
        "ma": ma,
        "price": price_now,
        "momentum_buy": momentum_buy,
        "reversion_buy": reversion_buy,
        "buy_signal": buy_signal,
        "momentum_profit": momentum_profit,
        "reversion_profit": reversion_profit,
        "profit_score": profit_score,
    }
