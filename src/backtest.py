from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.indicators import compute_asset_signals
from src.portfolio import Portfolio
from src.strategy import choose_buy_asset, extreme_sell, extreme_trigger, should_sell, size_buy
from src.utils import calc_drawdown


def _state_label(state_map: Dict[str, bool]) -> str:
    active = [k for k, v in state_map.items() if v]
    if not active:
        return "none"
    if len(active) == 2:
        return "both"
    return active[0]


def run_backtest(price_df: pd.DataFrame, config: Dict):
    fees = config["fees"]
    momentum_cfg = config["momentum"]
    thresholds = config["thresholds"]
    selling_cfg = config["selling"]
    extreme_cfg = config["extreme"]
    no_buy_cfg = config["no_buy"]
    holding_cfg = config["holding"]
    buy_cfg = config["buy_sizing"]
    weight_cfg = config["weight_factor"]
    exec_cfg = config["execution"]
    signals_cfg = config.get("signals", {})
    buy_logic_cfg = config.get("buy_logic", {})

    portfolio = Portfolio(
        cash=config["run"]["initial_cash"],
        avg_cost_method=selling_cfg.get("avg_cost_method", "weighted_avg"),
    )

    trades: List[Dict] = []
    records: List[Dict] = []

    grad_hist = {"gold": [], "btc": []}
    extreme_active = {"gold": False, "btc": False}
    extreme_max_price = {"gold": 0.0, "btc": 0.0}
    no_buy_active = {"gold": False, "btc": False}
    no_buy_days_left = {"gold": 0, "btc": 0}
    no_buy_max_price = {"gold": 0.0, "btc": 0.0}
    hold_left = {"gold": 0, "btc": 0}

    price_gold_series = price_df["price_gold"].to_numpy()
    price_btc_series = price_df["price_btc"].to_numpy()

    w_values = []

    reversion_mode = signals_cfg.get("reversion_mode", "below_ma")
    reversion_pct = signals_cfg.get("reversion_pct")
    if reversion_mode == "below_ma_pct" and reversion_pct is None:
        reversion_pct = no_buy_cfg.get("rebuy_pct_N")
    score_mode = signals_cfg.get("score_mode", "max")
    score_threshold = float(signals_cfg.get("score_threshold", 0.0) or 0.0)
    ma_include_current = bool(signals_cfg.get("ma_include_current", True))
    require_full_window = bool(signals_cfg.get("require_full_window", False))
    weight_mode = weight_cfg.get("apply_mode", "compare")
    w_use = [s.lower() for s in weight_cfg.get("W_use", [])]
    use_t_plus_one = bool(weight_cfg.get("use_t_plus_one", True))
    buy_mode = buy_logic_cfg.get("mode", "single")
    buy_size_mode = buy_cfg.get("mode", "score")
    buy_fixed_fraction = buy_cfg.get("fixed_fraction", 0.5)
    skip_sell_on_buy = bool(selling_cfg.get("skip_if_buy_signal", True))

    strategy_mode = config.get("strategy", {}).get("mode", "full")
    baseline_mode = strategy_mode != "full"
    baseline_done = False

    for i, row in price_df.iterrows():
        date = row["date"]
        t_index = int(row["t"]) if "t" in row else i
        price_gold = float(row["price_gold"])
        price_btc = float(row["price_btc"])
        can_trade_gold = bool(row["is_trading_gold"])
        can_trade_btc = bool(row["is_trading_btc"])
        chosen_asset = "none"
        sell_flag_gold = False
        sell_flag_btc = False

        # decrement holding counters
        for asset in hold_left:
            if hold_left[asset] > 0:
                hold_left[asset] -= 1

        # update no-buy counters
        for asset in no_buy_days_left:
            if no_buy_days_left[asset] > 0:
                no_buy_days_left[asset] -= 1

        # compute signals (guard against missing prices)
        if np.isnan(price_gold):
            signal_gold = {
                "gradient": 0.0,
                "ma": price_gold,
                "price": price_gold,
                "momentum_buy": False,
                "reversion_buy": False,
                "buy_signal": False,
                "momentum_profit": 0.0,
                "reversion_profit": 0.0,
                "profit_score": 0.0,
            }
            can_trade_gold = False
        else:
            gold_prices = price_gold_series[: i + 1]
            signal_gold = compute_asset_signals(
                gold_prices,
                momentum_cfg["n_window"],
                momentum_cfg["w_scheme"],
                thresholds["thr_grad"],
                thresholds["thr_ma_diff"],
                reversion_mode=reversion_mode,
                reversion_pct=reversion_pct,
                score_mode=score_mode,
                ma_include_current=ma_include_current,
                require_full_window=require_full_window,
            )

        if np.isnan(price_btc):
            signal_btc = {
                "gradient": 0.0,
                "ma": price_btc,
                "price": price_btc,
                "momentum_buy": False,
                "reversion_buy": False,
                "buy_signal": False,
                "momentum_profit": 0.0,
                "reversion_profit": 0.0,
                "profit_score": 0.0,
            }
            can_trade_btc = False
        else:
            btc_prices = price_btc_series[: i + 1]
            signal_btc = compute_asset_signals(
                btc_prices,
                momentum_cfg["n_window"],
                momentum_cfg["w_scheme"],
                thresholds["thr_grad"],
                thresholds["thr_ma_diff"],
                reversion_mode=reversion_mode,
                reversion_pct=reversion_pct,
                score_mode=score_mode,
                ma_include_current=ma_include_current,
                require_full_window=require_full_window,
            )

        # weight factor (recorded even if baseline)
        t_for_weight = t_index + 1 if use_t_plus_one else max(t_index, 1)
        w = weight_cfg["W_C"] / (t_for_weight ** 2)
        w_values.append(w)

        def apply_weight(score: float, asset: str) -> float:
            if weight_mode not in {"compare", "score"}:
                return score
            if asset in w_use or f"profit_{asset}" in w_use or "all" in w_use:
                return score * w
            return score

        score_gold_raw = float(signal_gold["profit_score"])
        score_btc_raw = float(signal_btc["profit_score"])

        score_gold_for_threshold = score_gold_raw
        score_btc_for_threshold = score_btc_raw
        if weight_mode == "score":
            score_gold_for_threshold = apply_weight(score_gold_raw, "gold")
            score_btc_for_threshold = apply_weight(score_btc_raw, "btc")

        compare_gold = score_gold_raw
        compare_btc = score_btc_raw
        if weight_mode in {"compare", "score"}:
            compare_gold = apply_weight(score_gold_raw, "gold")
            compare_btc = apply_weight(score_btc_raw, "btc")

        buy_gold_signal = bool(signal_gold["buy_signal"])
        buy_btc_signal = bool(signal_btc["buy_signal"])
        if score_threshold > 0:
            buy_gold_signal = buy_gold_signal and (score_gold_for_threshold > score_threshold)
            buy_btc_signal = buy_btc_signal and (score_btc_for_threshold > score_threshold)

        if baseline_mode:
            if (not baseline_done) and can_trade_btc:
                qty = portfolio.cash / (price_btc * (1 + fees["fee_btc"]))
                if qty > 0:
                    trade = portfolio.buy("btc", qty, price_btc, fees["fee_btc"])
                    trade.update(
                        {"date": date, "t": t_index, "reason": "baseline_buy", "signal_score": 0.0}
                    )
                    trades.append(trade)
                    baseline_done = True
                    hold_left["btc"] = holding_cfg["hold_days_T"]

            nav_total, nav_cash, nav_gold, nav_btc = portfolio.nav(price_gold, price_btc)
            records.append(
                {
                    "date": date,
                    "t": t_index,
                    "price_gold": price_gold,
                    "price_btc": price_btc,
                    "cash": portfolio.cash,
                    "gold_qty": portfolio.positions["gold"],
                    "btc_qty": portfolio.positions["btc"],
                    "nav_total": nav_total,
                    "nav_cash": nav_cash,
                    "nav_gold": nav_gold,
                    "nav_btc": nav_btc,
                    "state_extreme": "none",
                    "state_no_buy": "none",
                    "hold_left_gold": hold_left["gold"],
                    "hold_left_btc": hold_left["btc"],
                    "profit_gold": signal_gold["profit_score"],
                    "profit_btc": signal_btc["profit_score"],
                    "chosen_buy_asset": "btc" if baseline_done else "none",
                    "sell_flag_gold": False,
                    "sell_flag_btc": False,
                }
            )
            continue

        # extreme detection
        avg_grad_gold = float(np.mean(grad_hist["gold"])) if grad_hist["gold"] else 0.0
        avg_grad_btc = float(np.mean(grad_hist["btc"])) if grad_hist["btc"] else 0.0
        if portfolio.positions["gold"] > 0 and extreme_trigger(
            signal_gold["gradient"], avg_grad_gold, extreme_cfg["extreme_C"]
        ):
            extreme_active["gold"] = True
            extreme_max_price["gold"] = max(extreme_max_price["gold"], price_gold)
        if portfolio.positions["btc"] > 0 and extreme_trigger(
            signal_btc["gradient"], avg_grad_btc, extreme_cfg["extreme_C"]
        ):
            extreme_active["btc"] = True
            extreme_max_price["btc"] = max(extreme_max_price["btc"], price_btc)

        # update extreme max price while active
        if extreme_active["gold"]:
            extreme_max_price["gold"] = max(extreme_max_price["gold"], price_gold)
        if extreme_active["btc"]:
            extreme_max_price["btc"] = max(extreme_max_price["btc"], price_btc)

        grad_hist["gold"].append(signal_gold["gradient"])
        grad_hist["btc"].append(signal_btc["gradient"])

        # refresh no-buy state based on price threshold
        no_buy_mode = no_buy_cfg.get("mode", "both")
        for asset, price in [("gold", price_gold), ("btc", price_btc)]:
            if no_buy_active[asset]:
                allow_by_days = no_buy_days_left[asset] <= 0
                allow_by_price = price <= no_buy_cfg["rebuy_pct_N"] * no_buy_max_price[asset]
                if no_buy_mode == "days" and allow_by_days:
                    no_buy_active[asset] = False
                elif no_buy_mode == "price" and allow_by_price:
                    no_buy_active[asset] = False
                elif no_buy_mode == "either" and (allow_by_days or allow_by_price):
                    no_buy_active[asset] = False
                elif no_buy_mode == "both" and (allow_by_days and allow_by_price):
                    no_buy_active[asset] = False

        def log_blocked(asset: str, side: str, reason: str, price: float):
            trades.append(
                {
                    "date": date,
                    "t": t_index,
                    "asset": asset,
                    "side": side,
                    "qty": 0.0,
                    "price": price,
                    "notional": 0.0,
                    "fee_paid": 0.0,
                    "cash_change": 0.0,
                    "reason": reason,
                    "signal_score": 0.0,
                }
            )

        # daily order
        if any(extreme_active.values()):
            # only allow extreme sells, skip rest
            for asset, price, fee, can_trade in [
                ("gold", price_gold, fees["fee_gold"], can_trade_gold),
                ("btc", price_btc, fees["fee_btc"], can_trade_btc),
            ]:
                if not extreme_active[asset]:
                    continue
                if portfolio.positions[asset] <= 0:
                    continue
                if not can_trade:
                    log_blocked(asset, "sell", "blocked_by_calendar", price)
                    continue
                if hold_left[asset] > 0:
                    log_blocked(asset, "sell", "blocked_by_hold", price)
                    continue
                if extreme_sell(
                    price,
                    extreme_max_price[asset],
                    extreme_cfg["extreme_sell_pct_E"],
                    extreme_cfg.get("extreme_deltaP"),
                ):
                    trade = portfolio.sell(asset, portfolio.positions[asset], price, fee)
                    trade.update(
                        {
                            "date": date,
                            "t": t_index,
                            "reason": "extreme_sell",
                            "signal_score": signal_gold["gradient"]
                            if asset == "gold"
                            else signal_btc["gradient"],
                        }
                    )
                    trades.append(trade)
                    extreme_active[asset] = False
                    no_buy_active[asset] = True
                    no_buy_days_left[asset] = no_buy_cfg.get("no_buy_days", 0)
                    no_buy_max_price[asset] = extreme_max_price[asset]
                    extreme_max_price[asset] = 0.0
        else:
            # normal selling
            for asset, price, fee, can_trade in [
                ("gold", price_gold, fees["fee_gold"], can_trade_gold),
                ("btc", price_btc, fees["fee_btc"], can_trade_btc),
            ]:
                if portfolio.positions[asset] > 0:
                    buy_signal_active = buy_gold_signal if asset == "gold" else buy_btc_signal
                    if skip_sell_on_buy and buy_signal_active:
                        continue
                    net_sell_price = price * (1 - fee)
                    avg_cost = portfolio.avg_cost[asset]
                    sell_signal = should_sell(net_sell_price, avg_cost, selling_cfg["margin_L"])
                    if asset == "gold":
                        sell_flag_gold = sell_signal
                    else:
                        sell_flag_btc = sell_signal
                    if sell_signal and (not can_trade):
                        log_blocked(asset, "sell", "blocked_by_calendar", price)
                    elif sell_signal and hold_left[asset] > 0:
                        log_blocked(asset, "sell", "blocked_by_hold", price)
                    elif sell_signal and can_trade and hold_left[asset] == 0:
                        trade = portfolio.sell(asset, portfolio.positions[asset], price, fee)
                        trade.update(
                            {
                                "date": date,
                                "t": t_index,
                                "reason": "sell_signal",
                                "signal_score": net_sell_price - avg_cost - selling_cfg["margin_L"],
                            }
                        )
                        trades.append(trade)

            # buy decision
            def attempt_buy(asset: str, price: float, fee: float, can_trade: bool, score: float) -> bool:
                if no_buy_active[asset]:
                    log_blocked(asset, "buy", "blocked_by_no_buy", price)
                    return False
                if not can_trade:
                    log_blocked(asset, "buy", "blocked_by_calendar", price)
                    return False
                qty = size_buy(
                    portfolio.cash,
                    price,
                    score,
                    buy_cfg["buy_scale"],
                    buy_cfg["buy_cap"],
                    buy_cfg["buy_min_cash_reserve"],
                    mode=buy_size_mode,
                    fixed_fraction=buy_fixed_fraction,
                )
                if qty > 0:
                    trade = portfolio.buy(asset, qty, price, fee)
                    reason = "buy_momentum" if (
                        (asset == "gold" and signal_gold["momentum_buy"])
                        or (asset == "btc" and signal_btc["momentum_buy"])
                    ) else "buy_reversion"
                    trade.update({"date": date, "t": t_index, "reason": reason, "signal_score": score})
                    trades.append(trade)
                    hold_left[asset] = holding_cfg["hold_days_T"]
                    return True
                log_blocked(asset, "buy", "blocked_by_cash", price)
                return False

            if buy_mode == "multi":
                bought = []
                if buy_btc_signal:
                    if attempt_buy("btc", price_btc, fees["fee_btc"], can_trade_btc, score_btc_for_threshold):
                        bought.append("btc")
                if buy_gold_signal:
                    if attempt_buy("gold", price_gold, fees["fee_gold"], can_trade_gold, score_gold_for_threshold):
                        bought.append("gold")
                if bought:
                    if len(bought) == 2:
                        chosen_asset = "btc" if compare_btc >= compare_gold else "gold"
                    else:
                        chosen_asset = bought[0]
            else:
                if buy_gold_signal and no_buy_active["gold"]:
                    log_blocked("gold", "buy", "blocked_by_no_buy", price_gold)
                if buy_btc_signal and no_buy_active["btc"]:
                    log_blocked("btc", "buy", "blocked_by_no_buy", price_btc)
                if buy_gold_signal and not can_trade_gold:
                    log_blocked("gold", "buy", "blocked_by_calendar", price_gold)
                if buy_btc_signal and not can_trade_btc:
                    log_blocked("btc", "buy", "blocked_by_calendar", price_btc)

                buy_gold = buy_gold_signal and (not no_buy_active["gold"]) and can_trade_gold
                buy_btc = buy_btc_signal and (not no_buy_active["btc"]) and can_trade_btc

                chosen = choose_buy_asset(buy_gold, buy_btc, compare_gold, compare_btc)
                if chosen:
                    chosen_asset = chosen
                    score = score_gold_for_threshold if chosen == "gold" else score_btc_for_threshold
                    price = price_gold if chosen == "gold" else price_btc
                    fee = fees["fee_gold"] if chosen == "gold" else fees["fee_btc"]
                    can_trade = can_trade_gold if chosen == "gold" else can_trade_btc

                    if not can_trade:
                        log_blocked(chosen, "buy", "blocked_by_calendar", price)
                    else:
                        qty = size_buy(
                            portfolio.cash,
                            price,
                            score,
                            buy_cfg["buy_scale"],
                            buy_cfg["buy_cap"],
                            buy_cfg["buy_min_cash_reserve"],
                            mode=buy_size_mode,
                            fixed_fraction=buy_fixed_fraction,
                        )
                        if qty > 0:
                            trade = portfolio.buy(chosen, qty, price, fee)
                            reason = "buy_momentum" if (
                                (chosen == "gold" and signal_gold["momentum_buy"])
                                or (chosen == "btc" and signal_btc["momentum_buy"])
                            ) else "buy_reversion"
                            trade.update(
                                {"date": date, "t": t_index, "reason": reason, "signal_score": score}
                            )
                            trades.append(trade)
                            hold_left[chosen] = holding_cfg["hold_days_T"]
                        else:
                            log_blocked(chosen, "buy", "blocked_by_cash", price)
                else:
                    if buy_gold_signal and (not no_buy_active["gold"]) and can_trade_gold:
                        log_blocked("gold", "buy", "blocked_by_cash", price_gold)
                    if buy_btc_signal and (not no_buy_active["btc"]) and can_trade_btc:
                        log_blocked("btc", "buy", "blocked_by_cash", price_btc)

        nav_total, nav_cash, nav_gold, nav_btc = portfolio.nav(price_gold, price_btc)
        records.append(
            {
                "date": date,
                "t": t_index,
                "price_gold": price_gold,
                "price_btc": price_btc,
                "cash": portfolio.cash,
                "gold_qty": portfolio.positions["gold"],
                "btc_qty": portfolio.positions["btc"],
                "nav_total": nav_total,
                "nav_cash": nav_cash,
                "nav_gold": nav_gold,
                "nav_btc": nav_btc,
                "state_extreme": _state_label(extreme_active),
                "state_no_buy": _state_label(no_buy_active),
                "hold_left_gold": hold_left["gold"],
                "hold_left_btc": hold_left["btc"],
                "profit_gold": signal_gold["profit_score"],
                "profit_btc": signal_btc["profit_score"],
                "chosen_buy_asset": chosen_asset,
                "sell_flag_gold": sell_flag_gold,
                "sell_flag_btc": sell_flag_btc,
            }
        )

    results_df = pd.DataFrame(records)
    results_df["dd"] = calc_drawdown(results_df["nav_total"])
    trade_cols = [
        "date",
        "t",
        "asset",
        "side",
        "qty",
        "price",
        "notional",
        "fee_paid",
        "cash_change",
        "reason",
        "signal_score",
    ]
    trades_df = pd.DataFrame(trades, columns=trade_cols)

    w_stats = {
        "W_C": weight_cfg["W_C"],
        "W_use": weight_cfg.get("W_use", []),
        "W_min": float(min(w_values)) if w_values else None,
        "W_max": float(max(w_values)) if w_values else None,
    }
    return results_df, trades_df, w_stats
