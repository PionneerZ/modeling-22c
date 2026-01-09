from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Portfolio:
    cash: float
    avg_cost_method: str = "weighted_avg"
    positions: Dict[str, float] = field(default_factory=lambda: {"gold": 0.0, "btc": 0.0})
    avg_cost: Dict[str, float] = field(default_factory=lambda: {"gold": 0.0, "btc": 0.0})
    lots: Dict[str, List[Tuple[float, float]]] = field(
        default_factory=lambda: {"gold": [], "btc": []}
    )

    def buy(self, asset: str, qty: float, price: float, fee_rate: float):
        notional = qty * price
        fee_paid = notional * fee_rate
        cash_change = -(notional + fee_paid)
        self.cash += cash_change

        prev_qty = self.positions[asset]
        new_qty = prev_qty + qty
        self.positions[asset] = new_qty

        if self.avg_cost_method == "weighted_avg":
            if new_qty > 0:
                prev_cost = self.avg_cost[asset] * prev_qty
                added_cost = price * (1 + fee_rate) * qty
                self.avg_cost[asset] = (prev_cost + added_cost) / new_qty
        else:
            self.lots[asset].append((qty, price * (1 + fee_rate)))
            self.avg_cost[asset] = self._fifo_avg_cost(asset)

        return {
            "asset": asset,
            "side": "buy",
            "qty": qty,
            "price": price,
            "notional": notional,
            "fee_paid": fee_paid,
            "cash_change": cash_change,
        }

    def sell(self, asset: str, qty: float, price: float, fee_rate: float):
        notional = qty * price
        fee_paid = notional * fee_rate
        cash_change = notional - fee_paid
        self.cash += cash_change

        self.positions[asset] -= qty
        if self.positions[asset] <= 0:
            self.positions[asset] = 0.0
            self.avg_cost[asset] = 0.0
            self.lots[asset] = []
        else:
            if self.avg_cost_method != "weighted_avg":
                self._fifo_reduce(asset, qty)
                self.avg_cost[asset] = self._fifo_avg_cost(asset)

        return {
            "asset": asset,
            "side": "sell",
            "qty": qty,
            "price": price,
            "notional": notional,
            "fee_paid": fee_paid,
            "cash_change": cash_change,
        }

    def nav(self, price_gold: float, price_btc: float):
        nav_gold = self.positions["gold"] * price_gold
        nav_btc = self.positions["btc"] * price_btc
        nav_cash = self.cash
        nav_total = nav_cash + nav_gold + nav_btc
        return nav_total, nav_cash, nav_gold, nav_btc

    def _fifo_reduce(self, asset: str, qty: float) -> None:
        remaining = qty
        new_lots = []
        for lot_qty, lot_cost in self.lots[asset]:
            if remaining <= 0:
                new_lots.append((lot_qty, lot_cost))
                continue
            if lot_qty <= remaining:
                remaining -= lot_qty
            else:
                new_lots.append((lot_qty - remaining, lot_cost))
                remaining = 0.0
        self.lots[asset] = new_lots

    def _fifo_avg_cost(self, asset: str) -> float:
        total_qty = sum(q for q, _ in self.lots[asset])
        if total_qty == 0:
            return 0.0
        total_cost = sum(q * c for q, c in self.lots[asset])
        return total_cost / total_qty
