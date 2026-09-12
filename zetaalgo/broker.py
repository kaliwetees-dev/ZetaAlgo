"""Position accounting, position sizing and transaction costs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .config import BacktestConfig


@dataclass
class Position:
    """An open long position and the levels currently protecting it."""

    qty: float
    entry_price: float
    entry_index: int
    entry_ts: datetime
    stop_price: float
    target_price: Optional[float]
    initial_stop: float
    session: str
    direction: int = 1  # +1 long, -1 short
    entry_commission: float = 0.0
    breakeven_done: bool = False
    mfe: float = 0.0  # max favourable excursion, in price
    mae: float = 0.0  # max adverse excursion, in price

    @property
    def risk_per_share(self) -> float:
        """Distance from entry to the *initial* stop: the definition of 1R."""
        return max(1e-12, abs(self.entry_price - self.initial_stop))


@dataclass
class Trade:
    """A completed round trip."""

    entry_ts: datetime
    exit_ts: datetime
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    qty: float
    initial_stop: float
    target_price: Optional[float]
    direction: int
    exit_reason: str
    gross_pnl: float
    commission: float
    net_pnl: float
    r_multiple: float
    bars_held: int
    mfe_r: float
    mae_r: float
    session: str
    equity_after: float

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    @property
    def is_long(self) -> bool:
        return self.direction > 0

    @property
    def return_pct(self) -> float:
        notional = self.entry_price * self.qty
        return self.net_pnl / notional if notional else 0.0


def commission(qty: float, price: float, config: BacktestConfig) -> float:
    """Per-side commission: the larger of the bps, per-share and floor terms."""
    if qty <= 0:
        return 0.0
    bps = price * qty * config.commission_bps / 10_000.0
    per_share = qty * config.commission_per_share
    return max(bps + per_share, config.min_commission)


def position_size(
    equity: float, entry_price: float, stop_price: float, config: BacktestConfig
) -> float:
    """Quantity to buy, before the leverage cap is applied.

    ``risk`` sizing is the mode that matches how this setup defines a trade:
    the stop is structural (the cross candle's low), so risking a constant
    fraction of equity per trade keeps every trade the same size in R terms
    regardless of how wide that candle happened to be.
    """
    if entry_price <= 0:
        return 0.0
    risk_per_share = abs(entry_price - stop_price)
    if config.sizing == "fixed":
        qty = config.fixed_qty
    elif config.sizing == "notional":
        qty = (equity * config.notional_pct) / entry_price
    else:  # risk-based
        if risk_per_share <= 0:
            return 0.0
        qty = (equity * config.risk_pct) / risk_per_share

    # Leverage cap: never deploy more than max_notional_pct of equity.
    max_qty = (equity * config.max_notional_pct) / entry_price
    qty = min(qty, max_qty)
    if not config.allow_fractional_qty:
        qty = float(math.floor(qty))
    return max(0.0, qty)
