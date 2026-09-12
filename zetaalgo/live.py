"""Automated execution of the setup against a broker.

The point of this module is that the *decisions* come from exactly the same
``EmaVwapCrossoverStrategy`` object that was backtested.  Nothing about the
rules is re-implemented here; this layer only turns the strategy's state into
broker orders and keeps the two in sync.

Design notes
------------
*Closed bars only.*  ``LiveTrader.on_bar`` must be called once per **closed**
bar.  Acting on a forming bar means acting on a value that can still change,
which is the most common way a live system stops matching its backtest.

*State is rebuilt from a rolling window.*  Rather than maintaining a second,
incremental copy of the indicator math (a second chance to be wrong), the
trader re-runs the strategy over a trailing window of bars each time.  The
window always spans the whole current session, so the session-anchored VWAP
and the setup state come out identical to the backtest.

*Entries rest at the broker.*  The trigger is a buy-stop at the cross
candle's high, submitted with its protective stop attached, so the breakout
is not missed between bars.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from .broker import commission, position_size
from .config import BacktestConfig, StrategyConfig
from .data import Bar
from .strategy import EmaVwapCrossoverStrategy

logger = logging.getLogger("zetaalgo.live")


@dataclass
class Order:
    """An order intent handed to a broker adapter."""

    side: str  # buy | sell
    qty: float
    order_type: str  # market | stop | limit
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    # Target expressed as a multiple of *realised* risk.  A breakout order can
    # fill worse than its trigger, and a target fixed before the fill would
    # then pay less than the intended R while still risking a full 1R, so the
    # broker resolves this against the actual fill price.
    target_r: Optional[float] = None
    direction: int = 1  # +1 long, -1 short
    fill_through_ticks: int = 0
    reason: str = ""
    ts: Optional[datetime] = None

    def __str__(self) -> str:
        parts = [f"{self.side.upper()} {self.qty:g} {self.order_type}"]
        if self.price is not None:
            parts.append(f"@ {self.price:.2f}")
        if self.stop_loss is not None:
            parts.append(f"SL {self.stop_loss:.2f}")
        if self.take_profit is not None:
            parts.append(f"TP {self.take_profit:.2f}")
        parts.append(f"({self.reason})")
        return " ".join(parts)


@dataclass
class Fill:
    ts: datetime
    side: str
    qty: float
    price: float
    reason: str


class BrokerAdapter(ABC):
    """Minimal broker interface the trader needs.

    Implement this against a real API (Alpaca, IBKR, a crypto exchange, ...).
    Keep the semantics below exactly as documented or the live results will
    drift away from the backtest.
    """

    @abstractmethod
    def equity(self) -> float:
        """Current account equity in account currency."""

    @abstractmethod
    def position_qty(self) -> float:
        """Signed position size for the traded symbol (0 when flat)."""

    @abstractmethod
    def submit(self, order: Order) -> str:
        """Send an order; return a broker order id."""

    @abstractmethod
    def cancel(self, order_id: str) -> None:
        """Cancel a working order.  Must be safe to call on a filled order."""

    @abstractmethod
    def update_stop(self, price: float) -> None:
        """Move the protective stop of the open position."""

    @abstractmethod
    def position_avg_price(self) -> float:
        """Average fill price of the open position (0 when flat).

        The trader needs the *actual* fill, not the price it asked for: a
        breakout order that gaps through its trigger fills worse than planned,
        and a target sized off the trigger would then pay out less than the
        intended R multiple while still risking a full 1R.
        """

    @abstractmethod
    def update_target(self, price: float) -> None:
        """Set or move the profit target of the open position."""


class PaperBroker(BrokerAdapter):
    """An in-memory broker that fills orders from the bars it is shown.

    Useful for dry-running the automation end to end.  Fill assumptions match
    the backtest engine's (see ``backtest.py``): resting buy-stops fill at the
    trigger or the open if the bar gapped through it, stops fill at the stop
    price with slippage, targets fill at the limit price.
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        tick_size: float = 0.01,
        entry_bar_stop: str = "close",
    ) -> None:
        self.config = config or BacktestConfig()
        self.tick_size = tick_size
        self.entry_bar_stop = entry_bar_stop
        self.cash = self.config.initial_equity
        self.qty = 0.0
        self.avg_price = 0.0
        self.working: Dict[str, Order] = {}
        self.stop_price: Optional[float] = None
        self.target_price: Optional[float] = None
        self.entry_kind: str = "stop"
        self.fills: List[Fill] = []
        self._next_id = 1
        self._last_close = 0.0

    # -- BrokerAdapter -------------------------------------------------
    def equity(self) -> float:
        return self.cash + self.qty * self._last_close

    def position_qty(self) -> float:
        return self.qty

    def submit(self, order: Order) -> str:
        order_id = f"paper-{self._next_id}"
        self._next_id += 1
        if order.order_type == "market":
            self._fill(order, order.price if order.price is not None else self._last_close)
        else:
            self.working[order_id] = order
            logger.info("working order %s: %s", order_id, order)
        return order_id

    def cancel(self, order_id: str) -> None:
        removed = self.working.pop(order_id, None)
        if removed is not None:
            logger.info("cancelled %s: %s", order_id, removed)

    def update_stop(self, price: float) -> None:
        """Tighten the stop only: up for a long, down for a short."""
        if self.qty == 0:
            return
        way = 1 if self.qty > 0 else -1
        if self.stop_price is None or way * (price - self.stop_price) > 0:
            logger.info("stop moved to %.2f", price)
            self.stop_price = price

    def position_avg_price(self) -> float:
        return self.avg_price if self.qty != 0 else 0.0

    def update_target(self, price: float) -> None:
        if self.qty != 0:
            self.target_price = price

    # -- simulation ----------------------------------------------------
    def on_bar(self, bar: Bar) -> None:
        """Advance the simulation over one bar: brackets first, then entries."""
        self._last_close = bar.close
        slip = self.config.slippage_ticks * self.tick_size

        if self.qty != 0:
            way = 1 if self.qty > 0 else -1
            if self.stop_price is not None:
                touched = (bar.low <= self.stop_price if way > 0
                           else bar.high >= self.stop_price)
                if touched:
                    gapped = (bar.open <= self.stop_price if way > 0
                              else bar.open >= self.stop_price)
                    if gapped:
                        price = bar.open
                    else:
                        price = self.stop_price - way * slip
                    price = (max(bar.low, price) if way > 0 else min(bar.high, price))
                    self._exit(bar, price, "stop_gap" if gapped else "stop")
                    return
            if self.target_price is not None:
                reached = (bar.high >= self.target_price if way > 0
                           else bar.low <= self.target_price)
                if reached:
                    self._exit(bar, self.target_price, "target")
                    return

        if self.qty == 0:
            for order_id, order in list(self.working.items()):
                if order.order_type not in ("stop", "limit") or order.price is None:
                    continue
                way = order.direction
                through = order.fill_through_ticks * self.tick_size
                if order.order_type == "limit":
                    # Pullback entry: price has to come BACK to the level.
                    touched = (bar.low <= order.price - through if way > 0
                               else bar.high >= order.price + through)
                else:
                    touched = (bar.high >= order.price if way > 0
                               else bar.low <= order.price)
                if touched:
                    if order.order_type == "limit":
                        fill = (min(bar.open, order.price) if way > 0
                                else max(bar.open, order.price))
                    elif way > 0:
                        fill = min(max(bar.open, order.price) + slip, bar.high)
                    else:
                        fill = max(min(bar.open, order.price) - slip, bar.low)
                    self.working.pop(order_id, None)
                    self.entry_kind = order.order_type
                    self._fill(order, fill, ts=bar.ts)
                    self._entry_bar_brackets(bar, slip)

    def _entry_bar_brackets(self, bar: Bar, slip: float) -> None:
        """Apply stop/target to the bar that opened the position.

        Mirrors the backtest engine: the stop follows the configured entry-bar
        policy, while the target is always live because the range above the
        trigger is only reachable after the trigger was crossed.
        """
        if self.qty == 0:
            return
        way = 1 if self.qty > 0 else -1
        if self.entry_kind == "limit":
            # Mirrors backtest.py: a limit fills as price moves against the
            # trade, so this bar's favourable extreme may predate the fill.
            # Only outcomes the close proves are taken.
            if self.stop_price is not None and way * (bar.close - self.stop_price) <= 0:
                self._exit(bar, bar.close - way * slip, "stop")
            elif self.target_price is not None and way * (bar.close - self.target_price) >= 0:
                self._exit(bar, self.target_price, "target")
            return
        if self.stop_price is not None:
            touched = (bar.low <= self.stop_price if way > 0
                       else bar.high >= self.stop_price)
            if self.entry_bar_stop == "low" and touched:
                price = (max(bar.low, self.stop_price - slip) if way > 0
                         else min(bar.high, self.stop_price + slip))
                self._exit(bar, price, "stop")
                return
            if (self.entry_bar_stop == "close"
                    and way * (bar.close - self.stop_price) <= 0):
                self._exit(bar, bar.close - way * slip, "stop")
                return
        if self.target_price is not None:
            reached = (bar.high >= self.target_price if way > 0
                       else bar.low <= self.target_price)
            if reached:
                self._exit(bar, self.target_price, "target")

    def _fill(self, order: Order, price: float, ts: Optional[datetime] = None) -> None:
        # A resting limit provided liquidity, so it earns the maker rate --
        # as does a limit take-profit.  Stops and forced exits cross.
        maker = order.order_type == "limit" or order.reason == "target"
        fee = commission(order.qty, price, self.config, maker=maker)
        stamp = ts or order.ts or datetime.min
        opening = self.qty == 0
        if opening:
            way = order.direction
            # Respect available cash exactly as the backtest engine does, so a
            # fill beyond the trigger cannot quietly buy on margin here and
            # make the paper run diverge from the backtest.  A short receives
            # proceeds rather than paying cash, so only longs are capped.
            if way > 0 and order.qty * price + fee > self.cash:
                capped = float(int((self.cash - fee) / price)) if price > 0 else 0.0
                if capped <= 0:
                    logger.warning("insufficient cash for %s; order dropped", order)
                    return
                order = replace(order, qty=capped)
                fee = commission(order.qty, price, self.config, maker=maker)
            self.cash -= way * order.qty * price + fee
            self.qty = way * order.qty
            self.avg_price = price
            self.stop_price = order.stop_loss
            risk = way * (price - order.stop_loss) if order.stop_loss is not None else 0.0
            if order.target_r and risk > 0:
                self.target_price = price + way * order.target_r * risk
            else:
                self.target_price = order.take_profit
        else:
            # Closing: the order side is the opposite of the position.
            way = 1 if self.qty > 0 else -1
            self.cash += way * abs(self.qty) * price - fee
            self.qty = 0.0
            self.stop_price = self.target_price = None
        self.fills.append(Fill(stamp, order.side, order.qty, price, order.reason))
        logger.info("FILL %s %g @ %.2f (%s)", order.side, order.qty, price, order.reason)

    def _exit(self, bar: Bar, price: float, reason: str) -> None:
        self.submit(
            Order(
                side="sell" if self.qty > 0 else "buy",
                qty=abs(self.qty),
                order_type="market",
                price=price,
                direction=1 if self.qty > 0 else -1,
                reason=reason,
                ts=bar.ts,
            )
        )


class LiveTrader:
    """Drives the strategy against a broker adapter, one closed bar at a time."""

    def __init__(
        self,
        broker: BrokerAdapter,
        strategy_config: Optional[StrategyConfig] = None,
        backtest_config: Optional[BacktestConfig] = None,
        window: int = 800,
        strategy: Optional[object] = None,
    ) -> None:
        """``strategy`` may be any object implementing the strategy interface."""
        self.broker = broker
        if strategy is not None:
            self.strategy = strategy
            self.strategy_config = getattr(strategy, "config", strategy_config)
        else:
            self.strategy_config = strategy_config or StrategyConfig()
            self.strategy = EmaVwapCrossoverStrategy(self.strategy_config)
        self.config = backtest_config or BacktestConfig()
        self.window = window
        self.bars: List[Bar] = []
        self.entry_order_id: Optional[str] = None
        self.entry_index: Optional[int] = None
        self.orders: List[Order] = []
        self.planned_stop: Optional[float] = None
        self.traded_setups: set = set()
        self.working_setup_key: Optional[datetime] = None
        self.session_trades: Dict[str, int] = {}
        self.bars_seen: int = 0
        self.cooldown_until: int = -1
        self.entry_fill: Optional[float] = None
        # Bars-held is tracked against the monotonic bar counter, never against
        # a window-relative index: the rolling window slides, so an index
        # stored at entry stops pointing at the entry bar.
        self.entry_bars_seen: Optional[int] = None
        self.direction: int = 1
        self.entry_kind: str = "stop"
        self.mfe: float = 0.0
        self.breakeven_done: bool = False

    # ------------------------------------------------------------------
    def warm_up(self, bars: Sequence[Bar]) -> None:
        """Seed history with past bars without trading them."""
        self.bars = list(bars)[-self.window :]

    def on_bar(self, bar: Bar, last_of_session: bool = False) -> List[Order]:
        """Process one **closed** bar; returns the orders submitted."""
        self.bars.append(bar)
        self._trim_window()
        self.bars_seen += 1
        index = len(self.bars) - 1
        submitted: List[Order] = []

        # Rebuild strategy state over the window (see module docstring).
        self._rebuild_state(index)

        if self.broker.position_qty() != 0 and self.entry_index is None:
            # A resting order filled: reconcile the bracket against the real
            # fill price before anything else.
            self.entry_index = index
            self.entry_bars_seen = self.bars_seen
            self.entry_order_id = None
            if self.working_setup_key is not None:
                self.traded_setups.add(self.working_setup_key)
                self.working_setup_key = None
            self.session_trades[bar.session] = self.session_trades.get(bar.session, 0) + 1
            self._reconcile_bracket()

        # --- manage an open position ------------------------------------
        if self.broker.position_qty() != 0:
            bars_held = self.bars_held()
            reason = self.strategy.close_exit_reason(
                index, index - bars_held, self.direction
            )
            if reason is None and self.strategy_config.flat_at_session_end and last_of_session:
                reason = "session_end"
            if reason is not None:
                submitted.append(self._flatten(bar, reason))
            else:
                self._ratchet_stop(index, bar, entry_bar=bars_held == 0)

        # --- place/refresh the resting entry order ----------------------
        # Reached even on a bar where the position was just closed, because the
        # backtest engine also re-arms from the next bar onwards.
        if self.broker.position_qty() == 0:
            self._reset_position_state()
            plan_next = self._eligible_plan(index, last_of_session)
            if self.entry_order_id is not None and plan_next is None:
                self.broker.cancel(self.entry_order_id)
                self.entry_order_id = None
                self.working_setup_key = None
            if plan_next is not None and self.entry_order_id is None:
                order = self._build_entry_order(bar, plan_next)
                if order is not None:
                    self.planned_stop = plan_next.stop_price
                    self.direction = plan_next.direction
                    self.entry_kind = plan_next.order_kind
                    self.entry_order_id = self.broker.submit(order)
                    self.working_setup_key = self._setup_key(plan_next)
                    submitted.append(order)

        self.orders.extend(submitted)
        return submitted

    # ------------------------------------------------------------------
    def bars_held(self) -> int:
        """Bars elapsed since the entry fill, robust to window sliding."""
        if self.entry_bars_seen is None:
            return 0
        return max(0, self.bars_seen - self.entry_bars_seen)

    def _eligible_plan(self, index: int, last_of_session: bool):
        """The entry plan for the next bar, after the non-signal guards.

        The strategy object is rebuilt from the rolling window on every bar,
        so it has no memory of setups that already produced a fill.  Those
        guards live here instead, keyed by the cross candle's *timestamp*
        rather than its index, since indices shift when the window is trimmed.
        """
        cfg = self.strategy_config
        plan = self.strategy.entry_plan(index + 1)
        if plan is None:
            return None
        if last_of_session and cfg.setup_expires_at_session_end:
            return None  # the setup does not survive the session break
        if self._setup_key(plan) in self.traded_setups:
            return None  # already acted on: the backtest consumes the setup
        if cfg.cooldown_bars > 0 and self.bars_seen <= self.cooldown_until:
            return None
        limit = cfg.max_trades_per_session
        if limit > 0 and self.session_trades.get(self.bars[index].session, 0) >= limit:
            return None
        return plan

    def _setup_key(self, plan) -> Optional[datetime]:
        if 0 <= plan.cross_index < len(self.bars):
            return self.bars[plan.cross_index].ts
        return None

    def _reset_position_state(self) -> None:
        if self.entry_index is not None and self.strategy_config.cooldown_bars > 0:
            self.cooldown_until = self.bars_seen + self.strategy_config.cooldown_bars
        self.entry_index = None
        self.entry_bars_seen = None
        self.entry_fill = None
        self.mfe = 0.0
        self.breakeven_done = False

    # ------------------------------------------------------------------
    def _trim_window(self) -> None:
        """Keep the window bounded but never cut into the current session."""
        if len(self.bars) <= self.window:
            return
        current_session = self.bars[-1].session
        cutoff = len(self.bars) - self.window
        # Walk back so the whole current session stays in the window, because
        # the VWAP is anchored to its first bar.
        while cutoff > 0 and self.bars[cutoff].session == current_session:
            cutoff -= 1
        self.bars = self.bars[cutoff:]

    def _rebuild_state(self, index: int) -> None:
        self.strategy.prepare(self.bars)
        for i in range(index + 1):
            self.strategy.on_bar_close(i)

    def _reconcile_bracket(self) -> None:
        """Re-derive the target from the actual fill price.

        The protective stop is structural (the cross candle's low), so it does
        not move with the fill; the target is a multiple of realised risk and
        therefore must.
        """
        fill = self.broker.position_avg_price()
        if fill <= 0 or self.planned_stop is None:
            return
        self.entry_fill = fill
        self.mfe = 0.0
        self.breakeven_done = False
        risk = self.direction * (fill - self.planned_stop)
        if risk <= 0:
            return
        if self.strategy_config.target_r > 0:
            target = fill + self.direction * self.strategy_config.target_r * risk
            self.broker.update_target(target)
            logger.info("bracket set from fill %.2f: stop %.2f target %.2f",
                        fill, self.planned_stop, target)

    def _ratchet_stop(self, index: int, bar: Bar, entry_bar: bool = False) -> None:
        """Move the stop to breakeven and/or trail it, never downwards."""
        cfg = self.strategy_config
        if self.entry_fill is not None and self.planned_stop is not None:
            # On the entry bar the pre-fill part of the range is not ours to
            # claim, so the close is used unless the pessimistic policy is on.
            # This must match backtest.py or breakeven triggers at different
            # times live than in the backtest.
            way = self.direction
            if entry_bar and cfg.entry_bar_stop != "low":
                self.mfe = max(self.mfe, way * (bar.close - self.entry_fill), 0.0)
            else:
                best = bar.high if way > 0 else bar.low
                self.mfe = max(self.mfe, way * (best - self.entry_fill))
            risk = max(1e-12, abs(self.entry_fill - self.planned_stop))
            if (
                cfg.breakeven_at_r > 0
                and not self.breakeven_done
                and self.mfe >= cfg.breakeven_at_r * risk
            ):
                self.broker.update_stop(self.entry_fill)
                self.breakeven_done = True
        trail = self.strategy.trail_stop_level(index, self.direction)
        if trail is not None:
            self.broker.update_stop(trail)

    def _build_entry_order(self, bar: Bar, plan) -> Optional[Order]:
        qty = position_size(
            self.broker.equity(), plan.trigger_price, plan.stop_price, self.config
        )
        if qty <= 0:
            return None
        target_r = self.strategy_config.target_r if self.strategy_config.target_r > 0 else None
        return Order(
            side="buy" if plan.direction > 0 else "sell",
            qty=qty,
            order_type=plan.order_kind,
            direction=plan.direction,
            fill_through_ticks=plan.fill_through_ticks,
            price=plan.trigger_price,
            stop_loss=plan.stop_price,
            target_r=target_r,
            reason=f"ema9xvwap breakout of bar {plan.cross_index}",
            ts=bar.ts,
        )

    def _flatten(self, bar: Bar, reason: str) -> Order:
        slip = self.config.slippage_ticks * self.strategy_config.tick_size
        order = Order(
            side="sell" if self.direction > 0 else "buy",
            qty=abs(self.broker.position_qty()),
            direction=self.direction,
            order_type="market",
            price=bar.close - self.direction * slip,  # matches the backtest
            reason=reason,
            ts=bar.ts,
        )
        self.broker.submit(order)
        self.entry_index = None
        return order


def run_paper_session(
    bars: Sequence[Bar],
    strategy_config: Optional[StrategyConfig] = None,
    backtest_config: Optional[BacktestConfig] = None,
    strategy: Optional[object] = None,
) -> PaperBroker:
    """Stream bars through ``LiveTrader`` + ``PaperBroker``.

    This exercises the real automation path (resting orders, broker-side
    brackets, cancellations) rather than the vectorised backtest path, which
    makes it the right place to catch wiring bugs before going live.
    """
    scfg = getattr(strategy, "config", None) or strategy_config or StrategyConfig()
    broker = PaperBroker(
        backtest_config, tick_size=scfg.tick_size, entry_bar_stop=scfg.entry_bar_stop
    )
    trader = LiveTrader(broker, scfg, backtest_config, strategy=strategy)
    for i, bar in enumerate(bars):
        last_of_session = i + 1 >= len(bars) or bars[i + 1].session != bar.session
        broker.on_bar(bar)  # fill resting orders / brackets from this bar
        trader.on_bar(bar, last_of_session=last_of_session)
    return broker
