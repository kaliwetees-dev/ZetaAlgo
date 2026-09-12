"""Event-driven backtest engine for the EMA9 x VWAP crossover setup.

Execution assumptions, all deliberately pessimistic where the bar data is
ambiguous:

* **Entry** is a resting buy-stop at the cross candle's high (plus a buffer).
  It fills on the first bar whose high reaches the level.  If the bar opens
  above the level the fill is the open, not the level -- gaps are not free
  money.  Slippage is added on top.
* **Stop-outs** fill at the stop price plus slippage, or at the open when the
  bar opens below the stop (a gap through the stop).
* **Targets** are limit orders and fill at the target price with no slippage.
* **Stop before target.**  When one bar's range contains both levels the bar
  data cannot say which came first, so the loss is always assumed.
* **The entry bar is live.**  A position entered on a bar can be stopped out
  on that same bar.
* Nothing is decided using a bar's own close, EMA or VWAP before that bar is
  complete.  See ``strategy.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from .broker import Position, Trade, commission, position_size
from .config import BacktestConfig, StrategyConfig
from .data import Bar
from .strategy import EmaVwapCrossoverStrategy


@dataclass
class EquityPoint:
    ts: datetime
    equity: float
    exposure: float  # position notional / equity
    session: str


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[EquityPoint] = field(default_factory=list)
    initial_equity: float = 0.0
    final_equity: float = 0.0
    bars: int = 0
    sessions: int = 0
    signal_counters: Dict[str, int] = field(default_factory=dict)
    exit_reasons: Dict[str, int] = field(default_factory=dict)
    strategy_config: Optional[StrategyConfig] = None
    backtest_config: Optional[BacktestConfig] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None


class Backtester:
    """Runs the strategy over a bar series and records the resulting trades."""

    def __init__(
        self,
        strategy_config: Optional[StrategyConfig] = None,
        backtest_config: Optional[BacktestConfig] = None,
        strategy: Optional[object] = None,
    ) -> None:
        """``strategy`` may be any object implementing the strategy interface.

        Passing one lets a different set of rules reuse this engine unchanged,
        which is the point: execution realism, costs and the live/backtest
        parity guarantee are properties of the engine, not of the rules.
        """
        if strategy is not None:
            self.strategy = strategy
            self.strategy_config = getattr(strategy, "config", strategy_config)
        else:
            self.strategy_config = strategy_config or StrategyConfig()
            self.strategy = EmaVwapCrossoverStrategy(self.strategy_config)
        self.config = backtest_config or BacktestConfig()

    # ------------------------------------------------------------------
    def run(self, bars: Sequence[Bar]) -> BacktestResult:
        cfg = self.config
        scfg = self.strategy_config
        strategy = self.strategy
        strategy.prepare(bars)

        result = BacktestResult(
            initial_equity=cfg.initial_equity,
            bars=len(bars),
            strategy_config=scfg,
            backtest_config=cfg,
        )
        if not bars:
            result.final_equity = cfg.initial_equity
            return result

        result.start, result.end = bars[0].ts, bars[-1].ts
        result.sessions = len({b.session for b in bars})

        self.cash = cfg.initial_equity
        self.equity = cfg.initial_equity
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self._trades_this_session: Dict[str, int] = {}
        self._cooldown_until = -1

        slip = cfg.slippage_ticks * scfg.tick_size

        for i, bar in enumerate(bars):
            last_of_session = i + 1 >= len(bars) or bars[i + 1].session != bar.session
            exited_this_bar = False

            # --- 1. manage an already-open position with this bar ---------
            if self.position is not None:
                exited_this_bar = self._manage_position(i, bar, slip, last_of_session)

            # --- 2. look for an entry -------------------------------------
            # Skipped on a bar where a position was just closed: the intrabar
            # sequence is unknowable, so re-entering on the same bar would be
            # an assumption in the strategy's favour.
            if self.position is None and not exited_this_bar and i > self._cooldown_until:
                plan = strategy.entry_plan(i)
                if plan is not None and self._session_budget_left(bar.session):
                    long = plan.direction > 0
                    trigger = plan.trigger_price
                    through = plan.fill_through_ticks * scfg.tick_size
                    if plan.order_kind == "limit":
                        # A pullback entry: price has to come BACK to the level.
                        # ``fill_through_ticks`` can demand it trade through
                        # rather than merely touch, which is the honest way to
                        # model queue position on a resting limit.
                        touched = (bar.low <= trigger - through if long
                                   else bar.high >= trigger + through)
                    else:
                        touched = (bar.high >= trigger if long else bar.low <= trigger)
                    if touched:
                        if plan.order_kind == "limit":
                            # A limit never fills worse than its price; a bar
                            # that opened beyond it fills at the open instead.
                            fill = min(bar.open, trigger) if long else max(bar.open, trigger)
                        elif long:
                            fill = min(max(bar.open, trigger) + slip, bar.high)
                        else:
                            # Mirror image: a sell-stop fills at the trigger, or
                            # at the open when the bar gapped below it.
                            fill = max(min(bar.open, trigger) - slip, bar.low)
                        self._open_position(
                            i, bar, fill, plan.stop_price, size_price=plan.trigger_price,
                            direction=plan.direction,
                            maker=plan.order_kind == "limit",
                            entry_kind=plan.order_kind,
                        )
                        strategy.consume_setup()
                        if self.position is not None:
                            # The entry bar is live: it may stop us out at once.
                            exited_this_bar = self._manage_position(
                                i, bar, slip, last_of_session, entry_bar=True
                            )

            # --- 3. advance indicators/state machine ----------------------
            strategy.on_bar_close(i)

            # --- 4. trailing levels for the *next* bar --------------------
            if self.position is not None:
                self._update_protective_levels(i)

            # --- 5. mark to market ----------------------------------------
            notional = (self.position.direction * self.position.qty * bar.close
                        if self.position else 0.0)
            self.equity = self.cash + notional
            result.equity_curve.append(
                EquityPoint(
                    ts=bar.ts,
                    equity=self.equity,
                    exposure=(abs(notional) / self.equity) if self.equity else 0.0,
                    session=bar.session,
                )
            )

        result.trades = self.trades
        result.final_equity = self.equity
        result.signal_counters = dict(strategy.rejections)
        reasons: Dict[str, int] = {}
        for trade in self.trades:
            reasons[trade.exit_reason] = reasons.get(trade.exit_reason, 0) + 1
        result.exit_reasons = reasons
        return result

    # ------------------------------------------------------------------
    def _session_budget_left(self, session: str) -> bool:
        limit = self.strategy_config.max_trades_per_session
        if limit <= 0:
            return True
        return self._trades_this_session.get(session, 0) < limit

    def _open_position(
        self,
        index: int,
        bar: Bar,
        fill: float,
        stop: float,
        size_price: Optional[float] = None,
        direction: int = 1,
        maker: bool = False,
        entry_kind: str = "stop",
    ) -> None:
        cfg = self.config
        scfg = self.strategy_config
        if direction * (fill - stop) <= 0:
            return  # slippage swallowed the risk distance; refuse the trade
        # Size off the price the order was RESTING at, not the price it filled
        # at.  A live system has to commit to a quantity when it submits the
        # stop order, before the fill is known -- so a breakout that gaps
        # through its trigger really does risk more than the nominal
        # risk_pct, and the backtest has to show that rather than quietly
        # resize with hindsight.
        qty = position_size(self.equity, size_price or fill, stop, cfg)
        if qty <= 0:
            return
        fee = commission(qty, fill, cfg, maker=maker)
        if direction > 0:
            cost = qty * fill + fee
            if cost > self.cash:
                # Respect available cash (no implicit margin beyond the cap).
                lot = cfg.lot_size if cfg.lot_size > 0 else 1.0
                qty = math.floor(round((self.cash - fee) / fill / lot, 9)) * lot
                if qty <= 0:
                    return
                fee = commission(qty, fill, cfg, maker=maker)
        # Shorts receive proceeds instead of paying cash; the leverage cap in
        # position_size is what bounds them.
        self.cash -= direction * qty * fill + fee
        risk = direction * (fill - stop)
        target = fill + direction * scfg.target_r * risk if scfg.target_r > 0 else None
        self.position = Position(
            qty=qty,
            entry_price=fill,
            entry_index=index,
            entry_ts=bar.ts,
            stop_price=stop,
            target_price=target,
            initial_stop=stop,
            session=bar.session,
            direction=direction,
            entry_kind=entry_kind,
            entry_commission=fee,
        )
        self._trades_this_session[bar.session] = (
            self._trades_this_session.get(bar.session, 0) + 1
        )

    # ------------------------------------------------------------------
    def _manage_position(
        self,
        index: int,
        bar: Bar,
        slip: float,
        last_of_session: bool,
        entry_bar: bool = False,
    ) -> bool:
        """Apply this bar to the open position.  Returns True if it closed."""
        position = self.position
        assert position is not None
        scfg = self.strategy_config

        # Excursions are tracked from the entry price, in price terms.  On the
        # entry bar the pre-fill part of the range is not ours to claim, so
        # unless the pessimistic policy is selected the close is used for both
        # sides rather than contaminated wicks.
        way = position.direction
        # A breakout entry is crossed on the way INTO the trade, so the range
        # beyond the trigger is plausibly post-fill.  A pullback (limit) entry
        # is the mirror: it fills as price moves AGAINST the trade, so that
        # bar's favourable extreme may well have printed BEFORE the fill.
        # Claiming it would invent profit, so on the entry bar of a limit fill
        # only outcomes proven by the close are taken: if the bar closes beyond
        # a level, continuity says price crossed it after the fill.
        limit_entry_bar = entry_bar and position.entry_kind == "limit"
        use_full_range = not entry_bar or scfg.entry_bar_stop == "low"
        if limit_entry_bar:
            use_full_range = False
        if use_full_range:
            best, worst = (bar.high, bar.low) if way > 0 else (bar.low, bar.high)
            position.mfe = max(position.mfe, way * (best - position.entry_price))
            position.mae = min(position.mae, way * (worst - position.entry_price))
        else:
            moved = way * (bar.close - position.entry_price)
            position.mfe = max(position.mfe, moved, 0.0)
            position.mae = min(position.mae, moved, 0.0)

        # -- intrabar: stop first, then target ---------------------------
        # On the entry bar the open precedes the fill, so a gap-through-open
        # exit does not apply there.
        if limit_entry_bar:
            if way * (bar.close - position.stop_price) <= 0:
                return self._close_position(index, bar, bar.close - way * slip, "stop")
            if position.target_price is not None and (
                way * (bar.close - position.target_price) >= 0
            ):
                return self._close_position(index, bar, position.target_price, "target")
            # Neither level is proven; carry the position to the next bar.
            reason = self.strategy.close_exit_reason(index, position.entry_index, way)
            if reason is not None:
                return self._close_position(index, bar, bar.close - way * slip, reason)
            if scfg.flat_at_session_end and last_of_session:
                return self._close_position(index, bar, bar.close - way * slip, "session_end")
            return False

        stop_active = not entry_bar or scfg.entry_bar_stop == "low"
        stop_touched = (bar.low <= position.stop_price if way > 0
                        else bar.high >= position.stop_price)
        gapped = (bar.open <= position.stop_price if way > 0
                  else bar.open >= position.stop_price)
        if not entry_bar and gapped:
            return self._close_position(index, bar, bar.open, "stop_gap")
        if stop_active and stop_touched:
            fill = (max(bar.low, position.stop_price - slip) if way > 0
                    else min(bar.high, position.stop_price + slip))
            return self._close_position(index, bar, fill, "stop")
        if (entry_bar and scfg.entry_bar_stop == "close"
                and way * (bar.close - position.stop_price) <= 0):
            # The bar finished below the stop: the exit is real regardless of
            # the path taken to get there.
            return self._close_position(index, bar, bar.close - way * slip, "stop")
        # The target sits beyond the trigger, and the trigger was crossed on the
        # way there, so that part of the range is reachable post-fill.
        if position.target_price is not None:
            reached = (bar.high >= position.target_price if way > 0
                       else bar.low <= position.target_price)
            if reached:
                return self._close_position(index, bar, position.target_price, "target")

        # -- close-based exits (shared with the live trader) --------------
        reason = self.strategy.close_exit_reason(index, position.entry_index, way)
        if reason is not None:
            return self._close_position(index, bar, bar.close - way * slip, reason)
        if scfg.flat_at_session_end and last_of_session:
            return self._close_position(index, bar, bar.close - way * slip, "session_end")
        return False

    def _update_protective_levels(self, index: int) -> None:
        """Ratchet the stop up after this bar's close (never down)."""
        position = self.position
        assert position is not None
        scfg = self.strategy_config
        way = position.direction
        tighten = max if way > 0 else min  # never loosen the stop
        if scfg.breakeven_at_r > 0 and not position.breakeven_done:
            if position.mfe >= scfg.breakeven_at_r * position.risk_per_share:
                position.stop_price = tighten(position.stop_price, position.entry_price)
                position.breakeven_done = True
        trail = self.strategy.trail_stop_level(index, way)
        if trail is not None:
            position.stop_price = tighten(position.stop_price, trail)

    # ------------------------------------------------------------------
    def _close_position(self, index: int, bar: Bar, fill: float, reason: str) -> bool:
        position = self.position
        assert position is not None
        cfg = self.config
        fill = max(0.0, fill)
        way = position.direction
        # A take-profit rests as a limit and therefore earns the maker rate;
        # stops and forced exits cross the spread.
        fee = commission(position.qty, fill, cfg, maker=reason == "target")
        self.cash += way * position.qty * fill - fee
        gross = way * (fill - position.entry_price) * position.qty
        total_fees = fee + position.entry_commission
        net = gross - total_fees
        risk_per_share = position.risk_per_share
        equity_after = self.cash
        self.trades.append(
            Trade(
                entry_ts=position.entry_ts,
                exit_ts=bar.ts,
                entry_index=position.entry_index,
                exit_index=index,
                entry_price=position.entry_price,
                exit_price=fill,
                qty=position.qty,
                initial_stop=position.initial_stop,
                target_price=position.target_price,
                direction=way,
                exit_reason=reason,
                gross_pnl=gross,
                commission=total_fees,
                net_pnl=net,
                r_multiple=way * (fill - position.entry_price) / risk_per_share,
                bars_held=index - position.entry_index,
                mfe_r=position.mfe / risk_per_share,
                mae_r=position.mae / risk_per_share,
                session=position.session,
                equity_after=equity_after,
            )
        )
        self.position = None
        self.equity = self.cash
        if self.strategy_config.cooldown_bars > 0:
            self._cooldown_until = index + self.strategy_config.cooldown_bars
        return True


def run_backtest(
    bars: Sequence[Bar],
    strategy_config: Optional[StrategyConfig] = None,
    backtest_config: Optional[BacktestConfig] = None,
    strategy: Optional[object] = None,
) -> BacktestResult:
    """Convenience wrapper: build a ``Backtester`` and run it once."""
    return Backtester(strategy_config, backtest_config, strategy).run(bars)
