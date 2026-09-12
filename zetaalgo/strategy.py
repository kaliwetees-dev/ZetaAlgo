"""The EMA9 x VWAP crossover setup, expressed as a bar-by-bar state machine.

The published setup is four rules:

  1. The 9 EMA crosses the VWAP from below.
  2. The 9 EMA stays above the VWAP, with a clear gap between the two lines.
  3. Price (the candle) is above both the 9 EMA and the VWAP.
  4. Enter when the candle after the cross candle breaks the high of the
     cross candle.

Turning that into code forces three decisions the picture leaves open, and
each one is made explicitly here rather than by accident:

*"Clear gap" needs a number.*  The gap is measured against ATR (and
optionally price), so the same threshold means the same thing on a $20 stock
and a $2,000 one.

*Rules 2 and 3 cannot be judged on the cross candle.*  At the moment of the
cross the gap is by definition ~zero.  So the cross only *arms* the setup;
rules 2 and 3 are then checked on each following closed bar within
``confirm_window``.  Setting ``confirm_window=0`` demands the cross candle
itself satisfy everything, which is the strictest reading.

*No lookahead.*  Every gate is evaluated on **closed** bars only.  The entry
trigger is a resting stop order at a price that was known before the bar
opened, so the bar's own high is allowed to fill it -- but that bar's close,
EMA and VWAP are never consulted to make the decision to enter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .config import StrategyConfig
from .data import Bar
from .indicators import atr as atr_series
from .indicators import crossed_above, crossed_below, ema, session_vwap

LONG = 1
SHORT = -1


@dataclass
class Setup:
    """A crossover that has occurred and may still become a trade."""

    cross_index: int
    cross_high: float
    cross_low: float
    session: str
    direction: int = LONG
    confirm_index: Optional[int] = None

    @property
    def is_confirmed(self) -> bool:
        return self.confirm_index is not None


@dataclass
class EntryPlan:
    """A resting buy-stop derived from a confirmed setup."""

    trigger_price: float
    stop_price: float
    cross_index: int
    confirm_index: int
    setup_session: str
    direction: int = LONG

    @property
    def is_long(self) -> bool:
        return self.direction == LONG


class EmaVwapCrossoverStrategy:
    """Signal generator for the setup.

    Usage is driven by the backtest engine, one bar at a time::

        strategy.prepare(bars)
        for i, bar in enumerate(bars):
            plan = strategy.entry_plan(i)   # state as of bar i-1's close
            ...                             # engine may fill the plan
            strategy.on_bar_close(i)        # advance the state machine
    """

    def __init__(self, config: Optional[StrategyConfig] = None) -> None:
        self.config = config or StrategyConfig()
        self.bars: List[Bar] = []
        self.ema: List[Optional[float]] = []
        self.vwap: List[float] = []
        self.atr: List[Optional[float]] = []
        self.setup: Optional[Setup] = None
        self.rejections: dict = {}

    # ------------------------------------------------------------------
    # Setup / indicator preparation
    # ------------------------------------------------------------------
    def prepare(self, bars: Sequence[Bar]) -> None:
        """Compute indicator series once for the whole data set.

        The EMA runs continuously across sessions (an EMA has no natural
        anchor) while the VWAP resets every session.
        """
        self.bars = list(bars)
        closes = [b.close for b in self.bars]
        highs = [b.high for b in self.bars]
        lows = [b.low for b in self.bars]
        volumes = [b.volume for b in self.bars]
        sessions = [b.session for b in self.bars]
        self.ema = ema(closes, self.config.ema_period)
        self.vwap = session_vwap(highs, lows, closes, volumes, sessions)
        self.atr = atr_series(highs, lows, closes, self.config.atr_period)
        self.setup = None
        self.rejections = {}

    # ------------------------------------------------------------------
    # Rule helpers
    # ------------------------------------------------------------------
    def _round_to_tick(self, price: float, up: bool) -> float:
        tick = self.config.tick_size
        units = price / tick
        rounded = math.ceil(units - 1e-9) if up else math.floor(units + 1e-9)
        return round(rounded * tick, 10)

    def required_gap(self, index: int) -> Optional[float]:
        """Minimum EMA-to-VWAP distance that counts as a "clear gap"."""
        cfg = self.config
        bar = self.bars[index]
        gap_from_pct = bar.close * cfg.min_gap_pct
        if cfg.min_gap_atr > 0:
            current_atr = self.atr[index]
            if current_atr is None:
                return None  # ATR not warmed up: cannot judge the gap yet
            return max(gap_from_pct, current_atr * cfg.min_gap_atr)
        return gap_from_pct

    def _gap_ok(self, index: int, direction: int = LONG) -> bool:
        """Rule 2: the lines must be separated by more than the threshold.

        The gap is signed by direction, so a short setup needs the EMA that
        far BELOW the VWAP.
        """
        ema_value, vwap_value = self.ema[index], self.vwap[index]
        if ema_value is None:
            return False
        needed = self.required_gap(index)
        if needed is None:
            return False
        return direction * (ema_value - vwap_value) >= needed

    def _price_beyond_both(self, index: int, direction: int = LONG) -> bool:
        """Rule 3: the candle closes beyond both lines, in the trade's favour."""
        if not self.config.require_close_above_both:
            return True
        bar = self.bars[index]
        ema_value = self.ema[index]
        if ema_value is None:
            return False
        return (direction * (bar.close - ema_value) > 0
                and direction * (bar.close - self.vwap[index]) > 0)

    def _rules_satisfied(self, index: int, direction: int = LONG) -> bool:
        """Rules 2 and 3 evaluated on the closed bar ``index``."""
        ema_value = self.ema[index]
        if ema_value is None:
            return False
        if direction * (ema_value - self.vwap[index]) <= 0:
            return False
        if not self._gap_ok(index, direction):
            return False
        if not self._price_beyond_both(index, direction):
            return False
        if self.config.require_bullish_confirm_candle:
            # "With the trend": a bullish candle for longs, bearish for shorts.
            if self.bars[index].is_bullish != (direction == LONG):
                return False
        return True

    def _note_rejection(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    def on_bar_close(self, index: int) -> None:
        """Advance state using only information available at bar ``index``'s close."""
        cfg = self.config
        ema_value = self.ema[index]

        if self.setup is not None:
            setup = self.setup
            if index > setup.cross_index:
                if cfg.setup_expires_at_session_end and (
                    self.bars[index].session != setup.session
                ):
                    self.setup = None
                    self._note_rejection("expired_session_end")
                elif (ema_value is None
                      or setup.direction * (ema_value - self.vwap[index]) <= 0):
                    # Rule 2: the EMA failed to stay on its side of the VWAP.
                    self.setup = None
                    self._note_rejection("ema_fell_back_below_vwap")
                elif not setup.is_confirmed and index - setup.cross_index > cfg.confirm_window:
                    self.setup = None
                    self._note_rejection("not_confirmed_in_window")
                elif setup.is_confirmed and index - setup.confirm_index >= cfg.entry_window:
                    # No entry trigger fired inside the window; the level is stale.
                    self.setup = None
                    self._note_rejection("no_breakout_in_window")

        # An unconfirmed setup may confirm on this bar (including the cross bar).
        if self.setup is not None and not self.setup.is_confirmed:
            if index - self.setup.cross_index <= cfg.confirm_window and self._rules_satisfied(
                index, self.setup.direction
            ):
                self.setup.confirm_index = index

        # Detect a fresh cross.  A cross requires the EMA to have been at or
        # below the VWAP on the previous bar, which would already have
        # invalidated any live setup, so the two can never collide.
        if self.setup is None and index > 0:
            previous_ema, previous_vwap = self.ema[index - 1], self.vwap[index - 1]
            direction = None
            if cfg.trade_longs and crossed_above(
                previous_ema, previous_vwap, ema_value, self.vwap[index]
            ):
                direction = LONG
            elif cfg.trade_shorts and crossed_below(
                previous_ema, previous_vwap, ema_value, self.vwap[index]
            ):
                direction = SHORT
            if direction is not None:
                bar = self.bars[index]
                self.setup = Setup(
                    cross_index=index,
                    cross_high=bar.high,
                    cross_low=bar.low,
                    session=bar.session,
                    direction=direction,
                )
                self._note_rejection(
                    "crosses_detected" if direction == LONG else "crosses_detected_short"
                )
                if cfg.confirm_window >= 0 and self._rules_satisfied(index, direction):
                    self.setup.confirm_index = index

    # ------------------------------------------------------------------
    # Entry plan
    # ------------------------------------------------------------------
    def entry_plan(self, index: int) -> Optional[EntryPlan]:
        """The resting buy-stop applicable to bar ``index``, or ``None``.

        Called *before* bar ``index`` is processed, so the returned plan is
        built purely from bars up to ``index - 1``.  ``index`` may be one past
        the end of the series: that is the live-trading case, where the order
        has to rest at the broker before the next bar exists.
        """
        setup = self.setup
        cfg = self.config
        if setup is None or not setup.is_confirmed or index <= 0 or index > len(self.bars):
            return None
        # The break must happen on a bar after the confirming bar, inside the
        # entry window (window=1 means strictly the next candle).
        offset = index - setup.confirm_index
        if offset < 1 or offset > cfg.entry_window:
            return None
        # The next bar's session is unknown when planning ahead of the data;
        # the caller (live trader) handles the session rollover instead.
        if index < len(self.bars):
            if cfg.setup_expires_at_session_end and self.bars[index].session != setup.session:
                return None

        buffer = cfg.entry_buffer_ticks * cfg.tick_size
        if setup.direction == LONG:
            trigger = self._round_to_tick(setup.cross_high + buffer, up=True)
        else:
            # Rule 4 mirrored: a break of the cross candle's LOW.
            trigger = self._round_to_tick(setup.cross_low - buffer, up=False)
        stop = self._initial_stop(index, setup, trigger)
        if stop is None or setup.direction * (trigger - stop) <= 0:
            return None  # a non-positive risk distance is not tradeable
        return EntryPlan(
            trigger_price=trigger,
            stop_price=stop,
            cross_index=setup.cross_index,
            confirm_index=setup.confirm_index,
            setup_session=setup.session,
            direction=setup.direction,
        )

    def _initial_stop(self, index: int, setup: Setup, trigger: float) -> Optional[float]:
        """Initial protective stop, computed from closed-bar data only."""
        cfg = self.config
        buffer = cfg.stop_buffer_ticks * cfg.tick_size
        prev = index - 1  # last closed bar
        long = setup.direction == LONG
        if cfg.stop_mode == "cross_low":
            # The far side of the cross candle: its low for a long, high for a short.
            raw = setup.cross_low - buffer if long else setup.cross_high + buffer
        elif cfg.stop_mode == "vwap":
            raw = self.vwap[prev] - buffer if long else self.vwap[prev] + buffer
        else:  # atr
            current_atr = self.atr[prev]
            if current_atr is None:
                return None
            offset = cfg.stop_atr_mult * current_atr
            raw = trigger - offset if long else trigger + offset
        return self._round_to_tick(raw, up=not long)

    # ------------------------------------------------------------------
    # Exit helpers used by the engine
    # ------------------------------------------------------------------
    def trail_stop_level(self, index: int, direction: int = LONG) -> Optional[float]:
        """Trailing stop level implied by the closed bar ``index``."""
        cfg = self.config
        if cfg.trail_mode == "none":
            return None
        long = direction == LONG
        buffer = cfg.stop_buffer_ticks * cfg.tick_size
        if cfg.trail_mode == "ema":
            ema_value = self.ema[index]
            if ema_value is None:
                return None
            raw = ema_value - buffer if long else ema_value + buffer
        else:
            current_atr = self.atr[index]
            if current_atr is None:
                return None
            offset = cfg.trail_atr_mult * current_atr
            raw = self.bars[index].close - offset if long else self.bars[index].close + offset
        return self._round_to_tick(raw, up=not long)

    def close_exit_reason(
        self, index: int, entry_index: int, direction: int = LONG
    ) -> Optional[str]:
        """Rule-based exit for a bar's close, or ``None`` to stay in.

        Shared by the backtest engine and the live trader so the two can never
        disagree about when this setup says to get out.  Broker-side bracket
        orders (stop and target) are handled separately; this covers only the
        decisions that need the closed bar's indicators.
        """
        cfg = self.config
        bar = self.bars[index]
        ema_value = self.ema[index]
        # "Below" is relative to the trade: a short exits on a close back ABOVE.
        if (cfg.exit_on_close_below_ema and ema_value is not None
                and direction * (bar.close - ema_value) < 0):
            return "close_below_ema"
        if cfg.exit_on_close_below_vwap and direction * (bar.close - self.vwap[index]) < 0:
            return "close_below_vwap"
        if cfg.max_bars_in_trade > 0 and (index - entry_index) >= cfg.max_bars_in_trade:
            return "time_stop"
        return None

    def consume_setup(self) -> None:
        """Clear the active setup once it has produced a fill."""
        self.setup = None
