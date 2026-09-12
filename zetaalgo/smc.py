"""CHoCH -> BOS -> volume-profile POC retest.

The rules being implemented:

1. On 5-minute bars, track market structure from swing pivots.
2. A **CHoCH** (a break against the prevailing structure) sets the bias --
   bullish or bearish.
3. Wait for the **first BOS** in that same direction: confirmation that the new
   structure is real rather than a single failed poke.
4. Take the swing leg that produced it and run a **fixed-range volume profile**
   over it to find the **POC**, the price where most volume traded.
5. Wait for price to **retest the POC**, and take the trade in the CHoCH/BOS
   direction.

Three things the description leaves open, decided here and exposed as config:

*Which leg to profile.*  ``leg_anchor="swing"`` (default) profiles from the
extreme that launched the impulse -- the lowest low between the CHoCH and the
BOS for a long -- through to the BOS bar.  ``"choch"`` profiles from the CHoCH
bar instead.  The first is the "swing level" in the usual sense: the actual
impulse, not an arbitrary starting point.

*What "wait for it to be retested" means for an order.*  This is a **limit**
order resting at the POC, not a breakout stop.  That distinction matters for
fills: a limit is only filled if price actually trades to the level, and even
then only if you were far enough up the queue, which is why
``fill_through_ticks`` can demand price trade *through* the POC rather than
merely touch it.

*Where the stop goes.*  Beyond the origin of the leg by default: if price
returns through the low that launched a bullish impulse, the structure that
justified the trade is gone.

As everywhere in this repo, nothing is decided on a bar before that bar closes.
A pivot needs ``swing_right`` bars to confirm, and the engine never sees a plan
built from a bar it has not finished.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .data import Bar
from .indicators import atr as atr_series
from .strategy import EntryPlan
from .structure import (
    BEARISH,
    BULLISH,
    StructureEvent,
    StructureTracker,
    VolumeProfile,
    fixed_range_profile,
)


@dataclass
class SmcConfig:
    """Rules and risk settings for the CHoCH/BOS/POC strategy."""

    # --- direction ------------------------------------------------------
    trade_longs: bool = True
    trade_shorts: bool = True

    # --- structure ------------------------------------------------------
    swing_left: int = 2
    swing_right: int = 2
    use_close_break: bool = True  # a break needs a close through, not a wick
    bos_window: int = 30  # bars after the CHoCH to wait for the first BOS

    # --- volume profile -------------------------------------------------
    profile_bins: int = 24
    leg_anchor: str = "swing"  # swing | choch
    entry_level: str = "poc"  # poc | value_area

    # --- entry ----------------------------------------------------------
    retest_window: int = 40  # bars after the BOS to wait for the retest
    fill_through_ticks: int = 0  # require price through the level, not just to it
    tick_size: float = 0.01

    # --- risk -----------------------------------------------------------
    stop_mode: str = "leg"  # leg | atr | profile
    # Floor on the entry-to-stop distance, as a multiple of ATR.  Without it,
    # pairing an entry level with a stop derived from the same level (entering
    # at the value-area edge and stopping just beyond it) yields a risk of one
    # tick, an absurd position size, and a 0% win rate.
    min_risk_atr: float = 0.25
    stop_atr_mult: float = 1.5
    stop_buffer_ticks: int = 1
    atr_period: int = 14
    target_r: float = 2.0
    breakeven_at_r: float = 1.0
    trail_mode: str = "none"  # none | atr
    trail_atr_mult: float = 1.5
    max_bars_in_trade: int = 0
    entry_bar_stop: str = "close"  # see backtest.py

    # --- exits / housekeeping -------------------------------------------
    exit_on_opposite_structure: bool = True
    flat_at_session_end: bool = True
    setup_expires_at_session_end: bool = True
    max_trades_per_session: int = 0
    cooldown_bars: int = 0

    def __post_init__(self) -> None:
        if not (self.trade_longs or self.trade_shorts):
            raise ValueError("at least one of trade_longs/trade_shorts must be enabled")
        if self.swing_left < 1 or self.swing_right < 1:
            raise ValueError("swing_left/swing_right must be at least 1")
        if self.profile_bins < 2:
            raise ValueError("profile_bins must be at least 2")
        if self.leg_anchor not in ("swing", "choch"):
            raise ValueError(f"unknown leg_anchor: {self.leg_anchor!r}")
        if self.entry_level not in ("poc", "value_area"):
            raise ValueError(f"unknown entry_level: {self.entry_level!r}")
        if self.stop_mode not in ("leg", "atr", "profile"):
            raise ValueError(f"unknown stop_mode: {self.stop_mode!r}")
        if self.trail_mode not in ("none", "atr"):
            raise ValueError(f"unknown trail_mode: {self.trail_mode!r}")
        if self.entry_bar_stop not in ("close", "low", "next_bar"):
            raise ValueError(f"unknown entry_bar_stop: {self.entry_bar_stop!r}")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if self.retest_window < 1 or self.bos_window < 1:
            raise ValueError("bos_window and retest_window must be at least 1")


@dataclass
class PendingBias:
    """A CHoCH that is waiting for its confirming BOS."""

    direction: int
    choch_index: int
    session: str


@dataclass
class RetestSetup:
    """A confirmed CHoCH+BOS whose POC is waiting to be retested."""

    direction: int
    choch_index: int
    bos_index: int
    leg_start: int
    leg_low: float
    leg_high: float
    entry_price: float
    stop_price: float
    profile: VolumeProfile
    session: str


class SmcStrategy:
    """CHoCH -> first BOS -> POC retest, as a bar-by-bar state machine.

    Implements the same interface the backtest engine and live trader use for
    the EMA/VWAP strategy, so it inherits the identical execution model,
    costing and live/backtest parity checks.
    """

    def __init__(self, config: Optional[SmcConfig] = None) -> None:
        self.config = config or SmcConfig()
        self.bars: List[Bar] = []
        self.atr: List[Optional[float]] = []
        self.tracker = StructureTracker(
            self.config.swing_left, self.config.swing_right, self.config.use_close_break
        )
        self.events_by_index: Dict[int, List[StructureEvent]] = {}
        self.pending: Optional[PendingBias] = None
        self.setup: Optional[RetestSetup] = None
        self.rejections: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def prepare(self, bars: Sequence[Bar]) -> None:
        """Precompute structure events and ATR for the whole series.

        Precomputing is safe: an event at bar ``i`` depends only on pivots
        confirmed by bar ``i`` and on bar ``i``'s own close, never on the
        future.  The state machine below still walks the bars in order.
        """
        self.bars = list(bars)
        self.atr = atr_series(
            [b.high for b in self.bars],
            [b.low for b in self.bars],
            [b.close for b in self.bars],
            self.config.atr_period,
        )
        self.tracker.prepare(self.bars)
        self.events_by_index = {}
        for i in range(len(self.bars)):
            events = self.tracker.on_bar_close(i)
            if events:
                self.events_by_index[i] = events
        self.pending = None
        self.setup = None
        self.rejections = {}

    def _note(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def _round_to_tick(self, price: float, up: bool) -> float:
        import math

        tick = self.config.tick_size
        units = price / tick
        rounded = math.ceil(units - 1e-9) if up else math.floor(units + 1e-9)
        return round(rounded * tick, 10)

    # ------------------------------------------------------------------
    def on_bar_close(self, index: int) -> None:
        cfg = self.config
        bar = self.bars[index]

        # --- expire or invalidate what is already in flight --------------
        if self.setup is not None:
            setup = self.setup
            if cfg.setup_expires_at_session_end and bar.session != setup.session:
                self.setup = None
                self._note("retest_expired_session")
            elif index - setup.bos_index > cfg.retest_window:
                self.setup = None
                self._note("retest_never_came")
            elif setup.direction == BULLISH and bar.close < setup.leg_low:
                # Price went back through the origin of the impulse: the
                # structure that justified the trade no longer exists.
                self.setup = None
                self._note("leg_invalidated")
            elif setup.direction == BEARISH and bar.close > setup.leg_high:
                self.setup = None
                self._note("leg_invalidated")

        if self.pending is not None:
            pending = self.pending
            if cfg.setup_expires_at_session_end and bar.session != pending.session:
                self.pending = None
                self._note("choch_expired_session")
            elif index - pending.choch_index > cfg.bos_window:
                self.pending = None
                self._note("no_bos_after_choch")

        # --- react to this bar's structure events ------------------------
        for event in self.events_by_index.get(index, ()):
            self._handle_event(event, index)

    def _handle_event(self, event: StructureEvent, index: int) -> None:
        cfg = self.config
        allowed = (cfg.trade_longs if event.direction == BULLISH else cfg.trade_shorts)

        if event.kind == "CHoCH":
            self._note("choch_detected")
            # A fresh change of character supersedes anything in flight: the
            # market just argued against the previous read.
            self.setup = None
            self.pending = (
                PendingBias(event.direction, index, self.bars[index].session)
                if allowed
                else None
            )
            return

        # A BOS only matters while a CHoCH is waiting for confirmation.
        if self.pending is None:
            return
        if event.direction != self.pending.direction:
            self.pending = None
            self._note("bos_against_bias")
            return
        self._note("bos_confirmed")
        self._build_setup(self.pending, index)
        self.pending = None

    # ------------------------------------------------------------------
    def _build_setup(self, pending: PendingBias, bos_index: int) -> None:
        """Profile the swing leg and arm a POC retest."""
        cfg = self.config
        direction = pending.direction
        start = max(0, pending.choch_index)
        window = range(start, bos_index + 1)
        if bos_index <= start:
            self._note("leg_too_short")
            return

        if cfg.leg_anchor == "swing":
            # The impulse origin: the extreme the move launched from.
            if direction == BULLISH:
                anchor = min(window, key=lambda i: self.bars[i].low)
            else:
                anchor = max(window, key=lambda i: self.bars[i].high)
        else:
            anchor = start
        if bos_index - anchor < 1:
            self._note("leg_too_short")
            return

        profile = fixed_range_profile(
            self.bars, anchor, bos_index, cfg.profile_bins
        )
        if profile is None:
            self._note("degenerate_profile")
            return

        if cfg.entry_level == "poc":
            level = profile.poc_price
        else:
            # The far edge of the value area, i.e. the deeper pullback.
            level = (profile.value_area_low if direction == BULLISH
                     else profile.value_area_high)

        close = self.bars[bos_index].close
        # The retest has to still be ahead of us: for a long the level must sit
        # below price, or there is nothing to wait for.
        if direction == BULLISH and level >= close:
            self._note("level_already_passed")
            return
        if direction == BEARISH and level <= close:
            self._note("level_already_passed")
            return

        entry = self._round_to_tick(level, up=(direction == BEARISH))
        leg_low = min(self.bars[i].low for i in range(anchor, bos_index + 1))
        leg_high = max(self.bars[i].high for i in range(anchor, bos_index + 1))
        stop = self._stop_for(direction, bos_index, entry, leg_low, leg_high, profile)
        if stop is None or direction * (entry - stop) <= 0:
            self._note("bad_stop")
            return
        if cfg.min_risk_atr > 0:
            current_atr = self.atr[bos_index]
            if current_atr is None:
                self._note("atr_not_ready")
                return
            if abs(entry - stop) < cfg.min_risk_atr * current_atr:
                self._note("risk_too_tight")
                return

        self.setup = RetestSetup(
            direction=direction,
            choch_index=pending.choch_index,
            bos_index=bos_index,
            leg_start=anchor,
            leg_low=leg_low,
            leg_high=leg_high,
            entry_price=entry,
            stop_price=stop,
            profile=profile,
            session=self.bars[bos_index].session,
        )
        self._note("setup_armed")

    def _stop_for(
        self,
        direction: int,
        index: int,
        entry: float,
        leg_low: float,
        leg_high: float,
        profile: VolumeProfile,
    ) -> Optional[float]:
        cfg = self.config
        buffer = cfg.stop_buffer_ticks * cfg.tick_size
        if cfg.stop_mode == "leg":
            raw = leg_low - buffer if direction == BULLISH else leg_high + buffer
        elif cfg.stop_mode == "profile":
            raw = (profile.value_area_low - buffer if direction == BULLISH
                   else profile.value_area_high + buffer)
        else:  # atr
            current = self.atr[index]
            if current is None:
                return None
            offset = cfg.stop_atr_mult * current
            raw = entry - offset if direction == BULLISH else entry + offset
        return self._round_to_tick(raw, up=(direction == BEARISH))

    # ------------------------------------------------------------------
    def entry_plan(self, index: int) -> Optional[EntryPlan]:
        """The resting limit order applicable to bar ``index``, if any."""
        setup = self.setup
        cfg = self.config
        if setup is None or index <= setup.bos_index or index > len(self.bars):
            return None
        if index - setup.bos_index > cfg.retest_window:
            return None
        if cfg.setup_expires_at_session_end and index < len(self.bars):
            if self.bars[index].session != setup.session:
                return None
        return EntryPlan(
            trigger_price=setup.entry_price,
            stop_price=setup.stop_price,
            cross_index=setup.bos_index,
            confirm_index=setup.bos_index,
            setup_session=setup.session,
            direction=setup.direction,
            order_kind="limit",
            fill_through_ticks=cfg.fill_through_ticks,
        )

    def consume_setup(self) -> None:
        self.setup = None

    # ------------------------------------------------------------------
    def close_exit_reason(
        self, index: int, entry_index: int, direction: int = BULLISH
    ) -> Optional[str]:
        cfg = self.config
        if cfg.exit_on_opposite_structure:
            for event in self.events_by_index.get(index, ()):
                if event.direction != direction:
                    return "structure_flip"
        if cfg.max_bars_in_trade > 0 and (index - entry_index) >= cfg.max_bars_in_trade:
            return "time_stop"
        return None

    def trail_stop_level(self, index: int, direction: int = BULLISH) -> Optional[float]:
        cfg = self.config
        if cfg.trail_mode != "atr":
            return None
        current = self.atr[index]
        if current is None:
            return None
        offset = cfg.trail_atr_mult * current
        close = self.bars[index].close
        raw = close - offset if direction == BULLISH else close + offset
        return self._round_to_tick(raw, up=(direction == BEARISH))
