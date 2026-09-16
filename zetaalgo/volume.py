"""Volume-spike strategy: trade the bar that traded far more than usual.

The idea in one line
--------------------
Volume is attention.  A bar that trades several times the normal amount for
that market means something happened -- a headline, a liquidation cascade, a
level giving way -- and the two classic readings of it are opposites:

* **Breakout (continuation).**  The spike bar closed hard in one direction and
  took out recent range.  Join it on a break of the spike bar's extreme.
* **Fade (exhaustion).**  The spike bar made the extreme but could not hold it
  and closed back inside its own range, which is the shape of a climax.  Take
  the other side on a break of the spike bar's *opposite* extreme.

Both are implemented here (``mode="breakout"`` / ``mode="fade"``) because the
honest answer to "which one is right?" is a backtest, not an opinion.

Why a volume spike is the one intraday signal that can pay its fees
-------------------------------------------------------------------
Every other strategy in this repo ran into the same equation::

    fee_in_R = 2 * fee_rate / stop_percent

A scalp with a 0.1% stop pays roughly one full R in fees on a taker round
trip, which no win rate rescues.  A volume spike is different **by
construction**: the signal bar is an expansion bar, so a stop placed at the
far side of it is wide in percentage terms, and the same fee is a small
fraction of R.  ``min_risk_bps`` makes that structural advantage a gate rather
than a hope -- a setup whose stop is too tight to carry the round trip is
refused outright.

The three decisions a "sudden spike in volume" leaves open
-----------------------------------------------------------
*Spike relative to what?*  Raw volume is meaningless: 4,000 contracts is a
dead hour for BTC and a stampede for a small perp.  The baseline is a rolling
**median** of recent volume (``baseline="trailing"``), so the threshold is
scale-free -- and a median, not a mean, because a mean is dragged up by the
previous spikes and quietly raises the bar after every one.

*Volume has a time-of-day shape.*  On any market with a session, the open and
the close trade multiples of midday volume, so a trailing baseline fires
almost every open and almost never at lunch: the strategy would be a
"buy the open" system wearing a volume costume.  ``baseline="time_of_day"``
compares each bar against the same slot in previous sessions instead, which
removes the seasonality and leaves the genuinely unusual bars.

*A spike has no direction.*  Volume is unsigned, so the bar's own shape has to
supply the side: ``min_body_ratio`` demands a decisive close for a breakout,
``max_body_ratio`` demands a rejection wick for a fade, and
``require_range_break`` demands the bar actually did something to the recent
range rather than churning inside it.

As everywhere in this repo, every gate is evaluated on **closed** bars, and
the entry is a resting stop order at a level that was known before the bar
that fills it opened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .data import Bar
from .indicators import atr as atr_series
from .indicators import rolling_median
from .strategy import EntryPlan

LONG = 1
SHORT = -1


@dataclass
class VolumeSpikeConfig:
    """Rules and risk settings for the volume-spike strategy."""

    # --- direction ------------------------------------------------------
    trade_longs: bool = True
    trade_shorts: bool = True
    # breakout: go WITH the spike bar.  fade: take the other side of it.
    mode: str = "breakout"

    # --- what counts as a spike -----------------------------------------
    # "trailing"    - median volume of the previous ``baseline_period`` bars.
    # "time_of_day" - median volume of the same slot in the previous
    #                 ``baseline_sessions`` sessions, which is the only one of
    #                 the two that is not fooled by the opening bell.
    baseline: str = "trailing"
    baseline_period: int = 96  # 96 x 15m = one 24h day
    baseline_sessions: int = 20
    spike_mult: float = 3.0  # volume must be this many times the baseline

    # --- the spike bar's own shape --------------------------------------
    # |close - open| / (high - low).  A spike on a doji is indecision, not a
    # breakout, so a continuation trade needs a body; a fade needs the
    # opposite -- an extreme that was rejected.
    min_body_ratio: float = 0.5
    max_body_ratio: float = 0.35
    # Range expansion, as a multiple of ATR.  0 disables.
    min_range_atr: float = 0.0
    # The spike bar must extend the recent range (a new ``range_lookback``-bar
    # high for a long).  Without it, "spike" includes the churn bars inside a
    # range, which is where breakout entries go to die.
    require_range_break: bool = True
    range_lookback: int = 20

    # --- entry ----------------------------------------------------------
    # Bars after the spike in which the break is still accepted (1 = strictly
    # the next candle).  A stale level is not the same trade.
    entry_window: int = 3
    entry_buffer_ticks: int = 1
    fill_through_ticks: int = 0
    tick_size: float = 0.01

    # --- risk -----------------------------------------------------------
    # "spike_bar" - the far side of the spike bar (the level that says the
    #               interpretation was wrong).
    # "midpoint"  - half way across it, for markets where a full spike range
    #               is an unaffordable stop.
    # "atr"       - a fixed ATR distance from the trigger.
    stop_mode: str = "spike_bar"
    stop_atr_mult: float = 1.0
    stop_buffer_ticks: int = 1
    atr_period: int = 14
    # THE FEE FLOOR.  Entry-to-stop distance, in basis points of the entry
    # price, below which the setup is refused: a round trip costs
    # ``2 * fee_rate``, so a stop of the same order of magnitude means the fee
    # eats a whole R.  0 disables the gate.
    min_risk_bps: float = 0.0
    # The other tail: a spike bar can be enormous, and a stop at its far side
    # then risks more per trade than the idea is worth.  0 disables.
    max_risk_atr: float = 0.0
    target_r: float = 2.0
    breakeven_at_r: float = 1.0
    trail_mode: str = "none"  # none | atr
    trail_atr_mult: float = 1.5
    # A spike trade is an event trade: if the move has not happened within a
    # few bars, the event is over.  0 disables the time stop.
    max_bars_in_trade: int = 12
    entry_bar_stop: str = "close"  # see backtest.py

    # --- housekeeping ---------------------------------------------------
    # A perpetual has no session, so nothing is forced flat by default; set
    # this for instruments that actually close.
    flat_at_session_end: bool = False
    setup_expires_at_session_end: bool = False
    max_trades_per_session: int = 0
    cooldown_bars: int = 0

    @property
    def required_history(self) -> int:
        """Bars of history the state machine needs before it can decide anything.

        The live trader keeps a rolling window rather than the whole series, so
        it has to be told how much is enough: a window shorter than this
        silently computes a different baseline from the backtest, which is the
        quietest way for live trading to stop matching its own test.
        """
        tail = max(self.atr_period, self.range_lookback) + self.entry_window + 2
        if self.baseline == "time_of_day":
            return tail
        return self.baseline_period + tail

    @property
    def required_sessions(self) -> int:
        """Sessions of history needed, for the baseline that counts in sessions.

        The time-of-day baseline compares a bar against the same slot in
        previous sessions, so "enough history" is a number of sessions, not a
        number of bars -- a 15-minute window and a 1-minute one need wildly
        different bar counts to hold the same 20 sessions.
        """
        return self.baseline_sessions + 1 if self.baseline == "time_of_day" else 0

    def __post_init__(self) -> None:
        if not (self.trade_longs or self.trade_shorts):
            raise ValueError("at least one of trade_longs/trade_shorts must be enabled")
        if self.mode not in ("breakout", "fade"):
            raise ValueError(f"unknown mode: {self.mode!r}")
        if self.baseline not in ("trailing", "time_of_day"):
            raise ValueError(f"unknown baseline: {self.baseline!r}")
        if self.baseline_period < 2:
            raise ValueError("baseline_period must be at least 2")
        if self.baseline_sessions < 2:
            raise ValueError("baseline_sessions must be at least 2")
        if self.spike_mult <= 1.0:
            raise ValueError("spike_mult must be greater than 1 to mean anything")
        if not 0.0 <= self.min_body_ratio <= 1.0:
            raise ValueError("min_body_ratio must be within [0, 1]")
        if not 0.0 <= self.max_body_ratio <= 1.0:
            raise ValueError("max_body_ratio must be within [0, 1]")
        if self.range_lookback < 1:
            raise ValueError("range_lookback must be at least 1")
        if self.entry_window < 1:
            raise ValueError("entry_window must be at least 1")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if self.stop_mode not in ("spike_bar", "midpoint", "atr"):
            raise ValueError(f"unknown stop_mode: {self.stop_mode!r}")
        if self.trail_mode not in ("none", "atr"):
            raise ValueError(f"unknown trail_mode: {self.trail_mode!r}")
        if self.entry_bar_stop not in ("close", "low", "next_bar"):
            raise ValueError(f"unknown entry_bar_stop: {self.entry_bar_stop!r}")
        if self.min_risk_bps < 0 or self.max_risk_atr < 0:
            raise ValueError("risk bounds cannot be negative")


@dataclass
class SpikeSetup:
    """A spike bar that has been accepted and may still become a trade."""

    index: int
    high: float
    low: float
    session: str
    direction: int
    rvol: float


class VolumeSpikeStrategy:
    """Arms on an unusually heavy bar, enters on a break of that bar."""

    def __init__(self, config: Optional[VolumeSpikeConfig] = None) -> None:
        self.config = config or VolumeSpikeConfig()
        self.bars: List[Bar] = []
        self.baseline: List[Optional[float]] = []
        self.rvol: List[Optional[float]] = []
        self.atr: List[Optional[float]] = []
        self.setup: Optional[SpikeSetup] = None
        self.rejections: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------
    def prepare(self, bars: Sequence[Bar]) -> None:
        cfg = self.config
        self.bars = list(bars)
        volumes = [max(0.0, float(b.volume)) for b in self.bars]
        self.atr = atr_series(
            [b.high for b in self.bars],
            [b.low for b in self.bars],
            [b.close for b in self.bars],
            cfg.atr_period,
        )
        if cfg.baseline == "time_of_day":
            self.baseline = self._time_of_day_baseline(volumes)
        else:
            self.baseline = self._trailing_baseline(volumes)
        self.rvol = [
            (volumes[i] / self.baseline[i])
            if self.baseline[i] not in (None, 0.0)
            else None
            for i in range(len(self.bars))
        ]
        self.setup = None
        self.rejections = {}

    def _trailing_baseline(self, volumes: Sequence[float]) -> List[Optional[float]]:
        """Median volume of the ``baseline_period`` bars *before* each bar.

        Shifted by one on purpose: a bar may not be part of the baseline it is
        being judged against, or a large enough spike raises its own bar and
        partially hides itself.
        """
        medians = rolling_median(volumes, self.config.baseline_period)
        return [None] + medians[:-1] if medians else []

    def _time_of_day_baseline(self, volumes: Sequence[float]) -> List[Optional[float]]:
        """Median volume of the same slot in previous sessions.

        The slot is the bar's ordinal position within its session, which is
        the closest thing to "the same time of day" that bar data guarantees:
        it survives missing bars and half-days without a calendar.

        Only *previous* sessions contribute, so the value is causal even for
        the first bar of a session, where a trailing window would still be
        looking at yesterday's close.
        """
        cfg = self.config
        out: List[Optional[float]] = []
        history: Dict[int, List[float]] = {}
        slot = 0
        current: Optional[str] = None
        pending: List[tuple] = []  # (slot, volume) bars of the session in progress
        for i, bar in enumerate(self.bars):
            if bar.session != current:
                # Yesterday's bars only become part of the baseline once the
                # session they belong to is complete.
                for done_slot, volume in pending:
                    bucket = history.setdefault(done_slot, [])
                    bucket.append(volume)
                    if len(bucket) > cfg.baseline_sessions:
                        del bucket[0]
                pending = []
                current = bar.session
                slot = 0
            samples = history.get(slot, [])
            if len(samples) >= min(cfg.baseline_sessions, 3):
                ordered = sorted(samples)
                middle = len(ordered) // 2
                out.append(
                    ordered[middle]
                    if len(ordered) % 2
                    else (ordered[middle - 1] + ordered[middle]) / 2.0
                )
            else:
                out.append(None)
            pending.append((slot, volumes[i]))
            slot += 1
        return out

    # ------------------------------------------------------------------
    # Rule helpers
    # ------------------------------------------------------------------
    def _note(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def _round_to_tick(self, price: float, up: bool) -> float:
        tick = self.config.tick_size
        units = price / tick
        rounded = math.ceil(units - 1e-9) if up else math.floor(units + 1e-9)
        return round(rounded * tick, 10)

    @staticmethod
    def body_ratio(bar: Bar) -> float:
        """How much of the bar's range the body occupies, in ``[0, 1]``.

        A zero-range bar has no shape to read; calling it bodyless is the
        conservative answer, since it fails the breakout gate rather than
        passing it.
        """
        span = bar.high - bar.low
        if span <= 0:
            return 0.0
        return abs(bar.close - bar.open) / span

    def _breaks_range(self, index: int, direction: int) -> bool:
        """Did the spike bar extend the recent range in ``direction``?"""
        cfg = self.config
        start = max(0, index - cfg.range_lookback)
        if start >= index:
            return False  # not enough history to say the range was extended
        window = self.bars[start:index]
        bar = self.bars[index]
        if direction == LONG:
            return bar.high > max(b.high for b in window)
        return bar.low < min(b.low for b in window)

    def _spike_direction(self, index: int) -> Optional[int]:
        """The side the spike bar itself argues for, or ``None`` if it does not.

        Split out from :meth:`on_bar_close` because "which way does a volume
        spike point?" is the whole question, and the two modes answer it in
        opposite ways:

        * breakout -- with the body: a decisive close, in a bar that took out
          recent range.
        * fade -- against the excursion: a bar that reached a new extreme and
          closed back inside itself is a failed push, so the trade is the
          other way.
        """
        cfg = self.config
        bar = self.bars[index]
        ratio = self.body_ratio(bar)
        if cfg.mode == "breakout":
            if ratio < cfg.min_body_ratio:
                self._note("body_too_small")
                return None
            direction = LONG if bar.close > bar.open else SHORT
            if bar.close == bar.open:
                self._note("body_too_small")
                return None
        else:
            if ratio > cfg.max_body_ratio:
                self._note("body_too_large_to_fade")
                return None
            # The excursion, not the close, defines which extreme was rejected:
            # the fade is against whichever side the bar reached for.
            up_wick = bar.high - max(bar.open, bar.close)
            down_wick = min(bar.open, bar.close) - bar.low
            if up_wick == down_wick:
                self._note("no_rejected_extreme")
                return None
            direction = SHORT if up_wick > down_wick else LONG
        # ``require_range_break`` is asked about the side the BAR pushed to,
        # which for a fade is the opposite of the side being traded.
        if cfg.require_range_break and not self._breaks_range(
            index, direction if cfg.mode == "breakout" else -direction
        ):
            self._note("no_range_break")
            return None
        if (direction == LONG and not cfg.trade_longs) or (
            direction == SHORT and not cfg.trade_shorts
        ):
            self._note("direction_disabled")
            return None
        return direction

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    def on_bar_close(self, index: int) -> None:
        """Advance state using only information available at bar ``index``'s close."""
        cfg = self.config
        bar = self.bars[index]

        if self.setup is not None:
            if cfg.setup_expires_at_session_end and bar.session != self.setup.session:
                self.setup = None
                self._note("expired_session_end")
            elif index - self.setup.index >= cfg.entry_window:
                self.setup = None
                self._note("no_break_in_window")

        baseline = self.baseline[index]
        if baseline is None or baseline <= 0:
            self._note("baseline_not_ready")
            return
        if bar.volume < cfg.spike_mult * baseline:
            return
        self._note("spikes_detected")

        if cfg.min_range_atr > 0:
            current_atr = self.atr[index]
            if current_atr is None or (bar.high - bar.low) < cfg.min_range_atr * current_atr:
                self._note("range_not_expanded")
                return

        direction = self._spike_direction(index)
        if direction is None:
            return
        # A fresh spike replaces a stale one: the newest bar is the better
        # description of what the market is doing right now.
        self.setup = SpikeSetup(
            index=index,
            high=bar.high,
            low=bar.low,
            session=bar.session,
            direction=direction,
            rvol=bar.volume / baseline,
        )
        self._note("armed")

    def consume_setup(self) -> None:
        self.setup = None

    # ------------------------------------------------------------------
    # Entry plan
    # ------------------------------------------------------------------
    def entry_plan(self, index: int) -> Optional[EntryPlan]:
        """The resting stop order for bar ``index``, built from closed bars.

        Called before bar ``index`` exists as far as the strategy is
        concerned, so everything here comes from bar ``index - 1`` and earlier.
        """
        cfg = self.config
        setup = self.setup
        if setup is None or index <= 0 or index > len(self.bars):
            return None
        offset = index - setup.index
        if offset < 1 or offset > cfg.entry_window:
            return None
        # The same order rests for several bars, so counting it on each of
        # them would report a funnel in order-bars while every other line is
        # in setups.  Only the first look at a setup is counted.
        first_look = offset == 1
        if index < len(self.bars):
            if cfg.setup_expires_at_session_end and (
                self.bars[index].session != setup.session
            ):
                return None

        long = setup.direction == LONG
        buffer = cfg.entry_buffer_ticks * cfg.tick_size
        # Break of the spike bar, in the direction being traded.  For a fade
        # that is the extreme the bar FAILED at: the entry only triggers once
        # the rejection is confirmed by price leaving the bar the other way.
        raw_trigger = setup.high + buffer if long else setup.low - buffer
        trigger = self._round_to_tick(raw_trigger, up=long)
        stop = self._stop_for(setup, trigger)
        if stop is None:
            return None
        risk = setup.direction * (trigger - stop)
        if risk <= 0:
            if first_look:
                self._note("bad_stop")
            return None
        if cfg.min_risk_bps > 0 and (risk / trigger) * 10_000.0 < cfg.min_risk_bps:
            # Too tight to survive the round trip: see ``min_risk_bps``.
            if first_look:
                self._note("risk_below_fee_floor")
            return None
        if cfg.max_risk_atr > 0:
            current_atr = self.atr[setup.index]
            if current_atr is None or risk > cfg.max_risk_atr * current_atr:
                if first_look:
                    self._note("risk_too_wide")
                return None
        if first_look:
            self._note("orders_placed")
        return EntryPlan(
            trigger_price=trigger,
            stop_price=stop,
            cross_index=setup.index,
            confirm_index=setup.index,
            setup_session=setup.session,
            direction=setup.direction,
            order_kind="stop",
            fill_through_ticks=cfg.fill_through_ticks,
        )

    def _stop_for(self, setup: SpikeSetup, trigger: float) -> Optional[float]:
        cfg = self.config
        long = setup.direction == LONG
        buffer = cfg.stop_buffer_ticks * cfg.tick_size
        if cfg.stop_mode == "spike_bar":
            raw = setup.low - buffer if long else setup.high + buffer
        elif cfg.stop_mode == "midpoint":
            midpoint = (setup.high + setup.low) / 2.0
            raw = midpoint - buffer if long else midpoint + buffer
        else:
            current_atr = self.atr[setup.index]
            if current_atr is None:
                return None
            offset = cfg.stop_atr_mult * current_atr
            raw = trigger - offset if long else trigger + offset
        return self._round_to_tick(raw, up=not long)

    # ------------------------------------------------------------------
    # Exits
    # ------------------------------------------------------------------
    def close_exit_reason(
        self, index: int, entry_index: int, direction: int = LONG
    ) -> Optional[str]:
        cfg = self.config
        if cfg.max_bars_in_trade > 0 and (index - entry_index) >= cfg.max_bars_in_trade:
            return "time_stop"
        return None

    def trail_stop_level(self, index: int, direction: int = LONG) -> Optional[float]:
        cfg = self.config
        if cfg.trail_mode != "atr":
            return None
        current_atr = self.atr[index]
        if current_atr is None:
            return None
        offset = cfg.trail_atr_mult * current_atr
        close = self.bars[index].close
        raw = close - offset if direction == LONG else close + offset
        return self._round_to_tick(raw, up=direction != LONG)
