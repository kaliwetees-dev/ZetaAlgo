"""Volatility-band mean-reversion scalp, maker on both sides.

Why this shape and not another
------------------------------
Everything else in this repo died to one equation.  Fees are charged on
notional; risk is the stop distance; so the fee bill measured in R is::

    fee_in_R = 2 * fee_rate / stop_percent

and break-even needs a win rate of ``(1 + fee_in_R) / (1 + TP_in_R)``.  At 5
minutes the stop distance is a fraction of a percent, so a taker-fee round
trip (10 bps on OKX) makes ``fee_in_R`` about 1.0 and demands a ~99% win rate
at 1:1.  No signal survives that.

Two consequences drive this design:

1. **Never cross the spread to get in or to take profit.**  A resting limit
   pays the maker rate (2 bps on OKX VIP0 rather than 5), so the round trip is
   4 bps instead of 10.  Only the stop is a taker, and stops are the minority.
2. **The target must dwarf the fee, not resemble it.**  ``min_tp_bps`` refuses
   any setup whose take-profit does not clear the round-trip cost by a stated
   multiple.  This is the "cover the fees" requirement written as a gate the
   strategy cannot trade around.

The mechanism is liquidity provision: rest a bid below the market, get filled
when a burst pushes price to an extreme, and exit as it reverts.  Small,
frequent, high win rate -- a scalp.  The lagging trend filter is
*confirmation only*: it never generates a trade, it only vetoes fading a
market that is trending hard enough to keep going.

Honest limitation: a real maker strategy lives and dies on queue position and
adverse selection, and OHLCV bars cannot see either.  Being filled because
price touched your level is an assumption, not a fact; ``fill_through_ticks``
exists to stress it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .data import Bar
from .indicators import atr as atr_series
from .indicators import ema, sma
from .strategy import EntryPlan

LONG = 1
SHORT = -1


@dataclass
class ScalpConfig:
    """Rules and risk settings for the mean-reversion scalp."""

    # --- the mean price is reverting to -----------------------------------
    mean_period: int = 20
    mean_kind: str = "vwap"  # vwap | ema | sma  (rolling, not session-anchored)

    # --- the volatility unit the bands are measured in --------------------
    vol_period: int = 20
    entry_z: float = 2.0  # rest the limit this many vol-units from the mean

    # --- taking profit ----------------------------------------------------
    tp_fraction: float = 0.5  # exit this far back toward the mean
    # The fee gate.  A setup whose take-profit is worth less than this many
    # basis points is refused outright: a scalp that cannot pay for its own
    # round trip is not a small win, it is a slow loss.
    min_tp_bps: float = 12.0

    # --- risk -------------------------------------------------------------
    stop_z: float = 2.0  # stop this many vol-units beyond the entry band
    time_stop_bars: int = 12  # abandon a trade that has not reverted
    entry_bar_stop: str = "close"

    # --- lagging confirmation only ----------------------------------------
    # A slow EMA cannot start a trade here; it can only refuse one.  Fading a
    # market that is trending hard is how mean reversion dies, so the filter
    # vetoes longs while the slow mean is falling faster than the threshold
    # (and shorts while it is rising).
    trend_period: int = 200
    trend_filter: bool = True
    max_adverse_slope_bps: float = 4.0
    slope_lookback: int = 10

    # --- housekeeping -----------------------------------------------------
    trade_longs: bool = True
    trade_shorts: bool = True
    tick_size: float = 0.01
    fill_through_ticks: int = 0
    order_expiry_bars: int = 3  # how long a quote rests before being re-priced
    breakeven_at_r: float = 0.0
    trail_mode: str = "none"
    flat_at_session_end: bool = False  # a perpetual has no session
    setup_expires_at_session_end: bool = False
    max_trades_per_session: int = 0
    cooldown_bars: int = 0

    def __post_init__(self) -> None:
        if not (self.trade_longs or self.trade_shorts):
            raise ValueError("at least one of trade_longs/trade_shorts must be enabled")
        if self.mean_kind not in ("vwap", "ema", "sma"):
            raise ValueError(f"unknown mean_kind: {self.mean_kind!r}")
        if self.mean_period < 2 or self.vol_period < 2:
            raise ValueError("mean_period and vol_period must be at least 2")
        if self.entry_z <= 0 or self.stop_z <= 0:
            raise ValueError("entry_z and stop_z must be positive")
        if not 0 < self.tp_fraction <= 1:
            raise ValueError("tp_fraction must be in (0, 1]")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if self.entry_bar_stop not in ("close", "low", "next_bar"):
            raise ValueError(f"unknown entry_bar_stop: {self.entry_bar_stop!r}")

    @property
    def target_r(self) -> float:
        """Target as a multiple of risk.

        Both the target and the stop are measured in the same volatility unit,
        so their ratio is a constant the engine can apply directly::

            target = tp_fraction * entry_z * vol
            risk   = stop_z * vol
        """
        return (self.tp_fraction * self.entry_z) / self.stop_z

    @property
    def max_bars_in_trade(self) -> int:
        return self.time_stop_bars


class MeanReversionScalp:
    """Rests a maker quote at a volatility band and exits toward the mean."""

    def __init__(self, config: Optional[ScalpConfig] = None) -> None:
        self.config = config or ScalpConfig()
        self.bars: List[Bar] = []
        self.mean: List[Optional[float]] = []
        self.vol: List[Optional[float]] = []
        self.trend: List[Optional[float]] = []
        self.rejections: Dict[str, int] = {}
        self._quote_age = 0
        self._last_side: Optional[int] = None

    # ------------------------------------------------------------------
    def prepare(self, bars: Sequence[Bar]) -> None:
        cfg = self.config
        self.bars = list(bars)
        closes = [b.close for b in self.bars]
        highs = [b.high for b in self.bars]
        lows = [b.low for b in self.bars]

        if cfg.mean_kind == "vwap":
            self.mean = self._rolling_vwap(cfg.mean_period)
        elif cfg.mean_kind == "ema":
            self.mean = ema(closes, cfg.mean_period)
        else:
            self.mean = sma(closes, cfg.mean_period)

        self.vol = atr_series(highs, lows, closes, cfg.vol_period)
        self.trend = ema(closes, cfg.trend_period) if cfg.trend_filter else [None] * len(closes)
        self.rejections = {}
        self._quote_age = 0
        self._last_side = None

    def _rolling_vwap(self, period: int) -> List[Optional[float]]:
        """Volume-weighted mean over a trailing window.

        A rolling window rather than a session anchor: a perpetual trades
        continuously, so there is no open to anchor to, and a scalp cares
        about where volume traded in the last hour, not since midnight.
        """
        out: List[Optional[float]] = []
        pv = 0.0
        vol = 0.0
        tp_sum = 0.0
        for i, bar in enumerate(self.bars):
            typical = (bar.high + bar.low + bar.close) / 3.0
            volume = max(0.0, float(bar.volume))
            pv += typical * volume
            vol += volume
            tp_sum += typical
            if i >= period:
                old = self.bars[i - period]
                old_tp = (old.high + old.low + old.close) / 3.0
                old_v = max(0.0, float(old.volume))
                pv -= old_tp * old_v
                vol -= old_v
                tp_sum -= old_tp
            if i < period - 1:
                out.append(None)
            else:
                out.append(pv / vol if vol > 0 else tp_sum / period)
        return out

    def _note(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def _round_to_tick(self, price: float, up: bool) -> float:
        tick = self.config.tick_size
        units = price / tick
        rounded = math.ceil(units - 1e-9) if up else math.floor(units + 1e-9)
        return round(rounded * tick, 10)

    # ------------------------------------------------------------------
    def on_bar_close(self, index: int) -> None:
        """No multi-bar state to advance: the quote is re-derived each bar."""
        self._quote_age += 1

    def consume_setup(self) -> None:
        self._quote_age = 0

    # ------------------------------------------------------------------
    def _trend_allows(self, index: int, direction: int) -> bool:
        """The lagging filter: confirmation only, never a trade trigger."""
        cfg = self.config
        if not cfg.trend_filter:
            return True
        current = self.trend[index]
        past_index = index - cfg.slope_lookback
        if current is None or past_index < 0:
            return False
        past = self.trend[past_index]
        if past is None or past <= 0:
            return False
        slope_bps = (current - past) / past * 10_000.0
        # Refuse to buy a market whose slow mean is falling faster than the
        # threshold, and vice versa.
        return direction * slope_bps >= -cfg.max_adverse_slope_bps

    def entry_plan(self, index: int) -> Optional[EntryPlan]:
        """The resting maker quote for bar ``index``, built from closed bars."""
        cfg = self.config
        if index <= 0 or index > len(self.bars):
            return None
        prev = index - 1  # everything below comes from this closed bar
        mean = self.mean[prev]
        vol = self.vol[prev]
        if mean is None or vol is None or vol <= 0:
            return None

        close = self.bars[prev].close
        # Quote the side price has stretched away from: buy below the mean,
        # sell above it.
        direction = LONG if close < mean else SHORT
        if direction == LONG and not cfg.trade_longs:
            return None
        if direction == SHORT and not cfg.trade_shorts:
            return None
        if not self._trend_allows(prev, direction):
            self._note("trend_veto")
            return None

        band = mean - direction * cfg.entry_z * vol
        entry = self._round_to_tick(band, up=(direction == SHORT))
        stop = self._round_to_tick(
            entry - direction * cfg.stop_z * vol, up=(direction == SHORT)
        )
        if direction * (entry - stop) <= 0:
            self._note("bad_stop")
            return None

        # The exit is a PRICE (a fraction of the way back to the mean), not a
        # multiple of realised risk: a fill better than the quote must widen
        # the reward, never narrow it, or the fee floor below can be evaded.
        tp_distance = cfg.tp_fraction * cfg.entry_z * vol
        target = self._round_to_tick(
            entry + direction * tp_distance, up=(direction == LONG)
        )
        if entry <= 0:
            return None
        # The fee gate: is the take-profit actually worth the round trip?
        if (abs(target - entry) / entry) * 10_000.0 < cfg.min_tp_bps:
            self._note("tp_below_fee_floor")
            return None

        self._note("quoted")
        return EntryPlan(
            trigger_price=entry,
            stop_price=stop,
            cross_index=prev,
            confirm_index=prev,
            setup_session=self.bars[prev].session,
            direction=direction,
            order_kind="limit",  # maker on the way in
            fill_through_ticks=cfg.fill_through_ticks,
            target_price=target,  # maker on the way out too
        )

    # ------------------------------------------------------------------
    def close_exit_reason(
        self, index: int, entry_index: int, direction: int = LONG
    ) -> Optional[str]:
        cfg = self.config
        if cfg.time_stop_bars > 0 and (index - entry_index) >= cfg.time_stop_bars:
            return "time_stop"
        return None

    def trail_stop_level(self, index: int, direction: int = LONG) -> Optional[float]:
        return None
