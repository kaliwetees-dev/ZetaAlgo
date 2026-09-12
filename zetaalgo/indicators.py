"""Indicator primitives used by the EMA9 x VWAP crossover strategy.

Everything here is a pure function over plain Python lists so the library has
no third-party dependencies.  Values that cannot be computed yet (not enough
history) are returned as ``None`` rather than 0.0, so callers can never
accidentally trade on a warm-up value.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

Number = Optional[float]


def sma(values: Sequence[float], period: int) -> List[Number]:
    """Simple moving average.  ``None`` until ``period`` samples exist."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: List[Number] = []
    running = 0.0
    for i, value in enumerate(values):
        running += value
        if i >= period:
            running -= values[i - period]
        out.append(running / period if i >= period - 1 else None)
    return out


def ema(values: Sequence[float], period: int) -> List[Number]:
    """Exponential moving average seeded with the SMA of the first ``period``.

    Seeding with an SMA (rather than the first sample) keeps the series
    identical to what charting platforms such as TradingView display, which
    matters when a strategy is defined by eye off a chart.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    alpha = 2.0 / (period + 1.0)
    out: List[Number] = [None] * len(values)
    if len(values) < period:
        return out
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = (values[i] - prev) * alpha + prev
        out[i] = prev
    return out


def true_range(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> List[Number]:
    """Wilder's true range.  The first bar has no previous close, so ``None``."""
    out: List[Number] = []
    for i in range(len(closes)):
        if i == 0:
            out.append(None)
            continue
        prev_close = closes[i - 1]
        out.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - prev_close),
                abs(lows[i] - prev_close),
            )
        )
    return out


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> List[Number]:
    """Average true range using Wilder smoothing."""
    if period <= 0:
        raise ValueError("period must be positive")
    tr = true_range(highs, lows, closes)
    out: List[Number] = [None] * len(closes)
    # The first usable TR is at index 1, so the first ATR lands at ``period``.
    if len(closes) <= period:
        return out
    seed = sum(float(v) for v in tr[1 : period + 1]) / period
    out[period] = seed
    prev = seed
    for i in range(period + 1, len(closes)):
        prev = (prev * (period - 1) + float(tr[i])) / period
        out[i] = prev
    return out


def typical_price(high: float, low: float, close: float) -> float:
    """The HLC3 price that VWAP is conventionally volume-weighted against."""
    return (high + low + close) / 3.0


def session_vwap(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    session_ids: Sequence[str],
) -> List[float]:
    """Volume weighted average price, anchored to (and reset by) each session.

    Anchoring matters: VWAP is only meaningful intraday, where it resets at the
    start of every session.  A VWAP that runs continuously across days drifts
    into a slow moving average and silently changes the strategy.

    If a session has no traded volume yet, the unweighted typical price is used
    so the series stays defined instead of dividing by zero.
    """
    out: List[float] = []
    cum_pv = 0.0
    cum_vol = 0.0
    cum_tp = 0.0
    bars_in_session = 0
    current: Optional[str] = None
    for i in range(len(closes)):
        if session_ids[i] != current:
            current = session_ids[i]
            cum_pv = cum_vol = cum_tp = 0.0
            bars_in_session = 0
        tp = typical_price(highs[i], lows[i], closes[i])
        vol = max(0.0, float(volumes[i]))
        cum_pv += tp * vol
        cum_vol += vol
        cum_tp += tp
        bars_in_session += 1
        out.append(cum_pv / cum_vol if cum_vol > 0 else cum_tp / bars_in_session)
    return out


def crossed_above(
    fast_prev: Number, slow_prev: Number, fast_now: Number, slow_now: Number
) -> bool:
    """True when ``fast`` moves from at-or-below ``slow`` to strictly above it."""
    if None in (fast_prev, slow_prev, fast_now, slow_now):
        return False
    return fast_prev <= slow_prev and fast_now > slow_now
