"""Shared helpers for building deterministic bar sequences in tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional, Sequence

from zetaalgo.data import Bar


def make_bars(
    closes: Sequence[float],
    session: str = "2024-01-02",
    volume: float = 1000.0,
    pad: float = 0.05,
    start: Optional[datetime] = None,
    minutes: int = 5,
) -> List[Bar]:
    """Build bars from a close series.

    Each bar opens at the previous close and gets symmetric wicks, so the
    OHLC invariants hold and the range is predictable enough to assert on.
    """
    out: List[Bar] = []
    ts = start or datetime(2024, 1, 2, 9, 30)
    prev = closes[0]
    for i, close in enumerate(closes):
        open_price = prev
        high = max(open_price, close) + pad
        low = min(open_price, close) - pad
        out.append(
            Bar(
                ts=ts + timedelta(minutes=minutes * i),
                open=round(open_price, 4),
                high=round(high, 4),
                low=round(low, 4),
                close=round(close, 4),
                volume=volume,
                session=session,
            )
        )
        prev = close
    return out


def ramp(flat: int = 12, rise: int = 10, base: float = 100.0, step: float = 0.25) -> List[float]:
    """A flat stretch (so EMA and VWAP converge) followed by a clean rally.

    The flat part pins the EMA to the VWAP; the rally then pulls the faster
    EMA above it, which is the rule-1 crossover the strategy waits for.
    """
    return [base] * flat + [base + step * (i + 1) for i in range(rise)]
