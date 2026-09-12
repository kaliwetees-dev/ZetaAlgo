"""Market-structure primitives: swing pivots, CHoCH/BOS, and volume profile.

These are the building blocks of the "smart money concepts" vocabulary, kept
separate from any one strategy so they can be tested on their own.

The single most important property here is **when** a thing becomes known.  A
swing high at bar ``i`` is not visible at bar ``i``: it needs ``right`` more
bars to prove nothing exceeded it.  Every function below reports that
confirmation delay explicitly, because backtests of structure-based strategies
are usually wrong in exactly this way -- they mark a pivot on the bar it
occurred and then "enter" on information that nobody had at the time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .data import Bar

HIGH = "high"
LOW = "low"
BULLISH = 1
BEARISH = -1


@dataclass(frozen=True)
class Pivot:
    """A confirmed swing point."""

    index: int  # bar where the extreme occurred
    price: float
    kind: str  # HIGH | LOW
    confirmed_index: int  # bar where it became knowable

    @property
    def is_high(self) -> bool:
        return self.kind == HIGH


@dataclass(frozen=True)
class StructureEvent:
    """A break of a reference swing level."""

    index: int  # bar whose close broke the level
    kind: str  # "CHoCH" | "BOS"
    direction: int  # BULLISH | BEARISH
    level: float  # the swing level that was broken
    pivot_index: int  # bar of the swing that was broken


def find_pivot(
    values: Sequence[float], index: int, left: int, right: int, want_high: bool
) -> bool:
    """Is ``index`` a pivot, given ``left``/``right`` bars of context?

    Strict on the left and non-strict on the right, which is the usual
    convention: it makes the *first* bar of a flat top the pivot rather than
    silently registering several.
    """
    if index - left < 0 or index + right >= len(values):
        return False
    pivot = values[index]
    for j in range(index - left, index):
        if (values[j] >= pivot) if want_high else (values[j] <= pivot):
            return False
    for j in range(index + 1, index + right + 1):
        if (values[j] > pivot) if want_high else (values[j] < pivot):
            return False
    return True


def swing_points(
    bars: Sequence[Bar], left: int = 2, right: int = 2
) -> List[Pivot]:
    """All confirmed swing highs and lows, in order of confirmation."""
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    out: List[Pivot] = []
    for i in range(len(bars)):
        if find_pivot(highs, i, left, right, want_high=True):
            out.append(Pivot(i, highs[i], HIGH, i + right))
        if find_pivot(lows, i, left, right, want_high=False):
            out.append(Pivot(i, lows[i], LOW, i + right))
    out.sort(key=lambda p: (p.confirmed_index, p.index))
    return out


class StructureTracker:
    """Bar-by-bar market structure: which break is a CHoCH and which a BOS.

    The distinction is only about context, not about the break itself:

    * a break **against** the prevailing direction is a **CHoCH** -- the
      character of the market changed, and the bias flips;
    * a break **with** the prevailing direction is a **BOS** -- the existing
      structure simply continued.

    So the very same "close above the last swing high" is a CHoCH in a
    downtrend and a BOS in an uptrend.  Bias starts unknown, and the first
    break of either side sets it without being called a CHoCH, since there was
    no character to change yet.
    """

    def __init__(self, left: int = 2, right: int = 2, use_close: bool = True) -> None:
        self.left = left
        self.right = right
        self.use_close = use_close  # close-through vs wick-through breaks
        self.bias: int = 0
        self.reference_high: Optional[Pivot] = None
        self.reference_low: Optional[Pivot] = None
        self.events: List[StructureEvent] = []
        self._pivots_by_confirmation: dict = {}
        self._bars: List[Bar] = []

    def prepare(self, bars: Sequence[Bar]) -> None:
        self._bars = list(bars)
        self._pivots_by_confirmation = {}
        for pivot in swing_points(bars, self.left, self.right):
            self._pivots_by_confirmation.setdefault(pivot.confirmed_index, []).append(pivot)
        self.bias = 0
        self.reference_high = self.reference_low = None
        self.events = []

    def on_bar_close(self, index: int) -> List[StructureEvent]:
        """Advance one bar; returns any structure events it produced."""
        produced: List[StructureEvent] = []

        # 1. Register pivots that only became knowable on this bar.
        for pivot in self._pivots_by_confirmation.get(index, ()):
            if pivot.is_high:
                self.reference_high = pivot
            else:
                self.reference_low = pivot

        # 2. Test this bar's break against the standing references.
        bar = self._bars[index]
        up_price = bar.close if self.use_close else bar.high
        down_price = bar.close if self.use_close else bar.low

        if self.reference_high is not None and up_price > self.reference_high.price:
            produced.append(self._register(index, BULLISH, self.reference_high))
            self.reference_high = None  # consumed until a new pivot forms
        elif self.reference_low is not None and down_price < self.reference_low.price:
            produced.append(self._register(index, BEARISH, self.reference_low))
            self.reference_low = None
        return produced

    def _register(self, index: int, direction: int, pivot: Pivot) -> StructureEvent:
        # A break against the prevailing bias changes character; with it, it
        # is continuation.  With no bias yet, the break only establishes one.
        if self.bias != 0 and direction != self.bias:
            kind = "CHoCH"
        else:
            kind = "BOS"
        self.bias = direction
        event = StructureEvent(index, kind, direction, pivot.price, pivot.index)
        self.events.append(event)
        return event


# ----------------------------------------------------------------------
# Fixed-range volume profile
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class VolumeProfile:
    """A fixed-range volume profile over a slice of bars."""

    low: float
    high: float
    bins: int
    volumes: List[float]
    poc_price: float
    poc_bin: int
    value_area_low: float
    value_area_high: float
    total_volume: float

    def bin_price(self, index: int) -> float:
        width = (self.high - self.low) / self.bins if self.bins else 0.0
        return self.low + width * (index + 0.5)


def fixed_range_profile(
    bars: Sequence[Bar],
    start: int,
    end: int,
    bins: int = 24,
    value_area_pct: float = 0.70,
) -> Optional[VolumeProfile]:
    """Volume profile of ``bars[start:end + 1]``; ``None`` if degenerate.

    Each bar's volume is spread evenly across the price bins its range covers,
    which is the standard approximation when only OHLCV is available: the
    intrabar path is unknown, so no bin inside the range is favoured over
    another.
    """
    if start > end or start < 0 or end >= len(bars) or bins <= 0:
        return None
    window = bars[start : end + 1]
    low = min(b.low for b in window)
    high = max(b.high for b in window)
    if high <= low:
        return None

    width = (high - low) / bins
    volumes = [0.0] * bins

    def bin_of(price: float) -> int:
        return min(bins - 1, max(0, int((price - low) / width)))

    for bar in window:
        volume = max(0.0, float(bar.volume))
        if volume <= 0:
            continue
        first, last = bin_of(bar.low), bin_of(bar.high)
        share = volume / (last - first + 1)
        for b in range(first, last + 1):
            volumes[b] += share

    total = sum(volumes)
    if total <= 0:
        # No volume anywhere (some feeds report zero): fall back to the
        # midpoint so the caller still gets a usable level.
        poc_bin = bins // 2
    else:
        poc_bin = max(range(bins), key=lambda b: volumes[b])

    # Value area: expand from the POC, always taking the richer neighbour,
    # until the configured share of total volume is enclosed.
    lower = upper = poc_bin
    captured = volumes[poc_bin]
    target = total * value_area_pct
    while captured < target and (lower > 0 or upper < bins - 1):
        below = volumes[lower - 1] if lower > 0 else -1.0
        above = volumes[upper + 1] if upper < bins - 1 else -1.0
        if above >= below:
            upper += 1
            captured += volumes[upper]
        else:
            lower -= 1
            captured += volumes[lower]

    return VolumeProfile(
        low=low,
        high=high,
        bins=bins,
        volumes=volumes,
        poc_price=low + width * (poc_bin + 0.5),
        poc_bin=poc_bin,
        value_area_low=low + width * lower,
        value_area_high=low + width * (upper + 1),
        total_volume=total,
    )
