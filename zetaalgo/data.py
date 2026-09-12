"""Bar containers, CSV loading and a deterministic synthetic data generator."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Iterable, List, Optional, Sequence

TIMESTAMP_ALIASES = ("timestamp", "time", "datetime", "date", "ts", "open_time")
COLUMN_ALIASES = {
    "open": ("open", "o", "open_price"),
    "high": ("high", "h", "high_price"),
    "low": ("low", "l", "low_price"),
    "close": ("close", "c", "close_price", "adj_close", "adj close"),
    "volume": ("volume", "v", "vol", "qty", "quantity"),
}
TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
)


@dataclass(frozen=True)
class Bar:
    """A single OHLCV candle.

    ``session`` groups bars that share one VWAP anchor.  It is derived once at
    load time so every downstream consumer agrees on where sessions begin.
    """

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    session: str

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open


def parse_timestamp(raw: str) -> datetime:
    """Parse the timestamp formats that market data exports realistically use."""
    text = str(raw).strip()
    if not text:
        raise ValueError("empty timestamp")
    # Epoch seconds or milliseconds.
    if text.isdigit() and len(text) >= 9:
        value = int(text)
        if value > 10_000_000_000:  # milliseconds
            value //= 1000
        return datetime.utcfromtimestamp(value)
    normalised = text.replace("Z", "").split("+")[0].strip()
    try:
        return datetime.fromisoformat(normalised)
    except ValueError:
        pass
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(normalised, fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognised timestamp: {raw!r}")


def session_id(ts: datetime, session_start: Optional[time] = None) -> str:
    """Session label for a bar.

    By default a session is a calendar day.  ``session_start`` supports markets
    whose session opens in the evening and runs past midnight (futures, FX):
    bars before the start time are attributed to the previous day's session.
    """
    if session_start is None:
        return ts.date().isoformat()
    anchor = ts if ts.time() >= session_start else ts - timedelta(days=1)
    return anchor.date().isoformat()


def _resolve_columns(header: Sequence[str]) -> dict:
    lowered = {name.strip().lower(): name for name in header}
    resolved = {}
    for alias in TIMESTAMP_ALIASES:
        if alias in lowered:
            resolved["ts"] = lowered[alias]
            break
    else:
        raise ValueError(f"no timestamp column found in header: {list(header)}")
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                resolved[field] = lowered[alias]
                break
        else:
            if field == "volume":
                resolved[field] = None  # tolerated; treated as zero volume
            else:
                raise ValueError(f"no {field!r} column found in header: {list(header)}")
    return resolved


def load_csv(path: str, session_start: Optional[time] = None) -> List[Bar]:
    """Load OHLCV bars from a CSV with flexible column naming.

    Bars are sorted by timestamp and duplicate timestamps are dropped, keeping
    the last occurrence, because vendor exports are frequently unsorted or
    contain a repeated boundary bar.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} appears to be empty")
        cols = _resolve_columns(reader.fieldnames)
        by_ts = {}
        for row in reader:
            raw_ts = row.get(cols["ts"])
            if raw_ts is None or str(raw_ts).strip() == "":
                continue
            ts = parse_timestamp(raw_ts)
            vol_col = cols["volume"]
            raw_vol = row.get(vol_col) if vol_col else None
            try:
                bar = Bar(
                    ts=ts,
                    open=float(row[cols["open"]]),
                    high=float(row[cols["high"]]),
                    low=float(row[cols["low"]]),
                    close=float(row[cols["close"]]),
                    volume=float(raw_vol) if raw_vol not in (None, "") else 0.0,
                    session=session_id(ts, session_start),
                )
            except (TypeError, ValueError):
                continue  # skip malformed/blank rows rather than abort the run
            by_ts[ts] = bar
    return [by_ts[key] for key in sorted(by_ts)]


def write_csv(path: str, bars: Iterable[Bar]) -> None:
    """Write bars back out in the canonical column order."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for bar in bars:
            writer.writerow(
                [
                    bar.ts.isoformat(sep=" "),
                    f"{bar.open:.4f}",
                    f"{bar.high:.4f}",
                    f"{bar.low:.4f}",
                    f"{bar.close:.4f}",
                    f"{bar.volume:.0f}",
                ]
            )


def generate_synthetic(
    days: int = 60,
    bars_per_session: int = 78,
    start_price: float = 100.0,
    minutes_per_bar: int = 5,
    annual_drift: float = 0.05,
    annual_vol: float = 0.35,
    trend_persistence: float = 0.94,
    trend_strength: float = 1.6,
    seed: int = 7,
    session_open: time = time(9, 30),
    start_date: Optional[datetime] = None,
) -> List[Bar]:
    """Generate deterministic pseudo-market intraday bars.

    Pure Gaussian noise produces a series with no trends, which would make any
    crossover strategy look like a coin flip for uninteresting reasons.  So an
    Ornstein-Uhlenbeck style momentum term is layered on top of the random walk
    to create alternating trending and chopping regimes -- the environment this
    setup is actually meant to be judged in.

    The result is *synthetic*: useful to exercise and sanity-check the engine,
    never evidence that the strategy is profitable on real markets.
    """
    if days <= 0 or bars_per_session <= 0:
        raise ValueError("days and bars_per_session must be positive")
    rng = random.Random(seed)
    bars_per_year = 252 * bars_per_session
    mu = annual_drift / bars_per_year
    sigma = annual_vol / math.sqrt(bars_per_year)

    day = (start_date or datetime(2024, 1, 2)).replace(
        hour=session_open.hour, minute=session_open.minute, second=0, microsecond=0
    )
    price = start_price
    momentum = 0.0
    bars: List[Bar] = []
    produced = 0
    while produced < days:
        if day.weekday() >= 5:  # keep the calendar to weekdays
            day += timedelta(days=1)
            continue
        # Overnight gap, so each session's VWAP starts from a fresh location.
        price *= math.exp(rng.gauss(0.0, sigma * 3.0))
        for i in range(bars_per_session):
            momentum = trend_persistence * momentum + rng.gauss(0.0, 1.0)
            shock = rng.gauss(0.0, 1.0)
            ret = mu + sigma * (shock + trend_strength * momentum * (1 - trend_persistence))
            open_price = price
            close_price = open_price * math.exp(ret)
            body_high = max(open_price, close_price)
            body_low = min(open_price, close_price)
            wick = abs(ret) * 0.6 + sigma * 0.5
            high = body_high * math.exp(abs(rng.gauss(0.0, wick)))
            low = body_low * math.exp(-abs(rng.gauss(0.0, wick)))
            # U-shaped intraday volume profile with a lognormal surprise factor.
            phase = i / max(1, bars_per_session - 1)
            shape = 0.6 + 1.8 * ((2 * phase - 1) ** 2)
            volume = 20_000 * shape * math.exp(rng.gauss(0.0, 0.35))
            ts = day + timedelta(minutes=minutes_per_bar * i)
            bars.append(
                Bar(
                    ts=ts,
                    open=round(open_price, 4),
                    high=round(high, 4),
                    low=round(low, 4),
                    close=round(close_price, 4),
                    volume=round(volume),
                    session=session_id(ts),
                )
            )
            price = close_price
        produced += 1
        day += timedelta(days=1)
    return bars
