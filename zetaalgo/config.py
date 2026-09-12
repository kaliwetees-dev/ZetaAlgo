"""Tunable configuration for the strategy and the backtest engine."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional


@dataclass
class StrategyConfig:
    """Rules of the EMA9 x VWAP crossover setup.

    The defaults encode a practical reading of the setup.  The literal
    "entry on the very next candle" reading is available by setting
    ``confirm_window=0`` and ``entry_window=1``.
    """

    # --- Direction ------------------------------------------------------
    # The published setup is long-only, which is the default.  On instruments
    # that short as easily as they buy (perpetual futures, CFDs) the mirror
    # setup -- EMA crossing BELOW VWAP, entry on a break of the cross candle's
    # low -- is the other half of the same idea.
    trade_longs: bool = True
    trade_shorts: bool = False

    # --- Indicators -----------------------------------------------------
    ema_period: int = 9
    atr_period: int = 14

    # --- Rule 2: EMA stays above VWAP with a "clear gap" ----------------
    # A visual "clear gap" has to become a number.  Two scale-free measures
    # are supported and the gap must clear BOTH: a fraction of price and a
    # fraction of ATR.  ATR-relative is the more robust of the two because
    # what looks like a clear gap on a chart depends on volatility.
    min_gap_atr: float = 0.15
    min_gap_pct: float = 0.0

    # --- Timing windows -------------------------------------------------
    # Bars after the cross in which rules 2+3 may become true (0 = the cross
    # candle itself must already satisfy them).
    confirm_window: int = 6
    # Bars after confirmation in which a break of the cross candle's high is
    # still accepted as an entry (1 = strictly the next candle).
    entry_window: int = 3

    # --- Rule 3: price above both lines ---------------------------------
    require_close_above_both: bool = True
    require_bullish_confirm_candle: bool = False

    # --- Rule 4: entry on break of the cross candle high ----------------
    tick_size: float = 0.01
    entry_buffer_ticks: int = 1

    # --- Risk management ------------------------------------------------
    stop_mode: str = "cross_low"  # cross_low | atr | vwap
    # How the stop behaves on the entry bar itself.  A buy-stop is crossed on
    # the way UP, so anything above the trigger in that bar is plausibly
    # post-fill, while the bar's low often occurred BEFORE the fill.  Treating
    # that low as a stop-out fabricates losses on bars that closed higher.
    #   "close"    - stop honoured from the entry bar's close (default)
    #   "low"      - assume the low came after the fill (most pessimistic)
    #   "next_bar" - stop becomes active only on the following bar
    entry_bar_stop: str = "close"
    stop_atr_mult: float = 1.2
    stop_buffer_ticks: int = 1
    target_r: float = 2.0  # 0 disables the fixed target
    breakeven_at_r: float = 1.0  # 0 disables the move to breakeven
    trail_mode: str = "none"  # none | ema | atr
    trail_atr_mult: float = 1.5
    exit_on_close_below_ema: bool = True
    exit_on_close_below_vwap: bool = True
    max_bars_in_trade: int = 0  # 0 = no time stop

    # --- Session handling -----------------------------------------------
    flat_at_session_end: bool = True
    setup_expires_at_session_end: bool = True
    max_trades_per_session: int = 0  # 0 = unlimited
    cooldown_bars: int = 0

    def __post_init__(self) -> None:
        if not (self.trade_longs or self.trade_shorts):
            raise ValueError("at least one of trade_longs/trade_shorts must be enabled")
        if self.ema_period <= 0:
            raise ValueError("ema_period must be positive")
        if self.atr_period <= 0:
            raise ValueError("atr_period must be positive")
        if self.confirm_window < 0:
            raise ValueError("confirm_window cannot be negative")
        if self.entry_window < 1:
            raise ValueError("entry_window must be at least 1")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if self.stop_mode not in ("cross_low", "atr", "vwap"):
            raise ValueError(f"unknown stop_mode: {self.stop_mode!r}")
        if self.entry_bar_stop not in ("close", "low", "next_bar"):
            raise ValueError(f"unknown entry_bar_stop: {self.entry_bar_stop!r}")
        if self.trail_mode not in ("none", "ema", "atr"):
            raise ValueError(f"unknown trail_mode: {self.trail_mode!r}")
        if self.min_gap_atr < 0 or self.min_gap_pct < 0:
            raise ValueError("gap thresholds cannot be negative")


@dataclass
class BacktestConfig:
    """Account, sizing and cost assumptions for the simulation."""

    initial_equity: float = 100_000.0
    sizing: str = "risk"  # risk | fixed | notional
    risk_pct: float = 0.01  # fraction of equity risked between entry and stop
    fixed_qty: float = 100.0
    notional_pct: float = 0.25  # fraction of equity deployed per trade
    max_notional_pct: float = 1.0  # leverage cap: position value / equity
    commission_bps: float = 1.0  # taker, per side, on traded notional
    # Venues charge less for providing liquidity.  A pullback strategy resting
    # a limit order at a level is a MAKER on entry, and on a limit take-profit
    # too; only stop-outs and forced exits cross the spread.  Leaving this
    # None charges the taker rate on everything, which understates a
    # limit-entry strategy.  OKX VIP0 is 0.02% maker / 0.05% taker.
    maker_bps: Optional[float] = None
    commission_per_share: float = 0.0
    min_commission: float = 0.0
    slippage_ticks: float = 1.0  # applied adversely on entry and exit
    allow_fractional_qty: bool = False
    # Minimum tradeable increment.  Flooring to whole units is right for
    # shares but silently DELETES trades on high-priced contracts: risking 1%
    # of 100k through a 4xATR stop on BTC works out to 0.39 contracts, which
    # floors to zero and vanishes.  OKX quotes this as lotSz (0.01 for
    # BTC-USDT-SWAP, 1 for XAU-USDT-SWAP); see tools/fetch_okx.py --specs.
    lot_size: float = 1.0
    annualisation_days: int = 252

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if self.sizing not in ("risk", "fixed", "notional"):
            raise ValueError(f"unknown sizing mode: {self.sizing!r}")
        if self.risk_pct <= 0 and self.sizing == "risk":
            raise ValueError("risk_pct must be positive when sizing='risk'")
        if self.slippage_ticks < 0:
            raise ValueError("slippage_ticks cannot be negative")


def _coerce(target_type: Any, value: Any) -> Any:
    if target_type is bool and isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return target_type(value)


def apply_overrides(config: Any, overrides: Dict[str, Any]) -> Any:
    """Apply ``{field: value}`` overrides to a config, validating field names.

    Used by the CLI's parameter sweep so a grid can be expressed as plain
    strings without each combination re-listing every default.
    """
    types = {f.name: f.type for f in fields(config)}
    kwargs = {f.name: getattr(config, f.name) for f in fields(config)}
    for key, value in overrides.items():
        if key not in kwargs:
            raise ValueError(f"unknown config field: {key!r}")
        annotation = types[key]
        python_type = {"int": int, "float": float, "bool": bool, "str": str}.get(
            annotation if isinstance(annotation, str) else annotation.__name__, str
        )
        kwargs[key] = _coerce(python_type, value)
    return type(config)(**kwargs)
