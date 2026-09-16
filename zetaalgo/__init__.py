"""ZetaAlgo: an automated EMA9 x VWAP crossover trading system with backtester.

Quick start::

    from zetaalgo import StrategyConfig, generate_synthetic, run_backtest
    from zetaalgo.metrics import compute_metrics

    bars = generate_synthetic(days=120)
    result = run_backtest(bars, StrategyConfig(target_r=2.0))
    print(compute_metrics(result).total_return_pct)
"""

from .backtest import BacktestResult, Backtester, run_backtest
from .broker import Position, Trade
from .config import BacktestConfig, StrategyConfig
from .data import Bar, generate_synthetic, load_csv, write_csv
from .indicators import atr, ema, rolling_median, session_vwap, sma
from .live import BrokerAdapter, LiveTrader, Order, PaperBroker, run_paper_session
from .metrics import Metrics, compute_metrics
from .reporting import format_report
from .scalp import MeanReversionScalp, ScalpConfig
from .smc import SmcConfig, SmcStrategy
from .structure import StructureTracker, VolumeProfile, fixed_range_profile, swing_points
from .strategy import EmaVwapCrossoverStrategy, EntryPlan, Setup
from .volume import VolumeSpikeConfig, VolumeSpikeStrategy

__version__ = "1.0.0"

__all__ = [
    "Bar",
    "BacktestConfig",
    "BacktestResult",
    "Backtester",
    "BrokerAdapter",
    "EmaVwapCrossoverStrategy",
    "EntryPlan",
    "LiveTrader",
    "MeanReversionScalp",
    "Metrics",
    "Order",
    "PaperBroker",
    "Position",
    "Setup",
    "ScalpConfig",
    "SmcConfig",
    "SmcStrategy",
    "StrategyConfig",
    "StructureTracker",
    "VolumeProfile",
    "VolumeSpikeConfig",
    "VolumeSpikeStrategy",
    "Trade",
    "atr",
    "compute_metrics",
    "ema",
    "format_report",
    "generate_synthetic",
    "load_csv",
    "run_backtest",
    "fixed_range_profile",
    "rolling_median",
    "run_paper_session",
    "session_vwap",
    "swing_points",
    "sma",
    "write_csv",
]
