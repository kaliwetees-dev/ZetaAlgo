"""Performance statistics for a completed backtest."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

from .backtest import BacktestResult
from .broker import Trade


@dataclass
class Metrics:
    """Summary statistics.  Monetary values are in account currency."""

    trades: int = 0
    wins: int = 0
    losses: int = 0
    scratches: int = 0
    win_rate: float = 0.0
    initial_equity: float = 0.0
    final_equity: float = 0.0
    net_profit: float = 0.0
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    total_commission: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    payoff_ratio: float = 0.0
    expectancy: float = 0.0
    expectancy_r: float = 0.0
    total_r: float = 0.0
    avg_r: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_bars: int = 0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    exposure_pct: float = 0.0
    avg_bars_held: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    trades_per_session: float = 0.0
    avg_mfe_r: float = 0.0
    avg_mae_r: float = 0.0
    sessions: int = 0
    bars: int = 0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: Sequence[float]) -> float:
    """Sample standard deviation (n-1), which is the right one for returns."""
    if len(values) < 2:
        return 0.0
    average = _mean(values)
    variance = sum((v - average) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def session_equity(result: BacktestResult) -> List[float]:
    """Closing equity for each session, used for return-based ratios."""
    out: List[float] = []
    current: Optional[str] = None
    for point in result.equity_curve:
        if point.session != current:
            current = point.session
            out.append(point.equity)
        else:
            out[-1] = point.equity
    return out


def drawdown_series(equity: Sequence[float]) -> List[float]:
    """Fractional drawdown from the running peak at each point."""
    out: List[float] = []
    peak = -math.inf
    for value in equity:
        peak = max(peak, value)
        out.append((value - peak) / peak if peak > 0 else 0.0)
    return out


def max_consecutive(trades: Sequence[Trade], wins: bool) -> int:
    best = run = 0
    for trade in trades:
        if trade.is_win == wins and trade.net_pnl != 0:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def compute_metrics(result: BacktestResult) -> Metrics:
    """Reduce a backtest result to summary statistics."""
    trades = result.trades
    metrics = Metrics(
        trades=len(trades),
        initial_equity=result.initial_equity,
        final_equity=result.final_equity,
        sessions=result.sessions,
        bars=result.bars,
    )
    metrics.net_profit = result.final_equity - result.initial_equity
    if result.initial_equity > 0:
        metrics.total_return_pct = 100.0 * metrics.net_profit / result.initial_equity

    # --- trade-level statistics ---------------------------------------
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl < 0]
    metrics.wins, metrics.losses = len(wins), len(losses)
    metrics.scratches = len(trades) - len(wins) - len(losses)
    decided = len(wins) + len(losses)
    metrics.win_rate = 100.0 * len(wins) / decided if decided else 0.0
    metrics.gross_profit = sum(t.net_pnl for t in wins)
    metrics.gross_loss = abs(sum(t.net_pnl for t in losses))
    metrics.total_commission = sum(t.commission for t in trades)
    metrics.profit_factor = (
        metrics.gross_profit / metrics.gross_loss
        if metrics.gross_loss > 0
        else (math.inf if metrics.gross_profit > 0 else 0.0)
    )
    metrics.avg_win = _mean([t.net_pnl for t in wins])
    metrics.avg_loss = _mean([t.net_pnl for t in losses])
    metrics.payoff_ratio = (
        metrics.avg_win / abs(metrics.avg_loss) if metrics.avg_loss else 0.0
    )
    if trades:
        metrics.expectancy = _mean([t.net_pnl for t in trades])
        metrics.total_r = sum(t.r_multiple for t in trades)
        metrics.avg_r = metrics.expectancy_r = _mean([t.r_multiple for t in trades])
        metrics.best_trade = max(t.net_pnl for t in trades)
        metrics.worst_trade = min(t.net_pnl for t in trades)
        metrics.avg_bars_held = _mean([float(t.bars_held) for t in trades])
        metrics.avg_mfe_r = _mean([t.mfe_r for t in trades])
        metrics.avg_mae_r = _mean([t.mae_r for t in trades])
        metrics.max_consecutive_wins = max_consecutive(trades, True)
        metrics.max_consecutive_losses = max_consecutive(trades, False)
    if result.sessions:
        metrics.trades_per_session = len(trades) / result.sessions

    # --- curve-level statistics ---------------------------------------
    curve = [point.equity for point in result.equity_curve]
    if curve:
        drawdowns = drawdown_series(curve)
        metrics.max_drawdown_pct = -100.0 * min(drawdowns)
        peak = -math.inf
        worst = 0.0
        longest = current = 0
        for value in curve:
            if value >= peak:
                peak = value
                current = 0
            else:
                current += 1
                longest = max(longest, current)
                worst = max(worst, peak - value)
        metrics.max_drawdown = worst
        metrics.max_drawdown_bars = longest
        metrics.exposure_pct = 100.0 * _mean(
            [1.0 if point.exposure > 0 else 0.0 for point in result.equity_curve]
        )

    # Daily (per-session) returns drive the risk-adjusted ratios.
    daily = session_equity(result)
    if len(daily) > 1:
        returns = [
            (daily[i] - daily[i - 1]) / daily[i - 1]
            for i in range(1, len(daily))
            if daily[i - 1] > 0
        ]
        periods = result.backtest_config.annualisation_days if result.backtest_config else 252
        deviation = _stdev(returns)
        average = _mean(returns)
        if deviation > 0:
            metrics.sharpe = (average / deviation) * math.sqrt(periods)
        downside = [r for r in returns if r < 0]
        downside_dev = math.sqrt(_mean([r * r for r in downside])) if downside else 0.0
        if downside_dev > 0:
            metrics.sortino = (average / downside_dev) * math.sqrt(periods)
        years = len(daily) / periods
        if years > 0 and result.initial_equity > 0 and result.final_equity > 0:
            metrics.cagr_pct = 100.0 * (
                (result.final_equity / result.initial_equity) ** (1.0 / years) - 1.0
            )
        if metrics.max_drawdown_pct > 0:
            metrics.calmar = metrics.cagr_pct / metrics.max_drawdown_pct
    return metrics
