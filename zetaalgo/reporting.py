"""Human-readable reports and CSV exports for backtest results."""

from __future__ import annotations

import csv
import math
from dataclasses import fields
from typing import List, Optional, Sequence

from .backtest import BacktestResult
from .metrics import Metrics, compute_metrics

SPARK = " .:-=+*#%@"


def _fmt(value: float, width: int = 12, places: int = 2) -> str:
    if value == math.inf:
        return "inf".rjust(width)
    if value == -math.inf:
        return "-inf".rjust(width)
    return f"{value:>{width},.{places}f}"


def sparkline(values: Sequence[float], width: int = 68) -> str:
    """A one-line ASCII rendering of the equity curve."""
    if len(values) < 2:
        return ""
    step = len(values) / width
    sampled = [values[min(len(values) - 1, int(i * step))] for i in range(width)]
    low, high = min(sampled), max(sampled)
    if high - low < 1e-12:
        return SPARK[0] * width
    out = []
    for value in sampled:
        idx = int((value - low) / (high - low) * (len(SPARK) - 1))
        out.append(SPARK[idx])
    return "".join(out)


def _table(rows: Sequence[Sequence[str]]) -> str:
    widths: List[int] = []
    for row in rows:
        for i, cell in enumerate(row):
            if i >= len(widths):
                widths.append(0)
            widths[i] = max(widths[i], len(cell))
    lines = []
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    return "\n".join(lines)


def format_report(
    result: BacktestResult,
    metrics: Optional[Metrics] = None,
    title: str = "EMA9 x VWAP CROSSOVER BACKTEST",
    show_trades: int = 10,
) -> str:
    """Render a full text report for a backtest run."""
    metrics = metrics or compute_metrics(result)
    scfg, bcfg = result.strategy_config, result.backtest_config
    width = 74
    lines: List[str] = ["=" * width, title.center(width), "=" * width, ""]

    period = "n/a"
    if result.start and result.end:
        period = f"{result.start:%Y-%m-%d %H:%M} -> {result.end:%Y-%m-%d %H:%M}"
    lines += [
        f"Period           {period}",
        f"Bars / sessions  {result.bars:,} bars over {result.sessions:,} sessions",
        "",
    ]

    if scfg is not None and hasattr(scfg, "min_tp_bps"):
        lines += [
            "--- SETUP RULES " + "-" * (width - 16),
            f"  Mean        rolling {scfg.mean_kind} ({scfg.mean_period}), "
            f"ATR({scfg.vol_period}) as the volatility unit",
            f"  Quote       resting LIMIT {scfg.entry_z:g} ATR from the mean "
            f"(maker), longs {scfg.trade_longs} shorts {scfg.trade_shorts}",
            f"  Exit        {scfg.tp_fraction:g} of the way back to the mean "
            f"(maker), stop {scfg.stop_z:g} ATR beyond, time stop "
            f"{scfg.time_stop_bars} bars",
            f"  FEE FLOOR   refuse any take-profit worth under "
            f"{scfg.min_tp_bps:g} bps",
            f"  Confirm     lagging EMA({scfg.trend_period}) veto only, "
            f"enabled {scfg.trend_filter}",
            "",
        ]
    elif scfg is not None and not hasattr(scfg, "ema_period"):
        # CHoCH/BOS/POC configuration.
        lines += [
            "--- SETUP RULES " + "-" * (width - 16),
            f"  Structure   swings {scfg.swing_left}/{scfg.swing_right} bars, "
            f"{'close' if scfg.use_close_break else 'wick'}-through breaks",
            f"  Sequence    CHoCH -> first BOS within {scfg.bos_window} bars "
            f"-> {scfg.entry_level} retest within {scfg.retest_window} bars",
            f"  Profile     fixed range over the {scfg.leg_anchor} leg, "
            f"{scfg.profile_bins} bins",
            f"  Entry       resting LIMIT at the {scfg.entry_level}"
            + (f", needing {scfg.fill_through_ticks} tick(s) through"
               if scfg.fill_through_ticks else ", on touch"),
            f"  Stop        {scfg.stop_mode} (min {scfg.min_risk_atr:g} x ATR), "
            f"target {scfg.target_r:g}R, breakeven at {scfg.breakeven_at_r:g}R",
            f"  Direction   longs {scfg.trade_longs}, shorts {scfg.trade_shorts}",
            "",
        ]
    elif scfg is not None:
        lines += [
            "--- SETUP RULES " + "-" * (width - 16),
            f"  EMA period {scfg.ema_period}, session-anchored VWAP, ATR({scfg.atr_period})",
            f"  Rule 2 gap  >= max({scfg.min_gap_atr:g} x ATR, {scfg.min_gap_pct:.4%} of price)",
            f"  Rule 3      close above both lines: {scfg.require_close_above_both}",
            f"  Windows     confirm within {scfg.confirm_window} bars, "
            f"entry within {scfg.entry_window} bars of confirmation",
            f"  Stop        {scfg.stop_mode} (ATR x {scfg.stop_atr_mult:g}), "
            f"entry-bar policy '{scfg.entry_bar_stop}'",
            f"  Target      {scfg.target_r:g}R, breakeven at {scfg.breakeven_at_r:g}R, "
            f"trail '{scfg.trail_mode}'",
            f"  Exits       close<EMA {scfg.exit_on_close_below_ema}, "
            f"close<VWAP {scfg.exit_on_close_below_vwap}, "
            f"flat at session end {scfg.flat_at_session_end}",
            "",
        ]
    if bcfg is not None:
        lines += [
            "--- ACCOUNT & COSTS " + "-" * (width - 20),
            f"  Equity      {bcfg.initial_equity:,.2f} start, sizing '{bcfg.sizing}' "
            f"(risk {bcfg.risk_pct:.2%}/trade, max {bcfg.max_notional_pct:.0%} notional)",
            f"  Costs       {bcfg.commission_bps:g} bps/side + "
            f"{bcfg.commission_per_share:g}/share, slippage "
            f"{bcfg.slippage_ticks:g} tick(s)",
            "",
        ]

    lines += ["--- PERFORMANCE " + "-" * (width - 16)]
    lines.append(
        _table(
            [
                ["  Net profit", _fmt(metrics.net_profit), "Total return",
                 f"{metrics.total_return_pct:>9.2f} %"],
                ["  Final equity", _fmt(metrics.final_equity), "CAGR",
                 f"{metrics.cagr_pct:>9.2f} %"],
                ["  Gross profit", _fmt(metrics.gross_profit), "Max drawdown",
                 f"{metrics.max_drawdown_pct:>9.2f} %"],
                ["  Gross loss", _fmt(-metrics.gross_loss), "Max DD (cash)",
                 _fmt(metrics.max_drawdown, 9)],
                ["  Commission", _fmt(-metrics.total_commission), "Sharpe",
                 f"{metrics.sharpe:>9.2f}"],
                ["  Profit factor", _fmt(metrics.profit_factor), "Sortino",
                 f"{metrics.sortino:>9.2f}"],
                ["  Expectancy/trade", _fmt(metrics.expectancy), "Calmar",
                 f"{metrics.calmar:>9.2f}"],
                ["  Expectancy (R)", _fmt(metrics.expectancy_r), "Exposure",
                 f"{metrics.exposure_pct:>9.2f} %"],
            ]
        )
    )
    lines += ["", "--- TRADE STATISTICS " + "-" * (width - 21)]
    lines.append(
        _table(
            [
                ["  Trades", f"{metrics.trades:>12,}", "Win rate",
                 f"{metrics.win_rate:>9.2f} %"],
                ["  Wins / losses", f"{metrics.wins:>5,} / {metrics.losses:<6,}",
                 "Payoff ratio", f"{metrics.payoff_ratio:>9.2f}"],
                ["  Average win", _fmt(metrics.avg_win), "Best trade",
                 _fmt(metrics.best_trade, 9)],
                ["  Average loss", _fmt(metrics.avg_loss), "Worst trade",
                 _fmt(metrics.worst_trade, 9)],
                ["  Avg R multiple", _fmt(metrics.avg_r), "Total R",
                 f"{metrics.total_r:>9.2f}"],
                ["  Avg MFE (R)", _fmt(metrics.avg_mfe_r), "Avg MAE (R)",
                 f"{metrics.avg_mae_r:>9.2f}"],
                ["  Avg bars held", _fmt(metrics.avg_bars_held), "Trades/session",
                 f"{metrics.trades_per_session:>9.2f}"],
                ["  Max win streak", f"{metrics.max_consecutive_wins:>12,}",
                 "Max loss streak", f"{metrics.max_consecutive_losses:>9,}"],
            ]
        )
    )

    if result.exit_reasons:
        lines += ["", "--- EXIT BREAKDOWN " + "-" * (width - 19)]
        total = sum(result.exit_reasons.values())
        rows = []
        for reason, count in sorted(result.exit_reasons.items(), key=lambda kv: -kv[1]):
            subset = [t for t in result.trades if t.exit_reason == reason]
            pnl = sum(t.net_pnl for t in subset)
            rows.append(
                [
                    f"  {reason}",
                    f"{count:>5,}",
                    f"{100.0 * count / total:>6.1f} %",
                    f"net {pnl:>12,.2f}",
                    f"avg R {sum(t.r_multiple for t in subset) / len(subset):>6.2f}",
                ]
            )
        lines.append(_table(rows))

    if result.signal_counters:
        lines += ["", "--- SIGNAL FUNNEL " + "-" * (width - 18)]
        counters = dict(result.signal_counters)
        crosses = counters.pop("crosses_detected", 0)
        rows = [["  bullish EMA/VWAP crosses", f"{crosses:>6,}", ""]]
        for reason, count in sorted(counters.items(), key=lambda kv: -kv[1]):
            share = f"{100.0 * count / crosses:>6.1f} %" if crosses else ""
            rows.append([f"  discarded: {reason}", f"{count:>6,}", share])
        share = f"{100.0 * metrics.trades / crosses:>6.1f} %" if crosses else ""
        rows.append(["  entries taken", f"{metrics.trades:>6,}", share])
        lines.append(_table(rows))

    curve = [point.equity for point in result.equity_curve]
    if len(curve) > 2:
        spark = sparkline(curve)
        lines += [
            "",
            "--- EQUITY CURVE " + "-" * (width - 17),
            f"  {spark}",
            f"  low {min(curve):,.0f}".ljust(38) + f"high {max(curve):,.0f}",
        ]

    if show_trades and result.trades:
        lines += ["", f"--- LAST {min(show_trades, len(result.trades))} TRADES " + "-" * 40]
        rows = [["  entry", "exit", "qty", "entry", "exit", "R", "net", "reason"]]
        for trade in result.trades[-show_trades:]:
            rows.append(
                [
                    f"  {trade.entry_ts:%m-%d %H:%M}",
                    f"{trade.exit_ts:%m-%d %H:%M}",
                    f"{trade.qty:g}",
                    f"{trade.entry_price:.2f}",
                    f"{trade.exit_price:.2f}",
                    f"{trade.r_multiple:+.2f}",
                    f"{trade.net_pnl:+,.2f}",
                    trade.exit_reason,
                ]
            )
        lines.append(_table(rows))

    lines += ["", "=" * width]
    return "\n".join(lines)


def write_trades_csv(path: str, result: BacktestResult) -> None:
    """Export every trade, one row each, for external analysis."""
    from .broker import Trade  # local import keeps the module import graph flat

    columns = [f.name for f in fields(Trade)]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for trade in result.trades:
            writer.writerow(
                [
                    getattr(trade, name).isoformat(sep=" ")
                    if name.endswith("_ts")
                    else getattr(trade, name)
                    for name in columns
                ]
            )


def write_equity_csv(path: str, result: BacktestResult) -> None:
    """Export the bar-by-bar equity curve with drawdown."""
    peak = -math.inf
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "session", "equity", "exposure", "drawdown_pct"])
        for point in result.equity_curve:
            peak = max(peak, point.equity)
            dd = 100.0 * (point.equity - peak) / peak if peak > 0 else 0.0
            writer.writerow(
                [
                    point.ts.isoformat(sep=" "),
                    point.session,
                    f"{point.equity:.2f}",
                    f"{point.exposure:.4f}",
                    f"{dd:.4f}",
                ]
            )


def write_metrics_csv(path: str, metrics: Metrics) -> None:
    """Export the summary metrics as a two-column key/value CSV."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.as_dict().items():
            writer.writerow([key, value])
