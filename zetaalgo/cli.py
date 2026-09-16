"""Command line interface.

    python -m zetaalgo run        --synthetic 120
    python -m zetaalgo run        --csv data/spy_5m.csv --target-r 2.5
    python -m zetaalgo paper      --synthetic 30 --verbose
    python -m zetaalgo sweep      --synthetic 180 --grid "target_r=1.5,2,3"
    python -m zetaalgo walkforward --synthetic 300 --folds 4 --grid "target_r=1.5,2,3"
    python -m zetaalgo generate   --days 120 --out data/synthetic.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from datetime import time
from typing import Dict, List, Optional, Sequence, Tuple

from .backtest import run_backtest
from .config import BacktestConfig, StrategyConfig, apply_overrides
from .data import Bar, generate_synthetic, load_csv, write_csv
from .live import run_paper_session
from .scalp import MeanReversionScalp, ScalpConfig
from .smc import SmcConfig, SmcStrategy
from .volume import VolumeSpikeConfig, VolumeSpikeStrategy
from .metrics import compute_metrics
from .reporting import (
    format_report,
    write_equity_csv,
    write_metrics_csv,
    write_trades_csv,
)

STRATEGY_TITLES = {
    "emavwap": "EMA9 x VWAP CROSSOVER",
    "smc": "CHoCH -> BOS -> POC RETEST",
    "scalp": "MEAN-REVERSION SCALP",
    "spike": "VOLUME SPIKE",
}

SWEEP_METRICS = {
    "profit_factor": lambda m: m.profit_factor,
    "net_profit": lambda m: m.net_profit,
    "expectancy_r": lambda m: m.expectancy_r,
    "sharpe": lambda m: m.sharpe,
    "total_r": lambda m: m.total_r,
    "calmar": lambda m: m.calmar,
}


# ----------------------------------------------------------------------
# argument plumbing
# ----------------------------------------------------------------------
def _add_data_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("data")
    source = group.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="OHLCV CSV file with intraday bars")
    source.add_argument(
        "--synthetic",
        type=int,
        metavar="DAYS",
        help="generate N synthetic sessions instead of loading a file",
    )
    group.add_argument("--seed", type=int, default=7, help="synthetic data seed")
    group.add_argument(
        "--bars-per-session", type=int, default=78, help="synthetic bars per session"
    )
    group.add_argument(
        "--session-start",
        help="HH:MM session boundary for markets that trade past midnight",
    )
    group.add_argument("--symbol", default="SYNTHETIC", help="label used in reports")


def _add_smc_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("CHoCH/BOS/POC rules (--strategy smc)")
    group.add_argument("--swing", type=int, default=3,
                       help="bars either side of a swing pivot")
    group.add_argument("--bos-window", type=int, default=30,
                       help="bars after the CHoCH to wait for the first BOS")
    group.add_argument("--retest-window", type=int, default=40,
                       help="bars after the BOS to wait for the POC retest")
    group.add_argument("--profile-bins", type=int, default=24)
    group.add_argument("--leg-anchor", choices=("swing", "choch"), default="swing")
    group.add_argument("--entry-level", choices=("poc", "value_area"), default="poc")
    group.add_argument("--smc-stop", choices=("leg", "atr", "profile"), default="profile")
    group.add_argument("--fill-through-ticks", type=int, default=0,
                       help="require price to trade THROUGH the level, not just touch it")
    group.add_argument("--wick-breaks", action="store_true",
                       help="a wick through a swing counts as a break (default: close)")
    group.add_argument("--no-shorts", action="store_true",
                       help="long only (this strategy trades both ways by default)")


def build_smc_config(args: argparse.Namespace) -> SmcConfig:
    return SmcConfig(
        trade_longs=not args.no_longs,
        trade_shorts=not args.no_shorts,
        swing_left=args.swing,
        swing_right=args.swing,
        use_close_break=not args.wick_breaks,
        bos_window=args.bos_window,
        retest_window=args.retest_window,
        profile_bins=args.profile_bins,
        leg_anchor=args.leg_anchor,
        entry_level=args.entry_level,
        stop_mode=args.smc_stop,
        fill_through_ticks=args.fill_through_ticks,
        tick_size=args.tick_size,
        target_r=args.target_r,
        breakeven_at_r=args.breakeven_r,
        max_bars_in_trade=args.max_bars,
        entry_bar_stop=args.entry_bar_stop,
        flat_at_session_end=not args.hold_overnight,
        max_trades_per_session=args.max_trades_per_session,
        cooldown_bars=args.cooldown_bars,
    )


def _add_scalp_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("mean-reversion scalp (--strategy scalp)")
    group.add_argument("--mean-period", type=int, default=20)
    group.add_argument("--mean-kind", choices=("vwap", "ema", "sma"), default="vwap")
    group.add_argument("--vol-period", type=int, default=20)
    group.add_argument("--entry-z", type=float, default=2.0,
                       help="quote this many ATRs away from the rolling mean")
    group.add_argument("--tp-fraction", type=float, default=1.0,
                       help="take profit this far back toward the mean (1.0 = the mean)")
    group.add_argument("--min-tp-bps", type=float, default=12.0,
                       help="THE FEE FLOOR: refuse any setup whose take-profit is "
                            "worth less than this. OKX maker round trip is ~4 bps, "
                            "so keep this several times larger.")
    group.add_argument("--stop-z", type=float, default=2.0,
                       help="stop this many ATRs beyond the quote")
    group.add_argument("--time-stop", type=int, default=12,
                       help="abandon a trade that has not reverted in N bars")
    group.add_argument("--trend-period", type=int, default=200)
    group.add_argument("--no-trend-filter", action="store_true",
                       help="drop the lagging confirmation filter")
    group.add_argument("--max-adverse-slope-bps", type=float, default=4.0)
    group.add_argument("--max-variance-ratio", type=float, default=0.0,
                       help="refuse entries when the trailing variance ratio "
                            "is at or above this (1.0 = random walk, below = "
                            "mean reverting, above = trending). 0 disables. "
                            "The one regime gate that held out of sample.")
    group.add_argument("--vr-window", type=int, default=1920,
                       help="trailing bars used to measure the variance ratio")


def build_scalp_config(args: argparse.Namespace) -> ScalpConfig:
    return ScalpConfig(
        trade_longs=not args.no_longs,
        trade_shorts=not args.no_shorts,
        mean_period=args.mean_period,
        mean_kind=args.mean_kind,
        vol_period=args.vol_period,
        entry_z=args.entry_z,
        tp_fraction=args.tp_fraction,
        min_tp_bps=args.min_tp_bps,
        stop_z=args.stop_z,
        time_stop_bars=args.time_stop,
        trend_period=args.trend_period,
        trend_filter=not args.no_trend_filter,
        max_adverse_slope_bps=args.max_adverse_slope_bps,
        max_variance_ratio=args.max_variance_ratio,
        vr_window=args.vr_window,
        tick_size=args.tick_size,
        fill_through_ticks=args.fill_through_ticks,
        entry_bar_stop=args.entry_bar_stop,
        max_trades_per_session=args.max_trades_per_session,
        cooldown_bars=args.cooldown_bars,
    )


def _add_spike_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("volume spike (--strategy spike)")
    group.add_argument("--spike-mode", choices=("breakout", "fade"), default="breakout",
                       help="breakout: trade WITH the spike bar.  fade: take the "
                            "other side of a rejected extreme.")
    group.add_argument("--spike-baseline", choices=("trailing", "time_of_day"),
                       default="trailing",
                       help="what the volume is unusual RELATIVE TO. trailing: "
                            "median of the last --baseline-period bars. "
                            "time_of_day: median of the same slot in previous "
                            "sessions, which is the one that is not fooled by "
                            "the opening bell.")
    group.add_argument("--baseline-period", type=int, default=96,
                       help="bars in the trailing volume baseline (96 x 15m = 1 day)")
    group.add_argument("--baseline-sessions", type=int, default=20,
                       help="sessions used by the time_of_day baseline")
    group.add_argument("--spike-mult", type=float, default=3.0,
                       help="volume must be this many times the baseline median")
    group.add_argument("--min-body-ratio", type=float, default=0.5,
                       help="breakout: |close-open|/range required of the spike bar")
    group.add_argument("--max-body-ratio", type=float, default=0.35,
                       help="fade: the spike bar must be mostly wick, not body")
    group.add_argument("--min-range-atr", type=float, default=0.0,
                       help="require the spike bar's range to be this many ATRs")
    group.add_argument("--range-lookback", type=int, default=20,
                       help="bars the spike bar must break out of")
    group.add_argument("--no-range-break", action="store_true",
                       help="accept a spike that did not extend the recent range")
    group.add_argument("--spike-stop", choices=("spike_bar", "midpoint", "atr"),
                       default="spike_bar",
                       help="where the stop goes: the far side of the spike bar, "
                            "its midpoint, or an ATR distance from the trigger")
    group.add_argument("--min-risk-bps", type=float, default=0.0,
                       help="THE FEE FLOOR: refuse a setup whose entry-to-stop "
                            "distance is thinner than this. A taker round trip "
                            "on OKX is 10 bps, so anything near that spends a "
                            "whole R on fees.")
    group.add_argument("--max-risk-atr", type=float, default=0.0,
                       help="refuse a spike bar so large that its far side is an "
                            "unaffordable stop (0 = off)")
    group.add_argument("--spike-time-stop", type=int, default=12,
                       help="abandon a spike trade that has not moved in N bars "
                            "(0 = off): the event is over")
    group.add_argument("--spike-trail", choices=("none", "atr"), default="none")
    group.add_argument("--flat-session-end", action="store_true",
                       help="flatten at the session close (off by default: a "
                            "perpetual has no session)")


def build_spike_config(args: argparse.Namespace) -> VolumeSpikeConfig:
    return VolumeSpikeConfig(
        trade_longs=not args.no_longs,
        trade_shorts=not args.no_shorts,
        mode=args.spike_mode,
        baseline=args.spike_baseline,
        baseline_period=args.baseline_period,
        baseline_sessions=args.baseline_sessions,
        spike_mult=args.spike_mult,
        min_body_ratio=args.min_body_ratio,
        max_body_ratio=args.max_body_ratio,
        min_range_atr=args.min_range_atr,
        require_range_break=not args.no_range_break,
        range_lookback=args.range_lookback,
        entry_window=args.entry_window,
        entry_buffer_ticks=args.entry_buffer_ticks,
        fill_through_ticks=args.fill_through_ticks,
        tick_size=args.tick_size,
        stop_mode=args.spike_stop,
        stop_atr_mult=args.stop_atr_mult,
        min_risk_bps=args.min_risk_bps,
        max_risk_atr=args.max_risk_atr,
        target_r=args.target_r,
        breakeven_at_r=args.breakeven_r,
        trail_mode=args.spike_trail,
        max_bars_in_trade=args.spike_time_stop,
        entry_bar_stop=args.entry_bar_stop,
        flat_at_session_end=args.flat_session_end,
        max_trades_per_session=args.max_trades_per_session,
        cooldown_bars=args.cooldown_bars,
    )


def _add_strategy_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("strategy rules")
    group.add_argument(
        "--strategy",
        choices=("emavwap", "smc", "scalp", "spike"),
        default="emavwap",
        help="emavwap: EMA9 x VWAP crossover.  smc: CHoCH -> BOS -> volume "
             "profile POC retest.  scalp: maker mean-reversion at a "
             "volatility band, with a fee floor on the take-profit.  spike: "
             "break of a bar that traded several times the normal volume.",
    )
    group.add_argument(
        "--shorts",
        action="store_true",
        help="also trade the mirror setup (EMA crosses BELOW VWAP, entry on a "
             "break of the cross candle's low). Requires a shortable "
             "instrument such as a perpetual future.",
    )
    group.add_argument(
        "--no-longs", action="store_true", help="disable the long side"
    )
    group.add_argument("--ema", type=int, default=9, help="EMA period (rule 1)")
    group.add_argument(
        "--gap-atr",
        type=float,
        default=0.15,
        help="rule 2: minimum EMA-VWAP gap as a multiple of ATR",
    )
    group.add_argument(
        "--gap-pct",
        type=float,
        default=0.0,
        help="rule 2: minimum EMA-VWAP gap as a fraction of price",
    )
    group.add_argument(
        "--confirm-window",
        type=int,
        default=6,
        help="bars after the cross in which rules 2+3 may confirm (0 = strictest)",
    )
    group.add_argument(
        "--entry-window",
        type=int,
        default=3,
        help="bars after confirmation to accept the breakout (1 = next candle only)",
    )
    group.add_argument("--tick-size", type=float, default=0.01)
    group.add_argument("--entry-buffer-ticks", type=int, default=1)
    group.add_argument(
        "--stop-mode", choices=("cross_low", "atr", "vwap"), default="cross_low"
    )
    group.add_argument("--stop-atr-mult", type=float, default=1.2)
    group.add_argument(
        "--entry-bar-stop",
        choices=("close", "low", "next_bar"),
        default="close",
        help="intrabar path assumption for the stop on the entry bar",
    )
    group.add_argument("--target-r", type=float, default=2.0, help="0 disables the target")
    group.add_argument("--breakeven-r", type=float, default=1.0, help="0 disables")
    group.add_argument("--trail", choices=("none", "ema", "atr"), default="none")
    group.add_argument("--max-bars", type=int, default=0, help="time stop; 0 = off")
    group.add_argument("--max-trades-per-session", type=int, default=0)
    group.add_argument("--cooldown-bars", type=int, default=0)
    group.add_argument(
        "--no-ema-exit", action="store_true", help="do not exit on a close below the EMA"
    )
    group.add_argument(
        "--no-vwap-exit", action="store_true", help="do not exit on a close below the VWAP"
    )
    group.add_argument(
        "--hold-overnight",
        action="store_true",
        help="do not flatten at the session close (not recommended for a VWAP setup)",
    )
    group.add_argument(
        "--allow-loose-price",
        action="store_true",
        help="skip rule 3 (candle must close above both lines)",
    )


def _add_account_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("account & costs")
    group.add_argument("--equity", type=float, default=100_000.0)
    group.add_argument("--sizing",
                       choices=("risk", "fixed", "notional", "fixed_notional"),
                       default="risk")
    group.add_argument("--notional-per-trade", type=float, default=0.0,
                       help="absolute notional per trade for "
                            "--sizing fixed_notional (e.g. $5 margin at 100x "
                            "= 500)")
    group.add_argument("--margin", action="store_true",
                       help="derivatives accounting: reserve notional/leverage "
                            "instead of spending the notional, and model "
                            "liquidation. Required for any leveraged run.")
    group.add_argument("--margin-mode", choices=("cross", "isolated"),
                       default="cross",
                       help="cross: the whole balance backs each position "
                            "(loss NOT capped at the initial margin). "
                            "isolated: loss capped at notional/leverage, but "
                            "liquidation sits ~(1/leverage - mmr) from entry.")
    group.add_argument("--leverage", type=float, default=1.0)
    group.add_argument("--maintenance-margin-rate", type=float, default=0.005,
                       help="equity/notional floor before the exchange closes "
                            "you out (OKX small size is near 0.5%%)")
    group.add_argument("--risk", type=float, default=0.01, help="fraction of equity per trade")
    group.add_argument("--fixed-qty", type=float, default=100.0)
    group.add_argument("--notional-pct", type=float, default=0.25)
    group.add_argument("--max-notional", type=float, default=1.0, help="leverage cap")
    group.add_argument("--commission-bps", type=float, default=1.0,
                       help="taker fee per side, in basis points")
    group.add_argument("--maker-bps", type=float, default=None,
                       help="maker fee per side; limit entries and limit "
                            "take-profits are charged this instead (OKX VIP0: 2)")
    group.add_argument("--commission-per-share", type=float, default=0.0)
    group.add_argument("--slippage-ticks", type=float, default=1.0)
    group.add_argument("--fractional", action="store_true", help="allow fractional quantity")
    group.add_argument("--min-qty", type=float, default=0.0,
                       help="exchange minimum order size in underlying units "
                            "(OKX minSz x ctVal). Decisive on a small account.")
    group.add_argument("--lot-size", type=float, default=1.0,
                       help="minimum tradeable increment (OKX lotSz: 0.01 for "
                            "BTC/ETH/SOL swaps, 1 for XAU). Whole-unit flooring "
                            "silently drops trades on high-priced contracts.")


def build_strategy_config(args: argparse.Namespace) -> StrategyConfig:
    return StrategyConfig(
        trade_longs=not args.no_longs,
        trade_shorts=args.shorts,
        ema_period=args.ema,
        min_gap_atr=args.gap_atr,
        min_gap_pct=args.gap_pct,
        confirm_window=args.confirm_window,
        entry_window=args.entry_window,
        require_close_above_both=not args.allow_loose_price,
        tick_size=args.tick_size,
        entry_buffer_ticks=args.entry_buffer_ticks,
        stop_mode=args.stop_mode,
        stop_atr_mult=args.stop_atr_mult,
        entry_bar_stop=args.entry_bar_stop,
        target_r=args.target_r,
        breakeven_at_r=args.breakeven_r,
        trail_mode=args.trail,
        exit_on_close_below_ema=not args.no_ema_exit,
        exit_on_close_below_vwap=not args.no_vwap_exit,
        max_bars_in_trade=args.max_bars,
        flat_at_session_end=not args.hold_overnight,
        max_trades_per_session=args.max_trades_per_session,
        cooldown_bars=args.cooldown_bars,
    )


def build_strategy(args: argparse.Namespace):
    """The strategy object the engine should run, or None for the default."""
    kind = getattr(args, "strategy", "emavwap")
    if kind == "smc":
        return SmcStrategy(build_smc_config(args))
    if kind == "scalp":
        return MeanReversionScalp(build_scalp_config(args))
    if kind == "spike":
        return VolumeSpikeStrategy(build_spike_config(args))
    return None


def build_backtest_config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        initial_equity=args.equity,
        sizing=args.sizing,
        notional_per_trade=args.notional_per_trade,
        account_mode="margin" if args.margin else "cash",
        leverage=args.leverage,
        margin_mode=args.margin_mode,
        maintenance_margin_rate=args.maintenance_margin_rate,
        risk_pct=args.risk,
        fixed_qty=args.fixed_qty,
        notional_pct=args.notional_pct,
        max_notional_pct=args.max_notional,
        commission_bps=args.commission_bps,
        maker_bps=args.maker_bps,
        commission_per_share=args.commission_per_share,
        slippage_ticks=args.slippage_ticks,
        allow_fractional_qty=args.fractional,
        lot_size=args.lot_size,
        min_qty=args.min_qty,
    )


def load_bars(args: argparse.Namespace) -> List[Bar]:
    session_start = None
    if getattr(args, "session_start", None):
        hour, _, minute = args.session_start.partition(":")
        session_start = time(int(hour), int(minute or 0))
    if args.csv:
        bars = load_csv(args.csv, session_start=session_start)
        if not bars:
            raise SystemExit(f"no usable bars found in {args.csv}")
        return bars
    return generate_synthetic(
        days=args.synthetic,
        bars_per_session=args.bars_per_session,
        seed=args.seed,
    )


def parse_grid(spec: str) -> List[Dict[str, str]]:
    """Parse ``"a=1,2;b=3,4"`` into a list of override dicts (cartesian product)."""
    axes: List[Tuple[str, List[str]]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"bad grid term {chunk!r}; expected field=v1,v2")
        field, _, values = chunk.partition("=")
        axes.append((field.strip(), [v.strip() for v in values.split(",") if v.strip()]))
    if not axes:
        return [{}]
    combos = []
    for values in itertools.product(*[vals for _, vals in axes]):
        combos.append({field: value for (field, _), value in zip(axes, values)})
    return combos


# ----------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    bars = load_bars(args)
    scfg, bcfg = build_strategy_config(args), build_backtest_config(args)
    result = run_backtest(bars, scfg, bcfg, strategy=build_strategy(args))
    metrics = compute_metrics(result)

    if args.json:
        payload = {"symbol": args.symbol, "metrics": metrics.as_dict(),
                   "exit_reasons": result.exit_reasons,
                   "signal_counters": result.signal_counters}
        print(json.dumps(payload, indent=2, default=str))
    else:
        title = f"{STRATEGY_TITLES[getattr(args, 'strategy', 'emavwap')]} - {args.symbol}"
        print(format_report(result, metrics, title=title, show_trades=args.show_trades))
        if args.csv is None:
            print(
                "\nNOTE: these numbers come from SYNTHETIC data and say nothing about\n"
                "      the strategy's real edge.  Re-run with --csv on your own bars."
            )

    if args.trades_csv:
        write_trades_csv(args.trades_csv, result)
        print(f"wrote {args.trades_csv}", file=sys.stderr)
    if args.equity_csv:
        write_equity_csv(args.equity_csv, result)
        print(f"wrote {args.equity_csv}", file=sys.stderr)
    if args.metrics_csv:
        write_metrics_csv(args.metrics_csv, metrics)
        print(f"wrote {args.metrics_csv}", file=sys.stderr)
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    """Dry-run the automation: resting orders, brackets, cancellations."""
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
        stream=sys.stdout,
    )
    bars = load_bars(args)
    scfg, bcfg = build_strategy_config(args), build_backtest_config(args)
    broker = run_paper_session(bars, scfg, bcfg, strategy=build_strategy(args))

    # Fills alternate open/close, so the entries are the even ones.  Counting
    # buys instead would call every short's EXIT an entry, which silently
    # miscounts any strategy that trades both ways.
    entries = [f for i, f in enumerate(broker.fills) if i % 2 == 0]
    still_open = 1 if broker.qty != 0 else 0
    print(f"\npaper run over {len(bars):,} bars: {len(entries)} entries, "
          f"{len(broker.fills)} fills")
    print(f"final equity {broker.equity():,.2f} "
          f"(started {bcfg.initial_equity:,.2f}, flat at end: {broker.qty == 0})")

    if args.compare:
        result = run_backtest(bars, scfg, bcfg, strategy=build_strategy(args))
        gap = broker.equity() - result.final_equity
        # A position still open on the last bar has no trade record yet: the
        # backtest books a trade only when it closes.  Comparing that against
        # the live fill count reports a divergence that is not one.
        closed = len(entries) - still_open
        print(f"backtest equity {result.final_equity:,.2f} -> live/backtest gap {gap:+,.2f}")
        print(f"closed trades: paper {closed} vs backtest {len(result.trades)}"
              + (" (1 position still open, excluded)" if still_open else ""))
        if closed != len(result.trades) or (not still_open and abs(gap) > 0.01):
            print("WARNING: the automation diverges from the backtest; investigate "
                  "before trading it.")
        elif still_open:
            print("automation reproduces the backtest on every closed trade; the "
                  "equity gap is the position still open at the end.")
        else:
            print("automation reproduces the backtest exactly.")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    bars = load_bars(args)
    base_s, base_b = build_strategy_config(args), build_backtest_config(args)
    combos = parse_grid(args.grid)
    rank = SWEEP_METRICS[args.rank]

    rows = []
    for overrides in combos:
        strategy_over = {k: v for k, v in overrides.items() if hasattr(base_s, k)}
        account_over = {k: v for k, v in overrides.items() if hasattr(base_b, k)}
        unknown = set(overrides) - set(strategy_over) - set(account_over)
        if unknown:
            raise SystemExit(f"unknown grid field(s): {sorted(unknown)}")
        scfg = apply_overrides(base_s, strategy_over)
        bcfg = apply_overrides(base_b, account_over)
        metrics = compute_metrics(run_backtest(bars, scfg, bcfg))
        rows.append((overrides, metrics))

    rows.sort(key=lambda row: rank(row[1]), reverse=True)
    keys = sorted({k for overrides, _ in rows for k in overrides})
    header = [f"{k:>14s}" for k in keys] + [
        f"{'trades':>7s}", f"{'win%':>6s}", f"{'PF':>6s}", f"{'expR':>6s}",
        f"{'net':>11s}", f"{'maxDD%':>7s}", f"{'sharpe':>7s}",
    ]
    print(f"parameter sweep: {len(rows)} combinations, ranked by {args.rank}")
    print("  ".join(header))
    print("-" * (len("  ".join(header))))
    for overrides, m in rows:
        cells = [f"{overrides.get(k, ''):>14s}" for k in keys]
        pf = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
        cells += [
            f"{m.trades:>7d}", f"{m.win_rate:>6.1f}", f"{pf:>6s}",
            f"{m.expectancy_r:>6.3f}", f"{m.net_profit:>11,.0f}",
            f"{m.max_drawdown_pct:>7.2f}", f"{m.sharpe:>7.2f}",
        ]
        print("  ".join(cells))
    print(
        "\nA sweep ranks parameters on data you have already seen.  The top row is\n"
        "the best fit to this sample, not the best choice for the future -- use\n"
        "`walkforward` to see whether the ranking survives out of sample."
    )
    return 0


def cmd_walkforward(args: argparse.Namespace) -> int:
    """Pick parameters in-sample, then score them on the next unseen block.

    This is the part that decides whether a sweep found an edge or just fit the
    noise: each fold optimises on its own training block and is graded on data
    that optimisation never touched.
    """
    bars = load_bars(args)
    base_s, base_b = build_strategy_config(args), build_backtest_config(args)
    combos = parse_grid(args.grid)
    rank = SWEEP_METRICS[args.rank]

    sessions: List[str] = []
    for bar in bars:
        if not sessions or sessions[-1] != bar.session:
            sessions.append(bar.session)
    if len(sessions) < args.folds * 2:
        raise SystemExit(
            f"need at least {args.folds * 2} sessions for {args.folds} folds, "
            f"got {len(sessions)}"
        )

    block = len(sessions) // (args.folds + 1)
    by_session = {}
    for bar in bars:
        by_session.setdefault(bar.session, []).append(bar)

    def slice_bars(names: Sequence[str]) -> List[Bar]:
        out: List[Bar] = []
        for name in names:
            out.extend(by_session[name])
        return out

    print(f"walk-forward: {len(sessions)} sessions, {args.folds} folds, "
          f"train {block} / test {block} sessions per fold, ranked by {args.rank}")
    print(f"{'fold':>4s}  {'train':>21s}  {'chosen parameters':<34s} "
          f"{'IS ' + args.rank:>12s} {'OOS ' + args.rank:>12s} "
          f"{'OOS trades':>10s} {'OOS net':>10s}")
    oos_scores: List[float] = []
    oos_net = 0.0
    oos_trades = 0
    for fold in range(args.folds):
        train_names = sessions[fold * block : (fold + 1) * block]
        test_names = sessions[(fold + 1) * block : (fold + 2) * block]
        if not test_names:
            break
        train_bars, test_bars = slice_bars(train_names), slice_bars(test_names)

        best = None
        for overrides in combos:
            s_over = {k: v for k, v in overrides.items() if hasattr(base_s, k)}
            b_over = {k: v for k, v in overrides.items() if hasattr(base_b, k)}
            scfg = apply_overrides(base_s, s_over)
            bcfg = apply_overrides(base_b, b_over)
            score = rank(compute_metrics(run_backtest(train_bars, scfg, bcfg)))
            if best is None or score > best[0]:
                best = (score, overrides, scfg, bcfg)
        assert best is not None
        is_score, overrides, scfg, bcfg = best
        oos = compute_metrics(run_backtest(test_bars, scfg, bcfg))
        oos_scores.append(rank(oos))
        oos_net += oos.net_profit
        oos_trades += oos.trades
        label = ", ".join(f"{k}={v}" for k, v in sorted(overrides.items())) or "(defaults)"
        print(f"{fold + 1:>4d}  {train_names[0]}..{train_names[-1]}  {label:<34s} "
              f"{is_score:>12.3f} {rank(oos):>12.3f} {oos.trades:>10d} "
              f"{oos.net_profit:>10,.0f}")

    if oos_scores:
        average = sum(oos_scores) / len(oos_scores)
        print(f"\nout-of-sample mean {args.rank}: {average:.3f} across "
              f"{len(oos_scores)} folds ({oos_trades} trades, net {oos_net:,.0f})")
        positive = sum(1 for s in oos_scores if s > 0)
        print(f"folds with positive {args.rank}: {positive}/{len(oos_scores)}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    bars = generate_synthetic(
        days=args.days, bars_per_session=args.bars_per_session, seed=args.seed
    )
    write_csv(args.out, bars)
    print(f"wrote {len(bars):,} synthetic bars to {args.out}")
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zetaalgo",
        description="Automated EMA9 x VWAP crossover trading system and backtester.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="backtest the strategy")
    _add_data_args(run)
    _add_strategy_args(run)
    _add_smc_args(run)
    _add_scalp_args(run)
    _add_spike_args(run)
    _add_account_args(run)
    run.add_argument("--trades-csv")
    run.add_argument("--equity-csv")
    run.add_argument("--metrics-csv")
    run.add_argument("--json", action="store_true", help="emit metrics as JSON")
    run.add_argument("--show-trades", type=int, default=10)
    run.set_defaults(func=cmd_run)

    paper = subparsers.add_parser(
        "paper", help="dry-run the automated trader through a simulated broker"
    )
    _add_data_args(paper)
    _add_strategy_args(paper)
    _add_smc_args(paper)
    _add_scalp_args(paper)
    _add_spike_args(paper)
    _add_account_args(paper)
    paper.add_argument("--verbose", action="store_true", help="log every order and fill")
    paper.add_argument(
        "--compare",
        action="store_true",
        help="also run the backtest and assert the two agree",
    )
    paper.set_defaults(func=cmd_paper)

    sweep = subparsers.add_parser("sweep", help="grid-search parameters")
    _add_data_args(sweep)
    _add_strategy_args(sweep)
    _add_smc_args(sweep)
    _add_scalp_args(sweep)
    _add_spike_args(sweep)
    _add_account_args(sweep)
    sweep.add_argument(
        "--grid",
        required=True,
        help='e.g. "target_r=1.5,2,3;min_gap_atr=0.1,0.2"',
    )
    sweep.add_argument("--rank", choices=sorted(SWEEP_METRICS), default="profit_factor")
    sweep.set_defaults(func=cmd_sweep)

    walk = subparsers.add_parser(
        "walkforward", help="out-of-sample validation of a parameter grid"
    )
    _add_data_args(walk)
    _add_strategy_args(walk)
    _add_smc_args(walk)
    _add_scalp_args(walk)
    _add_spike_args(walk)
    _add_account_args(walk)
    walk.add_argument("--grid", required=True)
    walk.add_argument("--folds", type=int, default=4)
    walk.add_argument("--rank", choices=sorted(SWEEP_METRICS), default="expectancy_r")
    walk.set_defaults(func=cmd_walkforward)

    gen = subparsers.add_parser("generate", help="write a synthetic OHLCV CSV")
    gen.add_argument("--days", type=int, default=120)
    gen.add_argument("--bars-per-session", type=int, default=78)
    gen.add_argument("--seed", type=int, default=7)
    gen.add_argument("--out", default="synthetic.csv")
    gen.set_defaults(func=cmd_generate)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
