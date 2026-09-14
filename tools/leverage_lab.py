#!/usr/bin/env python3
"""What leverage suits a small account?  Three experiments, one $100 balance.

    python3 tools/leverage_lab.py --data-dir data_2y --bar 15m

Why this exists
---------------
"Which leverage is best?" is usually answered with a return column, and a
return column always points at the maximum the exchange allows.  It has to:
at a fixed stop distance, return is linear in size and drawdown is linear in
size too, so every ratio of the two is flat while the *level* of both keeps
climbing.  Ranking by return, or by return-per-drawdown, therefore cannot
select a leverage -- it can only rediscover the cap.

So this tool reports the three things that actually differ:

* **A. Fixed margin per trade.**  Posting the same $5 of margin at a higher
  leverage buys a bigger position.  This is the ladder people mean, and it is
  a *position-size* ladder wearing a leverage costume.
* **B. Fixed position size.**  The same $250 position held at 5x and at 100x.
  Identical trades, identical P&L.  Only the margin locked up and the
  distance to liquidation change -- which is the whole truth about what the
  leverage dial does.
* **C. Fixed fraction of equity, compounded.**  The only ladder with an
  interior optimum: too little exposure earns nothing, too much loses money
  even on a winning edge, because a drawdown shrinks the base that has to
  earn it back.  This is the question a $100 account is actually asking.

Every path is checked against cross-margin liquidation (the balance is the
collateral for all open positions at once) and against margin availability.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zetaalgo.backtest import run_backtest  # noqa: E402
from zetaalgo.broker import Trade  # noqa: E402
from zetaalgo.config import BacktestConfig  # noqa: E402
from zetaalgo.data import load_csv  # noqa: E402
from zetaalgo.scalp import MeanReversionScalp, ScalpConfig  # noqa: E402

# tickSz, lotSz x ctVal (the tradeable increment in underlying units), minSz x ctVal
SPECS: Dict[str, Tuple[float, float, float]] = {
    "btc_usdt_swap": (0.1, 0.0001, 0.0001),
    "eth_usdt_swap": (0.01, 0.001, 0.001),
    "sol_usdt_swap": (0.01, 0.01, 0.01),
    "xrp_usdt_swap": (0.0001, 1.0, 1.0),
    "doge_usdt_swap": (1e-05, 10.0, 10.0),
    "bnb_usdt_swap": (0.1, 0.01, 0.01),
}

# The config that survived the 2.2-year test: long-only, maker-only, with the
# variance-ratio regime gate.  Changing it here is a different experiment.
SCALP_KW = dict(
    trade_shorts=False,
    entry_z=3.5,
    tp_fraction=1.0,
    stop_z=6.0,
    time_stop_bars=24,
    min_tp_bps=8.0,
    trend_filter=True,
    max_variance_ratio=1.0,
)


def collect_trades(args: argparse.Namespace) -> List[Trade]:
    """Run every asset once, at the reference notional, and pool the trades.

    Position size scales P&L linearly, so one run per asset is enough to
    describe every size -- as long as the size is big enough that the lot
    increment and exchange minimum do not change *which* trades happen.
    ``--check-linearity`` verifies exactly that.
    """
    pooled: List[Trade] = []
    for asset in args.assets:
        path = os.path.join(args.data_dir, f"{asset}_{args.bar}.csv")
        if not os.path.exists(path):
            print(f"  {asset}: no data at {path}", file=sys.stderr)
            continue
        tick, lot, min_qty = SPECS[asset]
        bars = load_csv(path)
        config = BacktestConfig(
            initial_equity=args.reference_notional * 4,
            sizing="fixed_notional",
            notional_per_trade=args.reference_notional,
            account_mode="margin",
            margin_mode="cross",
            leverage=100.0,  # only lifts the margin constraint on this probe run
            maintenance_margin_rate=args.maintenance,
            commission_bps=args.taker_bps,
            maker_bps=args.maker_bps,
            slippage_ticks=1.0,
            lot_size=lot,
            min_qty=min_qty,
        )
        strategy = MeanReversionScalp(ScalpConfig(tick_size=tick, **SCALP_KW))
        pooled.extend(run_backtest(bars, backtest_config=config, strategy=strategy).trades)
    pooled.sort(key=lambda t: t.entry_ts)
    return pooled


def unit_return(trade: Trade, reference_notional: float) -> float:
    """P&L as a fraction of the position's entry notional."""
    notional = abs(trade.entry_price * trade.qty)
    return trade.net_pnl / notional if notional else 0.0


def stop_distance_pct(trade: Trade) -> float:
    return 100.0 * abs(trade.entry_price - trade.initial_stop) / trade.entry_price


def adverse_fraction(trade: Trade) -> float:
    """Deepest unrealised loss the trade ever showed, as a fraction of notional.

    ``mae_r`` is that excursion in units of the initial stop distance, signed
    negative, so negating it and scaling by the stop distance recovers the
    loss as a share of the position.
    """
    return abs(min(0.0, trade.mae_r)) * stop_distance_pct(trade) / 100.0


class Path:
    """One equity path: trades opened and settled in timestamp order.

    Cross margin is modelled the way the exchange does it -- the whole balance
    is collateral for every open position at once, so liquidation is checked
    against the summed maintenance requirement, not per position.
    """

    def __init__(self, equity: float, leverage: float, maintenance: float) -> None:
        self.equity = equity
        self.peak = equity
        self.low = equity
        self.max_dd = 0.0
        self.leverage = leverage
        self.maintenance = maintenance
        self.open_notional = 0.0
        self.peak_open_notional = 0.0
        self.peak_concurrent = 0
        self.peak_margin = 0.0
        self.concurrent = 0
        self.skipped_margin = 0
        self.taken = 0
        self.liquidated = False
        self.peak_stress = 0.0      # worst summed unrealised loss, in dollars
        self.peak_stress_pct = 0.0  # ... as a share of the balance at the time
        self.min_cushion = float("inf")  # (balance - stress) / maintenance owed
        self.stress_liquidated = False

    def margin_for(self, notional: float) -> float:
        return notional / self.leverage

    def open(self, notional: float) -> bool:
        used = self.margin_for(self.open_notional + notional)
        if notional <= 0 or used > self.equity:
            self.skipped_margin += 1
            return False
        self.open_notional += notional
        self.concurrent += 1
        self.taken += 1
        self.peak_open_notional = max(self.peak_open_notional, self.open_notional)
        self.peak_concurrent = max(self.peak_concurrent, self.concurrent)
        self.peak_margin = max(self.peak_margin, used)
        return True

    def close(self, notional: float, ret: float) -> None:
        self.open_notional = max(0.0, self.open_notional - notional)
        self.concurrent = max(0, self.concurrent - 1)
        self.equity = max(0.0, self.equity + notional * ret)
        self.peak = max(self.peak, self.equity)
        self.low = min(self.low, self.equity)
        if self.peak > 0:
            self.max_dd = max(self.max_dd, (self.peak - self.equity) / self.peak)
        if self.equity <= self.open_notional * self.maintenance or self.equity <= 0:
            self.liquidated = True


def simulate(
    trades: Sequence[Trade],
    reference_notional: float,
    leverage: float,
    maintenance: float,
    equity: float,
    size: str,
    param: float,
) -> Path:
    """Walk the pooled trades once.

    ``size`` picks how each position is sized: ``margin`` posts ``param``
    dollars of margin (so the position is ``param x leverage``), ``notional``
    takes a fixed ``param``-dollar position, ``fraction`` takes ``param`` of
    the *current* balance as margin and compounds.
    """
    path = Path(equity, leverage, maintenance)
    events: List[Tuple] = []
    for i, trade in enumerate(trades):
        events.append((trade.entry_ts, 0, i))
        events.append((trade.exit_ts, 1, i))
    events.sort(key=lambda e: (e[0], e[1]))

    live: Dict[int, float] = {}
    for _, kind, i in events:
        trade = trades[i]
        if kind == 0:
            if size == "margin":
                notional = param * leverage
            elif size == "notional":
                notional = param
            else:
                notional = param * path.equity * leverage
            if path.open(notional):
                live[i] = notional
        else:
            notional = live.pop(i, None)
            if notional is not None:
                path.close(notional, unit_return(trade, reference_notional))
        # Stress test the open book.  The engine settles P&L at the exit, so
        # the equity path above never sees an unrealised loss -- but the
        # exchange does, and liquidates on it.  Summing every open position's
        # deepest adverse excursion is an upper bound (they need not coincide),
        # so treat a breach as a warning, not a certainty.
        if live:
            stress = sum(live[j] * adverse_fraction(trades[j]) for j in live)
            path.peak_stress = max(path.peak_stress, stress)
            if path.equity > 0:
                path.peak_stress_pct = max(path.peak_stress_pct, 100 * stress / path.equity)
            owed = path.open_notional * path.maintenance
            if owed > 0:
                path.min_cushion = min(path.min_cushion, (path.equity - stress) / owed)
            if path.equity - stress <= owed:
                path.stress_liquidated = True
        if path.liquidated:
            break
    return path


def trade_stats(trades: Sequence[Trade], reference_notional: float) -> dict:
    rets = [unit_return(t, reference_notional) for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    n = len(rets)
    mean = sum(rets) / n
    sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / (n - 1)) if n > 1 else 0.0
    stops = sorted(stop_distance_pct(t) for t in trades)
    return {
        "n": n,
        "win": 100.0 * len(wins) / n,
        "pf": (sum(wins) / abs(sum(losses))) if losses else 99.0,
        "mean_bps": 10_000 * mean,
        "t": (mean / (sd / math.sqrt(n))) if sd else 0.0,
        "median_stop": stops[len(stops) // 2],
        "p95_stop": stops[int(0.95 * (len(stops) - 1))],
        "max_stop": stops[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data_2y")
    parser.add_argument("--bar", default="15m")
    parser.add_argument("--assets", default="eth_usdt_swap,sol_usdt_swap,"
                        "xrp_usdt_swap,doge_usdt_swap,bnb_usdt_swap",
                        help="comma separated; BTC is excluded by default because "
                             "it lost money over the same window")
    parser.add_argument("--equity", type=float, default=100.0)
    parser.add_argument("--reference-notional", type=float, default=50.0)
    parser.add_argument("--margin-per-trade", type=float, default=5.0)
    parser.add_argument("--fixed-notional", type=float, default=250.0)
    parser.add_argument("--maintenance", type=float, default=0.005)
    parser.add_argument("--maker-bps", type=float, default=2.0)
    parser.add_argument("--taker-bps", type=float, default=5.0)
    parser.add_argument("--levels", default="1,2,5,10,25,50,75,100")
    parser.add_argument("--fractions", default="0.005,0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.25")
    parser.add_argument("--check-linearity", action="store_true",
                        help="re-run at 4x the reference notional and confirm the "
                             "trade set and per-dollar P&L are unchanged")
    args = parser.parse_args()
    args.assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    for asset in args.assets:
        if asset not in SPECS:
            parser.error(f"no contract spec for {asset}")
    levels = [float(x) for x in args.levels.split(",")]
    fractions = [float(x) for x in args.fractions.split(",")]

    trades = collect_trades(args)
    if not trades:
        print("no trades")
        return 1
    stats = trade_stats(trades, args.reference_notional)
    span = (trades[-1].exit_ts - trades[0].entry_ts).days
    print(f"{len(args.assets)} assets  {args.bar}  "
          f"{trades[0].entry_ts:%Y-%m-%d}..{trades[-1].exit_ts:%Y-%m-%d} ({span} days)")
    print(f"{stats['n']} trades  win {stats['win']:.1f}%  PF {stats['pf']:.2f}  "
          f"avg {stats['mean_bps']:+.2f} bps/trade  t {stats['t']:+.2f}")
    print(f"stop distance: median {stats['median_stop']:.2f}%  "
          f"95th {stats['p95_stop']:.2f}%  worst {stats['max_stop']:.2f}%")

    if args.check_linearity:
        base = args.reference_notional
        args.reference_notional = base * 4
        again = collect_trades(args)
        args.reference_notional = base
        a = sum(unit_return(t, base) for t in trades)
        b = sum(unit_return(t, base * 4) for t in again)
        print(f"linearity check: {len(trades)} vs {len(again)} trades, "
              f"summed per-dollar P&L {a:+.6f} vs {b:+.6f}")

    print(f"\nA. FIXED ${args.margin_per_trade:.0f} MARGIN PER TRADE "
          f"(position = margin x leverage)")
    print(f"{'lev':>5s} {'position':>9s} {'final $':>9s} {'return':>9s} {'maxDD':>7s} "
          f"{'low $':>7s} {'peak exp':>9s} {'buffer':>7s} {'stress':>7s} {'cushion':>8s} {'risk':>5s}")
    for lev in levels:
        path = simulate(trades, args.reference_notional, lev, args.maintenance,
                        args.equity, "margin", args.margin_per_trade)
        buffer_pct = ((100.0 * (path.low - path.peak_open_notional * args.maintenance)
                       / path.peak_open_notional) if path.peak_open_notional else 0.0)
        risk = "LIQ" if path.liquidated else ("warn" if path.stress_liquidated else "ok")
        print(f"{lev:4.0f}x {args.margin_per_trade*lev:9,.0f} {path.equity:9,.2f} "
              f"{100*(path.equity/args.equity-1):+8.1f}% {100*path.max_dd:6.1f}% "
              f"{path.low:7.2f} {path.peak_open_notional:9,.0f} {buffer_pct:6.2f}% "
              f"{path.peak_stress_pct:6.0f}% {path.min_cushion:7.1f}x {risk:>5s}")
    print("  peak exp = largest total position value open at once (5 assets can all be open)")
    print("  buffer   = adverse move across that whole book that liquidates the account")
    print("  stress   = worst summed unrealised loss the open book actually showed,")
    print("             as a share of the balance")
    print("  cushion  = balance less that stress, over the maintenance margin owed:")
    print("             1.0x is the liquidation point, so this is the margin of safety")

    print(f"\nB. THE SAME ${args.fixed_notional:.0f} POSITION AT EVERY LEVERAGE")
    print(f"{'lev':>5s} {'margin ea':>9s} {'final $':>9s} {'return':>9s} {'maxDD':>7s} "
          f"{'iso liq':>8s} {'cross liq':>9s} {'skip':>5s} {'liq':>4s}")
    for lev in levels:
        path = simulate(trades, args.reference_notional, lev, args.maintenance,
                        args.equity, "notional", args.fixed_notional)
        iso = 100.0 * (1.0 / lev - args.maintenance)
        cross = (100.0 * (path.low - path.peak_open_notional * args.maintenance)
                 / path.peak_open_notional) if path.peak_open_notional else 0.0
        print(f"{lev:4.0f}x {args.fixed_notional/lev:9,.2f} {path.equity:9,.2f} "
              f"{100*(path.equity/args.equity-1):+8.1f}% {100*path.max_dd:6.1f}% "
              f"{iso:7.2f}% {cross:8.2f}% {path.skipped_margin:5d} "
              f"{'YES' if path.liquidated else 'no':>4s}")
    print("  iso liq   = adverse move that liquidates an ISOLATED position: 1/lev - maintenance")
    print("  cross liq = adverse move on the whole open book that would have liquidated the")
    print("              account at its lowest balance -- this is what actually applies")

    print("\nC. FIXED FRACTION OF EQUITY AS MARGIN, COMPOUNDED (100x)")
    print(f"{'frac':>6s} {'start pos':>9s} {'final $':>9s} {'return':>10s} {'maxDD':>7s} "
          f"{'low $':>7s} {'CAGR':>8s} {'liq':>4s}")
    best = None
    for frac in fractions:
        path = simulate(trades, args.reference_notional, 100.0, args.maintenance,
                        args.equity, "fraction", frac)
        years = max(span / 365.25, 1e-9)
        cagr = ((path.equity / args.equity) ** (1 / years) - 1) if path.equity > 0 else -1.0
        print(f"{100*frac:5.1f}% {frac*args.equity*100:9,.0f} {path.equity:9,.2f} "
              f"{100*(path.equity/args.equity-1):+9.1f}% {100*path.max_dd:6.1f}% "
              f"{path.low:7.2f} {100*cagr:+7.1f}% {'YES' if path.liquidated else 'no':>4s}")
        if not path.liquidated and (best is None or path.equity > best[1]):
            best = (frac, path.equity, path.max_dd)
    if best:
        print(f"  geometric optimum at {100*best[0]:.1f}% of equity per position "
              f"(${best[1]:,.2f} final, {100*best[2]:.1f}% drawdown).")
        print(f"  Half that fraction is the usual practical choice: it keeps about "
              f"three quarters of the growth for half the drawdown, and the optimum "
              f"itself is an in-sample estimate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
