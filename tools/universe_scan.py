#!/usr/bin/env python3
"""Run a strategy across many OKX perps and report the FULL distribution.

    python3 tools/fetch_okx.py --specs-all > /dev/null   # see fetch_okx.py
    python3 tools/universe_scan.py --bar 15m --notional 50

Why this exists
---------------
Testing one or three instruments cannot tell you whether an edge is broad or
whether you happened to pick the two that worked.  Scanning a whole universe
can -- but only if every result is reported.  Filtering to the profitable
names and presenting that as a result is selection, not evidence, and this
tool deliberately makes that hard: it prints losers, blown accounts and
instruments whose tick size makes the strategy unexecutable.

Two checks decide whether breadth means anything:

* **Persistence.** The Spearman correlation between each instrument's
  first-half and second-half profit factor.  If it is near zero, last
  period's winners tell you nothing about next period's, and choosing
  instruments by past performance cannot work -- only owning many of them can.
* **Feasibility.** A per-instrument result assumes a whole account behind each
  one.  The portfolio mode sizes every position from a single account and
  reports the peak leverage that implies, which is usually where a promising
  scan stops being tradeable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zetaalgo.backtest import run_backtest  # noqa: E402
from zetaalgo.config import BacktestConfig  # noqa: E402
from zetaalgo.data import load_csv  # noqa: E402
from zetaalgo.scalp import MeanReversionScalp, ScalpConfig  # noqa: E402

MIN_BARS = 4000
MAX_TICK_BPS = 4.0  # one tick this wide cannot support a ~12 bps target


def spearman(xs: List[float], ys: List[float]) -> float:
    def rank(v: List[float]) -> List[int]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0] * len(v)
        for pos, i in enumerate(order):
            out[i] = pos + 1
        return out
    if len(xs) < 3:
        return 0.0
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def summarise(trades) -> Optional[dict]:
    if not trades:
        return None
    wins = [t.net_pnl for t in trades if t.net_pnl > 0]
    losses = [t.net_pnl for t in trades if t.net_pnl < 0]
    net = sum(t.net_pnl for t in trades)
    n = len(trades)
    mean = net / n
    sd = math.sqrt(sum((t.net_pnl - mean) ** 2 for t in trades) / (n - 1)) if n > 1 else 0.0
    return {
        "n": n,
        "win": 100 * len(wins) / n,
        "pf": (sum(wins) / abs(sum(losses))) if losses else 99.0,
        "net": net,
        "t": (mean / (sd / math.sqrt(n))) if sd else 0.0,
        "liq": sum(1 for t in trades if t.exit_reason == "liquidation"),
    }


def build_config(spec: dict, args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        initial_equity=args.equity,
        sizing="fixed_notional",
        notional_per_trade=args.notional,
        account_mode="margin",
        margin_mode=args.margin_mode,
        leverage=args.leverage,
        maintenance_margin_rate=0.005,
        commission_bps=args.taker_bps,
        maker_bps=args.maker_bps,
        slippage_ticks=1.0,
        lot_size=spec["lotSz"] * spec["ctVal"],
        min_qty=spec["minSz"] * spec["ctVal"],
    )


def run_one(spec: dict, args: argparse.Namespace, half=None):
    name = spec["instId"].lower().replace("-", "_")
    path = os.path.join(args.data_dir, f"{name}_{args.bar}.csv")
    if not os.path.exists(path):
        return None, 0
    bars = load_csv(path)
    total = len(bars)
    if total < MIN_BARS:
        return None, total
    if half is not None:
        sessions = sorted({b.session for b in bars})
        keep = set(half(sessions))
        bars = [b for b in bars if b.session in keep]
    strategy = MeanReversionScalp(ScalpConfig(
        tick_size=spec["tickSz"], trade_shorts=False, entry_z=args.entry_z,
        tp_fraction=1.0, stop_z=args.stop_z, time_stop_bars=args.time_stop,
        min_tp_bps=args.min_tp_bps))
    return run_backtest(bars, backtest_config=build_config(spec, args),
                        strategy=strategy), total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universe", default="universe.json",
                        help="JSON list of {instId,tickSz,lotSz,minSz,ctVal}")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--bar", default="15m")
    parser.add_argument("--equity", type=float, default=100.0)
    parser.add_argument("--notional", type=float, default=50.0)
    parser.add_argument("--leverage", type=float, default=100.0)
    parser.add_argument("--margin-mode", choices=("cross", "isolated"), default="cross")
    parser.add_argument("--maker-bps", type=float, default=2.0)
    parser.add_argument("--taker-bps", type=float, default=5.0)
    parser.add_argument("--entry-z", type=float, default=3.5)
    parser.add_argument("--stop-z", type=float, default=6.0)
    parser.add_argument("--time-stop", type=int, default=24)
    parser.add_argument("--min-tp-bps", type=float, default=8.0)
    args = parser.parse_args()

    with open(args.universe) as handle:
        universe = json.load(handle)

    print(f"{'instrument':20s} {'tickbp':>7s} {'trades':>7s} {'win%':>6s} {'PF':>6s} "
          f"{'net':>9s} {'t':>6s} {'liq':>4s} | {'H1 PF':>6s} {'H2 PF':>6s}")
    usable, skipped = [], []
    for spec in universe:
        tick_bps = spec["tickSz"] / spec["last"] * 10_000 if spec.get("last") else 0.0
        result, total = run_one(spec, args)
        if result is None:
            skipped.append((spec["instId"], f"{total} bars"))
            continue
        full = summarise(result.trades)
        if full is None:
            skipped.append((spec["instId"], "no trades"))
            continue
        first, _ = run_one(spec, args, lambda s: s[: len(s) // 2])
        second, _ = run_one(spec, args, lambda s: s[len(s) // 2:])
        h1 = summarise(first.trades) if first else None
        h2 = summarise(second.trades) if second else None
        usable.append((spec["instId"], tick_bps, full, h1, h2))
        flag = "  UNEXECUTABLE tick" if tick_bps >= MAX_TICK_BPS else ""
        blown = "  ACCOUNT TO ZERO" if full["net"] <= -0.95 * args.equity else ""
        print(f'{spec["instId"]:20s} {tick_bps:7.2f} {full["n"]:7d} {full["win"]:6.1f} '
              f'{full["pf"]:6.2f} {full["net"]:+9.2f} {full["t"]:+6.2f} {full["liq"]:4d} | '
              f'{h1["pf"] if h1 else 0:6.2f} {h2["pf"] if h2 else 0:6.2f}{flag}{blown}')

    for inst, why in skipped:
        print(f"{inst:20s} skipped: {why}")

    if not usable:
        print("\nnothing usable")
        return 1

    positive = [u for u in usable if u[2]["net"] > 0]
    blown = [u for u in usable if u[2]["net"] <= -0.95 * args.equity]
    print(f"\nusable {len(usable)} of {len(universe)}   "
          f"positive {len(positive)}/{len(usable)} ({100*len(positive)/len(usable):.0f}%)   "
          f"accounts to zero {len(blown)}")
    nets = sorted(u[2]["net"] for u in usable)
    print(f"median net {nets[len(nets)//2]:+.2f}   worst {nets[0]:+.2f}   best {nets[-1]:+.2f}")

    pairs = [(u[3]["pf"], u[4]["pf"]) for u in usable if u[3] and u[4]]
    if len(pairs) >= 3:
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        ordered = sorted(pairs, reverse=True)
        half = max(1, len(ordered) // 3)
        top = sorted(p[1] for p in ordered[:half])
        bottom = sorted(p[1] for p in ordered[-half:])
        print(f"\npersistence: first-half vs second-half PF rank correlation {rho:+.3f}")
        print(f"  best third by first half  -> second-half median PF {top[len(top)//2]:.2f}")
        print(f"  worst third by first half -> second-half median PF {bottom[len(bottom)//2]:.2f}")
        if rho < 0.3:
            print("  -> past performance does NOT rank future performance here.")
            print("     Breadth can only be used as diversification, never as selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
