# ZetaAlgo — EMA 9 × VWAP crossover, automated and backtested

An automated intraday trading system for the "EMA 9 × VWAP crossover" setup,
with an event-driven backtester, walk-forward validation and a live-execution
layer that provably reproduces the backtest.

**Result up front: this setup does not make money on any data tested.**
On 60 sessions of real 5-minute US equity bars, pooled profit factor is 0.80
(still 0.93 with costs set to zero). On OKX perpetual futures — including
XAU-USDT-SWAP — it loses on every instrument, every timeframe and in both
directions. [Equity evidence](#does-it-actually-make-money) ·
[OKX perps evidence](#okx-perpetual-futures-including-xau-usdt-swap)

Pure standard library — **no numpy, no pandas, no install step.**

```bash
python3 tools/fetch_yahoo.py SPY                   # real 5-minute bars
python3 -m zetaalgo run --csv data/spy_5m.csv      # backtest on real data
python3 -m zetaalgo run --synthetic 250            # or on generated data
python3 -m zetaalgo paper --synthetic 30 --compare # dry-run the automation
python3 -m unittest discover -s tests              # 102 tests
```

---

## The setup

1. The 9 EMA crosses the VWAP from below.
2. The 9 EMA stays above the VWAP, with a **clear gap** between the two lines.
3. Price (the candle) is above **both** the 9 EMA and the VWAP.
4. Enter when the candle after the cross candle **breaks the high of the cross
   candle**.

## Turning four sentences into code

A chart annotation is not a specification. Three things had to be decided, and
each is a config flag rather than a silent assumption.

**"Clear gap" needed a number.** The gap is measured against ATR, so the same
threshold means the same thing on a $20 stock and a $2,000 one
(`--gap-atr`, default `0.15`; `--gap-pct` is also available).

**Rules 2 and 3 cannot be judged on the cross candle.** At the moment of the
cross the gap between the lines is by definition ~zero, so a cross candle can
almost never show a "clear gap". The cross therefore only *arms* the setup;
rules 2 and 3 are then tested on each following closed bar within
`--confirm-window` (default 6), and the breakout entry is accepted within
`--entry-window` bars of that confirmation (default 3).

The strictest literal reading — the cross candle itself must satisfy
everything, entry strictly on the next candle — is `--confirm-window 0
--entry-window 1`. It is worth seeing why that reading is not the default:

| reading | trades (250 sessions) | profit factor |
|---|---|---|
| `--confirm-window 0 --entry-window 1` (literal) | 21 | 0.96 |
| default (gap allowed to develop) | 199 | 1.34 |

**Stops and targets are not in the picture at all.** The cross candle's low is
the natural structural stop (`--stop-mode`, also `atr`/`vwap`), with a 2R
target, breakeven at 1R, and exits on a close back below the EMA or VWAP. All
are configurable; `--target-r 0` disables the target.

## Execution realism

Backtests lie by default. These are the assumptions, chosen pessimistically
wherever bar data is ambiguous:

- Entry is a resting **buy-stop** at the cross candle's high plus a buffer. If
  the bar opens above the level, the fill is the **open**, not the level.
- **Stop before target.** When one bar's range contains both levels, the bar
  cannot say which came first, so the loss is always assumed.
- Position size is computed from the price the order **rested** at, not the
  price it filled at — a live system must commit to a quantity before it knows
  the fill, so a breakout that gaps through its trigger really does risk more
  than the nominal `--risk`, and the backtest shows it.
- Commission and slippage are charged on both sides.
- Nothing is decided using a bar's own close, EMA or VWAP before that bar is
  complete. Two tests enforce this: mutating the execution bar cannot change
  the entry plan, and mutating the last bar cannot change any earlier trade.

### The entry-bar stop, and why it is a flag

A buy-stop is crossed *on the way up*, so anything above the trigger in that
bar is plausibly post-fill — but the bar's **low** often happened *before* the
fill. Treating that low as a stop-out fabricates losses on bars that closed
strongly higher. On 120 synthetic sessions this single assumption moved the
profit factor by roughly a third:

| `--entry-bar-stop` | same-bar stop-outs | profit factor |
|---|---|---|
| `low` (assume the low came after the fill) | 23 / 102 | 1.07 |
| `close` (**default**) | 3 / 102 | 1.41 |
| `next_bar` (stop active from the next bar) | 0 / 102 | 1.41 |

`close` and `next_bar` agreeing is the tell that most of those `low` stop-outs
were artifacts, not trades.

## Automation, and why you can trust the backtest

`zetaalgo/live.py` drives a broker with the **same strategy object** that was
backtested — the rules are not re-implemented. The exit logic both paths use is
literally one shared function.

That is checked, not asserted. `--compare` runs the automation through a
simulated broker (resting orders, broker-side brackets, cancellations) and
diffs it against the backtest:

```
$ python3 -m zetaalgo paper --synthetic 20 --compare
paper run over 1,560 bars: 13 entries, 26 fills
final equity 98,647.22 (started 100,000.00, flat at end: True)
backtest equity 98,647.22 -> live/backtest gap +0.00
automation reproduces the backtest exactly.
```

Every entry time, fill price, quantity, exit price and exit reason matches to
the cent across every configuration in the test suite. Getting there caught
four real bugs — including a live-only one where the rolling window slid and
`entry_index` silently stopped pointing at the entry bar, which would have
disabled the time stop and mistimed the breakeven after the first 800 bars.

### Going live

Implement `BrokerAdapter` (5 methods) against your broker's API and feed
`LiveTrader.on_bar()` one **closed** bar at a time:

```python
from zetaalgo import LiveTrader, StrategyConfig

trader = LiveTrader(MyBroker(), StrategyConfig())
trader.warm_up(history)          # seed indicators without trading
for bar in live_feed():          # closed bars only
    trader.on_bar(bar, last_of_session=is_final_bar_of_day(bar))
```

Acting on a forming bar is the most common way a live system stops matching
its backtest.

## Validation

A sweep ranks parameters on data it has already seen, so `walkforward` grades
each choice on the *next* unseen block:

```
$ python3 -m zetaalgo walkforward --synthetic 300 --folds 5 \
      --grid "target_r=1.5,2,3;min_gap_atr=0.05,0.15,0.3"

fold  chosen parameters              IS expectancy_r  OOS expectancy_r  OOS net
   1  min_gap_atr=0.3, target_r=1.5           0.129             0.517   11,141
   2  min_gap_atr=0.3, target_r=2             0.531            -0.064   -2,652
   3  min_gap_atr=0.15, target_r=2            0.115             0.134    7,123
   4  min_gap_atr=0.15, target_r=3            0.177             0.333    9,198
   5  min_gap_atr=0.3, target_r=3             0.487            -0.018   -2,278

folds with positive expectancy_r: 3/5
```

Note fold 2: the best in-sample parameters (0.531) went *negative* out of
sample. That gap is the whole reason this command exists.

## Does it actually make money?

**On 60 sessions of real 5-minute data across six US equities: no.**

```bash
python3 tools/fetch_yahoo.py SPY QQQ AAPL NVDA TSLA AMD   # real 5m bars
python3 -m zetaalgo run --csv data/spy_5m.csv
```

| symbol | trades | win% | profit factor | net | return |
|---|---|---|---|---|---|
| SPY | 54 | 27.8 | 0.60 | −1,412 | −1.41% |
| QQQ | 47 | 27.7 | 0.54 | −2,244 | −2.24% |
| AAPL | 56 | 41.1 | 1.53 | +3,896 | +3.90% |
| NVDA | 52 | 25.0 | 0.51 | −4,685 | −4.68% |
| TSLA | 54 | 38.9 | 1.04 | +420 | +0.42% |
| AMD | 49 | 24.5 | 0.60 | −6,214 | −6.21% |

Four of six lose; pooled profit factor 0.80, net −10,238.

**It is not a cost problem.** Set commission and slippage to zero — which no
one can actually trade at — and the pooled profit factor is still 0.93:

| costs | profit factor | expectancy (R) |
|---|---|---|
| realistic (1bp + 1 tick) | 0.80 | −0.014 |
| zero commission | 0.91 | −0.014 |
| zero costs (impossible) | 0.93 | +0.010 |

**Nor is it the parameters.** Of 72 combinations swept over gap threshold,
target, timing windows and exit rules, 5 beat break-even — and all five are
the same narrow setting. Tested where it was *not* selected, that winner
falls apart:

| slice | trades | profit factor | t-stat |
|---|---|---|---|
| all data (where it won the sweep) | 59 | 1.34 | +0.67 |
| SPY / QQQ / AAPL | 34 | 0.60 | −1.42 |
| NVDA / TSLA / AMD | 25 | 1.88 | +1.95 |

The entire profit comes from three high-volatility names, and TSLA alone
supplies +2,518 of the +3,295 across 8 trades. A t-statistic of +0.67 on 59
trades is indistinguishable from noise (roughly 2.0 is the usual bar). Per
symbol the samples are 5–13 trades, which can support no conclusion at all.

So the sweep did not find an edge; it found the three symbols that trended
during this particular 60-day window.

## OKX perpetual futures (including XAU-USDT-SWAP)

Perps change three things, so the code now supports all three: **shorts**
(`--shorts`, the mirror setup — EMA crossing *below* VWAP, entry on a break of
the cross candle's low), **24/7 sessions** (no exchange session, so VWAP
anchors at UTC midnight — the near-universal crypto convention), and **much
higher fees** (OKX taker is 5 bps/side, 10 bps round trip, versus ~1 bp for
equities).

```bash
python3 tools/fetch_okx.py --specs XAU-USDT-SWAP          # tickSz, ctVal
python3 tools/fetch_okx.py XAU-USDT-SWAP --bar 1H --days 365
python3 -m zetaalgo run --csv data/xau_usdt_swap_1H.csv \
        --shorts --tick-size 0.1 --commission-bps 5
```

**Every instrument, every timeframe, both directions, loses money.**
Longs *and* shorts, OKX taker fees, risk 1%/trade:

| | 5m (60d) | 15m (120d) | 1H (365d) |
|---|---|---|---|
| XAU-USDT-SWAP | PF 0.22 | PF 0.40 | PF 0.49 |
| BTC-USDT-SWAP | PF 0.35 | PF 0.57 | PF 0.67 |
| ETH-USDT-SWAP | PF 0.46 | PF 0.68 | PF 0.73 |
| SOL-USDT-SWAP | PF 0.35 | PF 0.57 | PF 0.89 |

### Why: fees are charged on notional, but risk is the stop distance

This setup's stop is the cross candle's low, which intraday is a *very* small
percentage of price. Risking a fixed fraction of equity through a tight stop
demands a large notional — and the round-trip fee lands on that notional, not
on the risk. The result is a fee bill measured in R:

| | stop distance | fees per trade |
|---|---|---|
| XAU 5m | 0.123% of price | **2.42 R** |
| BTC 5m | 0.172% | 1.71 R |
| SOL 5m | 0.330% | 0.42 R |
| XAU 1H | 0.505% | 0.42 R |
| SOL 1H | 1.569% | 0.09 R |

At 5m on gold, **every trade pays 2.4× the amount being risked, in fees,
before the market does anything at all.** That is not a strategy that needs
tuning; it is arithmetic that cannot be won.

**Gold is the worst candidate here, not the best.** XAU-USDT-SWAP has the
lowest profit factor at every timeframe, precisely because gold's intraday
range is small relative to its price, which makes the stop tight and the
fee-to-risk ratio brutal.

### Raising the timeframe fixes the fees — and it still loses

Moving to 1H cuts the fee drag from 2.42R to 0.09R, essentially eliminating
it. Profit factor improves (0.22 → 0.89) but **never reaches 1.0**. So fees
were not hiding an edge; once they are out of the way, there is still no edge.
Neither direction rescues it over a full year at 1H:

| | longs | shorts |
|---|---|---|
| XAU-USDT-SWAP | PF 0.43 | PF 0.58 |
| BTC-USDT-SWAP | PF 0.44 | PF 0.93 |
| ETH-USDT-SWAP | PF 0.82 | PF 0.65 |
| SOL-USDT-SWAP | PF 0.90 | PF 0.87 |

A 72-combination sweep at 1H (gap, target, timing windows, exit rules, stop
mode) found 6 settings above break-even — every one of them with **13 trades
across a full year and four instruments**. Three trades per instrument per
year is not a tradeable system; it is a sample too small to mean anything.

### If you intend to trade this anyway

Two things are not modelled, and both make the real result *worse*, not
better: **funding** (perps charge it every 8h; the engine ignores it) and
**contract granularity** (OKX sizes in contracts — `ctVal` for XAU is 0.001 oz
with `lotSz` 1 — so small positions round in ways the engine does not
simulate). Check `--specs` before sizing, and note that `--tick-size` must
match the instrument (0.1 for XAU and BTC, 0.01 for ETH and SOL) or the entry
buffer and stop rounding will be wrong.

The most useful thing this repo can do for you now is test a *different* idea
cheaply. The engine, the costs, the fee-drag diagnostic and the parity
guarantee are reusable; only `strategy.py` encodes these particular rules.

### What this does and does not establish

60 sessions is a small sample, one market regime, one asset class. This is
evidence that **the setup as published does not survive contact with recent
US equity data at 5-minute resolution** — not proof it never works anywhere.
What would change the answer is more data (years, not weeks), other sessions
and instruments, and a hypothesis about *why* the edge should exist. If you
have a longer intraday history, point `--csv` at it; the machinery is built
to give you a straight answer.

One real signal did survive: the strict, literal reading of the setup —
confirmation on the cross candle, entry strictly on the next one
(`--confirm-window 0 --entry-window 1`) — was the *only* region above
break-even on real data, the opposite of what the synthetic data suggested.
It trades rarely (59 trades across six symbols in 60 sessions) and, as shown
above, not significantly. But it is the version worth testing on a longer
history first.

### The synthetic numbers elsewhere in this README

The `--synthetic` figures quoted above exercise the engine; they are **not
evidence of an edge**. The generator produces trending and chopping regimes
by construction, and a breakout strategy flatters itself on trending
synthetic data — which is exactly the trap the real-data section above
avoids. Treat synthetic runs as tests of the machinery, never of the idea.

## Layout

| file | role |
|---|---|
| `indicators.py` | EMA, session-anchored VWAP, ATR, cross detection |
| `strategy.py` | the four rules as a bar-by-bar state machine |
| `backtest.py` | event-driven engine, fills and exit priority |
| `broker.py` | position sizing, commission, trade records |
| `live.py` | `BrokerAdapter`, `PaperBroker`, `LiveTrader` |
| `metrics.py` | win rate, PF, expectancy, drawdown, Sharpe/Sortino/Calmar |
| `reporting.py` | text report and CSV exports |
| `data.py` | CSV loading, session labelling, synthetic generator |
| `cli.py` | `run`, `paper`, `sweep`, `walkforward`, `generate` |
| `tools/fetch_yahoo.py` | pull real equity intraday bars into the CSV format |
| `tools/fetch_okx.py` | pull OKX perpetual candles and contract specs |

### Your own data

Any CSV with a timestamp and OHLCV columns; common column spellings and
timestamp formats are auto-detected. **Use intraday bars** — VWAP resets each
session and is meaningless on daily data. For markets whose session crosses
midnight, pass `--session-start 18:00` so the VWAP anchors correctly.

## Risk

This is software for studying a trading idea, not financial advice. A positive
backtest is not a prediction. Trade it on paper first, then with money you can
afford to lose.
