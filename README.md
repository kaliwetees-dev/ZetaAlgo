# ZetaAlgo — EMA 9 × VWAP crossover, automated and backtested

An automated intraday trading system for the "EMA 9 × VWAP crossover" setup,
with an event-driven backtester, walk-forward validation and a live-execution
layer that provably reproduces the backtest.

Pure standard library — **no numpy, no pandas, no install step.**

```bash
python3 -m zetaalgo run --synthetic 250            # backtest on generated data
python3 -m zetaalgo run --csv data/spy_5m.csv      # backtest on your own bars
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

## About the numbers in this README

**They come from synthetic data and are not evidence of an edge.** The
generator (`--synthetic`) produces trending and chopping regimes so the engine
can be exercised end to end, but it is not a market. Point `--csv` at your own
intraday bars before drawing any conclusion. The report says so too.

What the synthetic runs *do* show is that the mechanics behave sensibly: rule 2
is doing real work (a 0.05 ATR gap threshold is unprofitable, 0.30 is the best
of the grid), the edge is thin relative to costs, and the win rate is ~36% with
a ~2.4 payoff ratio — this is a low-win-rate, high-payoff profile, so it takes
long losing streaks (7 in a row here) to get paid.

Cost sensitivity, same 250 sessions:

| slippage | profit factor | expectancy (R) |
|---|---|---|
| 0 ticks | 1.36 | +0.171 |
| 1 tick (default) | 1.34 | +0.171 |
| 3 ticks | 1.24 | +0.126 |

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

### Your own data

Any CSV with a timestamp and OHLCV columns; common column spellings and
timestamp formats are auto-detected. **Use intraday bars** — VWAP resets each
session and is meaningless on daily data. For markets whose session crosses
midnight, pass `--session-start 18:00` so the VWAP anchors correctly.

## Risk

This is software for studying a trading idea, not financial advice. A positive
backtest is not a prediction. Trade it on paper first, then with money you can
afford to lose.
