# ZetaAlgo — EMA 9 × VWAP crossover, automated and backtested

An automated intraday trading system for the "EMA 9 × VWAP crossover" setup,
with an event-driven backtester, walk-forward validation and a live-execution
layer that provably reproduces the backtest.

**Three strategies are implemented. Two lose on all data tested; the third
(a maker mean-reversion scalp) clears costs at 15m but not at 5m.**
The EMA9 x VWAP crossover as published:
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

## Second strategy: CHoCH -> BOS -> volume-profile POC retest

A different idea entirely, implemented against the same engine
(`--strategy smc`): track market structure from swing pivots; a **CHoCH** (a
break against the prevailing structure) sets the bias; wait for the **first
BOS** in that direction to confirm it; run a **fixed-range volume profile**
over that swing leg to find the **POC**; then rest a limit order and take the
trade when price **retests** it.

```bash
python3 -m zetaalgo run --csv data/xau_usdt_swap_5m.csv --strategy smc \
        --tick-size 0.1 --commission-bps 5 --maker-bps 2
```

The three judgement calls, all configurable: which leg to profile
(`--leg-anchor`, default the impulse origin), whether to enter at the POC or
the far edge of the value area (`--entry-level`), and where the stop goes
(`--smc-stop`).

### The bug that made it look profitable

A first pass showed this strategy making money at 1H on OKX perps: profit
factor 2.62, **t = +4.28**, all four instruments positive, and a held-out
second half that was *stronger* than the first. It was an artifact.

**73% of trades opened and closed on the same bar, carrying 83% of the
profit.** The engine was crediting the target on the entry bar. For a
*breakout* entry that is defensible -- the trigger is crossed on the way up,
so the range above it is plausibly post-fill. For a **pullback** entry it is
exactly inverted: a limit fills as price moves *against* the trade, so that
bar's favourable extreme may well have printed *before* the fill. Claiming it
invents profit out of an unknowable intrabar path.

The fix is a continuity argument. If a bar **closes** beyond a level, price
must have traversed that level after the fill, whatever route it took. If the
bar merely *wicks* past and closes back inside, nothing is proven. So on the
entry bar of a limit fill, only outcomes the close proves are booked; the rest
carry to the next bar, where the whole range is legitimately post-fill.

| 1H OKX perps, same config | profit factor | net | t-stat |
|---|---|---|---|
| entry-bar target credited (wrong) | 2.62 | +24,299 | +4.28 |
| only what the close proves (correct) | 0.74 | −7,546 | −1.31 |

That single assumption *was* the entire edge. It is the same class of error as
the entry-bar stop documented above, and it is why this repo treats every
intrabar path assumption as an explicit, testable decision rather than a
default.

### What it does on real data

Corrected, with maker fees credited on the limit entry (OKX VIP0: 2 bps maker,
5 bps taker):

| pooled sweep | combinations above break-even |
|---|---|
| OKX perps 5m | **0 / 48** (best PF 0.56) |
| OKX perps 1H | **0 / 40** (best PF 0.74) |
| US equities 5m | 9 / 40 (best PF 1.34) |

The equity result is an in-sample sweep, and it does not survive: tested where
it was not selected, the winner falls to PF 0.73 on the held-out second half
and PF 0.65 on SPY/QQQ/AAPL, leaving NVDA/TSLA/AMD to carry it -- the same
three high-volatility names that carried the EMA/VWAP sweep.

One honest positive: at **zero fees** the 5m signal is mildly profitable on
perps (PF 1.48-1.65 on BTC/ETH/SOL). Unlike EMA/VWAP, which lost even for
free, there is something in this structure. It is just smaller than the cost
of trading it -- and the stops here are tighter still, so the fee-per-R
arithmetic from the section above bites harder, not less.

## Do TP and SL help? (and what a solution would look like)

Take-profit and stop-loss were in every backtest above -- `--target-r` is the
TP, `--smc-stop`/`--stop-mode` the SL. But they were never the *only* exits,
and that turned out to matter: on OKX 5m only 60% of EMA/VWAP exits and 54%
of CHoCH/BOS exits were an actual TP or SL. The rest were rule exits
(`close_below_ema`, `structure_flip`) and end-of-session flattening.

**Those rule exits were actively harmful.** Switching to TP/SL only:

| | with rule exits | TP/SL only |
|---|---|---|
| EMA/VWAP | PF 0.39 | **PF 0.52** |
| CHoCH/BOS/POC | PF 0.45 | **PF 0.63** |

Better, still losing. The reason is the fee-per-R arithmetic: a trade needs a
win rate of `(1 + fee_in_R) / (1 + TP_in_R)` just to break even, and at 5m
with tight stops `fee_in_R` is around 1.0, which demands a ~99% win rate at
1:1. That is the whole problem in one line.

So the lever is not the TP -- it is the **stop width**, because that is what
sets fee-per-R. Widening it on EMA/VWAP at 5m:

| stop | fee per R | profit factor |
|---|---|---|
| 1x ATR | 0.98 R | 0.39 |
| 3x ATR | 0.31 R | 0.86 |
| 6x ATR | 0.15 R | 0.84 |

Fees stop mattering and the strategy converges to **PF ~1.0 -- a coin flip**.
That is the signature of no edge, not of a cost problem, and it is the clearest
evidence in this repo that the EMA/VWAP setup has nothing to harvest.

### The one configuration that is not obviously dead

CHoCH/BOS/POC at **1H**, wide ATR stops, TP/SL as the only exits, limit entry
(maker), correct OKX lot sizes:

| | trades | PF | win% | t-stat |
|---|---|---|---|---|
| in-sample (best of 16 cells) | 224 | 1.32 | 47.8 | +2.01 |
| **held-out second half** | 104 | **1.13** | 43.3 | **+0.59** |
| XAU / BTC / ETH / SOL | all four | 1.43 / 1.13 / 1.39 / 1.39 | | |

Encouraging: 9 of 16 cells clear break-even in a contiguous plateau (stops of
3-4x ATR), not one lucky spike, and all four instruments are independently
positive. Discouraging: the held-out half falls to 1.13 with t = +0.59, which
is not distinguishable from luck, and +2.01 in-sample is unremarkable after
searching 16 cells.

Verdict: **plausible small edge, unproven.** Not something to fund. The
honest next step is more history (this is one year, one regime), not more
parameter search on the same year.

At the **5m** timeframe originally asked about, no TP/SL geometry clears
break-even: 0 of 20 on either strategy.

### A bug this experiment exposed: contract granularity

Position size was floored to whole units, which is correct for shares and
wrong for perps. Risking 1% of 100k through a 4x ATR stop on BTC at ~71,800
is 0.3887 contracts -- floored to **zero**, and the trade silently disappeared.
BTC showed 4 trades in a year where it should have had 63.

`--lot-size` now takes the instrument's real increment (OKX `lotSz`: 0.01 for
BTC/ETH/SOL swaps, 1 for XAU; see `tools/fetch_okx.py --specs`). The earlier
conclusions in this README were re-checked against the fix and are unchanged,
because they used tight structural stops where sizing rarely rounded to zero;
only the wide-stop experiment was affected.

## Third strategy: maker mean-reversion scalp (`--strategy scalp`)

Built to attack the constraint the first two died to, rather than to express a
chart pattern. The arithmetic that killed them is::

    fee_in_R = 2 * fee_rate / stop_percent

so a 5-minute scalp has exactly two ways out: **never cross the spread**, and
**make the target dwarf the fee**. Hence:

* Entry is a resting **limit** at a volatility band -- maker, 2 bps on OKX
  rather than 5. Filled when a burst pushes price to an extreme.
* The take-profit is also a **limit**, at a price (a fraction of the way back
  to the mean), so the round trip is ~4 bps instead of ~10. Only stop-outs
  cross.
* `--min-tp-bps` is a **hard fee floor**: any setup whose take-profit is worth
  less than N basis points is refused. This is "cover the fees" written as a
  gate the strategy cannot trade around.
* The lagging indicator is **confirmation only** -- a slow EMA that can veto
  fading a hard trend but can never start a trade.

### Measure the edge before building on it

The reversion edge is real and consistent. Forward return in bps after a dip
of 2+ ATR below the 20-bar mean, long side, **before costs** (the maker round
trip is 4 bps, so subtract that):

| 5m, gross bps | +3 bars | +12 | +24 |
|---|---|---|---|
| SOL | +2.7 | +4.5 | +6.7 |
| ETH | +2.1 | +6.0 | +10.7 |
| BTC | +0.4 | +0.6 | +3.1 |
| **XAU** | +0.9 | +2.5 | +4.5 |

Every instrument reverts; the sign is consistent across all four and grows
with horizon. But at 5m the numbers sit right on top of the 4 bps cost. **XAU
never clears it at any depth** (net -0.9 to +0.8 bps) -- gold's intraday range
is too small relative to its price, the same reason it was worst for the
earlier strategies.

### Result: 5m does not work, 15m does

Long-only, ETH+SOL+BTC, maker 2 / taker 5, correct lot sizes, grid over entry
depth, stop width and time stop:

| timeframe | cells above break-even | best |
|---|---|---|
| **5m** | **1 / 27** | PF 1.03 (t = +0.27) |
| 15m | **24 / 27** | PF 1.48 (t = +2.09) |

At 5m the edge and the cost are the same size and the result is a flat
nothing. At 15m the same strategy clears it on a broad plateau -- 24 of 27
cells, not one lucky cell -- and every instrument is independently positive:

| SL 6 ATR / TP at mean / 24-bar stop, 15m | trades | PF | t |
|---|---|---|---|
| in-sample (best of 27) | 161 | 1.48 | +2.09 |
| **held-out second half** | 93 | **1.22** | **+0.76** |
| ETH / SOL / BTC | 57 / 45 / 59 | 1.69 / 1.24 / 1.49 | |

It survives the assumptions that killed earlier results: only 1 trade in 161
closes on its entry bar (the intrabar artifact that faked the SMC edge), no
lookahead, and it stays positive at **taker fees on both sides**
(+9,178 instead of +13,619) and at 3 ticks of slippage.

**Why 15m works and 5m does not:** reversion grows with holding time while
the fee is a fixed cost per round trip. Five minutes is too short to amortise
4 bps; fifteen is not. That is the whole result, and it is a statement about
cost structure, not about indicators.

Verdict: **the most promising thing in this repo, and still not proven.**
Held-out t = +0.76 is not significant, and 161 trades is one regime of one
year. Trade it on paper first.

### A live-trading bug this strategy exposed

The first two strategies quote a *static* level (a cross-candle high, a POC).
This one re-derives its quote every bar, and that exposed a real defect: the
live trader left the old order resting instead of cancel/replacing it, so it
traded a stale price the backtest never used -- 81 fills live against 57 in
the backtest. `--compare` caught it. Any market-making strategy needs the
cancel/replace path, and it is now there and tested.

## What this looks like on a $100 account

Contract minimums, not the strategy, decide what a small account can trade.
On OKX one minimum order is worth **$1.02 (SOL) to $7.72 (BTC)** at recent
prices, while a 1%-risk position at these stop widths needs about $60 of
notional. So at $100 nothing is rounded away -- `--min-qty` reports **0
skipped signals** -- and the percentage returns are the same as on a large
account. Pass the real numbers or the backtest will lie to you:

| instrument | `--tick-size` | `--lot-size` | `--min-qty` |
|---|---|---|---|
| SOL-USDT-SWAP | 0.01 | 0.01 | 0.01 |
| ETH-USDT-SWAP | 0.01 | 0.001 | 0.001 |
| BTC-USDT-SWAP | 0.1 | 0.0001 | 0.0001 |
| XAU-USDT-SWAP | 0.1 | 0.001 | 0.001 |

(`lot_size` and `min_qty` are in underlying units: OKX `lotSz` x `ctVal`.)

**15m scalp, $100, risk 1%/trade, 120 days, full period** -- the window the
parameters were chosen in, so read it as a ceiling, not a forecast:

| instrument | trades | win% | PF | net $ | fees $ | max DD $ | return |
|---|---|---|---|---|---|---|---|
| ETH | 57 | 63.2 | 1.70 | **+6.93** | 1.73 | 1.79 | +6.93% |
| BTC | 59 | 69.5 | 1.50 | +4.27 | 2.62 | 3.57 | +4.27% |
| SOL | 45 | 62.2 | 1.24 | +2.16 | 1.25 | 2.80 | +2.16% |
| XAU | 80 | 51.2 | 0.69 | −5.09 | 4.09 | 7.73 | −5.09% |

**Held-out second half only** -- the honest expectation, on data no parameter
ever saw:

| instrument | trades | win% | PF | net $ | max DD $ | return |
|---|---|---|---|---|---|---|
| ETH | 31 | 61.3 | 1.46 | **+2.60** | 1.74 | +2.60% |
| SOL | 27 | 66.7 | 1.48 | +2.11 | 2.53 | +2.11% |
| BTC | 35 | 60.0 | 0.85 | −1.04 | 3.44 | −1.04% |
| XAU | 42 | 61.9 | 1.01 | +0.06 | 2.67 | +0.06% |

So the realistic figure is **two to three dollars over two months** on the
better instruments -- roughly a dollar a month -- against a drawdown of
similar size. And funding, which the engine does not model, takes a bite out
of even that: average hold is 4 hours, so about half of trades cross a funding
window.

| funding rate | cost over 57 ETH trades | share of the profit |
|---|---|---|
| typical 0.01% | ~$0.16 | 2.3% |
| elevated 0.05% | ~$0.81 | 11.7% |
| stressed 0.10% | ~$1.62 | 23.4% |

### The conclusion a $100 account should draw

The strategy does not scale badly -- the percentages hold. The problem is that
a percentage of $100 is not worth the operational risk of running an automated
system: one mis-set parameter, one unattended outage, one funding spike costs
more than a month of edge. And the edge itself is unproven (held-out
t = +0.76).

At $100 the correct use of this is **validation, not profit**: run it small and
check that your maker fills, fees and slippage actually match the backtest.
That question is worth a few dollars to answer, and it is the one thing no
amount of further backtesting can settle.

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
| `structure.py` | swing pivots, CHoCH/BOS, fixed-range volume profile |
| `smc.py` | CHoCH -> BOS -> POC retest strategy (`--strategy smc`) |
| `scalp.py` | maker mean-reversion scalp with a fee floor (`--strategy scalp`) |
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
