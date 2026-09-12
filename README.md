# ZetaAlgo — EMA 9 × VWAP crossover, automated and backtested

An automated intraday trading system for the "EMA 9 × VWAP crossover" setup,
with an event-driven backtester, walk-forward validation and a live-execution
layer that provably reproduces the backtest.

**Three strategies are implemented and none of them survives a full year of
data.** The maker mean-reversion scalp clears costs at 15m over 120 days and
across 30 instruments, but extended to 365 days through a bear market the
same parameters return a profit factor of 0.90. See
[the regime test](#regime-test-over-a-full-year-the-120-day-result-does-not-survive).
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

### Leveraged sizing: $100 with $5 margin per trade at 100x

`--sizing fixed_notional --notional-per-trade 500 --margin --leverage 100`
models this properly: margin accounting reserves notional/leverage instead of
spending the notional, and liquidation is checked against the bar's adverse
extreme (including on the entry bar, where a levered position can be closed
out before its own stop is reached).

First, what the numbers mean. **$5 of margin at 100x is a $500 position, which
on a $100 account is 5x account leverage** -- the 100x is only what the
exchange permits, not what you are running. The consequence is that risk per
trade is no longer 1%:

| | value |
|---|---|
| notional per trade | $500 |
| account leverage actually used | **5x** |
| avg stop distance | 2.00% of price |
| **loss if stopped** | **−$9.99 = 10% of the account** |
| gain if target hit | +$5.85 = 5.9% of the account |
| adverse move that liquidates one position | 19.5% |

**Held-out half, one $100 account per instrument** (the honest expectation):

| instrument | trades | liq | win% | PF | net $ | final $ | max DD | return |
|---|---|---|---|---|---|---|---|---|
| SOL | 27 | 0 | 66.7 | 2.13 | **+49.08** | 149.08 | 21.8% | +49.1% |
| ETH | 31 | 0 | 61.3 | 1.28 | +17.13 | 117.13 | 18.3% | +17.1% |
| BTC | 35 | 0 | 60.0 | 0.97 | −1.28 | 98.72 | 19.4% | −1.3% |
| XAU | 42 | 0 | 61.9 | 1.00 | +0.06 | 100.06 | 14.4% | +0.1% |

Full period (a ceiling, since the parameters were chosen in it) is +63.2% ETH,
+64.0% SOL, +40.5% BTC, −28.0% XAU.

Leverage is a pure multiplier on both sides -- it manufactures no edge:

| margin/trade | notional | acct leverage | loss per stop | net $ | max DD |
|---|---|---|---|---|---|
| $1 | $100 | 1x | $2.00 | +3.41 | 3.9% |
| **$5** | **$500** | **5x** | **$10.00** | **+17.13** | **18.3%** |
| $10 | $1,000 | 10x | $20.00 | +34.30 | 33.3% |
| $20 | $2,000 | 20x | $40.00 | +68.60 | 56.5% |

### Cross vs isolated: the $5 is not a $5 risk

A common and expensive misreading: in **cross** margin, $5 at 100x is the
*initial margin requirement*, not a loss cap. The whole balance backs the
position as it moves against you. The mode that caps a single trade's loss at
$5 is **isolated** -- and at 100x that cap arrives so early it destroys the
strategy.

`--margin-mode` models both. $5 x 100x = $500 notional, held-out half:

| instrument | mode | trades | **liquidated** | win% | net $ | max DD |
|---|---|---|---|---|---|---|
| ETH | cross | 31 | 0 | 61.3 | **+17.13** | 18.3% |
| ETH | **isolated** | 41 | **25 (61%)** | 39.0 | **−7.76** | 29.9% |
| SOL | cross | 27 | 0 | 66.7 | +49.08 | 21.8% |
| SOL | isolated | 31 | 14 (45%) | 54.8 | +56.66 | 11.1% |
| BTC | cross | 35 | 0 | 60.0 | −1.28 | 19.4% |
| BTC | isolated | 39 | 18 (46%) | 48.7 | −1.67 | 16.7% |

In isolated mode at 100x the liquidation price sits about
`1/leverage - maintenance` = **0.5%** from entry, while the strategy's stop is
at **2.00%**. Every losing trade is liquidated by noise long before its stop,
which is why ETH's win rate falls from 61% to 39% and +$17 becomes −$8.

| leverage | margin as % of notional | liquidation distance |
|---|---|---|
| 100x | 1.0% | **0.50%** |
| 50x | 2.0% | 1.50% |
| **40x** | 2.5% | **2.00%** (= the stop) |
| 25x | 4.0% | 3.50% |
| 10x | 10.0% | 9.50% |

So a 2% stop needs **under 40x** to be the thing that closes the trade.

### The 100x setting buys nothing

Position size is the **notional**, not the leverage. Holding notional at $500
and lowering the leverage setting changes only how close liquidation sits:

| leverage | margin required | liquidation dist | trades | liquidated | net $ |
|---|---|---|---|---|---|
| 100x | $5.00 | 0.50% | 41 | 25 | −7.76 |
| 40x | $12.50 | 2.00% | 32 | 3 | +18.11 |
| **25x** | **$20.00** | **3.50%** | **31** | **0** | **+17.13** |
| 10x | $50.00 | 9.50% | 31 | 0 | +17.13 |

25x and 10x produce **identical P&L** to each other, with no liquidations.
100x produces a worse result for the same position. High leverage does not
increase size or profit here; it only moves the liquidation price toward you.
Pick the lowest leverage that permits the notional you want.

### The thing that actually ends this account: cross margin

The table above is **one $100 account per instrument**. On *cross* margin, one
$100 account backs all of them at once, and three concurrent $500 positions is
$1,500 of notional:

| one $100 cross account, ETH+SOL+BTC | full period | held-out half |
|---|---|---|
| final equity | $267.69 (+167.7%) | $164.94 (+64.9%) |
| lowest equity | $77.60 | $66.04 |
| max combined drawdown | 31.2% | **43.1%** |
| peak account leverage | 19.3x | **20.5x** |
| liquidations in backtest | 0 | 0 |
| **simultaneous adverse move that would liquidate** | **4.7%** | **3.9%** |

Zero liquidations happened in this window, and that is not reassurance,
because these three are effectively one bet made three times: 15-minute
return correlations are **BTC/ETH 0.88, BTC/SOL 0.82, ETH/SOL 0.84**. In the
same 120 days, the number of starting bars from which all three subsequently
fell by the liquidating amount:

| window | all three fell ≥3.9% | ≥5% | ≥8% |
|---|---|---|---|
| 1 hour | 3 | 3 | 0 |
| 6 hours | 65 | 37 | 0 |
| 24 hours | 640 | 380 | 36 |

A 3.9% joint move is a routine crypto event that occurred repeatedly in the
backtest window. It did not coincide with peak exposure this time. Run this
configuration long enough and it will, and a cross-margin liquidation takes
the whole $100, not the one position.

If you want the leverage, the mitigations are structural, not parametric: one
position at a time, or isolated margin instead of cross, or sizing so total
notional across open positions stays inside what a 10% joint move can absorb.

### Scanning 30 OKX perps: breadth is diversification, not selection

`tools/universe_scan.py` runs the strategy across a whole universe and prints
every result, including losers and blown accounts. Top 30 USDT perps by real
USD turnover (base-currency volume x price -- OKX's `volCcy24h` alone ranks
memecoins absurdly high), 15m, 120 days:

| at $500 notional/position | at $50 notional/position |
|---|---|
| 25 / 29 positive (86%) | **26 / 29 positive (90%)** |
| median +$64.03 | median +$6.42 |
| **2 accounts to zero** (LAB, SOXL) | **0 accounts to zero** |

LAB took 9 liquidations with a worst trade of −$87.61 on a −17.61% move, and
SOXL 9 on −8.54%. So cross margin at $500 on a $100 account *does* blow up --
it simply did not happen on ETH/SOL/BTC, which is exactly the danger of
judging a strategy on three instruments.

**You cannot pick the winners.** The rank correlation between an
instrument's first-half and second-half profit factor is **−0.11**:

| selected on first half | second-half median PF |
|---|---|
| best third | 1.27 |
| worst third | **1.45** |
| whole universe | 1.39 |

Last period's laggards beat last period's leaders. Any procedure that chooses
instruments by past results is therefore worse than owning all of them.

### Shorts: tested with the final config, and they make it worse

The long-only choice was first made from a forward-return measurement at
`z > 2`, which is a weaker test than it should have been. Re-measured at the
depth the final config actually quotes (`z > 3`) across 12 instruments, the
asymmetry is larger, not smaller:

| average forward return, 15m | +6 bars | +12 | +24 |
|---|---|---|---|
| **long** after z < −3 | **+33.36** | **+25.39** | **+18.13** bps |
| **short** after z > +3 | −4.10 | −20.83 | **−63.45** bps |

Shorting an overbought extreme loses, and the loss compounds with holding
time: dips get bought while rallies keep running, so fading strength fights
the dominant direction. The strategy's trend-filter veto does not rescue it.

Run through the full 26-instrument portfolio at $50 per position:

| sides | trades | PF | net $ | max DD | t-stat |
|---|---|---|---|---|---|
| **long only** | 1,176 | **1.30** | +87.96 | **34.9%** | **+2.38** |
| long + short | 2,561 | 1.09 | +104.54 | 45.7% | +1.26 |
| short only | 1,418 | 0.95 | +15.27 | 69.0% | −0.63 |

Held-out half, which settles it:

| sides | trades | PF | net $ | max DD | return / drawdown |
|---|---|---|---|---|---|
| **long only** | 633 | **1.40** | **+45.50** | **23.4%** | **1.94** |
| long + short | 1,311 | 1.16 | +42.31 | 46.9% | 0.90 |
| short only | 699 | 0.99 | −0.53 | 49.3% | — |

Adding shorts doubles the trade count and the drawdown while returning
slightly *less* money out of sample. Risk-adjusted it is less than half as
good. Shorts stay off.

### The portfolio that follows from that

26 executable instruments (tick under 4 bps -- TRUMP at 5.0, NEAR at 4.2 and
BEAT at 11.25 bps cannot support a ~12 bps target), all traded small from one
$100 account. Peak concurrency is 16 positions, average 1.6:

| per position | peak notional | peak leverage | net | max DD | lowest equity | liq |
|---|---|---|---|---|---|---|
| $500 | $8,000 | 80x | +885.84 | 292% | **−$220** | 18 |
| $100 | $1,600 | 16x | +176.62 | 77.7% | $23.00 | 0 |
| **$50** | **$800** | **8x** | **+87.96** | **34.9%** | **$66.06** | **0** |
| $25 | $400 | 4x | +43.67 | 17.1% | $83.48 | 0 |

Held-out second half only:

| per position | net | return | max DD | lowest equity |
|---|---|---|---|---|
| $100 | +91.21 | +91.2% | 37.6% | $83.39 |
| **$50** | **+45.50** | **+45.5%** | **23.4%** | **$92.07** |
| $25 | +22.69 | +22.7% | 14.9% | $96.44 |

This is the best configuration in the repo: 26 instruments rather than three,
90% of them positive, no liquidations, and a held-out return that survives on
data no parameter touched. It is also still one 120-day window in one asset
class, 86-90% of a correlated universe being positive is consistent with a
favourable regime rather than a durable edge, and the 1H series over a full
year contains a single −94 month as a reminder of what the other kind of
regime does.

### Regime test over a full year: the 120-day result does not survive

Every figure above came from 120 days of 15m data, May to September 2026.
Extending the same config to **365 days of 15m** across six majors -- a window
in which BTC fell 32.8% -- reverses the conclusion:

| regime (BTC 20-day trailing) | trades | win% | PF | net $ | t |
|---|---|---|---|---|---|
| BULL (> +5%) | 142 | 54.9 | 0.92 | −2.90 | −0.42 |
| flat | 211 | 54.5 | **0.75** | −14.62 | −1.55 |
| BEAR (< −5%) | 129 | 53.5 | 1.14 | +4.53 | +0.60 |
| **all** | **504** | **54.4** | **0.90** | **−13.35** | −0.95 |

**It does not work in bull or bear. It loses in both, and loses most in
quiet markets.** Month by month, 6 of 13 are positive, and December 2025
alone (−16.86) exceeds the whole year's loss.

The one thing the regime split does establish is that **bear markets are not
the weakness** -- BEAR is the only regime above break-even (PF 1.14), because
a selloff produces the deep dips and sharp bounces mean reversion needs. The
killer is a *quiet* market (PF 0.75), where dips are too shallow to clear the
fee floor. That is the opposite of the intuition that a long-only dip-buyer
dies in downtrends.

The lagging trend filter does earn its keep, but not where expected:

| regime | filter ON | filter OFF | saved |
|---|---|---|---|
| BULL | −2.90 | +2.10 | **−5.00** |
| flat | −14.62 | −28.78 | **+14.16** |
| BEAR | +4.53 | +3.88 | +0.65 |

It rescues quiet markets by refusing trades, costs money in bull markets, and
is roughly neutral in bear.

**What this means for the earlier tables.** The 120-day window that produced
+45.5% held-out was the favourable stretch of a bad year: it contained June's
−20.6% selloff (the best month, +9.02) and the July-August recovery, and none
of December. On the full year the same parameters return −13.35. The
regime-dependence flagged as the main caveat throughout this README is not
hypothetical -- it is the result.

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
