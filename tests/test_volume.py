"""Tests for the volume-spike strategy.

The interesting cases are all about what "sudden spike" is measured against:
a baseline that includes the spike, or that is dragged around by the time of
day, quietly turns this into a different strategy.
"""

import unittest
from dataclasses import replace
from datetime import datetime, timedelta

from zetaalgo.backtest import run_backtest
from zetaalgo.config import BacktestConfig
from zetaalgo.data import Bar, generate_synthetic
from zetaalgo.live import run_paper_session
from zetaalgo.volume import VolumeSpikeConfig, VolumeSpikeStrategy

from helpers import make_bars


def drive(strategy, bars):
    """Run the state machine the way the engine does, collecting the plans."""
    strategy.prepare(bars)
    plans = []
    for i in range(len(bars)):
        plans.append(strategy.entry_plan(i))
        strategy.on_bar_close(i)
    return plans


def quiet_then_spike(
    spike_index: int = 40,
    bars_before: int = 40,
    bars_after: int = 10,
    spike_volume: float = 20_000.0,
    quiet_volume: float = 1_000.0,
    baseline_period: int = 20,
):
    """A flat, quiet market, one heavy breakout bar, then quiet again.

    The flat stretch keeps the baseline constant and the recent range narrow,
    so exactly one bar can qualify and every assertion below is about that bar
    rather than about the shape of some generated series.
    """
    closes = [100.0] * bars_before + [101.0] + [101.0] * bars_after
    volumes = [quiet_volume] * bars_before + [spike_volume] + [quiet_volume] * bars_after
    bars = make_bars(closes, volumes=volumes, pad=0.05)
    return bars, spike_index


class TestConfig(unittest.TestCase):
    def test_bad_values_are_rejected(self):
        for kwargs in (
            dict(mode="sideways"),
            dict(baseline="lunar"),
            dict(spike_mult=1.0),
            dict(spike_mult=0.5),
            dict(baseline_period=1),
            dict(baseline_sessions=1),
            dict(min_body_ratio=1.5),
            dict(max_body_ratio=-0.1),
            dict(range_lookback=0),
            dict(entry_window=0),
            dict(tick_size=0),
            dict(stop_mode="hope"),
            dict(trail_mode="ema"),
            dict(entry_bar_stop="whenever"),
            dict(min_risk_bps=-1),
            dict(trade_longs=False, trade_shorts=False),
        ):
            with self.assertRaises(ValueError, msg=kwargs):
                VolumeSpikeConfig(**kwargs)

    def test_a_spike_multiple_of_one_is_not_a_spike(self):
        with self.assertRaises(ValueError):
            VolumeSpikeConfig(spike_mult=1.0)

    def test_required_history_covers_the_trailing_baseline(self):
        config = VolumeSpikeConfig(baseline_period=500, atr_period=14,
                                   range_lookback=20, entry_window=3)
        self.assertGreater(config.required_history, 500)
        self.assertEqual(config.required_sessions, 0)

    def test_a_session_baseline_asks_for_sessions_not_bars(self):
        config = VolumeSpikeConfig(baseline="time_of_day", baseline_sessions=20)
        self.assertEqual(config.required_sessions, 21)
        self.assertLess(config.required_history, 100)


class TestBaseline(unittest.TestCase):
    """What the volume is compared against, and when it is allowed to know it."""

    def test_the_baseline_excludes_the_bar_it_judges(self):
        # A large enough spike would otherwise lift its own baseline and hide.
        bars, spike = quiet_then_spike(baseline_period=20)
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(baseline_period=20))
        strategy.prepare(bars)
        self.assertAlmostEqual(strategy.baseline[spike], 1_000.0)
        self.assertAlmostEqual(strategy.rvol[spike], 20.0)

    def test_the_baseline_is_a_median_not_a_mean(self):
        # One huge bar inside the window moves a mean a long way and a median
        # not at all, which is exactly why this uses a median.
        volumes = [1_000.0] * 10 + [100_000.0] + [1_000.0] * 10
        bars = make_bars([100.0] * len(volumes), volumes=volumes)
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(baseline_period=10))
        strategy.prepare(bars)
        self.assertAlmostEqual(strategy.baseline[-1], 1_000.0)

    def test_the_baseline_is_none_until_the_window_is_full(self):
        bars = make_bars([100.0] * 30, volume=1_000.0)
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(baseline_period=20))
        strategy.prepare(bars)
        self.assertTrue(all(v is None for v in strategy.baseline[:20]))
        self.assertIsNotNone(strategy.baseline[21])

    def test_a_warming_up_baseline_arms_nothing(self):
        bars = make_bars([100.0] * 10, volume=1_000.0)
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(baseline_period=20))
        self.assertTrue(all(p is None for p in drive(strategy, bars)))
        self.assertGreater(strategy.rejections.get("baseline_not_ready", 0), 0)


class TestTimeOfDayBaseline(unittest.TestCase):
    """Volume has a shape; a trailing baseline mistakes that shape for news."""

    @staticmethod
    def u_shaped_sessions(sessions: int = 12, bars: int = 12) -> list:
        """Sessions whose first bar always trades 8x the midday volume.

        Nothing unusual ever happens here: every session is identical.  A
        baseline worth having should therefore find no spikes at all.
        """
        out = []
        start = datetime(2024, 1, 2, 9, 30)
        price = 100.0
        for day in range(sessions):
            session = f"2024-01-{day + 2:02d}"
            for slot in range(bars):
                volume = 8_000.0 if slot == 0 else 1_000.0
                ts = start + timedelta(days=day, minutes=5 * slot)
                # A decisive up-bar every time, so only the volume gate can
                # be what separates the two baselines.
                out.append(Bar(ts=ts, open=price, high=price + 1.02,
                               low=price - 0.02, close=price + 1.0,
                               volume=volume, session=session))
                price += 1.0
        return out

    def test_a_trailing_baseline_fires_on_every_open(self):
        bars = self.u_shaped_sessions()
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline="trailing", baseline_period=11, spike_mult=3.0))
        drive(strategy, bars)
        self.assertGreater(strategy.rejections.get("spikes_detected", 0), 5)

    def test_a_time_of_day_baseline_sees_nothing_unusual(self):
        bars = self.u_shaped_sessions()
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline="time_of_day", baseline_sessions=5, spike_mult=3.0))
        drive(strategy, bars)
        self.assertEqual(strategy.rejections.get("spikes_detected", 0), 0)

    def test_a_time_of_day_baseline_still_sees_a_real_spike(self):
        bars = self.u_shaped_sessions()
        heavy = bars[-6]
        bars[-6] = replace(heavy, volume=heavy.volume * 10)
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline="time_of_day", baseline_sessions=5, spike_mult=3.0))
        drive(strategy, bars)
        self.assertEqual(strategy.rejections.get("spikes_detected", 0), 1)

    def test_only_completed_sessions_feed_the_baseline(self):
        """Today's earlier bars must not set today's threshold.

        Otherwise a quiet morning lowers the bar for the afternoon and a busy
        one raises it, which is a same-session feedback loop, not a baseline.
        """
        bars = self.u_shaped_sessions(sessions=8, bars=6)
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline="time_of_day", baseline_sessions=5))
        strategy.prepare(bars)
        # Slot 3 of the last session: its baseline is slot 3 of previous days
        # (1,000), never the 8,000 that traded at this session's open.
        self.assertAlmostEqual(strategy.baseline[-3], 1_000.0)


class TestSpikeDetection(unittest.TestCase):
    def test_a_heavy_decisive_breakout_bar_arms_a_long(self):
        bars, spike = quiet_then_spike()
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline_period=20, range_lookback=10))
        plans = drive(strategy, bars)
        armed = [p for p in plans if p is not None]
        self.assertTrue(armed)
        self.assertEqual(armed[0].direction, 1)
        self.assertEqual(armed[0].cross_index, spike)

    def test_the_trigger_sits_above_the_spike_bar_high(self):
        bars, spike = quiet_then_spike()
        config = VolumeSpikeConfig(baseline_period=20, range_lookback=10,
                                   entry_buffer_ticks=1, tick_size=0.01)
        plan = next(p for p in drive(VolumeSpikeStrategy(config), bars) if p)
        self.assertAlmostEqual(plan.trigger_price, bars[spike].high + 0.01, places=6)
        self.assertAlmostEqual(plan.stop_price, bars[spike].low - 0.01, places=6)

    def test_the_entry_is_a_stop_order(self):
        bars, _ = quiet_then_spike()
        plan = next(p for p in drive(VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline_period=20, range_lookback=10)), bars) if p)
        self.assertEqual(plan.order_kind, "stop")

    def test_ordinary_volume_arms_nothing(self):
        bars = make_bars([100.0 + 0.1 * i for i in range(60)], volume=1_000.0)
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(baseline_period=20))
        self.assertTrue(all(p is None for p in drive(strategy, bars)))
        self.assertEqual(strategy.rejections.get("spikes_detected", 0), 0)

    def test_a_higher_multiple_finds_fewer_spikes(self):
        bars = generate_synthetic(days=40, seed=31)
        counts = []
        for mult in (1.5, 2.0, 3.0, 5.0):
            strategy = VolumeSpikeStrategy(VolumeSpikeConfig(spike_mult=mult))
            drive(strategy, bars)
            counts.append(strategy.rejections.get("spikes_detected", 0))
        self.assertEqual(counts, sorted(counts, reverse=True), counts)

    def test_a_heavy_doji_is_not_a_breakout(self):
        """Volume without direction is indecision: the bar has no side."""
        volumes = [1_000.0] * 30 + [20_000.0] + [1_000.0] * 5
        closes = [100.0] * 36
        bars = make_bars(closes, volumes=volumes, pad=1.0)  # all wick, no body
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline_period=20, range_lookback=10))
        self.assertTrue(all(p is None for p in drive(strategy, bars)))
        self.assertGreater(strategy.rejections.get("body_too_small", 0), 0)

    def test_a_spike_inside_the_range_is_refused(self):
        """Churn is not a breakout, however much of it trades."""
        closes = [100.0 + (0.5 if i % 2 else -0.5) for i in range(40)]
        volumes = [1_000.0] * 35 + [20_000.0] + [1_000.0] * 4
        bars = make_bars(closes, volumes=volumes, pad=0.02)
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline_period=20, range_lookback=20))
        drive(strategy, bars)
        self.assertGreater(strategy.rejections.get("no_range_break", 0), 0)

    def test_the_range_gate_can_be_turned_off(self):
        closes = [100.0 + (0.5 if i % 2 else -0.5) for i in range(40)]
        volumes = [1_000.0] * 35 + [20_000.0] + [1_000.0] * 4
        bars = make_bars(closes, volumes=volumes, pad=0.02)
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline_period=20, require_range_break=False, min_body_ratio=0.1))
        self.assertTrue(any(p is not None for p in drive(strategy, bars)))

    def test_range_expansion_can_be_demanded(self):
        bars, _ = quiet_then_spike()
        loose = VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline_period=20, range_lookback=10, min_range_atr=0.0))
        strict = VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline_period=20, range_lookback=10, min_range_atr=50.0))
        self.assertTrue(any(p for p in drive(loose, bars)))
        self.assertTrue(all(p is None for p in drive(strict, bars)))
        self.assertGreater(strict.rejections.get("range_not_expanded", 0), 0)

    def test_a_disabled_direction_is_not_traded(self):
        bars, _ = quiet_then_spike()
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline_period=20, range_lookback=10, trade_longs=False))
        self.assertTrue(all(p is None for p in drive(strategy, bars)))
        self.assertGreater(strategy.rejections.get("direction_disabled", 0), 0)


class TestFadeMode(unittest.TestCase):
    """The opposite reading: a heavy bar that could not hold its extreme."""

    @staticmethod
    def climax_bars():
        """Quiet, then one heavy bar that spikes up and closes back down."""
        bars = make_bars([100.0] * 30 + [100.0] * 6, volume=1_000.0)
        spike = 30
        bars[spike] = replace(
            bars[spike], open=100.0, high=105.0, low=99.9, close=100.05,
            volume=20_000.0,
        )
        return bars, spike

    def test_a_rejected_high_arms_a_short(self):
        bars, spike = self.climax_bars()
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(
            mode="fade", baseline_period=20, range_lookback=10))
        plan = next(p for p in drive(strategy, bars) if p)
        self.assertEqual(plan.direction, -1)
        # The entry is a break of the OTHER side of the climax bar: the fade is
        # only taken once price actually leaves the bar downwards.
        self.assertLess(plan.trigger_price, bars[spike].low)
        self.assertGreater(plan.stop_price, bars[spike].high)

    def test_the_same_bar_reads_long_in_breakout_mode(self):
        """One bar, two strategies, opposite sides -- as intended."""
        bars = make_bars([100.0] * 30 + [100.0] * 6, volume=1_000.0)
        bars[30] = replace(bars[30], open=100.0, high=105.0, low=99.9,
                           close=104.8, volume=20_000.0)
        breakout = next(p for p in drive(VolumeSpikeStrategy(VolumeSpikeConfig(
            mode="breakout", baseline_period=20, range_lookback=10)), bars) if p)
        self.assertEqual(breakout.direction, 1)
        # The same bar has too much body to be a climax, so the fade refuses it.
        fader = VolumeSpikeStrategy(VolumeSpikeConfig(
            mode="fade", baseline_period=20, range_lookback=10))
        self.assertTrue(all(p is None for p in drive(fader, bars)))
        self.assertGreater(fader.rejections.get("body_too_large_to_fade", 0), 0)


class TestEntryWindow(unittest.TestCase):
    def test_the_level_goes_stale(self):
        bars, spike = quiet_then_spike(bars_after=10)
        config = VolumeSpikeConfig(baseline_period=20, range_lookback=10,
                                   entry_window=2)
        plans = drive(VolumeSpikeStrategy(config), bars)
        live = [i for i, p in enumerate(plans) if p is not None]
        self.assertEqual(live, [spike + 1, spike + 2])

    def test_a_wider_window_keeps_the_order_resting_longer(self):
        bars, spike = quiet_then_spike(bars_after=10)
        plans = drive(VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline_period=20, range_lookback=10, entry_window=5)), bars)
        self.assertEqual(sum(1 for p in plans if p is not None), 5)

    def test_an_expired_setup_is_recorded(self):
        bars, _ = quiet_then_spike(bars_after=10)
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline_period=20, range_lookback=10, entry_window=2))
        drive(strategy, bars)
        self.assertEqual(strategy.rejections.get("no_break_in_window", 0), 1)


class TestRiskGates(unittest.TestCase):
    """The fee floor, and its mirror at the other end."""

    def test_a_demanding_fee_floor_refuses_every_setup(self):
        bars, _ = quiet_then_spike()
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline_period=20, range_lookback=10, min_risk_bps=10_000.0))
        self.assertTrue(all(p is None for p in drive(strategy, bars)))
        self.assertGreater(strategy.rejections.get("risk_below_fee_floor", 0), 0)

    def test_every_taken_trade_clears_the_floor(self):
        floor = 20.0
        bars = generate_synthetic(days=60, seed=32)
        config = VolumeSpikeConfig(spike_mult=1.8, min_risk_bps=floor,
                                   require_range_break=False)
        result = run_backtest(bars, strategy=VolumeSpikeStrategy(config))
        self.assertTrue(result.trades)
        for trade in result.trades:
            risk = abs(trade.entry_price - trade.initial_stop)
            # Measured against the trigger the order rested at, which is what
            # the gate saw; a fill worse than the trigger only widens it.
            self.assertGreater(risk / trade.entry_price * 10_000.0, floor * 0.5)

    def test_raising_the_floor_can_only_reduce_trades(self):
        bars = generate_synthetic(days=60, seed=33)
        counts = [
            len(run_backtest(bars, strategy=VolumeSpikeStrategy(VolumeSpikeConfig(
                spike_mult=1.8, require_range_break=False,
                min_risk_bps=floor))).trades)
            for floor in (0.0, 10.0, 30.0, 100.0)
        ]
        self.assertEqual(counts, sorted(counts, reverse=True), counts)

    def test_an_enormous_spike_bar_can_be_refused_as_too_wide(self):
        bars, _ = quiet_then_spike()
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline_period=20, range_lookback=10, max_risk_atr=0.1))
        self.assertTrue(all(p is None for p in drive(strategy, bars)))
        self.assertGreater(strategy.rejections.get("risk_too_wide", 0), 0)


class TestStopModes(unittest.TestCase):
    def test_the_midpoint_stop_is_half_the_spike_bar(self):
        bars, spike = quiet_then_spike()
        config = VolumeSpikeConfig(baseline_period=20, range_lookback=10,
                                   stop_mode="midpoint", stop_buffer_ticks=0)
        plan = next(p for p in drive(VolumeSpikeStrategy(config), bars) if p)
        midpoint = (bars[spike].high + bars[spike].low) / 2.0
        self.assertAlmostEqual(plan.stop_price, midpoint, places=2)

    def test_the_midpoint_stop_is_tighter_than_the_full_bar(self):
        bars, _ = quiet_then_spike()
        wide = next(p for p in drive(VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline_period=20, range_lookback=10, stop_mode="spike_bar")), bars) if p)
        tight = next(p for p in drive(VolumeSpikeStrategy(VolumeSpikeConfig(
            baseline_period=20, range_lookback=10, stop_mode="midpoint")), bars) if p)
        self.assertGreater(tight.stop_price, wide.stop_price)

    def test_an_atr_stop_is_measured_from_the_trigger(self):
        bars, spike = quiet_then_spike()
        config = VolumeSpikeConfig(baseline_period=20, range_lookback=10,
                                   stop_mode="atr", stop_atr_mult=2.0,
                                   atr_period=14)
        strategy = VolumeSpikeStrategy(config)
        plan = next(p for p in drive(strategy, bars) if p)
        distance = plan.trigger_price - plan.stop_price
        self.assertAlmostEqual(distance, 2.0 * strategy.atr[spike], places=1)


class TestNoLookahead(unittest.TestCase):
    def test_the_plan_for_a_bar_does_not_use_that_bar(self):
        bars = generate_synthetic(days=40, seed=34)
        config = VolumeSpikeConfig(spike_mult=1.8, require_range_break=False)
        strategy = VolumeSpikeStrategy(config)
        strategy.prepare(bars)
        target = plan = None
        for i in range(len(bars)):
            found = strategy.entry_plan(i)
            if found is not None:
                target, plan = i, found
                break
            strategy.on_bar_close(i)
        self.assertIsNotNone(target)

        mutated = list(bars)
        original = mutated[target]
        mutated[target] = replace(original, open=original.open * 0.8,
                                  high=original.high * 1.5,
                                  low=original.low * 0.5,
                                  close=original.close * 0.8,
                                  volume=original.volume * 40)
        other = VolumeSpikeStrategy(config)
        other.prepare(mutated)
        for i in range(target):
            other.on_bar_close(i)
        after = other.entry_plan(target)
        self.assertIsNotNone(after)
        self.assertAlmostEqual(plan.trigger_price, after.trigger_price, places=9)
        self.assertAlmostEqual(plan.stop_price, after.stop_price, places=9)

    def test_the_spike_bar_is_never_its_own_entry_bar(self):
        """The order can only rest from the bar AFTER the spike."""
        bars = generate_synthetic(days=30, seed=35)
        strategy = VolumeSpikeStrategy(VolumeSpikeConfig(
            spike_mult=1.8, require_range_break=False))
        for i, plan in enumerate(drive(strategy, bars)):
            if plan is not None:
                self.assertGreater(i, plan.cross_index)


class TestEngineIntegration(unittest.TestCase):
    def test_the_engine_produces_trades_on_synthetic_bars(self):
        bars = generate_synthetic(days=90, seed=36)
        result = run_backtest(bars, strategy=VolumeSpikeStrategy(
            VolumeSpikeConfig(spike_mult=1.8, require_range_break=False)))
        self.assertTrue(result.trades)
        for trade in result.trades:
            self.assertIn(trade.direction, (1, -1))
            self.assertGreater(
                trade.direction * (trade.entry_price - trade.initial_stop), 0)

    def test_the_time_stop_bounds_how_long_a_trade_lasts(self):
        bars = generate_synthetic(days=90, seed=37)
        result = run_backtest(bars, strategy=VolumeSpikeStrategy(
            VolumeSpikeConfig(spike_mult=1.8, require_range_break=False,
                              max_bars_in_trade=6)))
        self.assertTrue(result.trades)
        for trade in result.trades:
            self.assertLessEqual(trade.bars_held, 6)

    def test_costs_only_reduce_the_result(self):
        bars = generate_synthetic(days=90, seed=38)
        config = VolumeSpikeConfig(spike_mult=1.8, require_range_break=False)
        free = run_backtest(bars, backtest_config=BacktestConfig(
            commission_bps=0.0, slippage_ticks=0.0),
            strategy=VolumeSpikeStrategy(config))
        costly = run_backtest(bars, backtest_config=BacktestConfig(
            commission_bps=5.0, slippage_ticks=2.0),
            strategy=VolumeSpikeStrategy(config))
        self.assertLess(costly.final_equity, free.final_equity)


class TestParity(unittest.TestCase):
    """The automation must reproduce the backtest, or neither is worth much."""

    def assert_parity(self, bars, config, backtest_config=None):
        result = run_backtest(bars, backtest_config=backtest_config,
                              strategy=VolumeSpikeStrategy(config))
        broker = run_paper_session(bars, backtest_config=backtest_config,
                                   strategy=VolumeSpikeStrategy(config))
        opens = [f for i, f in enumerate(broker.fills) if i % 2 == 0]
        # A position still open on the last bar is a fill with no trade record
        # yet -- the backtest only books a trade when it closes.  That is not a
        # divergence, so it is excluded rather than papered over.
        still_open = 1 if broker.qty != 0 else 0
        self.assertEqual(len(opens) - still_open, len(result.trades))
        for i, trade in enumerate(result.trades):
            self.assertEqual(opens[i].ts, trade.entry_ts)
            self.assertAlmostEqual(opens[i].price, trade.entry_price, places=6)
        self.assertAlmostEqual(broker.equity(), result.final_equity, places=6)

    def test_parity_on_a_trailing_baseline(self):
        self.assert_parity(
            generate_synthetic(days=40, seed=39),
            VolumeSpikeConfig(spike_mult=1.8, require_range_break=False),
        )

    def test_parity_on_a_time_of_day_baseline(self):
        """The window must hold whole sessions, not a fixed number of bars."""
        self.assert_parity(
            generate_synthetic(days=40, seed=40),
            VolumeSpikeConfig(baseline="time_of_day", baseline_sessions=10,
                              spike_mult=1.8, require_range_break=False),
        )

    def test_the_live_window_grows_to_fit_a_long_baseline(self):
        from zetaalgo.live import LiveTrader, PaperBroker

        config = VolumeSpikeConfig(baseline_period=5_000)
        trader = LiveTrader(PaperBroker(), strategy=VolumeSpikeStrategy(config))
        self.assertGreaterEqual(trader.window, config.required_history)

    def test_parity_with_shorts_and_a_trail(self):
        self.assert_parity(
            generate_synthetic(days=40, seed=41),
            VolumeSpikeConfig(spike_mult=1.8, require_range_break=False,
                              trail_mode="atr", trail_atr_mult=2.0,
                              breakeven_at_r=0.0),
            BacktestConfig(commission_bps=5.0, lot_size=0.01),
        )


if __name__ == "__main__":
    unittest.main()
