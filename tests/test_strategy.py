import unittest
from dataclasses import replace

from helpers import make_bars, ramp
from zetaalgo.config import StrategyConfig
from zetaalgo.strategy import EmaVwapCrossoverStrategy


def drive(strategy, bars, stop_at=None):
    """Replay bars through the state machine, collecting entry plans per bar."""
    strategy.prepare(bars)
    plans = []
    for i in range(len(bars) if stop_at is None else stop_at):
        plans.append(strategy.entry_plan(i))
        strategy.on_bar_close(i)
    return plans


class TestRule1Crossover(unittest.TestCase):
    def test_a_rally_after_a_flat_stretch_produces_a_bullish_cross(self):
        strategy = EmaVwapCrossoverStrategy()
        drive(strategy, make_bars(ramp()))
        self.assertGreaterEqual(strategy.rejections.get("crosses_detected", 0), 1)

    def test_a_flat_market_never_crosses(self):
        strategy = EmaVwapCrossoverStrategy()
        drive(strategy, make_bars([100.0] * 40))
        self.assertEqual(strategy.rejections.get("crosses_detected", 0), 0)

    def test_a_falling_market_never_produces_a_long_setup(self):
        strategy = EmaVwapCrossoverStrategy()
        closes = [100.0] * 12 + [100.0 - 0.25 * (i + 1) for i in range(12)]
        plans = drive(strategy, make_bars(closes))
        self.assertTrue(all(plan is None for plan in plans))


class TestRule2ClearGap(unittest.TestCase):
    """The EMA must stay above the VWAP with a measurable gap."""

    def test_a_demanding_gap_threshold_blocks_the_setup(self):
        bars = make_bars(ramp())
        loose = EmaVwapCrossoverStrategy(StrategyConfig(min_gap_atr=0.0))
        strict = EmaVwapCrossoverStrategy(StrategyConfig(min_gap_atr=50.0))
        self.assertTrue(any(p for p in drive(loose, bars)))
        self.assertFalse(any(p for p in drive(strict, bars)))

    def test_percentage_gap_threshold_is_also_enforced(self):
        bars = make_bars(ramp())
        strategy = EmaVwapCrossoverStrategy(
            StrategyConfig(min_gap_atr=0.0, min_gap_pct=0.5)
        )
        self.assertFalse(any(p for p in drive(strategy, bars)))

    def test_setup_is_discarded_when_the_ema_falls_back_below_the_vwap(self):
        # Rally to trigger the cross, then collapse before any breakout.
        closes = [100.0] * 12 + [100.05, 100.1] + [99.0] * 10
        strategy = EmaVwapCrossoverStrategy(StrategyConfig(confirm_window=8))
        drive(strategy, make_bars(closes))
        self.assertGreaterEqual(
            strategy.rejections.get("ema_fell_back_below_vwap", 0), 1
        )
        self.assertIsNone(strategy.setup)


class TestRule3PriceAboveBothLines(unittest.TestCase):
    def test_confirmation_requires_the_close_above_both_lines(self):
        strategy = EmaVwapCrossoverStrategy(StrategyConfig(min_gap_atr=0.0))
        bars = make_bars(ramp())
        strategy.prepare(bars)
        for i in range(len(bars)):
            strategy.on_bar_close(i)
            setup = strategy.setup
            if setup is not None and setup.is_confirmed:
                index = setup.confirm_index
                self.assertGreater(bars[index].close, strategy.ema[index])
                self.assertGreater(bars[index].close, strategy.vwap[index])
                break
        else:
            self.fail("no setup was ever confirmed")

    def test_disabling_rule_three_can_only_admit_more_setups(self):
        from zetaalgo.data import generate_synthetic

        bars = generate_synthetic(days=40, seed=11)
        strict = EmaVwapCrossoverStrategy(StrategyConfig(require_close_above_both=True))
        loose = EmaVwapCrossoverStrategy(StrategyConfig(require_close_above_both=False))
        strict_plans = sum(1 for p in drive(strict, bars) if p)
        loose_plans = sum(1 for p in drive(loose, bars) if p)
        self.assertGreater(loose_plans, strict_plans)


class TestRule4BreakoutEntry(unittest.TestCase):
    def test_trigger_sits_one_tick_above_the_cross_candle_high(self):
        strategy = EmaVwapCrossoverStrategy(StrategyConfig(min_gap_atr=0.0))
        bars = make_bars(ramp())
        plans = drive(strategy, bars)
        plan = next(p for p in plans if p is not None)
        self.assertAlmostEqual(
            plan.trigger_price, bars[plan.cross_index].high + 0.01, places=6
        )

    def test_stop_sits_one_tick_below_the_cross_candle_low(self):
        strategy = EmaVwapCrossoverStrategy(StrategyConfig(min_gap_atr=0.0))
        bars = make_bars(ramp())
        plan = next(p for p in drive(strategy, bars) if p is not None)
        self.assertAlmostEqual(
            plan.stop_price, bars[plan.cross_index].low - 0.01, places=6
        )

    def test_entry_is_never_offered_on_the_confirmation_bar_itself(self):
        strategy = EmaVwapCrossoverStrategy(StrategyConfig(min_gap_atr=0.0))
        bars = make_bars(ramp())
        plans = drive(strategy, bars)
        for index, plan in enumerate(plans):
            if plan is not None:
                self.assertGreater(index, plan.confirm_index)

    def test_entry_window_of_one_means_only_the_next_candle(self):
        config = StrategyConfig(min_gap_atr=0.0, entry_window=1)
        strategy = EmaVwapCrossoverStrategy(config)
        bars = make_bars(ramp())
        plans = drive(strategy, bars)
        for index, plan in enumerate(plans):
            if plan is not None:
                self.assertEqual(index - plan.confirm_index, 1)

    def test_a_wider_entry_window_offers_the_trigger_for_more_bars(self):
        bars = make_bars(ramp(flat=12, rise=14))
        narrow = EmaVwapCrossoverStrategy(StrategyConfig(min_gap_atr=0.0, entry_window=1))
        wide = EmaVwapCrossoverStrategy(StrategyConfig(min_gap_atr=0.0, entry_window=5))
        narrow_count = sum(1 for p in drive(narrow, bars) if p)
        wide_count = sum(1 for p in drive(wide, bars) if p)
        self.assertGreater(wide_count, narrow_count)

    def test_a_plan_with_non_positive_risk_is_refused(self):
        strategy = EmaVwapCrossoverStrategy(StrategyConfig(min_gap_atr=0.0))
        bars = make_bars(ramp())
        strategy.prepare(bars)
        for i in range(len(bars)):
            strategy.on_bar_close(i)
        if strategy.setup is not None:
            # Force the stop above the trigger: the plan must be rejected.
            strategy.setup.cross_low = strategy.setup.cross_high + 5.0
            self.assertIsNone(strategy.entry_plan(len(bars)))


class TestSessionHandling(unittest.TestCase):
    def test_a_setup_does_not_survive_the_session_break(self):
        first = make_bars([100.0] * 12 + [100.05, 100.1], session="2024-01-02")
        second = make_bars(
            [100.15] * 4,
            session="2024-01-03",
            start=first[-1].ts.replace(hour=9, minute=30) if False else None,
        )
        # Give the second session later timestamps so ordering stays sane.
        from datetime import timedelta

        second = [replace(bar, ts=first[-1].ts + timedelta(days=1)) for bar in second]
        strategy = EmaVwapCrossoverStrategy(StrategyConfig(min_gap_atr=0.0))
        bars = first + second
        plans = drive(strategy, bars)
        for index, plan in enumerate(plans):
            if plan is not None:
                self.assertEqual(bars[index].session, plan.setup_session)


class TestNoLookahead(unittest.TestCase):
    """An entry decision must never depend on the bar it is executed on."""

    def test_entry_plan_is_unchanged_when_the_current_bar_is_altered(self):
        bars = make_bars(ramp(flat=12, rise=10))
        strategy = EmaVwapCrossoverStrategy(StrategyConfig(min_gap_atr=0.0))
        strategy.prepare(bars)
        target_index = None
        for i in range(len(bars)):
            plan = strategy.entry_plan(i)
            if plan is not None:
                target_index = i
                baseline = plan
                break
            strategy.on_bar_close(i)
        self.assertIsNotNone(target_index, "no entry plan was produced")

        # Rebuild with the execution bar replaced by a wildly different candle.
        mutated = list(bars)
        original = mutated[target_index]
        mutated[target_index] = replace(
            original,
            open=original.open * 0.5,
            high=original.high * 2.0,
            low=original.low * 0.25,
            close=original.close * 0.5,
            volume=original.volume * 9,
        )
        other = EmaVwapCrossoverStrategy(StrategyConfig(min_gap_atr=0.0))
        other.prepare(mutated)
        for i in range(target_index):
            other.on_bar_close(i)
        after = other.entry_plan(target_index)
        self.assertIsNotNone(after)
        self.assertAlmostEqual(baseline.trigger_price, after.trigger_price, places=9)
        self.assertAlmostEqual(baseline.stop_price, after.stop_price, places=9)

    def test_entry_plan_can_be_requested_one_bar_past_the_data(self):
        """Live trading needs the resting order before the next bar exists."""
        bars = make_bars(ramp())
        strategy = EmaVwapCrossoverStrategy(StrategyConfig(min_gap_atr=0.0))
        strategy.prepare(bars)
        for i in range(len(bars)):
            strategy.on_bar_close(i)
        strategy.entry_plan(len(bars))  # must not raise
        # Further into the future there is nothing to plan against, and the
        # guard returns None rather than reading past the series.
        self.assertIsNone(strategy.entry_plan(len(bars) + 5))


if __name__ == "__main__":
    unittest.main()
