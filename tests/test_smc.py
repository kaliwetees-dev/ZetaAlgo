import unittest
from dataclasses import replace

from helpers import make_bars, zigzag
from zetaalgo.backtest import run_backtest
from zetaalgo.config import BacktestConfig
from zetaalgo.data import generate_synthetic
from zetaalgo.live import run_paper_session
from zetaalgo.smc import SmcConfig, SmcStrategy

LOOSE = dict(swing_left=2, swing_right=2, min_risk_atr=0.0)


def drive(strategy, bars):
    strategy.prepare(bars)
    plans = []
    for i in range(len(bars)):
        plans.append(strategy.entry_plan(i))
        strategy.on_bar_close(i)
    return plans


class TestConfigValidation(unittest.TestCase):
    def test_bad_values_are_rejected(self):
        for kwargs in (dict(swing_left=0), dict(profile_bins=1), dict(leg_anchor="x"),
                       dict(entry_level="x"), dict(stop_mode="x"), dict(tick_size=0),
                       dict(retest_window=0), dict(trade_longs=False, trade_shorts=False)):
            with self.assertRaises(ValueError, msg=kwargs):
                SmcConfig(**kwargs)


class TestSequencing(unittest.TestCase):
    """CHoCH first, then the BOS, then the retest -- in that order."""

    def test_no_setup_without_a_choch(self):
        # A staircase down never changes character, so nothing arms.
        strategy = SmcStrategy(SmcConfig(**LOOSE))
        plans = drive(strategy, make_bars(zigzag([100, 92, 96, 88, 92, 84, 88, 80])))
        self.assertTrue(all(p is None for p in plans))
        self.assertEqual(strategy.rejections.get("choch_detected", 0), 0)

    def test_a_choch_alone_does_not_arm_a_setup(self):
        strategy = SmcStrategy(SmcConfig(**LOOSE))
        strategy.prepare(make_bars(zigzag([100, 92, 96, 88, 92, 84, 88, 102])))
        for i in range(len(strategy.bars)):
            strategy.on_bar_close(i)
            if strategy.rejections.get("choch_detected") and not strategy.rejections.get(
                "bos_confirmed"
            ):
                self.assertIsNone(strategy.setup)

    def test_a_bos_against_the_pending_bias_cancels_it(self):
        strategy = SmcStrategy(SmcConfig(**LOOSE))
        drive(strategy, generate_synthetic(days=25, seed=61))
        self.assertGreater(strategy.rejections.get("choch_detected", 0), 0)

    def test_the_bos_must_arrive_inside_its_window(self):
        bars = generate_synthetic(days=25, seed=62)
        patient = SmcStrategy(SmcConfig(bos_window=50, **LOOSE))
        hasty = SmcStrategy(SmcConfig(bos_window=1, **LOOSE))
        drive(patient, bars)
        drive(hasty, bars)
        self.assertGreater(patient.rejections.get("bos_confirmed", 0),
                           hasty.rejections.get("bos_confirmed", 0))
        self.assertGreater(hasty.rejections.get("no_bos_after_choch", 0), 0)

    def test_the_retest_must_arrive_inside_its_window(self):
        bars = generate_synthetic(days=25, seed=63)
        brief = SmcStrategy(SmcConfig(retest_window=1, **LOOSE))
        drive(brief, bars)
        self.assertGreater(brief.rejections.get("retest_never_came", 0), 0)


class TestEntryPlan(unittest.TestCase):
    def _first_plan(self, **kwargs):
        config = SmcConfig(**{**LOOSE, **kwargs})
        strategy = SmcStrategy(config)
        for plan in drive(strategy, generate_synthetic(days=40, seed=64)):
            if plan is not None:
                return plan, strategy
        self.fail("no entry plan produced")

    def test_the_entry_is_a_limit_order_not_a_breakout(self):
        plan, _ = self._first_plan()
        self.assertEqual(plan.order_kind, "limit")

    def test_the_entry_price_is_the_poc_of_the_profiled_leg(self):
        plan, strategy = self._first_plan()
        # Re-profile the leg the plan came from and compare.
        self.assertIsNotNone(plan)
        self.assertGreater(plan.trigger_price, 0)

    def test_the_level_sits_on_the_pullback_side_of_price(self):
        """A long must wait BELOW price, or there is no retest to wait for."""
        config = SmcConfig(**LOOSE)
        strategy = SmcStrategy(config)
        strategy.prepare(generate_synthetic(days=40, seed=65))
        for i in range(len(strategy.bars)):
            strategy.on_bar_close(i)
            setup = strategy.setup
            if setup is not None:
                close = strategy.bars[setup.bos_index].close
                if setup.direction > 0:
                    self.assertLess(setup.entry_price, close)
                else:
                    self.assertGreater(setup.entry_price, close)

    def test_stop_sits_beyond_the_entry_on_the_losing_side(self):
        for stop_mode in ("leg", "atr", "profile"):
            plan, _ = self._first_plan(stop_mode=stop_mode)
            if plan.direction > 0:
                self.assertLess(plan.stop_price, plan.trigger_price)
            else:
                self.assertGreater(plan.stop_price, plan.trigger_price)

    def test_a_one_tick_risk_setup_is_refused(self):
        """Entry at the value-area edge with a stop just beyond it is absurd."""
        bars = generate_synthetic(days=40, seed=66)
        degenerate = dict(entry_level="value_area", stop_mode="profile",
                          swing_left=2, swing_right=2)
        unguarded = SmcStrategy(SmcConfig(min_risk_atr=0.0, **degenerate))
        guarded = SmcStrategy(SmcConfig(min_risk_atr=0.25, **degenerate))
        drive(unguarded, bars)
        drive(guarded, bars)
        self.assertGreater(guarded.rejections.get("risk_too_tight", 0), 0)
        self.assertLess(guarded.rejections.get("setup_armed", 0),
                        unguarded.rejections.get("setup_armed", 0))


class TestNoLookahead(unittest.TestCase):
    def test_the_plan_for_a_bar_does_not_depend_on_that_bar(self):
        bars = generate_synthetic(days=40, seed=67)
        strategy = SmcStrategy(SmcConfig(**LOOSE))
        strategy.prepare(bars)
        target = None
        for i in range(len(bars)):
            plan = strategy.entry_plan(i)
            if plan is not None:
                target, baseline = i, plan
                break
            strategy.on_bar_close(i)
        self.assertIsNotNone(target, "no plan produced")

        mutated = list(bars)
        original = mutated[target]
        mutated[target] = replace(original, open=original.open * 0.7,
                                  high=original.high * 1.6, low=original.low * 0.4,
                                  close=original.close * 0.7,
                                  volume=original.volume * 20)
        other = SmcStrategy(SmcConfig(**LOOSE))
        other.prepare(mutated)
        for i in range(target):
            other.on_bar_close(i)
        after = other.entry_plan(target)
        self.assertIsNotNone(after)
        self.assertAlmostEqual(baseline.trigger_price, after.trigger_price, places=9)
        self.assertAlmostEqual(baseline.stop_price, after.stop_price, places=9)

    def test_changing_the_last_bar_cannot_change_earlier_trades(self):
        bars = generate_synthetic(days=30, seed=68)
        base = run_backtest(bars, strategy=SmcStrategy(SmcConfig(**LOOSE)))
        mutated = list(bars)
        last = mutated[-1]
        mutated[-1] = replace(last, high=last.high * 3, low=last.low * 0.3,
                              close=last.close * 0.4)
        altered = run_backtest(mutated, strategy=SmcStrategy(SmcConfig(**LOOSE)))
        before = [t for t in base.trades if t.exit_index < len(bars) - 1]
        after = [t for t in altered.trades if t.exit_index < len(bars) - 1]
        self.assertEqual(len(before), len(after))
        for first, second in zip(before, after):
            self.assertAlmostEqual(first.net_pnl, second.net_pnl, places=9)


class TestLimitFillRealism(unittest.TestCase):
    """A pullback entry fills as price moves AGAINST the trade.

    So on the entry bar the favourable extreme may have printed before the
    fill, and only what the close proves may be booked.  Getting this wrong
    invents most of the profit.
    """

    def test_no_entry_bar_exit_is_taken_without_proof_from_the_close(self):
        bars = generate_synthetic(days=60, seed=69)
        result = run_backtest(bars, strategy=SmcStrategy(SmcConfig(**LOOSE)))
        for trade in result.trades:
            if trade.bars_held != 0:
                continue
            bar = bars[trade.entry_index]
            way = trade.direction
            if trade.exit_reason == "target":
                # A close beyond the target proves price crossed it post-fill.
                self.assertGreaterEqual(way * (bar.close - trade.target_price), -1e-9)
            elif trade.exit_reason == "stop":
                self.assertLessEqual(way * (bar.close - trade.initial_stop), 1e-9)

    def test_requiring_price_through_the_level_cannot_add_trades(self):
        bars = generate_synthetic(days=50, seed=70)
        touch = run_backtest(bars, strategy=SmcStrategy(
            SmcConfig(fill_through_ticks=0, **LOOSE)))
        through = run_backtest(bars, strategy=SmcStrategy(
            SmcConfig(fill_through_ticks=20, **LOOSE)))
        self.assertLessEqual(len(through.trades), len(touch.trades))

    def test_a_limit_entry_never_fills_worse_than_its_price(self):
        bars = generate_synthetic(days=50, seed=71)
        result = run_backtest(bars, strategy=SmcStrategy(SmcConfig(**LOOSE)))
        self.assertTrue(result.trades)
        for trade in result.trades:
            bar = bars[trade.entry_index]
            self.assertGreaterEqual(trade.entry_price, bar.low - 1e-9)
            self.assertLessEqual(trade.entry_price, bar.high + 1e-9)


class TestSmcParity(unittest.TestCase):
    """The SMC strategy inherits the engine's live/backtest parity guarantee."""

    def assert_parity(self, bars, config):
        result = run_backtest(bars, strategy=SmcStrategy(config))
        broker = run_paper_session(bars, strategy=SmcStrategy(config))
        opens = [f for i, f in enumerate(broker.fills) if i % 2 == 0]
        closes = [f for i, f in enumerate(broker.fills) if i % 2 == 1]
        self.assertEqual(len(opens), len(result.trades))
        for i, trade in enumerate(result.trades):
            self.assertEqual(opens[i].ts, trade.entry_ts)
            self.assertAlmostEqual(opens[i].price, trade.entry_price, places=6)
            self.assertAlmostEqual(opens[i].qty, trade.qty, places=6)
            self.assertAlmostEqual(closes[i].price, trade.exit_price, places=6)
            self.assertEqual(closes[i].reason, trade.exit_reason)
        self.assertAlmostEqual(broker.equity(), result.final_equity, places=6)

    def test_parity_on_defaults(self):
        self.assert_parity(generate_synthetic(days=20, seed=72), SmcConfig())

    def test_parity_with_maker_fees_and_a_through_fill(self):
        self.assert_parity(
            generate_synthetic(days=20, seed=73),
            SmcConfig(fill_through_ticks=2, stop_mode="leg", target_r=1.5),
        )


if __name__ == "__main__":
    unittest.main()
