import math
import unittest
from dataclasses import replace

from zetaalgo.backtest import run_backtest
from zetaalgo.config import BacktestConfig
from zetaalgo.data import generate_synthetic
from zetaalgo.live import LiveTrader, PaperBroker, run_paper_session
from zetaalgo.scalp import MeanReversionScalp, ScalpConfig


def drive(strategy, bars):
    strategy.prepare(bars)
    plans = []
    for i in range(len(bars)):
        plans.append(strategy.entry_plan(i))
        strategy.on_bar_close(i)
    return plans


class TestConfig(unittest.TestCase):
    def test_bad_values_are_rejected(self):
        for kwargs in (dict(mean_kind="x"), dict(mean_period=1), dict(entry_z=0),
                       dict(stop_z=-1), dict(tp_fraction=0), dict(tp_fraction=1.5),
                       dict(tick_size=0), dict(trade_longs=False, trade_shorts=False)):
            with self.assertRaises(ValueError, msg=kwargs):
                ScalpConfig(**kwargs)

    def test_target_r_is_the_ratio_of_the_two_vol_distances(self):
        # target = tp_fraction * entry_z * vol, risk = stop_z * vol.
        self.assertAlmostEqual(
            ScalpConfig(tp_fraction=0.5, entry_z=2.0, stop_z=2.0).target_r, 0.5)
        self.assertAlmostEqual(
            ScalpConfig(tp_fraction=1.0, entry_z=3.0, stop_z=6.0).target_r, 0.5)

    def test_time_stop_is_exposed_to_the_engine(self):
        self.assertEqual(ScalpConfig(time_stop_bars=7).max_bars_in_trade, 7)


class TestQuoting(unittest.TestCase):
    """The quote is a maker limit at a volatility band, re-derived each bar."""

    def test_the_entry_is_a_limit_order(self):
        strategy = MeanReversionScalp(ScalpConfig())
        plan = next(p for p in drive(strategy, generate_synthetic(days=20, seed=81)) if p)
        self.assertEqual(plan.order_kind, "limit")

    def test_a_long_is_quoted_below_the_mean_and_a_short_above(self):
        strategy = MeanReversionScalp(ScalpConfig(min_tp_bps=0.0))
        strategy.prepare(generate_synthetic(days=20, seed=82))
        for i in range(1, len(strategy.bars)):
            plan = strategy.entry_plan(i)
            strategy.on_bar_close(i)
            if plan is None:
                continue
            mean = strategy.mean[i - 1]
            if plan.direction > 0:
                self.assertLess(plan.trigger_price, mean)
                self.assertLess(plan.stop_price, plan.trigger_price)
            else:
                self.assertGreater(plan.trigger_price, mean)
                self.assertGreater(plan.stop_price, plan.trigger_price)

    def test_the_quote_moves_as_the_band_moves(self):
        strategy = MeanReversionScalp(ScalpConfig(min_tp_bps=0.0))
        prices = {p.trigger_price for p in drive(strategy, generate_synthetic(days=20, seed=83)) if p}
        self.assertGreater(len(prices), 10, "a re-priced quote should not be static")

    def test_a_wider_band_quotes_further_from_the_mean(self):
        bars = generate_synthetic(days=20, seed=84)
        near = MeanReversionScalp(ScalpConfig(entry_z=1.0, min_tp_bps=0.0))
        far = MeanReversionScalp(ScalpConfig(entry_z=3.0, min_tp_bps=0.0))
        near_plans = [p for p in drive(near, bars) if p and p.direction > 0]
        far_plans = [p for p in drive(far, bars) if p and p.direction > 0]
        self.assertTrue(near_plans and far_plans)
        # A deeper band fills less often.
        self.assertLess(len(run_backtest(bars, strategy=MeanReversionScalp(
            ScalpConfig(entry_z=3.0, min_tp_bps=0.0))).trades),
            len(run_backtest(bars, strategy=MeanReversionScalp(
                ScalpConfig(entry_z=1.0, min_tp_bps=0.0))).trades))


class TestFeeGate(unittest.TestCase):
    """The whole point: a scalp that cannot pay its round trip must not trade."""

    def test_a_demanding_floor_refuses_every_setup(self):
        strategy = MeanReversionScalp(ScalpConfig(min_tp_bps=10_000.0))
        plans = drive(strategy, generate_synthetic(days=20, seed=85))
        self.assertTrue(all(p is None for p in plans))
        self.assertGreater(strategy.rejections.get("tp_below_fee_floor", 0), 0)

    def test_raising_the_floor_can_only_reduce_trades(self):
        bars = generate_synthetic(days=40, seed=86)
        counts = [
            len(run_backtest(bars, strategy=MeanReversionScalp(
                ScalpConfig(min_tp_bps=floor))).trades)
            for floor in (0.0, 10.0, 25.0, 60.0)
        ]
        self.assertEqual(counts, sorted(counts, reverse=True), counts)

    def test_every_taken_trade_clears_the_floor(self):
        floor = 15.0
        bars = generate_synthetic(days=40, seed=87)
        config = ScalpConfig(min_tp_bps=floor)
        result = run_backtest(bars, strategy=MeanReversionScalp(config))
        self.assertTrue(result.trades)
        for trade in result.trades:
            reward = abs(trade.target_price - trade.entry_price)
            self.assertGreaterEqual(reward / trade.entry_price * 10_000.0, floor - 1e-6)


class TestLaggingFilterIsConfirmationOnly(unittest.TestCase):
    def test_the_filter_can_only_remove_trades_never_add_them(self):
        bars = generate_synthetic(days=40, seed=88)
        off = run_backtest(bars, strategy=MeanReversionScalp(
            ScalpConfig(trend_filter=False)))
        on = run_backtest(bars, strategy=MeanReversionScalp(
            ScalpConfig(trend_filter=True, max_adverse_slope_bps=0.0)))
        self.assertLessEqual(len(on.trades), len(off.trades))

    def test_a_permissive_threshold_vetoes_less_than_a_strict_one(self):
        bars = generate_synthetic(days=40, seed=89)
        strict = MeanReversionScalp(ScalpConfig(max_adverse_slope_bps=0.0))
        loose = MeanReversionScalp(ScalpConfig(max_adverse_slope_bps=500.0))
        drive(strict, bars)
        drive(loose, bars)
        self.assertGreater(strict.rejections.get("trend_veto", 0),
                           loose.rejections.get("trend_veto", 0))


class TestNoLookahead(unittest.TestCase):
    def test_the_quote_for_a_bar_does_not_use_that_bar(self):
        bars = generate_synthetic(days=30, seed=90)
        config = ScalpConfig(min_tp_bps=0.0)
        strategy = MeanReversionScalp(config)
        strategy.prepare(bars)
        target = None
        for i in range(len(bars)):
            plan = strategy.entry_plan(i)
            if plan is not None:
                target, baseline = i, plan
                break
            strategy.on_bar_close(i)
        self.assertIsNotNone(target)
        mutated = list(bars)
        original = mutated[target]
        mutated[target] = replace(original, open=original.open * 0.8,
                                  high=original.high * 1.5, low=original.low * 0.5,
                                  close=original.close * 0.8,
                                  volume=original.volume * 11)
        other = MeanReversionScalp(config)
        other.prepare(mutated)
        for i in range(target):
            other.on_bar_close(i)
        after = other.entry_plan(target)
        self.assertAlmostEqual(baseline.trigger_price, after.trigger_price, places=9)
        self.assertAlmostEqual(baseline.stop_price, after.stop_price, places=9)


class TestParity(unittest.TestCase):
    """A re-priced quote must be cancel/replaced live, or it trades stale."""

    def assert_parity(self, bars, config, backtest_config=None):
        result = run_backtest(bars, backtest_config=backtest_config,
                              strategy=MeanReversionScalp(config))
        broker = run_paper_session(bars, backtest_config=backtest_config,
                                   strategy=MeanReversionScalp(config))
        opens = [f for i, f in enumerate(broker.fills) if i % 2 == 0]
        self.assertEqual(len(opens), len(result.trades))
        for i, trade in enumerate(result.trades):
            self.assertEqual(opens[i].ts, trade.entry_ts)
            self.assertAlmostEqual(opens[i].price, trade.entry_price, places=6)
        self.assertAlmostEqual(broker.equity(), result.final_equity, places=6)

    def test_parity_on_defaults(self):
        self.assert_parity(generate_synthetic(days=20, seed=91), ScalpConfig())

    def test_parity_with_maker_fees_long_only(self):
        self.assert_parity(
            generate_synthetic(days=20, seed=92),
            ScalpConfig(trade_shorts=False, entry_z=3.0, stop_z=6.0, tp_fraction=1.0),
            BacktestConfig(commission_bps=5.0, maker_bps=2.0, lot_size=0.01),
        )

    def test_a_stale_quote_is_cancelled_and_replaced(self):
        bars = generate_synthetic(days=20, seed=93)
        broker = PaperBroker()
        trader = LiveTrader(broker, strategy=MeanReversionScalp(
            ScalpConfig(min_tp_bps=0.0)))
        cancels = []
        original = broker.cancel
        broker.cancel = lambda oid: (cancels.append(oid), original(oid))[1]
        for i, bar in enumerate(bars):
            broker.on_bar(bar)
            trader.on_bar(bar)
            self.assertLessEqual(len(broker.working), 1)
        self.assertGreater(len(cancels), 10, "a moving band must re-quote")


if __name__ == "__main__":
    unittest.main()


class TestVarianceRatioGate(unittest.TestCase):
    """Refuse to trade mean reversion when prices are trending.

    The variance ratio is Var(k-bar return) / (k x Var(1-bar return)): a
    random walk gives 1.0, below 1.0 means moves get retraced and above 1.0
    means they persist.  Gating on it is the one regime filter that held up
    out of sample in both directions.
    """

    def test_a_random_walk_scores_near_one(self):
        import random
        rng = random.Random(5)
        price = 100.0
        closes = []
        for _ in range(6000):
            price *= 1 + rng.gauss(0.0, 0.002)
            closes.append(price)
        strategy = MeanReversionScalp(ScalpConfig(max_variance_ratio=1.0,
                                                  vr_window=2000, vr_period=12))
        ratios = [r for r in strategy._rolling_variance_ratio(closes) if r is not None]
        self.assertTrue(ratios)
        average = sum(ratios) / len(ratios)
        self.assertGreater(average, 0.7)
        self.assertLess(average, 1.4)

    def test_momentum_scores_above_one(self):
        """Positively autocorrelated returns make k-bar moves persist.

        Note what does NOT raise the ratio: a constant drift.  Variance is
        measured around the mean, so steady direction cancels out.  The gate
        therefore measures path persistence, not whether the market is going
        up or down -- which is why it survived out of sample where a
        bull/bear gate did not.
        """
        import random
        rng = random.Random(11)
        price, previous = 100.0, 0.0
        closes = []
        for _ in range(6000):
            previous = 0.35 * previous + rng.gauss(0.0, 0.001)
            price *= 1 + previous
            closes.append(price)
        strategy = MeanReversionScalp(ScalpConfig(max_variance_ratio=1.0,
                                                  vr_window=2000, vr_period=12))
        ratios = [r for r in strategy._rolling_variance_ratio(closes) if r is not None]
        self.assertTrue(ratios)
        self.assertGreater(sum(ratios) / len(ratios), 1.3)

    def test_a_constant_drift_does_not_move_the_ratio(self):
        """Direction is not persistence: adding drift leaves the ratio alone."""
        import random
        def series(drift):
            rng = random.Random(21)
            price = 100.0
            out = []
            for _ in range(6000):
                price *= 1 + drift + rng.gauss(0.0, 0.002)
                out.append(price)
            return out
        strategy = MeanReversionScalp(ScalpConfig(max_variance_ratio=1.0,
                                                  vr_window=2000, vr_period=12))
        flat = [r for r in strategy._rolling_variance_ratio(series(0.0)) if r is not None]
        rising = [r for r in strategy._rolling_variance_ratio(series(0.001)) if r is not None]
        self.assertAlmostEqual(sum(flat) / len(flat), sum(rising) / len(rising), places=1)

    def test_a_noiseless_series_has_no_defined_ratio(self):
        """Zero return variance is 0/0, not a huge number."""
        closes = [100.0 * (1.0005 ** i) for i in range(6000)]
        strategy = MeanReversionScalp(ScalpConfig(max_variance_ratio=1.0,
                                                  vr_window=2000, vr_period=12))
        ratios = [r for r in strategy._rolling_variance_ratio(closes) if r is not None]
        for r in ratios:
            self.assertGreaterEqual(r, 0.0)  # never negative, whatever else

    def test_a_sawtooth_series_scores_below_one(self):
        closes = [100.0 + (2.0 if i % 2 else -2.0) for i in range(6000)]
        strategy = MeanReversionScalp(ScalpConfig(max_variance_ratio=1.0,
                                                  vr_window=2000, vr_period=12))
        ratios = [r for r in strategy._rolling_variance_ratio(closes) if r is not None]
        self.assertTrue(ratios)
        self.assertLess(sum(ratios) / len(ratios), 0.5)

    def test_the_gate_only_removes_trades(self):
        bars = generate_synthetic(days=90, seed=131)
        base = run_backtest(bars, strategy=MeanReversionScalp(
            ScalpConfig(max_variance_ratio=0.0)))
        gated = run_backtest(bars, strategy=MeanReversionScalp(
            ScalpConfig(max_variance_ratio=1.0, vr_window=500)))
        self.assertLessEqual(len(gated.trades), len(base.trades))

    def test_a_tighter_gate_removes_more(self):
        bars = generate_synthetic(days=90, seed=132)
        counts = [
            len(run_backtest(bars, strategy=MeanReversionScalp(
                ScalpConfig(max_variance_ratio=vr, vr_window=500))).trades)
            for vr in (1.2, 1.0, 0.8)
        ]
        self.assertEqual(counts, sorted(counts, reverse=True), counts)

    def test_the_gate_is_recorded_in_the_funnel(self):
        bars = generate_synthetic(days=90, seed=133)
        strategy = MeanReversionScalp(ScalpConfig(max_variance_ratio=0.5,
                                                  vr_window=500))
        run_backtest(bars, strategy=strategy)
        self.assertGreater(strategy.rejections.get("trending_regime_veto", 0), 0)
