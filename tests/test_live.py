import unittest

from zetaalgo.backtest import run_backtest
from zetaalgo.config import BacktestConfig, StrategyConfig
from zetaalgo.data import generate_synthetic
from zetaalgo.live import LiveTrader, Order, PaperBroker, run_paper_session


def paper_round_trips(broker):
    buys = [f for f in broker.fills if f.side == "buy"]
    sells = [f for f in broker.fills if f.side == "sell"]
    return buys, sells


class TestBacktestParity(unittest.TestCase):
    """The automation must reproduce the backtest exactly.

    This is the guarantee that makes the backtest worth anything: if the live
    path can drift, the reported edge describes a system nobody is running.
    """

    def assert_parity(self, bars, scfg=None, bcfg=None):
        scfg = scfg or StrategyConfig()
        result = run_backtest(bars, scfg, bcfg)
        broker = run_paper_session(bars, scfg, bcfg)
        buys, sells = paper_round_trips(broker)
        self.assertEqual(len(buys), len(result.trades), "trade count differs")
        for i, trade in enumerate(result.trades):
            self.assertEqual(buys[i].ts, trade.entry_ts, f"trade {i} entry time")
            self.assertAlmostEqual(buys[i].price, trade.entry_price, places=6,
                                   msg=f"trade {i} entry price")
            self.assertAlmostEqual(buys[i].qty, trade.qty, places=6,
                                   msg=f"trade {i} quantity")
            self.assertAlmostEqual(sells[i].price, trade.exit_price, places=6,
                                   msg=f"trade {i} exit price")
            self.assertEqual(sells[i].reason, trade.exit_reason, f"trade {i} reason")
        self.assertAlmostEqual(broker.equity(), result.final_equity, places=6)

    def test_parity_on_default_settings(self):
        self.assert_parity(generate_synthetic(days=22, seed=21))

    def test_parity_across_entry_bar_stop_policies(self):
        bars = generate_synthetic(days=18, seed=22)
        for policy in ("close", "low", "next_bar"):
            with self.subTest(policy=policy):
                self.assert_parity(bars, StrategyConfig(entry_bar_stop=policy))

    def test_parity_with_a_trailing_stop(self):
        bars = generate_synthetic(days=20, seed=23)
        self.assert_parity(bars, StrategyConfig(trail_mode="ema"))

    def test_parity_with_no_target_and_a_time_stop(self):
        bars = generate_synthetic(days=20, seed=24)
        self.assert_parity(
            bars, StrategyConfig(target_r=0.0, max_bars_in_trade=6, breakeven_at_r=0.0)
        )

    def test_parity_with_costs_and_strict_windows(self):
        bars = generate_synthetic(days=20, seed=25)
        scfg = StrategyConfig(confirm_window=2, entry_window=1, min_gap_atr=0.05)
        bcfg = BacktestConfig(commission_bps=5.0, slippage_ticks=3.0, risk_pct=0.005)
        self.assert_parity(bars, scfg, bcfg)


class TestRollingWindow(unittest.TestCase):
    """Regression tests for state that must survive the window sliding.

    The live trader keeps only a trailing window of bars, so any state stored
    as a window-relative index silently stops pointing at the right bar once
    the window starts sliding.
    """

    def _run(self, bars, scfg, window):
        broker = PaperBroker(tick_size=scfg.tick_size,
                             entry_bar_stop=scfg.entry_bar_stop)
        trader = LiveTrader(broker, scfg, window=window)
        for i, bar in enumerate(bars):
            last = i + 1 >= len(bars) or bars[i + 1].session != bar.session
            broker.on_bar(bar)
            trader.on_bar(bar, last_of_session=last)
        return broker

    def test_a_small_sliding_window_matches_a_large_one(self):
        bars = generate_synthetic(days=18, seed=26)
        scfg = StrategyConfig()
        small = self._run(bars, scfg, window=200)
        large = self._run(bars, scfg, window=100_000)
        self.assertAlmostEqual(small.equity(), large.equity(), places=6)
        self.assertEqual(len(small.fills), len(large.fills))

    def test_time_stop_still_fires_once_the_window_slides(self):
        bars = generate_synthetic(days=20, seed=27)
        scfg = StrategyConfig(max_bars_in_trade=4, target_r=0.0,
                              breakeven_at_r=0.0,
                              exit_on_close_below_ema=False,
                              exit_on_close_below_vwap=False)
        broker = self._run(bars, scfg, window=150)
        reasons = {f.reason for f in broker.fills if f.side == "sell"}
        self.assertIn("time_stop", reasons)

    def test_the_window_never_cuts_into_the_current_session(self):
        bars = generate_synthetic(days=10, seed=28)
        broker = PaperBroker()
        trader = LiveTrader(broker, StrategyConfig(), window=20)
        for i, bar in enumerate(bars):
            last = i + 1 >= len(bars) or bars[i + 1].session != bar.session
            broker.on_bar(bar)
            trader.on_bar(bar, last_of_session=last)
            # The VWAP anchor requires every bar of the live session.
            session_bars = [b for b in bars[: i + 1] if b.session == bar.session]
            held = [b for b in trader.bars if b.session == bar.session]
            self.assertEqual(len(held), len(session_bars))


class TestOrderManagement(unittest.TestCase):
    def test_a_resting_entry_order_is_cancelled_when_the_setup_dies(self):
        bars = generate_synthetic(days=20, seed=29)
        broker = PaperBroker()
        trader = LiveTrader(broker, StrategyConfig())
        cancelled = []
        original = broker.cancel
        broker.cancel = lambda oid: (cancelled.append(oid), original(oid))[1]
        for i, bar in enumerate(bars):
            last = i + 1 >= len(bars) or bars[i + 1].session != bar.session
            broker.on_bar(bar)
            trader.on_bar(bar, last_of_session=last)
        self.assertTrue(cancelled, "expected stale entry orders to be cancelled")
        self.assertEqual(broker.working, {})

    def test_the_trader_never_holds_two_positions_at_once(self):
        bars = generate_synthetic(days=18, seed=30)
        broker = PaperBroker()
        trader = LiveTrader(broker, StrategyConfig())
        for i, bar in enumerate(bars):
            last = i + 1 >= len(bars) or bars[i + 1].session != bar.session
            broker.on_bar(bar)
            trader.on_bar(bar, last_of_session=last)
            self.assertLessEqual(len(broker.working), 1)
            self.assertGreaterEqual(broker.qty, 0.0)

    def test_entry_orders_carry_a_protective_stop(self):
        bars = generate_synthetic(days=20, seed=31)
        broker = PaperBroker()
        trader = LiveTrader(broker, StrategyConfig())
        for i, bar in enumerate(bars):
            last = i + 1 >= len(bars) or bars[i + 1].session != bar.session
            broker.on_bar(bar)
            for order in trader.on_bar(bar, last_of_session=last):
                if order.side == "buy":
                    self.assertIsNotNone(order.stop_loss)
                    self.assertLess(order.stop_loss, order.price)
                    self.assertEqual(order.order_type, "stop")

    def test_target_is_resolved_against_the_actual_fill_not_the_trigger(self):
        broker = PaperBroker(BacktestConfig(commission_bps=0.0), tick_size=0.01)
        order = Order(side="buy", qty=10, order_type="market", price=101.0,
                      stop_loss=99.0, target_r=2.0, reason="test")
        broker._last_close = 101.0
        broker.submit(order)
        # Filled at 101 with a stop at 99 => 1R is 2.00, so the target is 105.
        self.assertAlmostEqual(broker.target_price, 105.0, places=6)

    def test_paper_broker_refuses_to_buy_beyond_available_cash(self):
        broker = PaperBroker(BacktestConfig(initial_equity=1_000.0,
                                            commission_bps=0.0))
        broker._last_close = 100.0
        broker.submit(Order(side="buy", qty=1_000, order_type="market",
                            price=100.0, stop_loss=99.0, reason="oversized"))
        self.assertGreaterEqual(broker.cash, -1e-9)
        self.assertLessEqual(broker.qty, 10.0)


if __name__ == "__main__":
    unittest.main()
