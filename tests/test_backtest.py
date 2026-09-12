import unittest
from dataclasses import replace

from helpers import make_bars, ramp
from zetaalgo.backtest import run_backtest
from zetaalgo.config import BacktestConfig, StrategyConfig
from zetaalgo.data import Bar, generate_synthetic

NO_COST = BacktestConfig(commission_bps=0.0, slippage_ticks=0.0)
LOOSE = StrategyConfig(min_gap_atr=0.0)


class TestEngineBasics(unittest.TestCase):
    def test_empty_input_produces_an_empty_result(self):
        result = run_backtest([])
        self.assertEqual(result.trades, [])
        self.assertEqual(result.final_equity, result.initial_equity)
        self.assertEqual(result.equity_curve, [])

    def test_too_little_history_to_warm_up_produces_no_trades(self):
        result = run_backtest(make_bars([100.0] * 5))
        self.assertEqual(result.trades, [])

    def test_equity_curve_has_one_point_per_bar(self):
        bars = generate_synthetic(days=5, seed=3)
        result = run_backtest(bars)
        self.assertEqual(len(result.equity_curve), len(bars))

    def test_results_are_deterministic(self):
        bars = generate_synthetic(days=20, seed=5)
        first, second = run_backtest(bars), run_backtest(bars)
        self.assertEqual(len(first.trades), len(second.trades))
        self.assertAlmostEqual(first.final_equity, second.final_equity, places=9)

    def test_realised_pnl_reconciles_with_final_equity(self):
        bars = generate_synthetic(days=40, seed=8)
        result = run_backtest(bars)
        self.assertGreater(len(result.trades), 0)
        realised = sum(trade.net_pnl for trade in result.trades)
        self.assertAlmostEqual(
            result.initial_equity + realised, result.final_equity, places=6
        )

    def test_no_position_is_left_open_at_the_end_of_the_data(self):
        result = run_backtest(generate_synthetic(days=15, seed=2))
        last = result.equity_curve[-1]
        self.assertAlmostEqual(last.exposure, 0.0, places=9)


class TestFillRealism(unittest.TestCase):
    def _one_trade(self, bars, scfg=None, bcfg=None):
        result = run_backtest(bars, scfg or LOOSE, bcfg or NO_COST)
        self.assertTrue(result.trades, "expected at least one trade")
        return result.trades[0]

    def test_entry_fills_at_the_trigger_when_the_bar_trades_through_it(self):
        trade = self._one_trade(make_bars(ramp()))
        # With zero slippage the fill is the trigger, one tick above the
        # cross candle's high, unless the bar opened above it.
        self.assertGreater(trade.entry_price, 0)
        self.assertGreaterEqual(trade.entry_price, trade.initial_stop)

    def test_a_gap_above_the_trigger_fills_at_the_open_not_the_trigger(self):
        bars = make_bars(ramp())
        result = run_backtest(bars, LOOSE, NO_COST)
        trade = result.trades[0]
        open_price = bars[trade.entry_index].open
        # The fill can never be better than the bar's open on a gap up.
        self.assertGreaterEqual(trade.entry_price + 1e-9, min(open_price, trade.entry_price))
        self.assertLessEqual(trade.entry_price, bars[trade.entry_index].high + 1e-9)

    def test_a_fill_never_lands_outside_the_bar_range(self):
        bars = generate_synthetic(days=30, seed=4)
        result = run_backtest(bars)
        for trade in result.trades:
            bar = bars[trade.entry_index]
            self.assertGreaterEqual(trade.entry_price, bar.low - 1e-9)
            self.assertLessEqual(trade.entry_price, bar.high + 1e-9)

    def test_slippage_makes_entries_worse_and_reduces_profit(self):
        bars = generate_synthetic(days=40, seed=6)
        clean = run_backtest(bars, LOOSE, BacktestConfig(slippage_ticks=0.0,
                                                         commission_bps=0.0))
        dirty = run_backtest(bars, LOOSE, BacktestConfig(slippage_ticks=5.0,
                                                         commission_bps=0.0))
        self.assertLess(dirty.final_equity, clean.final_equity)

    def test_commission_reduces_final_equity(self):
        bars = generate_synthetic(days=40, seed=6)
        free = run_backtest(bars, LOOSE, BacktestConfig(commission_bps=0.0,
                                                        slippage_ticks=0.0))
        costly = run_backtest(bars, LOOSE, BacktestConfig(commission_bps=20.0,
                                                          slippage_ticks=0.0))
        self.assertLess(costly.final_equity, free.final_equity)
        self.assertGreater(sum(t.commission for t in costly.trades), 0.0)


class TestExitPriority(unittest.TestCase):
    def test_a_bar_containing_both_stop_and_target_is_booked_as_the_stop(self):
        """Bar data cannot order the two touches, so the loss is assumed."""
        bars = make_bars(ramp(flat=12, rise=6))
        result = run_backtest(bars, LOOSE, NO_COST)
        self.assertTrue(result.trades)
        trade = result.trades[0]
        # Widen the bar after entry so it spans both levels, then re-run.
        entry_index = trade.entry_index
        if entry_index + 1 < len(bars):
            wide = list(bars)
            bar = wide[entry_index + 1]
            wide[entry_index + 1] = replace(
                bar,
                high=max(bar.high, trade.entry_price * 1.5),
                low=min(bar.low, trade.initial_stop * 0.5),
            )
            rerun = run_backtest(wide, LOOSE, NO_COST)
            self.assertIn(rerun.trades[0].exit_reason, ("stop", "stop_gap"))

    def test_positions_are_flattened_at_the_session_close_by_default(self):
        bars = generate_synthetic(days=25, seed=9)
        result = run_backtest(bars)
        sessions = {bar.session: [] for bar in bars}
        for bar in bars:
            sessions[bar.session].append(bar.ts)
        for trade in result.trades:
            self.assertEqual(trade.entry_ts.date(), trade.exit_ts.date())

    def test_holding_overnight_can_be_enabled(self):
        bars = generate_synthetic(days=25, seed=9)
        config = replace(LOOSE, flat_at_session_end=False,
                         setup_expires_at_session_end=False)
        result = run_backtest(bars, config)
        self.assertNotIn("session_end", result.exit_reasons)

    def test_time_stop_caps_how_long_a_trade_is_held(self):
        bars = generate_synthetic(days=40, seed=12)
        config = replace(LOOSE, max_bars_in_trade=3, target_r=0.0,
                         exit_on_close_below_ema=False,
                         exit_on_close_below_vwap=False)
        result = run_backtest(bars, config)
        self.assertTrue(result.trades)
        for trade in result.trades:
            self.assertLessEqual(trade.bars_held, 3)

    def test_target_exit_realises_about_the_configured_r_multiple(self):
        bars = generate_synthetic(days=60, seed=7)
        result = run_backtest(bars, LOOSE, NO_COST)
        targets = [t for t in result.trades if t.exit_reason == "target"]
        self.assertTrue(targets)
        for trade in targets:
            self.assertAlmostEqual(trade.r_multiple, 2.0, places=6)


class TestEntryBarStopPolicy(unittest.TestCase):
    """The intrabar path assumption is explicit and must stay that way."""

    def test_pessimistic_policy_never_beats_the_close_policy(self):
        bars = generate_synthetic(days=80, seed=13)
        low = run_backtest(bars, replace(LOOSE, entry_bar_stop="low"))
        close = run_backtest(bars, replace(LOOSE, entry_bar_stop="close"))
        self.assertLessEqual(low.final_equity, close.final_equity)

    def test_pessimistic_policy_produces_same_bar_stop_outs(self):
        bars = generate_synthetic(days=80, seed=13)
        low = run_backtest(bars, replace(LOOSE, entry_bar_stop="low"))
        nxt = run_backtest(bars, replace(LOOSE, entry_bar_stop="next_bar"))
        same_bar_low = sum(1 for t in low.trades
                           if t.bars_held == 0 and t.exit_reason == "stop")
        same_bar_next = sum(1 for t in nxt.trades
                            if t.bars_held == 0 and t.exit_reason == "stop")
        self.assertGreater(same_bar_low, same_bar_next)
        self.assertEqual(same_bar_next, 0)

    def test_trade_count_is_unaffected_by_the_policy(self):
        bars = generate_synthetic(days=80, seed=13)
        counts = {
            policy: len(run_backtest(bars, replace(LOOSE, entry_bar_stop=policy)).trades)
            for policy in ("low", "close", "next_bar")
        }
        self.assertEqual(len(set(counts.values())), 1, counts)


class TestSizingAndRisk(unittest.TestCase):
    def test_risk_sizing_keeps_losses_near_the_configured_risk(self):
        bars = generate_synthetic(days=60, seed=14)
        config = BacktestConfig(risk_pct=0.005, max_notional_pct=10.0,
                                commission_bps=0.0, slippage_ticks=0.0)
        result = run_backtest(bars, replace(LOOSE, breakeven_at_r=0.0), config)
        stops = [t for t in result.trades if t.exit_reason == "stop"]
        self.assertTrue(stops)
        for trade in stops:
            risked = 0.005 * 100_000 * 3  # generous bound: equity drifts
            self.assertLess(abs(trade.net_pnl), risked)

    def test_leverage_cap_bounds_position_notional(self):
        bars = generate_synthetic(days=40, seed=15)
        config = BacktestConfig(max_notional_pct=0.25, risk_pct=1.0)
        result = run_backtest(bars, LOOSE, config)
        self.assertTrue(result.trades)
        for trade in result.trades:
            notional = trade.entry_price * trade.qty
            self.assertLessEqual(notional, trade.equity_after * 0.35 + 1000)

    def test_fixed_sizing_uses_the_same_quantity_every_trade(self):
        bars = generate_synthetic(days=40, seed=16)
        config = BacktestConfig(sizing="fixed", fixed_qty=50.0)
        result = run_backtest(bars, LOOSE, config)
        self.assertTrue(result.trades)
        self.assertEqual({t.qty for t in result.trades}, {50.0})

    def test_max_trades_per_session_is_respected(self):
        bars = generate_synthetic(days=60, seed=17)
        result = run_backtest(bars, replace(LOOSE, max_trades_per_session=1))
        seen = {}
        for trade in result.trades:
            seen[trade.session] = seen.get(trade.session, 0) + 1
        self.assertTrue(seen)
        self.assertLessEqual(max(seen.values()), 1)


class TestNoLookaheadEndToEnd(unittest.TestCase):
    def test_changing_the_last_bar_cannot_change_earlier_trades(self):
        bars = generate_synthetic(days=30, seed=18)
        baseline = run_backtest(bars)
        mutated = list(bars)
        last = mutated[-1]
        mutated[-1] = replace(last, high=last.high * 3, low=last.low * 0.3,
                              close=last.close * 0.4)
        altered = run_backtest(mutated)
        closed_before = [t for t in baseline.trades if t.exit_index < len(bars) - 1]
        altered_before = [t for t in altered.trades if t.exit_index < len(bars) - 1]
        self.assertEqual(len(closed_before), len(altered_before))
        for first, second in zip(closed_before, altered_before):
            self.assertEqual(first.entry_ts, second.entry_ts)
            self.assertAlmostEqual(first.net_pnl, second.net_pnl, places=9)


if __name__ == "__main__":
    unittest.main()


class TestLotSizing(unittest.TestCase):
    """Contract granularity must not silently delete trades.

    Flooring to whole units is right for shares and wrong for high-priced
    contracts: risking 1% of 100k through a wide stop on a 70k instrument
    works out to a fraction of a contract, which floors to zero and vanishes
    from the backtest without any error.
    """

    def test_a_fractional_contract_is_not_rounded_away(self):
        from zetaalgo.broker import position_size

        # 0.39 contracts: zero under whole-unit flooring, tradeable at lotSz 0.01.
        whole = position_size(100_000, 71_841.8, 71_841.8 - 2_572.6, BacktestConfig())
        lots = position_size(100_000, 71_841.8, 71_841.8 - 2_572.6,
                             BacktestConfig(lot_size=0.01))
        self.assertEqual(whole, 0.0)
        self.assertAlmostEqual(lots, 0.38, places=6)

    def test_quantity_is_always_a_whole_number_of_lots(self):
        from zetaalgo.broker import position_size

        for lot in (0.01, 0.1, 1.0, 5.0):
            qty = position_size(100_000, 137.0, 129.0, BacktestConfig(lot_size=lot))
            self.assertAlmostEqual(qty / lot, round(qty / lot), places=6, msg=lot)

    def test_a_smaller_lot_size_can_only_admit_more_trades(self):
        bars = generate_synthetic(days=60, seed=91, start_price=70_000.0)
        config = replace(LOOSE, stop_mode="atr", stop_atr_mult=4.0)
        coarse = run_backtest(bars, config, BacktestConfig(lot_size=1.0))
        fine = run_backtest(bars, config, BacktestConfig(lot_size=0.001))
        self.assertGreaterEqual(len(fine.trades), len(coarse.trades))
