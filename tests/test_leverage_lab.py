"""Tests for the leverage experiment's account arithmetic.

The tool's conclusions rest on three claims: that position size scales P&L
linearly, that cross margin liquidates against the *summed* maintenance
requirement, and that a fixed fraction of equity compounds.  Each is tested
here against hand-computed numbers.
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.leverage_lab import Path, adverse_fraction, simulate, unit_return  # noqa: E402
from zetaalgo.broker import Trade  # noqa: E402

START = datetime(2025, 1, 1)


def trade(offset_h: float, hold_h: float, net: float, *, entry=100.0, qty=1.0,
          stop=98.0, mae_r=0.0) -> Trade:
    entry_ts = START + timedelta(hours=offset_h)
    return Trade(
        entry_ts=entry_ts, exit_ts=entry_ts + timedelta(hours=hold_h),
        entry_index=0, exit_index=1, entry_price=entry,
        exit_price=entry + net / qty, qty=qty, initial_stop=stop,
        target_price=None, direction=1, exit_reason="target",
        gross_pnl=net, commission=0.0, net_pnl=net,
        r_multiple=net / qty / abs(entry - stop), bars_held=int(hold_h),
        mfe_r=0.0, mae_r=mae_r, session="s", equity_after=0.0,
    )


class UnitReturnTests(unittest.TestCase):
    def test_return_is_pnl_over_entry_notional(self):
        # $2 on a 1-unit $100 position is 2%.
        self.assertAlmostEqual(unit_return(trade(0, 1, 2.0), 100.0), 0.02)

    def test_return_ignores_the_reference_notional_argument_value(self):
        # The fraction is a property of the trade, not of the run that found it.
        t = trade(0, 1, 2.0)
        self.assertAlmostEqual(unit_return(t, 50.0), unit_return(t, 500.0))

    def test_adverse_excursion_is_unsigned_and_scaled_by_the_stop(self):
        # mae_r is stored negative; a 0.5R excursion on a 2% stop is 1%.
        self.assertAlmostEqual(adverse_fraction(trade(0, 1, 1.0, mae_r=-0.5)), 0.01)

    def test_a_trade_that_never_went_against_you_has_no_excursion(self):
        self.assertEqual(adverse_fraction(trade(0, 1, 1.0, mae_r=0.0)), 0.0)


class PathTests(unittest.TestCase):
    def test_margin_is_notional_over_leverage(self):
        path = Path(100.0, 10.0, 0.005)
        self.assertAlmostEqual(path.margin_for(500.0), 50.0)

    def test_a_position_needing_more_margin_than_the_balance_is_refused(self):
        path = Path(100.0, 10.0, 0.005)
        self.assertFalse(path.open(1_001.0))
        self.assertEqual(path.skipped_margin, 1)
        self.assertEqual(path.taken, 0)

    def test_margin_is_pooled_across_open_positions(self):
        # Cross margin: the second position is refused because the pair would
        # need $110 of margin, not because either alone is too big.
        path = Path(100.0, 10.0, 0.005)
        self.assertTrue(path.open(600.0))
        self.assertFalse(path.open(500.0))
        self.assertEqual(path.peak_concurrent, 1)

    def test_drawdown_is_measured_from_the_running_peak(self):
        path = Path(100.0, 10.0, 0.005)
        path.open(100.0)
        path.close(100.0, 0.20)   # +$20 -> 120
        path.open(100.0)
        path.close(100.0, -0.30)  # -$30 -> 90, a 25% fall from 120
        self.assertAlmostEqual(path.equity, 90.0)
        self.assertAlmostEqual(path.max_dd, 0.25)
        self.assertAlmostEqual(path.low, 90.0)

    def test_equity_cannot_go_negative(self):
        path = Path(100.0, 100.0, 0.005)
        path.open(1_000.0)
        path.close(1_000.0, -0.50)  # a $500 loss on a $100 account
        self.assertEqual(path.equity, 0.0)
        self.assertTrue(path.liquidated)

    def test_liquidation_fires_while_other_positions_are_still_open(self):
        # $100 balance, two $2,000 positions: maintenance owed on the survivor
        # is $10, so settling a $95 loss leaves $5 and the account is gone.
        path = Path(100.0, 100.0, 0.005)
        path.open(2_000.0)
        path.open(2_000.0)
        path.close(2_000.0, -0.0475)
        self.assertAlmostEqual(path.equity, 5.0)
        self.assertTrue(path.liquidated)

    def test_a_healthy_account_is_not_liquidated(self):
        path = Path(100.0, 100.0, 0.005)
        path.open(500.0)
        path.close(500.0, -0.02)
        self.assertAlmostEqual(path.equity, 90.0)
        self.assertFalse(path.liquidated)


class SimulateTests(unittest.TestCase):
    def setUp(self):
        # Two sequential winners, each +2% of notional.
        self.trades = [trade(0, 1, 2.0), trade(2, 1, 2.0)]

    def test_fixed_margin_sizing_scales_the_position_by_leverage(self):
        low = simulate(self.trades, 100.0, 10.0, 0.005, 100.0, "margin", 5.0)
        high = simulate(self.trades, 100.0, 100.0, 0.005, 100.0, "margin", 5.0)
        # $50 then $500 of position, both +2% twice.
        self.assertAlmostEqual(low.equity, 100.0 + 2 * 0.02 * 50.0)
        self.assertAlmostEqual(high.equity, 100.0 + 2 * 0.02 * 500.0)

    def test_fixed_notional_sizing_ignores_leverage_entirely(self):
        a = simulate(self.trades, 100.0, 25.0, 0.005, 100.0, "notional", 250.0)
        b = simulate(self.trades, 100.0, 100.0, 0.005, 100.0, "notional", 250.0)
        self.assertAlmostEqual(a.equity, b.equity)

    def test_fractional_sizing_compounds(self):
        # 1% of equity as margin at 100x = a position worth the whole balance.
        path = simulate(self.trades, 100.0, 100.0, 0.005, 100.0, "fraction", 0.01)
        self.assertAlmostEqual(path.equity, 100.0 * 1.02 * 1.02)

    def test_overlapping_trades_are_open_at_once(self):
        overlapping = [trade(0, 5, 2.0), trade(1, 5, 2.0)]
        path = simulate(overlapping, 100.0, 100.0, 0.005, 100.0, "margin", 5.0)
        self.assertEqual(path.peak_concurrent, 2)
        self.assertAlmostEqual(path.peak_open_notional, 1_000.0)

    def test_the_stress_measure_sums_the_open_books_excursions(self):
        # Two overlapping $500 positions, each having shown a 1% adverse
        # excursion, is $10 of unrealised loss against a $100 balance.
        overlapping = [trade(0, 5, 2.0, mae_r=-0.5), trade(1, 5, 2.0, mae_r=-0.5)]
        path = simulate(overlapping, 100.0, 100.0, 0.005, 100.0, "margin", 5.0)
        self.assertAlmostEqual(path.peak_stress, 10.0)

    def test_the_run_stops_once_the_account_is_liquidated(self):
        wipeout = [trade(0, 1, -100.0), trade(2, 1, 2.0)]
        path = simulate(wipeout, 100.0, 100.0, 0.005, 100.0, "margin", 1.0)
        self.assertTrue(path.liquidated)
        self.assertEqual(path.equity, 0.0)
        self.assertEqual(path.taken, 1)


if __name__ == "__main__":
    unittest.main()
