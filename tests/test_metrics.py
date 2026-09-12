import math
import unittest
from datetime import datetime, timedelta

from zetaalgo.backtest import BacktestResult, EquityPoint
from zetaalgo.broker import Trade
from zetaalgo.config import BacktestConfig
from zetaalgo.metrics import compute_metrics, drawdown_series, max_consecutive


def make_trade(net_pnl: float, r_multiple: float = 1.0, day: int = 1) -> Trade:
    ts = datetime(2024, 1, day, 10, 0)
    return Trade(
        entry_ts=ts, exit_ts=ts, entry_index=0, exit_index=1,
        entry_price=100.0, exit_price=100.0 + net_pnl / 10.0, qty=10.0,
        initial_stop=99.0, target_price=102.0, exit_reason="target",
        gross_pnl=net_pnl, commission=0.0, net_pnl=net_pnl,
        r_multiple=r_multiple, bars_held=2, mfe_r=1.0, mae_r=-0.5,
        session=f"2024-01-{day:02d}", equity_after=100_000.0 + net_pnl,
    )


def make_result(trades, equity_points) -> BacktestResult:
    base = datetime(2024, 1, 1, 16, 0)
    curve = []
    for i, value in enumerate(equity_points):
        ts = base + timedelta(days=i)
        curve.append(
            EquityPoint(ts=ts, equity=value, exposure=0.0,
                        session=ts.date().isoformat())
        )
    return BacktestResult(
        trades=list(trades), equity_curve=curve,
        initial_equity=equity_points[0] if equity_points else 100_000.0,
        final_equity=equity_points[-1] if equity_points else 100_000.0,
        bars=len(curve), sessions=len(curve),
        backtest_config=BacktestConfig(),
    )


class TestDrawdown(unittest.TestCase):
    def test_a_rising_curve_never_draws_down(self):
        self.assertEqual(drawdown_series([1, 2, 3]), [0.0, 0.0, 0.0])

    def test_drawdown_is_measured_from_the_running_peak(self):
        result = drawdown_series([100, 120, 90, 150])
        self.assertAlmostEqual(result[2], -0.25)
        self.assertAlmostEqual(result[3], 0.0)

    def test_max_drawdown_percentage_is_reported_positive(self):
        metrics = compute_metrics(make_result([], [100.0, 120.0, 60.0, 90.0]))
        self.assertAlmostEqual(metrics.max_drawdown_pct, 50.0)
        self.assertAlmostEqual(metrics.max_drawdown, 60.0)


class TestStreaks(unittest.TestCase):
    def test_longest_losing_streak_is_found(self):
        trades = [make_trade(x) for x in (-1, -1, 5, -1, -1, -1, 5)]
        self.assertEqual(max_consecutive(trades, wins=False), 3)
        self.assertEqual(max_consecutive(trades, wins=True), 1)


class TestTradeStatistics(unittest.TestCase):
    def setUp(self):
        # Three winners of +200, two losers of -100.
        self.trades = [make_trade(200.0, 2.0, d) for d in (1, 2, 3)]
        self.trades += [make_trade(-100.0, -1.0, d) for d in (4, 5)]
        self.result = make_result(self.trades, [100_000.0, 100_400.0])

    def test_win_rate_counts_only_decided_trades(self):
        self.assertAlmostEqual(compute_metrics(self.result).win_rate, 60.0)

    def test_profit_factor_is_gross_profit_over_gross_loss(self):
        self.assertAlmostEqual(compute_metrics(self.result).profit_factor, 3.0)

    def test_expectancy_is_the_average_net_result(self):
        metrics = compute_metrics(self.result)
        self.assertAlmostEqual(metrics.expectancy, (600.0 - 200.0) / 5)
        self.assertAlmostEqual(metrics.expectancy_r, (6.0 - 2.0) / 5)

    def test_payoff_ratio_compares_average_win_to_average_loss(self):
        self.assertAlmostEqual(compute_metrics(self.result).payoff_ratio, 2.0)

    def test_best_and_worst_trades_are_reported(self):
        metrics = compute_metrics(self.result)
        self.assertAlmostEqual(metrics.best_trade, 200.0)
        self.assertAlmostEqual(metrics.worst_trade, -100.0)

    def test_profit_factor_is_infinite_when_nothing_ever_lost(self):
        result = make_result([make_trade(100.0)], [100_000.0, 100_100.0])
        self.assertEqual(compute_metrics(result).profit_factor, math.inf)

    def test_an_empty_run_yields_zeroed_statistics(self):
        metrics = compute_metrics(make_result([], [100_000.0]))
        self.assertEqual(metrics.trades, 0)
        self.assertEqual(metrics.win_rate, 0.0)
        self.assertEqual(metrics.profit_factor, 0.0)


class TestRiskAdjustedRatios(unittest.TestCase):
    def test_a_flat_curve_has_no_sharpe(self):
        metrics = compute_metrics(make_result([], [100_000.0] * 10))
        self.assertEqual(metrics.sharpe, 0.0)

    def test_a_steadily_rising_curve_has_a_high_sharpe(self):
        curve = [100_000.0 * (1.001 ** i) for i in range(60)]
        metrics = compute_metrics(make_result([], curve))
        self.assertGreater(metrics.sharpe, 10.0)
        self.assertGreater(metrics.cagr_pct, 0.0)

    def test_sortino_ignores_upside_volatility(self):
        curve = [100_000.0]
        for i in range(40):
            curve.append(curve[-1] * (1.02 if i % 2 == 0 else 1.001))
        metrics = compute_metrics(make_result([], curve))
        self.assertEqual(metrics.sortino, 0.0)  # no losing sessions at all


if __name__ == "__main__":
    unittest.main()
