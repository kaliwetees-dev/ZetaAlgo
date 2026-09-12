import math
import unittest

from zetaalgo.indicators import (
    atr,
    crossed_above,
    ema,
    session_vwap,
    sma,
    true_range,
    typical_price,
)


class TestMovingAverages(unittest.TestCase):
    def test_sma_is_none_until_window_is_full(self):
        self.assertEqual(sma([1, 2, 3, 4], 3), [None, None, 2.0, 3.0])

    def test_ema_is_seeded_with_the_sma_of_the_first_period(self):
        values = list(range(1, 13))
        result = ema(values, 9)
        self.assertEqual(result[:8], [None] * 8)
        self.assertAlmostEqual(result[8], 5.0)  # SMA of 1..9

    def test_ema_of_a_constant_series_is_that_constant(self):
        result = ema([42.0] * 20, 9)
        for value in result[8:]:
            self.assertAlmostEqual(value, 42.0)

    def test_ema_tracks_a_step_change_faster_than_a_longer_ema(self):
        values = [100.0] * 20 + [110.0] * 20
        fast, slow = ema(values, 9), ema(values, 21)
        self.assertGreater(fast[-15], slow[-15])

    def test_ema_returns_all_none_when_history_is_too_short(self):
        self.assertEqual(ema([1.0, 2.0], 9), [None, None])

    def test_invalid_period_is_rejected(self):
        for period in (0, -3):
            with self.assertRaises(ValueError):
                ema([1.0, 2.0], period)


class TestTrueRangeAndAtr(unittest.TestCase):
    def test_true_range_has_no_value_on_the_first_bar(self):
        self.assertIsNone(true_range([2], [1], [1.5])[0])

    def test_true_range_uses_the_widest_of_the_three_measures(self):
        # A gap up: the high-to-previous-close distance dominates.
        result = true_range([10, 20], [9, 19], [9.5, 19.5])
        self.assertAlmostEqual(result[1], 20 - 9.5)

    def test_atr_of_a_flat_series_equals_the_bar_range(self):
        result = atr([11.0] * 30, [9.0] * 30, [10.0] * 30, 14)
        self.assertIsNone(result[13])  # first ATR lands at index == period
        self.assertAlmostEqual(result[14], 2.0, places=6)
        self.assertAlmostEqual(result[-1], 2.0, places=6)

    def test_atr_includes_the_gap_to_the_previous_close_on_a_trending_series(self):
        # Rising 1.00 per bar: high-to-previous-close (1.5) is wider than the
        # bar's own range (1.0), so ATR must reflect the larger measure.
        highs = [x + 1.0 for x in range(30)]
        lows = [float(x) for x in range(30)]
        closes = [x + 0.5 for x in range(30)]
        result = atr(highs, lows, closes, 14)
        self.assertAlmostEqual(result[-1], 1.5, places=6)

    def test_atr_needs_more_than_period_bars(self):
        self.assertEqual(atr([1] * 5, [1] * 5, [1] * 5, 14), [None] * 5)


class TestSessionVwap(unittest.TestCase):
    def test_vwap_is_the_volume_weighted_typical_price(self):
        highs, lows, closes = [10, 20], [10, 20], [10, 20]
        result = session_vwap(highs, lows, closes, [1, 3], ["d", "d"])
        self.assertAlmostEqual(result[0], 10.0)
        self.assertAlmostEqual(result[1], (10 * 1 + 20 * 3) / 4)

    def test_vwap_resets_at_the_start_of_each_session(self):
        highs = lows = closes = [10, 10, 50]
        result = session_vwap(highs, lows, closes, [100, 100, 1], ["d1", "d1", "d2"])
        # The new session ignores the previous session's volume entirely.
        self.assertAlmostEqual(result[2], 50.0)

    def test_zero_volume_falls_back_to_the_unweighted_typical_price(self):
        result = session_vwap([12], [8], [10], [0], ["d"])
        self.assertAlmostEqual(result[0], typical_price(12, 8, 10))

    def test_negative_volume_is_clamped_rather_than_corrupting_the_average(self):
        result = session_vwap([10, 10], [10, 10], [10, 10], [5, -5], ["d", "d"])
        self.assertAlmostEqual(result[1], 10.0)
        self.assertFalse(any(math.isnan(v) for v in result))


class TestCrossDetection(unittest.TestCase):
    def test_cross_requires_moving_from_at_or_below_to_strictly_above(self):
        self.assertTrue(crossed_above(1.0, 1.0, 2.0, 1.0))
        self.assertTrue(crossed_above(0.9, 1.0, 1.1, 1.0))

    def test_no_cross_when_already_above(self):
        self.assertFalse(crossed_above(2.0, 1.0, 3.0, 1.0))

    def test_touching_without_exceeding_is_not_a_cross(self):
        self.assertFalse(crossed_above(0.9, 1.0, 1.0, 1.0))

    def test_missing_history_is_not_a_cross(self):
        self.assertFalse(crossed_above(None, 1.0, 2.0, 1.0))


if __name__ == "__main__":
    unittest.main()
