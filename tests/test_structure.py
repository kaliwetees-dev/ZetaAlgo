import unittest

from helpers import make_bars, zigzag
from zetaalgo.data import generate_synthetic
from zetaalgo.structure import (
    BEARISH,
    BULLISH,
    StructureTracker,
    fixed_range_profile,
    find_pivot,
    swing_points,
)


class TestPivots(unittest.TestCase):
    def test_a_peak_is_a_pivot_high(self):
        values = [1, 2, 5, 2, 1]
        self.assertTrue(find_pivot(values, 2, 2, 2, want_high=True))
        self.assertFalse(find_pivot(values, 2, 2, 2, want_high=False))

    def test_a_trough_is_a_pivot_low(self):
        values = [5, 4, 1, 4, 5]
        self.assertTrue(find_pivot(values, 2, 2, 2, want_high=False))

    def test_edges_cannot_be_pivots_without_enough_context(self):
        values = [1, 5, 1]
        self.assertFalse(find_pivot(values, 0, 2, 2, want_high=True))
        self.assertFalse(find_pivot(values, 2, 2, 2, want_high=True))

    def test_a_pivot_is_only_confirmed_after_its_right_bars(self):
        """The whole correctness of structure trading rests on this lag."""
        bars = generate_synthetic(days=4, seed=3)
        for right in (1, 2, 4):
            pivots = swing_points(bars, left=2, right=right)
            self.assertTrue(pivots)
            for pivot in pivots:
                self.assertEqual(pivot.confirmed_index, pivot.index + right)

    def test_a_monotonic_series_has_no_interior_pivots(self):
        bars = make_bars([100.0 + i for i in range(30)])
        highs = [p for p in swing_points(bars) if p.is_high]
        self.assertEqual(highs, [])


class TestChochVersusBos(unittest.TestCase):
    """The same break is a CHoCH or a BOS depending only on context."""

    def _run(self, closes):
        bars = make_bars(closes)
        tracker = StructureTracker(left=2, right=2)
        tracker.prepare(bars)
        events = []
        for i in range(len(bars)):
            events += tracker.on_bar_close(i)
        return events

    def test_a_reversal_after_a_downtrend_is_a_choch(self):
        # Lower highs and lower lows, then a push back up through the last
        # swing high: that break is a change of character.
        closes = zigzag([100, 92, 96, 88, 92, 84, 88, 102])
        events = self._run(closes)
        kinds = [(e.kind, e.direction) for e in events]
        self.assertIn(BEARISH, [d for _, d in kinds])
        self.assertIn(("CHoCH", BULLISH), kinds)

    def test_continuation_in_the_same_direction_is_a_bos(self):
        # A staircase down: each new low breaks the previous one.
        events = self._run(zigzag([100, 92, 96, 88, 92, 84, 88, 80]))
        self.assertTrue(events)
        self.assertTrue(all(e.direction == BEARISH for e in events), events)
        self.assertTrue(all(e.kind == "BOS" for e in events), events)

    def test_the_first_break_only_establishes_bias(self):
        events = self._run(zigzag([100, 92, 96, 88, 92, 84]))
        self.assertTrue(events)
        self.assertEqual(events[0].kind, "BOS")

    def test_a_market_that_never_retraces_has_no_structure_to_break(self):
        """A monotonic decline makes no swing lows, so nothing can break."""
        self.assertEqual(self._run([100 - i * 0.8 for i in range(40)]), [])

    def test_bias_flips_on_a_choch(self):
        bars = make_bars(zigzag([100, 92, 96, 88, 92, 84, 88, 104]))
        tracker = StructureTracker(left=2, right=2)
        tracker.prepare(bars)
        for i in range(len(bars)):
            tracker.on_bar_close(i)
        self.assertEqual(tracker.bias, BULLISH)


class TestVolumeProfile(unittest.TestCase):
    def test_poc_lands_where_the_volume_is(self):
        # One bar carries far more volume than its neighbours.
        bars = (make_bars([100.0] * 3, volume=10)
                + make_bars([110.0] * 1, volume=100_000)
                + make_bars([100.0] * 3, volume=10))
        profile = fixed_range_profile(bars, 0, len(bars) - 1, bins=20)
        self.assertIsNotNone(profile)
        self.assertGreater(profile.poc_price, 105.0)

    def test_poc_always_sits_inside_the_profiled_range(self):
        bars = generate_synthetic(days=3, seed=8)
        profile = fixed_range_profile(bars, 0, 100, bins=30)
        self.assertLessEqual(profile.low, profile.poc_price)
        self.assertLessEqual(profile.poc_price, profile.high)

    def test_value_area_brackets_the_poc(self):
        bars = generate_synthetic(days=3, seed=8)
        profile = fixed_range_profile(bars, 0, 100, bins=30)
        self.assertLessEqual(profile.value_area_low, profile.poc_price)
        self.assertLessEqual(profile.poc_price, profile.value_area_high)

    def test_value_area_captures_roughly_the_requested_share(self):
        bars = generate_synthetic(days=3, seed=8)
        profile = fixed_range_profile(bars, 0, 150, bins=40, value_area_pct=0.70)
        low_bin = int((profile.value_area_low - profile.low)
                      / ((profile.high - profile.low) / profile.bins) + 1e-9)
        high_bin = int((profile.value_area_high - profile.low)
                       / ((profile.high - profile.low) / profile.bins) - 1e-9)
        captured = sum(profile.volumes[low_bin : high_bin + 1])
        self.assertGreaterEqual(captured / profile.total_volume, 0.68)

    def test_a_flat_range_has_no_profile(self):
        bars = make_bars([100.0] * 5, pad=0.0)
        self.assertIsNone(fixed_range_profile(bars, 0, 4))

    def test_zero_volume_still_yields_a_level(self):
        bars = make_bars([100.0, 101.0, 102.0], volume=0.0)
        profile = fixed_range_profile(bars, 0, 2)
        self.assertIsNotNone(profile)
        self.assertLessEqual(profile.low, profile.poc_price)

    def test_out_of_range_slices_are_rejected(self):
        bars = generate_synthetic(days=1, seed=1)
        self.assertIsNone(fixed_range_profile(bars, 5, 2))
        self.assertIsNone(fixed_range_profile(bars, 0, len(bars) + 5))


if __name__ == "__main__":
    unittest.main()
