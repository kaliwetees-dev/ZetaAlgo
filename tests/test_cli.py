import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from zetaalgo.cli import main, parse_grid


def run_cli(*argv):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(list(argv))
    return code, buffer.getvalue()


class TestGridParsing(unittest.TestCase):
    def test_single_axis_expands_to_one_combination_per_value(self):
        self.assertEqual(
            parse_grid("target_r=1,2"),
            [{"target_r": "1"}, {"target_r": "2"}],
        )

    def test_two_axes_produce_the_cartesian_product(self):
        combos = parse_grid("a=1,2;b=3,4")
        self.assertEqual(len(combos), 4)
        self.assertIn({"a": "1", "b": "4"}, combos)

    def test_an_empty_spec_yields_the_defaults(self):
        self.assertEqual(parse_grid(""), [{}])

    def test_a_malformed_term_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_grid("targetr")


class TestCommands(unittest.TestCase):
    def test_run_prints_a_report(self):
        code, output = run_cli("run", "--synthetic", "12")
        self.assertEqual(code, 0)
        self.assertIn("PERFORMANCE", output)
        self.assertIn("SIGNAL FUNNEL", output)
        self.assertIn("SYNTHETIC", output)

    def test_run_warns_that_synthetic_results_prove_nothing(self):
        _, output = run_cli("run", "--synthetic", "10")
        self.assertIn("SYNTHETIC data", output)

    def test_run_emits_valid_json(self):
        code, output = run_cli("run", "--synthetic", "12", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertIn("metrics", payload)
        self.assertIn("win_rate", payload["metrics"])

    def test_run_writes_requested_csv_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            trades = os.path.join(tmp, "t.csv")
            equity = os.path.join(tmp, "e.csv")
            metrics = os.path.join(tmp, "m.csv")
            code, _ = run_cli("run", "--synthetic", "20", "--trades-csv", trades,
                              "--equity-csv", equity, "--metrics-csv", metrics)
            self.assertEqual(code, 0)
            for path in (trades, equity, metrics):
                self.assertTrue(os.path.exists(path), path)
                with open(path) as handle:
                    self.assertGreater(len(handle.readlines()), 1)

    def test_generate_then_run_round_trips_through_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bars.csv")
            code, _ = run_cli("generate", "--days", "10", "--out", path)
            self.assertEqual(code, 0)
            code, output = run_cli("run", "--csv", path, "--symbol", "ROUNDTRIP")
            self.assertEqual(code, 0)
            self.assertIn("ROUNDTRIP", output)
            # Real (file-based) data must not carry the synthetic disclaimer.
            self.assertNotIn("SYNTHETIC data", output)

    def test_paper_compare_reports_exact_agreement(self):
        code, output = run_cli("paper", "--synthetic", "12", "--compare")
        self.assertEqual(code, 0)
        self.assertIn("automation reproduces the backtest exactly", output)

    def test_sweep_ranks_every_combination(self):
        code, output = run_cli("sweep", "--synthetic", "25",
                               "--grid", "target_r=1.5,3", "--rank", "net_profit")
        self.assertEqual(code, 0)
        self.assertIn("parameter sweep: 2 combinations", output)

    def test_sweep_rejects_an_unknown_parameter(self):
        with self.assertRaises(SystemExit):
            run_cli("sweep", "--synthetic", "10", "--grid", "nonsense=1")

    def test_walkforward_reports_out_of_sample_scores(self):
        code, output = run_cli("walkforward", "--synthetic", "40", "--folds", "2",
                               "--grid", "target_r=1.5,3")
        self.assertEqual(code, 0)
        self.assertIn("out-of-sample mean", output)

    def test_walkforward_refuses_too_few_sessions(self):
        with self.assertRaises(SystemExit):
            run_cli("walkforward", "--synthetic", "4", "--folds", "8",
                    "--grid", "target_r=2")

    def test_a_missing_data_source_is_an_error(self):
        with self.assertRaises(SystemExit):
            run_cli("run")


if __name__ == "__main__":
    unittest.main()
