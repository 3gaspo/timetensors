"""Regression checks for complete synthetic benchmark-family coverage."""

import unittest
from pathlib import Path
import sys

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.synthetic_smoke import BENCHMARK_FAMILIES, EXPECTED_METHODS, SEEDS, run_benchmark


class SyntheticSmokeCoverageTest(unittest.TestCase):
    def test_every_benchmark_family_and_seed_is_present(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=(4, 360)).cumsum(axis=1)

        results, _ = run_benchmark(values)

        self.assertEqual(set(results["family"]), set(BENCHMARK_FAMILIES))
        for family in BENCHMARK_FAMILIES:
            family_results = results.loc[results["family"] == family]
            family_seeds = set(family_results["seed"])
            self.assertEqual(family_seeds, set(SEEDS), family)
            self.assertEqual(set(family_results["method"]), EXPECTED_METHODS[family], family)
            counts = family_results.groupby("method")["seed"].nunique()
            self.assertTrue((counts == len(SEEDS)).all(), family)


if __name__ == "__main__":
    unittest.main()
