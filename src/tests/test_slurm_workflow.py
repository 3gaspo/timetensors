"""Static contract checks for every TimeTensors Slurm family."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SlurmWorkflowTest(unittest.TestCase):
    def test_fronts_use_standard_scale_and_stage_controls(self):
        fronts = sorted(ROOT.glob("*.slurm"))
        self.assertEqual(len(fronts), 7)
        for path in fronts:
            text = path.read_text(encoding="utf-8")
            self.assertIn('EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"', text)
            self.assertIn('STAGES="${STAGES:-train,tables}"', text)
            self.assertNotIn("BENCHMARK_PROFILE", text)
            self.assertNotIn("RUN_MODE", text)
            self.assertNotIn("TEST_MODE", text)

    def test_internal_workflow_is_signature_aware(self):
        common = (ROOT / "src/slurm/benchmark_common.sh").read_text(encoding="utf-8")
        for mode in ("test)", "full)", "ultra)"):
            self.assertIn(mode, common)
        self.assertNotIn("  small)", common)
        self.assertIn('SETTINGS_OVERRIDE:-504:168', common)
        self.assertEqual(
            common.count(
                'SETTINGS_OVERRIDE:-168:24 336:48 504:168 720:168 720:720}'
            ),
            2,
        )
        self.assertEqual(
            common.count(
                'DATASETS_OVERRIDE:-ETTh1 electricity traffic solar weather exchange_rate}'
            ),
            2,
        )
        self.assertEqual(common.count('SEEDS_OVERRIDE:-1 2 3}'), 2)
        self.assertNotIn('SEEDS_OVERRIDE:-1 2 3 4 5}', common)
        self.assertEqual(common.count('MODELS_OVERRIDE:-patchtst}'), 2)
        self.assertIn('MODELS_OVERRIDE:-patchtst dlinear}', common)
        self.assertIn("run.complete", common)
        self.assertNotIn("BENCHMARK_PROFILE", common)
        table_stage = (ROOT / "src/slurm/stage_tables.sh").read_text(encoding="utf-8")
        self.assertIn("tables.complete", table_stage)
        self.assertTrue((ROOT / "src/slurm/stage_train.sh").is_file())


if __name__ == "__main__":
    unittest.main()
