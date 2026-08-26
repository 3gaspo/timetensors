"""Static contract checks for every TimeTensors Slurm family."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SlurmWorkflowTest(unittest.TestCase):
    def test_fronts_use_standard_scale_and_stage_controls(self):
        fronts = sorted(ROOT.glob("[0-9][0-9]_*.slurm"))
        self.assertEqual(len(fronts), 8)
        self.assertTrue((ROOT / "publish_job.sh").is_file())
        for path in fronts:
            text = path.read_text(encoding="utf-8")
            self.assertIn('EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"', text)
            expected_stages = (
                'STAGES="${STAGES:-evaluate,tables}"'
                if path.name == "08_foundation_models.slurm"
                else 'STAGES="${STAGES:-train,tables}"'
            )
            self.assertIn(expected_stages, text)
            self.assertIn("#SBATCH --partition=h100", text)
            self.assertNotIn("#SBATCH --partition=a100", text)
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
                'SETTINGS_OVERRIDE:-168:24 336:48 504:168}'
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
        self.assertNotIn("run.complete", common)
        self.assertIn("python -m pipeline.runs allocate", common)
        self.assertIn("python -m pipeline.runs pending-seeds", common)
        self.assertIn("python -m pipeline.runs status", common)
        self.assertIn("--status ready", common)
        self.assertIn("python -m pipeline.runs ready", common)
        self.assertIn("python -m pipeline.runs complete-launch", common)
        self.assertIn("python -m pipeline.runs complete --run-dir", common)
        self.assertIn('srun --ntasks=1 python -m "$module"', common)
        self.assertIn("srun --ntasks=1 python -m scripts.report", common)
        self.assertNotIn("--status completed", common)
        self.assertIn("dataset_has_tensor_payload", common)
        self.assertIn("missing_tensor_payload", common)
        self.assertIn('"$ROOT/../datasets"', common)
        self.assertIn('"$ROOT/../weights"', common)
        self.assertIn('"$SHARED_ROOT/datasets"', common)
        self.assertIn('"$SHARED_ROOT/weights"', common)
        self.assertNotIn("BENCHMARK_PROFILE", common)
        table_stage = (ROOT / "src/slurm/stage_tables.sh").read_text(encoding="utf-8")
        self.assertNotIn("tables.complete", table_stage)
        self.assertTrue((ROOT / "src/slurm/stage_train.sh").is_file())

    def test_constants_policy_contract(self):
        constants = (ROOT / "src/slurm/benchmark_constants.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'DEFAULT_POLICIES="keep remove_train_windows remove_eval_windows '
            'remove_all_windows drop_all_users"',
            constants,
        )
        self.assertIn(
            'DEFAULT_POLICIES="keep remove_all_windows drop_all_users"', constants
        )
        self.assertNotIn("drop_train_users", constants)
        self.assertNotIn("drop_eval_users", constants)

    def test_foundation_models_share_one_runnable_front(self):
        foundation = (
            ROOT / "src/slurm/benchmark_foundation_models.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "chronos2 chronos_bolt ts_icl tabpfn_ts",
            foundation,
        )
        self.assertIn("# tirex2 remains adapter-supported", foundation)
        self.assertIn("+experiment.skip_training=true", foundation)
        self.assertIn('stage_evaluate.sh', foundation)
        self.assertTrue((ROOT / "08_foundation_models.slurm").is_file())
        for adapter in ("chronos2.py", "chronos_bolt.py", "ts_icl.py", "tirex2.py", "tabpfn.py"):
            source = (ROOT / "src/external_models" / adapter).read_text(
                encoding="utf-8"
            )
            self.assertIn('project.parent / "weights"', source)


if __name__ == "__main__":
    unittest.main()
