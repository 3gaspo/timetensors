"""Static contract checks for every TimeTensors Slurm family."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SlurmWorkflowTest(unittest.TestCase):
    def test_cluster_sync_scripts(self):
        code = (ROOT / "sync_code_to_selena.sh").read_text(encoding="utf-8")
        results = (ROOT / "sync_results_to_dgx.sh").read_text(encoding="utf-8")
        for script in (code, results):
            self.assertIn('PROJECT_NAME="$(basename "$PROJECT_ROOT")"', script)
            self.assertIn("sed -n '1p'", script)
        for excluded in (
            ".git/",
            ".venv/",
            ".secrets/",
            "pyproject.toml",
            "uv.lock",
            "datasets/",
            "weights/",
            "outputs/",
            "logs/",
        ):
            self.assertIn(f"--exclude='{excluded}'", code)
        self.assertIn("selena.hpc.edf.fr", code)
        self.assertIn("--delete", code)
        self.assertIn("dgx-front.retd.edf.fr", results)
        self.assertIn("--include='outputs_selena/.gitkeep'", code)
        self.assertIn("--exclude='outputs_selena/***'", code)
        self.assertIn("--include='logs_selena/.gitkeep'", code)
        self.assertIn("--exclude='logs_selena/***'", code)
        self.assertIn('"$SOURCE_ROOT/outputs_selena/"', results)
        self.assertIn('"$SOURCE_ROOT/logs_selena/"', results)
        self.assertNotIn("--delete", results)

    def test_fronts_use_standard_scale_and_stage_controls(self):
        fronts = sorted(ROOT.glob("[0-9][0-9]_*.slurm"))
        dgx_fronts = [path for path in fronts if "_selena" not in path.stem]
        selena_fronts = [path for path in fronts if "_selena" in path.stem]
        self.assertEqual(len(dgx_fronts), 8)
        self.assertEqual(len(selena_fronts), 8)
        self.assertEqual(
            {path.stem for path in selena_fronts},
            {f"{path.stem}_selena" for path in dgx_fronts},
        )
        self.assertTrue((ROOT / "publish_job.sh").is_file())
        for path in dgx_fronts:
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
        for path in selena_fronts:
            text = path.read_text(encoding="utf-8")
            expected_stages = (
                'STAGES="${STAGES:-evaluate,tables}"'
                if path.name == "08_foundation_models_selena.slurm"
                else 'STAGES="${STAGES:-train,tables}"'
            )
            self.assertIn(expected_stages, text)
            self.assertIn("#SBATCH --partition=an", text)
            self.assertIn("#SBATCH --output=logs_selena/%x_%j.out", text)
            self.assertIn("#SBATCH --exclusive", text)
            self.assertIn("#SBATCH --no-requeue", text)
            self.assertIn("#SBATCH --wckey=P12CU:DATASCIENCE", text)
            self.assertIn('OUTPUTS_ROOT="$PROJECT_ROOT/outputs_selena"', text)
            self.assertIn('LOGS_ROOT="$PROJECT_ROOT/logs_selena"', text)
            self.assertIn('EXPERIMENT_LAUNCH_ID="selena_${SLURM_JOB_ID', text)

    def test_internal_workflow_is_signature_aware(self):
        common = (ROOT / "src/slurm/benchmark_common.sh").read_text(encoding="utf-8")
        self.assertIn('LOGS_ROOT="${LOGS_ROOT:-$ROOT/logs}"', common)
        self.assertIn('OUTPUTS_ROOT="${OUTPUTS_ROOT:-$ROOT/outputs}"', common)
        self.assertIn('--root "$OUTPUTS_ROOT"', common)
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
        for family in (
            "constants",
            "sampling",
            "normalizations",
            "reference",
            "losses",
            "linear_models",
            "central_per_user",
            "foundation_models",
        ):
            source = (ROOT / "src/slurm" / f"benchmark_{family}.sh").read_text(
                encoding="utf-8"
            )
            self.assertIn(f'OUT_ROOT="${{OUT_ROOT:-$OUTPUTS_ROOT/{family}}}"', source)

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
