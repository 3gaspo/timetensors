# Local Smoke Tests

These scripts exercise package components on tiny synthetic inputs. They are for
the local coding machine and do not submit Slurm jobs or run benchmarks.

Run from the repository root:

```bash
python src/tests/test_config_defaults.py
python src/tests/test_dataloaders.py
python src/tests/test_models.py
python src/tests/test_results_table.py
python src/tests/test_sklearn.py
python src/tests/test_slurm_workflow.py
python src/tests/test_synthetic_smoke.py
```

The scripts intentionally avoid large datasets, checkpoints, and long training
runs. Use them before moving changes into the remote cluster workflow.
