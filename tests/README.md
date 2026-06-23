# Local Smoke Tests

These scripts exercise package components on tiny synthetic inputs. They are for
the local coding machine and do not submit Slurm jobs or run benchmarks.

Run from the repository root:

```bash
python tests/test_config_defaults.py
python tests/test_dataloaders.py
python tests/test_models.py
python tests/test_results_table.py
```

The scripts intentionally avoid large datasets, checkpoints, and long training
runs. Use them before moving changes into the remote cluster workflow.
