# TimeTensors

TimeTensors is the reference trainable forecasting benchmark for this thesis.
It measures how data filtering, sampling, normalization, training loss, model
family, and centralized versus per-user fitting affect accuracy, convergence,
and population-level dispersion.

Each numbered workflow changes one declared factor while retaining common
chronological splits, seed ownership, evaluation rows, and reporting rules.
Frozen foundation models are evaluated in a separate inference-only family.

## Documentation map

| Need | Document |
|---|---|
| Formal benchmark task and controlled factors | [`latex/method_overview.pdf`](latex/method_overview.pdf) |
| Prepared-data, model, training, and reporting flow | [`docs/architecture.md`](docs/architecture.md) |
| Numbered experiment families and order | [`docs/experiment_catalog.md`](docs/experiment_catalog.md) |
| Finalized evidence and full restart scope | [`docs/results_recap.md`](docs/results_recap.md) |
| Complete reproducibility specification | [`latex/experiment_guideline.pdf`](latex/experiment_guideline.pdf) |
| Full historical and analyzed evidence record | [`latex/executive_summary.pdf`](latex/executive_summary.pdf) |

## Setup

Use the project-managed environment from the repository root:

```bash
uv sync
export PYTHONPATH=src
```

Raw wide CSVs belong under `datasets/<name>/`; prepared variants are created
under `datasets/prepared/<dataset>/<signature>/`. Foundation checkpoints
belong under `weights/`. An adjacent `config.json` owns portable target,
exclusion, date, orientation, rename, and aggregation settings.

## Main executions

Run the numbered families in order because later reference choices depend on
earlier controls:

```bash
EXPERIMENT_MODE=test sbatch 01_constants.slurm
EXPERIMENT_MODE=full sbatch 01_constants.slurm
EXPERIMENT_MODE=full sbatch 02_sampling.slurm
EXPERIMENT_MODE=full sbatch 03_normalizations.slurm
EXPERIMENT_MODE=full sbatch 04_reference.slurm
EXPERIMENT_MODE=full sbatch 05_losses.slurm
EXPERIMENT_MODE=full sbatch 06_linear_models.slurm
EXPERIMENT_MODE=full sbatch 07_central_per_user.slurm
EXPERIMENT_MODE=test sbatch 08_foundation_models.slurm
EXPERIMENT_MODE=full sbatch 08_foundation_models.slurm
```

`test` is the narrow Electricity `504:168`, seed-1 gate. `full` is the primary
six-dataset, three-setting PatchTST study with seeds 1--3; `ultra` adds
DLinear where the family shares a trainable backbone axis. Exact treatments
and dependencies are in the [experiment catalog](docs/experiment_catalog.md).

Every front defaults to `STAGES=train,tables`; a stage subset is a recovery
override. Matching `*_selena.slurm` fronts execute the same family, for example:

```bash
EXPERIMENT_MODE=test sbatch 01_constants_selena.slurm
```

## Outputs and cluster operations

- Prepared tensors: `datasets/prepared/<dataset>/<signature>/`.
- Family runs: `outputs/<family>/dataset/L_H/backbone/.../run_n/seed_n/`.
- Publishable tables: `outputs/reports/<family>/<mode>/`.
- Runtime streams: `logs/` or `logs_selena/`.

Preview and then mirror maintained code from DGX:

```bash
bash sync_code_to_selena.sh --dry-run
bash sync_code_to_selena.sh
```

The preview marks stale maintained files with `*deleting`. The real transfer
delays deletion and protects excluded environments, dependency manifests,
datasets, weights, outputs, and logs.

Pull Selena artifacts from DGX at the smallest useful tier:

```bash
bash sync_results_to_dgx.sh
bash sync_results_to_dgx.sh --size detailed
bash sync_results_to_dgx.sh --size full
```

The default retrieves logs and aggregate reports. `detailed` adds non-binary
runs and diagnostics; `full` adds binary recovery payloads. Use
`bash publish_job.sh <job-id>` for one terminal log pair or
`bash publish_job.sh` for all logs plus aggregate reports.

## Documentation maintenance

```bash
PYTHONPATH=src python -m scripts.build_docs
PYTHONPATH=src python -m scripts.build_docs --render method
PYTHONPATH=src python -m scripts.build_docs --render all
```

The default validates the documentation map and all eight DGX fronts. The
method note owns the formal benchmark, architecture owns implementation,
the catalog owns planned families, and the recap plus executive summary own
analyzed evidence.
