# TimeTensors

TimeTensors is a time-series forecasting benchmark focused on how experimental parameterization affects performance and convergence. It supports tensorized multivariate datasets, multiple sampling policies, constant-window filtering, PyTorch and scikit-learn models, normalization and loss variants, central/per-user training, seed aggregation, and publication-ready LaTeX tables.

## Layout

```text
src/
  conf/       Hydra configuration
  dataset/    CSV/tensor loading, splits, sampling, and statistics
  models/     forecasting models and normalization layers
  training/   losses, PyTorch/sklearn training, evaluation, per-user runs
  scripts/    runnable Hydra entrypoints
  slurm/      benchmark implementations (`.sh`)
  visu/       plots, notebooks, dashboards, and result tables
  tests/      lightweight smoke tests
datasets/     remote dataset payloads
weights/      remote pretrained weights
outputs/      generated models, metrics, figures, and tables
logs/         runtime and Slurm logs
latex/        experiment protocol
01_*.slurm ... 07_*.slurm   ordered submission files
```

All Python commands below are run from the repository root with `PYTHONPATH=src`.

## Entry points

```bash
python -m scripts.experiment       # dataset, training, and evaluation
python -m scripts.load_dataset     # dataset stage only
python -m scripts.train            # PyTorch training only
python -m scripts.evaluate         # evaluation only
python -m scripts.train_sklearn    # sklearn linear regression
python -m visu.results_table outputs/results --show-std
```

Hydra accepts the experiment sections `data`, `task`, `model`, `normalization`, `training`, `evaluation`, `experiment`, and `output`. A typical run is:

```bash
python -m scripts.experiment \
  +data.raw_path=datasets/electricity +data.path=datasets/electricity \
  +data.name=electricity +task.lags=168 +task.horizon=24 \
  +model.name=patchtst +model.path=patchtst \
  +normalization.name=instance +training.loss=nmse \
  +training.lr=1e-5 +training.batch_size=256 +training.epochs=10000 \
  +training.valid_eval_freq=1000 +training.logging_eval_freq=1000 \
  +experiment.seeds='[1,2,3]' \
  +output.dir=outputs/results/electricity/168_24 +output.name=patchtst
```

`experiment.seeds` creates `seed_N/` subdirectories. The first seed may rebuild the tensor dataset; later seeds reuse it. Training history stores raw optimizer-step losses, interval-average train losses, and validation losses at `training.valid_eval_freq`. Set `training.plot_step_train_loss=false` for the clearer interval-train/validation plot.

## Experiment controls

- Window anchor: every sampled date `t` is the last observed date, with
  `X_t = X(t-L:t] = {x_(t-L+1), ..., x_t}` and
  `Y_t = X(t:t+H] = {x_(t+1), ..., x_(t+H)}`. Date splits own target dates:
  a full horizon must stay inside its split, while its lookback may cross the
  preceding boundary. Evaluation target dates therefore do not move when `L`
  changes. Batch metadata records `query_indices`/`query_ids`; the legacy
  `date_indices`/`date_ids` keys are cutoff-date aliases.
- Sampling modes: `random`, `dates`, `individuals`, and `all`; train and evaluation strides are independent.
- Saved date/pair subset specifications carry `date_anchor=query_t`; legacy
  start-index subset files are rejected and must be regenerated.
- Constant handling: remove individual constant windows independently in train/evaluation, or drop users containing an accessible constant/non-finite lookback independently in train/evaluation.
- Losses: `mse`, `mae`, `nmse`, `nmae`, and `relative_mse`.
- Normalization: identity, global standard, global min-max, instance min-max, instance normalization/RevIN, and the research variants retained under `models/`.
- Models: persistence and linear baselines, DLinear, PatchTST, Chronos, TabPFN, and sklearn linear regression.
- Training scope: `experiment.training_scope=central` or `per_user`.
- Per-user evaluation saves equal-user means and `w10_*`, the mean loss of the worst 10% of users.

Global standard and min-max statistics are computed from accessible training
lookbacks under the same cutoff and target-split rules. Final artifacts are
written below `<output.dir>/<output.name>/`, including `model_state.pt`,
`train_history.pt`, `criterion_loss.pdf`, `all_losses.pt`, and
`per_user_all_losses.pt`.

## Slurm benchmarks

The jobs under `src/slurm/` cover:

- a reference order-of-error comparison between persistence, PatchTST, and Chronos-2;
- constant-window and constant-user removal;
- sampling mode and batch size;
- training losses, including relative MSE;
- normalization methods, including global min-max;
- linear and sklearn baselines;
- central versus per-user PatchTST and Chronos with W10 metrics.

Submit only the numbered `.slurm` files in the project root. Their names show
the recommended order. Every front is a complete resumable workflow with
`EXPERIMENT_MODE=test|full|ultra` and default
`STAGES=train,tables`. Family orchestrators and the shared stage implementations
remain under `src/slurm/`:

- `01_constants.slurm` -> `benchmark_constants.sh` compares constant-window/user policies.
- `02_sampling.slurm` -> `benchmark_sampling.sh` compares sampling modes and batch sizes.
- `03_normalizations.slurm` -> `benchmark_normalizations.sh` compares normalization parameterizations.
- `04_reference.slurm` -> `benchmark_reference.sh` establishes persistence, PatchTST, and Chronos-2
  orders of error.
- `05_losses.slurm` -> `benchmark_losses.sh` compares training losses under a fixed normalization.
- `06_linear_models.slurm` -> `benchmark_linear_models.sh` compares trainable/closed-form linear models and
  saves coefficient plots.
- `07_central_per_user.slurm` -> `benchmark_central_per_user.sh` compares centralized and per-user training.
- `benchmark_common.sh` resolves resources, applies the common split/stride and
  training controls, and launches exactly one Python task. `stage_train.sh`
  and `stage_tables.sh` execute the separate stages. These internal scripts are
  sourced by the family workflows and are not submitted directly.

Each configuration log includes the full model-specific Hydra overrides, so
otherwise similar rows can be distinguished without consulting the output
directory.

The launchers have three scale profiles. `test` is the safe default and narrows the
common axes to one Electricity/504--168/seed-1 path check; each family still
runs its own narrow method list (for example, the reference job keeps
persistence, PatchTST, and Chronos-2). It uses 20 optimizer steps with
validation/logging every 10 steps. `full` is the primary methodology grid:
ETTh1, Electricity, Traffic, Solar, Weather, and Exchange Rate;
168--24, 336--48, 504--168, 720--168, and 720--720; seeds 1--3; and PatchTST for
families that use the shared model axis. `ultra` uses the full axes and adds
DLinear where the family uses the shared model axis. Full and ultra
use 10,000 optimizer steps with validation/logging every 1,000 steps. Every
profile uses learning rate `1e-5`, batch size 256, random sampling, and
evaluation stride equal to the horizon.

The reference and linear families have their own fixed model/method axes rather
than the shared DLinear/PatchTST axis. Central-versus-per-user uses PatchTST in
test and full; ultra adds Chronos.

Every benchmark job exposes the same quick check:

```bash
EXPERIMENT_MODE=test sbatch 05_losses.slurm
EXPERIMENT_MODE=full sbatch 05_losses.slurm
EXPERIMENT_MODE=ultra sbatch 05_losses.slurm
```

Use `EXPERIMENT_MODE=full`, then `ultra` after the test profile.
Every sweep can be narrowed without editing a launcher through
`DATASETS_OVERRIDE`, `SETTINGS_OVERRIDE`, `SEEDS_OVERRIDE`, and
`MODELS_OVERRIDE`; values are whitespace-separated. Family-specific overrides
include `POLICIES_OVERRIDE`, `SAMPLING_CASES_OVERRIDE` (space-separated
`mode:batch_size` pairs), `SAMPLING_MODES_OVERRIDE`,
`BATCH_SIZES_OVERRIDE`, `LOSSES_OVERRIDE`, `NORMS_OVERRIDE`,
`LINEAR_METHODS_OVERRIDE`, and `LINEAR_NORMS_OVERRIDE`.

The full front runs `STAGES=train,tables`. The training stage writes one
signature-matched `run.complete` beside each seed's `all_losses.pt` and a
grid-level `.workflow/train.complete`; it skips only exact completed work. The table stage first requires every
selected dataset/setting/method/seed, then writes
`.workflow/tables.complete`; it is skipped when its signature still matches and
no training completion marker is newer. Outputs created before this completion
contract must be rerun once. `STAGES=train` and `STAGES=tables` are recovery
overrides, while `SKIP_COMPLETED=false` forces selected work. Resubmitting the
same complete front is therefore the normal recovery procedure after a time
limit.

The recommended execution order is:

1. Run one bare-launcher `test` path check.
2. Run `EXPERIMENT_MODE=full sbatch 01_constants.slurm`, then retain the
   selected constant-user policy.
3. Run `02_sampling.slurm` with `EXPERIMENT_MODE=full` for all four sampling
   modes and batch sizes 64/256/1024.
4. Run `03_normalizations.slurm` with `EXPERIMENT_MODE=full`.
5. Run `04_reference.slurm` with `EXPERIMENT_MODE=full` for
   persistence/PatchTST/Chronos-2 orders of error under the retained choices.
6. Run `05_losses.slurm` with `EXPERIMENT_MODE=full`.
7. Run `06_linear_models.slurm`, followed by `07_central_per_user.slurm`, with
   `EXPERIMENT_MODE=full`.

The reference benchmark uses non-trainable persistence, instance-normalized
nMSE PatchTST with constant users removed, and frozen Chronos-2. The linear
benchmark plots learned coefficients separately for identity, global standard,
and instance normalization. Its normalization list is configurable through
`LINEAR_NORMS_OVERRIDE`. If the preceding controls select different choices,
pass them through `REFERENCE_PATCHTST_NORM`, `REFERENCE_LOSS`,
`REFERENCE_DROP_TRAIN_CONSTANT_USERS`, and
`REFERENCE_DROP_EVAL_CONSTANT_USERS`.

Dataset and weight roots are resolved in this order: an explicit `DATA_ROOT`
or `WEIGHTS_ROOT`, a non-empty project-local directory, a non-empty parent
directory, then one additional shared-parent candidate. When this repository
is copied elsewhere, set the two environment variables explicitly.
Set `REBUILD_DATASETS=true` only on the first job that prepares a clean dataset
root; dependent jobs should reuse it rather than rebuilding concurrently.

When tensors are rebuilt, `config.json` is discovered beside `data.raw_path`
(whether that value is a dataset directory or a CSV). `data.config_path` may
instead name an explicit JSON file or directory. Portable loading fields live at
the top level and TimeTensors-only overrides under `timetensors`; scoped and
run values override other fields, while every `drop_users` list is merged
additively. The selected path and applied keys are timestamped in the job log.
ETTh1 is evaluated with every non-date variable, so its source CSV must contain
all seven variables rather than only `OT`.

All non-filtering launchers default to dropping users with accessible constant
look-backs in both training and evaluation. The shared
`DROP_TRAIN_CONSTANT_USERS` and `DROP_EVAL_CONSTANT_USERS` overrides carry a
different decision from `01_constants.slurm` into sampling,
normalization, loss, linear, and central/per-user studies. The reference job
uses the corresponding `REFERENCE_DROP_*` overrides listed above.

Test/full/ultra benchmark tables are emitted once per selected model. The
linear-model and reference comparisons intentionally use combined tables.

Tables aggregate seed means and sample standard deviations, display values with
two decimals, and include an explicit per-row `\times 10^{m}` multiplier. A
single-seed test cell is valid but has no estimable sample deviation, so it is
shown without `±`. See `latex/benchmark_experiments.tex` for the complete
protocol.

All benchmark launchers use the `a100` partition, one CPU per task, concise
one-word job names, and `logs/%x_%j.{out,err}` for Slurm output. Hydra logs to
stdout only so the Slurm files remain the canonical run logs. Every profile
remains sequential within one allocation; resubmit the same front to continue
from its completion markers.

## Lightweight checks

With the project environment prepared:

```bash
python src/tests/test_config_defaults.py
python src/tests/test_dataloaders.py
python src/tests/test_models.py
python src/tests/test_results_table.py
python src/tests/test_sklearn.py
python src/tests/test_slurm_workflow.py
```

## Synthetic smoke benchmark

`src/scripts/synthetic_smoke.py` loads the generator definitions and the
two-cluster population from `archive/synthetic_generator.ipynb`, writes a small
16-user dataset to `datasets/synthetic_smoke/`, and exercises all six benchmark
families with a NumPy linear forecaster for six epochs and seeds 1 and 2. It is
dependency-light and is intended for local end-to-end checks when the PyTorch
`uv` environment is unavailable.

```bash
python src/scripts/synthetic_smoke.py
```

Seed-level results, aggregated CSV/Markdown/LaTeX tables, training history, and
SVG plots are written to `outputs/synthetic_smoke/`. This smoke runner validates
the experiment controls and reporting path; the Slurm benchmarks remain the
authoritative DLinear/PatchTST experiments.

## Experiment protocol

`latex/benchmark_experiments.tex` is the single LaTeX source for the forecasting
task, all seven benchmark families, the shared datasets and settings, and the
main implementation classes. Its compiled PDF is written to
`outputs/pdf/benchmark_experiments.pdf`.
