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
python -m visu.results_table outputs/reference --show-std
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
  +output.dir=outputs/manual_debug/electricity/168_24 +output.name=patchtst
```

`experiment.seeds` creates `seed_N/` subdirectories. The first seed may rebuild the tensor dataset; later seeds reuse it. Training history stores raw optimizer-step losses, interval-average train losses, and validation losses at `training.valid_eval_freq`. Set `training.plot_step_train_loss=false` for the clearer interval-train/validation plot.
This direct Hydra command is for debugging; table-eligible runs must be
allocated by a manifest-aware numbered Slurm workflow.

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
- Constant handling: remove individual constant windows independently in
  train/evaluation, or compare against dropping affected users from both.
- Losses: `mse`, `mae`, `nmse`, `nmae`, and `relative_mse`.
- Normalization: identity, global standard, global min-max, instance min-max, instance normalization/RevIN, and the research variants retained under `models/`.
- Normalization classes use the acronymic type names `RevIN` and `GRevIN`
  without a redundant `Normalization` suffix.
- Models: persistence and linear baselines, DLinear, PatchTST, Chronos, TabPFN, and sklearn linear regression.
- Training scope: `experiment.training_scope=central` or `per_user`.
- Evaluation performs one inference pass per configured run and stores each
  metric as one elementwise tensor aligned with compact user, query, and run
  ID tensors. Equal-user means and `w10_*`, the mean loss of the worst 10% of
  users, are derived from those rows without another inference pass.

Global standard and min-max statistics are computed from accessible training
lookbacks under the same cutoff and target-split rules. Final artifacts are
written below `<output.dir>/<output.name>/`, including `model_state.pt`,
`train_history.pt`, `criterion_loss.pdf`, `all_losses.pt`, and
optional `example_prediction.pdf`. Each split in `all_losses.pt` contains
`losses`, aligned `metadata`, and scalar `summaries`; per-user tensor copies are
not written.

## Slurm benchmarks

The jobs under `src/slurm/` cover:

- a reference order-of-error comparison between persistence, PatchTST, and Chronos-2;
- constant-window removal with a drop-all-affected-users comparison;
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

- `01_constants.slurm` -> `benchmark_constants.sh` compares keeping all pairs,
  train/evaluation/both window removal, and dropping all affected users.
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
168--24, 336--48, and 504--168; seeds 1--3; and PatchTST for
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

The full front runs `STAGES=train,tables`. Training allocates current-manifest
runs and executes only incomplete seeds; tables are lightweight and are rebuilt
from the selected completed manifests. `STAGES=train` and `STAGES=tables` are
recovery overrides, while `SKIP_COMPLETED=false` forces the exact selected
computation and retains its previous manifest. Resubmitting the same complete
front is therefore the normal recovery procedure after a time limit.

## Result identity and manifests

Every family has its own workflow root and ordered model configs:

| Root | Backbone | Ordered model-config folders |
|---|---|---|
| `outputs/constants` | selected forecaster | `policy` |
| `outputs/sampling` | selected forecaster | `sampling_mode/batch_size` |
| `outputs/normalizations` | selected forecaster | `normalization` |
| `outputs/losses` | selected forecaster | `loss` |
| `outputs/reference` | persistence/PatchTST/Chronos-2 | `normalization/loss` |
| `outputs/linear_models` | linear method | `normalization` |
| `outputs/central_per_user` | selected forecaster | `scope` |

Each continues as `dataset/L_H/backbone/<configs>/run_n/seed_n/`. Step budget,
optimizer, split, strides, plotting cadence, and other scientific execution
choices are pipeline configs in `manifest.json`; device and scheduler placement
are runtime configs. One seed fixes all stochasticity in that repetition.

Run identity contains only the manifest schema, ordered identity/model configs,
pipeline and experiment parameters, and seeds. Source files, Slurm fronts,
datasets, weights, logs, outputs, and directories are never fingerprinted or hashed.
Plain provenance paths may be recorded but do not affect reuse. Code and data
changes are manual rerun decisions; use `RUN_CONFLICT_POLICY=new` for another
repeat with unchanged parameters. Change `schema_version` only for a deliberate
global artifact-contract break.

The current `schema_version` is 2. Schema 1 runs use the replaced duplicated
per-user loss contract and must be rerun. Only completed manifests can enter a report.
The overall run remains `running` with `ready_at_utc` while finished seed
states are `ready`; completion is written immediately after that
configuration's producer process returns successfully with every required
artifact. Later configuration or table failures preserve completed runs and
interrupt only unfinished work. The completed manifest is authoritative and reuse does not hash
or revalidate synchronized files. `RUN_CONFLICT_POLICY=overwrite_exact`
skips identical completed runs, resumes identical interrupted runs, and creates
the next `run_n` for changed pipeline configs. `overwrite_path` and `new` are
explicit alternatives. Reports support the common distinct/latest/average
config policy and selected/latest/distinct/average repeat policy. Explicit
pipeline filters select a pipeline configuration and must match even with one
candidate. Exact-repeat selection is recorded in `SELECTED_RUNS.txt`, and every
`report_manifest.json` records requested filters and obtained inputs.

The former result trees could not be migrated because synchronized seed folders
lacked the excluded `all_losses.pt` payloads required to prove completion. They
are preserved under `outputs/archive/legacy_pre_schema_v1_2026-08-07/` and are
never consumed by current tables.

The later remote commit `99b4d80` was imported without merging its pre-schema
code under the commit-addressed `cluster_full_99b4d80/` archive directory. It
contains a partial full constants run (132 of 270 intended seed runs) and the
job-42527 log pair, but no sampling-family tree and no synchronized metric
payload from which to rebuild a policy table. The quantitative coverage and
failure analysis is retained in `latex/executive_summary.tex`.

The recommended execution order is:

1. Run one bare-launcher `test` path check.
2. Run `EXPERIMENT_MODE=full sbatch 01_constants.slurm` and compare the three
   window-removal scopes against keeping all pairs and dropping all affected users.
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
If a selected dataset has no `*values.pt` tensor payload, its first pending
configuration rebuilds the tensors automatically. `REBUILD_DATASETS=true`
forces a rebuild even when tensors exist; do not force concurrent rebuilds of
the same dataset.

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
`DROP_TRAIN_CONSTANT_USERS` and `DROP_EVAL_CONSTANT_USERS` overrides can disable
that behavior for sampling, normalization, loss, linear, and central/per-user
studies. Window removal is varied only by `01_constants.slurm`. The reference
job uses the corresponding `REFERENCE_DROP_*` overrides listed above.

Test/full/ultra benchmark tables are emitted once per selected model. The
linear-model and reference comparisons intentionally use combined tables.

Tables aggregate seed means and sample standard deviations, display values with
two decimals, and include an explicit per-row `\times 10^{m}` multiplier. A
single-seed test cell is valid but has no estimable sample deviation, so it is
shown without `±`. See `latex/experiment_guideline.tex` for the complete
protocol.

All benchmark launchers use the `h100` partition, one CPU per task, concise
one-word job names, and `logs/%x_%j.{out,err}` for Slurm output. Hydra logs to
stdout only so the Slurm files remain the canonical run logs. Every profile
remains sequential within one allocation; resubmit the same front to continue
from its completion markers.

## Publishing terminal Slurm artifacts

Slurm jobs never submit a publisher or run Git commands. After any job reaches
a terminal state, including failure, cancellation, or timeout, run the manual
publisher from that project's Git root:

```bash
bash publish_job.sh <job-id>
```

The script first verifies `main`, sources `$HOME/codes/proxy.sh`, and runs
`git pull --ff-only origin main`. With a job ID, it selects only the exact
`logs/*_<job-id>.out`/`.err` pair. It force-adds only those paths while excluding
`*.pt`, `*.npy`, and `*.cbm`, commits them, and pushes `origin main`. A
non-fast-forward pull stops without creating a merge commit, and the script
never creates a pull request. Existing unrelated staged paths are excluded from
the commit.

Omit the job ID to force-add, commit, and push the complete `logs/` and
lightweight `outputs/` trees:

```bash
bash publish_job.sh
```

`PROXY_SCRIPT_PATH` overrides the default `$HOME/codes/proxy.sh`. The publisher
sources that script once for both the pull and push and leaves the shell's
existing GitHub credential and askpass context untouched.

## Lightweight checks

With the project environment prepared:

```bash
python src/tests/test_config_defaults.py
python src/tests/test_dataloaders.py
python src/tests/test_models.py
python src/tests/test_results_table.py
python src/tests/test_sklearn.py
python src/tests/test_slurm_workflow.py
python src/tests/test_synthetic_smoke.py
```

## Synthetic smoke benchmark

`src/scripts/synthetic_smoke.py` loads the generator definitions and the
two-cluster population from `archive/synthetic_generator.ipynb`, writes a small
16-user dataset to `datasets/synthetic_smoke/`, and exercises all seven benchmark
families with NumPy forecasters for six epochs and seeds 1 and 2. Every family
has an explicit expected-method set, and the smoke test requires every expected
method to contain both completed seeds. The reference family uses persistence
plus explicitly named NumPy proxies for PatchTST and Chronos-2; it validates the
three-method workflow and reporting contract, not
the numerical behavior of those unavailable backbones. The runner is
dependency-light and is intended for local end-to-end checks when the PyTorch
`uv` environment is unavailable.

```bash
python src/scripts/synthetic_smoke.py
```

Seed-level manifests and artifacts use
`outputs/synthetic_smoke/<family>/synthetic_smoke/24_6/numpy_linear_proxy/<method>/run_n/seed_n/`;
aggregated reports use `outputs/reports/synthetic_smoke/`. This validates every
family's methods, both seeds, selection, and reporting path; the Slurm
benchmarks remain the authoritative DLinear/PatchTST experiments.

## Experiment protocol

`latex/experiment_guideline.tex` specifies the forecasting task, all seven
benchmark families, shared datasets and settings, artifact contracts, and
practical workflow. `latex/executive_summary.tex` records only completed and
analyzed results. Both compiled PDFs are kept beside their sources.

## Maintenance workflow

Every project change is recorded in `PENDING_UPDATES.md` with its scope,
affected contracts, focused checks already completed, deferred integration
coverage, documentation impact, and rerun requirements. Routine edits use only
the smallest relevant smoke check. Periodic maintenance verifies pending entries
against the implementation, runs complementary generic lightweight smoke tests,
reconciles this README and the project LaTeX documents, and renders affected
PDFs before resolving the entries.
