# timetensors

Time series forecasting experiments built around Hydra overrides, PyTorch
models, tensorized datasets, and Slurm benchmark scripts.

## Entrypoints

Run from the repository root so relative paths in the Slurm scripts resolve
against this folder.

```bash
python -m timetensors.experiment  # dataset build, train, and eval orchestration
python -m timetensors.load_dataset # build/load tensor dataset artifacts only
python -m timetensors.train_model  # train only
python -m timetensors.eval_model   # evaluate only
```

Example:

```bash
python -m timetensors.experiment \
  +data.raw_path="../datasets/electricity" \
  +data.path="../datasets/electricity" \
  +data.name=electricity \
  +experiment.rebuild_dataset=true \
  +task.lags=168 \
  +task.horizon=24 \
  +model.name=dlinear \
  +model.path=dlinear \
  +normalization.name=instance \
  +training.bs=128 \
  +training.epochs=50 \
  +output.dir=outputs/timetensor_models/electricity/168_24 \
  +output.name=dlinear
```

## Config Reference

Hydra is used with no config file by default, so most runs pass values as
`+section.key=value` overrides. Nested dictionaries can be passed with dot
paths, for example `+data.sampling.eval_stride=24`.

### `data`

Dataset input/output and tensor-building options.

| Key | Default | Description |
| --- | --- | --- |
| `data.path` | `run/dataset` | Directory containing tensor artifacts such as `values.pt`; also the output directory when rebuilding. |
| `data.built_path` | unset | Alias with priority over `data.path` for loading built tensors. |
| `data.raw_path` | `data.path` | Directory containing `<data.name>.csv` when rebuilding. |
| `data.source_path` | unset | Alias for `data.raw_path`. |
| `data.name` | required for rebuild | CSV stem used as `<data.raw_path>/<data.name>.csv`. |
| `data.dataset` | unset | Alias for `data.name`. |
| `data.prefix` | `""` | Prefix for tensor files, producing `<prefix>_values.pt`, etc. |
| `data.legacy_context_kind` | unset | Interpret legacy `context.pt` as `individual` or `global`. |
| `data.save_loaded_copy` | `false` | Re-save loaded tensors to `data.path`. |
| `data.prepare_loaders` | `true` | Build train/eval loaders during dataset stage. |

CSV-building options used only when `experiment.rebuild_dataset=true`:

| Key | Default | Description |
| --- | --- | --- |
| `data.global_context_cols` | unset | CSV columns moved into global context. |
| `data.context_cols` | unset | Alias for `data.global_context_cols`. |
| `data.drop_users` | unset | Series/columns to drop. |
| `data.drop` | unset | Alias/general drop list. |
| `data.build_individual_ids_context` | `false` | Add individual IDs as individual context. |
| `data.rename_cols` | unset | Mapping of source column names to display names. |
| `data.aggr` | unset | Resample aggregation: `sum`, `mean`, `last`, `first`, or `asfreq`. |
| `data.aggregation` | unset | Alias for `data.aggr`. |
| `data.aggr_period` | `h` | Pandas resampling period. |
| `data.users_dim` | `1` | `1` means series are columns; `0` means series are rows. |
| `data.date_col` | unset | CSV date column to parse and use as index. |
| `data.dates` | unset | Explicit dates replacing the CSV index. |

### `data.splits` or top-level `splits`

Controls train/validation/test temporal and individual splits.

| Key | Default | Description |
| --- | --- | --- |
| `date_splits` | `[0.6,0.2,0.2]` | Temporal split ratios; one to three positive values summing to 1. |
| `indiv_split` | `1.0` | Fraction of individuals kept per split. |
| `reshuffle` | `true` | Reshuffle generated split assignments. |
| `shuffle_individuals` | `true` | Randomize individual split membership. |
| `by_cluster` / `by_group` | unset | Per-cluster/per-group overrides. |

### `data.sampling` or top-level `sampling`

Controls how dataset items are sampled from split tensors.

| Key | Default | Description |
| --- | --- | --- |
| `train_idx_mode` | `random` | Train sampler mode. |
| `eval_idx_mode` | `all` | Eval sampler mode. |
| `train_stride` | `1` | Step between train candidate start dates. |
| `eval_stride` | `1` | Step between eval candidate start dates. |
| `shuffle_train` | `true` | Shuffle train loader. |
| `shuffle_eval` | `false` | Shuffle eval loaders. |
| `remove_train_cte` | `false` | Remove constant train windows. |
| `remove_eval_cte` | `false` | Remove constant eval windows. |
| `train_block_individuals` | `1` | Individuals returned in each train item. |
| `eval_block_individuals` | `1` | Individuals returned in each eval item. |
| `train_len_multiplier` | `1` | Multiplier for train sampler length. |
| `eval_len_multiplier` | `1` | Multiplier for eval sampler length. |
| `use_context` | `true` | Legacy switch for both context types. |
| `use_individual_context` | `true` | Include individual context in batches. |
| `use_global_context` | `true` | Include global context in batches. |

Valid sampler modes are `random`, `dates`, `individuals`, and `all`.

### `data.subsets` or top-level `subsets`

Optional subsampling on top of train/eval splits.

| Key | Default | Description |
| --- | --- | --- |
| `mode` | sampler mode | Subset mode for all splits. |
| `modes.<split>` | unset | Per-split subset mode. |
| `sizes.<split>` | `1.0` | Ratio of candidates to keep for a split. |
| `specs.<split>` | unset | Precomputed indices or `{mode, indices, stride}`. |
| `by_cluster` / `by_group` | unset | Per-cluster/per-group overrides. |

Valid subset modes are `dates`, `individuals`, and `all`.

### `task`

| Key | Default | Description |
| --- | --- | --- |
| `task.lags` | `168` | Lookback/context length. |
| `task.horizon` | `24` | Forecast horizon. |

### `training`

| Key | Default | Description |
| --- | --- | --- |
| `training.bs` | `32` | Batch size alias. |
| `training.batch_size` | `32` | Batch size. |
| `training.epochs` | `1` | Training epochs; `0` skips optimization but still saves/evaluates. |
| `training.loss` | `MSE` | Training loss config or alias. |
| `training.complete_evaluation` | `true` | Include extra eval metrics beyond `MSE` and `nMSE`. |
| `training.lr` | `1e-3` | Learning rate. |
| `training.optimizer` | `adam` | Optimizer name. |
| `training.optimizer_kwargs` | `{}` | Extra optimizer kwargs. |
| `training.grad_clip` | unset | Max gradient norm. |
| `training.eval_freq` | unset | Validation frequency in epochs. |
| `training.device` | `auto` | Device selector: `auto`, `gpu`, `cuda`, or `cpu`. |
| `training.eval_runs` | `1` | Fallback for `evaluation.runs`. |
| `training.pretrained_path` | unset | Model state dict path. |
| `training.init` | unset | Alias for `training.pretrained_path`. |

Optimizers: `adam`, `adamw`, `sgd`, `rmsprop`.

Loss aliases: `MSE`, `MAE`, `nMSE`, `nMAE`, `rMSE`.
Loss dictionaries may use `name`, `base`, `loss`, `scaling`, `mode`,
`normalization`, `reduction`, `eps`, and `kwargs`. Base losses are `mse`/`l2`
and `mae`/`l1`; scalings are `normal`/`instance`/`std` and
`relative_mean`/`rmean`.

### `model`

| Key | Default | Description |
| --- | --- | --- |
| `model.name` | `model.path` or `linear` | Run/model name. |
| `model.path` | `model.name` | Built-in model key, import path, `module:attr`, or `.py` file. |
| `model.specs` | unset | Path to a model YAML file; bypasses inline model fields. |
| `model.class` | unset | Class name when loading from a Python file. |
| `model.kwargs` | `{}` | Constructor kwargs. |
| `model.configs` | `{}` | Alias for `model.kwargs`. |
| `model.state_dict_path` | unset | State dict to load. |
| `model.repeat_constant` | `false` | Repeat last value for constant lookbacks. |
| `model.covariate_augmentation` | unset | Covariate augmentation config. |
| `model.augmentation` | unset | Alias for `model.covariate_augmentation`. |

Built-in model keys:

`persistence`, `expected`, `repeat`, `lookback`, `linear`,
`periodic_linear`, `period`, `dlinear`, `DLinear`, `patchtst`, `PatchTST`.

Inline model kwargs automatically receive `lags`, `dim`, and `horizon` where
needed.

### `model.covariate_augmentation`

| Key | Default | Description |
| --- | --- | --- |
| `mode` / `modes` | none | Augmentation modes. A dash-separated string is accepted. |
| `kwargs.noise_scale` | `1.0` | Scale for `noise` mode. |
| `kwargs.constant_value` | `1.0` | Value for `constant` mode. |
| `kwargs.kernel_size` | `5` | Odd smoothing kernel size for `kernel` mode. |
| `kwargs.eps` | `1e-8` | Numerical epsilon. |

Modes: `identity`, `square`, `root`, `sign`, `mirror`, `kernel`, `noise`,
`constant`.

### `normalization`

| Key | Default | Description |
| --- | --- | --- |
| `normalization.name` | `identity` | Normalization applied inside `TimeTensorModel`. |
| `normalization.kwargs` | `{}` | Constructor kwargs for the normalization. |
| `normalization.configs` | `{}` | Alias for `normalization.kwargs`. |

Valid normalization names and aliases:

`none`, `identity`, `default`, `standard`, `standardnorm`, `standard_norm`,
`zscore`, `minmax`, `min_max`, `imm`, `instance_minmax`,
`instance_min_max`, `instance`, `instancenorm`, `instance_norm`, `revin`,
`reversible_instance_norm`, `revin_last`, `last_revin`, `revin_arcsinh`,
`revin_asinh`, `revin_arcsinsh`, `revin_arcinsh`, `sigmoid`, `logistic`,
`tanh`, `hyperbolic_tangent`, `relative_mean`, `rmean`, `rms`, `rms_norm`,
`rmsnorm`.

Common kwargs include `eps`, `detach_stats`, `mean`, `std`, `min_value`,
`max_value`, `dim`, `affine`, and `center`.

### `experiment`

| Key | Default | Description |
| --- | --- | --- |
| `experiment.rebuild_dataset` | `false` | Build CSV/raw data into tensor artifacts before training/eval. |
| `experiment.prepare_loaders` | `data.prepare_loaders` or `true` | Build loaders during dataset stage. |
| `experiment.evaluate` | `true` | Run evaluation after training/skipping training. |
| `experiment.skip_training` | `false` | Skip training stage. |
| `experiment.bypass_training_with_pretrained` | `true` | Skip training if pretrained state is supplied. |
| `experiment.pretrained_path` | unset | State dict path. |
| `experiment.seed` | `misc.seed` | Random seed alias. |

Legacy aliases still accepted: `experiment.build_dataset`, `data.rebuild`,
`data.build`, and `experiment.skip_dataset_build`.

### `evaluation`

| Key | Default | Description |
| --- | --- | --- |
| `evaluation.splits` | all loader splits | Split name or list of split names to evaluate. |
| `evaluation.runs` | `training.eval_runs` or `1` | Repeated evaluation passes. |
| `evaluation.plot_example` | `false` | Generate an example prediction plot. |
| `evaluation.save_example_plot` | `false` | Save example plot to `example_prediction.pdf`. |

### `output`

| Key | Default | Description |
| --- | --- | --- |
| `output.dir` | `misc.output_dir` or `outputs` | Parent output directory. |
| `output.name` | `misc.save_name` or `model.name` | Run directory name under `output.dir`. |

The run directory is:

```text
<output.dir>/<output.name>/
```

Saved artifacts include `model_state.pt`, `train_history.pt`,
`train_metadata.pt`, `all_losses.pt`, `per_user_all_losses.pt`, and
`experiment_summary.json`.

### `misc`

| Key | Default | Description |
| --- | --- | --- |
| `misc.seed` | unset | Random seed. |
| `misc.device` | `auto` | Fallback device selector. |
| `misc.output_dir` | `outputs` | Fallback for `output.dir`. |
| `misc.save_name` | `model.name` | Fallback for `output.name`. |
| `misc.log_level` | `INFO` | Python logging level. |

### Hydra

Hydra's own job directory is separate from TimeTensor's `output.dir`.

```bash
hydra.run.dir='outputs/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}'
```

Use single quotes or escape the `$` in `${now:...}` when writing Bash/Slurm
scripts, otherwise Bash tries to expand `now`.

## Dataset Layout

The Slurm scripts expect CSVs and built tensors in the same dataset directory:

```text
../datasets/
  electricity/
    electricity.csv
    values.pt
    datetimes.pt
    individual_ids.pt
    date_ids.pt
    dataset_metadata.json
```

`experiment.rebuild_dataset=true` writes the tensor files into `data.path`.
Set `REBUILD_DATASETS=false` for Slurm runs when tensors already exist.

## Slurm Scripts

Scripts live in `timetensors/slurm/` and use these shell variables:

| Variable | Default | Description |
| --- | --- | --- |
| `DATA_ROOT` | `../datasets` in scripts | Parent dataset directory. |
| `OUT_ROOT` | per-script `outputs/...` | Parent output directory. |
| `REBUILD_DATASETS` | `true` | Rebuild tensors for the first run of each dataset. |
| `SEED` | `1` where used | Experiment seed. |

Submit from the repository root:

```bash
sbatch timetensors/slurm/benchmark_models.slurm
REBUILD_DATASETS=false sbatch timetensors/slurm/benchmark_models.slurm
```

## Device Logging

Training and evaluation log the selected device to stdout, which appears in
`script_outputs/*.out` for Slurm jobs:

```text
Device selected: requested=auto resolved=cuda:0 {'cuda_available': True, ...}
```

If `resolved=cpu` or `cuda_available=False`, PyTorch did not find a usable GPU
inside that job.
