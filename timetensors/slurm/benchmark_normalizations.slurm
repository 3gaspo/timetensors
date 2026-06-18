#!/bin/bash
#SBATCH --job-name=tt_norms
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/tt_norms_%j.out
#SBATCH --error=script_outputs/tt_norms_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

DATA_ROOT="../datasets"
OUT_ROOT="outputs/timetensor_normalizations"
REBUILD_DATASETS="${REBUILD_DATASETS:-true}"
RECOMPUTE_STATS="${RECOMPUTE_STATS:-true}"
STATS_ARGS=()
[ -n "${STATS_MAX_WINDOWS:-}" ] && STATS_ARGS+=(+experiment.stats_max_windows="$STATS_MAX_WINDOWS")
DATASETS=("electricity" "traffic" "solar")
SETTINGS=("168 24" "672 168")
NORMS=("identity" "standard" "instance" "revin" "revin-center-last" "revin-arcsinh" "in-min-max" "relative_mean" "rms" "min-max")

for dataset in "${DATASETS[@]}"; do
  DATASET_DIR="${DATA_ROOT}/${dataset}"
  rebuild_dataset="$REBUILD_DATASETS"
  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    for norm in "${NORMS[@]}"; do
      norm_name="$norm"
      extra_args=()
      if [ "$norm" = "standard" ]; then
        extra_args+=(+normalization.kwargs.mean=0.0 +normalization.kwargs.std=1.0)
      fi
      if [ "$norm" = "min-max" ]; then
        extra_args+=(+normalization.kwargs.min_value=0.0 +normalization.kwargs.max_value=1.0)
      fi
      if [ "$norm" = "revin-center-last" ]; then
        norm_name="revin"
        extra_args+=(+normalization.kwargs.center=last)
      fi
      if [ "$norm" = "revin-arcsinh" ]; then
        norm_name="revin"
        extra_args+=(+normalization.kwargs.transform=arcsinh)
      fi
      srun python -m timetensors.experiment \
        +data.raw_path="${DATA_ROOT}/${dataset}" \
        +data.path="$DATASET_DIR" \
        +data.name="$dataset" \
        +task.lags="$L" \
        +task.horizon="$H" \
        +data.splits.date_splits=[0.6,0.2,0.2] \
        +data.sampling.train_idx_mode=random \
        +data.sampling.eval_idx_mode=all \
        +data.sampling.eval_stride="$H" \
        +training.batch_size=256 \
        +training.device=gpu \
        +training.epochs=50 \
        +training.loss=nmse \
        +model.name=dlinear \
        +model.path=dlinear \
        +normalization.name="$norm_name" \
        "${extra_args[@]}" \
        +experiment.rebuild_dataset="$rebuild_dataset" \
        +experiment.recompute_stats="$RECOMPUTE_STATS" \
        "${STATS_ARGS[@]}" \
        +output.dir="${OUT_ROOT}/${dataset}/${L}_${H}" \
        +output.name="dlinear_norm_${norm}"
      rebuild_dataset=false
    done
  done
done
