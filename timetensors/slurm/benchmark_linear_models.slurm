#!/bin/bash
#SBATCH --job-name=tt_linear_cmp
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/tt_linear_cmp_%j.out
#SBATCH --error=script_outputs/tt_linear_cmp_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

DATA_ROOT="../datasets"
OUT_ROOT="outputs/timetensor_linear_compare"
REBUILD_DATASETS=true
RECOMPUTE_STATS=true
ELECTRICITY_DROP_USERS="57;106;127;182;298;6;113;143"
STATS_MAX_WINDOWS=""
STATS_ARGS=()
[ -n "$STATS_MAX_WINDOWS" ] && STATS_ARGS+=(+experiment.stats_max_windows="$STATS_MAX_WINDOWS")

DATASETS=("electricity" "traffic" "solar")
SETTINGS=("336 24" "672 168")
LOSSES=("mse" "nmse" "rmse")
TRAIN_STRIDE=24
EVAL_STRIDE=24

for dataset in "${DATASETS[@]}"; do
  DATASET_DIR="${DATA_ROOT}/${dataset}"
  DATA_ARGS=()
  [ "$dataset" = "electricity" ] && DATA_ARGS+=("+data.drop_users='$ELECTRICITY_DROP_USERS'")
  rebuild_dataset="$REBUILD_DATASETS"
  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    for loss in "${LOSSES[@]}"; do
      srun python -m timetensors.experiment \
        +data.raw_path="${DATA_ROOT}/${dataset}" \
        +data.path="$DATASET_DIR" \
        +data.name="$dataset" \
        "${DATA_ARGS[@]}" \
        +task.lags="$L" \
        +task.horizon="$H" \
        +data.splits.date_splits=[0.6,0.2,0.2] \
        +data.sampling.train_idx_mode=all \
        +data.sampling.eval_idx_mode=all \
        +data.sampling.train_stride="$TRAIN_STRIDE" \
        +data.sampling.eval_stride="$EVAL_STRIDE" \
        +training.batch_size=256 \
        +training.device=gpu \
        +training.epochs=200 \
        +training.loss="$loss" \
        +model.name=linear \
        +model.path=linear \
        +normalization.name=instance \
        +experiment.rebuild_dataset="$rebuild_dataset" \
        +experiment.recompute_stats="$RECOMPUTE_STATS" \
        +experiment.plot_weights=true \
        "${STATS_ARGS[@]}" \
        +output.dir="${OUT_ROOT}/${dataset}/${L}_${H}" \
        +output.name="torch_linear_${loss}" \
        hydra.run.dir='outputs/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}'

      srun python -m timetensors.train_sklearn \
        +data.raw_path="${DATA_ROOT}/${dataset}" \
        +data.path="$DATASET_DIR" \
        +data.name="$dataset" \
        "${DATA_ARGS[@]}" \
        +task.lags="$L" \
        +task.horizon="$H" \
        +data.splits.date_splits=[0.6,0.2,0.2] \
        +data.sampling.train_idx_mode=all \
        +data.sampling.eval_idx_mode=all \
        +data.sampling.train_stride="$TRAIN_STRIDE" \
        +data.sampling.eval_stride="$EVAL_STRIDE" \
        +training.batch_size=256 \
        +training.loss="$loss" \
        +model.name=sklinear \
        +normalization.name=instance \
        +sklearn.unroll_mode=accessible \
        +sklearn.eval_unroll_mode=accessible \
        +experiment.rebuild_dataset=false \
        +experiment.recompute_stats="$RECOMPUTE_STATS" \
        +experiment.plot_weights=true \
        "${STATS_ARGS[@]}" \
        +output.dir="${OUT_ROOT}/${dataset}/${L}_${H}" \
        +output.name="sklinear_${loss}" \
        hydra.run.dir='outputs/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}'

      rebuild_dataset=false
    done
  done
done
