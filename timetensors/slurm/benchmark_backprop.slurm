#!/bin/bash
#SBATCH --job-name=tt_backprop
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/tt_backprop_%j.out
#SBATCH --error=script_outputs/tt_backprop_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

DATA_ROOT="../datasets"
OUT_ROOT="outputs/timetensor_backprop"
REBUILD_DATASETS="${REBUILD_DATASETS:-true}"
RECOMPUTE_STATS="${RECOMPUTE_STATS:-true}"
STATS_ARGS=()
[ -n "${STATS_MAX_WINDOWS:-}" ] && STATS_ARGS+=(+experiment.stats_max_windows="$STATS_MAX_WINDOWS")
DATASETS=("electricity" "traffic")
SETTINGS=("168 24" "672 168")
LOSSES=("mse" "nmse" "mae" "nmae" "rmse")
OPTIMS=("adam" "adamw" "sgd")

for dataset in "${DATASETS[@]}"; do
  DATASET_DIR="${DATA_ROOT}/${dataset}"
  rebuild_dataset="$REBUILD_DATASETS"
  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    for loss in "${LOSSES[@]}"; do
      for optim in "${OPTIMS[@]}"; do
        lr="0.001"
        [ "$optim" = "sgd" ] && lr="0.01"
        srun python -m timetensors.experiment \
          +data.raw_path="${DATA_ROOT}/${dataset}" \
          +data.path="$DATASET_DIR" \
          +data.name="$dataset" \
          +task.lags="$L" \
          +task.horizon="$H" \
          +data.splits.date_splits=[0.6,0.2,0.2] \
          +data.sampling.train_idx_mode=individuals \
          +data.sampling.eval_idx_mode=all \
          +data.sampling.eval_stride="$H" \
          +training.batch_size=256 \
          +training.device=gpu \
          +training.epochs=200 \
          +training.loss="$loss" \
          +training.optimizer="$optim" \
          +training.lr="$lr" \
          +training.grad_clip=1.0 \
          +model.name=linear \
          +model.path=linear \
          +normalization.name=instance \
          +experiment.rebuild_dataset="$rebuild_dataset" \
          +experiment.recompute_stats="$RECOMPUTE_STATS" \
          "${STATS_ARGS[@]}" \
          +output.dir="${OUT_ROOT}/${dataset}/${L}_${H}" \
          +output.name="linear_${loss}_${optim}"
        rebuild_dataset=false
      done
    done
  done
done
