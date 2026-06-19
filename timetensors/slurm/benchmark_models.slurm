#!/bin/bash
#SBATCH --job-name=tt_models
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/tt_models_%j.out
#SBATCH --error=script_outputs/tt_models_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=h100-bis

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

DATA_ROOT="../datasets"
OUT_ROOT="outputs/timetensor_models"
REBUILD_DATASETS=true
RECOMPUTE_STATS=true
ELECTRICITY_DROP_USERS="57;106;127;182;298;6;113;143"
STATS_MAX_WINDOWS=""
STATS_ARGS=()
[ -n "$STATS_MAX_WINDOWS" ] && STATS_ARGS+=(+experiment.stats_max_windows="$STATS_MAX_WINDOWS")

DATASETS=("electricity") # "solar")
SETTINGS=("168 24") # "672 168")
EVAL_STRIDE=256
MODELS=("patchtst") #persistence" "dlinear")

for dataset in "${DATASETS[@]}"; do
  DATASET_DIR="${DATA_ROOT}/${dataset}"
  DATA_ARGS=()
  [ "$dataset" = "electricity" ] && DATA_ARGS+=("+data.drop_users='$ELECTRICITY_DROP_USERS'")
  rebuild_dataset="$REBUILD_DATASETS"
  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    for model in "${MODELS[@]}"; do
      epochs=1000
      if [[ "$model" =~ ^(persistence|expected|repeat)$ ]]; then
        epochs=0
      fi
      srun python -m timetensors.experiment \
        +data.raw_path="${DATA_ROOT}/${dataset}" \
        +data.path="$DATASET_DIR" \
        +data.name="$dataset" \
        "${DATA_ARGS[@]}" \
        +task.lags="$L" \
        +task.horizon="$H" \
        +data.splits.date_splits=[0.6,0.2,0.2] \
        +data.sampling.train_idx_mode=individuals \
        +data.sampling.eval_idx_mode=all \
        +data.sampling.eval_stride="$EVAL_STRIDE" \
        +training.batch_size=256 \
        +training.device=gpu \
        +training.epochs="$epochs" \
        +training.loss=nmse \
        +model.name="$model" \
        +model.path="$model" \
        +normalization.name=instance \
        +experiment.rebuild_dataset="$rebuild_dataset" \
        +experiment.recompute_stats="$RECOMPUTE_STATS" \
        "${STATS_ARGS[@]}" \
        +output.dir="${OUT_ROOT}/${dataset}/${L}_${H}" \
        +output.name="$model" \
        hydra.run.dir='outputs/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}'
      rebuild_dataset=false
    done
  done
done
