#!/bin/bash
#SBATCH --job-name=tt_models
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/tt_models_%j.out
#SBATCH --error=script_outputs/tt_models_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=h100-bis

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

DATA_ROOT="../datasets"
OUT_ROOT="outputs/timetensor_models"
REBUILD_DATASETS="${REBUILD_DATASETS:-true}"

DATASETS=("electricity" "solar")
SETTINGS=("168 24" "672 168")
EVAL_STRIDE=256
MODELS=("persistence" "dlinear")

for dataset in "${DATASETS[@]}"; do
  DATASET_DIR="${DATA_ROOT}/${dataset}"
  rebuild_dataset="$REBUILD_DATASETS"
  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    for model in "${MODELS[@]}"; do
      epochs=100
      if [[ "$model" =~ ^(persistence|expected|repeat)$ ]]; then
        epochs=0
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
        +data.sampling.eval_stride="$EVAL_STRIDE" \
        +training.batch_size=128 \
        +training.device=gpu \
        +training.epochs="$epochs" \
        +training.loss=MSE \
        +model.name="$model" \
        +model.path="$model" \
        +normalization.name=instance \
        +experiment.rebuild_dataset="$rebuild_dataset" \
        +output.dir="${OUT_ROOT}/${dataset}/${L}_${H}" \
        +output.name="$model" \
        hydra.run.dir='outputs/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}'
      rebuild_dataset=false
    done
  done
done
