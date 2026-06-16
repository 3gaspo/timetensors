#!/bin/bash
#SBATCH --job-name=tt_models
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/tt_models_%j.out
#SBATCH --error=script_outputs/tt_models_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

DATA_ROOT="${DATA_ROOT:-../datasets}"
OUT_ROOT="${OUT_ROOT:-../outputs/timetensor_models}"
DATASETS=("electricity" "traffic" "solar")
SETTINGS=("168 24" "672 168")
MODELS=("persistence" "expected" "repeat" "linear" "dlinear" "patchtst")

for dataset in "${DATASETS[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    for model in "${MODELS[@]}"; do
      epochs=50
      if [[ "$model" =~ ^(persistence|expected|repeat)$ ]]; then
        epochs=0
      fi
      srun python -m timetensor.experiment \
        +data.path="${DATA_ROOT}/${dataset}/run/${dataset}" \
        +task.lags="$L" \
        +task.horizon="$H" \
        +data.splits.date_splits=[0.6,0.2,0.2] \
        +data.sampling.train_idx_mode=random \
        +data.sampling.eval_idx_mode=all \
        +data.sampling.eval_stride="$H" \
        +training.bs=128 \
        +training.epochs="$epochs" \
        +training.loss=MSE \
        +model.name="$model" \
        +model.path="$model" \
        +normalization.name=instance \
        +experiment.skip_dataset_build=true \
        +output.dir="${OUT_ROOT}/${dataset}/${L}_${H}" \
        +output.name="$model"
    done
  done
done
