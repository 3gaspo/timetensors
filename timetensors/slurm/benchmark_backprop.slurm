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
DATASETS=("electricity" "traffic")
SETTINGS=("168 24" "672 168")
LOSSES=("MSE" "nMSE" "MAE" "nMAE" "rMSE")
OPTIMS=("adam" "adamw" "sgd")

for dataset in "${DATASETS[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    for loss in "${LOSSES[@]}"; do
      for optim in "${OPTIMS[@]}"; do
        lr="0.001"
        [ "$optim" = "sgd" ] && lr="0.01"
        srun python -m timetensors.experiment \
          +data.path="${DATA_ROOT}/${dataset}/${dataset}" \
          +task.lags="$L" \
          +task.horizon="$H" \
          +data.splits.date_splits=[0.6,0.2,0.2] \
          +data.sampling.train_idx_mode=random \
          +data.sampling.eval_idx_mode=all \
          +data.sampling.eval_stride="$H" \
          +training.bs=128 \
          +training.epochs=50 \
          +training.loss="$loss" \
          +training.optimizer="$optim" \
          +training.lr="$lr" \
          +training.grad_clip=1.0 \
          +model.name=linear \
          +model.path=linear \
          +normalization.name=instance \
          +experiment.skip_dataset_build=true \
          +output.dir="${OUT_ROOT}/${dataset}/${L}_${H}" \
          +output.name="linear_${loss}_${optim}"
      done
    done
  done
done
