#!/bin/bash
#SBATCH --job-name=tt_sampling
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/tt_sampling_%j.out
#SBATCH --error=script_outputs/tt_sampling_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

DATA_ROOT="${DATA_ROOT:-../datasets}"
OUT_ROOT="${OUT_ROOT:-../outputs/timetensor_sampling}"
DATASETS=("electricity" "traffic" "solar")
SETTINGS=("168 24" "672 168" "1344 336")
SEED="${SEED:-1}"

for dataset in "${DATASETS[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    for sampling_name in random individuals dates all; do
      train_mode="$sampling_name"
      if [ "$sampling_name" = "all" ]; then
        train_mode="all"
      fi
      srun python -m timetensor.experiment \
        +data.path="${DATA_ROOT}/${dataset}/run/${dataset}" \
        +task.lags="$L" \
        +task.horizon="$H" \
        +data.splits.date_splits=[0.6,0.2,0.2] \
        +data.sampling.train_idx_mode="$train_mode" \
        +data.sampling.eval_idx_mode=all \
        +data.sampling.train_stride=1 \
        +data.sampling.eval_stride="$H" \
        +training.bs=128 \
        +training.epochs=50 \
        +training.loss=MSE \
        +model.name=patchtst \
        +model.path=patchtst \
        +normalization.name=instance \
        +experiment.skip_dataset_build=true \
        +experiment.evaluate=true \
        +misc.seed="$SEED" \
        +output.dir="${OUT_ROOT}/${dataset}/${L}_${H}" \
        +output.name="patchtst_sampling_${sampling_name}"
    done
  done
done
