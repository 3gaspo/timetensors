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

DATA_ROOT="${DATA_ROOT:-../datasets}"
OUT_ROOT="${OUT_ROOT:-../outputs/timetensor_normalizations}"
DATASETS=("electricity" "traffic" "solar")
SETTINGS=("168 24" "672 168")
NORMS=("identity" "standard" "instance" "revin" "revin_last" "revin_arcsinh" "imm" "relative_mean" "rms" "minmax")

for dataset in "${DATASETS[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    for norm in "${NORMS[@]}"; do
      extra_args=()
      if [ "$norm" = "standard" ]; then
        extra_args+=(+normalization.kwargs.mean=0.0 +normalization.kwargs.std=1.0)
      fi
      if [ "$norm" = "minmax" ]; then
        extra_args+=(+normalization.kwargs.min_value=0.0 +normalization.kwargs.max_value=1.0)
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
        +training.epochs=50 \
        +training.loss=MSE \
        +model.name=dlinear \
        +model.path=dlinear \
        +normalization.name="$norm" \
        "${extra_args[@]}" \
        +experiment.skip_dataset_build=true \
        +output.dir="${OUT_ROOT}/${dataset}/${L}_${H}" \
        +output.name="dlinear_norm_${norm}"
    done
  done
done
