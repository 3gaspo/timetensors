#!/bin/bash
#SBATCH --job-name=tt_constants
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/tt_constants_%j.out
#SBATCH --error=script_outputs/tt_constants_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
srun python -m timetensors.check_cuda --expected-torch 2.5.1

DATA_ROOT="../datasets"
OUT_ROOT="outputs/timetensor_constants"
REBUILD_DATASETS="${REBUILD_DATASETS:-true}"
DATASETS=("raw_electricity" "electricity" "traffic" "solar")
SETTINGS=("168 24" "672 168")
SEED="${SEED:-1}"

for dataset in "${DATASETS[@]}"; do
  DATASET_DIR="${DATA_ROOT}/${dataset}"
  rebuild_dataset="$REBUILD_DATASETS"
  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    for policy in keep remove_train remove_eval remove_all repeat_constant; do
      rm_train=false
      rm_eval=false
      repeat=false
      [ "$policy" = "remove_train" ] && rm_train=true
      [ "$policy" = "remove_eval" ] && rm_eval=true
      [ "$policy" = "remove_all" ] && rm_train=true && rm_eval=true
      [ "$policy" = "repeat_constant" ] && repeat=true
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
        +data.sampling.remove_train_cte="$rm_train" \
        +data.sampling.remove_eval_cte="$rm_eval" \
        +training.batch_size=128 \
        +training.device=gpu \
        +training.epochs=50 \
        +model.name=linear \
        +model.path=linear \
        +model.repeat_constant="$repeat" \
        +normalization.name=instance \
        +experiment.rebuild_dataset="$rebuild_dataset" \
        +output.dir="${OUT_ROOT}/${dataset}/${L}_${H}" \
        +output.name="linear_constants_${policy}"
      rebuild_dataset=false
    done
  done
done
