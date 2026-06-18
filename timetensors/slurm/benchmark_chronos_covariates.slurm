#!/bin/bash
#SBATCH --job-name=tt_chr_cov
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/tt_chr_cov_%j.out
#SBATCH --error=script_outputs/tt_chr_cov_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

DATA_ROOT="${DATA_ROOT:-../datasets}"
OUT_ROOT="${OUT_ROOT:-outputs/timetensor_chronos_covariates}"
REBUILD_DATASETS="${REBUILD_DATASETS:-true}"
RECOMPUTE_STATS="${RECOMPUTE_STATS:-true}"
STATS_ARGS=()
[ -n "${STATS_MAX_WINDOWS:-}" ] && STATS_ARGS+=(+experiment.stats_max_windows="$STATS_MAX_WINDOWS")

DATASETS=("electricity" "traffic" "solar" "exchange_rate")
SETTINGS=("1344 336" "672 168" "168 24" "24 24")
EVAL_STRIDE="${EVAL_STRIDE:-512}"
BATCH_SIZE="${SOTA_BATCH_SIZE:-350}"
AUGMENT_TARGET="${AUGMENT_TARGET:-past_only}"

MODEL_ARGS=(+model.kwargs.device_map="${CHRONOS_DEVICE_MAP:-cuda}")
[ -n "${CHRONOS_WEIGHTS_PATH:-}" ] && MODEL_ARGS+=(+model.kwargs.weights_path="$CHRONOS_WEIGHTS_PATH")

# Format: output_suffix kind count value
AUGMENTS=(
  "none none 0 0"
  "noise_0p01 noise 1 0.01"
  "noise_0p1 noise 1 0.1"
  "noise_1 noise 1 1.0"
  "noise_1_x2 noise 2 1.0"
  "noise_1_x3 noise 3 1.0"
  "noise_1_x5 noise 5 1.0"
  "constant_0 constant 1 0.0"
  "constant_1 constant 1 1.0"
  "constant_0_x2 constant 2 0.0"
  "constant_0_x3 constant 3 0.0"
  "constant_0_x5 constant 5 0.0"
)

for dataset in "${DATASETS[@]}"; do
  DATASET_DIR="${DATA_ROOT}/${dataset}"
  rebuild_dataset="$REBUILD_DATASETS"
  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    for augment in "${AUGMENTS[@]}"; do
      read -r suffix kind count value <<< "$augment"
      augment_args=()
      if [ "$kind" != "none" ]; then
        augment_args+=(
          "+model.covariate_augmentation.modes=[{name:${kind},count:${count},value:${value},target:${AUGMENT_TARGET}}]"
        )
      fi

      srun python -m timetensors.experiment \
        +data.raw_path="${DATA_ROOT}/${dataset}" \
        +data.path="$DATASET_DIR" \
        +data.name="$dataset" \
        +task.lags="$L" \
        +task.horizon="$H" \
        +data.splits.date_splits=[0.6,0.2,0.2] \
        +data.sampling.train_idx_mode=all \
        +data.sampling.eval_idx_mode=all \
        +data.sampling.train_stride="$EVAL_STRIDE" \
        +data.sampling.eval_stride="$EVAL_STRIDE" \
        +data.sampling.remove_train_cte=false \
        +data.sampling.remove_eval_cte=false \
        +data.sampling.use_context=false \
        +data.sampling.use_individual_context=false \
        +data.sampling.use_global_context=false \
        +training.batch_size="$BATCH_SIZE" \
        +training.device=gpu \
        +training.epochs=0 \
        +training.loss=MSE \
        +model.name=chronos \
        +model.path=chronos \
        "${MODEL_ARGS[@]}" \
        "${augment_args[@]}" \
        +normalization.name=instance \
        +experiment.rebuild_dataset="$rebuild_dataset" \
        +experiment.recompute_stats="$RECOMPUTE_STATS" \
        "${STATS_ARGS[@]}" \
        +experiment.evaluate=true \
        +experiment.seed="${SEED:-1}" \
        +output.dir="${OUT_ROOT}/${dataset}/${L}_${H}" \
        +output.name="chronos_${suffix}_${AUGMENT_TARGET}" \
        hydra.run.dir='outputs/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}'
      rebuild_dataset=false
    done
  done
done
