#!/bin/bash
#SBATCH --job-name=tt_sota_cmp
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/tt_sota_cmp_%j.out
#SBATCH --error=script_outputs/tt_sota_cmp_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

DATA_ROOT="../datasets"
OUT_ROOT="outputs/timetensor_sota_compare"
REBUILD_DATASETS=true
RECOMPUTE_STATS=true
ELECTRICITY_DROP_USERS="57;106;127;182;298;6;113;143"
STATS_MAX_WINDOWS=""
SEED=1
CHRONOS_DEVICE_MAP="cuda"
TABPFN_DEVICE="cuda"
CHRONOS_WEIGHTS_PATH="../weights/chronos2"
TABPFN_WEIGHTS_PATH="../weights/tabpfnts/tabpfn-v2.5-regressor-v2.5_default.ckpt"
STATS_ARGS=()
[ -n "$STATS_MAX_WINDOWS" ] && STATS_ARGS+=(+experiment.stats_max_windows="$STATS_MAX_WINDOWS")

DATASETS=("electricity" "traffic" "solar" "exchange_rate")
SETTINGS=("1344 336" "672 168" "168 24" "24 24")
MODELS=("patchtst" "chronos" "tabpfn")
EVAL_STRIDE=512
PATCHTST_BATCH_SIZE=256
PATCHTST_EPOCHS=200
SOTA_BATCH_SIZE=350

for dataset in "${DATASETS[@]}"; do
  DATASET_DIR="${DATA_ROOT}/${dataset}"
  DATA_ARGS=()
  [ "$dataset" = "electricity" ] && DATA_ARGS+=("+data.drop_users='$ELECTRICITY_DROP_USERS'")
  rebuild_dataset="$REBUILD_DATASETS"
  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    for model in "${MODELS[@]}"; do
      epochs="$PATCHTST_EPOCHS"
      batch_size="$PATCHTST_BATCH_SIZE"
      model_args=()
      if [ "$model" = "chronos" ]; then
        epochs=0
        batch_size="$SOTA_BATCH_SIZE"
        model_args+=(+model.kwargs.device_map="$CHRONOS_DEVICE_MAP")
        [ -n "$CHRONOS_WEIGHTS_PATH" ] && model_args+=(+model.kwargs.weights_path="$CHRONOS_WEIGHTS_PATH")
      elif [ "$model" = "tabpfn" ]; then
        epochs=0
        batch_size="$SOTA_BATCH_SIZE"
        model_args+=(+model.kwargs.device="$TABPFN_DEVICE")
        [ -n "$TABPFN_WEIGHTS_PATH" ] && model_args+=(+model.kwargs.weights_path="$TABPFN_WEIGHTS_PATH")
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
        +data.sampling.train_stride="$EVAL_STRIDE" \
        +data.sampling.eval_stride="$EVAL_STRIDE" \
        +data.sampling.remove_train_cte=true \
        +data.sampling.remove_eval_cte=true \
        +data.sampling.use_context=false \
        +data.sampling.use_individual_context=false \
        +data.sampling.use_global_context=false \
        +training.batch_size="$batch_size" \
        +training.device=gpu \
        +training.epochs="$epochs" \
        +training.loss=nmse \
        +model.name="$model" \
        +model.path="$model" \
        "${model_args[@]}" \
        +normalization.name=instance \
        +experiment.rebuild_dataset="$rebuild_dataset" \
        +experiment.recompute_stats="$RECOMPUTE_STATS" \
        "${STATS_ARGS[@]}" \
        +experiment.evaluate=true \
        +experiment.seed="$SEED" \
        +output.dir="${OUT_ROOT}/${dataset}/${L}_${H}" \
        +output.name="$model" \
        hydra.run.dir='outputs/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}'
      rebuild_dataset=false
    done
  done
done

srun python -m timetensors.results_table "$OUT_ROOT" --metric mse --split test1
