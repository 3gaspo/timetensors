#!/bin/bash
# Evaluate every supported frozen foundation model through TimeTensors.
set -euo pipefail
FAMILY=foundation_models
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_common.sh"
OUT_ROOT="$ROOT/outputs/foundation_models"
read -ra FOUNDATION_MODELS <<< "${FOUNDATION_MODELS_OVERRIDE:-chronos2 chronos_bolt ts_icl tabpfn_ts}"
# tirex2 remains adapter-supported but is excluded from foundation launches for now.
DROP_FOUNDATION_CONSTANT_USERS="${DROP_FOUNDATION_CONSTANT_USERS:-false}"

run_evaluation() {
  local model weight_path batch
  for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      for model in "${FOUNDATION_MODELS[@]}"; do
        case "$model" in
          chronos2)
            weight_path="$(resolve_weight_path chronos2)"
            batch=64
            ;;
          chronos_bolt)
            weight_path="$(resolve_weight_path chronos-bolt-base)"
            batch=128
            ;;
          ts_icl)
            weight_path="$(resolve_weight_path tsicl/tsicl-v1.ckpt)"
            batch=32
            ;;
          tirex2)
            weight_path="$(resolve_weight_path tirex2)"
            batch=64
            ;;
          tabpfn_ts)
            weight_path="$(resolve_weight_path tabpfnts/tabpfn-v2.5-regressor-v2.5_default.ckpt)"
            batch=1
            ;;
          *)
            log_error "unknown foundation model=$model"
            exit 2
            ;;
        esac
        MODEL_CONFIG_ORDER=normalization
        MODEL_CONFIG_VALUES=("normalization=identity")
        TABLE_ROW_CONFIG=
        TABLE_COLUMN_CONFIG=
        CASE_DISPLAY_NAME="$model"
        CASE_BATCH_SIZE="${FOUNDATION_BATCH_SIZE_OVERRIDE:-$batch}"
        run_case scripts.experiment "$dataset" "$setting" "$model" \
          +model.name="$model" +model.path="$model" +normalization.name=identity \
          +model.kwargs.weights_path="$weight_path" \
          ++data.sampling.drop_train_constant_individuals="$DROP_FOUNDATION_CONSTANT_USERS" \
          ++data.sampling.drop_eval_constant_individuals="$DROP_FOUNDATION_CONSTANT_USERS" \
          +experiment.skip_training=true ++training.epochs=0
      done
    done
  done
}

run_tables() {
  METHOD_ARG="$(IFS=,; echo "${FOUNDATION_MODELS[*]}")"
  write_table combined mse "$METHOD_ARG"
  write_table combined user_mse "$METHOD_ARG"
  write_table combined w10_mse "$METHOD_ARG"
}

TABLE_REQUIRED_OUTPUTS=(
  "$OUT_ROOT/results_combined_mse.tex"
  "$OUT_ROOT/results_combined_user_mse.tex"
  "$OUT_ROOT/results_combined_w10_mse.tex"
)
TABLE_EXPECTED_METHODS=("${FOUNDATION_MODELS[@]}")
log_section "workflow start family=foundation_models mode=$EXPERIMENT_MODE stages=$STAGES_SPEC"
source "$ROOT/src/slurm/stage_evaluate.sh"
source "$ROOT/src/slurm/stage_tables.sh"
log_section "workflow done family=foundation_models output=$OUT_ROOT"
