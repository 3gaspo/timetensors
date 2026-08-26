#!/bin/bash
# Compare one centralized model with one independently fitted model per user.
set -euo pipefail
FAMILY=central_per_user
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_common.sh"
OUT_ROOT="${OUT_ROOT:-$OUTPUTS_ROOT/central_per_user}"
if [ "$EXPERIMENT_MODE" = ultra ]; then
  DEFAULT_USER_MODELS="patchtst chronos2"
else
  DEFAULT_USER_MODELS="patchtst"
fi
read -ra USER_MODELS <<< "${USER_MODELS_OVERRIDE:-$DEFAULT_USER_MODELS}"
CHRONOS_WEIGHTS_PATH="$(resolve_weight_path chronos2)"
METHODS=()
for model in "${USER_MODELS[@]}"; do
  METHODS+=("${model}_central" "${model}_per_user")
done

run_training() {
  for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      for scope in central per_user; do
        if [[ " ${USER_MODELS[*]} " == *" patchtst "* ]]; then
          MODEL_CONFIG_ORDER=scope
          MODEL_CONFIG_VALUES=("scope=$scope")
          TABLE_ROW_CONFIG=scope
          TABLE_COLUMN_CONFIG=
          CASE_DISPLAY_NAME="patchtst_${scope}"
          run_case scripts.experiment "$dataset" "$setting" patchtst \
            +model.name=patchtst +model.path=patchtst +normalization.name=instance \
            +training.loss=nmse +experiment.training_scope="$scope"
        fi
        if [[ " ${USER_MODELS[*]} " == *" chronos2 "* ]]; then
          cross_learning=false; [ "$scope" = central ] && cross_learning=true
          MODEL_CONFIG_ORDER=scope
          MODEL_CONFIG_VALUES=("scope=$scope")
          TABLE_ROW_CONFIG=scope
          TABLE_COLUMN_CONFIG=
          CASE_DISPLAY_NAME="chronos2_${scope}"
          run_case scripts.experiment "$dataset" "$setting" chronos2 \
            +model.name=chronos2 +model.path=chronos2 +normalization.name=identity \
            +model.kwargs.weights_path="$CHRONOS_WEIGHTS_PATH" \
            +model.kwargs.cross_learning="$cross_learning" \
            ++training.epochs=0 +experiment.training_scope="$scope"
        fi
      done
    done
  done
}

run_tables() {
  METHOD_ARG="$(IFS=,; echo "${METHODS[*]}")"
  write_table combined mse "$METHOD_ARG"
  write_table combined user_mse "$METHOD_ARG"
  write_table combined w10_mse "$METHOD_ARG"
}

TABLE_REQUIRED_OUTPUTS=(
  "$OUT_ROOT/results_combined_mse.tex"
  "$OUT_ROOT/results_combined_user_mse.tex"
  "$OUT_ROOT/results_combined_w10_mse.tex"
)
TABLE_EXPECTED_METHODS=("${METHODS[@]}")
log_section "workflow start family=central_per_user mode=$EXPERIMENT_MODE stages=$STAGES_SPEC"
source "$ROOT/src/slurm/stage_train.sh"
source "$ROOT/src/slurm/stage_tables.sh"
log_section "workflow done family=central_per_user output=$OUT_ROOT"
