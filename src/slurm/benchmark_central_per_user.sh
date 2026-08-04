#!/bin/bash
# Compare one centralized model with one independently fitted model per user.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_common.sh"
OUT_ROOT="$ROOT/outputs/central_per_user"
if [ "$EXPERIMENT_MODE" = ultra ]; then
  DEFAULT_USER_MODELS="patchtst chronos"
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
          run_case scripts.experiment "$dataset" "$setting" "patchtst_${scope}" \
            +model.name=patchtst +model.path=patchtst +normalization.name=instance \
            +training.loss=nmse +experiment.training_scope="$scope"
        fi
        if [[ " ${USER_MODELS[*]} " == *" chronos "* ]]; then
          cross_learning=false; [ "$scope" = central ] && cross_learning=true
          run_case scripts.experiment "$dataset" "$setting" "chronos_${scope}" \
            +model.name=chronos +model.path=chronos +normalization.name=identity \
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
  write_table combined w10_mse "$METHOD_ARG"
}

WORKFLOW_STATE_DIR="$OUT_ROOT/.workflow"
TABLE_INPUT_NAME=run.complete
TABLE_STAGE_SIGNATURE="v1|family=central_per_user|mode=$EXPERIMENT_MODE|datasets=$DATASETS_CSV|settings=$SETTINGS_CSV|seeds=$SEEDS_CSV|models=${USER_MODELS[*]}"
TRAIN_STAGE_SIGNATURE="$TABLE_STAGE_SIGNATURE|$COMMON_TRAIN_SIGNATURE"
TABLE_REQUIRED_OUTPUTS=("$OUT_ROOT/results_combined_mse.tex" "$OUT_ROOT/results_combined_w10_mse.tex")
TABLE_EXPECTED_METHODS=("${METHODS[@]}")
log_section "workflow start family=central_per_user mode=$EXPERIMENT_MODE stages=$STAGES_SPEC"
source "$ROOT/src/slurm/stage_train.sh"
source "$ROOT/src/slurm/stage_tables.sh"
log_section "workflow done family=central_per_user output=$OUT_ROOT"
