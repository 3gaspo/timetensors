#!/bin/bash
# Compare one centralized model with one independently fitted model per user.
set -euo pipefail
TEST_MODE="${TEST_MODE:-false}"
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_common.sh"
OUT_ROOT="$ROOT/outputs/central_per_user"
if [ "$BENCHMARK_PROFILE" = test ]; then
  DEFAULT_USER_MODELS="patchtst"
else
  DEFAULT_USER_MODELS="patchtst chronos"
fi
read -ra USER_MODELS <<< "${USER_MODELS_OVERRIDE:-$DEFAULT_USER_MODELS}"
CHRONOS_WEIGHTS_PATH="$(resolve_weight_path chronos2)"
METHODS=()
for model in "${USER_MODELS[@]}"; do
  METHODS+=("${model}_central" "${model}_per_user")
done

if [ "$RUN_MODE" != tables ]; then
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
fi
if [ "$RUN_MODE" != train ]; then
  METHOD_ARG="$(IFS=,; echo "${METHODS[*]}")"
  write_table combined mse "$METHOD_ARG"
  write_table combined w10_mse "$METHOD_ARG"
fi
