#!/bin/bash
# Establish persistence, PatchTST, and Chronos-2 error references.
set -euo pipefail
FAMILY=reference
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_common.sh"
OUT_ROOT="$ROOT/outputs/reference"
read -ra REFERENCE_METHODS <<< "${REFERENCE_METHODS_OVERRIDE:-persistence patchtst chronos2}"
REFERENCE_PATCHTST_NORM="${REFERENCE_PATCHTST_NORM:-instance}"
REFERENCE_LOSS="${REFERENCE_LOSS:-nmse}"
REFERENCE_DROP_TRAIN_CONSTANT_USERS="${REFERENCE_DROP_TRAIN_CONSTANT_USERS:-true}"
REFERENCE_DROP_EVAL_CONSTANT_USERS="${REFERENCE_DROP_EVAL_CONSTANT_USERS:-true}"
CHRONOS_WEIGHTS_PATH="$(resolve_weight_path chronos2)"

run_training() {
  for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      for method in "${REFERENCE_METHODS[@]}"; do
        case "$method" in
          persistence)
            MODEL_CONFIG_ORDER=normalization,loss
            MODEL_CONFIG_VALUES=("normalization=instance" "loss=none")
            TABLE_ROW_CONFIG=normalization
            TABLE_COLUMN_CONFIG=loss
            CASE_DISPLAY_NAME=persistence
            run_case scripts.experiment "$dataset" "$setting" persistence \
              +model.name=persistence +model.path=persistence +normalization.name=instance \
              ++data.sampling.drop_train_constant_individuals="$REFERENCE_DROP_TRAIN_CONSTANT_USERS" \
              ++data.sampling.drop_eval_constant_individuals="$REFERENCE_DROP_EVAL_CONSTANT_USERS" \
              ++training.epochs=0
            ;;
          patchtst)
            MODEL_CONFIG_ORDER=normalization,loss
            MODEL_CONFIG_VALUES=("normalization=$REFERENCE_PATCHTST_NORM" "loss=$REFERENCE_LOSS")
            TABLE_ROW_CONFIG=normalization
            TABLE_COLUMN_CONFIG=loss
            CASE_DISPLAY_NAME=patchtst
            run_case scripts.experiment "$dataset" "$setting" patchtst \
              +model.name=patchtst +model.path=patchtst +normalization.name="$REFERENCE_PATCHTST_NORM" \
              +training.loss="$REFERENCE_LOSS" \
              ++data.sampling.drop_train_constant_individuals="$REFERENCE_DROP_TRAIN_CONSTANT_USERS" \
              ++data.sampling.drop_eval_constant_individuals="$REFERENCE_DROP_EVAL_CONSTANT_USERS"
            ;;
          chronos2)
            MODEL_CONFIG_ORDER=normalization,loss
            MODEL_CONFIG_VALUES=("normalization=identity" "loss=none")
            TABLE_ROW_CONFIG=normalization
            TABLE_COLUMN_CONFIG=loss
            CASE_DISPLAY_NAME=chronos2
            run_case scripts.experiment "$dataset" "$setting" chronos2 \
              +model.name=chronos +model.path=chronos +normalization.name=identity \
              +model.kwargs.weights_path="$CHRONOS_WEIGHTS_PATH" \
              ++data.sampling.drop_train_constant_individuals="$REFERENCE_DROP_TRAIN_CONSTANT_USERS" \
              ++data.sampling.drop_eval_constant_individuals="$REFERENCE_DROP_EVAL_CONSTANT_USERS" \
              +model.kwargs.cross_learning=false ++training.epochs=0
            ;;
          *)
            log_error "unknown reference method=$method"
            exit 2
            ;;
        esac
      done
    done
  done
}

run_tables() {
  METHOD_ARG="$(IFS=,; echo "${REFERENCE_METHODS[*]}")"
  write_table combined mse "$METHOD_ARG"
  write_table combined w10_mse "$METHOD_ARG"
}

TABLE_REQUIRED_OUTPUTS=("$OUT_ROOT/results_combined_mse.tex" "$OUT_ROOT/results_combined_w10_mse.tex")
TABLE_EXPECTED_METHODS=("${REFERENCE_METHODS[@]}")
log_section "workflow start family=reference mode=$EXPERIMENT_MODE stages=$STAGES_SPEC"
source "$ROOT/src/slurm/stage_train.sh"
source "$ROOT/src/slurm/stage_tables.sh"
log_section "workflow done family=reference output=$OUT_ROOT"
