#!/bin/bash
# Compare input normalization parameterizations under a fixed loss.
set -euo pipefail
FAMILY=normalizations
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_common.sh"
OUT_ROOT="$ROOT/outputs/normalizations"
read -ra NORMS <<< "${NORMS_OVERRIDE:-identity standard min-max in-min-max instance revin}"

run_training() {
  for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      for model in "${MODELS[@]}"; do
        for norm in "${NORMS[@]}"; do
          MODEL_CONFIG_ORDER=normalization
          MODEL_CONFIG_VALUES=("normalization=$norm")
          TABLE_ROW_CONFIG=normalization
          TABLE_COLUMN_CONFIG=
          CASE_DISPLAY_NAME="${model}_${norm}"
          run_case scripts.experiment "$dataset" "$setting" "$model" \
            +model.name="$model" +model.path="$model" +normalization.name="$norm" +training.loss=nmse
        done
      done
    done
  done
}

run_tables() {
  for model in "${MODELS[@]}"; do
    methods=(); for norm in "${NORMS[@]}"; do methods+=("${model}_${norm}"); done
    write_table "$model" mse "$(IFS=,; echo "${methods[*]}")"
  done
}

TABLE_REQUIRED_OUTPUTS=()
TABLE_EXPECTED_METHODS=()
for model in "${MODELS[@]}"; do
  TABLE_REQUIRED_OUTPUTS+=("$OUT_ROOT/results_${model}_mse.tex")
  for norm in "${NORMS[@]}"; do TABLE_EXPECTED_METHODS+=("${model}_${norm}"); done
done
log_section "workflow start family=normalizations mode=$EXPERIMENT_MODE stages=$STAGES_SPEC"
source "$ROOT/src/slurm/stage_train.sh"
source "$ROOT/src/slurm/stage_tables.sh"
log_section "workflow done family=normalizations output=$OUT_ROOT"
