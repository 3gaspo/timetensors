#!/bin/bash
# Compare forecasting losses under instance normalization.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_common.sh"
OUT_ROOT="$ROOT/outputs/losses"
read -ra LOSSES <<< "${LOSSES_OVERRIDE:-mse mae nmse nmae relative_mse}"

run_training() {
  for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      for model in "${MODELS[@]}"; do
        for loss in "${LOSSES[@]}"; do
          run_case scripts.experiment "$dataset" "$setting" "${model}_${loss}" \
            +model.name="$model" +model.path="$model" +normalization.name=instance +training.loss="$loss"
        done
      done
    done
  done
}

run_tables() {
  for model in "${MODELS[@]}"; do
    methods=(); for loss in "${LOSSES[@]}"; do methods+=("${model}_${loss}"); done
    write_table "$model" mse "$(IFS=,; echo "${methods[*]}")"
  done
}

WORKFLOW_STATE_DIR="$OUT_ROOT/.workflow"
TABLE_INPUT_NAME=run.complete
TABLE_STAGE_SIGNATURE="v1|family=losses|mode=$EXPERIMENT_MODE|datasets=$DATASETS_CSV|settings=$SETTINGS_CSV|models=${MODELS[*]}|seeds=$SEEDS_CSV|losses=${LOSSES[*]}"
TRAIN_STAGE_SIGNATURE="$TABLE_STAGE_SIGNATURE|$COMMON_TRAIN_SIGNATURE"
TABLE_REQUIRED_OUTPUTS=()
TABLE_EXPECTED_METHODS=()
for model in "${MODELS[@]}"; do
  TABLE_REQUIRED_OUTPUTS+=("$OUT_ROOT/results_${model}_mse.tex")
  for loss in "${LOSSES[@]}"; do TABLE_EXPECTED_METHODS+=("${model}_${loss}"); done
done
log_section "workflow start family=losses mode=$EXPERIMENT_MODE stages=$STAGES_SPEC"
source "$ROOT/src/slurm/stage_train.sh"
source "$ROOT/src/slurm/stage_tables.sh"
log_section "workflow done family=losses output=$OUT_ROOT"
