#!/bin/bash
# Compare constant-window and constant-user filtering policies.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_common.sh"
OUT_ROOT="$ROOT/outputs/constants"
if [ "$EXPERIMENT_MODE" = full ] || [ "$EXPERIMENT_MODE" = ultra ]; then
  DEFAULT_POLICIES="keep remove_train_windows remove_eval_windows remove_all_windows drop_train_users drop_eval_users drop_all_users"
else
  DEFAULT_POLICIES="keep drop_train_users drop_all_users"
fi
read -ra POLICIES <<< "${POLICIES_OVERRIDE:-$DEFAULT_POLICIES}"

run_training() {
  for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      for model in "${MODELS[@]}"; do
        for policy in "${POLICIES[@]}"; do
          train_windows=false; eval_windows=false; train_users=false; eval_users=false
          [[ "$policy" =~ remove_train|remove_all ]] && train_windows=true
          [[ "$policy" =~ remove_eval|remove_all ]] && eval_windows=true
          [[ "$policy" =~ drop_train|drop_all ]] && train_users=true
          [[ "$policy" =~ drop_eval|drop_all ]] && eval_users=true
          run_case scripts.experiment "$dataset" "$setting" "${model}_${policy}" \
            +model.name="$model" +model.path="$model" +normalization.name=instance \
            +data.sampling.remove_train_cte="$train_windows" \
            +data.sampling.remove_eval_cte="$eval_windows" \
            ++data.sampling.drop_train_constant_individuals="$train_users" \
            ++data.sampling.drop_eval_constant_individuals="$eval_users"
        done
      done
    done
  done
}

run_tables() {
  for model in "${MODELS[@]}"; do
    methods=(); for policy in "${POLICIES[@]}"; do methods+=("${model}_${policy}"); done
    write_table "$model" mse "$(IFS=,; echo "${methods[*]}")"
  done
}

WORKFLOW_STATE_DIR="$OUT_ROOT/.workflow"
TABLE_INPUT_NAME=run.complete
TABLE_STAGE_SIGNATURE="v1|family=constants|mode=$EXPERIMENT_MODE|datasets=$DATASETS_CSV|settings=$SETTINGS_CSV|models=${MODELS[*]}|seeds=$SEEDS_CSV|policies=${POLICIES[*]}"
TRAIN_STAGE_SIGNATURE="$TABLE_STAGE_SIGNATURE|$COMMON_TRAIN_SIGNATURE"
TABLE_REQUIRED_OUTPUTS=()
TABLE_EXPECTED_METHODS=()
for model in "${MODELS[@]}"; do
  TABLE_REQUIRED_OUTPUTS+=("$OUT_ROOT/results_${model}_mse.tex")
  for policy in "${POLICIES[@]}"; do TABLE_EXPECTED_METHODS+=("${model}_${policy}"); done
done
log_section "workflow start family=constants mode=$EXPERIMENT_MODE stages=$STAGES_SPEC"
source "$ROOT/src/slurm/stage_train.sh"
source "$ROOT/src/slurm/stage_tables.sh"
log_section "workflow done family=constants output=$OUT_ROOT"
