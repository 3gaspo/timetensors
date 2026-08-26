#!/bin/bash
# Compare constant-window filtering against keeping all pairs and dropping all
# affected users.
set -euo pipefail
FAMILY=constants
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_common.sh"
OUT_ROOT="${OUT_ROOT:-$OUTPUTS_ROOT/constants}"
if [ "$EXPERIMENT_MODE" = full ] || [ "$EXPERIMENT_MODE" = ultra ]; then
  DEFAULT_POLICIES="keep remove_train_windows remove_eval_windows remove_all_windows drop_all_users"
else
  DEFAULT_POLICIES="keep remove_all_windows drop_all_users"
fi
read -ra POLICIES <<< "${POLICIES_OVERRIDE:-$DEFAULT_POLICIES}"

run_training() {
  for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      for model in "${MODELS[@]}"; do
        for policy in "${POLICIES[@]}"; do
          train_windows=false; eval_windows=false; train_users=false; eval_users=false
          case "$policy" in
            keep) ;;
            remove_train_windows) train_windows=true ;;
            remove_eval_windows) eval_windows=true ;;
            remove_all_windows) train_windows=true; eval_windows=true ;;
            drop_all_users) train_users=true; eval_users=true ;;
            *) log_error "unknown constants policy=$policy"; exit 2 ;;
          esac
          MODEL_CONFIG_ORDER=policy
          MODEL_CONFIG_VALUES=("policy=$policy")
          TABLE_ROW_CONFIG=policy
          TABLE_COLUMN_CONFIG=
          CASE_DISPLAY_NAME="${model}_${policy}"
          run_case scripts.experiment "$dataset" "$setting" "$model" \
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
