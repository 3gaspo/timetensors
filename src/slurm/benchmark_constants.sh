#!/bin/bash
# Compare constant-window and constant-user filtering policies.
set -euo pipefail
TEST_MODE="${TEST_MODE:-false}"
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_common.sh"
OUT_ROOT="$ROOT/outputs/constants"
if [ "$BENCHMARK_PROFILE" = full ]; then
  DEFAULT_POLICIES="keep remove_train_windows remove_eval_windows remove_all_windows drop_train_users drop_eval_users drop_all_users"
else
  DEFAULT_POLICIES="keep drop_train_users drop_all_users"
fi
read -ra POLICIES <<< "${POLICIES_OVERRIDE:-$DEFAULT_POLICIES}"

if [ "$RUN_MODE" != tables ]; then
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
fi
if [ "$RUN_MODE" != train ]; then
  for model in "${MODELS[@]}"; do
    methods=(); for policy in "${POLICIES[@]}"; do methods+=("${model}_${policy}"); done
    write_table "$model" mse "$(IFS=,; echo "${methods[*]}")"
  done
fi
