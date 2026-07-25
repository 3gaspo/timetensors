#!/bin/bash
# Compare forecasting losses under instance normalization.
set -euo pipefail
TEST_MODE="${TEST_MODE:-false}"
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_common.sh"
OUT_ROOT="$ROOT/outputs/losses"
read -ra LOSSES <<< "${LOSSES_OVERRIDE:-mse mae nmse nmae relative_mse}"

if [ "$RUN_MODE" != tables ]; then
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
fi
if [ "$RUN_MODE" != train ]; then
  for model in "${MODELS[@]}"; do
    methods=(); for loss in "${LOSSES[@]}"; do methods+=("${model}_${loss}"); done
    write_table "$model" mse "$(IFS=,; echo "${methods[*]}")"
  done
fi
