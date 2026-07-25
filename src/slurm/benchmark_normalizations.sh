#!/bin/bash
# Compare input normalization parameterizations under a fixed loss.
set -euo pipefail
TEST_MODE="${TEST_MODE:-false}"
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_common.sh"
OUT_ROOT="$ROOT/outputs/normalizations"
read -ra NORMS <<< "${NORMS_OVERRIDE:-identity standard min-max in-min-max instance revin}"

if [ "$RUN_MODE" != tables ]; then
  for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      for model in "${MODELS[@]}"; do
        for norm in "${NORMS[@]}"; do
          run_case scripts.experiment "$dataset" "$setting" "${model}_${norm}" \
            +model.name="$model" +model.path="$model" +normalization.name="$norm" +training.loss=nmse
        done
      done
    done
  done
fi
if [ "$RUN_MODE" != train ]; then
  for model in "${MODELS[@]}"; do
    methods=(); for norm in "${NORMS[@]}"; do methods+=("${model}_${norm}"); done
    write_table "$model" mse "$(IFS=,; echo "${methods[*]}")"
  done
fi
