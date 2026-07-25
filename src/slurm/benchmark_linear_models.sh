#!/bin/bash
# Compare trainable and closed-form linear models and save their coefficients.
set -euo pipefail
TEST_MODE="${TEST_MODE:-false}"
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_common.sh"
OUT_ROOT="$ROOT/outputs/linear_models"
if [ "$BENCHMARK_PROFILE" = test ]; then
  DEFAULT_LINEAR_METHODS="linear sklinear"
  DEFAULT_LINEAR_NORMS="instance"
else
  DEFAULT_LINEAR_METHODS="linear periodic_linear sklinear"
  DEFAULT_LINEAR_NORMS="identity standard instance"
fi
read -ra LINEAR_METHODS <<< "${LINEAR_METHODS_OVERRIDE:-$DEFAULT_LINEAR_METHODS}"
read -ra LINEAR_NORMS <<< "${LINEAR_NORMS_OVERRIDE:-$DEFAULT_LINEAR_NORMS}"
METHODS=(persistence)
for method in "${LINEAR_METHODS[@]}"; do
  for norm in "${LINEAR_NORMS[@]}"; do
    METHODS+=("${method}_${norm}")
  done
done

if [ "$RUN_MODE" != tables ]; then
  for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      run_case scripts.experiment "$dataset" "$setting" persistence \
        +model.name=persistence +model.path=persistence +normalization.name=instance \
        ++training.epochs=0 +experiment.plot_weights=true
      for method in "${LINEAR_METHODS[@]}"; do
        for norm in "${LINEAR_NORMS[@]}"; do
          name="${method}_${norm}"
          if [ "$method" = sklinear ]; then
            run_case scripts.train_sklearn "$dataset" "$setting" "$name" \
              +model.name=sklinear +normalization.name="$norm" \
              +sklearn.unroll_mode=accessible +sklearn.eval_unroll_mode=accessible \
              +experiment.plot_weights=true
          else
            run_case scripts.experiment "$dataset" "$setting" "$name" \
              +model.name="$method" +model.path="$method" +normalization.name="$norm" \
              +experiment.plot_weights=true
          fi
        done
      done
    done
  done
fi
if [ "$RUN_MODE" != train ]; then
  write_table combined mse "$(IFS=,; echo "${METHODS[*]}")"
fi
