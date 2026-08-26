#!/bin/bash
# Compare trainable and closed-form linear models and save their coefficients.
set -euo pipefail
FAMILY=linear_models
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_common.sh"
OUT_ROOT="${OUT_ROOT:-$OUTPUTS_ROOT/linear_models}"
if [ "$EXPERIMENT_MODE" = test ]; then
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

run_training() {
  for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      MODEL_CONFIG_ORDER=normalization
      MODEL_CONFIG_VALUES=("normalization=instance")
      TABLE_ROW_CONFIG=
      TABLE_COLUMN_CONFIG=normalization
      CASE_DISPLAY_NAME=persistence
      run_case scripts.experiment "$dataset" "$setting" persistence \
        +model.name=persistence +model.path=persistence +normalization.name=instance \
        ++training.epochs=0 +experiment.plot_weights=true
      for method in "${LINEAR_METHODS[@]}"; do
        for norm in "${LINEAR_NORMS[@]}"; do
          name="${method}_${norm}"
          if [ "$method" = sklinear ]; then
            MODEL_CONFIG_ORDER=normalization
            MODEL_CONFIG_VALUES=("normalization=$norm")
            TABLE_ROW_CONFIG=
            TABLE_COLUMN_CONFIG=normalization
            CASE_DISPLAY_NAME="$name"
            run_case scripts.train_sklearn "$dataset" "$setting" "$method" \
              +model.name=sklinear +normalization.name="$norm" \
              +sklearn.unroll_mode=accessible +sklearn.eval_unroll_mode=accessible \
              +experiment.plot_weights=true
          else
            MODEL_CONFIG_ORDER=normalization
            MODEL_CONFIG_VALUES=("normalization=$norm")
            TABLE_ROW_CONFIG=
            TABLE_COLUMN_CONFIG=normalization
            CASE_DISPLAY_NAME="$name"
            run_case scripts.experiment "$dataset" "$setting" "$method" \
              +model.name="$method" +model.path="$method" +normalization.name="$norm" \
              +experiment.plot_weights=true
          fi
        done
      done
    done
  done
}

run_tables() {
  write_table combined mse "$(IFS=,; echo "${METHODS[*]}")"
}

TABLE_REQUIRED_OUTPUTS=("$OUT_ROOT/results_combined_mse.tex")
TABLE_EXPECTED_METHODS=("${METHODS[@]}")
log_section "workflow start family=linear_models mode=$EXPERIMENT_MODE stages=$STAGES_SPEC"
source "$ROOT/src/slurm/stage_train.sh"
source "$ROOT/src/slurm/stage_tables.sh"
log_section "workflow done family=linear_models output=$OUT_ROOT"
