#!/bin/bash
# Compare random/individual sampling and batch-size choices.
set -euo pipefail
FAMILY=sampling
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_common.sh"
OUT_ROOT="${OUT_ROOT:-$OUTPUTS_ROOT/sampling}"
SAMPLING_CASES=()
if [ -n "${SAMPLING_CASES_OVERRIDE:-}" ]; then
  read -ra SAMPLING_CASES <<< "$SAMPLING_CASES_OVERRIDE"
elif [ -n "${SAMPLING_MODES_OVERRIDE:-}" ] || [ -n "${BATCH_SIZES_OVERRIDE:-}" ]; then
  read -ra SAMPLING_MODES <<< "${SAMPLING_MODES_OVERRIDE:-random individuals}"
  read -ra BATCH_SIZES <<< "${BATCH_SIZES_OVERRIDE:-256}"
  for mode in "${SAMPLING_MODES[@]}"; do
    for batch in "${BATCH_SIZES[@]}"; do
      SAMPLING_CASES+=("${mode}:${batch}")
    done
  done
elif [ "$EXPERIMENT_MODE" = full ] || [ "$EXPERIMENT_MODE" = ultra ]; then
  for mode in random dates individuals all; do
    for batch in 64 256 1024; do
      SAMPLING_CASES+=("${mode}:${batch}")
    done
  done
else
  SAMPLING_CASES=(random:256)
fi

for sampling_case in "${SAMPLING_CASES[@]}"; do
  mode="${sampling_case%%:*}"
  batch="${sampling_case##*:}"
  if [ "$mode" = "$sampling_case" ] || ! [[ "$batch" =~ ^[1-9][0-9]*$ ]]; then
    log_error "invalid sampling case=$sampling_case expected=mode:positive_batch_size"
    exit 2
  fi
done

run_training() {
  for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      for model in "${MODELS[@]}"; do
        for sampling_case in "${SAMPLING_CASES[@]}"; do
          mode="${sampling_case%%:*}"
          batch="${sampling_case##*:}"
          MODEL_CONFIG_ORDER=sampling_mode,batch_size
          MODEL_CONFIG_VALUES=("sampling_mode=$mode" "batch_size=$batch")
          TABLE_ROW_CONFIG=sampling_mode
          TABLE_COLUMN_CONFIG=batch_size
          CASE_DISPLAY_NAME="${model}_${mode}_bs${batch}"
          CASE_BATCH_SIZE="$batch"
          run_case scripts.experiment "$dataset" "$setting" "$model" \
            +model.name="$model" +model.path="$model" +normalization.name=instance \
            ++data.sampling.train_idx_mode="$mode" ++training.batch_size="$batch"
        done
      done
    done
  done
}

run_tables() {
  for model in "${MODELS[@]}"; do
    methods=(); for sampling_case in "${SAMPLING_CASES[@]}"; do mode="${sampling_case%%:*}"; batch="${sampling_case##*:}"; methods+=("${model}_${mode}_bs${batch}"); done
    write_table "$model" mse "$(IFS=,; echo "${methods[*]}")"
  done
}

TABLE_REQUIRED_OUTPUTS=()
TABLE_EXPECTED_METHODS=()
for model in "${MODELS[@]}"; do
  TABLE_REQUIRED_OUTPUTS+=("$OUT_ROOT/results_${model}_mse.tex")
  for sampling_case in "${SAMPLING_CASES[@]}"; do
    mode="${sampling_case%%:*}"; batch="${sampling_case##*:}"
    TABLE_EXPECTED_METHODS+=("${model}_${mode}_bs${batch}")
  done
done
log_section "workflow start family=sampling mode=$EXPERIMENT_MODE stages=$STAGES_SPEC"
source "$ROOT/src/slurm/stage_train.sh"
source "$ROOT/src/slurm/stage_tables.sh"
log_section "workflow done family=sampling output=$OUT_ROOT"
