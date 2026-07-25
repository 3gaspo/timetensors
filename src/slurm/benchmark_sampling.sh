#!/bin/bash
# Compare random/individual sampling and batch-size choices.
set -euo pipefail
TEST_MODE="${TEST_MODE:-false}"
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_common.sh"
OUT_ROOT="$ROOT/outputs/sampling"
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
elif [ "$BENCHMARK_PROFILE" = full ]; then
  for mode in random dates individuals all; do
    for batch in 64 256 1024; do
      SAMPLING_CASES+=("${mode}:${batch}")
    done
  done
elif [ "$BENCHMARK_PROFILE" = study ]; then
  SAMPLING_CASES=(random:64 random:256 random:1024 individuals:256)
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

if [ "$RUN_MODE" != tables ]; then
  for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      for model in "${MODELS[@]}"; do
        for sampling_case in "${SAMPLING_CASES[@]}"; do
          mode="${sampling_case%%:*}"
          batch="${sampling_case##*:}"
          run_case scripts.experiment "$dataset" "$setting" "${model}_${mode}_bs${batch}" \
            +model.name="$model" +model.path="$model" +normalization.name=instance \
            ++data.sampling.train_idx_mode="$mode" ++training.batch_size="$batch"
        done
      done
    done
  done
fi
if [ "$RUN_MODE" != train ]; then
  for model in "${MODELS[@]}"; do
    methods=(); for sampling_case in "${SAMPLING_CASES[@]}"; do mode="${sampling_case%%:*}"; batch="${sampling_case##*:}"; methods+=("${model}_${mode}_bs${batch}"); done
    write_table "$model" mse "$(IFS=,; echo "${methods[*]}")"
  done
fi
