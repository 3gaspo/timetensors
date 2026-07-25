#!/bin/bash
# Shared TimeTensors resource resolution, configuration, launch, and table helpers.

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT/src"

log() { printf '%s %s\n' "$(date -Is)" "$*"; }
log_section() { printf '\n%s %s\n' "$(date -Is)" "$*"; }
log_error() { printf '%s %s\n' "$(date -Is)" "$*" >&2; }

# Storage may be project-local or in a shared parent directory. Set DATA_ROOT
# and WEIGHTS_ROOT explicitly on another machine, or edit these candidates.
directory_has_payload() {
  local directory="$1" first
  [ -d "$directory" ] || return 1
  first="$(find "$directory" -mindepth 1 ! -name .gitkeep -print -quit 2>/dev/null || true)"
  [ -n "$first" ]
}

resolve_storage_root() {
  local explicit="$1" local_root="$2" parent_root="$3" workspace_root="$4"
  if [ -n "$explicit" ]; then
    echo "$explicit"
  elif directory_has_payload "$local_root"; then
    echo "$local_root"
  elif directory_has_payload "$parent_root"; then
    echo "$parent_root"
  elif directory_has_payload "$workspace_root"; then
    echo "$workspace_root"
  else
    echo "$local_root"
  fi
}

SHARED_ROOT="$(cd "$ROOT/../../.." 2>/dev/null && pwd || true)"
DATA_ROOT_EXPLICIT="${DATA_ROOT:-}"
WEIGHTS_ROOT_EXPLICIT="${WEIGHTS_ROOT:-}"
DATA_ROOT="$(resolve_storage_root "$DATA_ROOT_EXPLICIT" "$ROOT/datasets" "$ROOT/../datasets" "$SHARED_ROOT/datasets")"
WEIGHTS_ROOT="$(resolve_storage_root "$WEIGHTS_ROOT_EXPLICIT" "$ROOT/weights" "$ROOT/../weights" "$SHARED_ROOT/weights")"
LR="${LR:-1e-5}"
BS="${BS:-256}"
TRAIN_MODE="${TRAIN_MODE:-random}"
DROP_TRAIN_CONSTANT_USERS="${DROP_TRAIN_CONSTANT_USERS:-true}"
DROP_EVAL_CONSTANT_USERS="${DROP_EVAL_CONSTANT_USERS:-true}"
REBUILD_DATASETS="${REBUILD_DATASETS:-false}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-test}"
RUN_MODE="${RUN_MODE:-both}"

case "$RUN_MODE" in
  train|tables|both) ;;
  *)
    log_error "unknown RUN_MODE=$RUN_MODE expected=train,tables,both"
    exit 2
    ;;
esac

if [ "$TEST_MODE" = true ]; then
  BENCHMARK_PROFILE=test
fi

case "$BENCHMARK_PROFILE" in
  test)
    EPOCHS="${EPOCHS:-20}"
    VALID_EVAL_FREQ="${VALID_EVAL_FREQ:-10}"
    LOGGING_EVAL_FREQ="${LOGGING_EVAL_FREQ:-10}"
    read -ra DATASETS <<< "${DATASETS_OVERRIDE:-electricity}"
    read -ra SETTINGS <<< "${SETTINGS_OVERRIDE:-168:24}"
    read -ra SEEDS <<< "${SEEDS_OVERRIDE:-1}"
    read -ra MODELS <<< "${MODELS_OVERRIDE:-dlinear}"
    ;;
  study)
    EPOCHS="${EPOCHS:-10000}"
    VALID_EVAL_FREQ="${VALID_EVAL_FREQ:-1000}"
    LOGGING_EVAL_FREQ="${LOGGING_EVAL_FREQ:-1000}"
    read -ra DATASETS <<< "${DATASETS_OVERRIDE:-etth1 electricity traffic solar weather exchange_rate}"
    read -ra SETTINGS <<< "${SETTINGS_OVERRIDE:-168:24 504:168 504:504}"
    read -ra SEEDS <<< "${SEEDS_OVERRIDE:-1 2 3}"
    read -ra MODELS <<< "${MODELS_OVERRIDE:-dlinear}"
    ;;
  full)
    EPOCHS="${EPOCHS:-10000}"
    VALID_EVAL_FREQ="${VALID_EVAL_FREQ:-1000}"
    LOGGING_EVAL_FREQ="${LOGGING_EVAL_FREQ:-1000}"
    read -ra DATASETS <<< "${DATASETS_OVERRIDE:-etth1 electricity traffic solar weather exchange_rate}"
    read -ra SETTINGS <<< "${SETTINGS_OVERRIDE:-168:24 168:168 504:24 504:168 504:504 720:168 720:720}"
    read -ra SEEDS <<< "${SEEDS_OVERRIDE:-1 2 3 4 5}"
    read -ra MODELS <<< "${MODELS_OVERRIDE:-dlinear patchtst}"
    ;;
  *)
    log_error "unknown BENCHMARK_PROFILE=$BENCHMARK_PROFILE expected=test,study,full"
    exit 2
    ;;
esac

SEEDS_CSV="$(IFS=,; echo "${SEEDS[*]}")"
DATASETS_CSV="$(IFS=,; echo "${DATASETS[*]}")"
SETTING_NAMES=()
for setting in "${SETTINGS[@]}"; do
  SETTING_NAMES+=("${setting/:/_}")
done
SETTINGS_CSV="$(IFS=,; echo "${SETTING_NAMES[*]}")"
declare -A DATASET_BUILT

log_section "benchmark start profile=$BENCHMARK_PROFILE run_mode=$RUN_MODE datasets=$DATASETS_CSV settings=$SETTINGS_CSV models=${MODELS[*]} seeds=$SEEDS_CSV data_root=$DATA_ROOT weights_root=$WEIGHTS_ROOT train_sampling=$TRAIN_MODE batch_size=$BS learning_rate=$LR epochs=$EPOCHS valid_eval_frequency=$VALID_EVAL_FREQ logging_frequency=$LOGGING_EVAL_FREQ drop_train_constant_users=$DROP_TRAIN_CONSTANT_USERS drop_eval_constant_users=$DROP_EVAL_CONSTANT_USERS"

resolve_dataset_root() {
  local dataset="$1" candidate
  if [ -n "$DATA_ROOT_EXPLICIT" ]; then
    echo "$DATA_ROOT"
    return
  fi
  for candidate in "$ROOT/datasets" "$ROOT/../datasets" "$SHARED_ROOT/datasets"; do
    if [ -d "$candidate/$dataset" ]; then
      echo "$candidate"
      return
    fi
  done
  echo "$DATA_ROOT"
}

resolve_weight_path() {
  local relative_path="$1" candidate
  if [ -n "$WEIGHTS_ROOT_EXPLICIT" ]; then
    echo "$WEIGHTS_ROOT/$relative_path"
    return
  fi
  for candidate in "$ROOT/weights" "$ROOT/../weights" "$SHARED_ROOT/weights"; do
    if [ -e "$candidate/$relative_path" ]; then
      echo "$candidate/$relative_path"
      return
    fi
  done
  echo "$WEIGHTS_ROOT/$relative_path"
}

run_case() {
  local module="$1" dataset="$2" setting="$3" method="$4"
  shift 4
  local lags="${setting%%:*}" horizon="${setting##*:}"
  local case_data_root
  case_data_root="$(resolve_dataset_root "$dataset")"
  local rebuild=false
  if [ "$REBUILD_DATASETS" = true ] && [ -z "${DATASET_BUILT[$dataset]:-}" ]; then
    rebuild=true
    DATASET_BUILT[$dataset]=1
  fi
  local config_args=()
  if [ -f "$case_data_root/$dataset/config.json" ]; then
    config_args+=(+data.config_path="$case_data_root/$dataset/config.json")
  fi
  log_section "configuration dataset=$dataset lags=$lags horizon=$horizon method=$method module=$module seeds=$SEEDS_CSV data_root=$case_data_root train_sampling=$TRAIN_MODE train_stride=1 eval_sampling=all eval_stride=$horizon batch_size=$BS learning_rate=$LR epochs=$EPOCHS rebuild_dataset=$rebuild overrides=$*"
  srun --ntasks=1 python -m "$module" \
    +data.raw_path="$case_data_root/$dataset" \
    +data.path="$case_data_root/$dataset" \
    +data.name="$dataset" \
    "${config_args[@]}" \
    +task.lags="$lags" \
    +task.horizon="$horizon" \
    +data.splits.date_splits='[0.6,0.2,0.2]' \
    +data.sampling.train_idx_mode="$TRAIN_MODE" \
    +data.sampling.eval_idx_mode=all \
    +data.sampling.train_stride=1 \
    +data.sampling.eval_stride="$horizon" \
    +data.sampling.drop_train_constant_individuals="$DROP_TRAIN_CONSTANT_USERS" \
    +data.sampling.drop_eval_constant_individuals="$DROP_EVAL_CONSTANT_USERS" \
    +training.batch_size="$BS" \
    +training.lr="$LR" \
    +training.epochs="$EPOCHS" \
    +training.valid_eval_freq="$VALID_EVAL_FREQ" \
    +training.logging_eval_freq="$LOGGING_EVAL_FREQ" \
    +training.plot_step_train_loss=false \
    +training.device=gpu \
    +experiment.rebuild_dataset="$rebuild" \
    +experiment.recompute_stats=true \
    +experiment.seeds="[$SEEDS_CSV]" \
    +output.dir="$OUT_ROOT/$dataset/${lags}_${horizon}" \
    +output.name="$method" \
    "$@" \
    hydra.run.dir="$OUT_ROOT/hydra/$dataset/${lags}_${horizon}/$method/${SLURM_JOB_ID:-local}"
}

write_table() {
  local model="$1" metric="$2" methods="$3"
  log_section "table model=$model metric=$metric methods=$methods output=$OUT_ROOT/results_${model}_${metric}.tex"
  srun --ntasks=1 python -m visu.results_table "$OUT_ROOT" \
    --split test1 --metric "$metric" \
    --datasets "$DATASETS_CSV" --settings "$SETTINGS_CSV" \
    --methods "$methods" --show-std \
    --output "$OUT_ROOT/results_${model}_${metric}.tex"
}
