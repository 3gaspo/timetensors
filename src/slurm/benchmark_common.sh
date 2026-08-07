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
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
STAGES_SPEC="${STAGES:-train,tables}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
RUN_CONFLICT_POLICY="${RUN_CONFLICT_POLICY:-overwrite_exact}"
FORCE_RUN="${FORCE_RUN:-false}"
TABLE_CONFIG_POLICY="${TABLE_CONFIG_POLICY:-distinct}"
TABLE_REPEAT_POLICY="${TABLE_REPEAT_POLICY:-selected}"
if [ "$EXPERIMENT_MODE" = test ]; then TABLE_PURPOSE="${TABLE_PURPOSE:-smoke}"; else TABLE_PURPOSE="${TABLE_PURPOSE:-publication}"; fi
EXPERIMENT_LAUNCH_ID="${EXPERIMENT_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
export EXPERIMENT_LAUNCH_ID
trap 'status=$?; if [ "$status" -ne 0 ]; then python -m experiment_runs interrupt-launch --root "$ROOT/outputs" --launch-id "$EXPERIMENT_LAUNCH_ID" || true; fi' EXIT

case "$EXPERIMENT_MODE" in
  test)
    EPOCHS="${EPOCHS:-20}"
    VALID_EVAL_FREQ="${VALID_EVAL_FREQ:-10}"
    LOGGING_EVAL_FREQ="${LOGGING_EVAL_FREQ:-10}"
    read -ra DATASETS <<< "${DATASETS_OVERRIDE:-electricity}"
    read -ra SETTINGS <<< "${SETTINGS_OVERRIDE:-504:168}"
    read -ra SEEDS <<< "${SEEDS_OVERRIDE:-1}"
    read -ra MODELS <<< "${MODELS_OVERRIDE:-patchtst}"
    ;;
  full)
    EPOCHS="${EPOCHS:-10000}"
    VALID_EVAL_FREQ="${VALID_EVAL_FREQ:-1000}"
    LOGGING_EVAL_FREQ="${LOGGING_EVAL_FREQ:-1000}"
    read -ra DATASETS <<< "${DATASETS_OVERRIDE:-ETTh1 electricity traffic solar weather exchange_rate}"
    read -ra SETTINGS <<< "${SETTINGS_OVERRIDE:-168:24 336:48 504:168}"
    read -ra SEEDS <<< "${SEEDS_OVERRIDE:-1 2 3}"
    read -ra MODELS <<< "${MODELS_OVERRIDE:-patchtst}"
    ;;
  ultra)
    EPOCHS="${EPOCHS:-10000}"
    VALID_EVAL_FREQ="${VALID_EVAL_FREQ:-1000}"
    LOGGING_EVAL_FREQ="${LOGGING_EVAL_FREQ:-1000}"
    read -ra DATASETS <<< "${DATASETS_OVERRIDE:-ETTh1 electricity traffic solar weather exchange_rate}"
    read -ra SETTINGS <<< "${SETTINGS_OVERRIDE:-168:24 336:48 504:168}"
    read -ra SEEDS <<< "${SEEDS_OVERRIDE:-1 2 3}"
    read -ra MODELS <<< "${MODELS_OVERRIDE:-patchtst dlinear}"
    ;;
  *)
    log_error "unknown EXPERIMENT_MODE=$EXPERIMENT_MODE expected=test,full,ultra"
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
read -ra STAGE_LIST <<< "${STAGES_SPEC//,/ }"
declare -A DATASET_BUILT

stage_requested() {
  local wanted="$1" stage
  for stage in "${STAGE_LIST[@]}"; do
    [ "$stage" = "$wanted" ] && return 0
  done
  return 1
}
for stage in "${STAGE_LIST[@]}"; do
  case "$stage" in
    train|tables) ;;
    *) log_error "STAGES must contain only train,tables (got $STAGES_SPEC)"; exit 2 ;;
  esac
done

log_section "benchmark workflow mode=$EXPERIMENT_MODE stages=$STAGES_SPEC skip_completed=$SKIP_COMPLETED datasets=$DATASETS_CSV settings=$SETTINGS_CSV models=${MODELS[*]} seeds=$SEEDS_CSV data_root=$DATA_ROOT weights_root=$WEIGHTS_ROOT train_sampling=$TRAIN_MODE batch_size=$BS learning_rate=$LR epochs=$EPOCHS valid_eval_frequency=$VALID_EVAL_FREQ logging_frequency=$LOGGING_EVAL_FREQ drop_train_constant_users=$DROP_TRAIN_CONSTANT_USERS drop_eval_constant_users=$DROP_EVAL_CONSTANT_USERS"

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

dataset_has_tensor_payload() {
  local dataset_directory="$1" values_file
  [ -d "$dataset_directory" ] || return 1
  values_file="$(find "$dataset_directory" -maxdepth 1 -type f -name '*values.pt' -print -quit 2>/dev/null || true)"
  [ -n "$values_file" ]
}

run_case() {
  local module="$1" dataset="$2" setting="$3" backbone="$4"
  shift 4
  local lags="${setting%%:*}" horizon="${setting##*:}"
  local case_data_root config_path seed seed_root run_seeds_csv identity_root pair value
  local run_dir run_action run_signature purpose effective_batch
  local -a allocation_args pending_seeds required_artifacts
  : "${FAMILY:?family launcher must set FAMILY}"
  : "${MODEL_CONFIG_ORDER:=}"
  : "${TABLE_ROW_CONFIG:=}"
  : "${TABLE_COLUMN_CONFIG:=}"
  : "${CASE_DISPLAY_NAME:=$backbone}"
  effective_batch="${CASE_BATCH_SIZE:-$BS}"
  case_data_root="$(resolve_dataset_root "$dataset")"
  config_path="$case_data_root/$dataset/config.json"
  identity_root="$OUT_ROOT/$dataset/${lags}_${horizon}/${backbone,,}"
  for pair in "${MODEL_CONFIG_VALUES[@]}"; do
    value="${pair#*=}"
    value="${value,,}"
    identity_root="$identity_root/${value// /-}"
  done
  if [ "$EXPERIMENT_MODE" = test ]; then purpose=smoke; else purpose=publication; fi
  allocation_args=(
    --identity-root "$identity_root" --project timetensors --workflow "$FAMILY"
    --dataset "$dataset" --lookback "$lags" --horizon "$horizon" --backbone "$backbone"
    --model-config-order "$MODEL_CONFIG_ORDER" --purpose "$purpose" --mode "$EXPERIMENT_MODE"
    --display-name "$CASE_DISPLAY_NAME" --row-config "$TABLE_ROW_CONFIG" --column-config "$TABLE_COLUMN_CONFIG"
    --pipeline-config "entrypoint=$module" --pipeline-config "data.date_splits=0.6,0.2,0.2"
    --pipeline-config "data.train_idx_mode=$TRAIN_MODE" --pipeline-config "data.eval_idx_mode=all"
    --pipeline-config "data.train_stride=1" --pipeline-config "data.eval_stride=$horizon"
    --pipeline-config "data.drop_train_constant_users=$DROP_TRAIN_CONSTANT_USERS"
    --pipeline-config "data.drop_eval_constant_users=$DROP_EVAL_CONSTANT_USERS"
    --pipeline-config "training.batch_size=$effective_batch" --pipeline-config "training.learning_rate=$LR"
    --pipeline-config "training.epochs=$EPOCHS" --pipeline-config "training.valid_eval_freq=$VALID_EVAL_FREQ"
    --pipeline-config "training.logging_eval_freq=$LOGGING_EVAL_FREQ" --pipeline-config "hydra_overrides=$*"
    --runtime-config "training.device=gpu" --runtime-config "slurm.job_id=${SLURM_JOB_ID:-}"
    --project-root "$ROOT" --policy "$RUN_CONFLICT_POLICY" --skip-completed "$SKIP_COMPLETED"
    --force "$FORCE_RUN" --launch-id "$EXPERIMENT_LAUNCH_ID"
  )
  for pair in "${MODEL_CONFIG_VALUES[@]}"; do allocation_args+=(--model-config "$pair"); done
  for seed in "${SEEDS[@]}"; do allocation_args+=(--seed "$seed"); done
  if [ -f "$config_path" ]; then allocation_args+=(--input "dataset_config=$config_path"); fi
  if [ -n "${RUN_INDEX:-}" ]; then allocation_args+=(--run-index "$RUN_INDEX"); fi
  IFS=$'\t' read -r run_dir run_action run_signature < <(python -m experiment_runs allocate "${allocation_args[@]}")
  if [ "$run_action" = skip ]; then
    log "skip complete dataset=$dataset lags=$lags horizon=$horizon backbone=$backbone model_configs=${MODEL_CONFIG_VALUES[*]:-none} run=$run_dir"
    return
  fi
  run_seeds_csv="$(python -m experiment_runs pending-seeds --run-dir "$run_dir")"
  IFS=, read -ra pending_seeds <<< "$run_seeds_csv"
  if [ "${#pending_seeds[@]}" -eq 0 ] || [ -z "${pending_seeds[0]:-}" ]; then
    log_error "allocated non-skipped run has no pending seeds: $run_dir"
    exit 1
  fi
  local rebuild=false rebuild_reason=not_needed
  if [ -z "${DATASET_BUILT[$dataset]:-}" ]; then
    if [ "$REBUILD_DATASETS" = true ]; then
      rebuild=true
      rebuild_reason=forced
    elif ! dataset_has_tensor_payload "$case_data_root/$dataset"; then
      rebuild=true
      rebuild_reason=missing_tensor_payload
    fi
    if [ "$rebuild" = true ]; then
      DATASET_BUILT[$dataset]=1
    fi
  fi
  local config_args=()
  if [ -f "$case_data_root/$dataset/config.json" ]; then
    config_args+=(+data.config_path="$case_data_root/$dataset/config.json")
  fi
  for seed in "${pending_seeds[@]}"; do
    python -m experiment_runs status --run-dir "$run_dir" --status running --seed "$seed"
  done
  log_section "configuration dataset=$dataset lags=$lags horizon=$horizon backbone=$backbone model_configs=${MODEL_CONFIG_VALUES[*]:-none} module=$module requested_seeds=$SEEDS_CSV run_seeds=$run_seeds_csv run=$run_dir computation_signature=$run_signature data_root=$case_data_root train_sampling=$TRAIN_MODE train_stride=1 eval_sampling=all eval_stride=$horizon batch_size=$effective_batch learning_rate=$LR epochs=$EPOCHS rebuild_dataset=$rebuild rebuild_reason=$rebuild_reason overrides=$*"
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
    +training.batch_size="$effective_batch" \
    +training.lr="$LR" \
    +training.epochs="$EPOCHS" \
    +training.valid_eval_freq="$VALID_EVAL_FREQ" \
    +training.logging_eval_freq="$LOGGING_EVAL_FREQ" \
    +training.plot_step_train_loss=false \
    +training.device=gpu \
    +experiment.rebuild_dataset="$rebuild" \
    +experiment.recompute_stats=true \
    +experiment.seeds="[$run_seeds_csv]" \
    +output.dir="$run_dir" \
    +output.name= \
    "$@" \
    hydra.run.dir="$run_dir/hydra/$EXPERIMENT_LAUNCH_ID"
  required_artifacts=()
  for seed in "${pending_seeds[@]}"; do
    seed_root="$run_dir/seed_$seed"
    if [ ! -s "$seed_root/all_losses.pt" ]; then
      log_error "training completed without required result $seed_root/all_losses.pt"
      exit 1
    fi
    python -m experiment_runs status --run-dir "$run_dir" --status completed --seed "$seed" --artifact "seed_$seed/all_losses.pt"
  done
  for seed in "${SEEDS[@]}"; do required_artifacts+=(--artifact "seed_$seed/all_losses.pt"); done
  python -m experiment_runs status --run-dir "$run_dir" --status completed "${required_artifacts[@]}"
  unset CASE_BATCH_SIZE CASE_DISPLAY_NAME
  MODEL_CONFIG_VALUES=()
}

write_table() {
  local model="$1" metric="$2" methods="$3"
  local pair
  local -a table_args
  log_section "table model=$model metric=$metric methods=$methods output=$OUT_ROOT/results_${model}_${metric}.tex"
  table_args=(
    "$OUT_ROOT" --split test1 --metric "$metric"
    --datasets "$DATASETS_CSV" --settings "$SETTINGS_CSV"
    --methods "$methods" --show-std
    --config-policy "$TABLE_CONFIG_POLICY" --repeat-policy "$TABLE_REPEAT_POLICY"
    --output "$OUT_ROOT/results_${model}_${metric}.tex"
  )
  if [ -n "${TABLE_PIPELINE_CONFIGS:-}" ]; then
    for pair in ${TABLE_PIPELINE_CONFIGS}; do table_args+=(--pipeline-config "$pair"); done
  fi
  if [ -n "${TABLE_PURPOSE:-}" ]; then table_args+=(--purpose "$TABLE_PURPOSE"); fi
  srun --ntasks=1 python -m visu.results_table "${table_args[@]}"
}

verify_table_inputs() {
  find "$OUT_ROOT" -type f -name manifest.json -not -path '*/archive/*' -print -quit | grep -q .
}
