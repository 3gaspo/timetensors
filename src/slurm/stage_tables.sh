#!/bin/bash
# Execute the workflow's table stage and retain a signature-aware completion
# marker. The sourced family orchestrator defines the table inputs and outputs.

tables_stage_current() {
  local marker="$WORKFLOW_STATE_DIR/tables.complete"
  local required
  [ "$SKIP_COMPLETED" = true ] || return 1
  [ -s "$marker" ] || return 1
  [ "$(head -n 1 "$marker")" = "$TABLE_STAGE_SIGNATURE" ] || return 1
  for required in "${TABLE_REQUIRED_OUTPUTS[@]}"; do
    [ -s "$required" ] || return 1
  done
  if find "$OUT_ROOT" -type f -name "$TABLE_INPUT_NAME" -newer "$marker" -print -quit 2>/dev/null | grep -q .; then
    return 1
  fi
  return 0
}

if stage_requested tables; then
  if [ ! -s "$WORKFLOW_STATE_DIR/train.complete" ] ||
    [ "$(head -n 1 "$WORKFLOW_STATE_DIR/train.complete" 2>/dev/null || true)" != "$TRAIN_STAGE_SIGNATURE" ]; then
    log_error "table stage requires a signature-matched completed training stage"
    exit 1
  fi
  if declare -F verify_table_inputs >/dev/null && ! verify_table_inputs; then
    log_error "table stage requires every selected training result; rerun with STAGES=train,tables"
    exit 1
  fi
  if tables_stage_current; then
    log "stage skip name=tables reason=complete signature=$TABLE_STAGE_SIGNATURE"
  else
    log_section "stage start name=tables"
    run_tables
    for required in "${TABLE_REQUIRED_OUTPUTS[@]}"; do
      if [ ! -s "$required" ]; then
        log_error "table stage completed without required output $required"
        exit 1
      fi
    done
    mkdir -p "$WORKFLOW_STATE_DIR"
    printf '%s\n' "$TABLE_STAGE_SIGNATURE" > "$WORKFLOW_STATE_DIR/tables.complete.tmp"
    mv "$WORKFLOW_STATE_DIR/tables.complete.tmp" "$WORKFLOW_STATE_DIR/tables.complete"
    log_section "stage done name=tables"
  fi
else
  log "stage skip name=tables reason=not_requested"
fi
