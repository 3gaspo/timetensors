#!/bin/bash
# Execute the workflow's training stage. The sourced family orchestrator
# defines stage_requested, log_section, and run_training.

training_stage_current() {
  local marker="$WORKFLOW_STATE_DIR/train.complete"
  [ "$SKIP_COMPLETED" = true ] || return 1
  [ -s "$marker" ] || return 1
  [ "$(head -n 1 "$marker")" = "$TRAIN_STAGE_SIGNATURE" ] || return 1
  verify_table_inputs
}

if stage_requested train; then
  if training_stage_current; then
    log "stage skip name=train reason=complete signature=$TRAIN_STAGE_SIGNATURE"
  else
    log_section "stage start name=train"
    run_training
    verify_table_inputs
    mkdir -p "$WORKFLOW_STATE_DIR"
    printf '%s\n' "$TRAIN_STAGE_SIGNATURE" > "$WORKFLOW_STATE_DIR/train.complete.tmp"
    mv "$WORKFLOW_STATE_DIR/train.complete.tmp" "$WORKFLOW_STATE_DIR/train.complete"
    log_section "stage done name=train"
  fi
else
  log "stage skip name=train reason=not_requested"
fi
