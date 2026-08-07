#!/bin/bash
# Execute the workflow's training stage. The sourced family orchestrator
# defines stage_requested, log_section, and run_training.

if stage_requested train; then
  log_section "stage start name=train"
  run_training
  log_section "stage done name=train"
else
  log "stage skip name=train reason=not_requested"
fi
