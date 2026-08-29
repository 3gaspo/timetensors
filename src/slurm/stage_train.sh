#!/bin/bash
# Execute the workflow's training stage. The sourced family orchestrator
# defines stage_requested, log_section, and run_training.

if stage_requested train; then
  stage_start train
  run_training
  stage_complete
else
  log "stage skip name=train reason=not_requested"
fi
