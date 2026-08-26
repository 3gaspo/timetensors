#!/bin/bash
# Execute an evaluation-only workflow stage. The sourced family orchestrator
# defines stage_requested, log_section, and run_evaluation.

if stage_requested evaluate; then
  log_section "stage start name=evaluate"
  run_evaluation
  log_section "stage done name=evaluate"
else
  log "stage skip name=evaluate reason=not_requested"
fi
