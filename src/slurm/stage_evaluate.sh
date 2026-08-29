#!/bin/bash
# Execute an evaluation-only workflow stage. The sourced family orchestrator
# defines stage_requested, log_section, and run_evaluation.

if stage_requested evaluate; then
  stage_start evaluate
  run_evaluation
  stage_complete
else
  log "stage skip name=evaluate reason=not_requested"
fi
