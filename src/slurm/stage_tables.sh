#!/bin/bash
# Execute the workflow's table stage and retain a signature-aware completion
# marker. The sourced family orchestrator defines the table inputs and outputs.

if stage_requested tables; then
  log_section "stage start name=tables"
  run_tables
  for required in "${TABLE_REQUIRED_OUTPUTS[@]}"; do
    if [ ! -s "$required" ]; then
      log_error "table stage completed without required output $required"
      exit 1
    fi
  done
  log_section "stage done name=tables"
else
  log "stage skip name=tables reason=not_requested"
fi
