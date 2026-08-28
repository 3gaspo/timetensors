#!/bin/bash
#SBATCH --job-name=s_norms
#SBATCH --gres=gpu:1
#SBATCH --output=logs_selena/%x_%j.out
#SBATCH --error=logs_selena/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=an
#SBATCH --qos=an_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --exclusive
#SBATCH --wckey=P12CU:DATASCIENCE

# Complete resumable workflow; implementation: src/slurm/benchmark_normalizations.sh.
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
STAGES="${STAGES:-train,tables}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUTPUTS_ROOT="$PROJECT_ROOT/outputs_selena"
LOGS_ROOT="$PROJECT_ROOT/logs_selena"
EXPERIMENT_LAUNCH_ID="selena_${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE STAGES OUTPUTS_ROOT LOGS_ROOT EXPERIMENT_LAUNCH_ID
source "$PROJECT_ROOT/src/slurm/benchmark_normalizations.sh"
