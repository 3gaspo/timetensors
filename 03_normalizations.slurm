#!/bin/bash
#SBATCH --job-name=norms
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# Complete resumable workflow; implementation: src/slurm/benchmark_normalizations.sh.
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
STAGES="${STAGES:-train,tables}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE STAGES
source "$PROJECT_ROOT/src/slurm/benchmark_normalizations.sh"
