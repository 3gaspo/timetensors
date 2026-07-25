#!/bin/bash
#SBATCH --job-name=losses
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# Submit from the project root; implementation: src/slurm/benchmark_losses.sh.
TEST_MODE="${TEST_MODE:-false}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-test}"
RUN_MODE="${RUN_MODE:-both}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT TEST_MODE BENCHMARK_PROFILE RUN_MODE
source "$PROJECT_ROOT/src/slurm/benchmark_losses.sh"
