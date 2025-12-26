#!/bin/bash
#SBATCH -J finetune
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -A ${TACC_PROJECT}
#SBATCH -o slurm_%A_%a.out
#SBATCH -e slurm_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -d benchmarks || { echo "ERROR: submit from repo root"; exit 1; }

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MANIFEST=${MANIFEST:?set MANIFEST env var}
python -u benchmarks/finetune/run_manifest_entry.py \
  --manifest "$MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID"
