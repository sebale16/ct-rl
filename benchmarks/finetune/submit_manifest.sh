#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <system:{stampede3|ls6|vista|ls6-gpu-a100|vista-gpu-gh}> <manifest.jsonl> <max_parallel>"
  exit 1
fi

SYSTEM="$1"
MANIFEST_PATH="$2"
MAX_PARALLEL="$3"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Manifest not found: $MANIFEST_PATH"
  exit 1
fi

N=$(wc -l < "$MANIFEST_PATH")
if [[ "$N" -le 0 ]]; then
  echo "Empty manifest: $MANIFEST_PATH"
  exit 1
fi

export MANIFEST="$MANIFEST_PATH"

case "$SYSTEM" in
  stampede3)
    SBATCH_FILE="benchmarks/finetune/slurm/stampede3_cpu.sbatch"
    ;;
  ls6)
    SBATCH_FILE="benchmarks/finetune/slurm/ls6_cpu.sbatch"
    ;;
  vista)
    SBATCH_FILE="benchmarks/finetune/slurm/vista_cpu.sbatch"
    ;;
  ls6-gpu-a100)
    SBATCH_FILE="benchmarks/finetune/slurm/ls6_gpu_a100.sbatch"
    ;;
  vista-gpu-gh)
    SBATCH_FILE="benchmarks/finetune/slurm/vista_gh_gpu.sbatch"
    ;;
  *)
    echo "Unknown system: $SYSTEM"
    exit 1
    ;;
esac

echo "Submitting $N jobs via $SBATCH_FILE with max parallel = $MAX_PARALLEL"
sbatch --array=0-$((N-1))%${MAX_PARALLEL} "$SBATCH_FILE"
