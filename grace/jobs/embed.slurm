#!/bin/bash
# Embed every chunk with one model; the .npz goes back to the laptop/cluster for `tamu-ai load`.
# Submit from the grace folder:
#   cd $SCRATCH/tamu-ai-search/grace && sbatch --export=ALL,MODEL=BAAI/bge-base-en-v1.5 jobs/embed.slurm
#SBATCH --job-name=tamu-ai-embed
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x.%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
activate
export HF_HUB_OFFLINE=1

MODEL="${MODEL:-BAAI/bge-base-en-v1.5}"
SLUG="$(echo "$MODEL" | tr -c 'A-Za-z0-9\n' '-' | tr 'A-Z' 'a-z')"
OUT="$TAMU_AI_DATA/vectors-$SLUG.npz"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
time tamu-ai embed --chunks "$TAMU_AI_DATA/chunks.jsonl" --model "$MODEL" --out "$OUT" --batch-size 128
echo "copy back: $OUT"
