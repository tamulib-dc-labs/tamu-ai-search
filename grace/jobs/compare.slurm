#!/bin/bash
# Compare embedding models, keyword weights and rerankers on the test questions (one A100, about an hour).
# Submit from the grace folder:   cd $SCRATCH/tamu-ai-search/grace && sbatch jobs/compare.slurm
# Override the lists without editing:
#   sbatch --export=ALL,MODELS="BAAI/bge-base-en-v1.5",RERANKERS="BAAI/bge-reranker-base" jobs/compare.slurm
#SBATCH --job-name=tamu-ai-compare
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x.%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
activate
export HF_HUB_OFFLINE=1  # models were downloaded by setup.sh; compute nodes have no internet

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

MODELS="${MODELS:-BAAI/bge-base-en-v1.5 BAAI/bge-m3 codefuse-ai/F2LLM-0.6B}"
RERANKERS="${RERANKERS:-BAAI/bge-reranker-base BAAI/bge-reranker-v2-m3}"
RUN="$TAMU_AI_DATA/runs/compare-$(date +%Y%m%d-%H%M)-$SLURM_JOB_ID"
mkdir -p "$RUN"

# shellcheck disable=SC2086
tamu-ai compare \
    --chunks "$TAMU_AI_DATA/chunks.jsonl" \
    --evalpack "$TAMU_AI_DATA/evalpack.json" \
    --models $MODELS \
    --rerankers $RERANKERS \
    --keyword-weights 1.0 0.5 0.25 \
    --rerank-depths 20 50 \
    --cache-dir "$TAMU_AI_DATA/vectors-cache" \
    --out "$RUN/report.json" | tee "$RUN/summary.txt"
