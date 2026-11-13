#!/bin/bash
# Compare embedding models, keyword weights and rerankers on the test questions.
# Submit from this folder:  sbatch compare.slurm
# The results table is at the end of compare.<job id>.log
#
# The #SBATCH lines ask Slurm for resources: 1 A100 GPU, 8 CPU cores, 64 GB memory, up to 2 hours.
#SBATCH --job-name=compare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=compare.%j.log

source ./env.sh
source "$VENV/bin/activate"
module load WebProxy   # compute nodes reach the internet (Hugging Face) only through this proxy

nvidia-smi
tamu-ai compare "$DATA" \
    --models BAAI/bge-base-en-v1.5 BAAI/bge-m3 codefuse-ai/F2LLM-0.6B \
    --rerankers BAAI/bge-reranker-base BAAI/bge-reranker-v2-m3 \
    --keyword-weights 1.0 0.5 0.25
