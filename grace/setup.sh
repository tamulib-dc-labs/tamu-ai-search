#!/bin/bash
# One-time setup on a Grace login node:
#   cd $SCRATCH/tamu-ai-search/grace && bash setup.sh
# Run it again after pulling new code if service/pyproject.toml changed.
set -e
cd "$(dirname "$0")"
source ./env.sh

# A virtual environment keeps our packages separate from everything else on Grace.
[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip
# PyTorch built for NVIDIA GPUs (CUDA 12.4), then our service and its dependencies.
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e "../service[dev]"

mkdir -p "$DATA"
echo "Done. Copy chunks.jsonl and questions.json into $DATA, then: sbatch compare.slurm"
