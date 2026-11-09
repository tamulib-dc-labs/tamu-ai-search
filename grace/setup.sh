#!/bin/bash
# One-time setup, run on a Grace LOGIN node (it has internet; compute nodes do not, and it has no GPU).
#   cd $SCRATCH/tamu-ai-search && bash grace/setup.sh
# Safe to run again: it reuses the venv and skips models already downloaded.
set -euo pipefail
source "$(dirname "$0")/env.sh"

load_python
echo "Python: $(python3 --version)"
[ -d "$TAMU_AI_VENV" ] || python3 -m venv "$TAMU_AI_VENV"
# shellcheck disable=SC1091
source "$TAMU_AI_VENV/bin/activate"

pip install --upgrade pip
pip install torch --index-url "$TAMU_AI_TORCH_INDEX"
pip install -e "$TAMU_AI_HOME/service[dev]"

mkdir -p "$TAMU_AI_DATA" "$HF_HOME" "$TAMU_AI_HOME/grace/logs"

# Every model the jobs may use, fetched now because jobs run offline.
tamu-ai download-models \
    BAAI/bge-base-en-v1.5 BAAI/bge-m3 codefuse-ai/F2LLM-0.6B \
    BAAI/bge-reranker-base BAAI/bge-reranker-v2-m3

(cd "$TAMU_AI_HOME/service" && python -m pytest -q)
echo "Setup done. Data files go in $TAMU_AI_DATA; submit jobs from $TAMU_AI_HOME/grace."
