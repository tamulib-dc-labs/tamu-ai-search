#!/bin/bash
# Run on the LAPTOP (WSL Ubuntu), next to the local Solr. Exports the data files and bundles them with the code
# into one archive to copy to Grace:
#   bash grace/pack.sh             -> /tmp/tamu-ai-grace.tgz
# Then from Git Bash or WSL:
#   scp /tmp/tamu-ai-grace.tgz <NetID>@grace.hprc.tamu.edu:/scratch/user/<NetID>/
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TAMU_AI="${TAMU_AI:-$HOME/.venvs/tamu-ai/bin/tamu-ai}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/tamu-ai-data"
"$TAMU_AI" export --out "$STAGE/tamu-ai-data/chunks.jsonl"
"$TAMU_AI" export-eval --out "$STAGE/tamu-ai-data/evalpack.json"

# Code without caches, results or the local git history.
tar -C "$(dirname "$REPO")" -cf "$STAGE/code.tar" \
    --exclude='.git' --exclude='__pycache__' --exclude='*.egg-info' --exclude='.pytest_cache' \
    --exclude='eval/results' --exclude='grace/logs' "$(basename "$REPO")"
tar -C "$STAGE" -xf "$STAGE/code.tar" && rm "$STAGE/code.tar"

tar -C "$STAGE" -czf /tmp/tamu-ai-grace.tgz "$(basename "$REPO")" tamu-ai-data
ls -lh /tmp/tamu-ai-grace.tgz
echo "On Grace: cd \$SCRATCH && tar xzf tamu-ai-grace.tgz   (gives tamu-ai-search/ and tamu-ai-data/)"
