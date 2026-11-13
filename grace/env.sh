# Shared settings for Grace, used by setup.sh and the job scripts (`source env.sh`).

# Python from Grace's module system.
module purge
module load GCCcore/13.2.0 Python/3.11.5

export VENV=$SCRATCH/venvs/tamu-ai      # our Python packages
export DATA=$SCRATCH/tamu-ai-data       # chunks.jsonl, questions.json, vectors
export HF_HOME=$SCRATCH/hf              # downloaded models

# $HOME only has 10 GB, so keep pip's and other caches on scratch.
export PIP_CACHE_DIR=$SCRATCH/.cache/pip
export XDG_CACHE_HOME=$SCRATCH/.cache
