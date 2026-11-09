# Shared settings for Grace. Sourced by setup.sh and the Slurm jobs; change values here, not in each script.

# Where things live. $SCRATCH (/scratch/user/<NetID>) has room; $HOME is small. Scratch is NOT backed up.
export TAMU_AI_HOME="${TAMU_AI_HOME:-$SCRATCH/tamu-ai-search}"   # the code, copied from the laptop
export TAMU_AI_VENV="${TAMU_AI_VENV:-$SCRATCH/venvs/tamu-ai}"     # Python packages
export TAMU_AI_DATA="${TAMU_AI_DATA:-$SCRATCH/tamu-ai-data}"      # chunks, evalpacks, vectors, reports
export HF_HOME="${HF_HOME:-$SCRATCH/hf}"                          # downloaded models

# Python module. Check what Grace offers with `module spider Python` and change this line if needed.
export TAMU_AI_PY_MODULES="${TAMU_AI_PY_MODULES:-GCCcore/13.2.0 Python/3.11.5}"
# PyTorch build. CUDA 12.x wheels need a recent NVIDIA driver; a GPU job prints the driver version.
export TAMU_AI_TORCH_INDEX="${TAMU_AI_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

load_python() {
    module purge
    # shellcheck disable=SC2086
    module load $TAMU_AI_PY_MODULES
}

activate() {
    load_python
    # shellcheck disable=SC1091
    source "$TAMU_AI_VENV/bin/activate"
}
