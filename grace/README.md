# Running on HPRC Grace

The laptop is for writing code. Grace does the heavy work on A100 GPUs: embedding every chunk, comparing models,
and later fine-tuning. Grace cannot reach our Solr, so everything moves as files.

```
laptop (WSL, next to local Solr)         Grace (login node + GPU jobs)              laptop
--------------------------------         -----------------------------              ------
bash grace/pack.sh                 --->  tar xzf, bash grace/setup.sh (once)
  code + chunks.jsonl + evalpack         sbatch jobs/compare.slurm   -> report
                                         sbatch jobs/embed.slurm     -> vectors  ---> tamu-ai load
```

## Before the first run

1. An HPRC account with an allocation on Grace (apply at hprc.tamu.edu).
2. Off campus: connect to the TAMU VPN first. Logging in asks for Duo.
3. Log in once from WSL to check: `ssh <NetID>@grace.hprc.tamu.edu`

## 1. Pack on the laptop (WSL Ubuntu, local stack running)

```bash
cd /mnt/c/lib_work/tamu-ai-search
bash grace/pack.sh                     # exports chunks + test questions, writes /tmp/tamu-ai-grace.tgz
scp /tmp/tamu-ai-grace.tgz <NetID>@grace.hprc.tamu.edu:/scratch/user/<NetID>/
```

## 2. Unpack and set up on Grace (login node, once)

```bash
cd $SCRATCH && tar xzf tamu-ai-grace.tgz       # -> $SCRATCH/tamu-ai-search and $SCRATCH/tamu-ai-data
module spider Python                           # pick a Python 3.11+ module; edit grace/env.sh if the default is missing
cd $SCRATCH/tamu-ai-search && bash grace/setup.sh
```

`setup.sh` makes a virtual environment in `$SCRATCH/venvs/tamu-ai`, installs PyTorch (CUDA) and the service,
downloads the models into `$SCRATCH/hf` (compute nodes have no internet), and runs the unit tests.

After a code change, re-pack and unpack again; the venv and models stay, so only `pip install -e` is needed
if `pyproject.toml` changed.

## 3. Compare models (GPU job)

```bash
cd $SCRATCH/tamu-ai-search/grace
sbatch jobs/compare.slurm           # prints "Submitted batch job <id>"
squeue -u $USER                     # PD = waiting for a GPU, R = running
tail -f logs/tamu-ai-compare.<id>.out
```

It scores keyword search, each embedding model (bge-base, bge-m3, F2LLM-0.6B), hybrid search at keyword weights
1, 0.5 and 0.25, and each reranker on the top 20 and 50. Results land in `$TAMU_AI_DATA/runs/compare-*/`
(`summary.txt` is the table, `report.json` has per-question and per-style detail). Copy them back with:

```bash
scp -r <NetID>@grace.hprc.tamu.edu:/scratch/user/<NetID>/tamu-ai-data/runs ./grace-runs
```

To try other models, download them on the login node first (`tamu-ai download-models <name>`), then
`sbatch --export=ALL,MODELS="<a> <b>",RERANKERS="<c>" jobs/compare.slurm`.

## 4. Embed with the chosen model and load into Solr

```bash
# Grace
sbatch --export=ALL,MODEL=BAAI/bge-base-en-v1.5 jobs/embed.slurm
# laptop (WSL): copy the .npz back, then
TAMU_AI_EMBED_MODEL=BAAI/bge-base-en-v1.5 ~/.venvs/tamu-ai/bin/tamu-ai load \
    --chunks <chunks.jsonl> --vectors vectors-baai-bge-base-en-v1-5.npz
```

A model with a different vector size (bge-m3 and F2LLM-0.6B use 1024) needs `TAMU_AI_EMBED_DIM=1024` and a fresh
`ai_chunks` core (`tamu-ai setup` adds a `knn_vector_1024` type, but the `vector` field keeps its first type).

## Settings (grace/env.sh)

| Variable | Default | Meaning |
| --- | --- | --- |
| `TAMU_AI_HOME` | `$SCRATCH/tamu-ai-search` | code |
| `TAMU_AI_DATA` | `$SCRATCH/tamu-ai-data` | chunks, evalpack, vectors, runs |
| `TAMU_AI_VENV` | `$SCRATCH/venvs/tamu-ai` | Python packages |
| `HF_HOME` | `$SCRATCH/hf` | model files |
| `TAMU_AI_PY_MODULES` | `GCCcore/13.2.0 Python/3.11.5` | check with `module spider Python` |
| `TAMU_AI_TORCH_INDEX` | CUDA 12.4 wheels | must suit the GPU driver (printed at the top of each job log) |

Scratch is not backed up: keep anything that matters (reports, chosen vectors) on the laptop too.
