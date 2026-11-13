# tamu-ai-search

AI Mode for TAMU Digital Collections (digitalcollections.library.tamu.edu): search by meaning, AI answers that
cite real items, follow-up questions, "ask about this item", and related items.

Right now (milestone 1) it is the **search part only**, with no LLM yet:

1. Read published items from Archipelago's Solr index.
2. Turn each item's metadata into a readable "card" (`tamu_ai/cards.py`).
3. Turn each card into a vector with an embedding model, and store cards and vectors in a separate Solr core,
   `ai_chunks` (`pipeline.py`).
4. Search by keywords and by vectors, merge the two lists, and re-order the top results with a reranker
   (`search.py`).
5. Score search on test questions (`evaluate.py`, `eval/questions.yaml`).

## Layout

```
service/                 the Python package (tamu-ai)
  tamu_ai/
    config.py            settings (Solr address, model names), overridable with TAMU_AI_* variables
    solr.py              small helpers for Solr's HTTP API
    cards.py             item metadata -> readable card text, split into chunks
    models.py            embedding model and reranker
    pipeline.py          export -> embed -> load, through files
    search.py            keyword, vector, hybrid and reranked search
    evaluate.py          recall@10, nDCG@10, MRR
    compare.py           compare models without Solr (for Grace)
    api.py               web API (FastAPI)
    cli.py               the `tamu-ai` command
  eval/questions.yaml    test questions with the items each should find
  tests/
grace/                   scripts for the HPRC Grace cluster
drupal/                  (later) Drupal module: settings, export endpoint, change notifications
```

## Laptop setup (WSL Ubuntu, next to the local Archipelago testbed)

```bash
python3 -m venv ~/.venvs/tamu-ai
~/.venvs/tamu-ai/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/.venvs/tamu-ai/bin/pip install -e "/mnt/c/lib_work/tamu-ai-search/service[dev]"
source ~/.venvs/tamu-ai/bin/activate
```

Create the core once, then add our fields:

```bash
docker exec esmero-solr solr create -c ai_chunks -d _default
tamu-ai setup
```

## Commands

```bash
tamu-ai card 26570                 # show the card text for one item (by node id)
tamu-ai export data                # data/chunks.jsonl + data/questions.json
tamu-ai embed data                 # data/vectors-<model>.npy  (slow on a laptop; use Grace for everything)
tamu-ai load data                  # replace ai_chunks with these chunks and vectors
tamu-ai search "photos of cattle"  # one search (default mode: hybrid+rerank)
tamu-ai search "photos of cattle" --mode keyword
tamu-ai eval                       # score all four modes on the test questions
pytest -q                          # unit tests (from service/)
uvicorn tamu_ai.api:app --port 8100   # web API; try it at http://localhost:8100/docs
```

`export` saves Solr's keyword results for each test question, and those come from `ai_chunks`, so load the chunks
first (`load` without a vectors file loads the text only).

Models download from Hugging Face on first use. To try another embedding model, set `TAMU_AI_EMBED_MODEL` and
`TAMU_AI_EMBED_DIM` (bge-m3 and F2LLM-0.6B use 1024); a new size needs a fresh `ai_chunks` core.

## Grace (HPRC)

The laptop is for writing code; embedding everything and comparing models runs on Grace's A100 GPUs. Grace can't
reach our Solr, so data goes over as files.

```bash
# laptop: export the data and copy it over
tamu-ai export data
scp data/chunks.jsonl data/questions.json <NetID>@grace.hprc.tamu.edu:/scratch/user/<NetID>/tamu-ai-data/

# Grace login node, once: get the code and install
cd $SCRATCH && git clone https://github.com/tamulib-dc-labs/tamu-ai-search.git
cd tamu-ai-search/grace && bash setup.sh

# Grace: run the comparison on a GPU node
sbatch compare.slurm        # prints the job id
squeue -u $USER             # PD = waiting for a GPU, R = running
tail -f compare.<id>.log    # the results table is at the end

# Grace: embed everything with the chosen model, then copy the vectors back and `tamu-ai load data`
sbatch --export=ALL,MODEL=BAAI/bge-base-en-v1.5 embed.slurm
```

Compute nodes reach Hugging Face through the `WebProxy` module, which the job scripts load. Scratch is not
backed up, so copy anything worth keeping back to the laptop.

## Not done yet

- Only metadata cards; OCR text and A/V transcripts come later.
- `load` replaces the whole core; updating single items when librarians edit them comes with the Drupal module.
- The test questions are a draft and need a librarian to approve them.
