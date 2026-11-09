# tamu-ai service

Python service behind AI Mode. Milestone 1 covers retrieval only: no LLM yet.

## How it fits together

```
Archipelago Solr (core "drupal")  --corpus.py-->  Item  --cards.py / chunking.py-->  Chunk (metadata card)
        --embed.py (bge-base, 768 numbers)-->  Solr core "ai_chunks"  (text + vector + filter fields)

query --retrieve.py--> keyword search (edismax)  \
                      vector search ({!knn})      >-- RRF merge --> reranker (bge-reranker-base) --> top 10
```

| Module | Job |
| --- | --- |
| `config.py` | Settings from `TAMU_AI_*` environment variables |
| `solr.py` | Minimal Solr HTTP client |
| `schema.py` | Adds the vector type and fields to `ai_chunks` (safe to re-run) |
| `corpus.py` | Reads public items (published, not Page/Leaf) from the Archipelago index |
| `cards.py` | Turns metadata JSON into a readable card; years from EDTF dates |
| `chunking.py` | Splits cards (and later OCR/transcripts) into chunks |
| `embed.py` | Embedding and reranking models |
| `ingest.py` | Chunks, embeds and writes items; replaces an item's old chunks |
| `retrieve.py` | Keyword, vector, hybrid (RRF) and hybrid+rerank search, filters |
| `evaluate.py` | recall@10, nDCG@10, MRR over `eval/questions.yaml` |
| `api.py` | FastAPI: `GET /ai/v1/health`, `POST /ai/v1/search` |
| `cli.py` | `tamu-ai setup | ingest | search | eval | card` |

## Local setup (WSL Ubuntu, next to the archipelago-1.6.0-local testbed)

```bash
python3 -m venv ~/.venvs/tamu-ai
~/.venvs/tamu-ai/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
cd /mnt/c/lib_work/tamu-ai-search/service
~/.venvs/tamu-ai/bin/pip install -e ".[dev]"
```

Create the chunk core once (local Solr runs in the `esmero-solr` container):

```bash
docker exec esmero-solr solr create -c ai_chunks -d _default
~/.venvs/tamu-ai/bin/tamu-ai setup
```

## Commands

```bash
~/.venvs/tamu-ai/bin/tamu-ai card 26570                  # show the chunks built for one item (by nid)
~/.venvs/tamu-ai/bin/tamu-ai ingest --fresh              # (re)index every public item
~/.venvs/tamu-ai/bin/tamu-ai search "photos of cattle"   # one query, hybrid+rerank
~/.venvs/tamu-ai/bin/tamu-ai search "photos of cattle" --mode keyword
~/.venvs/tamu-ai/bin/tamu-ai eval                        # score all modes on the test questions
~/.venvs/tamu-ai/bin/pytest -q                           # unit tests
~/.venvs/tamu-ai/bin/uvicorn tamu_ai.api:app --port 8100 # HTTP API
```

The first run downloads the two models from Hugging Face (about 1.5 GB) into `~/.cache/huggingface`.

## Settings

| Variable | Default |
| --- | --- |
| `TAMU_AI_SOLR_URL` | `http://localhost:8983/solr` |
| `TAMU_AI_SOURCE_CORE` | `drupal` |
| `TAMU_AI_CHUNK_CORE` | `ai_chunks` |
| `TAMU_AI_SITE_URL` | `https://digitalcollections.library.tamu.edu` |
| `TAMU_AI_EMBED_MODEL` / `TAMU_AI_EMBED_DIM` | `BAAI/bge-base-en-v1.5` / `768` |
| `TAMU_AI_RERANK_MODEL` | `BAAI/bge-reranker-base` |

## Where things run

The laptop is for writing code and small smoke tests only. Embedding the full corpus, model comparisons and all
training run on HPRC Grace. Grace cannot reach our Solr, so that path goes through files:

```bash
tamu-ai export --out chunks.jsonl          # next to Solr
tamu-ai export-eval --out evalpack.json    # next to Solr: questions + Solr keyword rankings
tamu-ai embed --chunks chunks.jsonl --model BAAI/bge-base-en-v1.5 --out vectors.npz   # Grace GPU
tamu-ai compare --chunks chunks.jsonl --evalpack evalpack.json --models ... --rerankers ... --out report.json  # Grace
tamu-ai load --chunks chunks.jsonl --vectors vectors.npz   # next to Solr
```

`grace/README.md` has the step-by-step runbook and the Slurm job scripts.

## Known limits of milestone 1

- Metadata cards only: the local testbed has no OCR or transcript flavors, so content chunks are not built yet
  (`chunking.word_windows` is ready for them).
- Reads from Solr's stored `zs_raw_json`; the Drupal export endpoint and change notifications come in milestone 2.
- Test questions are a draft from item metadata and need librarian approval.
