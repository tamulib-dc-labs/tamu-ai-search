# tamu-ai-search

AI Mode for TAMU Digital Collections (digitalcollections.library.tamu.edu): answers grounded in Archipelago
items, follow-up conversation, "ask about this item", and related items.

Design and roadmap: the shared guide "AI Mode for TAMU Digital Collections: a beginner's guide".

| Folder | What lives there |
| --- | --- |
| `service/` | Python service (FastAPI): ingest, chunking, embeddings, hybrid retrieval, reranking, evaluation |
| `drupal/` | Drupal module `tamu_ai_search` (milestone 2: settings, corpus export, change notifications) |

## Milestone 1: search core (no LLM yet)

1. Read published items from the Archipelago Solr index (`zs_raw_json`).
2. Turn each item into a readable *metadata card* chunk.
3. Embed chunks with `BAAI/bge-base-en-v1.5` (768 numbers each) and store them in a separate Solr core, `ai_chunks`.
4. Search both ways (keywords and vectors), merge with reciprocal rank fusion, rerank with `BAAI/bge-reranker-base`.
5. Score retrieval on test questions (`service/eval/questions.yaml`).

See `service/README.md` for setup and commands.
