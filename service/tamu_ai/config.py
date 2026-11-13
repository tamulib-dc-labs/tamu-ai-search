# Settings for the whole service.
#
# Each value can be overridden with an environment variable, so the same code runs on the laptop,
# on Grace and (later) in the cluster without editing files. Example:
#   TAMU_AI_EMBED_MODEL=BAAI/bge-m3 TAMU_AI_EMBED_DIM=1024 tamu-ai search "old maps"

import os

# Where Solr is, and which cores (databases) we use.
SOLR_URL = os.environ.get("TAMU_AI_SOLR_URL", "http://localhost:8983/solr").rstrip("/")
SOURCE_CORE = os.environ.get("TAMU_AI_SOURCE_CORE", "drupal")    # Archipelago's own search index (we only read it)
CHUNK_CORE = os.environ.get("TAMU_AI_CHUNK_CORE", "ai_chunks")   # our core: chunk text + vectors

# Used to build links back to items: SITE_URL/do/<uuid>
SITE_URL = os.environ.get("TAMU_AI_SITE_URL", "https://digitalcollections.library.tamu.edu").rstrip("/")

# The embedding model turns text into a list of numbers (a "vector"); EMBED_DIM is how many numbers.
# It must match the model: bge-base-en-v1.5 gives 768, bge-m3 and F2LLM-0.6B give 1024.
EMBED_MODEL = os.environ.get("TAMU_AI_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_DIM = int(os.environ.get("TAMU_AI_EMBED_DIM", "768"))

# The reranker reads the question and a passage together and scores how well they match.
RERANK_MODEL = os.environ.get("TAMU_AI_RERANK_MODEL", "BAAI/bge-reranker-base")

# How much the keyword results count when merged with the vector results (vector results count 1.0).
KEYWORD_WEIGHT = float(os.environ.get("TAMU_AI_KEYWORD_WEIGHT", "1.0"))
