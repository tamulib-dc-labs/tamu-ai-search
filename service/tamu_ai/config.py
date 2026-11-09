"""Settings, read from environment variables so the same code runs locally and in the cluster."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    solr_url: str = "http://localhost:8983/solr"
    source_core: str = "drupal"  # Archipelago's Search API core (default_solr_index)
    chunk_core: str = "ai_chunks"  # our own core for chunks + vectors
    site_url: str = "https://digitalcollections.library.tamu.edu"
    embed_model: str = "BAAI/bge-base-en-v1.5"
    embed_dim: int = 768
    rerank_model: str = "BAAI/bge-reranker-base"
    keyword_weight: float = 1.0  # RRF weight of the keyword list relative to the vector list

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(
            solr_url=os.environ.get("TAMU_AI_SOLR_URL", defaults.solr_url).rstrip("/"),
            source_core=os.environ.get("TAMU_AI_SOURCE_CORE", defaults.source_core),
            chunk_core=os.environ.get("TAMU_AI_CHUNK_CORE", defaults.chunk_core),
            site_url=os.environ.get("TAMU_AI_SITE_URL", defaults.site_url).rstrip("/"),
            embed_model=os.environ.get("TAMU_AI_EMBED_MODEL", defaults.embed_model),
            embed_dim=int(os.environ.get("TAMU_AI_EMBED_DIM", defaults.embed_dim)),
            rerank_model=os.environ.get("TAMU_AI_RERANK_MODEL", defaults.rerank_model),
            keyword_weight=float(os.environ.get("TAMU_AI_KEYWORD_WEIGHT", defaults.keyword_weight)),
        )
