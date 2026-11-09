"""Embedding and reranking models, loaded once and shared. Runs on a GPU when one is present (Grace), else CPU."""

from __future__ import annotations

from functools import cached_property

from .config import Settings

# Query/passage prefixes for models whose files do not declare them. Models that ship prompts in their
# sentence-transformers config (F2LLM, bge-m3, ...) use those instead.
PREFIXES = {
    "bge-small-en": ("Represent this sentence for searching relevant passages: ", ""),
    "bge-base-en": ("Represent this sentence for searching relevant passages: ", ""),
    "bge-large-en": ("Represent this sentence for searching relevant passages: ", ""),
    "e5-": ("query: ", "passage: "),
}


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        kwargs = {"torch_dtype": "bfloat16"} if device() == "cuda" else {}
        self.model = SentenceTransformer(model_name, device=device(), model_kwargs=kwargs)
        self.uses_model_prompts = "query" in (self.model.prompts or {})
        self.query_prefix, self.passage_prefix = next(
            (p for key, p in PREFIXES.items() if key in model_name.lower()), ("", ""))

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str], role: str, batch_size: int):
        options = dict(batch_size=batch_size, normalize_embeddings=True, convert_to_tensor=True,
                       show_progress_bar=False)
        if self.uses_model_prompts:
            encode = self.model.encode_query if role == "query" else self.model.encode_document
            tensor = encode(texts, **options)
        else:
            prefix = self.query_prefix if role == "query" else self.passage_prefix
            tensor = self.model.encode([prefix + t for t in texts], **options)
        # bfloat16 tensors cannot go straight to numpy; float32 is what Solr and the scoring code expect.
        return tensor.float().cpu().numpy()

    def passages(self, texts: list[str], batch_size: int = 32):
        """Normalized float32 numpy array, one row per text."""
        return self._encode(texts, "passage", batch_size)

    def queries(self, texts: list[str], batch_size: int = 32):
        return self._encode(texts, "query", batch_size)


class Reranker:
    def __init__(self, model_name: str, max_length: int = 512):
        from sentence_transformers import CrossEncoder
        self.name = model_name
        self.model = CrossEncoder(model_name, device=device(), max_length=max_length)

    def scores(self, query: str, passages: list[str], batch_size: int = 32) -> list[float]:
        if not passages:
            return []
        return self.model.predict([(query, p) for p in passages], batch_size=batch_size,
                                  show_progress_bar=False).tolist()


class Models:
    """The models named in Settings, loaded on first use (the API and CLI share one instance)."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def embedder(self) -> Embedder:
        return Embedder(self.settings.embed_model)

    @cached_property
    def reranker(self) -> Reranker:
        return Reranker(self.settings.rerank_model)

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return self.embedder.passages(texts, batch_size).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.queries([text])[0].tolist()

    def rerank_scores(self, query: str, passages: list[str]) -> list[float]:
        return self.reranker.scores(query, passages)
