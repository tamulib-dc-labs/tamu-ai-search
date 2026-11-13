# The two kinds of AI model we use for search, both from the sentence-transformers library.
#
# 1. Embedding model: turns a piece of text into a vector (a list of numbers). Texts with similar
#    meaning get similar vectors, even if they share no words ("parking" vs "where to leave my car").
# 2. Reranker: reads the question and one passage together and gives a match score. Slower but
#    more accurate, so we only use it to re-order the top results.
#
# Models download from Hugging Face the first time they are used and are cached after that.

from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

# Use the GPU if there is one (Grace), otherwise the CPU (laptop).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The English bge models were trained with this phrase in front of every search query,
# so we have to add it too. Newer models (bge-m3, F2LLM) need nothing, or bring their own prompt.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# lru_cache means each model is loaded once and then reused (loading takes seconds).
@lru_cache
def load_embedder(name):
    return SentenceTransformer(name, device=DEVICE)


@lru_cache
def load_reranker(name):
    return CrossEncoder(name, device=DEVICE, max_length=512)


def embed(texts, model, query=False, batch_size=32):
    # Turn a list of texts into vectors: a numpy array with one row per text.
    # query=True for search questions, False for the passages being searched.
    # Vectors are normalized (length 1), so the dot product of two vectors is their cosine similarity.
    embedder = load_embedder(model)
    options = dict(batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    if query and embedder.prompts.get("query"):
        # The model ships its own query instruction (e.g. F2LLM); let the library add it.
        # (Some models, bge-base included, list an empty "query" prompt; those fall through to below.)
        return embedder.encode(texts, prompt_name="query", **options)
    if query and "bge-" in model and "-en" in model:
        texts = [BGE_QUERY_PREFIX + text for text in texts]
    return embedder.encode(texts, **options)


def rerank(query, texts, model):
    # A score for each text: how well it answers the query. Higher is better.
    if not texts:
        return []
    return load_reranker(model).predict([(query, text) for text in texts], show_progress_bar=False).tolist()
