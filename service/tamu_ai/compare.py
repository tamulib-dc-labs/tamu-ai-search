"""Compare embedding models, merge weights and rerankers on the test questions, entirely in memory (for Grace).

Inputs are two files made next to Solr: chunks.jsonl (`tamu-ai export`) and evalpack.json (`tamu-ai export-eval`).
Vector search here is a plain matrix product over every chunk: exact, and fast enough for tens of thousands
of chunks on a GPU node. Solr's HNSW index returns nearly the same neighbours.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from .evaluate import Question, score
from .files import model_slug, read_chunks, read_vectors, save_vectors
from .retrieve import rrf

log = logging.getLogger(__name__)


def _load_pack(path: Path) -> tuple[list[Question], dict[str, list[str]]]:
    pack = json.loads(path.read_text(encoding="utf-8"))
    questions = [Question(id=q["id"], question=q["question"], relevant=q["relevant"], style=q.get("style", ""))
                 for q in pack["questions"]]
    return questions, {q["id"]: q["keyword"] for q in pack["questions"]}


def _passage_vectors(model_name: str, chunk_ids: list[str], texts: list[str], cache_dir: Path, embedder):
    """Embed every chunk once per model; reuse the saved file while the chunk list is unchanged."""
    cache = cache_dir / f"vectors-{model_slug(model_name)}.npz"
    if cache.exists():
        ids, vectors, cached_model = read_vectors(cache)
        if ids == chunk_ids and cached_model == model_name:
            log.info("using cached vectors %s", cache)
            return vectors
    started = time.monotonic()
    vectors = embedder.passages(texts, batch_size=64)
    log.info("%s: embedded %d chunks in %.0fs", model_name, len(texts), time.monotonic() - started)
    save_vectors(cache, chunk_ids, vectors, model_name)
    return vectors


def _by_style(rows: dict, questions: list[Question]) -> dict:
    out = {}
    for style in sorted({q.style for q in questions if q.style}):
        ids = [q.id for q in questions if q.style == style]
        out[style] = {"questions": len(ids), "ndcg": sum(rows[i]["ndcg"] for i in ids) / len(ids)}
    return out


def run_compare(chunks_path: Path, pack_path: Path, embed_models: list[str], rerankers: list[str],
                keyword_weights: list[float], rerank_depths: list[int], cache_dir: Path,
                depth: int = 50, k: int = 10) -> dict:
    from .embed import Embedder, Reranker

    chunks = list(read_chunks(chunks_path))
    chunk_ids = [c.id for c in chunks]
    item_of = {c.id: c.item_uuid for c in chunks}
    text_of = {c.id: c.text for c in chunks}
    questions, keyword_ranks = _load_pack(pack_path)
    # The pack and the chunk file should come from the same export; drop ids the chunk file does not have.
    dropped = sum(1 for ids in keyword_ranks.values() for c in ids if c not in item_of)
    if dropped:
        log.warning("%d keyword results are not in %s; was the evalpack made from another export?",
                    dropped, chunks_path)
    keyword_ranks = {qid: [c for c in ids if c in item_of] for qid, ids in keyword_ranks.items()}
    report: dict = {"k": k, "chunks": len(chunks), "questions": len(questions), "modes": {}, "per_question": {},
                    "by_style": {}}

    def record(name: str, rank_chunks) -> None:
        averages, rows = score(questions, lambda q: [item_of[c] for c in rank_chunks(q) if c in item_of], k)
        report["modes"][name] = averages
        report["by_style"][name] = _by_style(rows, questions)
        for qid, row in rows.items():
            report["per_question"].setdefault(qid, {})[name] = row
        log.info("%-60s nDCG@10 %.3f", name, averages["ndcg"])

    record("keyword (solr)", lambda q: keyword_ranks[q.id])
    loaded_rerankers = {name: Reranker(name) for name in rerankers}

    for model_name in embed_models:
        embedder = Embedder(model_name)
        vectors = _passage_vectors(model_name, chunk_ids, [c.text for c in chunks], cache_dir, embedder)
        query_vectors = embedder.queries([q.question for q in questions])
        similarity = query_vectors @ vectors.T  # cosine, since every vector is normalized
        dense = {q.id: [chunk_ids[i] for i in np.argsort(-similarity[n])[:depth]] for n, q in enumerate(questions)}
        short = model_name.split("/")[-1]

        record(f"{short} vector", lambda q: dense[q.id])
        for weight in keyword_weights:
            fused = {q.id: [cid for cid, _ in rrf([keyword_ranks[q.id], dense[q.id]], weights=[weight, 1.0])]
                     for q in questions}
            record(f"{short} hybrid kw={weight:g}", lambda q, fused=fused: fused[q.id])
            for reranker_name, reranker in loaded_rerankers.items():
                for rerank_depth in rerank_depths:
                    def reranked(q, fused=fused, reranker=reranker, rerank_depth=rerank_depth):
                        pool = fused[q.id][:rerank_depth]
                        scores = reranker.scores(q.question, [text_of[c] for c in pool])
                        return [c for c, _ in sorted(zip(pool, scores), key=lambda pair: pair[1], reverse=True)]
                    record(f"{short} hybrid kw={weight:g} + {reranker_name.split('/')[-1]}@{rerank_depth}",
                           reranked)
        del embedder
        _free_gpu()
    return report


def _free_gpu() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
