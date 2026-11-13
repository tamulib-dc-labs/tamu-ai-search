# Compare embedding models, keyword weights and rerankers on the test questions (runs on Grace).
#
# Uses only the files from `tamu-ai export`, no Solr. Vector search here is done by hand: multiply the
# question vectors by all chunk vectors and take the highest scores. That's exact; Solr's vector index
# gives almost the same results, just faster on big collections.
#
# Every "variant" (e.g. "bge-m3 hybrid kw=0.5 + bge-reranker-base") produces a ranking for each question,
# and all variants are scored the same way, so the printed table compares them fairly.

import json
from pathlib import Path

import numpy as np

from . import models
from .evaluate import score
from .pipeline import read_chunks, vectors_path
from .search import rrf


def compare(folder, embed_models, rerankers=(), keyword_weights=(1.0, 0.5), depth=50):
    chunks = read_chunks(folder)
    chunk_ids = [c["id"] for c in chunks]
    item_of = {c["id"]: c["item_uuid"] for c in chunks}
    text_of = {c["id"]: c["text"] for c in chunks}
    questions = json.loads((Path(folder) / "questions.json").read_text(encoding="utf-8"))

    # Solr's keyword results, saved by export. Drop any chunk ids that aren't in chunks.jsonl
    # (happens if the export used --limit but Solr had more chunks loaded).
    keyword = {q["id"]: [cid for cid in q["keyword"] if cid in item_of] for q in questions}

    # rankings[variant name][question id] = chunk ids, best first
    rankings = {"keyword": keyword}

    for model in embed_models:
        name = model.split("/")[-1]
        print(f"embedding with {model} ...", flush=True)

        # Reuse vectors from an earlier run (or from `tamu-ai embed`) if they match the chunk file.
        path = vectors_path(folder, model)
        vectors = np.load(path) if path.exists() else None
        if vectors is None or len(vectors) != len(chunks):
            vectors = models.embed([c["text"] for c in chunks], model, batch_size=64)
            np.save(path, vectors.astype("float32"))

        # Cosine similarity of every question with every chunk (vectors are normalized, so a dot product).
        question_vectors = models.embed([q["question"] for q in questions], model, query=True)
        similarity = question_vectors @ vectors.T
        vector = {}
        for row, q in enumerate(questions):
            best = np.argsort(-similarity[row])[:depth]  # indexes of the highest scores
            vector[q["id"]] = [chunk_ids[i] for i in best]
        rankings[f"{name} vector"] = vector

        for weight in keyword_weights:
            hybrid = {q["id"]: rrf([keyword[q["id"]], vector[q["id"]]], [weight, 1.0])[:depth] for q in questions}
            rankings[f"{name} hybrid kw={weight:g}"] = hybrid

            for reranker in rerankers:
                reranked = {}
                for q in questions:
                    pool = hybrid[q["id"]]
                    scores = models.rerank(q["question"], [text_of[cid] for cid in pool], reranker)
                    order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
                    reranked[q["id"]] = [pool[i] for i in order]
                rankings[f"{name} hybrid kw={weight:g} + {reranker.split('/')[-1]}"] = reranked

    # Score every variant. Scoring works on items, so turn chunk ids into item uuids first.
    results = {}
    for variant, ranking in rankings.items():
        items = {qid: [item_of[cid] for cid in chunk_list] for qid, chunk_list in ranking.items()}
        results[variant] = score(questions, items)
    return results
