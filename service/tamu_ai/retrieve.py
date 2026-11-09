"""Hybrid retrieval: keyword (BM25) + vector (kNN) search, merged with reciprocal rank fusion, then reranked."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .embed import Models
from .solr import Solr

MODES = ("keyword", "vector", "hybrid", "hybrid+rerank")
RETURN_FIELDS = "id,item_uuid,nid,title_s,chunk_kind,anchor,text,object_type,collection"
RRF_K = 60  # the constant from Cormack et al. (2009); it damps the gap between rank 1 and rank 2


@dataclass
class Filters:
    object_types: list[str] = field(default_factory=list)
    collections: list[str] = field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None

    def to_fq(self) -> list[str]:
        fq = []
        if self.object_types:
            fq.append("object_type:(" + " OR ".join(_quote(t) for t in self.object_types) + ")")
        if self.collections:
            fq.append("collection:(" + " OR ".join(_quote(c) for c in self.collections) + ")")
        if self.year_from is not None or self.year_to is not None:
            # An item matches when its date span overlaps the requested range.
            low = self.year_from if self.year_from is not None else "*"
            high = self.year_to if self.year_to is not None else "*"
            fq.append(f"year_max:[{low} TO *] AND year_min:[* TO {high}]")
        return fq


@dataclass
class Hit:
    id: str
    item_uuid: str
    nid: int
    title: str
    kind: str
    anchor: str
    text: str
    score: float
    object_types: list[str] = field(default_factory=list)
    collections: list[str] = field(default_factory=list)

    @classmethod
    def from_doc(cls, doc: dict[str, Any], score: float) -> "Hit":
        return cls(id=doc["id"], item_uuid=doc["item_uuid"], nid=doc.get("nid", 0), title=doc.get("title_s", ""),
                   kind=doc.get("chunk_kind", ""), anchor=doc.get("anchor", ""), text=doc.get("text", ""),
                   score=score, object_types=doc.get("object_type", []), collections=doc.get("collection", []))


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def rrf(ranked_lists: list[list[str]], k: int = RRF_K,
        weights: list[float] | None = None) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: each list gives an id weight/(k + rank); ids high in any list rise to the top.

    Equal weights treat both lists the same; a lower weight on the keyword list lets meaning search lead.
    """
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def diversify(hits: list[Hit], per_item: int) -> list[Hit]:
    """Keep at most `per_item` chunks of the same item, so one long book cannot crowd out everything else."""
    counts: dict[str, int] = {}
    kept = []
    for hit in hits:
        if counts.get(hit.item_uuid, 0) < per_item:
            kept.append(hit)
            counts[hit.item_uuid] = counts.get(hit.item_uuid, 0) + 1
    return kept


class Retriever:
    def __init__(self, settings: Settings, models: Models):
        self.solr = Solr(settings.solr_url, settings.chunk_core)
        self.models = models
        self.keyword_weight = settings.keyword_weight

    def keyword(self, query: str, filters: Filters, k: int) -> list[Hit]:
        data = self.solr.select({"q": query, "defType": "edismax", "qf": "title^2 text", "fq": filters.to_fq(),
                                 "fl": RETURN_FIELDS + ",score", "rows": k})
        return [Hit.from_doc(d, d["score"]) for d in data["response"]["docs"]]

    def vector(self, query: str, filters: Filters, k: int) -> list[Hit]:
        vec = self.models.embed_query(query)
        knn = "{!knn f=vector topK=%d}[%s]" % (k, ",".join(f"{x:.6f}" for x in vec))
        data = self.solr.select({"q": knn, "fq": filters.to_fq(), "fl": RETURN_FIELDS + ",score", "rows": k})
        return [Hit.from_doc(d, d["score"]) for d in data["response"]["docs"]]

    def search(self, query: str, filters: Filters | None = None, mode: str = "hybrid+rerank",
               limit: int = 10, candidates: int = 50, per_item: int = 2) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        filters = filters or Filters()
        if mode == "keyword":
            return diversify(self.keyword(query, filters, candidates), per_item)[:limit]
        if mode == "vector":
            return diversify(self.vector(query, filters, candidates), per_item)[:limit]

        keyword_hits = self.keyword(query, filters, candidates)
        vector_hits = self.vector(query, filters, candidates)
        by_id = {h.id: h for h in keyword_hits + vector_hits}
        fused = [Hit(**{**vars(by_id[doc_id]), "score": score})
                 for doc_id, score in rrf([[h.id for h in keyword_hits], [h.id for h in vector_hits]],
                                          weights=[self.keyword_weight, 1.0])]
        if mode == "hybrid":
            return diversify(fused, per_item)[:limit]

        pool = fused[:candidates]
        scores = self.models.rerank_scores(query, [h.text for h in pool])
        reranked = [Hit(**{**vars(h), "score": s}) for h, s in zip(pool, scores)]
        reranked.sort(key=lambda h: h.score, reverse=True)
        return diversify(reranked, per_item)[:limit]
