"""HTTP API. Milestone 1 exposes retrieval only; the answer endpoint (/ai/v1/chat) comes with the LLM."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings
from .embed import Models
from .retrieve import MODES, Filters, Retriever

settings = Settings.from_env()
models = Models(settings)
retriever = Retriever(settings, models)
app = FastAPI(title="TAMU AI search", version="0.1.0")


class SearchFilters(BaseModel):
    object_types: list[str] = []
    collections: list[str] = []
    year_from: int | None = None
    year_to: int | None = None


class SearchRequest(BaseModel):
    q: str = Field(min_length=1, max_length=500)
    filters: SearchFilters = SearchFilters()
    mode: str = "hybrid+rerank"
    limit: int = Field(default=10, ge=1, le=50)


@app.get("/ai/v1/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ai/v1/search")
def search(request: SearchRequest) -> dict:
    if request.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    hits = retriever.search(request.q, Filters(**request.filters.model_dump()), mode=request.mode,
                            limit=request.limit)
    return {"results": [{
        "item_uuid": h.item_uuid, "nid": h.nid, "title": h.title, "url": f"{settings.site_url}/do/{h.item_uuid}",
        "kind": h.kind, "anchor": h.anchor, "score": round(h.score, 4), "object_types": h.object_types,
        "collections": h.collections, "snippet": h.text[:400],
    } for h in hits]}
