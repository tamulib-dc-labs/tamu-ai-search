# The web API, built with FastAPI. Run it with:
#   uvicorn tamu_ai.api:app --port 8100
# then open http://localhost:8100/docs to try it in the browser.
#
# For now it only does search. The AI answer endpoint (/ai/v1/chat) comes later, with the LLM.

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import config
from .search import filter_queries, search

app = FastAPI(title="TAMU AI search")


# What a search request looks like. FastAPI checks incoming JSON against this
# and returns a clear error if, say, `limit` is 500 or `q` is empty.
class SearchRequest(BaseModel):
    q: str = Field(min_length=1, max_length=500)
    mode: Literal["keyword", "vector", "hybrid", "hybrid+rerank"] = "hybrid+rerank"
    limit: int = Field(default=10, ge=1, le=50)
    object_types: list[str] = []
    collections: list[str] = []
    year_from: int | None = None
    year_to: int | None = None


@app.get("/ai/v1/health")
def health():
    # For monitoring: "is the service up?"
    return {"status": "ok"}


@app.post("/ai/v1/search")
def search_items(request: SearchRequest):
    fq = filter_queries(request.object_types, request.collections, request.year_from, request.year_to)
    docs = search(request.q, request.mode, fq, request.limit)
    return {"results": [{
        "item_uuid": doc["item_uuid"],
        "nid": doc["nid"],
        "title": doc["title"],
        "url": f"{config.SITE_URL}/do/{doc['item_uuid']}",
        "object_types": doc.get("object_type", []),
        "collections": doc.get("collection", []),
        "snippet": doc["text"][:400],
    } for doc in docs]}
