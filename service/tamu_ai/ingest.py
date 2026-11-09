"""Corpus -> chunks -> vectors -> the ai_chunks core."""

from __future__ import annotations

import logging
import time

from .chunking import Chunk, card_chunks
from .config import Settings
from .corpus import iter_items
from .embed import Models
from .solr import Solr

log = logging.getLogger(__name__)


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_batch(solr: Solr, models: Models, chunks: list[Chunk]) -> None:
    # Replace, never append: drop every chunk the items had before, so edited or shortened items leave nothing stale.
    uuids = sorted({c.item_uuid for c in chunks})
    solr.delete_by_query("item_uuid:(" + " OR ".join(_quote(u) for u in uuids) + ")")
    vectors = models.embed_passages([c.text for c in chunks])
    solr.add([c.to_solr(v) for c, v in zip(chunks, vectors)])


def ingest(settings: Settings, models: Models, limit: int | None = None, fresh: bool = False,
           batch_items: int = 64) -> dict[str, int]:
    solr = Solr(settings.solr_url, settings.chunk_core)
    if fresh:
        solr.delete_by_query("*:*")
    started = time.monotonic()
    items = chunks_total = 0
    pending: list[Chunk] = []
    pending_items = 0
    for item in iter_items(settings, limit=limit):
        pending.extend(card_chunks(item))
        pending_items += 1
        if pending_items >= batch_items:
            write_batch(solr, models, pending)
            items += pending_items
            chunks_total += len(pending)
            log.info("%d items, %d chunks (%.0fs)", items, chunks_total, time.monotonic() - started)
            pending, pending_items = [], 0
    if pending:
        write_batch(solr, models, pending)
        items += pending_items
        chunks_total += len(pending)
    solr.commit()
    log.info("done: %d items, %d chunks in %.0fs", items, chunks_total, time.monotonic() - started)
    return {"items": items, "chunks": chunks_total}
