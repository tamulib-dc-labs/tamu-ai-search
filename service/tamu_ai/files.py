"""The file path between our Solr and Grace: export chunks -> embed them on a GPU -> load vectors back.

Grace cannot reach any of our Solr servers, so the heavy step (embedding) works only on files:

    laptop/cluster:  tamu-ai export      -> chunks.jsonl        (text + filter fields, no vectors)
    Grace:           tamu-ai embed       -> vectors-<model>.npz (one row per chunk id)
    laptop/cluster:  tamu-ai load        -> ai_chunks core
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .chunking import Chunk, card_chunks
from .config import Settings
from .corpus import iter_items
from .evaluate import Question
from .solr import Solr

log = logging.getLogger(__name__)


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_chunks(path: Path) -> Iterator[Chunk]:
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                yield Chunk.from_record(json.loads(line))


def export_chunks(settings: Settings, path: Path, limit: int | None = None) -> int:
    """Every public item's chunks, one JSON object per line; an item's chunks are always consecutive."""
    return write_jsonl(path, (c.to_record() for item in iter_items(settings, limit=limit)
                              for c in card_chunks(item)))


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", model_name).strip("-").lower()


def save_vectors(path: Path, ids: list[str], vectors: np.ndarray, model_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids), vectors=vectors.astype("float32"), model=np.array(model_name),
             created=np.array(datetime.now(timezone.utc).isoformat()))


def read_vectors(path: Path) -> tuple[list[str], np.ndarray, str]:
    data = np.load(path)
    return [str(i) for i in data["ids"]], data["vectors"], str(data["model"])


def embed_file(chunks_path: Path, out_path: Path, model_name: str, batch_size: int = 64) -> tuple[int, int]:
    from .embed import Embedder
    chunks = list(read_chunks(chunks_path))
    embedder = Embedder(model_name)
    vectors = embedder.passages([c.text for c in chunks], batch_size=batch_size)
    save_vectors(out_path, [c.id for c in chunks], vectors, model_name)
    return len(chunks), vectors.shape[1]


def _item_batches(chunks: Iterable[Chunk], max_chunks: int) -> Iterator[list[Chunk]]:
    """Batches that never split an item, because each batch first deletes its items' old chunks."""
    batch: list[Chunk] = []
    for _, group in groupby(chunks, key=lambda c: c.item_uuid):
        item_chunks = list(group)
        if batch and len(batch) + len(item_chunks) > max_chunks:
            yield batch
            batch = []
        batch.extend(item_chunks)
    if batch:
        yield batch


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def load_into_solr(settings: Settings, chunks_path: Path, vectors_path: Path | None = None,
                   batch_chunks: int = 500) -> int:
    """Write chunks (and their vectors, if given) into ai_chunks, replacing each item's previous chunks."""
    vector_of: dict[str, list[float]] = {}
    if vectors_path:
        ids, vectors, model_name = read_vectors(vectors_path)
        if model_name != settings.embed_model:
            raise ValueError(f"vectors were made with {model_name} but TAMU_AI_EMBED_MODEL is "
                             f"{settings.embed_model}; queries would not match them")
        if vectors.shape[1] != settings.embed_dim:
            raise ValueError(f"vectors have {vectors.shape[1]} numbers but the core expects {settings.embed_dim}")
        vector_of = {chunk_id: row.tolist() for chunk_id, row in zip(ids, vectors)}

    solr = Solr(settings.solr_url, settings.chunk_core)
    written = 0
    for batch in _item_batches(read_chunks(chunks_path), batch_chunks):
        uuids = sorted({c.item_uuid for c in batch})
        solr.delete_by_query("item_uuid:(" + " OR ".join(_quote(u) for u in uuids) + ")")
        solr.add([c.to_solr(vector_of.get(c.id) if vector_of else None) for c in batch])
        written += len(batch)
    solr.commit()
    missing = written - sum(1 for c in read_chunks(chunks_path) if c.id in vector_of) if vector_of else 0
    if missing:
        log.warning("%d chunks had no vector and were loaded as text only", missing)
    return written


def export_evalpack(settings: Settings, questions: list[Question], path: Path, depth: int = 50) -> int:
    """Questions plus Solr's keyword ranking for each, so Grace can score hybrid search without Solr.

    The keyword list depends only on the chunk text, not on the embedding model, so it is computed once here.
    """
    from .retrieve import Filters, Retriever
    retriever = Retriever(settings, models=None)
    pack = {
        "created": datetime.now(timezone.utc).isoformat(),
        "source": f"{settings.solr_url}/{settings.chunk_core}",
        "depth": depth,
        "questions": [{
            "id": q.id, "question": q.question, "relevant": q.relevant, "style": q.style,
            "keyword": [h.id for h in retriever.keyword(q.question, Filters(), depth)],
        } for q in questions],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pack, indent=1, ensure_ascii=False), encoding="utf-8")
    return len(questions)
