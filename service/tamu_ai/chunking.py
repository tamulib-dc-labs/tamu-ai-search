"""Split items into chunks: passages small enough to embed and specific enough to cite."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .cards import build_card, subjects, years
from .corpus import Item

# bge models read at most 512 tokens; ~300 words leaves room for the header and for longer words.
MAX_WORDS = 300
OVERLAP_WORDS = 50


@dataclass
class Chunk:
    id: str
    item_uuid: str
    nid: int
    title: str
    kind: str  # card | ocr | transcript
    anchor: str  # where in the item: "" for cards; "page/14", "time/1844" for content chunks
    text: str
    object_types: list[str] = field(default_factory=list)
    collections: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    rights: str = ""

    def to_record(self) -> dict:
        """One line of a chunks.jsonl file: everything except the vector."""
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict) -> "Chunk":
        return cls(**record)

    def to_solr(self, vector: list[float] | None) -> dict:
        doc = {
            "id": self.id, "item_uuid": self.item_uuid, "nid": self.nid, "title": self.title,
            "title_s": self.title, "chunk_kind": self.kind, "anchor": self.anchor, "text": self.text,
            "object_type": self.object_types, "collection": self.collections, "subject": self.subjects,
            "rights": self.rights,
        }
        if vector is not None:  # text-only chunks still work for keyword search
            doc["vector"] = vector
        if self.year_min is not None:
            doc["year_min"] = self.year_min
            doc["year_max"] = self.year_max
        return doc


def word_windows(text: str, max_words: int = MAX_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping windows of words, so a sentence cut at one edge appears whole in the next."""
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)] if words else []
    step = max_words - overlap
    windows = []
    for start in range(0, len(words), step):
        windows.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break
    return windows


def card_chunks(item: Item) -> list[Chunk]:
    """One chunk for a short card; a long card is split, and every part repeats the header for context."""
    card = build_card(item.title, item.metadata, item.collections)
    header = "\n".join(card.header)
    body = "\n".join(card.body)
    header_words = len(header.split())
    room = max(MAX_WORDS - header_words, 100)
    parts = word_windows(body, max_words=room, overlap=min(OVERLAP_WORDS, room // 3)) if body else [""]

    year_min, year_max = years(item.metadata)
    common = dict(
        item_uuid=item.uuid, nid=item.nid, title=item.title, kind="card", anchor="",
        object_types=item.object_types, collections=item.collections, subjects=subjects(item.metadata),
        year_min=year_min, year_max=year_max, rights=str(item.metadata.get("rights") or ""),
    )
    return [
        Chunk(id=f"{item.uuid}:card:{n}", text=f"{header}\n{part}".strip(), **common)
        for n, part in enumerate(parts)
    ]
