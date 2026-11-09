"""Fields of the ai_chunks core. Applied through Solr's Schema API, so re-running it is safe."""

from __future__ import annotations

from .config import Settings
from .solr import Solr

VECTOR_TYPE = "knn_vector_{dim}"

FIELDS = [
    # name, type, multiValued
    ("item_uuid", "string", False),
    ("nid", "pint", False),
    ("title", "text_en", False),
    ("title_s", "string", False),
    ("chunk_kind", "string", False),  # card | ocr | transcript
    ("anchor", "string", False),  # "" for cards, "page/14" or "time/1844" later
    ("text", "text_en", False),
    ("object_type", "string", True),
    ("collection", "string", True),
    ("subject", "string", True),
    ("year_min", "pint", False),
    ("year_max", "pint", False),
    ("rights", "string", False),
]


def apply(settings: Settings) -> list[str]:
    """Create the vector type and fields that are missing; return what was added."""
    solr = Solr(settings.solr_url, settings.chunk_core)
    # The _default configset guesses field types for unknown fields; we want a fixed schema.
    solr.set_user_property("update.autoCreateFields", "false")

    schema = solr.schema_fields()
    have_types = {t["name"] for t in schema["fieldTypes"]}
    have_fields = {f["name"] for f in schema["fields"]}
    added: list[str] = []

    vector_type = VECTOR_TYPE.format(dim=settings.embed_dim)
    if vector_type not in have_types:
        solr.schema({"add-field-type": {
            "name": vector_type, "class": "solr.DenseVectorField",
            "vectorDimension": settings.embed_dim, "similarityFunction": "cosine",
        }})
        added.append(vector_type)

    commands = [
        {"name": name, "type": ftype, "multiValued": multi, "stored": True, "indexed": True}
        for name, ftype, multi in FIELDS if name not in have_fields
    ]
    if "vector" not in have_fields:
        commands.append({"name": "vector", "type": vector_type, "stored": False, "indexed": True})
    if commands:
        solr.schema({"add-field": commands})
        added.extend(c["name"] for c in commands)
    return added
