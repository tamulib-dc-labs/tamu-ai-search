# Getting chunks into the ai_chunks core, in steps that pass files to each other.
#
# Grace (where the GPUs are) cannot reach our Solr, so the slow step, embedding, works on files:
#
#   1. export  (next to Solr)  -> data/chunks.jsonl      one chunk per line, as JSON
#                              -> data/questions.json    test questions + Solr's keyword results for each
#   2. embed   (on Grace)      -> data/vectors-<model>.npy   one vector per line of chunks.jsonl, same order
#   3. load    (next to Solr)  -> chunks + vectors written into the ai_chunks core
#
# On the laptop you can run all three in a row for a small test.

import json
import re
from pathlib import Path

import numpy as np

from . import cards, config, models, search, solr
from .evaluate import load_questions

# Fields of the ai_chunks core: (name, Solr type, can hold several values?)
FIELDS = [
    ("item_uuid", "string", False),
    ("nid", "pint", False),
    ("title", "text_en", False),       # text_en = English text, searchable by word (with stemming)
    ("text", "text_en", False),
    ("object_type", "string", True),   # string = exact value, used for filters
    ("collection", "string", True),
    ("subject", "string", True),
    ("year_min", "pint", False),
    ("year_max", "pint", False),
    ("rights", "string", False),
]


def setup_schema():
    # Add our fields to the ai_chunks core. Fields that already exist are skipped,
    # so it is safe to run again.
    core = config.CHUNK_CORE
    # By default Solr guesses a type for any unknown field it sees. We'd rather get an error.
    solr.post(core, "config", json={"set-user-property": {"update.autoCreateFields": "false"}})

    schema = solr.get(core, "schema")["schema"]
    existing_types = {t["name"] for t in schema["fieldTypes"]}
    existing_fields = {f["name"] for f in schema["fields"]}
    added = []

    # The vector field type: a fixed number of floats, compared by cosine similarity.
    vector_type = f"knn_vector_{config.EMBED_DIM}"
    if vector_type not in existing_types:
        solr.post(core, "schema", json={"add-field-type": {
            "name": vector_type, "class": "solr.DenseVectorField",
            "vectorDimension": config.EMBED_DIM, "similarityFunction": "cosine"}})
        added.append(vector_type)

    new_fields = [{"name": name, "type": type_, "multiValued": multi, "stored": True, "indexed": True}
                  for name, type_, multi in FIELDS if name not in existing_fields]
    if "vector" not in existing_fields:
        # Not stored: we search by the vector but never need to read it back.
        new_fields.append({"name": "vector", "type": vector_type, "stored": False, "indexed": True})
    if new_fields:
        solr.post(core, "schema", json={"add-field": new_fields})
        added += [f["name"] for f in new_fields]
    return added


def vectors_path(folder, model):
    # "BAAI/bge-base-en-v1.5" -> data/vectors-baai-bge-base-en-v1-5.npy
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    return Path(folder) / f"vectors-{slug}.npy"


def read_chunks(folder):
    with open(Path(folder) / "chunks.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def export(folder, questions_file, limit=None):
    # Step 1: write every public item's chunks, and the test questions, into `folder`.
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(folder / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in cards.all_chunks(limit):
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            count += 1

    # Grace has no Solr, so we save Solr's keyword results for each test question now.
    # They come from the ai_chunks core, so load the chunks (text is enough) before exporting.
    questions = load_questions(questions_file)
    for q in questions:
        q["keyword"] = [doc["id"] for doc in search.keyword_search(q["question"], rows=50)]
    (folder / "questions.json").write_text(json.dumps(questions, indent=1, ensure_ascii=False), encoding="utf-8")
    return count


def embed(folder, model, batch_size=64):
    # Step 2: embed every chunk in folder/chunks.jsonl and save the vectors next to it.
    chunks = read_chunks(folder)
    vectors = models.embed([c["text"] for c in chunks], model, batch_size=batch_size)
    path = vectors_path(folder, model)
    np.save(path, vectors.astype("float32"))
    return path, vectors.shape


def load(folder, model=None):
    # Step 3: replace everything in ai_chunks with the chunks in `folder`.
    # If a vectors file for `model` exists, the vectors go in too; otherwise only the text
    # (keyword search still works, vector search doesn't).
    chunks = read_chunks(folder)
    path = vectors_path(folder, model or config.EMBED_MODEL)
    vectors = np.load(path) if path.exists() else None
    if vectors is not None and len(vectors) != len(chunks):
        raise ValueError(f"{path.name} has {len(vectors)} vectors but chunks.jsonl has {len(chunks)} chunks; "
                         "run embed again")

    solr.delete_all(config.CHUNK_CORE)
    for start in range(0, len(chunks), 500):  # send 500 at a time
        batch = []
        for i in range(start, min(start + 500, len(chunks))):
            # Solr doesn't accept empty values, so leave out fields that are None (e.g. no dates).
            doc = {key: value for key, value in chunks[i].items() if value is not None}
            if vectors is not None:
                doc["vector"] = vectors[i].tolist()
            batch.append(doc)
        solr.add(config.CHUNK_CORE, batch)
    solr.commit(config.CHUNK_CORE)
    return len(chunks), vectors is not None
