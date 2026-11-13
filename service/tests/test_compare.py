# Run compare() on a tiny data folder with fake models, so the test needs no GPU or downloads.

import json

import numpy as np

from tamu_ai import compare as compare_module
from tamu_ai import models


def fake_embed(texts, model, query=False, batch_size=32):
    # A 2-number "vector": [1, 0] if the text mentions maps, otherwise [0, 1].
    return np.array([[1.0, 0.0] if "map" in t.lower() else [0.0, 1.0] for t in texts], dtype="float32")


def fake_rerank(query, texts, model):
    return [1.0 if "cuba" in t.lower() else 0.0 for t in texts]


def test_compare_scores_every_variant(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "embed", fake_embed)
    monkeypatch.setattr(models, "rerank", fake_rerank)

    chunks = [{"id": "m1:card:0", "item_uuid": "m1", "text": "Map of Texas"},
              {"id": "m2:card:0", "item_uuid": "m2", "text": "Map of Cuba"},
              {"id": "p1:card:0", "item_uuid": "p1", "text": "Photo of a football game"}]
    (tmp_path / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")
    questions = [{"id": "q1", "question": "old maps of Cuba", "relevant": ["m2"], "style": "keyword",
                  "keyword": ["p1:card:0", "m1:card:0", "missing:card:0"]}]
    (tmp_path / "questions.json").write_text(json.dumps(questions), encoding="utf-8")

    results = compare_module.compare(tmp_path, ["test/fake-model"], rerankers=["test/fake-reranker"],
                                     keyword_weights=[1.0])

    assert list(results) == ["keyword", "fake-model vector", "fake-model hybrid kw=1",
                             "fake-model hybrid kw=1 + fake-reranker"]
    assert results["keyword"]["recall"] == 0.0                         # keyword never found m2
    assert results["fake-model hybrid kw=1 + fake-reranker"]["mrr"] == 1.0  # the reranker puts Cuba first
    assert (tmp_path / "vectors-test-fake-model.npy").exists()           # vectors saved for reuse
