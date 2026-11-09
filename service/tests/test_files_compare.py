import json

import numpy as np

from tamu_ai import embed
from tamu_ai.chunking import Chunk
from tamu_ai.compare import run_compare
from tamu_ai.files import _item_batches, read_chunks, read_vectors, save_vectors, write_jsonl
from tamu_ai.retrieve import rrf


def _chunk(item: str, n: int, text: str = "") -> Chunk:
    return Chunk(id=f"{item}:card:{n}", item_uuid=item, nid=1, title=item, kind="card", anchor="",
                 text=text or f"{item} text", object_types=["Image"], year_min=1930, year_max=1939)


def test_chunk_records_round_trip_through_jsonl(tmp_path):
    chunks = [_chunk("a", 0), _chunk("a", 1)]
    write_jsonl(tmp_path / "c.jsonl", (c.to_record() for c in chunks))
    assert list(read_chunks(tmp_path / "c.jsonl")) == chunks
    assert "vector" not in chunks[0].to_solr(None)


def test_vectors_round_trip(tmp_path):
    save_vectors(tmp_path / "v.npz", ["x", "y"], np.eye(2), "some/model")
    ids, vectors, model = read_vectors(tmp_path / "v.npz")
    assert ids == ["x", "y"] and model == "some/model" and vectors.dtype == np.float32


def test_item_batches_never_split_an_item():
    chunks = [_chunk("a", 0), _chunk("a", 1), _chunk("b", 0), _chunk("c", 0), _chunk("c", 1)]
    batches = list(_item_batches(chunks, max_chunks=3))
    assert [[c.id for c in b] for b in batches] == [["a:card:0", "a:card:1", "b:card:0"], ["c:card:0", "c:card:1"]]


def test_weighted_rrf_lets_the_second_list_lead():
    equal = [i for i, _ in rrf([["kw"], ["vec"]])]
    assert set(equal) == {"kw", "vec"}
    assert [i for i, _ in rrf([["kw"], ["vec"]], weights=[0.5, 1.0])] == ["vec", "kw"]


class FakeEmbedder:
    """Puts 'cows' texts on one axis and 'maps' texts on the other."""

    def __init__(self, name):
        self.name = name

    def _vec(self, text):
        v = np.array([text.count("cow") + 0.01, text.count("map") + 0.01], dtype="float32")
        return v / np.linalg.norm(v)

    def passages(self, texts, batch_size=64):
        return np.stack([self._vec(t) for t in texts])

    queries = passages


class FakeReranker:
    def __init__(self, name):
        self.name = name

    def scores(self, query, passages):
        return [float(p.count("cow")) for p in passages]


def test_compare_scores_every_variant_without_solr(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "Embedder", FakeEmbedder)
    monkeypatch.setattr(embed, "Reranker", FakeReranker)
    chunks = [_chunk("cows", 0, "cow cow in a field"), _chunk("maps", 0, "map of Cuba")]
    write_jsonl(tmp_path / "c.jsonl", (c.to_record() for c in chunks))
    (tmp_path / "pack.json").write_text(json.dumps({"questions": [
        {"id": "q1", "question": "photos of a cow", "relevant": ["cows"], "style": "meaning",
         "keyword": ["maps:card:0"]},
    ]}))
    report = run_compare(tmp_path / "c.jsonl", tmp_path / "pack.json", ["fake/emb"], ["fake/rr"],
                         keyword_weights=[1.0], rerank_depths=[10], cache_dir=tmp_path / "cache")
    modes = report["modes"]
    assert modes["keyword (solr)"]["mrr"] == 0.0  # keyword list only found the map
    assert modes["emb vector"]["mrr"] == 1.0
    assert modes["emb hybrid kw=1 + rr@10"]["mrr"] == 1.0
    assert report["by_style"]["emb vector"]["meaning"]["questions"] == 1
    assert (tmp_path / "cache" / "vectors-fake-emb.npz").exists()
