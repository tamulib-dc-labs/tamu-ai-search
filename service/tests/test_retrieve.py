from tamu_ai.evaluate import mrr, ndcg_at, ranked_items, recall_at
from tamu_ai.retrieve import Filters, Hit, diversify, rrf


def test_rrf_rewards_ids_found_by_both_lists():
    fused = rrf([["a", "b", "c"], ["c", "d", "a"]])
    order = [doc_id for doc_id, _ in fused]
    assert order[:2] == ["a", "c"]  # in both lists
    assert set(order) == {"a", "b", "c", "d"}
    assert fused[0][1] == 1 / 61 + 1 / 63


def _hit(chunk: str, item: str) -> Hit:
    return Hit(id=chunk, item_uuid=item, nid=0, title="", kind="card", anchor="", text="", score=0.0)


def test_diversify_caps_chunks_per_item_and_keeps_order():
    hits = [_hit("a1", "a"), _hit("a2", "a"), _hit("a3", "a"), _hit("b1", "b")]
    assert [h.id for h in diversify(hits, per_item=2)] == ["a1", "a2", "b1"]


def test_filters_build_solr_filter_queries():
    fq = Filters(object_types=["Image", "Map"], collections=['Say "hi"'], year_from=1930, year_to=1939).to_fq()
    assert fq == [
        'object_type:("Image" OR "Map")',
        'collection:("Say \\"hi\\"")',
        "year_max:[1930 TO *] AND year_min:[* TO 1939]",
    ]
    assert Filters(year_from=1900).to_fq() == ["year_max:[1900 TO *] AND year_min:[* TO *]"]


def test_metrics():
    ranked = ranked_items(["x", "a", "x", "b"])
    assert ranked == ["x", "a", "b"]
    relevant = {"a", "b"}
    assert recall_at(ranked, relevant, 2) == 0.5
    assert mrr(ranked, relevant) == 0.5
    assert ndcg_at(["a", "b"], relevant, 10) == 1.0
    assert 0 < ndcg_at(ranked, relevant, 10) < 1
