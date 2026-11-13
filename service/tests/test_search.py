from tamu_ai.search import filter_queries, max_per_item, rrf


def test_rrf_rewards_ids_found_by_both_lists():
    merged = rrf([["a", "b", "c"], ["c", "d"]], weights=[1.0, 1.0])
    assert merged[0] == "c"  # "c" is in both lists, so it beats "a", which is first in only one
    assert set(merged) == {"a", "b", "c", "d"}


def test_rrf_weight_lets_one_list_lead():
    assert rrf([["kw"], ["vec"]], weights=[0.5, 1.0])[0] == "vec"
    assert rrf([["kw"], ["vec"]], weights=[1.0, 0.5])[0] == "kw"


def test_filter_queries():
    assert filter_queries() == []
    fq = filter_queries(object_types=["Map"], collections=['Say "hi"'], year_from=1940, year_to=1950)
    assert fq[0] == 'object_type:("Map")'
    assert fq[1] == 'collection:("Say \\"hi\\"")'
    assert fq[2] == "year_max:[1940 TO *] AND year_min:[* TO 1950]"


def test_max_per_item():
    docs = [{"item_uuid": "a"}, {"item_uuid": "a"}, {"item_uuid": "a"}, {"item_uuid": "b"}]
    assert [d["item_uuid"] for d in max_per_item(docs, 2)] == ["a", "a", "b"]
