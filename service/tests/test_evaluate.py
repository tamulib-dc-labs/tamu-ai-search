import pytest

from tamu_ai.evaluate import mrr, ndcg, recall, score


def test_metrics():
    relevant = {"a", "b"}
    assert recall(["a", "x", "b"], relevant) == 1.0
    assert recall(["x", "a"], relevant) == 0.5
    assert ndcg(["a", "b", "x"], relevant) == pytest.approx(1.0)
    assert ndcg(["x", "x", "a"], relevant) < ndcg(["a", "x", "x"], relevant)
    assert mrr(["x", "b"], relevant) == 0.5
    assert mrr(["x"], relevant) == 0.0


def test_score_counts_an_item_once_and_splits_by_style():
    questions = [{"id": "q1", "relevant": ["a"], "style": "keyword"},
                 {"id": "q2", "relevant": ["b"], "style": "meaning"}]
    # "a" appears twice (two chunks of one item); it should count once, at rank 1.
    result = score(questions, {"q1": ["a", "a"], "q2": ["x", "b"]})
    assert result["recall"] == 1.0
    assert result["ndcg_keyword"] == pytest.approx(1.0)
    assert result["mrr"] == pytest.approx(0.75)
