from evaluation.evaluate import mrr, ndcg_at_k, precision_at_k, recall_at_k
from retrieval.ranker import preference_score


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, 3) == 2 / 3


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c", "d"}, 2) == 1 / 3


def test_mrr():
    assert mrr(["x", "b", "a"], {"a", "b"}) == 0.5


def test_ndcg():
    assert 0.0 < ndcg_at_k(["a", "x", "b"], {"a", "b"}, 3) <= 1.0


def test_preference_score():
    assert preference_score("Taylor Swift", {"Taylor Swift": 2}) > 0
    assert preference_score("Drake", {"Drake": -2}) < 0
