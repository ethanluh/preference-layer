from preferencelayer.eval import metrics


def test_ndcg_perfect_ranking_is_one():
    ranking = ["a", "b", "c", "x", "y"]
    assert metrics.ndcg_at_k(ranking, {"a", "b", "c"}, k=10) == 1.0


def test_ndcg_zero_when_no_relevant_in_topk():
    ranking = ["x", "y", "z", "a"]
    assert metrics.ndcg_at_k(ranking, {"a"}, k=3) == 0.0


def test_ndcg_rewards_higher_placement():
    rel = {"a"}
    high = metrics.ndcg_at_k(["a", "x", "y"], rel, k=10)
    low = metrics.ndcg_at_k(["x", "y", "a"], rel, k=10)
    assert high > low
    assert high == 1.0


def test_ndcg_empty_relevant_is_zero():
    assert metrics.ndcg_at_k(["a", "b"], set(), k=10) == 0.0


def test_recall_and_hit_rate():
    ranking = ["a", "x", "b", "y"]
    assert abs(metrics.recall_at_k(ranking, {"a", "b", "c"}, k=4) - 2 / 3) < 1e-9
    assert metrics.hit_rate_at_k(ranking, {"a"}, k=1) == 1.0
    assert metrics.hit_rate_at_k(ranking, {"b"}, k=1) == 0.0


def test_mrr():
    assert metrics.mrr(["x", "a", "y"], {"a"}) == 0.5
    assert metrics.mrr(["x", "y"], {"a"}) == 0.0


def test_graded_relevance():
    ranking = ["a", "b"]
    rel_map = {"a": 3.0, "b": 1.0}
    # Perfect order with graded gains -> 1.0
    assert abs(metrics.ndcg_at_k(ranking, {"a", "b"}, k=10, relevance_map=rel_map) - 1.0) < 1e-9
    # Reversed order -> < 1.0
    assert metrics.ndcg_at_k(["b", "a"], {"a", "b"}, k=10, relevance_map=rel_map) < 1.0
