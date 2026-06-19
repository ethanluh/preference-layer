import numpy as np

from preferencelayer.data import synthetic
from preferencelayer.eval import ExperimentHarness
from preferencelayer.models import (
    FlatAttributeRecommender,
    FlatItemEmbeddingRecommender,
    SparsePreferenceGraph,
)


def _facs():
    return {
        "flat_item_embedding": lambda: FlatItemEmbeddingRecommender(),
        "flat_attribute": lambda: FlatAttributeRecommender(),
        "preference_graph": lambda: SparsePreferenceGraph(),
    }


def test_graph_discovers_true_interaction_edges():
    ds = synthetic.generate(n_users=200, seed=7)
    src = ds.categories["laptops"]
    idx = src.item_index()
    per_user = [np.stack([idx[i].attributes for i in src.purchases[u]]) for u in ds.user_ids]
    _, cat = src.item_matrix()
    g = SparsePreferenceGraph(n_edges=10)
    g.prepare(cat, per_user, src.schema.n_shared)
    # Nearly all planted interaction pairs are recovered within the top-10 edges
    # (variance-of-within-user-correlation reliably finds >= 7 of 8).
    recovered = len(set(ds.phi_pairs) & set(g.edges))
    assert recovered >= 7


def test_all_models_beat_popularity_on_transfer():
    ds = synthetic.generate(n_users=150, seed=7)
    h = ExperimentHarness(ds, k=10, seed=13)
    res = h.run_transfer(_facs(), "laptops", "headphones")
    pop = res["popularity"].ndcg
    for name in ("flat_attribute", "preference_graph"):
        assert res[name].ndcg > pop


def test_graph_beats_flat_on_cross_category_transfer():
    """The headline Phase 0 claim and go/no-go gate (>= 5% NDCG@10 lift)."""
    ds = synthetic.generate(n_users=250, seed=7)
    h = ExperimentHarness(ds, k=10, seed=13)
    res = h.run_transfer(_facs(), "laptops", "headphones")
    comp = next(c for c in h.compare(res) if c.baseline == "flat_attribute")
    assert comp.rel_gain_pct >= 5.0
    assert comp.p_value < 0.05


def test_scores_are_finite_and_sized():
    ds = synthetic.generate(n_users=20, items_per_category=60, seed=1)
    src = ds.categories["laptops"]
    idx = src.item_index()
    per_user = [np.stack([idx[i].attributes for i in src.purchases[u]]) for u in ds.user_ids]
    _, cat = src.item_matrix()
    model = SparsePreferenceGraph()
    model.prepare(cat, per_user, src.schema.n_shared)
    uid = ds.user_ids[0]
    purchased = np.stack([idx[i].attributes for i in src.purchases[uid]])
    state = model.fit(purchased, cat, src.schema.n_shared)
    scores = model.score(state, cat, src.schema.n_shared)
    assert scores.shape == (cat.shape[0],)
    assert np.all(np.isfinite(scores))
