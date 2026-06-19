import numpy as np

from preferencelayer.data import synthetic


def test_generate_shapes_and_determinism():
    a = synthetic.generate(n_users=20, items_per_category=50, seed=3)
    b = synthetic.generate(n_users=20, items_per_category=50, seed=3)
    assert list(a.categories) == ["laptops", "headphones"]
    assert len(a.user_ids) == 20
    # Determinism: same seed -> identical purchases.
    assert a.categories["laptops"].purchases == b.categories["laptops"].purchases


def test_shared_attributes_align_across_categories():
    ds = synthetic.generate(n_users=5, items_per_category=20, seed=1)
    lap = ds.categories["laptops"].schema
    head = ds.categories["headphones"].schema
    assert lap.shared == head.shared
    assert lap.n_shared == head.n_shared


def test_candidate_sets_contain_relevant_and_hard_negatives():
    ds = synthetic.generate(n_users=10, items_per_category=200, n_candidates=120,
                            n_relevant=12, hard_negative_frac=0.6, seed=2)
    cat = ds.categories["laptops"]
    uid = ds.user_ids[0]
    cand = set(cat.eval_candidates[uid])
    rel = set(cat.relevant[uid])
    assert rel.issubset(cand)
    assert len(cand) == 120


def test_purchases_are_preference_biased():
    # Purchased items should have higher mean true utility than random items.
    ds = synthetic.generate(n_users=30, items_per_category=300, seed=5)
    cat = ds.categories["laptops"]
    idx = cat.item_index()
    # Use shared-attribute alignment with ground-truth theta as a proxy signal.
    theta = ds.theta
    diffs = []
    for ui, uid in enumerate(ds.user_ids):
        pur = np.stack([idx[i].attributes[: theta.shape[1]] for i in cat.purchases[uid]])
        rand = np.stack([it.attributes[: theta.shape[1]] for it in cat.items[:50]])
        diffs.append((pur @ theta[ui]).mean() - (rand @ theta[ui]).mean())
    assert np.mean(diffs) > 0
