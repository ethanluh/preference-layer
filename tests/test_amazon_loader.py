"""Offline tests for the Amazon Reviews 2023 assembly logic.

These exercise ``build_category_data`` (the pure, network-free core of the loader) on
small in-memory inputs — the live ``load_category`` fetch is not run in CI.
"""

import numpy as np

from preferencelayer.data.amazon import _collect_metas, build_category_data
from preferencelayer.data.synthetic import CategoryData


def _meta(pid, title="", price=None, rating=4.0, n=10):
    return {"parent_asin": pid, "title": title, "features": "", "description": "",
            "store": "", "average_rating": rating, "rating_number": n, "price": price}


def _metas():
    return {
        "p_perf": _meta("p_perf", "fast powerful performance laptop", price="500"),
        "p_port": _meta("p_port", "lightweight portable slim travel", price="300"),
        "p_build": _meta("p_build", "premium aluminum sturdy metal", price="900"),
        "p_plain1": _meta("p_plain1", "thing one", price="50"),
        "p_plain2": _meta("p_plain2", "thing two", price="60"),
        "p_plain3": _meta("p_plain3", "thing three", price="70"),
        "p_plain4": _meta("p_plain4", "thing four", price="80"),
    }


def test_build_category_data_shapes_and_relevance():
    metas = _metas()
    # One user rates 6 items; the two 5-star items should be the top of 'relevant'.
    interactions = [
        ("u1", "p_perf", 5.0), ("u1", "p_port", 5.0), ("u1", "p_build", 3.0),
        ("u1", "p_plain1", 2.0), ("u1", "p_plain2", 1.0), ("u1", "p_plain3", 4.0),
    ]
    cat = build_category_data(metas, interactions, "laptops",
                              min_user_reviews=5, n_relevant=2, n_candidates=7, n_hard_negatives=2, seed=1)
    assert isinstance(cat, CategoryData)
    assert len(cat.items) == len(metas)
    assert set(cat.relevant["u1"]) == {"p_perf", "p_port"}        # top-2 by rating
    # Candidate set contains the relevant items and is drawn from the catalog.
    assert set(cat.relevant["u1"]).issubset(set(cat.eval_candidates["u1"]))
    assert all(c in metas for c in cat.eval_candidates["u1"])


def test_featurization_reflects_keywords():
    metas = _metas()
    cat = build_category_data(metas, [("u1", p, 5.0) for p in metas] , "laptops",
                              min_user_reviews=1, seed=1)
    idx = cat.item_index()
    names = cat.schema.names
    # The 'performance' item scores higher on the performance attribute than a plain item.
    perf_col = names.index("performance")
    assert idx["p_perf"].attributes[perf_col] > idx["p_plain1"].attributes[perf_col]
    port_col = names.index("portability")
    assert idx["p_port"].attributes[port_col] > idx["p_plain1"].attributes[port_col]
    assert np.all((idx["p_perf"].attributes >= 0) & (idx["p_perf"].attributes <= 1))


def test_min_user_reviews_filters_sparse_users():
    metas = _metas()
    interactions = [("u_sparse", "p_perf", 5.0), ("u_sparse", "p_port", 4.0)]   # only 2
    cat = build_category_data(metas, interactions, "laptops", min_user_reviews=5, seed=1)
    assert "u_sparse" not in cat.purchases     # dropped: too few reviews
    assert cat.purchases == {}


def test_unknown_items_in_interactions_are_ignored():
    metas = _metas()
    interactions = [("u1", "p_perf", 5.0), ("u1", "ghost", 5.0)] + [("u1", p, 4.0) for p in list(metas)[:4]]
    cat = build_category_data(metas, interactions, "laptops", min_user_reviews=3, seed=1)
    assert "ghost" not in cat.purchases.get("u1", [])    # item not in metadata is skipped


def _shard(n, start=0):
    return [{"parent_asin": f"p{i}"} for i in range(start, start + n)]


def test_collect_metas_caps_mid_shard():
    # A single shard of 10 items, capped at 3 -> stops mid-shard, no overshoot.
    metas = _collect_metas([_shard(10)], max_items=3)
    assert len(metas) == 3
    assert set(metas) == {"p0", "p1", "p2"}


def test_collect_metas_dedups_and_spans_shards():
    # Cap reached only in the second shard; duplicate ids across shards are not recounted.
    shards = [_shard(2), _shard(3, start=1)]  # p0,p1 | p1,p2,p3 -> unique p0..p3
    metas = _collect_metas(shards, max_items=4)
    assert set(metas) == {"p0", "p1", "p2", "p3"}


def test_collect_metas_is_lazy_no_extra_shards():
    # Once the cap is hit the generator must not be asked for further shards (no extra
    # downloads in the real loader). A second shard that raises proves we never reach it.
    def gen():
        yield _shard(5)
        raise AssertionError("second shard fetched after cap was already reached")

    metas = _collect_metas(gen(), max_items=3)
    assert len(metas) == 3


def test_collect_metas_under_cap_returns_all():
    metas = _collect_metas([_shard(2), _shard(2, start=2)], max_items=100)
    assert len(metas) == 4
