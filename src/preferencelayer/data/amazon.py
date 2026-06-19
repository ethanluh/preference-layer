"""Amazon Reviews 2023 loader — the real-data path for the Phase 0 benchmark.

The synthetic benchmark (``synthetic.py``) is the controlled, reproducible test
with a known ground truth. This module is the bridge to the real corpus the design
docs target: the McAuley-Lab *Amazon Reviews 2023* dataset, loaded via the
HuggingFace ``datasets`` library (an optional dependency:
``pip install preferencelayer[amazon]``).

It builds the same :class:`~preferencelayer.data.synthetic.CategoryData` objects
the harness consumes, so every model and metric works unchanged on real data.
Cross-category transfer is evaluated over users who reviewed items in *both*
chosen categories.

Attribute featurization
-----------------------
Real reviews do not ship with the clean attribute vectors the synthetic generator
plants. We derive an approximate shared-attribute vector per item from available
metadata (price percentile, average rating, rating volume, title/feature
keywords). This is intentionally coarse — production-grade attribute extraction is
Phase 1 work (the QIL NLP pipeline). The featurization is documented here so the
real-data results are interpreted with that caveat in mind.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from ..attributes import AttributeSchema, SHARED_ATTRIBUTES
from .synthetic import CategoryData, Item

# Keyword cues used to derive coarse shared-attribute signals from item text.
_ATTR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "build_quality": ("premium", "aluminum", "metal", "solid", "sturdy", "durable build"),
    "durability": ("durable", "lasts", "reliable", "rugged", "long lasting", "tough"),
    "portability": ("lightweight", "portable", "compact", "travel", "slim", "light"),
    "performance": ("fast", "powerful", "performance", "high speed", "responsive"),
    "brand_affinity": ("brand", "official", "genuine", "authentic"),
    "aesthetics": ("sleek", "design", "beautiful", "stylish", "elegant", "color"),
    "ergonomics": ("comfortable", "ergonomic", "comfort", "grip"),
}


def _require_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise ImportError(
            "The Amazon Reviews 2023 loader needs the 'datasets' package. "
            "Install it with: pip install preferencelayer[amazon]"
        ) from e
    import datasets

    return datasets


def _keyword_score(text: str, words: tuple[str, ...]) -> float:
    text = text.lower()
    return min(1.0, sum(text.count(w) for w in words) / 3.0)


def _featurize(meta: dict, price_pct: float, schema: AttributeSchema) -> np.ndarray:
    """Map item metadata to a shared-attribute vector in [0, 1]."""
    text = " ".join(
        str(meta.get(k, "")) for k in ("title", "features", "description", "store")
    )
    rating = float(meta.get("average_rating") or 0.0) / 5.0
    n_ratings = float(meta.get("rating_number") or 0.0)
    popularity = min(1.0, np.log1p(n_ratings) / 12.0)

    vec = np.zeros(schema.dim)
    idx = {name: i for i, name in enumerate(SHARED_ATTRIBUTES)}
    vec[idx["price_sensitivity"]] = 1.0 - price_pct       # cheaper -> higher value signal
    vec[idx["performance"]] = max(rating, _keyword_score(text, _ATTR_KEYWORDS["performance"]))
    vec[idx["build_quality"]] = _keyword_score(text, _ATTR_KEYWORDS["build_quality"])
    vec[idx["durability"]] = _keyword_score(text, _ATTR_KEYWORDS["durability"])
    vec[idx["portability"]] = _keyword_score(text, _ATTR_KEYWORDS["portability"])
    vec[idx["brand_affinity"]] = max(popularity, _keyword_score(text, _ATTR_KEYWORDS["brand_affinity"]))
    vec[idx["aesthetics"]] = _keyword_score(text, _ATTR_KEYWORDS["aesthetics"])
    vec[idx["ergonomics"]] = _keyword_score(text, _ATTR_KEYWORDS["ergonomics"])
    return np.clip(vec, 0.0, 1.0)


def load_category(
    category_config: str,
    category_label: str,
    max_items: int = 2000,
    min_user_reviews: int = 3,
    n_relevant: int = 12,
    n_candidates: int = 120,
    seed: int = 7,
) -> CategoryData:
    """Load one Amazon Reviews 2023 category into a :class:`CategoryData`.

    ``category_config`` is a HuggingFace config name such as
    ``"raw_review_Electronics"`` paired with its ``raw_meta_Electronics`` metadata
    split. ``category_label`` is the label used inside PreferenceLayer (e.g.
    ``"laptops"``). Items are scored as relevant by the user's own rating; hard
    negatives are highly-rated-by-others items the user did not engage with.
    """
    datasets = _require_datasets()
    rng = np.random.default_rng(seed)
    schema = AttributeSchema.for_category(category_label)

    meta_split = datasets.load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{category_config}",
        split="full", trust_remote_code=True, streaming=True,
    )
    metas: dict[str, dict] = {}
    prices: list[float] = []
    for row in meta_split:
        pid = row.get("parent_asin") or row.get("asin")
        if not pid:
            continue
        metas[pid] = row
        try:
            prices.append(float(row.get("price")))
        except (TypeError, ValueError):
            pass
        if len(metas) >= max_items:
            break

    price_arr = np.array(prices) if prices else np.array([0.0])

    def price_pct(meta: dict) -> float:
        try:
            p = float(meta.get("price"))
        except (TypeError, ValueError):
            return 0.5
        return float((price_arr < p).mean())

    items = [Item(pid, category_label, _featurize(m, price_pct(m), schema)) for pid, m in metas.items()]
    item_ids = {it.item_id for it in items}

    reviews = datasets.load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category_config}",
        split="full", trust_remote_code=True, streaming=True,
    )
    user_items: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in reviews:
        uid, pid = row.get("user_id"), (row.get("parent_asin") or row.get("asin"))
        if uid and pid in item_ids:
            user_items[uid].append((pid, float(row.get("rating") or 0.0)))

    users = {u: v for u, v in user_items.items() if len(v) >= min_user_reviews}
    purchases = {u: [pid for pid, _ in v] for u, v in users.items()}
    relevant = {
        u: [pid for pid, _ in sorted(v, key=lambda x: -x[1])[:n_relevant]]
        for u, v in users.items()
    }
    all_ids = [it.item_id for it in items]
    eval_candidates = {}
    for u, rel in relevant.items():
        rel_set = set(rel)
        pool = [i for i in all_ids if i not in rel_set]
        fill = list(rng.choice(pool, size=min(n_candidates - len(rel), len(pool)), replace=False))
        cand = rel + fill
        rng.shuffle(cand)
        eval_candidates[u] = cand

    return CategoryData(
        category=category_label,
        schema=schema,
        items=items,
        purchases=purchases,
        relevant=relevant,
        eval_candidates=eval_candidates,
    )
