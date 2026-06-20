"""Amazon Reviews 2023 loader — the real-data path for the Phase 0/1 benchmark.

The synthetic benchmark (``synthetic.py``) is the controlled, reproducible test with a
known ground truth. This module is the bridge to the real corpus the design docs
target: the McAuley-Lab *Amazon Reviews 2023* dataset. It builds the same
:class:`~preferencelayer.data.synthetic.CategoryData` objects the harness consumes, so
every model and metric works unchanged on real data.

How the data is loaded
----------------------
The dataset is read directly from its Hugging Face **Parquet** (item metadata) and the
ready-made **0-core ``last_out`` benchmark CSV** (user→item→rating interactions) —
*not* the legacy loading script. (The original script-based path stopped working once
``datasets`` dropped ``trust_remote_code`` / script datasets; the Parquet + CSV files
need only ``huggingface_hub`` + ``pandas``/``pyarrow``, the ``[amazon]`` extra.) Item
relevance is the user's own rating; candidate sets mix the user's relevant items with
**hard negatives** — globally popular items the user did *not* engage with — so a
popularity baseline cannot win for free, exactly as the synthetic benchmark does.

Attribute featurization
-----------------------
Real items do not ship with the clean attribute vectors the synthetic generator plants.
We derive a coarse shared-attribute vector per item from metadata (price percentile,
average rating, rating volume, title/feature keywords). This is intentionally coarse —
production-grade attribute extraction is Phase 1 work (the QIL NLP pipeline) — and the
real-data results in ``docs/phase1-amazon-realdata.md`` are interpreted with that caveat.
The assembly logic (:func:`build_category_data`) is split from the network fetch so it
can be unit-tested offline.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np

from ..attributes import AttributeSchema, SHARED_ATTRIBUTES
from .synthetic import CategoryData, Item

_REPO = "McAuley-Lab/Amazon-Reviews-2023"
_META_COLUMNS = ("parent_asin", "title", "features", "description", "store",
                 "average_rating", "rating_number", "price")

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


def _require_deps():
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise ImportError(
            "The Amazon Reviews 2023 loader needs pandas + pyarrow + huggingface_hub. "
            "Install them with: pip install preferencelayer[amazon]"
        ) from e
    return pd, hf_hub_download, list_repo_files


def _keyword_score(text: str, words: tuple[str, ...]) -> float:
    text = text.lower()
    return min(1.0, sum(text.count(w) for w in words) / 3.0)


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _featurize(meta: dict, price_pct: float, schema: AttributeSchema) -> np.ndarray:
    """Map item metadata to a shared-attribute vector in [0, 1]."""
    text = " ".join(str(meta.get(k, "")) for k in ("title", "features", "description", "store"))
    rating = (_to_float(meta.get("average_rating")) or 0.0) / 5.0
    n_ratings = _to_float(meta.get("rating_number")) or 0.0
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


def build_category_data(
    metas: dict[str, dict],
    interactions: Iterable[tuple[str, str, float]],
    category_label: str,
    *,
    min_user_reviews: int = 5,
    n_relevant: int = 12,
    n_candidates: int = 120,
    n_hard_negatives: int = 50,
    seed: int = 7,
) -> CategoryData:
    """Assemble a :class:`CategoryData` from item metadata + rating interactions.

    Pure (no network), so it is unit-testable on small in-memory inputs. ``metas`` maps
    item id -> metadata dict; ``interactions`` yields ``(user_id, item_id, rating)``.
    Relevance is the user's top-rated items; candidate sets are
    ``relevant + hard negatives (popular items the user didn't touch) + random fill`` so
    a popularity baseline cannot trivially win.
    """
    schema = AttributeSchema.for_category(category_label)
    rng = np.random.default_rng(seed)

    price_arr = np.array([p for p in (_to_float(m.get("price")) for m in metas.values()) if p is not None]
                         or [0.0])

    def price_pct(meta: dict) -> float:
        p = _to_float(meta.get("price"))
        return float((price_arr < p).mean()) if p is not None else 0.5

    items = [Item(pid, category_label, _featurize(m, price_pct(m), schema)) for pid, m in metas.items()]
    item_ids = {it.item_id for it in items}

    user_items: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for uid, pid, rating in interactions:
        if uid and pid in item_ids:
            user_items[uid].append((pid, float(rating)))
    users = {u: v for u, v in user_items.items() if len(v) >= min_user_reviews}

    purchases = {u: [pid for pid, _ in v] for u, v in users.items()}
    relevant = {
        u: [pid for pid, _ in sorted(v, key=lambda x: -x[1])[:n_relevant]] for u, v in users.items()
    }

    # Hard negatives: globally popular items, so popularity ranks them high and cannot
    # coast on the fact that a user's favorites are also broadly popular.
    pop = Counter(pid for v in users.values() for pid, _ in v)
    popular = [pid for pid, _ in pop.most_common()]
    all_ids = [it.item_id for it in items]

    eval_candidates: dict[str, list[str]] = {}
    for u, rel in relevant.items():
        seen = set(rel) | set(purchases[u])
        hard = [pid for pid in popular if pid not in seen][:n_hard_negatives]
        seen |= set(hard)
        pool = [i for i in all_ids if i not in seen]
        n_fill = max(0, n_candidates - len(rel) - len(hard))
        fill = list(rng.choice(pool, size=min(n_fill, len(pool)), replace=False)) if pool else []
        cand = rel + hard + fill
        rng.shuffle(cand)
        eval_candidates[u] = cand

    return CategoryData(
        category=category_label, schema=schema, items=items,
        purchases=purchases, relevant=relevant, eval_candidates=eval_candidates,
    )


def _collect_metas(shard_records: Iterable[Iterable[dict]], max_items: int) -> dict[str, dict]:
    """Collect unique item metadata across shards up to a **row-precise** ``max_items`` cap.

    ``shard_records`` yields one record iterable per metadata shard. Collection stops as
    soon as ``max_items`` unique items (keyed by ``parent_asin``) have been seen — mid-shard
    if necessary — rather than overshooting to a shard boundary. Because the cap short-
    circuits, a lazy ``shard_records`` generator never produces shards beyond the one that
    fills the cap (so no extra downloads). Split out from :func:`load_category` so it is
    unit-testable offline. A non-positive ``max_items`` yields an empty result.
    """
    metas: dict[str, dict] = {}
    if max_items <= 0:
        return metas
    for records in shard_records:
        for row in records:
            pid = row.get("parent_asin")
            if pid and pid not in metas:
                metas[pid] = row
                if len(metas) >= max_items:
                    return metas
    return metas


def load_category(
    category_config: str,
    category_label: str,
    max_items: int = 4000,
    max_interactions: int | None = None,
    **build_kwargs,
) -> CategoryData:
    """Load one Amazon Reviews 2023 category into a :class:`CategoryData`.

    ``category_config`` is the dataset category name (e.g. ``"All_Beauty"``,
    ``"Electronics"``); ``category_label`` is the label used inside PreferenceLayer.
    Reads item metadata from the ``raw_meta_<config>`` Parquet shards (up to
    ``max_items``) and interactions from the ``0core/last_out`` benchmark CSV (up to
    ``max_interactions`` rows, if given), then delegates to :func:`build_category_data`.
    """
    pd, hf_hub_download, list_repo_files = _require_deps()

    shards = sorted(f for f in list_repo_files(_REPO, repo_type="dataset")
                    if f.startswith(f"raw_meta_{category_config}/") and f.endswith(".parquet"))
    if not shards:
        raise ValueError(f"no Parquet metadata for category '{category_config}' "
                         f"(it may only exist via the legacy loading script)")
    def _shard_records():
        # Lazy: a shard is downloaded only when _collect_metas asks for it, so once the
        # cap is hit no further shards are fetched.
        for shard in shards:
            frame = pd.read_parquet(hf_hub_download(_REPO, shard, repo_type="dataset"),
                                    columns=list(_META_COLUMNS))
            yield frame.to_dict("records")

    metas = _collect_metas(_shard_records(), max_items)

    csv = hf_hub_download(_REPO, f"benchmark/0core/last_out/{category_config}.train.csv",
                          repo_type="dataset")
    reviews = pd.read_csv(csv, usecols=["user_id", "parent_asin", "rating"], nrows=max_interactions)
    interactions = zip(reviews.user_id, reviews.parent_asin, reviews.rating)

    return build_category_data(metas, interactions, category_label, **build_kwargs)
