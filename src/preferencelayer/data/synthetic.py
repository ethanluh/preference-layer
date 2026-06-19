"""Synthetic benchmark with a *planted*, transferable cross-category preference signal.

Why synthetic data? The headline Phase 0 claim is about *cross-category transfer*:
preference learned in one category improving recommendations in another. To test
that cleanly we need users whose preferences genuinely carry across categories,
with a known ground truth to score against. The Amazon Reviews 2023 loader
(``data/amazon.py``) provides the real-data path; this module provides the
controlled, fully reproducible benchmark.

The generative model
--------------------
Each user ``u`` has a latent preference over the *shared* attribute vocabulary:

* ``theta_u`` : linear taste over shared attributes (same across all categories).
* ``phi_u``   : sparse *interaction* tastes over pairs of shared attributes
                (e.g. "I only care about battery life when portability is also
                high"). Also identical across categories.

A user's utility for an item with attribute vector ``x`` is::

    utility(u, item) =  theta_u · x_shared
                      + sum_(a,b) phi_u[a,b] * x_a * x_b      (interactions)
                      + theta_local_u · x_specific           (category-local taste)
                      + noise

The linear term ``theta_u · x_shared`` transfers across categories and is fully
recoverable by a *flat attribute* baseline. The interaction term is the part a
linear/mean model structurally cannot represent — it is recoverable only by a
model with edges between attributes. That is the planted advantage the sparse
preference graph is designed to exploit. The category-local term is deliberately
non-transferable noise from the transfer task's point of view.

Nothing here hard-codes "the graph wins": both the flat-attribute baseline and the
graph see exactly the same shared attributes. The graph wins only if attribute
*interactions* matter, which is an empirical, falsifiable property of the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..attributes import AttributeSchema


@dataclass
class Item:
    item_id: str
    category: str
    attributes: np.ndarray  # length == schema.dim, values in [0, 1]


@dataclass
class CategoryData:
    category: str
    schema: AttributeSchema
    items: list[Item]
    # user_id -> ordered list of purchased item_ids (the implicit-feedback sequence)
    purchases: dict[str, list[str]]
    # user_id -> item_ids the user would genuinely buy (ground-truth top set)
    relevant: dict[str, list[str]]
    # user_id -> evaluation candidate set (relevant + hard negatives + random fill).
    # Hard negatives are items that rank high on *linear* attributes but are
    # dispreferred once conditional (interaction) effects are accounted for — the
    # discriminating cases that separate a graph model from a flat one.
    eval_candidates: dict[str, list[str]]

    def item_matrix(self) -> tuple[list[str], np.ndarray]:
        ids = [it.item_id for it in self.items]
        mat = np.stack([it.attributes for it in self.items])
        return ids, mat

    def item_index(self) -> dict[str, Item]:
        return {it.item_id: it for it in self.items}


@dataclass
class SyntheticDataset:
    categories: dict[str, CategoryData]
    user_ids: list[str]
    # ground-truth latent params (for diagnostics / sanity checks only)
    theta: np.ndarray            # (n_users, n_shared)
    phi_pairs: list[tuple[int, int]]
    phi: np.ndarray              # (n_users, n_phi_pairs)


def _softmax_sample(rng: np.random.Generator, utilities: np.ndarray, n: int, temp: float) -> np.ndarray:
    """Sample ``n`` distinct item indices proportional to ``softmax(utility/temp)``."""
    logits = utilities / max(temp, 1e-6)
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    n = min(n, np.count_nonzero(p > 0))
    return rng.choice(len(utilities), size=n, replace=False, p=p)


def generate(
    n_users: int = 600,
    categories: tuple[str, ...] = ("laptops", "headphones"),
    items_per_category: int = 400,
    purchases_per_user: int = 30,
    n_relevant: int = 12,
    n_candidates: int = 120,
    hard_negative_frac: float = 0.6,
    n_interactions: int = 8,
    linear_strength: float = 1.0,
    interaction_strength: float = 2.5,
    local_strength: float = 0.6,
    noise: float = 0.30,
    seed: int = 7,
) -> SyntheticDataset:
    """Generate a synthetic multi-category preference dataset.

    Parameters control how strong the transferable interaction signal is relative
    to linear taste, category-local taste, and noise. With the defaults the
    interaction term is substantial but not dominant, so a graph model has a real
    but earned advantage.
    """
    rng = np.random.default_rng(seed)
    schemas = {c: AttributeSchema.for_category(c) for c in categories}
    n_shared = schemas[categories[0]].n_shared

    user_ids = [f"user_{i:04d}" for i in range(n_users)]

    # Latent linear taste over shared attributes, centered so signs vary.
    theta = rng.normal(0.0, linear_strength, size=(n_users, n_shared))

    # Sparse interaction structure over shared attribute pairs, shared by all users
    # at the *structural* level (same pairs matter) but with per-user coefficients.
    all_pairs = [(a, b) for a in range(n_shared) for b in range(a + 1, n_shared)]
    pair_idx = rng.choice(len(all_pairs), size=min(n_interactions, len(all_pairs)), replace=False)
    phi_pairs = [all_pairs[i] for i in pair_idx]
    phi = rng.normal(0.0, interaction_strength, size=(n_users, len(phi_pairs)))

    def make_items(category: str) -> list[Item]:
        schema = schemas[category]
        items = []
        attrs = rng.random(size=(items_per_category, schema.dim))  # uniform [0,1]
        for j in range(items_per_category):
            items.append(Item(f"{category}_item_{j:04d}", category, attrs[j]))
        return items

    def utilities(user: int, category: str, items: list[Item]) -> tuple[np.ndarray, np.ndarray]:
        """Return (full_utility, linear_only_utility) for every item.

        The linear-only utility is what a flat attribute model can at best recover;
        the gap between the two is the interaction signal only a graph can capture.
        """
        X = np.stack([it.attributes for it in items])          # (N, dim)
        x_shared = X[:, :n_shared]                              # (N, n_shared)
        x_local = X[:, n_shared:]                               # (N, dim - n_shared)

        lin = x_shared @ theta[user]                           # linear shared taste
        if x_local.shape[1] > 0:                               # category-local taste
            local_w = rng.normal(0.0, local_strength, size=x_local.shape[1])
            lin = lin + x_local @ local_w

        u = lin.copy()
        for k, (a, b) in enumerate(phi_pairs):                 # interactions
            u += phi[user, k] * x_shared[:, a] * x_shared[:, b]

        eps = rng.normal(0.0, noise, size=len(items))
        return u + eps, lin

    n_hard = int(round(hard_negative_frac * (n_candidates - n_relevant)))
    n_rand = n_candidates - n_relevant - n_hard

    cat_data: dict[str, CategoryData] = {}
    for category in categories:
        items = make_items(category)
        ids = [it.item_id for it in items]
        purchases: dict[str, list[str]] = {}
        relevant: dict[str, list[str]] = {}
        eval_candidates: dict[str, list[str]] = {}
        for ui, uid in enumerate(user_ids):
            u_full, u_lin = utilities(ui, category, items)
            full_order = np.argsort(-u_full)
            top = full_order[:n_relevant]
            rel_ids = [items[t].item_id for t in top]
            relevant[uid] = rel_ids
            rel_set = set(rel_ids)

            # Observed purchases: noisy, preference-biased sample (implicit feedback).
            chosen = _softmax_sample(rng, u_full, purchases_per_user, temp=0.5)
            purchases[uid] = [items[c].item_id for c in chosen]

            # Hard negatives: items a linear model ranks highly but that are NOT in
            # the true preferred set (because interactions pull them down).
            lin_order = np.argsort(-u_lin)
            hard = [items[i].item_id for i in lin_order if items[i].item_id not in rel_set][:n_hard]
            hard_set = set(hard)
            # Random fill from the remainder.
            pool = [i for i in ids if i not in rel_set and i not in hard_set]
            fill = list(rng.choice(pool, size=min(n_rand, len(pool)), replace=False))
            cand = rel_ids + hard + fill
            rng.shuffle(cand)
            eval_candidates[uid] = cand

        cat_data[category] = CategoryData(
            category=category,
            schema=schemas[category],
            items=items,
            purchases=purchases,
            relevant=relevant,
            eval_candidates=eval_candidates,
        )

    return SyntheticDataset(
        categories=cat_data,
        user_ids=user_ids,
        theta=theta,
        phi_pairs=phi_pairs,
        phi=phi,
    )
