"""Integrated benchmark where ranking needs *both* preference and quality.

Phase 0 validated the two layers in isolation: the preference graph transfers
taste across categories (Claim 1) and the QIL extracts use-profile quality from
text (Claim 2). This module builds the controlled scenario that tests the thing
neither Phase 0 experiment could — the **combination**. It is to the α-blend
what ``data/synthetic.py`` is to the preference graph and ``qil/corpus.py`` is to
extraction: a fully reproducible, planted-ground-truth world where the headline
integration claim is *falsifiable*.

The generative model
--------------------
Every product carries two independent things:

* an **attribute vector** ``x_p`` (shared + category-specific), the substrate of
  *preference*; and
* a **planted quality** ``q[p, use_profile, dim]`` in ``[0, 1]``, the substrate
  of QIL *quality* — deliberately *uncorrelated* with attributes, because a
  product's real-world reliability is not readable off its spec sheet.

A user ``u`` has a latent attribute taste (``theta_u`` linear + ``phi_u``
interactions, as in ``data/synthetic.py``), a ``use_profile``, and — crucially —
a **history length** ``h_u`` spanning sparse cold-start to rich. Their true
utility for a product is::

    utility(u, p) = pref_term(x_p; theta_u, phi_u)
                  + quality_weight * mean_quality(p, use_profile_u)
                  + noise

Both terms matter, so **neither signal alone suffices**. Preference is the
larger driver (``quality_weight`` keeps quality a real but minority share); this
matters for the result, see below.

Why the α-blend should win — and why *adaptive* α beats any fixed α
-------------------------------------------------------------------
The quality estimate the agent gets is community-derived: its accuracy is the
same for every user (the QIL aggregates the same posts regardless of who is
shopping). The *preference* estimate is per-user — fit on that user's ``h_u``
purchases — so its reliability grows with history. That asymmetry is the whole
point:

* **Cold-start users** (small ``h_u``): the preference fit collapses toward the
  population prior and is a poor guide to *their* taste, while quality is
  estimated just as well as for anyone. The right move is a *low* α — lean on
  quality. The credential's confidence is low, so ``alpha_from_confidence``
  produces exactly that.
* **Rich-history users**: the preference fit is both reliable *and* points at the
  dominant utility term, so a *high* α is right. Their confidence is high, so the
  adaptive α rises to match.

A single fixed α cannot be right for both ends: 0.5 over-trusts the garbage
preference estimate of cold-start users and under-trusts the good, dominant
preference signal of rich users. Confidence-adaptive α tracks the per-user
optimum, so it should beat preference-only, quality-only, *and* any fixed blend.
Nothing here hard-codes that outcome — it is an empirical property of the planted
weights and is reported honestly (including if it fails) by the harness.

Credentials and confidence
---------------------------
Per-user ``mean_confidence`` is tied to history exactly the way the preference
graph's own cold-start blend is (``models/graph.py``: ``lam = n / (n + pivot)``).
That is faithful: the credential is confident precisely to the degree the user's
own signal — rather than the population prior — drives their fitted graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..attributes import AttributeSchema
from ..qil.extract import ExtractedSignal
from ..qil.schema import QUALITY_DIMS


@dataclass
class IntegratedProduct:
    product_id: str
    attributes: np.ndarray                       # length schema.dim, values in [0, 1]
    quality: dict[tuple[str, str], float]        # (use_profile, dim) -> planted mean


@dataclass
class IntegratedUser:
    user_id: str
    use_profile: str
    history_len: int
    cohort: str                  # "cold" | "warm" | "rich" (for subgroup analysis)
    purchases: list[str]         # observed purchase product ids (the fit data)
    relevant: list[str]          # ground-truth top set by true utility
    candidates: list[str]        # eval candidate set (relevant + traps + fill)
    mean_confidence: float       # credential confidence -> drives adaptive alpha


@dataclass
class IntegratedScenario:
    schema: AttributeSchema
    products: list[IntegratedProduct]
    users: list[IntegratedUser]
    signals: list[ExtractedSignal]   # feed straight into a QualityAggregator
    use_profiles: tuple[str, ...]
    quality_weight: float
    # latent params, for diagnostics / sanity checks only
    theta: np.ndarray = field(default_factory=lambda: np.empty(0))
    phi_pairs: list[tuple[int, int]] = field(default_factory=list)
    phi: np.ndarray = field(default_factory=lambda: np.empty(0))

    def product_index(self) -> dict[str, IntegratedProduct]:
        return {p.product_id: p for p in self.products}

    def catalog_matrix(self) -> tuple[list[str], np.ndarray]:
        ids = [p.product_id for p in self.products]
        return ids, np.stack([p.attributes for p in self.products])


# History-length cohorts. Cold-start users have a handful of purchases; rich
# users have a deep history. The spread is what makes adaptive alpha exercisable.
_COHORTS: tuple[tuple[str, int, int], ...] = (
    ("cold", 1, 3),
    ("warm", 6, 12),
    ("rich", 22, 40),
)

# A brand-new user with *zero* history: no personal signal at all, so the
# preference fit collapses to the population prior and credential confidence is 0.
# This is the regime the architecture's "lean on quality for new users" intuition
# targets. Prepended to the cohorts only when ``include_new_cohort=True`` so the
# default benchmark (and the integration / quality-robustness experiments) is
# unchanged.
_NEW_COHORT: tuple[str, int, int] = ("new", 0, 0)


def _softmax_sample(rng: np.random.Generator, utilities: np.ndarray, n: int, temp: float) -> list[int]:
    logits = utilities / max(temp, 1e-6)
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    n = min(n, int(np.count_nonzero(p > 0)))
    return list(rng.choice(len(utilities), size=n, replace=False, p=p))


def generate(
    n_users: int = 300,
    category: str = "laptops",
    n_products: int = 160,
    use_profiles: tuple[str, ...] = ("gaming", "heavy_use", "travel"),
    quality_weight: float = 0.6,
    n_relevant: int = 10,
    n_pref_traps: int = 8,
    n_quality_traps: int = 20,
    n_candidates: int = 80,
    purchase_quality_weight: float = 0.1,
    n_interactions: int = 6,
    linear_strength: float = 1.5,
    interaction_strength: float = 3.0,
    noise: float = 0.15,
    cold_start_pivot: int = 8,
    signals_per_cell: int = 14,
    evidence_lo: int | None = None,
    evidence_hi: int | None = None,
    signal_obs_noise: float = 0.10,
    signal_confidence: float = 0.85,
    include_new_cohort: bool = False,
    seed: int = 23,
) -> IntegratedScenario:
    """Generate the integrated preference+quality benchmark.

    ``quality_weight`` sets how much planted quality contributes to true utility
    relative to attribute preference (kept a minority share so preference is the
    dominant — and, for rich users, well-estimated — driver). ``signals_per_cell``
    and ``signal_obs_noise`` control how sharply the QIL can estimate quality.

    **Evidence uniformity.** By default every product gets ``signals_per_cell``
    quality observations (a uniform-evidence world). Pass ``evidence_lo`` /
    ``evidence_hi`` to instead draw each *product's* evidence count from
    ``[evidence_lo, evidence_hi]`` — the realistic case where some products are
    heavily reviewed and others barely. Evidence is drawn *independently* of true
    quality, so it is a pure reliability signal, not a quality proxy. Thin-evidence
    products get unreliable posteriors (the Normal-Normal aggregator shrinks them
    toward the neutral prior), which is precisely what an *evidence-aware* α should
    detect and route around by leaning on preference for those items. Setting
    ``evidence_lo == evidence_hi`` recovers a uniform regime (the honest control).
    """
    rng = np.random.default_rng(seed)
    schema = AttributeSchema.for_category(category)
    n_shared = schema.n_shared

    user_ids = [f"user_{i:04d}" for i in range(n_users)]
    theta = rng.normal(0.0, linear_strength, size=(n_users, n_shared))
    all_pairs = [(a, b) for a in range(n_shared) for b in range(a + 1, n_shared)]
    pair_idx = rng.choice(len(all_pairs), size=min(n_interactions, len(all_pairs)), replace=False)
    phi_pairs = [all_pairs[i] for i in pair_idx]
    phi = rng.normal(0.0, interaction_strength, size=(n_users, len(phi_pairs)))

    # ---------------------------------------------------------------- products
    attrs = rng.random(size=(n_products, schema.dim))
    products: list[IntegratedProduct] = []
    for j in range(n_products):
        # Planted quality is independent of attributes: a wide per-product base
        # plus per-(profile,dim) variation, so quality genuinely discriminates
        # products and is not recoverable from the spec sheet.
        base = rng.uniform(0.2, 0.85)
        q: dict[tuple[str, str], float] = {}
        for prof in use_profiles:
            prof_off = rng.normal(0.0, 0.12)
            for dim in QUALITY_DIMS:
                q[(prof, dim)] = float(np.clip(base + prof_off + rng.normal(0.0, 0.07), 0.02, 0.98))
        products.append(IntegratedProduct(f"{category}_prod_{j:04d}", attrs[j], q))

    prod_ids = [p.product_id for p in products]
    X = np.stack([p.attributes for p in products])
    x_shared = X[:, :n_shared]

    def mean_quality(prod: IntegratedProduct, profile: str) -> float:
        return float(np.mean([prod.quality[(profile, d)] for d in QUALITY_DIMS]))

    # Per-product mean quality, by profile, reused for utility and traps.
    qual_by_profile = {
        prof: np.array([mean_quality(p, prof) for p in products]) for prof in use_profiles
    }

    def pref_utility(ui: int) -> np.ndarray:
        u = x_shared @ theta[ui]
        for k, (a, b) in enumerate(phi_pairs):
            u += phi[ui, k] * x_shared[:, a] * x_shared[:, b]
        return u

    # --------------------------------------------------------------- the users
    cohorts = (_NEW_COHORT, *_COHORTS) if include_new_cohort else _COHORTS
    users: list[IntegratedUser] = []
    for ui, uid in enumerate(user_ids):
        cohort_name, lo, hi = cohorts[ui % len(cohorts)]
        history_len = int(rng.integers(lo, hi + 1))
        profile = use_profiles[ui % len(use_profiles)]

        pref_u = pref_utility(ui)
        qual_u = qual_by_profile[profile]
        # Standardize the two terms before combining so quality_weight is a clean
        # variance ratio rather than an artifact of raw scales.
        pref_z = (pref_u - pref_u.mean()) / (pref_u.std() + 1e-9)
        qual_z = (qual_u - qual_u.mean()) / (qual_u.std() + 1e-9)
        true_util = pref_z + quality_weight * qual_z
        full = true_util + rng.normal(0.0, noise, size=n_products)

        order = np.argsort(-full)
        relevant = [prod_ids[i] for i in order[:n_relevant]]
        rel_set = set(relevant)

        # Observed purchases are *taste-driven*: a shopper picks on the attributes
        # they can see at purchase time, with only a small pull from quality
        # (``purchase_quality_weight``) since real-world reliability is largely
        # discovered afterward — exactly the gap the QIL fills. So a rich history
        # yields a clean, strong preference fit, while a sparse history is weak
        # purely from lack of data (it collapses toward the population prior). The
        # quality dimension of true utility is therefore *not* learnable from the
        # user's own purchases; it has to come from the community-derived QIL.
        purchase_util = pref_z + purchase_quality_weight * qual_z
        purchase_util = purchase_util + rng.normal(0.0, noise, size=n_products)
        purchases = [prod_ids[i] for i in _softmax_sample(rng, purchase_util, history_len, temp=0.5)]

        # Two kinds of hard negative make *both* signals necessary:
        #  - preference traps: high preference, low true utility (quality drags
        #    them down) -> a preference-only ranker is fooled;
        #  - quality traps: high quality, low true utility (taste drags them
        #    down) -> a quality-only ranker is fooled.
        pref_order = [prod_ids[i] for i in np.argsort(-pref_z) if prod_ids[i] not in rel_set]
        qual_order = [prod_ids[i] for i in np.argsort(-qual_z) if prod_ids[i] not in rel_set]
        pref_traps = pref_order[:n_pref_traps]
        trap_set = set(pref_traps)
        quality_traps = [p for p in qual_order if p not in trap_set][:n_quality_traps]
        trap_set |= set(quality_traps)

        fill_pool = [p for p in prod_ids if p not in rel_set and p not in trap_set]
        n_fill = max(0, n_candidates - len(relevant) - len(pref_traps) - len(quality_traps))
        fill = list(rng.choice(fill_pool, size=min(n_fill, len(fill_pool)), replace=False))
        candidates = relevant + pref_traps + quality_traps + fill
        rng.shuffle(candidates)

        # Credential confidence == trust in the user's own fit over the prior,
        # mirroring the preference graph's cold-start blend weight.
        mean_confidence = history_len / (history_len + cold_start_pivot)

        users.append(IntegratedUser(
            user_id=uid, use_profile=profile, history_len=history_len, cohort=cohort_name,
            purchases=purchases, relevant=relevant, candidates=candidates,
            mean_confidence=float(mean_confidence),
        ))

    # ------------------------------------------------- QIL evidence -> signals
    # Emit per-(product, use_profile, dim) performance observations scattered
    # around the planted mean. These feed a real QualityAggregator, so the agent
    # queries genuine posteriors (the QIL extraction NLP step, validated in
    # Phase 0, is short-circuited here — we inject ExtractedSignal directly, the
    # same path the QIL query tests use).
    non_uniform = evidence_lo is not None and evidence_hi is not None
    signals: list[ExtractedSignal] = []
    for prod in products:
        # Evidence per product: uniform by default, or drawn per product when a
        # range is given (independent of the product's true quality).
        n_cell = int(rng.integers(evidence_lo, evidence_hi + 1)) if non_uniform else signals_per_cell
        for prof in use_profiles:
            for dim in QUALITY_DIMS:
                mean = prod.quality[(prof, dim)]
                for _ in range(n_cell):
                    val = float(np.clip(rng.normal(mean, signal_obs_noise), 0.0, 1.0))
                    signals.append(ExtractedSignal(
                        product_id=prod.product_id, category=category, use_profile=prof,
                        signal_type="performance", failure_mode=None, quality_dim=dim,
                        signal_value=val, confidence=signal_confidence,
                    ))

    return IntegratedScenario(
        schema=schema, products=products, users=users, signals=signals,
        use_profiles=use_profiles, quality_weight=quality_weight,
        theta=theta, phi_pairs=phi_pairs, phi=phi,
    )
