"""Controlled, labeled corpus for the QIL extraction feasibility study.

Why a synthetic corpus? Claim 2 of the implementation plan is that
use-profile-conditioned quality signals are *extractable* from public unstructured
text at >= 70% precision. The production path (``docs/implementation-plan.md``
Work Stream B) collects ~2,000 Reddit posts/category and has two annotators label
300 of them. That pipeline is not reproducible offline and not appropriate for
CI. So, exactly as ``data/synthetic.py`` does for Claim 1, this module generates a
controlled corpus with **known ground truth** and **deliberately injected
ambiguity**, so the measured precision is an *earned*, falsifiable number rather
than an artifact of templated text.

Anti-rigging design
-------------------
A classifier could trivially hit 100% precision on cleanly templated text. To make
the result meaningful, the generator:

* draws use-profile cue words from per-profile lexicons that **overlap** (gaming
  and heavy-use both say "hours"/"load"; travel and light-use both say
  "comfortable") — genuine confusability;
* mixes in a large pool of **shared filler** words carrying no profile signal;
* with probability ``ambiguity_frac`` emits a *weak-signal* post dominated by
  filler plus a single, possibly cross-profile, cue — these cap achievable
  precision below 1.0;
* overlays **failure / performance phrasing** that is irrelevant to the
  use-profile label and acts as distractor vocabulary.

The gold labels are always the true generative profile; difficulty comes from the
observable text, not from corrupting the labels. Tune the knobs and precision
moves smoothly — there is no hard-coded "the classifier wins".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import CATEGORIES, FAILURE_MODES, QUALITY_DIMS, SIGNAL_TYPES, USE_PROFILES

# Per-use-profile signature lexicons. Note the intentional overlaps across
# profiles (e.g. "hours", "long", "fast", "comfortable") that create real
# confusability — the classifier must weigh co-occurring evidence, not key off a
# single token.
_PROFILE_LEXICON: dict[str, tuple[str, ...]] = {
    "light_use": ("browsing", "email", "casual", "occasional", "notes", "everyday",
                  "students", "streaming", "comfortable", "simple", "basic"),
    "heavy_use": ("rendering", "compile", "sustained", "multitasking", "intensive",
                  "workstation", "batch", "hours", "load", "long", "demanding"),
    "gaming":    ("gaming", "fps", "frames", "gpu", "esports", "competitive",
                  "refresh", "hours", "load", "fast", "rig"),
    "professional": ("work", "office", "productivity", "spreadsheets", "client",
                     "meetings", "deadlines", "coding", "developer", "fast"),
    "travel":    ("travel", "commute", "airport", "backpack", "lightweight",
                  "portable", "flights", "cafe", "comfortable", "long"),
}

# Filler words with no use-profile signal, sprinkled into every post.
_FILLER = ("the", "this", "device", "machine", "really", "pretty", "honestly",
           "bought", "month", "year", "setup", "experience", "daily", "use",
           "overall", "would", "recommend", "after", "still", "though")

# Failure-mode phrasing (distractor vocabulary for the use-profile task; the
# actual label source for failure-mode extraction).
_FAILURE_LEXICON: dict[str, tuple[str, ...]] = {
    "thermal_throttling": ("throttling", "overheats", "hot", "fans", "thermal", "temps"),
    "battery_degradation": ("battery", "drains", "charge", "degraded", "swollen"),
    "structural_failure": ("hinge", "cracked", "broke", "flex", "snapped", "loose"),
    "connectivity_issue": ("wifi", "drops", "bluetooth", "disconnects", "unstable"),
    "switch_failure": ("chatter", "double", "sticky", "stabilizer", "rattle", "switch"),
    "display_defect": ("pixel", "backlight", "bleed", "flickering", "ghosting"),
}

_PERFORMANCE_PHRASES = ("handles", "smooth", "solid", "great", "flawless", "reliable")
_COMPARISON_PHRASES = ("better", "compared", "versus", "worse", "prefer", "beats")

# Failure modes that only apply to keyboards / laptops respectively.
_CATEGORY_FAILURES: dict[str, tuple[str, ...]] = {
    "keyboards": ("switch_failure", "structural_failure", "connectivity_issue"),
    "laptops": ("thermal_throttling", "battery_degradation", "structural_failure",
                "connectivity_issue", "display_defect"),
}


@dataclass
class Sample:
    """One labeled unstructured-text signal (a stand-in for a Reddit post)."""

    text: str
    category: str
    product_id: str
    use_profile: str        # gold label for the Phase 0 gate
    signal_type: str
    failure_mode: str | None
    quality_dim: str | None
    signal_value: float     # normalized quality observation in [0, 1]
    label_confidence: float


@dataclass
class Corpus:
    train: list[Sample]
    test: list[Sample]
    products: dict[str, list[str]]            # category -> product ids
    # Planted ground-truth quality means per (product, use_profile, dim), for
    # diagnostics and to give aggregation a recoverable target.
    quality_means: dict[tuple[str, str, str], float]


def _tokens_for_profile(rng: np.random.Generator, profile: str, ambiguity_frac: float,
                        overlap_frac: float) -> list[str]:
    """Emit profile cue tokens, weak or cross-profile a controlled fraction of the time."""
    weak = rng.random() < ambiguity_frac
    n_cue = 1 if weak else int(rng.integers(2, 5))
    toks: list[str] = []
    for _ in range(n_cue):
        src = profile
        if rng.random() < overlap_frac:
            src = USE_PROFILES[int(rng.integers(0, len(USE_PROFILES)))]
        lex = _PROFILE_LEXICON[src]
        toks.append(lex[int(rng.integers(0, len(lex)))])
    return toks


def generate(
    n_train: int = 1400,
    n_test: int = 400,
    products_per_category: int = 8,
    ambiguity_frac: float = 0.18,
    overlap_frac: float = 0.15,
    profile_skew: float = 0.2,
    seed: int = 17,
) -> Corpus:
    """Generate a labeled QIL extraction corpus.

    ``ambiguity_frac`` / ``overlap_frac`` control how hard the use-profile task is;
    ``profile_skew`` (>0) makes the class distribution non-uniform so the
    most-frequent-class baseline is a meaningful (but beatable) reference.
    """
    rng = np.random.default_rng(seed)

    products = {
        c: [f"{c[:-1]}_model_{i:02d}" for i in range(products_per_category)]
        for c in CATEGORIES
    }

    # Non-uniform class prior (Dirichlet-ish via softmax of skewed logits).
    logits = rng.normal(0.0, profile_skew, size=len(USE_PROFILES))
    class_p = np.exp(logits - logits.max())
    class_p /= class_p.sum()

    # Planted quality means per (product, use_profile, dim): some products are
    # genuinely better for some use profiles than others.
    quality_means: dict[tuple[str, str, str], float] = {}
    for c in CATEGORIES:
        for pid in products[c]:
            base = rng.uniform(0.35, 0.75)
            for prof in USE_PROFILES:
                prof_off = rng.normal(0.0, 0.12)
                for dim in QUALITY_DIMS:
                    m = float(np.clip(base + prof_off + rng.normal(0.0, 0.08), 0.02, 0.98))
                    quality_means[(pid, prof, dim)] = m

    def make_sample() -> Sample:
        category = CATEGORIES[int(rng.integers(0, len(CATEGORIES)))]
        pid = products[category][int(rng.integers(0, len(products[category])))]
        profile = USE_PROFILES[int(rng.choice(len(USE_PROFILES), p=class_p))]
        signal_type = SIGNAL_TYPES[int(rng.integers(0, len(SIGNAL_TYPES)))]

        words = _tokens_for_profile(rng, profile, ambiguity_frac, overlap_frac)

        failure_mode: str | None = None
        quality_dim: str | None = None
        if signal_type == "failure":
            fm_choices = _CATEGORY_FAILURES[category]
            failure_mode = fm_choices[int(rng.integers(0, len(fm_choices)))]
            words += list(rng.choice(_FAILURE_LEXICON[failure_mode],
                                     size=int(rng.integers(2, 4)), replace=True))
            signal_value = float(np.clip(rng.normal(0.25, 0.12), 0.0, 1.0))
        else:
            quality_dim = QUALITY_DIMS[int(rng.integers(0, len(QUALITY_DIMS)))]
            phrases = _PERFORMANCE_PHRASES if signal_type == "performance" else _COMPARISON_PHRASES
            words += list(rng.choice(phrases, size=int(rng.integers(1, 3)), replace=True))
            mean = quality_means[(pid, profile, quality_dim)]
            signal_value = float(np.clip(rng.normal(mean, 0.10), 0.0, 1.0))

        # Filler dominates weak-signal posts, keeping precision earned.
        words += list(rng.choice(_FILLER, size=int(rng.integers(4, 9)), replace=True))
        rng.shuffle(words)

        # Annotator confidence: lower when the post is short / cue-poor.
        cue_count = sum(1 for w in words if any(w in lex for lex in _PROFILE_LEXICON.values()))
        label_confidence = float(np.clip(0.5 + 0.12 * cue_count, 0.5, 0.98))

        return Sample(
            text=" ".join(words),
            category=category,
            product_id=pid,
            use_profile=profile,
            signal_type=signal_type,
            failure_mode=failure_mode,
            quality_dim=quality_dim,
            signal_value=signal_value,
            label_confidence=label_confidence,
        )

    samples = [make_sample() for _ in range(n_train + n_test)]
    return Corpus(
        train=samples[:n_train],
        test=samples[n_train:],
        products=products,
        quality_means=quality_means,
    )
