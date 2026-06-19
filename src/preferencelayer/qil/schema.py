"""QIL vocabularies: use profiles, failure modes, and quality dimensions.

The Quality Intelligence Layer (QIL) answers a different question from the
preference graph. The graph models *what a user wants*; the QIL models *how a
product actually performs for a given kind of use*. Both are conditioned on a
profile, but here the profile is a **use profile** (how the product is used),
not a taste profile.

This module pins down the controlled vocabulary the Phase 0 feasibility study
classifies into and aggregates over. It is intentionally small and matches the
schema sketched in ``docs/architecture.md`` (``failure_mode``, ``quality_dim``,
``use_profile``, ``signal_type``).
"""

from __future__ import annotations

# How a product is used. This is the field the Phase 0 go/no-go gate is about:
# "use-profile-conditioned quality signals extractable at >= 70% precision".
USE_PROFILES: tuple[str, ...] = (
    "light_use",     # browsing, email, occasional casual use
    "heavy_use",     # sustained compute, rendering, all-day multitasking
    "gaming",        # high-framerate gaming, competitive play
    "professional",  # office / developer / enterprise productivity
    "travel",        # commuting, flights, on-the-go battery-bound use
)

# Discrete failure modes a signal may report.
FAILURE_MODES: tuple[str, ...] = (
    "thermal_throttling",
    "battery_degradation",
    "structural_failure",
    "connectivity_issue",
    "switch_failure",     # keyboards
    "display_defect",
)

# Continuous quality dimensions aggregated with a conjugate Bayesian posterior.
QUALITY_DIMS: tuple[str, ...] = (
    "thermal",
    "build_quality",
    "battery_longevity",
    "display",
    "ergonomics",
)

# What kind of statement a signal is. Only 'failure' counts toward the
# Beta-Binomial failure rate; 'performance'/'comparison' carry continuous
# quality-dimension observations.
SIGNAL_TYPES: tuple[str, ...] = ("failure", "performance", "comparison")

# Categories with rich community discourse, per the implementation plan.
CATEGORIES: tuple[str, ...] = ("laptops", "keyboards")


def use_profile_index(profile: str) -> int:
    return USE_PROFILES.index(profile)
