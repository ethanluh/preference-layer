"""Attribute schemas: the shared preference vocabulary across product categories.

The central architectural bet of PreferenceLayer is that user preferences live in
an *attribute* space, not an *item-embedding* space. Attributes such as
``price_sensitivity`` or ``durability`` carry meaning across product categories: a
user who values durability in a laptop tends to value it in headphones too. Flat
per-platform item embeddings cannot express that, because the item-embedding space
of one category does not align with another's. Attributes do.

This module defines:

* A small set of *shared* attributes that are meaningful in every category.
* A few *category-specific* attributes that only apply locally.

Cross-category transfer is only possible through the shared attributes. The graph
model exploits them; a flat item-embedding baseline cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Attributes whose semantics are stable across every product category. These are
# the channel through which preference signal transfers between categories.
SHARED_ATTRIBUTES: tuple[str, ...] = (
    "price_sensitivity",   # preference for lower price / value
    "build_quality",       # premium materials, solid construction
    "durability",          # longevity, resistance to failure
    "portability",         # light, compact, travel-friendly
    "performance",         # raw capability / power
    "brand_affinity",      # preference for established premium brands
    "aesthetics",          # design / looks
    "ergonomics",          # comfort during sustained use
)

# Attributes that only make sense within a single category.
CATEGORY_SPECIFIC_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "laptops": ("battery_life", "thermal_performance", "keyboard_quality", "display_quality"),
    "headphones": ("noise_cancellation", "bass_response", "soundstage", "wireless_range"),
    "keyboards": ("switch_feel", "key_travel", "rgb_lighting", "layout_compactness"),
    "monitors": ("refresh_rate", "color_accuracy", "panel_uniformity", "stand_adjustability"),
}


@dataclass(frozen=True)
class AttributeSchema:
    """Ordered attribute vocabulary for a single category.

    The vector layout is ``[shared..., category_specific...]``. The shared block
    occupies the same indices in every category, which is what lets a preference
    vector or graph trained on one category be applied to another.
    """

    category: str
    shared: tuple[str, ...] = SHARED_ATTRIBUTES
    specific: tuple[str, ...] = ()

    @classmethod
    def for_category(cls, category: str) -> "AttributeSchema":
        return cls(
            category=category,
            shared=SHARED_ATTRIBUTES,
            specific=CATEGORY_SPECIFIC_ATTRIBUTES.get(category, ()),
        )

    @property
    def names(self) -> tuple[str, ...]:
        return self.shared + self.specific

    @property
    def dim(self) -> int:
        return len(self.names)

    @property
    def n_shared(self) -> int:
        return len(self.shared)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def shared_indices(self) -> list[int]:
        """Indices of the shared block (always ``range(n_shared)`` by construction)."""
        return list(range(self.n_shared))


def shared_attribute_count() -> int:
    return len(SHARED_ATTRIBUTES)
