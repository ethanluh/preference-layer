"""Product-id normalization for QIL ingestion.

The same laptop is called "ThinkPad X1 Carbon Gen 12", "X1 Carbon (Gen 12)",
"thinkpad-x1c-g12", ... across Reddit, iFixit and Notebookcheck. The ingestion
``product_signal`` schema (``schema.sql``) needs a single canonical
``product_id`` so signals about one product aggregate together, plus a
``model_normalized`` string used to *match* free-text mentions to that canonical
id.

This module is dependency-free: a deterministic slug normalizer plus a small
token-overlap fuzzy matcher against a canonical model registry. A production
pipeline can swap in a stronger entity-resolution model behind the same
``ProductRegistry`` interface; nothing downstream depends on the matching
internals.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Marketing words and separators that carry no disambiguating signal.
_STOP_TOKENS = frozenset({
    "the", "and", "with", "for", "gen", "generation", "edition", "laptop",
    "notebook", "keyboard", "model", "series", "inch", "in",
})
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_model_string(raw: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace -> match string."""
    text = unicodedata.normalize("NFKD", raw)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    tokens = _TOKEN_RE.findall(text)
    return " ".join(tokens)


def _signature(raw: str) -> frozenset[str]:
    """Content tokens (stopwords removed) used for fuzzy overlap matching."""
    return frozenset(t for t in normalize_model_string(raw).split() if t not in _STOP_TOKENS)


@dataclass(frozen=True)
class CanonicalProduct:
    product_id: str
    category: str
    display_name: str
    # Extra surface forms (model numbers, nicknames) that should match this product.
    aliases: tuple[str, ...] = ()

    def signatures(self) -> list[frozenset[str]]:
        return [_signature(self.display_name)] + [_signature(a) for a in self.aliases]


@dataclass
class ProductRegistry:
    """Canonical model list with a token-overlap fuzzy matcher.

    ``match`` returns ``(CanonicalProduct, score)`` for the best candidate whose
    Jaccard token overlap clears ``threshold``, else ``None``. Deterministic and
    offline so ingestion is testable without a network entity-resolution service.
    """

    products: list[CanonicalProduct] = field(default_factory=list)
    threshold: float = 0.5

    def add(self, product: CanonicalProduct) -> "ProductRegistry":
        self.products.append(product)
        return self

    def match(self, mention: str, category: str | None = None) -> tuple[CanonicalProduct, float] | None:
        """Best canonical product whose model name is *contained* in the mention.

        Uses a containment score -- the fraction of the canonical model's tokens
        present in the mention -- rather than symmetric Jaccard, because a free-
        text post is far longer than a model name and symmetric overlap would be
        diluted to near-zero by the post's other words.

        A short (2-token) signature also requires *both* tokens present, not just
        the ratio -- otherwise a 2-token alias like "xps 15" clears the 0.5
        threshold on the generic token "15" alone (e.g. any post mentioning a
        15" screen size), with zero evidence of the distinguishing token ("xps").
        Signatures of 3+ tokens are already protected by the ratio (matching a
        bare minority of tokens can't clear 0.5), so this only tightens the
        specific case a ratio-only check leaves open.
        """
        mention_sig = _signature(mention)
        if not mention_sig:
            return None
        best: tuple[CanonicalProduct, float] | None = None
        for prod in self.products:
            if category is not None and prod.category != category:
                continue
            for sig in prod.signatures():
                overlap = len(sig & mention_sig)
                if not sig or overlap < min(2, len(sig)):
                    continue
                score = overlap / len(sig)
                if score >= self.threshold and (best is None or score > best[1]):
                    best = (prod, score)
        return best
