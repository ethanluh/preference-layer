"""Heuristic quality-dimension span tagger (QIL ingestion).

The QIL extractor (``qil/extract.py``) learns ``use_profile`` and ``signal_type``
but does **not** extract ``quality_dim`` -- it passes through whatever was on the
input ``Sample``. In a real ingest the pipeline has no gold ``quality_dim`` (it is
*predicting* it), so ``_doc_to_sample`` defaults it to ``None``. The aggregator
then skips every ``None``-dim signal (``aggregate.py``: "if s.quality_dim is None:
continue"), so a real ingest writes **zero** GP quality posteriors.

This module closes that gap with a lexicon/rule tagger: a dependency-free stand-in
for the eventual B2 span model. It maps a post to the single **dominant** quality
dimension it discusses (plus a sentiment-derived ``signal_value``), so the
ingest -> refit path produces real GP posteriors end to end.

Why one dimension per document: the pipeline is 1 document -> 1 ``Sample`` ->
1 ``ExtractedSignal`` -> 1 row, deduplicated on ``content_hash``. Emitting one
signal per detected dimension would produce multiple rows sharing a
``content_hash`` and collide on dedup. Picking the dominant dimension keeps the
1:1 shape; the span model that replaces this can revisit multi-span emission
alongside a dedup-key change.

This is a heuristic stand-in, not the B2 deliverable: it is not measured against
an annotated corpus and makes no precision claim. It exists so the aggregation
story is genuinely end-to-end in-sandbox; swap in the trained span model behind
the same ``QualityDimTagger.tag`` interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema import QUALITY_DIMS

_TOKEN_RE = re.compile(r"[a-z]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# Keyword cues per quality dimension. Keyed by the five ``schema.QUALITY_DIMS``;
# mirrors the lexicon style in ``corpus.py`` (which generates text) but is used
# here to *read* it. Deliberately high-precision cues -- a real span model will
# do better, this only needs to route the obvious cases so GP posteriors form.
QUALITY_DIM_LEXICON: dict[str, tuple[str, ...]] = {
    "thermal": (
        "thermal", "throttling", "throttle", "overheats", "overheating", "hot",
        "heat", "temps", "temperature", "fans", "fan", "cooling", "warm",
    ),
    "build_quality": (
        "build", "chassis", "hinge", "flex", "creak", "sturdy", "solid", "premium",
        "plastic", "cheap", "rigid", "wobble", "rattle", "construction",
    ),
    "battery_longevity": (
        "battery", "charge", "charging", "drain", "drains", "longevity", "cycles",
        "endurance", "lasts", "runtime", "unplugged",
    ),
    "display": (
        "display", "screen", "panel", "brightness", "nits", "color", "colour",
        "ips", "oled", "bezel", "backlight", "glare", "contrast",
    ),
    "ergonomics": (
        "ergonomics", "ergonomic", "keyboard", "typing", "wrist", "layout",
        "trackpad", "keys", "keycaps", "comfortable", "comfort", "feel", "actuation",
    ),
}

# Sentiment cues for the [0,1] ``signal_value`` (the GP observation). A real
# pipeline parses a graded value with the span model; this gives a non-degenerate
# observation instead of the prior-default 0.5 so posteriors actually move.
_POSITIVE = frozenset((
    "great", "good", "excellent", "solid", "smooth", "reliable", "flawless",
    "love", "amazing", "fast", "comfortable", "sturdy", "premium", "best", "crisp",
))
_NEGATIVE = frozenset((
    "bad", "poor", "terrible", "awful", "throttling", "overheats", "drains",
    "cheap", "flex", "creak", "wobble", "rattle", "disappointing", "worst",
    "broke", "cracked", "dim", "glare",
))

# signal_value for a fully-positive vs fully-negative post; neutral is the prior.
_NEUTRAL_VALUE = 0.5
_POS_VALUE = 0.85
_NEG_VALUE = 0.15


@dataclass
class QualityDimTagger:
    """Lexicon tagger: post text -> (dominant quality_dim, signal_value).

    ``tag`` returns ``(None, 0.5)`` when no dimension cue is present, so callers
    can leave ``quality_dim`` unset (and the aggregator skips it) exactly as today.
    """

    lexicon: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {k: v for k, v in QUALITY_DIM_LEXICON.items()}
    )

    def tag(self, text: str) -> tuple[str | None, float]:
        toks = _tokens(text)
        if not toks:
            return None, _NEUTRAL_VALUE
        token_set = set(toks)

        # Dominant dimension = most cue hits; tie-break by QUALITY_DIMS order so
        # the result is deterministic and independent of dict insertion order.
        best_dim: str | None = None
        best_hits = 0
        for dim in QUALITY_DIMS:
            cues = self.lexicon.get(dim, ())
            hits = sum(1 for c in cues if c in token_set)
            if hits > best_hits:
                best_dim, best_hits = dim, hits

        if best_dim is None:
            return None, _NEUTRAL_VALUE
        return best_dim, self._signal_value(toks)

    @staticmethod
    def _signal_value(toks: list[str]) -> float:
        pos = sum(1 for t in toks if t in _POSITIVE)
        neg = sum(1 for t in toks if t in _NEGATIVE)
        if pos == neg:
            return _NEUTRAL_VALUE
        # Linear blend toward the dominant polarity, scaled by its margin share.
        total = pos + neg
        if pos > neg:
            return round(_NEUTRAL_VALUE + (_POS_VALUE - _NEUTRAL_VALUE) * (pos - neg) / total, 4)
        return round(_NEUTRAL_VALUE - (_NEUTRAL_VALUE - _NEG_VALUE) * (neg - pos) / total, 4)
