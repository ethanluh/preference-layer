"""Unit tests for the heuristic quality-dimension span tagger."""

from preferencelayer.qil.quality_spans import QUALITY_DIM_LEXICON, QualityDimTagger
from preferencelayer.qil.schema import QUALITY_DIMS


def test_lexicon_covers_every_schema_dimension():
    # The tagger can only ever emit dims it has cues for; keep it aligned to schema.
    assert set(QUALITY_DIM_LEXICON) == set(QUALITY_DIMS)


def test_detects_each_dimension_from_representative_text():
    tagger = QualityDimTagger()
    cases = {
        "thermal": "It overheats and the fans scream, constant thermal throttling.",
        "build_quality": "The chassis has noticeable flex and the hinge feels cheap.",
        "battery_longevity": "Battery drains fast; charge barely lasts a few hours unplugged.",
        "display": "The OLED screen is gorgeous, high brightness and great color.",
        "ergonomics": "Typing on this keyboard is comfortable, the key layout feels great.",
    }
    for expected_dim, text in cases.items():
        dim, _value = tagger.tag(text)
        assert dim == expected_dim, f"{text!r} -> {dim}, expected {expected_dim}"


def test_no_cue_returns_none_and_neutral_value():
    dim, value = QualityDimTagger().tag("I bought this last month and it arrived on time.")
    assert dim is None
    assert value == 0.5


def test_empty_text_is_neutral():
    assert QualityDimTagger().tag("") == (None, 0.5)


def test_dominant_dimension_wins_on_more_hits():
    # Two display cues vs one thermal cue -> display dominates.
    dim, _ = QualityDimTagger().tag("the screen brightness is great though it runs a little hot")
    assert dim == "display"


def test_tie_breaks_by_schema_order():
    # One thermal cue ("hot") and one ergonomics cue ("comfortable"): equal hits,
    # so the QUALITY_DIMS order decides -> thermal precedes ergonomics.
    dim, _ = QualityDimTagger().tag("runs hot but the grip is comfortable")
    assert dim == "thermal"


def test_signal_value_reflects_sentiment():
    tagger = QualityDimTagger()
    _, pos = tagger.tag("the display is great, excellent crisp color, best screen")
    _, neg = tagger.tag("the display is terrible, awful dim screen, worst panel")
    assert pos > 0.5
    assert neg < 0.5
