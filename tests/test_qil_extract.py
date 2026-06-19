"""Tests for QIL extraction, including the Phase 0 Claim 2 go/no-go gate."""

from preferencelayer.qil import QILExtractor, corpus as C, evaluate
from preferencelayer.qil.eval import GATE_PRECISION


def _trained(seed=17, n_train=1400, n_test=400):
    cp = C.generate(n_train=n_train, n_test=n_test, seed=seed)
    ex = QILExtractor().fit(cp.train)
    report = evaluate([s.use_profile for s in cp.test], ex.predict_use_profiles(cp.test))
    return cp, ex, report


def test_phase0_qil_gate():
    """Claim 2: use-profile precision >= 70% on the held-out set."""
    _, _, report = _trained()
    assert report.macro_precision >= GATE_PRECISION, report.macro_precision
    # And the classifier must clearly beat the most-frequent-class baseline.
    assert report.macro_precision > report.baseline_precision + 0.2


def test_gate_robust_across_seeds():
    for seed in (1, 7, 42):
        _, _, report = _trained(seed=seed, n_train=900, n_test=300)
        assert report.macro_precision >= GATE_PRECISION, (seed, report.macro_precision)


def test_extract_produces_structured_signals():
    cp, ex, _ = _trained(n_train=300, n_test=100)
    signals = ex.extract(cp.test)
    assert len(signals) == len(cp.test)
    s = signals[0]
    assert 0.0 <= s.confidence <= 1.0
    assert 0.0 <= s.signal_value <= 1.0
    # Failure-typed signals carry a failure mode; others carry a quality dim.
    for sig in signals:
        if sig.signal_type == "failure":
            assert sig.quality_dim is not None or sig.failure_mode is not None


def test_tfidf_rows_unit_normalized():
    cp = C.generate(n_train=100, n_test=20, seed=1)
    ex = QILExtractor().fit(cp.train)
    X = ex.vectorizer.transform([s.text for s in cp.test])
    norms = (X ** 2).sum(axis=1) ** 0.5
    assert ((norms > 0.99) & (norms < 1.01)).all()
