"""Tests for the controlled QIL corpus generator."""

from preferencelayer.qil import corpus as C
from preferencelayer.qil.schema import CATEGORIES, USE_PROFILES


def test_deterministic():
    a = C.generate(n_train=200, n_test=80, seed=3)
    b = C.generate(n_train=200, n_test=80, seed=3)
    assert [s.text for s in a.train] == [s.text for s in b.train]
    assert [s.use_profile for s in a.test] == [s.use_profile for s in b.test]


def test_split_sizes_and_fields():
    cp = C.generate(n_train=300, n_test=120, seed=5)
    assert len(cp.train) == 300 and len(cp.test) == 120
    for s in cp.train[:20]:
        assert s.use_profile in USE_PROFILES
        assert s.category in CATEGORIES
        assert 0.0 <= s.signal_value <= 1.0
        assert s.text  # non-empty


def test_all_profiles_present():
    cp = C.generate(n_train=600, n_test=200, seed=9)
    seen = {s.use_profile for s in cp.train}
    assert seen == set(USE_PROFILES)


def test_failure_samples_carry_failure_mode():
    cp = C.generate(n_train=400, n_test=100, seed=2)
    for s in cp.train:
        if s.signal_type == "failure":
            assert s.failure_mode is not None
        else:
            assert s.quality_dim is not None
