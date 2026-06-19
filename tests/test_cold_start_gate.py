"""Regression test for the zero-history cold-start finding.

Robust facts from ``docs/phase1-cold-start-results.md``:

1. Zero-history users are a first-class cohort: no purchases, credential confidence 0,
   and the preference fit returns the population prior without crashing.
2. The adaptive mechanism's *premise* holds there — quality alone beats preference
   alone (the only cohort where it does), and the adaptive blend (which leans on
   quality at confidence 0) beats preference-only.

We assert these (stable at this sample size) and leave the adaptive-vs-fixed magnitude
and the noisy per-cohort optimal-α to the experiment/report.
"""

import numpy as np

from preferencelayer.agent import IntegrationHarness
from preferencelayer.agent._harness import prepare_preference_model, purchase_matrix
from preferencelayer.agent.combine import alpha_from_confidence
from preferencelayer.data import integrated


def test_new_cohort_is_zero_history():
    s = integrated.generate(n_users=120, seed=23, include_new_cohort=True)
    new = [u for u in s.users if u.cohort == "new"]
    assert new, "include_new_cohort should create a 'new' cohort"
    assert all(u.history_len == 0 for u in new)
    assert all(u.mean_confidence == 0.0 for u in new)
    assert all(u.purchases == [] for u in new)


def test_default_scenario_has_no_new_cohort():
    s = integrated.generate(n_users=60, seed=23)   # default: include_new_cohort=False
    assert not any(u.cohort == "new" for u in s.users)


def test_fit_on_empty_history_returns_prior():
    s = integrated.generate(n_users=60, seed=23, include_new_cohort=True)
    model, idx, catalog, n_shared = prepare_preference_model(s)
    empty = purchase_matrix(idx, [], s.schema.dim)
    assert empty.shape == (0, s.schema.dim)
    state = model.fit(empty, catalog, n_shared)        # must not raise (no per-user fit)
    assert np.allclose(state["w"], model.prior_w)


def test_zero_history_leans_on_quality():
    s = integrated.generate(n_users=200, seed=23, include_new_cohort=True)
    rep = IntegrationHarness(s, k=10, seed=13).run()
    new = next(c for c in rep.cohorts if c.cohort == "new")
    b = new.by_condition
    # The documented premise: for a brand-new user, quality alone beats preference alone.
    assert b["quality_only"] > b["preference_only"]
    # The adaptive blend leans on quality (α = sigmoid(3*(0-0.5)) ≈ 0.18 < 0.5) and so
    # ranks better than preference alone for these users.
    assert alpha_from_confidence(new.mean_confidence) < 0.5
    assert b["adaptive_alpha"] >= b["preference_only"]
