"""Bayesian aggregation of extracted signals into quality posteriors.

Implements the aggregation layer from ``docs/architecture.md``:

* **Failure rate** per ``(product_id, use_profile)``: hierarchical Beta-Binomial.
  A category-level base rate sets the prior ``Beta(a0, b0)``; each extracted
  ``failure`` signal increments failures, each non-failure increments
  non-failures. Posterior mean ``= a / (a + b)``.
* **Quality dimensions** per ``(product_id, use_profile, quality_dim)``: the doc
  specifies a Gaussian process over release time. Phase 0 uses the conjugate
  **Normal-Normal** special case (no temporal kernel) as an honest, dependency-
  free stand-in — it yields the same posterior-mean + credible-interval contract
  the ``/quality`` API needs. The GP upgrade is Phase 1 work.

Each observation is weighted by its extraction ``confidence`` so low-confidence
signals move the posterior less.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .extract import ExtractedSignal

# Z for a 90% credible interval (matches the architecture's "90% credible interval").
_Z90 = 1.6448536269514722


@dataclass
class QualityPosterior:
    product_id: str
    use_profile: str
    quality_dim: str
    posterior_mean: float
    posterior_std: float
    credible_lo_90: float
    credible_hi_90: float
    evidence_count: int


@dataclass
class FailureRatePosterior:
    product_id: str
    use_profile: str
    alpha: float
    beta: float
    rate_mean: float
    evidence_count: int


class QualityAggregator:
    """Aggregates extracted signals into per-(product, use_profile, dim) posteriors."""

    def __init__(
        self,
        prior_strength: float = 4.0,     # pseudo-observations for the quality-dim prior
        prior_mean: float = 0.5,         # neutral quality prior
        obs_std: float = 0.15,           # per-observation noise std
        failure_prior_count: float = 2.0,
    ):
        self.prior_strength = prior_strength
        self.prior_mean = prior_mean
        self.obs_std = obs_std
        self.failure_prior_count = failure_prior_count
        self.quality: dict[tuple[str, str, str], QualityPosterior] = {}
        self.failure: dict[tuple[str, str], FailureRatePosterior] = {}

    def fit(self, signals: list[ExtractedSignal]) -> "QualityAggregator":
        # --- continuous quality dimensions: weighted Normal-Normal conjugate ----
        # Prior precision and (precision-weighted) mean accumulator.
        prior_prec = self.prior_strength / (self.obs_std ** 2)
        sum_prec: dict[tuple[str, str, str], float] = defaultdict(lambda: prior_prec)
        sum_prec_mean: dict[tuple[str, str, str], float] = defaultdict(lambda: prior_prec * self.prior_mean)
        counts: dict[tuple[str, str, str], int] = defaultdict(int)

        # --- failure rate: category-base-rate Beta-Binomial -----------------------
        fail_counts: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])  # [failures, non]

        for s in signals:
            if s.signal_type == "failure":
                key = (s.product_id, s.use_profile)
                fail_counts[key][0] += s.confidence
            else:
                if s.quality_dim is None:
                    continue
                key3 = (s.product_id, s.use_profile, s.quality_dim)
                prec = s.confidence / (self.obs_std ** 2)
                sum_prec[key3] += prec
                sum_prec_mean[key3] += prec * s.signal_value
                counts[key3] += 1
                # A non-failure observation is also evidence of non-failure.
                fail_counts[(s.product_id, s.use_profile)][1] += s.confidence

        for key3, prec in sum_prec.items():
            mean = sum_prec_mean[key3] / prec
            std = math.sqrt(1.0 / prec)
            self.quality[key3] = QualityPosterior(
                product_id=key3[0], use_profile=key3[1], quality_dim=key3[2],
                posterior_mean=float(mean), posterior_std=float(std),
                credible_lo_90=float(max(0.0, mean - _Z90 * std)),
                credible_hi_90=float(min(1.0, mean + _Z90 * std)),
                evidence_count=counts[key3],
            )

        a0 = self.failure_prior_count
        b0 = self.failure_prior_count
        for (pid, prof), (f, nf) in fail_counts.items():
            a, b = a0 + f, b0 + nf
            self.failure[(pid, prof)] = FailureRatePosterior(
                product_id=pid, use_profile=prof, alpha=a, beta=b,
                rate_mean=float(a / (a + b)), evidence_count=int(round(f + nf)),
            )
        return self
