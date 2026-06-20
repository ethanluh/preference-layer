"""Bayesian aggregation of extracted signals into quality posteriors.

Implements the aggregation layer from ``docs/architecture.md``:

* **Failure rate** per ``(product_id, use_profile)``: hierarchical Beta-Binomial.
  A category-level base rate sets the prior ``Beta(a0, b0)``; each extracted
  ``failure`` signal increments failures, each non-failure increments
  non-failures. Posterior mean ``= a / (a + b)``.
* **Quality dimensions** per ``(product_id, use_profile, quality_dim)``: a
  **Gaussian process** with a squared-exponential kernel over product release
  time, exactly as ``docs/architecture.md`` specifies (Work Stream B3). This
  replaces the Phase 0 Normal-Normal stand-in. The GP reduces to that conjugate
  estimate when every observation shares the query time, so the
  ``posterior_mean + 90% credible interval + evidence_count`` contract the
  ``/quality`` API consumes is unchanged. See ``gp.py``.

Each observation is weighted by its extraction ``confidence`` (low-confidence
signals get inflated GP noise, so they move the posterior less) and carries an
optional ``observed_at`` time in days since release.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .extract import ExtractedSignal
from .gp import GPHyperparams, fit_gp_posterior


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
    """Aggregates extracted signals into per-(product, use_profile, dim) posteriors.

    Quality dimensions use a Gaussian process over release time (``gp.py``);
    failure rates use a category-base-rate Beta-Binomial. The GP query is
    evaluated at ``query_time`` (days since release; 0 == release time), so a
    caller can ask "current quality" by passing the product's age.
    """

    def __init__(
        self,
        prior_mean: float = 0.5,             # neutral-quality GP prior mean
        obs_std: float = 0.15,               # per-observation noise std (-> GP obs_var)
        failure_prior_count: float = 2.0,
        lengthscale_days: float = 180.0,     # GP squared-exponential lengthscale
        prior_strength: float = 4.0,         # prior "pseudo-observations"; -> GP signal_var
        signal_var: float | None = None,     # GP prior variance; overrides prior_strength
        query_time: float = 0.0,             # days since release to evaluate the GP at
    ):
        self.prior_mean = prior_mean
        self.obs_std = obs_std
        self.failure_prior_count = failure_prior_count
        self.prior_strength = prior_strength
        self.query_time = query_time
        # Map the conjugate "prior_strength" onto the GP prior variance so the old
        # lever is preserved: a strong prior (large prior_strength) is a tight
        # (small) signal_var; prior_strength -> 0 is a near-flat prior, so the GP
        # posterior mean collapses to the confidence-weighted sample mean ("raw").
        if signal_var is None:
            signal_var = (obs_std ** 2) / prior_strength if prior_strength > 0 else 1e9
        self.gp_hp = GPHyperparams(
            lengthscale_days=lengthscale_days,
            signal_var=signal_var,
            obs_var=obs_std ** 2,
            prior_mean=prior_mean,
        )
        self.quality: dict[tuple[str, str, str], QualityPosterior] = {}
        self.failure: dict[tuple[str, str], FailureRatePosterior] = {}

    def fit(self, signals: list[ExtractedSignal]) -> "QualityAggregator":
        # --- continuous quality dimensions: GP over release time ------------------
        # Collect timed, confidence-weighted observations per (product, profile, dim).
        Obs = tuple[list[float], list[float], list[float]]  # (times, values, confidences)
        obs: dict[tuple[str, str, str], Obs] = defaultdict(lambda: ([], [], []))

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
                t = 0.0 if s.observed_at is None else float(s.observed_at)
                times, values, confs = obs[key3]
                times.append(t)
                values.append(s.signal_value)
                confs.append(s.confidence)
                # A non-failure observation is also evidence of non-failure.
                fail_counts[(s.product_id, s.use_profile)][1] += s.confidence

        for key3, (times, values, confs) in obs.items():
            post = fit_gp_posterior(times, values, confs,
                                    query_time=self.query_time, hp=self.gp_hp)
            self.quality[key3] = QualityPosterior(
                product_id=key3[0], use_profile=key3[1], quality_dim=key3[2],
                posterior_mean=post.mean, posterior_std=post.std,
                credible_lo_90=post.credible_lo_90, credible_hi_90=post.credible_hi_90,
                evidence_count=post.evidence_count,
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
