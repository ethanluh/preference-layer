"""Gaussian-process quality model over product release time (Work Stream B3).

``docs/architecture.md`` specifies the continuous quality dimensions as a Gaussian
process with a squared-exponential kernel over release time (e.g. "thermal
performance degrades with thermal-paste age; battery capacity degrades with cycle
count"). Phase 0 shipped a conjugate Normal-Normal *stand-in* (no temporal
kernel). This module is the GP upgrade.

It is dependency-free (NumPy only) and deliberately a thin, well-understood GP:

* **Kernel:** squared-exponential ``k(t, t') = signal_var * exp(-(t-t')^2 /
  (2 * lengthscale^2))`` over time in days since release.
* **Likelihood:** per-observation Gaussian noise, with variance inflated for
  low-confidence extractions (``obs_var / confidence``) so weak signals move the
  posterior less -- preserving the Phase 0 confidence-weighting semantics.
* **Prior mean:** a constant neutral-quality prior (default 0.5), so a product
  with no observations sits at the prior with wide credible bounds.

The posterior at a query time ``t*`` yields a mean and variance, from which we
build the same ``mean + posterior_std + 90% credible interval + evidence_count``
contract the ``/quality`` API already consumes -- the GP is a drop-in for the
Normal-Normal special case (and reduces to it when all observations share the
query time).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_Z90 = 1.6448536269514722  # matches aggregate.py / the architecture's 90% CI


@dataclass
class GPHyperparams:
    """Squared-exponential GP hyperparameters (the parameters refit nightly)."""

    lengthscale_days: float = 180.0   # how fast quality varies over release time
    signal_var: float = 0.08          # prior variance of the quality function
    obs_var: float = 0.0225           # base observation noise (== 0.15^2, the Phase 0 obs_std)
    prior_mean: float = 0.5           # neutral-quality prior mean


@dataclass
class GPPosterior:
    """GP posterior at a query time, plus the fitted parameters (for storage)."""

    mean: float
    std: float
    credible_lo_90: float
    credible_hi_90: float
    evidence_count: int
    query_time: float
    hyperparams: GPHyperparams


def _sq_exp(t1: np.ndarray, t2: np.ndarray, lengthscale: float, signal_var: float) -> np.ndarray:
    d = t1[:, None] - t2[None, :]
    return signal_var * np.exp(-(d ** 2) / (2.0 * lengthscale ** 2))


def fit_gp_posterior(
    times: list[float],
    values: list[float],
    confidences: list[float],
    query_time: float = 0.0,
    hp: GPHyperparams | None = None,
) -> GPPosterior:
    """Posterior of the quality function at ``query_time`` given timed observations.

    Standard GP regression with a constant prior mean and heteroscedastic noise
    ``obs_var / confidence`` per point. With zero observations the posterior is
    the prior. With one or more, it is the exact conjugate GP posterior.
    """
    hp = hp or GPHyperparams()
    n = len(values)
    if n == 0:
        std = math.sqrt(hp.signal_var)
        return GPPosterior(
            mean=hp.prior_mean, std=std,
            credible_lo_90=max(0.0, hp.prior_mean - _Z90 * std),
            credible_hi_90=min(1.0, hp.prior_mean + _Z90 * std),
            evidence_count=0, query_time=query_time, hyperparams=hp,
        )

    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float) - hp.prior_mean  # center on prior mean
    # Per-point noise: weaker (lower-confidence) observations get more noise.
    conf = np.clip(np.asarray(confidences, dtype=float), 1e-3, None)
    noise = hp.obs_var / conf

    K = _sq_exp(t, t, hp.lengthscale_days, hp.signal_var) + np.diag(noise)
    ts = np.asarray([query_time], dtype=float)
    Ks = _sq_exp(ts, t, hp.lengthscale_days, hp.signal_var)   # (1, n)
    Kss = float(_sq_exp(ts, ts, hp.lengthscale_days, hp.signal_var)[0, 0])

    # Solve K alpha = y  (symmetric PD -> Cholesky for stability).
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mean = hp.prior_mean + float((Ks @ alpha)[0])

    v = np.linalg.solve(L, Ks.T)            # (n, 1)
    var = max(Kss - float((v.T @ v).item()), 1e-9)
    std = math.sqrt(var)

    return GPPosterior(
        mean=float(mean), std=std,
        credible_lo_90=max(0.0, mean - _Z90 * std),
        credible_hi_90=min(1.0, mean + _Z90 * std),
        evidence_count=n, query_time=query_time, hyperparams=hp,
    )
