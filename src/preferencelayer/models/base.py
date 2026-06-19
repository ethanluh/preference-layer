"""Recommender interface.

A recommender is fit from a single user's observed purchases in a *source*
category and then scores candidate items in a (possibly different) *target*
category. Attribute vectors are laid out as ``[shared..., category_specific...]``;
the first ``n_shared`` columns are aligned across every category, so a model that
restricts itself to that block transfers across categories for free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Recommender(ABC):
    """Base class for personalized recommenders fit per user."""

    name: str = "base"

    def prepare(self, catalog: np.ndarray, per_user_purchased: list[np.ndarray], n_shared: int) -> None:
        """Optional corpus-level setup (edge topology, priors). No-op by default.

        ``per_user_purchased`` is one ``(n_purchases_u, dim)`` array per user, so
        models can mine *within-user* structure (e.g. attribute interactions).
        """

    @abstractmethod
    def fit(
        self,
        purchased: np.ndarray,
        catalog: np.ndarray,
        n_shared: int,
        population: "PopulationPrior | None" = None,
    ) -> Any:
        """Return an opaque per-user state from observed purchases.

        ``purchased`` is ``(n_purchases, dim)``; ``catalog`` is the full source
        catalog ``(n_items, dim)`` (used for negatives / co-occurrence statistics).
        """

    @abstractmethod
    def score(self, state: Any, candidates: np.ndarray, n_shared: int) -> np.ndarray:
        """Score ``(n_candidates, dim)`` items for the fitted user. Higher is better."""


class PopulationPrior:
    """Population-level statistics used for cold-start initialization.

    Mirrors the ``coldStartPrior`` field in the PTP credential schema: a new user
    with little history leans on the population mean until their own signal
    accumulates.
    """

    def __init__(self, theta_mean: np.ndarray, phi_mean: np.ndarray):
        self.theta_mean = theta_mean
        self.phi_mean = phi_mean

    @classmethod
    def neutral(cls, n_shared: int, n_pairs: int) -> "PopulationPrior":
        return cls(np.zeros(n_shared), np.zeros(n_pairs))
