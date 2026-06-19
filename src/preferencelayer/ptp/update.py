"""On-device differentially private credential update (PTP spec §5).

Outcome signals (purchase / return / dwell / rating / elicitation) produce a
gradient on the affected attribute nodes. The gradient is clipped to a sensitivity
bound and perturbed with Gaussian noise calibrated to (epsilon, delta) before it
touches the stored weights, so the update satisfies (epsilon, delta)-differential
privacy. Raw behavioral data never leaves the device; only the noised weight delta
is applied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .credential import PreferenceCredential

# Map outcome types to a base gradient sign / magnitude on matched attributes.
_OUTCOME_GRADIENT = {
    "purchase": 1.0,
    "return": -1.0,
    "dwell": 0.3,
    "rating": None,        # uses the supplied rating (centered at 0.5)
    "elicitation": None,   # uses the stated response directly
}


@dataclass
class DPConfig:
    epsilon: float = 2.0
    delta: float = 1e-5
    clip_norm: float = 1.0      # C, sensitivity bound
    learning_rate: float = 0.01
    confidence_lr: float = 0.02
    budget_max: float = 20.0


def gaussian_sigma(cfg: DPConfig) -> float:
    """Noise scale for the Gaussian mechanism: sigma = C * sqrt(2 ln(1.25/delta)) / eps."""
    return cfg.clip_norm * math.sqrt(2.0 * math.log(1.25 / cfg.delta)) / cfg.epsilon


def _clip(grad: np.ndarray, c: float) -> np.ndarray:
    norm = np.linalg.norm(grad)
    return grad / max(1.0, norm / c)


def apply_outcome(
    cred: PreferenceCredential,
    affected_nodes: list[str],
    outcome_type: str,
    rating: float | None = None,
    elicitation_weights: dict[str, float] | None = None,
    cfg: DPConfig | None = None,
    rng: np.random.Generator | None = None,
) -> PreferenceCredential:
    """Apply a single DP outcome update to the credential in place and return it.

    ``affected_nodes`` are the attribute node ids the outcome bears on (the
    use-context -> attribute mapping is the caller's responsibility). Raises
    ``BudgetExhausted`` if the privacy budget would be exceeded.
    """
    cfg = cfg or DPConfig()
    rng = rng or np.random.default_rng()

    if cred.graph.privacyBudgetConsumed + cfg.epsilon > cfg.budget_max:
        raise BudgetExhausted(
            f"privacy budget {cfg.budget_max} would be exceeded "
            f"(consumed={cred.graph.privacyBudgetConsumed}, step={cfg.epsilon})"
        )

    node_index = {n.id: n for n in cred.graph.attributeNodes}
    affected = [n for n in affected_nodes if n in node_index]
    if not affected:
        return cred

    # Build the raw gradient over affected nodes.
    grad = np.zeros(len(affected))
    base = _OUTCOME_GRADIENT.get(outcome_type, 0.0)
    for i, node_id in enumerate(affected):
        if outcome_type == "rating" and rating is not None:
            grad[i] = (rating - 0.5) * 2.0
        elif outcome_type == "elicitation" and elicitation_weights:
            grad[i] = elicitation_weights.get(node_id, 0.0)
        else:
            grad[i] = base if base is not None else 0.0

    # Clip, then add Gaussian noise (the DP mechanism).
    grad = _clip(grad, cfg.clip_norm)
    sigma = gaussian_sigma(cfg)
    grad = grad + rng.normal(0.0, sigma, size=grad.shape)

    # Apply to node weights and bump confidence.
    for i, node_id in enumerate(affected):
        node = node_index[node_id]
        node.weight = float(np.clip(node.weight + cfg.learning_rate * grad[i], -1.0, 1.0))
        node.confidence = float(min(1.0, node.confidence + cfg.confidence_lr * (1.0 - node.confidence)))

    cred.graph.updateCount += 1
    cred.graph.privacyBudgetConsumed = round(cred.graph.privacyBudgetConsumed + cfg.epsilon, 6)
    from datetime import datetime, timezone

    cred.graph.lastUpdated = datetime.now(timezone.utc).isoformat()
    return cred


class BudgetExhausted(RuntimeError):
    """Raised when an update would exceed the configured privacy budget."""
