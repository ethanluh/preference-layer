"""Agent integration layer: fuse portable preference with QIL quality.

This is the keystone that joins the project's two halves. The preference graph
(``models/`` + ``ptp/``) scores *what a user wants*; the Quality Intelligence
Layer (``qil/``) scores *how a product performs for a use profile*. An agent
ranking products needs both, combined by the confidence-adaptive α-blend from
``docs/architecture.md``:

    score = alpha * pref_score + (1 - alpha) * quality_score
    alpha = sigmoid(3.0 * (mean_confidence - 0.5))

* :mod:`~preferencelayer.agent.combine` — the blend math (pure functions).
* :class:`~preferencelayer.agent.recommender.AgentRecommender` — orchestration:
  calls the preference model and the QIL query service, then blends.
* :class:`~preferencelayer.agent.evaluate.IntegrationHarness` — the falsifiable
  benchmark showing adaptive α beats preference-only, quality-only, and fixed α.
"""

from . import combine
from .ablation import (
    CellResult,
    QualityHandlingHarness,
    QualityHandlingResult,
)
from .evaluate import (
    CohortBreakdown,
    ConditionResult,
    IntegrationHarness,
    IntegrationReport,
)
from .protocol import (
    ProtocolAgent,
    ProtocolRecommendation,
    credential_from_arrays,
    score_from_credential,
)
from .recommender import AgentRecommender, BlendResult

__all__ = [
    "combine",
    "AgentRecommender",
    "BlendResult",
    "IntegrationHarness",
    "IntegrationReport",
    "ConditionResult",
    "CohortBreakdown",
    "QualityHandlingHarness",
    "QualityHandlingResult",
    "CellResult",
    "ProtocolAgent",
    "ProtocolRecommendation",
    "score_from_credential",
    "credential_from_arrays",
]
