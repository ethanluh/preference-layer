"""Protocol-level integration: rank products from the PTP and QIL MCP tools alone.

Everything in :mod:`preferencelayer.agent.recommender` ranks with an *in-process*
preference model — it holds the fitted :class:`SparsePreferenceGraph` object and
calls its ``score`` method directly. A real shopping agent never has that object.
It has two **MCP tools**:

* PTP ``get_preference`` → a signed, selectively-disclosed *preference credential*
  (a graph of attribute weights + interaction edges) plus a confidence score;
* QIL ``get_quality`` → use-profile-conditioned quality posteriors per product.

This module is the agent that lives on the far side of those tools. It reconstructs
a preference score for each candidate **from the disclosed credential graph** — the
proof that the credential is a sufficient, portable carrier of preference — queries
the QIL tool for quality, and blends them with the documented confidence-adaptive α
(:mod:`preferencelayer.agent.combine`). The point is to exercise the real
agent-facing seams (``PTPToolHandler.call`` / ``QILToolHandler.call``), including
auth, selective disclosure, and signing, rather than reaching into Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..attributes import AttributeSchema
from ..ptp.credential import AttributeNode, Edge, PreferenceCredential, PreferenceGraph
from ..qil.schema import QUALITY_DIMS
from . import combine


def score_from_credential(
    credential: dict | PreferenceCredential, candidate_attrs: np.ndarray, schema: AttributeSchema
) -> np.ndarray:
    """Score candidates from a disclosed PTP preference graph.

    Reconstructs the same linear-plus-interaction utility the preference graph
    learns, but from the *published* credential weights rather than an in-process
    model: ``score(x) = Σ_node w_node · x[node] + Σ_edge w_edge · x[a]·x[b]``.
    Attribute ids are mapped to columns via ``schema``; nodes/edges naming an
    attribute outside the schema are skipped (a credential may disclose a subset).
    """
    cred = credential if isinstance(credential, PreferenceCredential) else PreferenceCredential.from_dict(credential)
    graph = cred.graph
    col = {name: i for i, name in enumerate(schema.names)}
    scores = np.zeros(len(candidate_attrs))
    for node in graph.attributeNodes:
        j = col.get(node.id)
        if j is not None:
            scores += node.weight * candidate_attrs[:, j]
    for edge in graph.edges:
        a, b = col.get(edge.source), col.get(edge.target)
        if a is not None and b is not None:
            scores += edge.weight * candidate_attrs[:, a] * candidate_attrs[:, b]
    return scores


def quality_from_response(response: dict, *, neutral_quality: float = 0.5, failure_penalty: float = 0.0) -> float:
    """Collapse a QIL ``get_quality`` response to one quality score.

    Mean posterior over disclosed dimensions, optionally discounted by
    ``failure_penalty * failure_rate``; falls back to ``neutral_quality`` when the
    QIL has no evidence (a 404 or empty dimensions). Mirrors the scoring in
    :meth:`AgentRecommender.query_quality` but reads the MCP tool's dict.
    """
    if response.get("status") != 200 or not response.get("dimensions"):
        return neutral_quality
    score = float(np.mean([d["posterior_mean"] for d in response["dimensions"].values()]))
    fail = response.get("failure_rate")
    if fail is not None and failure_penalty:
        score -= failure_penalty * fail
    return score


@dataclass
class ProtocolRecommendation:
    """A ranking produced purely from the PTP + QIL tool responses."""

    status: int                       # 200, or the PTP error status if preference is unavailable
    order: list[int] = field(default_factory=list)
    blended: np.ndarray | None = None
    pref: np.ndarray | None = None
    quality: np.ndarray | None = None
    alpha: float = float("nan")
    confidence: float = float("nan")
    coverage: list[str] = field(default_factory=list)   # attribute ids the credential disclosed
    missing: list[str] = field(default_factory=list)
    elicitation_recommended: bool = False


class ProtocolAgent:
    """Ranks products using only the PTP and QIL MCP tool handlers.

    ``ptp_handler`` and ``qil_handler`` are anything with a
    ``call(name, arguments) -> dict`` method — the in-process
    :class:`PTPToolHandler` / :class:`QILToolHandler`, or a thin shim over a live
    MCP client. ``schema`` lets the agent map disclosed attribute ids to the columns
    of the candidate attribute matrix it is ranking.
    """

    def __init__(self, ptp_handler, qil_handler, schema: AttributeSchema, *,
                 neutral_quality: float = 0.5, failure_penalty: float = 0.0):
        self.ptp = ptp_handler
        self.qil = qil_handler
        self.schema = schema
        self.neutral_quality = neutral_quality
        self.failure_penalty = failure_penalty

    def recommend(
        self,
        category: str,
        use_profile: str,
        candidate_ids: list[str],
        candidate_attrs: np.ndarray,
        *,
        query_context: str = "",
        disclosure_scope: list[str] | None = None,
    ) -> ProtocolRecommendation:
        """Fetch preference + quality over the tools, blend, and rank.

        Returns the PTP error status (e.g. 403 on a revoked token, 404 with no
        credential) without ranking when preference is unavailable — the agent
        cannot personalize, and surfaces that rather than guessing.
        """
        pref_resp = self.ptp.call("get_preference", {
            "category": category,
            "query_context": query_context,
            "disclosure_scope": disclosure_scope,
        })
        if pref_resp.get("status") != 200:
            return ProtocolRecommendation(status=pref_resp.get("status", 500))

        pref = score_from_credential(pref_resp["credential"], candidate_attrs, self.schema)
        quality = np.array([
            quality_from_response(
                self.qil.call("get_quality", {"product_id": pid, "use_profile": use_profile,
                                               "dimensions": list(QUALITY_DIMS)}),
                neutral_quality=self.neutral_quality, failure_penalty=self.failure_penalty)
            for pid in candidate_ids
        ])
        confidence = float(pref_resp.get("confidence", 0.0))
        alpha = combine.alpha_from_confidence(confidence)
        blended = combine.blend(pref, quality, alpha)
        return ProtocolRecommendation(
            status=200,
            order=list(np.argsort(-blended)),
            blended=blended, pref=pref, quality=quality,
            alpha=alpha, confidence=confidence,
            coverage=pref_resp.get("coverage", []),
            missing=pref_resp.get("missing", []),
            elicitation_recommended=bool(pref_resp.get("elicitation_recommended", False)),
        )


def credential_from_arrays(
    schema: AttributeSchema,
    theta: np.ndarray,
    phi_pairs: list[tuple[int, int]],
    phi: np.ndarray,
    *,
    category: str,
    issuer_did: str,
    node_confidence: float,
    cold_start_prior: str | None = None,
) -> PreferenceCredential:
    """Build a PTP preference credential from learned/planted preference weights.

    Maps the shared-attribute weight vector ``theta`` to attribute nodes and the
    interaction ``phi`` over ``phi_pairs`` to edges, using the schema's shared
    attribute names. This is the bridge a real client would implement when
    *exporting* a fitted preference model into a portable credential; here it lets
    the protocol path be driven from the same preference parameters the benchmark
    plants, so a credential round-trip can be measured end to end.
    """
    names = schema.shared
    nodes = [
        AttributeNode(id=names[i], weight=float(theta[i]), confidence=float(node_confidence))
        for i in range(min(len(theta), len(names)))
    ]
    edges = [
        Edge(source=names[a], target=names[b], weight=float(phi[k]))
        for k, (a, b) in enumerate(phi_pairs)
        if a < len(names) and b < len(names)
    ]
    graph = PreferenceGraph(
        category=category, attributeNodes=nodes, edges=edges, coldStartPrior=cold_start_prior,
    )
    return PreferenceCredential(issuer_did, graph)
