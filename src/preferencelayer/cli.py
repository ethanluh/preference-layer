"""``preflayer`` command-line interface.

Subcommands:

* ``demo``           — run an end-to-end PTP lifecycle (create, authorize, read,
                       update, elicit) and print each step.
* ``experiment``     — run the Phase 0 cross-category transfer benchmark (Claim 1).
* ``qil-experiment`` — run the Phase 0 QIL extraction feasibility study (Claim 2).
* ``agent-demo``     — rank candidates with the preference+quality α-blend, showing
                       how a cold-start and a rich-history user diverge.
* ``integration``    — run the Phase 1 preference+quality integration benchmark.
* ``view``           — summarize a saved credential store.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _demo() -> int:
    from .ptp import (
        AttributeNode,
        CredentialStore,
        Edge,
        PreferenceCredential,
        PreferenceGraph,
        new_user_keypair,
    )

    sk, did = new_user_keypair()
    print(f"Created user identity: {did[:32]}...")

    graph = PreferenceGraph(
        category="laptops",
        attributeNodes=[
            AttributeNode("performance", weight=0.8, confidence=0.7),
            AttributeNode("portability", weight=0.6, confidence=0.5),
            AttributeNode("price_sensitivity", weight=-0.3, confidence=0.6),
            AttributeNode("build_quality", weight=0.5, confidence=0.3),
        ],
        edges=[Edge("performance", "portability", weight=-0.4, contextKey="travel")],
        coldStartPrior="laptops_population_v0",
    )
    cred = PreferenceCredential(did, graph)
    cred.sign(sk)
    print(f"Signed credential valid: {cred.verify(sk.verify_key)}")

    store = CredentialStore(sk, did)
    store.put_credential(cred)

    token = store.authorize_agent("agent.shopping.example", scope=["laptops"])
    print(f"Authorized agent, token: {token[:20]}...")

    pref = store.get_preference(token, "laptops", query_context="sustained ML workload, frequent travel",
                                disclosure_scope=["performance", "portability", "build_quality"])
    print(f"\nget_preference -> confidence={pref['confidence']} "
          f"coverage={pref['coverage']} elicit_recommended={pref['elicitation_recommended']}")

    elicit = store.elicit(token, "laptops", max_questions=2)
    print("\nrequest_elicitation -> top questions:")
    for q in elicit["questions"]:
        print(f"  [{q['target_attribute']}] {q['text']} (IG={q['information_gain']})")

    out = store.submit_outcome(token, "laptops", product_id="thinkpad-x1-carbon-gen12",
                               outcome_type="purchase", use_context="software development, travel")
    print(f"\nsubmit_outcome -> {out['status']} affected={out['affected_nodes']} "
          f"budget_consumed={out['privacy_budget_consumed']}")

    from .ptp import AuthError

    print(f"\nRevoked {store.revoke_agent('agent.shopping.example')} token(s).")
    try:
        store.get_preference(token, "laptops")
        print("Post-revocation get_preference: UNEXPECTEDLY allowed")
    except AuthError as e:
        print(f"Post-revocation get_preference correctly denied: {e}")
    return 0


def _experiment(args) -> int:
    from .data import synthetic
    from .eval import ExperimentHarness
    from .models import (
        FlatAttributeRecommender,
        FlatItemEmbeddingRecommender,
        SparsePreferenceGraph,
    )

    ds = synthetic.generate(n_users=args.users, seed=args.seed)
    facs = {
        "flat_item_embedding": lambda: FlatItemEmbeddingRecommender(),
        "flat_attribute": lambda: FlatAttributeRecommender(),
        "preference_graph": lambda: SparsePreferenceGraph(),
    }
    h = ExperimentHarness(ds, k=10, seed=13)
    transfer = h.run_transfer(facs, "laptops", "headphones")
    for name, r in sorted(transfer.items(), key=lambda kv: -kv[1].ndcg):
        print(f"{name:<22} NDCG@10={r.ndcg:.4f}")
    comp = next(c for c in h.compare(transfer) if c.baseline == "flat_attribute")
    print(f"\nGraph vs flat_attribute (transfer): {comp.rel_gain_pct:+.1f}%  p={comp.p_value:.4f}")
    return 0


def _qil_experiment(args) -> int:
    from .qil import QILExtractor, QualityAggregator, QualityService, corpus as corpus_mod, evaluate
    from .qil.eval import GATE_PRECISION

    cp = corpus_mod.generate(n_train=args.train, n_test=args.test, seed=args.seed)
    ex = QILExtractor().fit(cp.train)
    report = evaluate([s.use_profile for s in cp.test], ex.predict_use_profiles(cp.test))
    for m in sorted(report.per_class, key=lambda m: -m.support):
        print(f"{m.label:<16} precision={m.precision:.3f} recall={m.recall:.3f} (n={m.support})")
    print(f"\nmacro precision={report.macro_precision:.3f}  baseline(mfc)={report.baseline_precision:.3f}")
    status = "PASS" if report.gate_pass else "FAIL"
    print(f"Phase 0 QIL gate (>= {GATE_PRECISION:.0%}): {status}")

    # Show the aggregation + query layer on the extracted signals.
    svc = QualityService(QualityAggregator().fit(ex.extract(cp.train + cp.test)))
    cat0 = list(cp.products)[0]
    pa, pb = cp.products[cat0][0], cp.products[cat0][1]
    q = svc.quality(pa, "gaming")
    print(f"\n/quality {pa} (gaming): failure_rate={q.get('failure_rate')} "
          f"dims={list(q.get('dimensions', {}))}")
    c = svc.compare(pa, pb, "gaming")
    if c["status"] == 200:
        print(f"/compare {pa} vs {pb}: {c['dimensions']}")
    return 0 if report.gate_pass else 1


def _agent_demo(args) -> int:
    """Show the α-blend ranking the same candidates for a cold vs. rich user."""
    import numpy as np

    from .agent import AgentRecommender
    from .agent.combine import alpha_from_confidence
    from .data import integrated
    from .models import SparsePreferenceGraph
    from .qil import QualityAggregator, QualityService

    s = integrated.generate(n_users=120, seed=args.seed)
    idx = s.product_index()
    n_shared = s.schema.n_shared
    _, catalog = s.catalog_matrix()
    per_user = [np.stack([idx[p].attributes for p in u.purchases]) for u in s.users]
    model = SparsePreferenceGraph(seed=13)
    model.prepare(catalog, per_user, n_shared)
    service = QualityService(QualityAggregator().fit(s.signals))

    # Pick one low-confidence (cold-start) and one high-confidence (rich) user.
    cold = min(s.users, key=lambda u: u.mean_confidence)
    rich = max(s.users, key=lambda u: u.mean_confidence)
    print("An agent ranks the same candidate products by blending the user's portable")
    print("preference with use-profile quality. alpha = sigmoid(3*(confidence-0.5)):\n")
    for label, u in (("COLD-START", cold), ("RICH-HISTORY", rich)):
        purchased = np.stack([idx[p].attributes for p in u.purchases])
        state = model.fit(purchased, catalog, n_shared)
        agent = AgentRecommender(model, state, service, n_shared)
        cand_attrs = np.stack([idx[c].attributes for c in u.candidates])
        res = agent.rank(u.candidates, cand_attrs, u.use_profile, u.mean_confidence)
        relevant = set(u.relevant)
        top = [u.candidates[i] for i in res.order[:5]]
        hits = sum(1 for t in top if t in relevant)
        print(f"{label}: history={u.history_len} use_profile={u.use_profile} "
              f"confidence={u.mean_confidence:.2f} -> alpha={alpha_from_confidence(u.mean_confidence):.2f}")
        print(f"  top-5 ranked: {top}")
        print(f"  relevant hits in top-5: {hits}/5  (alpha near 0 leans quality, near 1 leans preference)\n")
    return 0


def _integration(args) -> int:
    from .agent import IntegrationHarness
    from .data import integrated

    s = integrated.generate(n_users=args.users, seed=args.seed)
    rep = IntegrationHarness(s, k=10, seed=13).run()
    for name in ("preference_only", "quality_only", "fixed_alpha", "adaptive_alpha"):
        print(f"{name:<18} NDCG@10={rep.conditions[name].ndcg:.4f}")
    gp, pp = rep.comparisons["preference_only"]
    gq, pq = rep.comparisons["quality_only"]
    print(f"\nBlend vs preference_only: {gp:+.4f} (p={pp:.4f}); "
          f"vs quality_only: {gq:+.4f} (p={pq:.4f})")
    print(f"Milestone (blend beats both single layers): "
          f"{'PASS' if rep.milestone_pass else 'FAIL'}")
    return 0 if rep.milestone_pass else 1


def _view(args) -> int:
    from .ptp import CredentialStore, new_user_keypair

    print("`view` requires a saved store and key; this prototype generates ephemeral keys per run.")
    print("Run `preflayer demo` for a full in-memory lifecycle.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="preflayer", description="PreferenceLayer prototype CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("demo", help="Run an end-to-end PTP lifecycle demo.")
    exp = sub.add_parser("experiment", help="Run the Phase 0 transfer benchmark (Claim 1).")
    exp.add_argument("--users", type=int, default=400)
    exp.add_argument("--seed", type=int, default=7)
    qil = sub.add_parser("qil-experiment", help="Run the Phase 0 QIL feasibility study (Claim 2).")
    qil.add_argument("--train", type=int, default=1400)
    qil.add_argument("--test", type=int, default=400)
    qil.add_argument("--seed", type=int, default=17)
    ad = sub.add_parser("agent-demo", help="Show the preference+quality α-blend ranking candidates.")
    ad.add_argument("--seed", type=int, default=23)
    integ = sub.add_parser("integration", help="Run the Phase 1 integration benchmark.")
    integ.add_argument("--users", type=int, default=300)
    integ.add_argument("--seed", type=int, default=23)
    sub.add_parser("view", help="Summarize a saved credential store.")

    args = parser.parse_args(argv)
    if args.command == "demo":
        return _demo()
    if args.command == "experiment":
        return _experiment(args)
    if args.command == "qil-experiment":
        return _qil_experiment(args)
    if args.command == "agent-demo":
        return _agent_demo(args)
    if args.command == "integration":
        return _integration(args)
    if args.command == "view":
        return _view(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
