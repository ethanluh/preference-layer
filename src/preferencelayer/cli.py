"""``preflayer`` command-line interface.

Subcommands:

* ``demo``           — run an end-to-end PTP lifecycle (create, authorize, read,
                       update, elicit) and print each step.
* ``experiment``     — run the Phase 0 cross-category transfer benchmark (Claim 1).
* ``qil-experiment`` — run the Phase 0 QIL extraction feasibility study (Claim 2).
* ``agent-demo``     — rank candidates with the preference+quality α-blend, showing
                       how a cold-start and a rich-history user diverge.
* ``integration``    — run the Phase 1 preference+quality integration benchmark.

Persistent credential store (Phase 1, Work Stream A — operate on a real on-disk,
encrypted store under ``$PREFLAYER_HOME`` or ``~/.preflayer``):

* ``init``           — create the user's identity + store (``--seed-demo`` adds a
                       starter laptops credential).
* ``authorize``      — mint a scoped, expiring agent token.
* ``view``           — summarize the identity, credentials and active agent tokens.
* ``revoke``         — revoke all tokens for an agent id.
* ``export``         — export the signed credentials as JSON.
* ``delete``         — irreversibly wipe the store.
"""

from __future__ import annotations

import argparse
import sys
import time

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


def _protocol_demo(args) -> int:
    """Rank products end-to-end through the real PTP + QIL MCP tool handlers."""
    import numpy as np

    from .agent.protocol import ProtocolAgent
    from .attributes import AttributeSchema
    from .mcp.server import PTPToolHandler
    from .ptp import (
        AttributeNode,
        CredentialStore,
        Edge,
        PreferenceCredential,
        PreferenceGraph,
        new_user_keypair,
    )
    from .qil import QualityAggregator, QualityService
    from .qil.extract import ExtractedSignal
    from .qil.mcp_server import QILToolHandler

    schema = AttributeSchema.for_category("laptops")

    # 1. The user's portable preference credential, held in their own store.
    sk, did = new_user_keypair()
    store = CredentialStore(sk, did)
    store.put_credential(PreferenceCredential(did, PreferenceGraph(
        category="laptops",
        attributeNodes=[
            AttributeNode("performance", weight=0.8, confidence=0.8),
            AttributeNode("portability", weight=0.6, confidence=0.7),
            AttributeNode("price_sensitivity", weight=-0.3, confidence=0.6),
        ],
        edges=[Edge("performance", "portability", weight=-0.4, contextKey="travel")],
    )))
    token = store.authorize_agent("agent.shopping.example", scope=["laptops"])
    ptp = PTPToolHandler(store, token)

    # 2. A QIL service with quality evidence for the candidate products.
    products = {"workhorse": 0.85, "ultrabook": 0.55, "budget": 0.30}
    sigs = [
        ExtractedSignal(pid, "laptops", "travel", "performance", None, dim, mean, 0.9)
        for pid, mean in products.items() for dim in ("thermal", "build_quality") for _ in range(12)
    ]
    qil = QILToolHandler(QualityService(QualityAggregator().fit(sigs)))

    # 3. Candidate products as attribute vectors over the laptops schema.
    def vec(**kw):
        x = np.zeros(schema.dim)
        for name, v in kw.items():
            x[schema.index(name)] = v
        return x

    cand_ids = list(products)
    cand_attrs = np.stack([
        vec(performance=0.9, portability=0.3, price_sensitivity=0.2),   # workhorse
        vec(performance=0.6, portability=0.9, price_sensitivity=0.4),   # ultrabook
        vec(performance=0.3, portability=0.5, price_sensitivity=0.9),   # budget
    ])

    agent = ProtocolAgent(ptp, qil, schema)
    rec = agent.recommend("laptops", "travel", cand_ids, cand_attrs,
                          query_context="frequent travel, sustained workloads")

    print("Agent ranks products using ONLY the PTP get_preference + QIL get_quality tools.\n")
    print(f"get_preference -> confidence={rec.confidence:.2f} coverage={rec.coverage}")
    print(f"blended with alpha = sigmoid(3*(confidence-0.5)) = {rec.alpha:.2f}\n")
    print(f"{'product':<12}{'pref':>8}{'quality':>9}{'blended':>9}")
    print("-" * 38)
    for i in rec.order:
        print(f"{cand_ids[i]:<12}{rec.pref[i]:>8.2f}{rec.quality[i]:>9.2f}{rec.blended[i]:>9.2f}")
    print(f"\nTop recommendation: {cand_ids[rec.order[0]]}")

    # 4. Revoke the agent: the same call is now denied at the protocol layer.
    store.revoke_agent("agent.shopping.example")
    denied = agent.recommend("laptops", "travel", cand_ids, cand_attrs)
    print(f"\nAfter revocation: get_preference denied -> status {denied.status}, no ranking produced.")
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


def _starter_credential(did: str):
    """A small, realistic laptops credential so `view`/`export` show content."""
    from .ptp import AttributeNode, Edge, PreferenceCredential, PreferenceGraph

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
    return PreferenceCredential(did, graph)


def _open_store(create: bool = False):
    from .ptp import IdentityLocked, PersistentCredentialStore, StoreNotFound

    try:
        return PersistentCredentialStore.open(args_home(), create=create)
    except StoreNotFound as e:
        print(f"error: {e}", file=sys.stderr)
    except IdentityLocked as e:
        print(f"error: {e}", file=sys.stderr)
    return None


# `args_home` is set by `main` from the resolved --home/$PREFLAYER_HOME.
_HOME = None


def args_home():
    return _HOME


def _init(args) -> int:
    from .ptp import PersistentCredentialStore, StoreNotFound

    # Refuse to clobber an existing store unless --force.
    try:
        existing = PersistentCredentialStore.open(args_home(), create=False)
        existing.close()
        if not args.force:
            print(f"error: a store already exists at {args_home() or '~/.preflayer'} "
                  f"(use --force to reinitialize)", file=sys.stderr)
            return 1
        existing = PersistentCredentialStore.open(args_home(), create=False)
        existing.delete_all()
    except StoreNotFound:
        pass

    store = PersistentCredentialStore.open(args_home(), create=True)
    print(f"Initialized PreferenceLayer store at {store.home}")
    print(f"Identity (did:key): {store.issuer_did}")
    if args.seed_demo:
        store.put_credential(_starter_credential(store.issuer_did))
        print("Seeded a starter 'laptops' credential.")
    store.close()
    return 0


def _authorize(args) -> int:
    store = _open_store()
    if store is None:
        return 1
    scope = args.scope or ["*"]
    token = store.authorize_agent(args.agent_id, scope=scope, ttl_seconds=args.ttl)
    store.close()
    print(f"Authorized '{args.agent_id}' scope={scope} ttl={args.ttl}s")
    print(f"token: {token}")
    return 0


def _revoke(args) -> int:
    store = _open_store()
    if store is None:
        return 1
    n = store.revoke_agent(args.agent_id)
    store.close()
    print(f"Revoked {n} active token(s) for '{args.agent_id}'.")
    return 0


def _export(args) -> int:
    import json as _json

    store = _open_store()
    if store is None:
        return 1
    bundle = store.export_bundle()
    store.close()
    text = _json.dumps(bundle, indent=2)
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(text)
        print(f"Exported {len(bundle['credentials'])} credential(s) to {args.out}")
    else:
        print(text)
    return 0


def _delete(args) -> int:
    store = _open_store()
    if store is None:
        return 1
    if not args.yes:
        print(f"This will irreversibly delete the store and identity at {store.home}.")
        resp = input("Type 'delete' to confirm: ").strip()
        if resp != "delete":
            store.close()
            print("Aborted.")
            return 1
    store.delete_all()
    print("Store deleted.")
    return 0


def _view(args) -> int:
    store = _open_store()
    if store is None:
        return 1
    store.prune_expired()
    print(f"Store:    {store.home}")
    print(f"Identity: {store.issuer_did}")
    cats = store.categories()
    print(f"\nCredentials ({len(cats)}):")
    if not cats:
        print("  (none — run `preflayer init --seed-demo` or have an agent elicit one)")
    for c in sorted(cats):
        cred = store._creds[c]
        g = cred.graph
        confs = [n.confidence for n in g.attributeNodes]
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        valid = cred.verify(store.signing_key.verify_key)
        print(f"  [{c}] nodes={len(g.attributeNodes)} edges={len(g.edges)} "
              f"mean_conf={mean_conf:.2f} updates={g.updateCount} "
              f"budget={g.privacyBudgetConsumed} signed={'ok' if valid else 'INVALID'}")
    tokens = store.agent_tokens()
    print(f"\nActive agent tokens ({len(tokens)}):")
    if not tokens:
        print("  (none)")
    for at in tokens:
        ttl = max(0, int(at.expires_at - time.time()))
        print(f"  {at.agent_id:<28} scope={at.scope} expires_in={ttl}s")
    store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    global _HOME
    parser = argparse.ArgumentParser(prog="preflayer", description="PreferenceLayer prototype CLI")
    parser.add_argument("--home", default=None,
                        help="Credential store directory (default: $PREFLAYER_HOME or ~/.preflayer).")
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
    sub.add_parser("protocol-demo", help="Rank products through the real PTP + QIL MCP tool handlers.")
    integ = sub.add_parser("integration", help="Run the Phase 1 integration benchmark.")
    integ.add_argument("--users", type=int, default=300)
    integ.add_argument("--seed", type=int, default=23)

    ini = sub.add_parser("init", help="Create the user's identity + persistent credential store.")
    ini.add_argument("--force", action="store_true", help="Reinitialize, deleting any existing store.")
    ini.add_argument("--seed-demo", action="store_true", help="Seed a starter 'laptops' credential.")
    auth = sub.add_parser("authorize", help="Mint a scoped, expiring agent token.")
    auth.add_argument("agent_id")
    auth.add_argument("--scope", nargs="*", default=None, help="Category scope(s); default '*' (all).")
    auth.add_argument("--ttl", type=int, default=86_400, help="Token lifetime in seconds (default 1 day).")
    rev = sub.add_parser("revoke", help="Revoke all active tokens for an agent id.")
    rev.add_argument("agent_id")
    sub.add_parser("view", help="Summarize the persistent credential store.")
    exp_cmd = sub.add_parser("export", help="Export signed credentials as JSON.")
    exp_cmd.add_argument("--out", default=None, help="Write to a file instead of stdout.")
    dele = sub.add_parser("delete", help="Irreversibly delete the store and identity.")
    dele.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")

    args = parser.parse_args(argv)
    _HOME = args.home
    if args.command == "demo":
        return _demo()
    if args.command == "experiment":
        return _experiment(args)
    if args.command == "qil-experiment":
        return _qil_experiment(args)
    if args.command == "agent-demo":
        return _agent_demo(args)
    if args.command == "protocol-demo":
        return _protocol_demo(args)
    if args.command == "integration":
        return _integration(args)
    if args.command == "init":
        return _init(args)
    if args.command == "authorize":
        return _authorize(args)
    if args.command == "revoke":
        return _revoke(args)
    if args.command == "view":
        return _view(args)
    if args.command == "export":
        return _export(args)
    if args.command == "delete":
        return _delete(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
