"""``preflayer`` command-line interface.

Subcommands:

* ``demo``       — run an end-to-end PTP lifecycle (create, authorize, read,
                   update, elicit) and print each step.
* ``experiment`` — run the Phase 0 cross-category transfer benchmark.
* ``view``       — summarize a saved credential store.
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


def _view(args) -> int:
    from .ptp import CredentialStore, new_user_keypair

    print("`view` requires a saved store and key; this prototype generates ephemeral keys per run.")
    print("Run `preflayer demo` for a full in-memory lifecycle.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="preflayer", description="PreferenceLayer prototype CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("demo", help="Run an end-to-end PTP lifecycle demo.")
    exp = sub.add_parser("experiment", help="Run the Phase 0 transfer benchmark.")
    exp.add_argument("--users", type=int, default=400)
    exp.add_argument("--seed", type=int, default=7)
    sub.add_parser("view", help="Summarize a saved credential store.")

    args = parser.parse_args(argv)
    if args.command == "demo":
        return _demo()
    if args.command == "experiment":
        return _experiment(args)
    if args.command == "view":
        return _view(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
