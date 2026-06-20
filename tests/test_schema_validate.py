"""Tests for the published PTP v0.1 JSON Schema + standalone validator.

These exercise the validator both ways: against a credential produced by our own
``PreferenceCredential`` (proving the implementation conforms to the frozen
schema) and against hand-built good/bad documents (proving the validator rejects
the right things). The validator must not import our credential logic, so an
external party can run it standalone.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from preferencelayer.ptp.credential import (
    AttributeNode,
    ContextConditioner,
    Edge,
    PreferenceCredential,
    PreferenceGraph,
    new_user_keypair,
)
from preferencelayer.ptp.schema_validate import (
    ValidationFailed,
    is_valid,
    load_schema,
    main,
    schema_path,
    validate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEXTS = REPO_ROOT / "contexts"


def _full_credential() -> dict:
    sk, did = new_user_keypair(seed=b"7" * 32)
    graph = PreferenceGraph(
        category="laptops",
        attributeNodes=[
            AttributeNode("performance", weight=0.8, confidence=0.7, embedding=[0.1, -0.2]),
            AttributeNode("portability", weight=0.6, confidence=0.5),
        ],
        edges=[Edge("performance", "portability", weight=-0.4, contextKey="travel")],
        contextConditioners=[ContextConditioner("travel", activeNodes=["portability"])],
        coldStartPrior="laptops_population_v0",
    )
    # Required flat update fields must be populated for a frozen-schema credential.
    graph.updateCount = 3
    graph.privacyBudgetConsumed = 6.0
    graph.lastUpdated = "2026-05-10T14:22:00+00:00"
    cred = PreferenceCredential(did, graph)
    cred.sign(sk)
    return cred.to_dict()


# ------------------------------------------------------------------ published files
def test_context_document_is_valid_json():
    ctx = json.loads((CONTEXTS / "ptp-v1.jsonld").read_text())
    assert "@context" in ctx
    # The published context must alias the spec's canonical field names.
    inner = ctx["@context"]["preferenceGraph"]["@context"]
    assert "attributeNodes" in inner and "edges" in inner and "contextConditioners" in inner


def test_schema_document_is_valid_json_and_locatable():
    assert schema_path().is_file()
    schema = load_schema()
    assert schema["title"].startswith("PTP Preference Credential")


def test_packaged_schema_matches_published_context_copy():
    """The package-data schema copy must stay byte-identical to contexts/ (no drift)."""
    canonical = (CONTEXTS / "ptp-credential-v0.1.schema.json").read_text()
    packaged = (REPO_ROOT / "src" / "preferencelayer" / "ptp" / "ptp-credential-v0.1.schema.json").read_text()
    assert json.loads(canonical) == json.loads(packaged)


# -------------------------------------------------------------- implementation conforms
def test_real_credential_validates():
    validate(_full_credential())  # must not raise
    assert is_valid(_full_credential())


def test_cold_start_credential_without_edges_validates():
    """A freshly built cold-start credential (attribute nodes, NO edges) must
    serialize and PASS the frozen schema. ``edges`` is OPTIONAL (absent == [])."""
    sk, did = new_user_keypair(seed=b"9" * 32)
    graph = PreferenceGraph(
        category="laptops",
        attributeNodes=[
            AttributeNode("performance", weight=0.5, confidence=0.3),
            AttributeNode("portability", weight=0.4, confidence=0.2),
        ],
        coldStartPrior="laptops_population_v0",
    )  # no edges, no contextConditioners
    # Required flat update fields must be populated for a frozen-schema credential.
    graph.lastUpdated = "2026-05-10T14:22:00+00:00"
    cred = PreferenceCredential(did, graph).sign(sk)
    doc = cred.to_dict()
    # The tidy serializer drops the empty edges array entirely.
    assert "edges" not in doc["credentialSubject"]["preferenceGraph"]
    validate(doc)  # must not raise
    assert is_valid(doc)


# ------------------------------------------------------------------ rejection cases
def test_missing_required_graph_field_rejected():
    doc = _full_credential()
    del doc["credentialSubject"]["preferenceGraph"]["updateCount"]
    with pytest.raises(ValidationFailed) as ei:
        validate(doc)
    assert any("updateCount" in e for e in ei.value.errors)


def test_weight_out_of_range_rejected():
    doc = _full_credential()
    doc["credentialSubject"]["preferenceGraph"]["attributeNodes"][0]["weight"] = 5.0
    assert not is_valid(doc)


def test_wrong_version_rejected():
    doc = _full_credential()
    doc["credentialSubject"]["preferenceGraph"]["version"] = "0.2"
    assert not is_valid(doc)


def test_missing_context_uri_rejected():
    doc = _full_credential()
    doc["@context"] = ["https://www.w3.org/ns/credentials/v2"]  # drop the PTP context
    assert not is_valid(doc)


def test_non_did_issuer_rejected():
    doc = _full_credential()
    doc["issuer"] = "https://example.com/not-a-did"
    assert not is_valid(doc)


# ------------------------------------------------------------------ static fixtures
def test_good_fixture_valid_bad_fixture_invalid():
    good = json.loads((REPO_ROOT / "tests" / "fixtures" / "credential_valid.json").read_text())
    bad = json.loads((REPO_ROOT / "tests" / "fixtures" / "credential_invalid.json").read_text())
    assert is_valid(good)
    assert not is_valid(bad)


# ------------------------------------------------------------------ CLI entry point
def test_cli_accepts_good_rejects_bad(tmp_path, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_full_credential()))
    assert main([str(good)]) == 0

    bad_doc = _full_credential()
    bad_doc["credentialSubject"]["preferenceGraph"]["attributeNodes"][0]["confidence"] = 9.9
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(bad_doc))
    assert main([str(bad)]) == 1


def test_cli_runs_as_module_without_importing_credential_logic(tmp_path):
    """The validator must be runnable purely as a script over a JSON file."""
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_full_credential()))
    proc = subprocess.run(
        [sys.executable, "-m", "preferencelayer.ptp.schema_validate", str(good)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
