"""Standalone validator for PTP credentials against the frozen v0.1 schema.

This module deliberately depends on **nothing** in ``preferencelayer`` — it loads
the published JSON Schema (``contexts/ptp-credential-v0.1.schema.json``) and
checks an arbitrary credential document against it. An external party can run it
to validate a credential without reading our credential implementation::

    python -m preferencelayer.ptp.schema_validate path/to/credential.json

It uses the ``jsonschema`` library when available (full Draft 2020-12 checking);
if that optional dependency is absent it falls back to a small built-in checker
that covers the required-field / type / range / enum constraints the schema pins,
so CI and casual use never hard-fail purely on a missing dependency.

The exit code is 0 for a valid credential and 1 for an invalid one (or a usage
error), so it composes cleanly in CI and shell pipelines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The published, frozen schema lives at the repo root under contexts/. Resolve it
# relative to this file so the validator works from an installed package too
# (the schema is shipped as package data — see pyproject packaging).
_SCHEMA_FILENAME = "ptp-credential-v0.1.schema.json"


class ValidationFailed(Exception):
    """Raised with a human-readable list of schema violations."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def schema_path() -> Path:
    """Locate the published schema document.

    Searches, in order: the package-data copy next to this module, then the
    in-repo ``contexts/`` directory (for running straight from a checkout).
    """
    candidates = [
        Path(__file__).with_name(_SCHEMA_FILENAME),
        Path(__file__).resolve().parents[3] / "contexts" / _SCHEMA_FILENAME,
        Path.cwd() / "contexts" / _SCHEMA_FILENAME,
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"could not locate {_SCHEMA_FILENAME} (looked in: {', '.join(str(c) for c in candidates)})"
    )


def load_schema() -> dict:
    return json.loads(schema_path().read_text())


def validate(document: dict, schema: dict | None = None) -> None:
    """Validate ``document`` against the PTP v0.1 schema.

    Raises :class:`ValidationFailed` with all collected errors if invalid.
    Returns ``None`` on success.
    """
    schema = schema if schema is not None else load_schema()
    try:
        import jsonschema  # type: ignore
    except ImportError:
        errors = _fallback_validate(document, schema)
    else:
        validator = jsonschema.Draft202012Validator(schema)
        errors = [
            f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
        ]
    if errors:
        raise ValidationFailed(errors)


def is_valid(document: dict, schema: dict | None = None) -> bool:
    try:
        validate(document, schema)
        return True
    except ValidationFailed:
        return False


# --------------------------------------------------------------------------- #
# Minimal built-in checker (used only when `jsonschema` is not installed).      #
# Covers exactly the constraints the v0.1 schema pins; not a general engine.    #
# --------------------------------------------------------------------------- #
def _fallback_validate(doc: dict, schema: dict) -> list[str]:
    errors: list[str] = []

    def req(obj, fields, where):
        if not isinstance(obj, dict):
            errors.append(f"{where}: expected object")
            return False
        for f in fields:
            if f not in obj:
                errors.append(f"{where}: missing required field '{f}'")
        return isinstance(obj, dict)

    def num_in(name, val, lo, hi, where):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            errors.append(f"{where}/{name}: expected number")
        elif not (lo <= val <= hi):
            errors.append(f"{where}/{name}: {val} out of range [{lo}, {hi}]")

    if not req(doc, ["@context", "type", "issuer", "issuanceDate", "credentialSubject"], "<root>"):
        return errors

    ctx = doc.get("@context")
    if not isinstance(ctx, list) or "https://www.w3.org/ns/credentials/v2" not in ctx:
        errors.append("@context: must include 'https://www.w3.org/ns/credentials/v2'")
    if not isinstance(ctx, list) or "https://preferencelayer.io/context/v1" not in ctx:
        errors.append("@context: must include 'https://preferencelayer.io/context/v1'")

    typ = doc.get("type")
    if not isinstance(typ, list) or "VerifiableCredential" not in typ or "PreferenceCredential" not in typ:
        errors.append("type: must include 'VerifiableCredential' and 'PreferenceCredential'")

    issuer = doc.get("issuer")
    if not (isinstance(issuer, str) and issuer.startswith("did:")):
        errors.append("issuer: must be a DID string ('did:...')")

    subj = doc.get("credentialSubject", {})
    if req(subj, ["id", "preferenceGraph"], "credentialSubject"):
        sid = subj.get("id")
        if not (isinstance(sid, str) and sid.startswith("did:")):
            errors.append("credentialSubject/id: must be a DID string ('did:...')")
        g = subj.get("preferenceGraph", {})
        gw = "credentialSubject/preferenceGraph"
        if req(g, ["category", "version", "attributeNodes", "edges",
                   "updateCount", "privacyBudgetConsumed", "lastUpdated"], gw):
            if g.get("version") != "0.1":
                errors.append(f"{gw}/version: must be '0.1'")
            if not isinstance(g.get("updateCount"), int) or isinstance(g.get("updateCount"), bool) or g.get("updateCount", -1) < 0:
                errors.append(f"{gw}/updateCount: must be a non-negative integer")
            pbc = g.get("privacyBudgetConsumed")
            if not isinstance(pbc, (int, float)) or isinstance(pbc, bool) or pbc < 0:
                errors.append(f"{gw}/privacyBudgetConsumed: must be a non-negative number")
            for i, n in enumerate(g.get("attributeNodes", []) or []):
                nw = f"{gw}/attributeNodes/{i}"
                if req(n, ["id", "weight", "confidence"], nw):
                    num_in("weight", n.get("weight"), -1, 1, nw)
                    num_in("confidence", n.get("confidence"), 0, 1, nw)
            for i, e in enumerate(g.get("edges", []) or []):
                ew = f"{gw}/edges/{i}"
                if req(e, ["source", "target", "weight"], ew):
                    num_in("weight", e.get("weight"), -1, 1, ew)
    return errors


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("usage: python -m preferencelayer.ptp.schema_validate <credential.json> [...]")
        return 0 if argv and argv[0] in ("-h", "--help") else 1

    schema = load_schema()
    rc = 0
    for path in argv:
        try:
            doc = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"{path}: ERROR could not read/parse: {exc}")
            rc = 1
            continue
        try:
            validate(doc, schema)
            print(f"{path}: OK (valid PTP credential v0.1)")
        except ValidationFailed as exc:
            print(f"{path}: INVALID")
            for err in exc.errors:
                print(f"  - {err}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
