"""End-to-end tests for the `preflayer` persistent-store CLI commands."""

import json

from preferencelayer.cli import main


def _run(home, *argv) -> int:
    return main(["--home", str(home), *argv])


def test_init_view_authorize_revoke_export_delete(tmp_path, capsys):
    home = tmp_path / "store"

    # init seeds a starter credential
    assert _run(home, "init", "--seed-demo") == 0
    out = capsys.readouterr().out
    assert "Initialized PreferenceLayer store" in out
    assert "did:key:" in out

    # re-init without --force is refused
    assert _run(home, "init") == 1
    assert "already exists" in capsys.readouterr().err

    # view shows the seeded credential and no tokens
    assert _run(home, "view") == 0
    out = capsys.readouterr().out
    assert "[laptops]" in out and "signed=ok" in out
    assert "Active agent tokens (0)" in out

    # authorize then view shows the live token
    assert _run(home, "authorize", "agent.shop", "--scope", "laptops", "--ttl", "3600") == 0
    assert "token:" in capsys.readouterr().out
    assert _run(home, "view") == 0
    assert "agent.shop" in capsys.readouterr().out

    # export writes signed credentials to a file
    out_file = tmp_path / "bundle.json"
    assert _run(home, "export", "--out", str(out_file)) == 0
    bundle = json.loads(out_file.read_text())
    assert bundle["credentials"][0]["credentialSubject"]["preferenceGraph"]["category"] == "laptops"
    assert "proof" in bundle["credentials"][0]

    # revoke removes the token
    assert _run(home, "revoke", "agent.shop") == 0
    assert "Revoked 1" in capsys.readouterr().out
    assert _run(home, "view") == 0
    assert "Active agent tokens (0)" in capsys.readouterr().out

    # delete --yes wipes the store; subsequent view fails cleanly
    assert _run(home, "delete", "--yes") == 0
    assert _run(home, "view") == 1
    assert "no PreferenceLayer store" in capsys.readouterr().err


def test_view_before_init_errors(tmp_path, capsys):
    assert _run(tmp_path / "nope", "view") == 1
    assert "no PreferenceLayer store" in capsys.readouterr().err
