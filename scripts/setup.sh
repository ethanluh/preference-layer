#!/usr/bin/env bash
# Idempotent environment setup for PreferenceLayer.
# Used by the Claude Code SessionStart hook and runnable by hand.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
# Install the package + dev deps if pytest isn't importable yet.
if ! python -c "import pytest, numpy, nacl" >/dev/null 2>&1; then
  pip install --quiet -e ".[dev]"
fi

echo "PreferenceLayer environment ready (.venv). Run: python -m pytest"
