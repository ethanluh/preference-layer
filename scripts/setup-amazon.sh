#!/usr/bin/env bash
# Opt-in setup for the real-data (Amazon Reviews 2023) benchmark.
# Installs the heavier `[amazon]` extra (pandas + pyarrow + huggingface_hub) on demand,
# so the always-on SessionStart hook (scripts/setup.sh) can stay light and fast.
# Idempotent and runnable by hand:  bash scripts/setup-amazon.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
# Install the amazon extra (plus dev, so the documented pytest verification works on a
# cold venv). Guard on the editable package too, so a venv that happens to have the deps
# system-wide but no preferencelayer install still gets one.
if ! python -c "import preferencelayer, pandas, pyarrow, huggingface_hub" >/dev/null 2>&1; then
  pip install --quiet -e ".[dev,amazon]"
fi

echo "PreferenceLayer real-data deps ready (.venv + [amazon])."
echo "Run: python experiments/run_amazon_realdata.py"
