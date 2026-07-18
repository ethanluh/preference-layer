#!/usr/bin/env bash
# Blocks "gh pr create"/"gh pr edit" commands whose body is missing Quire's
# <!-- declared-direction: ... --> marker. Installed by Quire's repo setup — see CLAUDE.md.
set -euo pipefail

input="$(cat)"

if command -v jq >/dev/null 2>&1; then
  # Proper JSON parsing: handles embedded/escaped quotes and newlines in the
  # command string (e.g. `--title "..."` followed by a multi-line --body),
  # which the naive regex below truncates at the first embedded quote.
  command="$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null || true)"
else
  # jq unavailable: fall back to the naive extraction (breaks on embedded quotes).
  command="$(printf '%s' "$input" | grep -o '"command"[[:space:]]*:[[:space:]]*"[^"]*"' | head -n1 || true)"
fi

if [[ -z "$command" ]]; then
  exit 0
fi

if ! printf '%s' "$command" | grep -qE 'gh pr (create|edit)'; then
  exit 0
fi

if printf '%s' "$command" | grep -qP '<!--\s*declared-direction:\s*\S.*-->'; then
  exit 0
fi

echo "This PR body is missing a <!-- declared-direction: ... --> marker. Add one describing this PR's product-direction intent before opening/editing the PR." >&2
exit 2
