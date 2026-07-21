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

# --body-file points at a file whose *content* carries the marker, not the
# command line itself -- read it so that case is checked too. Portable bash
# parameter expansion (not grep -P, which isn't available under BSD/macOS
# grep, only the interactive shell's aliased ugrep) so this works wherever
# the hook actually runs.
body_file=""
if [[ "$command" == *--body-file* ]]; then
  rest="${command#*--body-file}"
  rest="${rest#"${rest%%[![:space:]]*}"}"  # strip leading whitespace
  first_char="${rest:0:1}"
  if [[ "$first_char" == '"' || "$first_char" == "'" ]]; then
    body_file="${rest#?}"
    body_file="${body_file%%["$first_char"]*}"
  else
    body_file="${rest%%[[:space:]]*}"
  fi
fi

body_content=""
if [[ -n "$body_file" && -f "$body_file" ]]; then
  body_content="$(cat "$body_file" 2>/dev/null || true)"
fi

haystack="$command"$'\n'"$body_content"

# Portable ERE (works under BSD grep, no -P/PCRE needed): require the marker,
# a colon, and at least one non-whitespace character before the closing "-->".
if printf '%s' "$haystack" | grep -qE '<!--[[:space:]]*declared-direction:[[:space:]]*[^[:space:]].*-->'; then
  exit 0
fi

echo "This PR body is missing a <!-- declared-direction: ... --> marker. Add one describing this PR's product-direction intent before opening/editing the PR." >&2
exit 2
