#!/usr/bin/env bash
# Refuse to publish private infrastructure.
#
# This is the second half of the public-repo rule. `gitleaks` catches credentials; this catches
# the identifiers around them: internal hostnames, tailnet addresses, host names and private
# repository paths. It scans every tracked file, so it also covers docs, fixtures and recorded
# test data — the places a scrub forgets.
#
# Exit 1 lists every violation. Exit 0 prints the allowlisted and warn-only hits.
#
# Allowlist: `.ci/private-refs.allow`, one rule per line:
#     path/glob:pattern  # why this line is safe to publish
# `path/glob` is a shell glob matched against the repo-relative path; `pattern` is an ERE matched
# against the whole `git grep` hit (`path:line:text`). A justification is mandatory in review —
# the guard does not enforce it, humans do.
#
# Run it locally before pushing: `bash scripts/ci/no-private-refs.sh`.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 2

# Hard failures. No \b: BSD/macOS regcomp has no word boundary, and this must behave the same on
# the dev box and on the runner.
HARD=(
  '\.internal'
  '100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}'
  'ts\.net'
  'strato-(server|vps)'
  'nonagenticai/gitops-production'
)

# Warnings only: RFC1918 literals are usually a test fixture, but they are still an address.
SOFT=(
  '192\.168\.[0-9]{1,3}\.[0-9]{1,3}'
  '(^|[^0-9.])10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'
)

allowed() {
  # $1 = full hit line; returns 0 when an allowlist rule covers it.
  local hit="$1" file="${1%%:*}" rule glob pat
  [ -f .ci/private-refs.allow ] || return 1
  while IFS= read -r rule || [ -n "$rule" ]; do
    rule="${rule%%#*}"
    rule="$(printf '%s' "$rule" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [ -n "$rule" ] || continue
    glob="${rule%%:*}"
    pat="${rule#*:}"
    pat="$(printf '%s' "$pat" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [ -n "$glob" ] && [ -n "$pat" ] || continue
    # shellcheck disable=SC2254 # glob match is intended
    case "$file" in
      $glob) printf '%s\n' "$hit" | grep -qE -- "$pat" && return 0 ;;
    esac
  done < .ci/private-refs.allow
  return 1
}

hard_args=()
for p in "${HARD[@]}"; do hard_args+=(-e "$p"); done
soft_args=()
for p in "${SOFT[@]}"; do soft_args+=(-e "$p"); done

# This script and its allowlist necessarily spell the patterns out; everything else is scanned.
SCAN=(-- . ':(exclude)scripts/ci/no-private-refs.sh' ':(exclude).ci/private-refs.allow')

fail=0
allow_count=0
while IFS= read -r hit; do
  [ -n "$hit" ] || continue
  if allowed "$hit"; then
    allow_count=$((allow_count + 1))
    echo "allowed: $hit"
  else
    echo "PRIVATE: $hit"
    fail=1
  fi
done < <(git grep -nI -E "${hard_args[@]}" "${SCAN[@]}" || true)

while IFS= read -r hit; do
  [ -n "$hit" ] || continue
  echo "warn (RFC1918): $hit"
done < <(git grep -nI -E "${soft_args[@]}" "${SCAN[@]}" || true)

echo
if [ "$fail" -ne 0 ]; then
  echo "FAILED: private infrastructure is referenced in tracked files (see PRIVATE lines above)."
  echo "Remove it, or add a reviewed rule to .ci/private-refs.allow with a justification."
  exit 1
fi
echo "OK: no private infrastructure references in $(git ls-files | wc -l | tr -d ' ') tracked files ($allow_count allowlisted)."
