#!/usr/bin/env bash
set -euo pipefail

pr_number="${1:-0}"

git config --global --add safe.directory "$(pwd)"

if [[ -z "$pr_number" || "$pr_number" == "0" ]]; then
  exit 0
fi

echo "pr number $pr_number"
git fetch origin "pull/${pr_number}/head:pr-${pr_number}"
git checkout "pr-${pr_number}"
