#!/usr/bin/env bash
set -euo pipefail

packages=(defuser pypcre tokenicer logbar evalution)
urls=(
  "${DEFUSER_GIT_URL:?DEFUSER_GIT_URL is required}"
  "${PYPCRE_GIT_URL:?PYPCRE_GIT_URL is required}"
  "${TOKENICER_GIT_URL:?TOKENICER_GIT_URL is required}"
  "${LOGBAR_GIT_URL:?LOGBAR_GIT_URL is required}"
  "${EVALUTION_GIT_URL:?EVALUTION_GIT_URL is required}"
)

echo "===== remove ModelCloud git deps if installed ====="
uv pip uninstall -y "${packages[@]}" || true

echo "===== install ModelCloud git deps ====="
printf '  - %s\n' "${urls[@]}"
uv pip install "${urls[@]}"
