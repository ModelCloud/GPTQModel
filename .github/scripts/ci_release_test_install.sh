#!/usr/bin/env bash
set -euo pipefail

pkg_name="${1:?package filename is required}"
venv_dir="${2:-}"
if [[ -z "$venv_dir" ]]; then
  venv_dir="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/gptqmodel-release-install.XXXXXX")"
  trap 'rm -rf "$venv_dir"' EXIT
fi
constraint="$PWD/.github/ci/torch-cu132.txt"

uv venv --python "$(command -v python)" "$venv_dir"
uv pip install --python "$venv_dir/bin/python" \
  --torch-backend cu132 --constraint "$constraint" "dist/$pkg_name"
uv pip check --python "$venv_dir/bin/python"
"$venv_dir/bin/python" -I -X gil=0 .github/scripts/ci_assert_free_threaded.py --installed
