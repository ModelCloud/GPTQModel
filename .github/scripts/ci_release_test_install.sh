#!/usr/bin/env bash
set -euo pipefail

pkg_name="${1:?package filename is required}"
venv_dir="${2:-local_uv_env}"

python -m venv "$venv_dir"
source "$venv_dir/bin/activate"
pip install uv
uv pip install "dist/$pkg_name"
