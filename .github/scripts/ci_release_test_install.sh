#!/usr/bin/env bash
set -euo pipefail

pkg_name="${1:?package filename is required}"
venv_dir="${2:-local_uv_env}"

uv venv "$venv_dir"
source "$venv_dir/bin/activate"
uv pip install "dist/$pkg_name" torch
