#!/usr/bin/env bash
set -euo pipefail

pkg_name="${1:?package filename is required}"
run_id="${2:?run_id is required}"

sha256sum "dist/$pkg_name"
target_dir="/opt/dist/$run_id"
mkdir -p "$target_dir"
cp "dist/$pkg_name" "$target_dir/"
echo "UPLOADED=1" >> "$GITHUB_ENV"
