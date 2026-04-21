#!/usr/bin/env bash
set -euo pipefail

dist_dir="${1:-dist}"

ls -ahl "$dist_dir"
pkg="$(ls -t "$dist_dir"/*.tar.gz | head -n 1 | xargs basename)"
echo "PKG_NAME=$pkg" >> "$GITHUB_ENV"
twine check "$dist_dir/$pkg"
