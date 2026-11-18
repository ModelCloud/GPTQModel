#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

destination="${1:?uv destination is required}"
version=0.13.8
archive=uv-x86_64-unknown-linux-gnu.tar.gz
sha256=5acfd9177cfecf0bf87578f059728215ccc884388988f55f7a65041b57387d91
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

curl --fail --silent --show-error --location \
  --connect-timeout 15 --max-time 120 --retry 3 --retry-max-time 180 \
  "https://github.com/astral-sh/uv/releases/download/$version/$archive" \
  --output "$tmpdir/$archive"
(
  cd "$tmpdir"
  printf '%s  %s\n' "$sha256" "$archive" | sha256sum --check
)
tar --no-same-owner -xzf "$tmpdir/$archive" -C "$tmpdir"
mkdir -p "$destination"
install -m 755 "$tmpdir/${archive%.tar.gz}/uv" "$destination/uv"
"$destination/uv" --version
