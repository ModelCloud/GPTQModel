#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

: "${CARGO_HOME:?Cargo directory is required}"
: "${RUSTUP_HOME:?Rustup directory is required}"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

installer="$tmpdir/rustup-init"
curl --fail --silent --show-error --location \
  --connect-timeout 15 --max-time 120 --retry 3 --retry-max-time 180 \
  https://static.rust-lang.org/rustup/archive/1.29.0/x86_64-unknown-linux-gnu/rustup-init \
  --output "$installer"
printf '%s  %s\n' 4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10 "$installer" | sha256sum --check
chmod +x "$installer"
"$installer" -y --no-modify-path --profile minimal --default-toolchain 1.94.0
"$CARGO_HOME/bin/rustc" --version
