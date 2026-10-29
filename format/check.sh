#!/bin/bash
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
#
# Strict lint gate. It never modifies files, so its exit status reports the real
# number of remaining violations. Use format/format.sh for local autofixing.

set -o errexit
set -o nounset
set -o pipefail

cd "$(dirname "$0")/.." || exit 1

# Keep the version in lockstep with pyproject.toml's `quality` extra. Installing
# here keeps the gate usable in a bare CI checkout and is a no-op locally.
pip install --disable-pip-version-check --quiet ruff==0.14.2

ruff --version

# Configuration lives in pyproject.toml at the repository root, so these paths
# must stay repository-relative. Never add `--fix` here: a gate that rewrites
# files before checking the exit status can never fail.
#
# Lint the whole `gptqmodel` package instead of an allowlist of subdirectories.
# The allowlist existed only to skip the vendored `gptqmodel/integration` tree,
# which was deleted in #1332; it was never updated afterwards and silently
# skipped every subpackage added since then (`looper`, `adapter`, `eora`,
# `exllamav3`, `hf_kernels`, `hf_minimax_m2`) despite those shipping in the
# wheel. A new subpackage must be covered automatically.
ruff check \
  gptqmodel \
  docs/eora \
  tests \
  setup.py
