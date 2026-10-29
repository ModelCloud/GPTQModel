#!/bin/bash
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
#
# Local autofixer. The CI gate is format/check.sh, which never rewrites files.

set -o errexit
set -o nounset
set -o pipefail

cd "$(dirname "$0")/.." || exit 1

# Keep the version in lockstep with pyproject.toml's `quality` extra and with
# format/check.sh.
pip install --disable-pip-version-check --quiet ruff==0.14.2

# Configuration lives in pyproject.toml at the repository root, so these paths
# must stay repository-relative.
#
# Lint the whole `gptqmodel` package, matching format/check.sh: the old
# allowlist of subdirectories silently skipped the subpackages added after
# `gptqmodel/integration` was removed, so their violations were never fixed.
#
# `--unsafe-fixes` is intentional: this is the local autofixer and it is allowed
# to rewrite code the safe pass refuses to touch. `ruff check --fix` exits
# non-zero when anything is left unfixable, so under `errexit` the exit status
# always reports real leftovers; the previous script swallowed it with
# `ruff_status=$?` and therefore always exited 0.
ruff check \
  gptqmodel \
  docs/eora \
  tests \
  setup.py \
  --fix \
  --unsafe-fixes
