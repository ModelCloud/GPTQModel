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
ruff check \
  gptqmodel/models \
  gptqmodel/nn_modules \
  gptqmodel/quantization \
  gptqmodel/utils \
  gptqmodel/__init__.py \
  docs/eora \
  tests \
  setup.py
