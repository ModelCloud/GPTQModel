#!/usr/bin/env bash
set -euo pipefail

uv pip install build twine
python -m build --sdist
