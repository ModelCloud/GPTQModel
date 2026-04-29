#!/usr/bin/env bash
set -euo pipefail

python -m pip install uv
uv pip install -U build twine setuptools
python -m build --sdist
