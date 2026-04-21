#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U build twine setuptools
python -m build --sdist
