#!/bin/bash

cd "$(dirname "$0")" || exit

# force ruff to be same version as setup.py
pip install -U ruff==0.13.0

ruff check ../gptqmodel/models ../gptqmodel/nn_modules ../gptqmodel/quantization ../gptqmodel/utils ../gptqmodel/__init__.py ../examples ../tests ../setup.py --fix --unsafe-fixes
ruff_status=$?

# Exit with the status code of ruff check
exit $ruff_status
