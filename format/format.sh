#!/bin/bash

cd "$(dirname "$0")" || exit

# force ruff/isort to be same version as setup.py
pip install -U ruff==0.9.5 isort==6.0.0

ruff check ../gptqmodel/models ../gptqmodel/nn_modules ../gptqmodel/quantization ../gptqmodel/utils ../gptqmodel/__init__.py ../examples ../tests ../setup.py --fix --unsafe-fixes
ruff_status=$?

isort -l 119 -e ../

# Exit with the status code of ruff check
exit $ruff_status