#!/bin/bash

cd "$(dirname "$0")" || exit

# force ruff/isort to be same version as setup.py
pip install -U ruff==0.4.9 isort==5.13.2

ruff check ../gptqmodel ../examples ../tests ../setup.py --fix
ruff_status=$?

isort -l 119 -e ../

# Exit with the status code of ruff check
exit $ruff_status