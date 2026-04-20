#!/usr/bin/env bash
set -euo pipefail

test_group="${1:?test_group is required}"
test_script="${2:?test_script is required}"
cuda_version="${3:?cuda_version is required}"
torch_version="${4:?torch_version is required}"
python_version="${5:?python_version is required}"

python -V
which python
which pip || true
echo "uv cache dir: $(uv cache dir)"

echo "setting env... group=${test_group} cuda=${cuda_version} torch=${torch_version} python=${python_version}"
bash /opt/env/init_compiler_no_env.sh "$cuda_version" "$torch_version" "$python_version"
python .github/scripts/ci_deps.py install "$test_script"
python .github/scripts/ci_deps.py uninstall "$test_script"
