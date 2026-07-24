#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
venv_path="${CUDAQ_VENV_PATH:-${repo_root}/venv/cudaq}"
python_bin="${CUDAQ_PYTHON_BIN:-/usr/bin/python3.13}"
cudaq_version="${CUDAQ_VERSION:-0.15.0}"

if [[ ! -x "${python_bin}" ]]; then
    echo "CUDA-Q setup requires regular CPython 3.13 at ${python_bin}." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "CUDA-Q setup requires uv on PATH." >&2
    exit 1
fi

if [[ ! -x "${venv_path}/bin/python" ]]; then
    uv venv --seed --python "${python_bin}" "${venv_path}"
fi

uv pip install --python "${venv_path}/bin/python" "cudaq==${cudaq_version}"
"${venv_path}/bin/python" -m pip show cudaq | sed -n '1,3p'

echo "CUDA-Q environment ready at ${venv_path}"
