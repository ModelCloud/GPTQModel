#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python3.12}"
repo_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
manifest="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tenstorrent-py312.txt"
tt_index="https://pypi.eng.aws.tenstorrent.com/"

"${python_bin}" - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Tenstorrent runtime requires CPython 3.12, got {sys.version.split()[0]}")
PY

"${python_bin}" -m pip install --upgrade pip setuptools wheel
"${python_bin}" -m pip install \
  --index-url "${tt_index}" \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  -r "${manifest}"
"${python_bin}" -m pip install --no-deps -e "${repo_dir}"

"${python_bin}" - <<'PY'
import torch
import torch_xla

count = int(torch_xla.runtime.addressable_device_count())
device_type = str(torch_xla.runtime.device_type())
print(f"torch={torch.__version__} xla={torch_xla.__version__}")
print(f"device_type={device_type} addressable_devices={count}")
if count < 1:
    raise SystemExit("Tenstorrent PJRT exposed no addressable devices")
PY

printf '%s\n' "Install the Python packages above, then verify the system packages with:"
printf '%s\n' "  dpkg-query -W sfpi tenstorrent-dkms tenstorrent-tools"
printf '%s\n' "  tt-smi -ls"
