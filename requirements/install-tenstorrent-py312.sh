#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python3.12}"
repo_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
manifest="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tenstorrent-py312.txt"
tt_index="https://pypi.eng.aws.tenstorrent.com/"

# This validated environment pins torch 2.11.0+cpu, which is below the patch for
# GHSA-rrmf-rvhw-rf47 (CVE-2025-3000). Tenstorrent's PJRT plugin has no newer
# compatible Torch/XLA stack, so the supported path is the documented startup
# gate override. It is exported here for the verification below and must also be
# set in every shell that imports GPT-QModel from this environment; see
# docs/torch-runtime-safety-gate.md.
export GPTQ_BYPASS_SAFETY_CHECKS=1

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

# The override is load-bearing: without it the gate must reject this runtime. If
# this import starts succeeding, the pinned manifest has moved past the advisory
# range and the override above should be removed.
if GPTQ_BYPASS_SAFETY_CHECKS= "${python_bin}" -c 'import gptqmodel' >/dev/null 2>&1; then
  printf '%s\n' "warning: GPT-QModel imported this pinned Torch without the safety override." >&2
  printf '%s\n' "The pinned manifest is no longer below the advisory range; remove the override." >&2
fi

"${python_bin}" - <<'PY'
import gptqmodel
import torch
import torch_xla

count = int(torch_xla.runtime.addressable_device_count())
device_type = str(torch_xla.runtime.device_type())
print(f"gptqmodel={gptqmodel.__version__}")
print(f"torch={torch.__version__} xla={torch_xla.__version__}")
print(f"device_type={device_type} addressable_devices={count}")
if count < 1:
    raise SystemExit("Tenstorrent PJRT exposed no addressable devices")
PY

printf '%s\n' ""
printf '%s\n' "This environment pins a Torch release below the startup-gate patch boundary."
printf '%s\n' "Set the documented override in every shell that imports GPT-QModel here:"
printf '%s\n' "  export GPTQ_BYPASS_SAFETY_CHECKS=1"
printf '%s\n' ""
printf '%s\n' "Install the Python packages above, then verify the system packages with:"
printf '%s\n' "  dpkg-query -W sfpi tenstorrent-dkms tenstorrent-tools"
printf '%s\n' "  tt-smi -ls"
