#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
set -euo pipefail

# Switch CUDA toolkit to match the CUDA version PyTorch was built with on Debian/Ubuntu.

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "auto_switch_cuda_toolkit.sh: This script only supports Linux." >&2
    exit 1
fi

if [[ ! -f /etc/debian_version ]]; then
    echo "auto_switch_cuda_toolkit.sh: This script only supports Debian/Ubuntu systems." >&2
    exit 1
fi

if ! command -v update-alternatives >/dev/null 2>&1; then
    echo "auto_switch_cuda_toolkit.sh: 'update-alternatives' command not found." >&2
    exit 1
fi

if [[ $EUID -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        echo "auto_switch_cuda_toolkit.sh: Run as root or install sudo." >&2
        exit 1
    fi
else
    SUDO=""
fi

python_cmd=$(command -v python3 || true)
if [[ -z "${python_cmd}" ]]; then
    python_cmd=$(command -v python || true)
fi
if [[ -z "${python_cmd}" ]]; then
    echo "auto_switch_cuda_toolkit.sh: Python interpreter not found." >&2
    exit 1
fi

torch_cuda_version=$("${python_cmd}" - <<'PY'
import sys
try:
    import torch
except Exception as exc:  # pragma: no cover - runtime check
    print(f"Failed to import torch: {exc}", file=sys.stderr)
    sys.exit(1)

cuda_version = torch.version.cuda
if not cuda_version:
    print("Torch is not compiled with CUDA support.", file=sys.stderr)
    sys.exit(1)

parts = cuda_version.split('.')
if len(parts) >= 2:
    normalized = f"{parts[0]}.{parts[1]}"
else:
    normalized = cuda_version

print(normalized)
PY
)

if [[ -z "${torch_cuda_version}" ]]; then
    echo "auto_switch_cuda_toolkit.sh: Unable to determine torch CUDA version." >&2
    exit 1
fi

target_version=${torch_cuda_version//$'\r'/$''}
target_version=${target_version//$'\n'/$''}

echo "Detected torch CUDA version: ${target_version}"

config_output=$({ printf '\n'; } | ${SUDO} update-alternatives --config cuda 2>&1 || true)

selection=$(CONFIG_OUTPUT="${config_output}" "${python_cmd}" - "${target_version}" <<'PY'
import os
import pcre as re
import sys

target = sys.argv[1]
data = os.environ.get("CONFIG_OUTPUT", "")
lines = data.splitlines()

candidates = []
for line in lines:
    if "manual mode" not in line or "/cuda-" not in line:
        continue
    stripped = line.lstrip("*").strip()
    if not stripped:
        continue
    parts = stripped.split()
    if len(parts) < 2:
        continue
    sel = parts[0]
    if not sel.isdigit():
        continue
    path = parts[1]
    match = re.search(r"cuda-([0-9.]+)", path)
    if not match:
        continue
    version = match.group(1)
    candidates.append((sel, version, path))

if not candidates:
    print("No CUDA alternatives found in update-alternatives output.", file=sys.stderr)
    sys.exit(1)

split = lambda text: text.split('.')
target_parts = split(target)

chosen = None
for sel, version, _ in candidates:
    if version == target:
        chosen = sel
        break

if chosen is None and len(target_parts) >= 2:
    for sel, version, _ in candidates:
        parts = split(version)
        if len(parts) >= 2 and parts[0] == target_parts[0] and parts[1] == target_parts[1]:
            chosen = sel
            break

if chosen is None and target_parts:
    for sel, version, _ in candidates:
        parts = split(version)
        if parts and parts[0] == target_parts[0]:
            chosen = sel
            break

if chosen is None:
    available = ", ".join(sorted({ver for _, ver, _ in candidates}))
    print(
        f"Could not find CUDA alternative matching torch CUDA {target}. Available versions: {available}",
        file=sys.stderr,
    )
    sys.exit(1)

print(chosen)
PY
)

if [[ -z "${selection}" ]]; then
    echo "auto_switch_cuda_toolkit.sh: Failed to identify matching CUDA alternative." >&2
    exit 1
fi

echo "Selecting CUDA alternative entry: ${selection}"
printf '%s\n' "${selection}" | ${SUDO} update-alternatives --config cuda

current_value=$(${SUDO} update-alternatives --query cuda 2>/dev/null | awk '/^Value:/ {print $2; exit}')
if [[ -n "${current_value}" ]]; then
    echo "CUDA toolkit is now set to: ${current_value}"
else
    echo "CUDA toolkit update complete."
fi
