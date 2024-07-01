#!/bin/bash
export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0"
INSTALL_COMMAND="pip install -vvv --no-build-isolation ."

check_uv_version() {
    python - <<EOF
from importlib.metadata import version, PackageNotFoundError
try:
    print(version('uv'))
except PackageNotFoundError:
    print('not found')
EOF
}

UV_VERSION=$(check_uv_version)

if [[ $UV_VERSION != "not found" && $(echo -e "$UV_VERSION\n0.1.16" | sort -V | head -n1) == "0.1.16" ]]; then
    INSTALL_COMMAND="uv $INSTALL_COMMAND"
fi

$INSTALL_COMMAND

