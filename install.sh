#!/bin/bash
export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0"

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --vllm     Install the vllm module"
    echo "  --sglang   Install the sglang module"
    echo "  --bitblas  Install the bitblas module"
    echo "  --lm_eval  Install the lm_eval module"
    echo "  -h, --help Show this help message"
}

check_uv_version() {
    python - <<EOF
from importlib.metadata import version, PackageNotFoundError
try:
    print(version('uv'))
except PackageNotFoundError:
    print('not found')
EOF
}

INSTALL_COMMAND="pip install -vvv --no-build-isolation ."

MODULES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vllm)
            MODULES="${MODULES}vllm,"
            shift
            ;;
        --sglang)
            MODULES="${MODULES}sglang,"
            shift
            ;;
        --bitblas)
            MODULES="${MODULES}bitblas,"
            shift
            ;;
        --lm_eval)
            MODULES="${MODULES}lm_eval,"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

MODULES=${MODULES%,}

if [ -n "$MODULES" ]; then
    INSTALL_COMMAND="$INSTALL_COMMAND[$MODULES]"
fi

UV_VERSION=$(check_uv_version)

if [[ $UV_VERSION != "not found" && $(echo -e "$UV_VERSION\n0.1.16" | sort -V | head -n1) == "0.1.16" ]]; then
    INSTALL_COMMAND="uv $INSTALL_COMMAND"
fi

$INSTALL_COMMAND

