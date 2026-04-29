#!/usr/bin/env bash
set -euo pipefail

test_script="${1:?test_script is required}"
cuda_version="${2:?cuda_version is required}"
torch_version="${3:?torch_version is required}"
run_id="${4:?run_id is required}"
run_attempt="${5:?run_attempt is required}"

echo "-- loading unit test's config --"
source /opt/uv/setup_uv_venv.sh unit_test_env
uv pip install requests packaging -U

env_output="$(python3 .github/scripts/ci_workflow.py resolve-env \
  --group tests \
  --test-name "$test_script" \
  --cuda-version "$cuda_version" \
  --torch-version "$torch_version" \
  --shell)"
eval "$env_output"

runtime_output="$(python3 .github/scripts/ci_workflow.py resolve-test-runtime \
  --test-name "$test_script" \
  --shell)"
eval "$runtime_output"

test_cache_key="$test_script"
test_cache_key="${test_cache_key//\//_}"
test_cache_key="${test_cache_key//./_}"
test_cache_key="${test_cache_key//-/_}"

export GPTQMODEL_TORCH_EXTENSIONS_DIR="/tmp/gptqmodel/torch_extensions/${run_id}/${run_attempt}/${ENV_NAME}/${test_cache_key}"
export TORCH_EXTENSIONS_DIR="$GPTQMODEL_TORCH_EXTENSIONS_DIR"
mkdir -p "$GPTQMODEL_TORCH_EXTENSIONS_DIR"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo "GPTQMODEL_TORCH_EXTENSIONS_DIR=$GPTQMODEL_TORCH_EXTENSIONS_DIR" >> "$GITHUB_ENV"
  echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR" >> "$GITHUB_ENV"
fi

echo "-- setting up env --"
echo "env_name: $ENV_NAME"
echo "torch extensions dir: $GPTQMODEL_TORCH_EXTENSIONS_DIR"

echo "-- ls -ahl /opt/uv/venvs before --"
ls -ahl /opt/uv/venvs

target="/opt/uv/tmp/${ENV_NAME}_$(date +%s)"
mkdir -p "$target"
mv /opt/uv/venvs/* "$target"/ || true

echo "-- ls -ahl /opt/uv/venvs after --"
ls -ahl /opt/uv/venvs

/opt/uv/setup_uv_venv.sh "$ENV_NAME"

echo "-- ls -ahl /opt/uv/venvs new --"
ls -ahl /opt/uv/venvs
echo "-- set --"
