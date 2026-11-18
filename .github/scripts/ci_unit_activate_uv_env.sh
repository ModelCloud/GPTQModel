#!/usr/bin/env bash
set -euo pipefail

test_script="${1:?test_script is required}"
run_id="${2:?run_id is required}"
run_attempt="${3:?run_attempt is required}"

uv pip install pyyaml
torch_version="$(sed -n 's/^torch==\(.*\)+cu132$/\1/p' .github/ci/torch-cu132.txt)"
env_output="$(python .github/scripts/ci_workflow.py resolve-env \
  --group tests \
  --test-name "$test_script" \
  --cuda-version 132 \
  --torch-version "$torch_version" \
  --shell)"
eval "$env_output"

python .github/scripts/ci_workflow.py resolve-test-runtime --test-name "$test_script"
test_cache_key="$test_script"
test_cache_key="${test_cache_key//\//_}"
test_cache_key="${test_cache_key//./_}"
test_cache_key="${test_cache_key//-/_}"

extensions_dir="/tmp/gptqmodel/torch_extensions/${run_id}/${run_attempt}/${ENV_NAME}/${test_cache_key}"
mkdir -p "$extensions_dir"
venv="$RUNNER_TEMP/$ENV_NAME"
uv venv --managed-python --python "$PYTHON_VERSION" "$venv"
"$venv/bin/python" .github/scripts/ci_assert_free_threaded.py --python-only
printf '%s\n' "$venv/bin" >> "$GITHUB_PATH"
printf 'VIRTUAL_ENV=%s\n' "$venv" >> "$GITHUB_ENV"
printf 'GPTQMODEL_TORCH_EXTENSIONS_DIR=%s\n' "$extensions_dir" >> "$GITHUB_ENV"
printf 'TORCH_EXTENSIONS_DIR=%s\n' "$extensions_dir" >> "$GITHUB_ENV"
