#!/usr/bin/env bash
set -euo pipefail

runner_ip="${1:?runner ip is required}"
github_run_id="${2:?github run id is required}"
max_parallel="${3:-}"

if [[ -z "${GITHUB_OUTPUT:-}" ]]; then
  echo "GITHUB_OUTPUT is required" >&2
  exit 1
fi

run_id="$github_run_id"

echo "ip=$runner_ip" >> "$GITHUB_OUTPUT"
echo "ip: $runner_ip"
echo "run_id=$run_id" >> "$GITHUB_OUTPUT"
echo "run_id=$run_id"

if [[ -n "$max_parallel" ]]; then
  max_parallel_json="{\"size\": ${max_parallel:-20}}"
  echo "max-parallel=$max_parallel_json" >> "$GITHUB_OUTPUT"
  echo "max-parallel=$max_parallel_json"
fi
