#!/usr/bin/env bash
set -euo pipefail

mode="${1:?mode is required}"

case "$mode" in
  sleep)
    for _ in {1..5}; do sleep 5; done
    ;;
  manual)
    runner_host="${RUNNER:?RUNNER is required}"
    run_id="${GITHUB_RUN_ID:?GITHUB_RUN_ID is required}"
    timestamp="$(date +%s%3N)"
    echo "open http://${runner_host}/gpu/ci/confirm?id=${run_id}&timestamp=$timestamp&confirmed=1 to confirm releasing to pypi"
    for _ in {1..5}; do echo "."; done
    echo "click http://${runner_host}/gpu/ci/confirm?id=${run_id}&timestamp=$timestamp&denied=1 to DENY"

    status=-1
    while [[ "$status" -lt 0 ]]; do
      status="$(curl -s "http://${runner_host}/gpu/ci/confirm?id=${run_id}&timestamp=$timestamp")"
      if [[ "$status" == "2" ]]; then
        echo "PYPI_RELEASE_CONFIRMATION=$status" >> "$GITHUB_ENV"
      elif [[ "$status" -lt 0 ]]; then
        sleep 5
      else
        echo "release has been confirmed"
        echo "PYPI_RELEASE_CONFIRMATION=$status" >> "$GITHUB_ENV"
      fi
    done
    ;;
  *)
    echo "Unsupported confirmation mode: $mode" >&2
    exit 1
    ;;
esac
