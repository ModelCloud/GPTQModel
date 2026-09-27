#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if (( $# < 1 )); then
    printf 'Usage: %s TEST_PATH | quality-eval ARGS | dtype-audit ARGS | benchmark\n' "$0" >&2
    exit 2
fi

case "$1" in
    tests/models/test_llama3_2_w4afp8.py|tests/models/test_llama3_2_w4a16_reference.py|tests/models/test_w4a_tiny_llama_lifecycle.py|tests/models/test_w4a_replay_stream.py|tests/kernels/test_w4a_stream.py|tests/kernels/test_w4afp8_gb10.py)
        if (( $# != 1 )); then
            printf 'Unexpected arguments after test path.\n' >&2
            exit 2
        fi
        run_kind=pytest ;;
    quality-eval)
        shift
        run_kind=quality ;;
    dtype-audit)
        shift
        run_kind=dtype_audit ;;
    benchmark)
        shift
        if (( $# != 0 )); then
            printf 'The benchmark command takes no arguments.\n' >&2
            exit 2
        fi
        run_kind=benchmark ;;
    *) printf 'Only the W4A tests, quality tools, and benchmark are supported.\n' >&2; exit 2 ;;
esac

cd "$(dirname "$0")/../.."
exec 9>/tmp/gptqmodel-w4a-gb10.lock
if ! flock -n 9; then
    printf 'Another 1B W4A test is already running.\n' >&2
    exit 1
fi

python_bin=${GPTQMODEL_TEST_PYTHON:-python}
if [[ $run_kind == pytest ]]; then
    run_command=("$python_bin" -m pytest -q -o log_cli=false "$1")
elif [[ $run_kind == quality ]]; then
    run_command=("$python_bin" -m tests.models.w4a_quality_regression eval "$@")
elif [[ $run_kind == dtype_audit ]]; then
    run_command=("$python_bin" -m tests.models.w4a_dtype_audit "$@")
elif [[ $run_kind == benchmark ]]; then
    run_command=("$python_bin" tests/benchmark/benchmark_w4a_gb10.py)

fi
budget=$("$python_bin" tests/models/w4a_gb10_memory.py --budget)
read -r hard_limit soft_limit <<<"$budget"
export GPTQMODEL_W4A_MEMORY_MAX_BYTES=$hard_limit
printf 'W4A test memory limits: hard=%s MiB soft=%s MiB; host reserve=2048 MiB\n' \
    "$((hard_limit / 1024 / 1024))" "$((soft_limit / 1024 / 1024))"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
unit="gptqmodel-w4a-gb10-${BASHPID}.scope"
systemd-run --user --scope --unit="${unit%.scope}" --expand-environment=no \
    -p "MemoryHigh=$soft_limit" -p "MemoryMax=$hard_limit" -p MemorySwapMax=0 \
    "${run_command[@]}" &
scope_pid=$!

stop_scope() {
    if kill -0 "$scope_pid" 2>/dev/null; then
        systemctl --user kill --signal=SIGKILL "$unit" >/dev/null 2>&1 || true
        wait "$scope_pid" 2>/dev/null || true
    fi
}
trap stop_scope EXIT INT TERM

# GB10 CUDA allocations can reduce physical RAM without appearing in the
# process scope's memory.current, so also watch the host reserve directly.
while kill -0 "$scope_pid" 2>/dev/null; do
    available_kib=$(awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo)
    if (( available_kib < 2 * 1024 * 1024 )); then
        printf 'Stopping 1B W4A test: host headroom fell below 2 GiB.\n' >&2
        stop_scope
        exit 1
    fi
    root_memory_max=$(</sys/fs/cgroup/memory.max)
    if [[ $root_memory_max != max ]]; then
        root_memory_current=$(</sys/fs/cgroup/memory.current)
        if (( root_memory_max - root_memory_current < 2 * 1024 * 1024 * 1024 )); then
            printf 'Stopping 1B W4A test: root cgroup headroom fell below 2 GiB.\n' >&2
            stop_scope
            exit 1
        fi
    fi
    sleep 0.1
done

wait "$scope_pid"
