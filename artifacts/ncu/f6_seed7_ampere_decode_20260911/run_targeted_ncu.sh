#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32-r8-zml-cuda-v3__qvq-p32__yaqa125x__seed7__20260908/benchmarks/zml_native_fast_abi__zml-native-fast-abi-f6-seed7-sm80-20260910T045313Z/runs/zml-native-fast-abi-f6-seed7-sm80-decode-profile-20260911T011105Z"
MODEL="/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32-r8-zml-cuda-v3__qvq-p32__yaqa125x__seed7__20260908"
LIB="/root/.cache/bazel/_bazel_root/c9182d3aa498eed8a636d81ed7c893e2/execroot/_main/bazel-out/k8-opt/bin/examples/llm/libzml_llama.so"
CUDA_COMPAT="/root/.cache/bazel/_bazel_root/c9182d3aa498eed8a636d81ed7c893e2/execroot/_main/bazel-out/k8-opt/bin/platforms/cuda/source_sandbox/lib/compat/libcuda.so.1"
PYTHON="/root/venv-py3.14t-gil0/bin/python"
GPU_UUID="GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2"
ALLOCATOR_URL="http://127.0.0.1:17351"
SESSION_ID="devin-660bd16c357849a09b7bf0ee52d664c2"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="/root/devin-worker/repos/QvQ${PYTHONPATH:+:${PYTHONPATH}}"
export RUNFILES_DIR="/root/devin-worker/repos/ZML-Ultra/bazel-bin/examples/llm/llama_paged_token_runner.runfiles"
export ZML_PROFILE_DECODE_SAMPLES=1
export ZML_PROFILE_GPU_UUID="${GPU_UUID}"

run_profile() {
    local prefix="$1"
    local kernel_regex="$2"
    shift 2

    "${PYTHON}" -m gpu_allocator.cli \
        --base-url "${ALLOCATOR_URL}" \
        --session-id "${SESSION_ID}" \
        run -n 1 --style uuid -- \
        "${PYTHON}" "${RUN_DIR}/idle_gate_exec.py" \
        --uuid "${GPU_UUID}" \
        --samples 3 \
        --max-attempts 30 \
        --max-memory-mib 1024 \
        -- \
        ncu \
        --target-processes all \
        --preload-library "${CUDA_COMPAT}" \
        --profile-from-start on \
        --kernel-name-base demangled \
        --kernel-name "regex:${kernel_regex}" \
        --launch-count 1 \
        --cache-control none \
        --export="${RUN_DIR}/${prefix}" \
        --force-overwrite \
        --csv \
        --page details \
        "$@" \
        "${PYTHON}" "${RUN_DIR}/zml_native_fast_abi_decode_profile.py" \
        --child \
        --child-kind batch \
        --mode "${prefix}" \
        --model "${MODEL}" \
        --lib "${LIB}" \
        --context 131072 \
        --batch 1 \
        --capacity 8192 \
        --decode-samples 1 \
        --decode-warmup 3 \
        --seed 7308 \
        --output-json "${RUN_DIR}/${prefix}_workload.json" \
        > "${RUN_DIR}/${prefix}.stdout.csv" \
        2> "${RUN_DIR}/${prefix}.stderr.log"
}

case "${1:-}" in
    attention)
        run_profile \
            ncu_attention_full \
            '.*kernel_unified_attention_3d_ptr.*' \
            --section SpeedOfLight \
            --section ComputeWorkloadAnalysis \
            --section MemoryWorkloadAnalysis \
            --section LaunchStats \
            --section Occupancy \
            --section WarpStateStats \
            --section SchedulerStats \
            --section InstructionStats
        ;;
    p32-main)
        run_profile \
            ncu_p32_m1_main_full \
            '.*p32_window_ampere_m1_kernel<\(int\)6, \(int\)1, \(int\)128, \(int\)16, \(int\)2, \(int\)0, \(int\)0, \(int\)0, \(bool\)1>.*' \
            --section SpeedOfLight \
            --section ComputeWorkloadAnalysis \
            --section MemoryWorkloadAnalysis \
            --section LaunchStats \
            --section Occupancy \
            --section WarpStateStats \
            --section SchedulerStats \
            --section InstructionStats
        ;;
    p32-minor)
        run_profile \
            ncu_p32_m1_minor_full \
            '.*p32_window_ampere_m1_kernel<\(int\)5, \(int\)1, \(int\)128, \(int\)16, \(int\)2, \(int\)512, \(int\)32, \(int\)2048, \(bool\)0>.*' \
            --section SpeedOfLight \
            --section ComputeWorkloadAnalysis \
            --section MemoryWorkloadAnalysis \
            --section LaunchStats \
            --section Occupancy \
            --section WarpStateStats \
            --section SchedulerStats \
            --section InstructionStats
        ;;
    rank8)
        run_profile \
            ncu_p32_rank8 \
            '.*p32_rank8_project_kernel.*' \
            --section SpeedOfLight \
            --section LaunchStats \
            --section Occupancy \
            --section WarpStateStats \
            --section SchedulerStats \
            --section InstructionStats
        ;;
    *)
        echo "usage: $0 {attention|p32-main|p32-minor|rank8}" >&2
        exit 2
        ;;
esac
