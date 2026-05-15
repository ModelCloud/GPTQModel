# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


def _load_build_helper():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_cannoe_ascendc.py"
    spec = importlib.util.spec_from_file_location("build_cannoe_ascendc", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_raw_validator():
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_cannoe_ascendc_raw.py"
    spec = importlib.util.spec_from_file_location("validate_cannoe_ascendc_raw", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ascendc_host_tiler_caps_logical_blocks():
    host_tiler = (
        Path(__file__).resolve().parents[1]
        / "gptqmodel_ext"
        / "cannoe"
        / "ascendc"
        / "op_host"
        / "cannoe_w4_a16_matmul.cpp"
    )
    text = host_tiler.read_text(encoding="utf-8")

    assert "constexpr uint32_t kMaxLogicalBlocks = 8;" in text
    assert "return available_blocks < kMaxLogicalBlocks ? available_blocks : kMaxLogicalBlocks;" in text


def test_raw_validator_finds_embedded_json_after_cann_warning():
    validator = _load_raw_validator()
    text = (
        "[Warning]: tiling struct [TopkTiling] "
        '{"device": 7, "rows": 8, "pass": true}\n'
        "is conflict with one in file topk_tilingdata.h, line 21\n"
    )

    assert validator._last_json_object(text) == {"device": 7, "rows": 8, "pass": True}


def test_raw_validator_custom_timing_stats():
    validator = _load_raw_validator()

    assert validator._timing_stats([]) == {
        "custom_ms_min": None,
        "custom_ms_mean": None,
        "custom_ms_max": None,
    }
    assert validator._timing_stats(
        [
            {"custom_ms": 3.0},
            {"custom_ms": 1.0},
            {"custom_ms": 2.0},
            {"custom_ms": None},
        ]
    ) == {
        "custom_ms_min": 1.0,
        "custom_ms_mean": 2.0,
        "custom_ms_max": 3.0,
    }


def test_raw_validator_batches_one_case_per_device():
    validator = _load_raw_validator()
    cases = [{"case": idx} for idx in range(5)]

    batches = validator._device_case_batches([0, 1], cases)

    assert batches == [
        [(0, {"case": 0}), (1, {"case": 1})],
        [(0, {"case": 2}), (1, {"case": 3})],
        [(0, {"case": 4})],
    ]


def test_raw_validator_quiet_cann_env_defaults_preserve_overrides():
    validator = _load_raw_validator()
    env = {"ASCEND_GLOBAL_LOG_LEVEL": "2"}

    validator._apply_quiet_cann_env(env)

    assert env["ASCEND_GLOBAL_LOG_LEVEL"] == "2"
    assert env["ASCEND_SLOG_PRINT_TO_STDOUT"] == "0"


def test_raw_validator_emit_json_result_to_saved_fd():
    validator = _load_raw_validator()
    read_fd, write_fd = os.pipe()
    try:
        validator._emit_json_result({"device": 0, "pass": True}, write_fd)
        os.close(write_fd)
        write_fd = -1
        payload = os.read(read_fd, 4096).decode("utf-8")
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)

    assert validator._last_json_object(payload) == {"device": 0, "pass": True}


def test_enable_kernel_define_coalesces_experimental_options(tmp_path):
    build_helper = _load_build_helper()
    cmake_path = tmp_path / "op_kernel" / "CMakeLists.txt"
    cmake_path.parent.mkdir()
    cmake_path.write_text(
        "# set custom compile options\n"
        'if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")\n'
        "    add_ops_compile_options(ALL OPTIONS -g -O0)\n"
        "endif()\n"
        "\n"
        "add_kernels_compile()\n"
    )

    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_STAGED_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_CUBE_CONSUMER")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_MIXED_LAUNCH")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_MIXED_AIV_BASELINE")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_CANN9_VECTOR_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_VECOUT_CONSUMER")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_VECOUT_LOCAL_A")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_CONSUMER")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_DIRECT_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_DIRECT_MULTIK")

    text = cmake_path.read_text()
    assert text.count("add_ops_compile_options(ALL OPTIONS -DCANNOE_EXPERIMENTAL_") == 1
    assert (
        "add_ops_compile_options(ALL OPTIONS "
        "-DCANNOE_EXPERIMENTAL_STAGED_DEQUANT=1 "
        "-DCANNOE_EXPERIMENTAL_CUBE_CONSUMER=1 "
        "-DCANNOE_EXPERIMENTAL_MIXED_LAUNCH=1 "
        "-DCANNOE_EXPERIMENTAL_MIXED_AIV_BASELINE=1 "
        "-DCANNOE_EXPERIMENTAL_CANN9_VECTOR_DEQUANT=1 "
        "-DCANNOE_EXPERIMENTAL_VECOUT_CONSUMER=1 "
        "-DCANNOE_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF=1 "
        "-DCANNOE_EXPERIMENTAL_VECOUT_LOCAL_A=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_CONSUMER=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_DIRECT_DEQUANT=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_DIRECT_MULTIK=1)"
    ) in text


def test_enable_kernel_define_handles_cann9_kernel_cmake(tmp_path):
    build_helper = _load_build_helper()
    cmake_path = tmp_path / "op_kernel" / "CMakeLists.txt"
    cmake_path.parent.mkdir()
    cmake_path.write_text(
        "npu_op_kernel_sources(ascendc_kernels\n"
        "    KERNEL_DIR ./\n"
        ")\n"
    )

    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_STAGED_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_CUBE_CONSUMER")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_MIXED_LAUNCH")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_CANN9_VECTOR_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_VECOUT_CONSUMER")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_VECOUT_LOCAL_A")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_CONSUMER")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_DIRECT_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_DIRECT_MULTIK")

    text = cmake_path.read_text()
    assert text.count("npu_op_kernel_options(ascendc_kernels ALL OPTIONS -DCANNOE_EXPERIMENTAL_") == 1
    assert (
        "npu_op_kernel_options(ascendc_kernels ALL OPTIONS "
        "-DCANNOE_EXPERIMENTAL_STAGED_DEQUANT=1 "
        "-DCANNOE_EXPERIMENTAL_CUBE_CONSUMER=1 "
        "-DCANNOE_EXPERIMENTAL_MIXED_LAUNCH=1 "
        "-DCANNOE_EXPERIMENTAL_CANN9_VECTOR_DEQUANT=1 "
        "-DCANNOE_EXPERIMENTAL_VECOUT_CONSUMER=1 "
        "-DCANNOE_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF=1 "
        "-DCANNOE_EXPERIMENTAL_VECOUT_LOCAL_A=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_CONSUMER=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_DIRECT_DEQUANT=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_DIRECT_MULTIK=1)"
    ) in text


def _strategy_args(strategy: str):
    return argparse.Namespace(
        strategy=strategy,
        experimental_staged_dequant=False,
        experimental_cube_consumer=False,
        experimental_mixed_launch=False,
        experimental_mixed_aiv_baseline=False,
        experimental_cann9_vector_dequant=False,
        experimental_vecout_consumer=False,
        experimental_vecout_runtime_handoff=False,
        experimental_vecout_local_a=False,
        experimental_tscm_consumer=False,
        experimental_tscm_runtime_handoff=False,
        experimental_tscm_direct_dequant=False,
        experimental_tscm_direct_multik=False,
    )


def test_public_strategy_vecout_local_a_expands_flags():
    build_helper = _load_build_helper()
    args = _strategy_args("vecout-local-a")

    build_helper._resolve_experimental_flags(args, argparse.ArgumentParser())

    assert args.experimental_staged_dequant
    assert args.experimental_cube_consumer
    assert args.experimental_mixed_launch
    assert args.experimental_cann9_vector_dequant
    assert args.experimental_vecout_consumer
    assert args.experimental_vecout_runtime_handoff
    assert args.experimental_vecout_local_a
    assert not args.experimental_tscm_consumer


def test_public_strategy_tscm_direct_multik_expands_flags():
    build_helper = _load_build_helper()
    args = _strategy_args("tscm-direct-multik")

    build_helper._resolve_experimental_flags(args, argparse.ArgumentParser())

    assert args.experimental_staged_dequant
    assert args.experimental_cube_consumer
    assert args.experimental_mixed_launch
    assert args.experimental_cann9_vector_dequant
    assert args.experimental_tscm_consumer
    assert args.experimental_tscm_runtime_handoff
    assert args.experimental_tscm_direct_dequant
    assert args.experimental_tscm_direct_multik
    assert not args.experimental_vecout_consumer


def test_command_for_python_entrypoint_uses_active_interpreter(tmp_path):
    build_helper = _load_build_helper()
    entrypoint = tmp_path / "msopgen"
    entrypoint.write_text("#!/missing/python\nprint('x')\n")

    command = build_helper._command_for_entrypoint(str(entrypoint))

    assert command[0].endswith("python")
    assert command[1] == str(entrypoint)
