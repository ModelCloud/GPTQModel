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


def test_ascendc_host_tiler_disables_staging_for_unsupported_aic_tscm_shapes():
    host_tiler = (
        Path(__file__).resolve().parents[1]
        / "gptqmodel_ext"
        / "cannoe"
        / "ascendc"
        / "op_host"
        / "cannoe_w4_a16_matmul.cpp"
    )
    text = host_tiler.read_text(encoding="utf-8")

    assert "bool enable_staged_dequant = true;" in text
    assert "bool aic_tscm_supported" in text
    assert "CANNOE_EXPERIMENTAL_AIC_TSCM_UNSAFE_RUNTIME" in text
    assert "enable_staged_dequant = false;" in text
    assert "if (enable_staged_dequant) {" in text


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
    assert validator._timing_stats([{"custom_ms": float("nan")}, {"custom_ms": 4.0}]) == {
        "custom_ms_min": 4.0,
        "custom_ms_mean": 4.0,
        "custom_ms_max": 4.0,
    }


def test_raw_validator_groups_nonfinite_indices_by_output_tile():
    validator = _load_raw_validator()

    assert validator._nonfinite_tile_counts([0, 1, 255, 256, 511, 512, 1024], n=5120, base_n=-256) == {
        "0": 3,
        "1": 2,
        "2": 1,
        "4": 1,
    }


def test_raw_validator_expands_sparse_active_k_and_values():
    validator = _load_raw_validator()

    case = {"active_k_count": 4, "active_value_mode": "ramp"}

    assert validator._expand_active_k(case, 16) == [0, 4, 8, 12]
    assert validator._active_values(case, 4) == [-1.0, 0.625, 0.125, -0.375]
    assert validator._preview_values([1, 2, 3]) == [1, 2, 3]
    assert validator._preview_values(list(range(20))) == list(range(8)) + list(range(12, 20))


def test_raw_validator_records_failed_json_metrics():
    validator = _load_raw_validator()

    class Proc:
        returncode = 2

        def poll(self):
            return self.returncode

    results = []
    failures = []
    validator._record_worker_result(
        device=0,
        case={"rows": 1, "k": 17408, "n": 5120},
        proc=Proc(),
        stdout=(
            '{"custom_ms": 907.8, "diff_nonfinite_count": 5120, '
            '"max_abs": null, "mean_abs": null, "pass": false}'
        ),
        stderr="",
        results=results,
        failures=failures,
    )

    assert len(results) == 1
    assert len(failures) == 1
    assert results[0]["worker_returncode"] == 2
    assert results[0]["diff_nonfinite_count"] == 5120
    assert failures[0]["result"]["custom_ms"] == 907.8


def test_raw_validator_qwen_down_preset_targets_large_projection():
    validator = _load_raw_validator()

    assert list(validator.CASE_PRESETS["qwen3_27b_down"]) == [
        {"rows": 1, "k": 17408, "n": 5120, "group": 32, "seed": 2701}
    ]


def test_raw_validator_qwen_down_onehot_preset_targets_accumulation_boundaries():
    validator = _load_raw_validator()

    cases = list(validator.CASE_PRESETS["qwen3_27b_down_onehot"])

    assert [case["one_hot_k"] for case in cases] == [0, 31, 32, 127, 128, 8703, 8704, 17407]
    assert {case["rows"] for case in cases} == {1}
    assert {case["k"] for case in cases} == {17408}
    assert {case["n"] for case in cases} == {5120}
    assert {case["group"] for case in cases} == {32}


def test_raw_validator_qwen_down_pairwise_preset_targets_cross_tile_accumulation():
    validator = _load_raw_validator()

    cases = list(validator.CASE_PRESETS["qwen3_27b_down_pairwise"])

    assert [tuple(case["active_k"]) for case in cases] == [
        (0, 31),
        (0, 32),
        (0, 127),
        (0, 128),
        (127, 128),
        (8703, 8704),
        (0, 8704),
        (8704, 17407),
    ]
    assert {case["rows"] for case in cases} == {1}
    assert {case["k"] for case in cases} == {17408}
    assert {case["n"] for case in cases} == {5120}
    assert {case["group"] for case in cases} == {32}


def test_raw_validator_qwen_down_sparse_preset_targets_active_k_density():
    validator = _load_raw_validator()

    cases = list(validator.CASE_PRESETS["qwen3_27b_down_sparse"])

    assert [case["active_k_count"] for case in cases] == [4, 8, 16, 32, 64, 128, 256, 512]
    assert {case["active_value_mode"] for case in cases} == {"ramp"}
    assert {case["rows"] for case in cases} == {1}
    assert {case["k"] for case in cases} == {17408}
    assert {case["n"] for case in cases} == {5120}
    assert {case["group"] for case in cases} == {32}


def test_raw_validator_qwen_down_random_scale_preset_targets_activation_distribution():
    validator = _load_raw_validator()

    cases = list(validator.CASE_PRESETS["qwen3_27b_down_random_scale"])

    assert [case["input_scale"] for case in cases] == [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    assert {case["rows"] for case in cases} == {1}
    assert {case["k"] for case in cases} == {17408}
    assert {case["n"] for case in cases} == {5120}
    assert {case["group"] for case in cases} == {32}


def test_raw_validator_batches_one_case_per_device():
    validator = _load_raw_validator()
    cases = [{"case": idx} for idx in range(5)]

    batches = validator._device_case_batches([0, 1], cases)

    assert batches == [
        [(0, {"case": 0}), (1, {"case": 1})],
        [(0, {"case": 2}), (1, {"case": 3})],
        [(0, {"case": 4})],
    ]


def test_raw_validator_planner_tiles_match_validated_cannoe_policy():
    validator = _load_raw_validator()
    args = argparse.Namespace(planner_tiles=True, base_m=16, base_n=-256, base_k=-128)

    fast_case = validator._case_payload(
        {"rows": 4, "k": 512, "n": 512, "group": 32, "seed": 1},
        args,
    )
    unsafe_narrow_case = validator._case_payload(
        {"rows": 8, "k": 1024, "n": 512, "group": 32, "seed": 2},
        args,
    )

    assert fast_case["base_m"] == 16
    assert fast_case["base_n"] == -128
    assert fast_case["base_k"] == -128
    assert unsafe_narrow_case["base_n"] == -256
    assert unsafe_narrow_case["base_k"] == -128


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
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_VECOUT_TILE_CAST_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_VECOUT_TILE_FILL_DIAGNOSTIC")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_VECOUT_INPLACE_CAST_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_VECOUT_CAST_SCRATCH_PROBE")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_INT4_LANE_DIAGNOSTIC")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_CONSUMER")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_DIRECT_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_DIRECT_MULTIK")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_ITERATE_GETC_DIAGNOSTIC")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_TBUF_HANDOFF")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_AIC_TSCM_HANDOFF")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_AIC_TSCM_ZERO_B_DIAGNOSTIC")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_AIC_TSCM_PATH_DIAGNOSTIC")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_AIC_TSCM_UNSAFE_RUNTIME")

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
        "-DCANNOE_EXPERIMENTAL_VECOUT_TILE_CAST_DEQUANT=1 "
        "-DCANNOE_EXPERIMENTAL_VECOUT_TILE_FILL_DIAGNOSTIC=1 "
        "-DCANNOE_EXPERIMENTAL_VECOUT_INPLACE_CAST_DEQUANT=1 "
        "-DCANNOE_EXPERIMENTAL_VECOUT_CAST_SCRATCH_PROBE=1 "
        "-DCANNOE_EXPERIMENTAL_INT4_LANE_DIAGNOSTIC=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_CONSUMER=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_DIRECT_DEQUANT=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_DIRECT_MULTIK=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_ITERATE_GETC_DIAGNOSTIC=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_TBUF_HANDOFF=1 "
        "-DCANNOE_EXPERIMENTAL_AIC_TSCM_HANDOFF=1 "
        "-DCANNOE_EXPERIMENTAL_AIC_TSCM_ZERO_B_DIAGNOSTIC=1 "
        "-DCANNOE_EXPERIMENTAL_AIC_TSCM_PATH_DIAGNOSTIC=1 "
        "-DCANNOE_EXPERIMENTAL_AIC_TSCM_UNSAFE_RUNTIME=1)"
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
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_VECOUT_TILE_CAST_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_VECOUT_TILE_FILL_DIAGNOSTIC")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_VECOUT_INPLACE_CAST_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_VECOUT_CAST_SCRATCH_PROBE")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_INT4_LANE_DIAGNOSTIC")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_CONSUMER")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_DIRECT_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_DIRECT_MULTIK")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_ITERATE_GETC_DIAGNOSTIC")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_TSCM_TBUF_HANDOFF")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_AIC_TSCM_HANDOFF")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_AIC_TSCM_ZERO_B_DIAGNOSTIC")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_AIC_TSCM_PATH_DIAGNOSTIC")
    build_helper._enable_kernel_define(tmp_path, "CANNOE_EXPERIMENTAL_AIC_TSCM_UNSAFE_RUNTIME")

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
        "-DCANNOE_EXPERIMENTAL_VECOUT_TILE_CAST_DEQUANT=1 "
        "-DCANNOE_EXPERIMENTAL_VECOUT_TILE_FILL_DIAGNOSTIC=1 "
        "-DCANNOE_EXPERIMENTAL_VECOUT_INPLACE_CAST_DEQUANT=1 "
        "-DCANNOE_EXPERIMENTAL_VECOUT_CAST_SCRATCH_PROBE=1 "
        "-DCANNOE_EXPERIMENTAL_INT4_LANE_DIAGNOSTIC=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_CONSUMER=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_DIRECT_DEQUANT=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_DIRECT_MULTIK=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_ITERATE_GETC_DIAGNOSTIC=1 "
        "-DCANNOE_EXPERIMENTAL_TSCM_TBUF_HANDOFF=1 "
        "-DCANNOE_EXPERIMENTAL_AIC_TSCM_HANDOFF=1 "
        "-DCANNOE_EXPERIMENTAL_AIC_TSCM_ZERO_B_DIAGNOSTIC=1 "
        "-DCANNOE_EXPERIMENTAL_AIC_TSCM_PATH_DIAGNOSTIC=1 "
        "-DCANNOE_EXPERIMENTAL_AIC_TSCM_UNSAFE_RUNTIME=1)"
    ) in text


def test_enable_host_define_coalesces_experimental_options(tmp_path):
    build_helper = _load_build_helper()
    cmake_path = tmp_path / "op_host" / "CMakeLists.txt"
    cmake_path.parent.mkdir()
    cmake_path.write_text(
        "add_compile_options(-DCANNOE_EXPERIMENTAL_MIXED_LAUNCH=1)\n"
        "\n"
        "aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)\n"
    )

    build_helper._enable_host_define(tmp_path, "CANNOE_EXPERIMENTAL_AIC_TSCM_HANDOFF")
    build_helper._enable_host_define(tmp_path, "CANNOE_EXPERIMENTAL_AIC_TSCM_HANDOFF")

    text = cmake_path.read_text()
    assert text.count("-DCANNOE_EXPERIMENTAL_AIC_TSCM_HANDOFF=1") == 1
    assert (
        "add_compile_options("
        "-DCANNOE_EXPERIMENTAL_MIXED_LAUNCH=1 "
        "-DCANNOE_EXPERIMENTAL_AIC_TSCM_HANDOFF=1)"
    ) in text


def test_aic_tscm_handoff_enables_matching_host_define():
    build_helper = _load_build_helper()
    script_text = Path(build_helper.__file__).read_text(encoding="utf-8")

    assert '_enable_kernel_define(output, "CANNOE_EXPERIMENTAL_AIC_TSCM_HANDOFF")' in script_text
    assert '_enable_host_define(output, "CANNOE_EXPERIMENTAL_AIC_TSCM_HANDOFF")' in script_text
    assert '_enable_host_define(output, "CANNOE_EXPERIMENTAL_AIC_TSCM_UNSAFE_RUNTIME")' in script_text


def _strategy_args(strategy: str):
    return argparse.Namespace(
        strategy=strategy,
        experimental_staged_dequant=False,
        experimental_cube_consumer=False,
        experimental_mixed_launch=False,
        experimental_mixed_aiv_baseline=False,
        experimental_mixed_entry_diagnostic=False,
        experimental_mixed_matmul_reg_diagnostic=False,
        experimental_cann9_vector_dequant=False,
        experimental_vecout_consumer=False,
        experimental_vecout_runtime_handoff=False,
        experimental_vecout_local_a=False,
        experimental_vecout_tile_cast_dequant=False,
        experimental_vecout_tile_fill_diagnostic=False,
        experimental_vecout_inplace_cast_dequant=False,
        experimental_vecout_cast_scratch_probe=False,
        experimental_int4_lane_diagnostic=False,
        experimental_tscm_consumer=False,
        experimental_tscm_runtime_handoff=False,
        experimental_tscm_direct_dequant=False,
        experimental_tscm_direct_multik=False,
        experimental_tscm_local_a=False,
        experimental_tscm_iterate_getc_diagnostic=False,
        experimental_tscm_tbuf_handoff=False,
        experimental_aic_tscm_handoff=False,
        experimental_aic_tscm_zero_b_diagnostic=False,
        experimental_aic_tscm_path_diagnostic=False,
        experimental_aic_tscm_index_diagnostic=False,
        experimental_aic_tscm_syncall_diagnostic=False,
        experimental_aic_tscm_ping_diagnostic=False,
        experimental_aic_tscm_unsafe_runtime=False,
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


def test_public_strategy_tscm_iterate_getc_diagnostic_expands_local_a_flags():
    build_helper = _load_build_helper()
    args = _strategy_args("tscm-iterate-getc-diagnostic")

    build_helper._resolve_experimental_flags(args, argparse.ArgumentParser())

    assert args.experimental_staged_dequant
    assert args.experimental_cube_consumer
    assert args.experimental_mixed_launch
    assert args.experimental_cann9_vector_dequant
    assert args.experimental_tscm_consumer
    assert args.experimental_tscm_runtime_handoff
    assert args.experimental_tscm_direct_dequant
    assert args.experimental_tscm_direct_multik
    assert args.experimental_tscm_local_a
    assert args.experimental_tscm_iterate_getc_diagnostic
    assert not args.experimental_vecout_consumer


def test_int4_lane_diagnostic_expands_minimal_vector_flags():
    build_helper = _load_build_helper()
    args = _strategy_args("manual")
    args.experimental_int4_lane_diagnostic = True

    build_helper._resolve_experimental_flags(args, argparse.ArgumentParser())

    assert args.experimental_staged_dequant
    assert args.experimental_cann9_vector_dequant
    assert args.experimental_int4_lane_diagnostic
    assert not args.experimental_cube_consumer
    assert not args.experimental_mixed_launch


def test_tscm_tbuf_handoff_expands_runtime_flags():
    build_helper = _load_build_helper()
    args = _strategy_args("manual")
    args.experimental_tscm_tbuf_handoff = True

    build_helper._resolve_experimental_flags(args, argparse.ArgumentParser())

    assert args.experimental_staged_dequant
    assert args.experimental_cube_consumer
    assert args.experimental_mixed_launch
    assert args.experimental_tscm_consumer
    assert args.experimental_tscm_runtime_handoff
    assert args.experimental_tscm_tbuf_handoff


def test_aic_tscm_handoff_expands_direct_multik_flags():
    build_helper = _load_build_helper()
    args = _strategy_args("aic-tscm-handoff")

    build_helper._resolve_experimental_flags(args, argparse.ArgumentParser())

    assert args.experimental_staged_dequant
    assert args.experimental_cube_consumer
    assert args.experimental_mixed_launch
    assert args.experimental_cann9_vector_dequant
    assert args.experimental_tscm_consumer
    assert args.experimental_tscm_runtime_handoff
    assert args.experimental_tscm_direct_dequant
    assert args.experimental_tscm_direct_multik
    assert args.experimental_tscm_tbuf_handoff
    assert args.experimental_aic_tscm_handoff


def test_aic_tscm_zero_b_diagnostic_expands_handoff_flags():
    build_helper = _load_build_helper()
    args = _strategy_args("aic-tscm-zero-b-diagnostic")

    build_helper._resolve_experimental_flags(args, argparse.ArgumentParser())

    assert args.experimental_staged_dequant
    assert args.experimental_cube_consumer
    assert args.experimental_mixed_launch
    assert args.experimental_cann9_vector_dequant
    assert args.experimental_tscm_consumer
    assert args.experimental_tscm_runtime_handoff
    assert args.experimental_tscm_direct_dequant
    assert args.experimental_tscm_direct_multik
    assert args.experimental_tscm_tbuf_handoff
    assert args.experimental_aic_tscm_handoff
    assert args.experimental_aic_tscm_zero_b_diagnostic


def test_aic_tscm_path_diagnostic_expands_handoff_flags():
    build_helper = _load_build_helper()
    args = _strategy_args("aic-tscm-path-diagnostic")

    build_helper._resolve_experimental_flags(args, argparse.ArgumentParser())

    assert args.experimental_staged_dequant
    assert args.experimental_cube_consumer
    assert args.experimental_mixed_launch
    assert args.experimental_cann9_vector_dequant
    assert args.experimental_tscm_consumer
    assert args.experimental_tscm_runtime_handoff
    assert args.experimental_tscm_direct_dequant
    assert args.experimental_tscm_direct_multik
    assert args.experimental_tscm_tbuf_handoff
    assert args.experimental_aic_tscm_handoff
    assert args.experimental_aic_tscm_path_diagnostic


def test_aic_tscm_unsafe_runtime_expands_handoff_flags():
    build_helper = _load_build_helper()
    args = _strategy_args("aic-tscm-unsafe-runtime")

    build_helper._resolve_experimental_flags(args, argparse.ArgumentParser())

    assert args.experimental_staged_dequant
    assert args.experimental_cube_consumer
    assert args.experimental_mixed_launch
    assert args.experimental_cann9_vector_dequant
    assert args.experimental_tscm_consumer
    assert args.experimental_tscm_runtime_handoff
    assert args.experimental_tscm_direct_dequant
    assert args.experimental_tscm_direct_multik
    assert args.experimental_tscm_tbuf_handoff
    assert args.experimental_aic_tscm_handoff
    assert args.experimental_aic_tscm_unsafe_runtime


def test_vecout_tile_cast_dequant_expands_vecout_local_a_flags():
    build_helper = _load_build_helper()
    args = _strategy_args("manual")
    args.experimental_vecout_tile_cast_dequant = True

    build_helper._resolve_experimental_flags(args, argparse.ArgumentParser())

    assert args.experimental_staged_dequant
    assert args.experimental_cube_consumer
    assert args.experimental_mixed_launch
    assert args.experimental_cann9_vector_dequant
    assert args.experimental_vecout_consumer
    assert args.experimental_vecout_runtime_handoff
    assert args.experimental_vecout_local_a
    assert args.experimental_vecout_tile_cast_dequant


def test_vecout_tile_fill_diagnostic_expands_tile_cast_flags():
    build_helper = _load_build_helper()
    args = _strategy_args("manual")
    args.experimental_vecout_tile_fill_diagnostic = True

    build_helper._resolve_experimental_flags(args, argparse.ArgumentParser())

    assert args.experimental_staged_dequant
    assert args.experimental_cube_consumer
    assert args.experimental_mixed_launch
    assert args.experimental_cann9_vector_dequant
    assert args.experimental_vecout_consumer
    assert args.experimental_vecout_runtime_handoff
    assert args.experimental_vecout_local_a
    assert args.experimental_vecout_tile_cast_dequant
    assert args.experimental_vecout_tile_fill_diagnostic


def test_vecout_inplace_cast_dequant_expands_vecout_local_a_flags():
    build_helper = _load_build_helper()
    args = _strategy_args("manual")
    args.experimental_vecout_inplace_cast_dequant = True

    build_helper._resolve_experimental_flags(args, argparse.ArgumentParser())

    assert args.experimental_staged_dequant
    assert args.experimental_cube_consumer
    assert args.experimental_mixed_launch
    assert args.experimental_cann9_vector_dequant
    assert args.experimental_vecout_consumer
    assert args.experimental_vecout_runtime_handoff
    assert args.experimental_vecout_local_a
    assert args.experimental_vecout_inplace_cast_dequant
    assert not args.experimental_vecout_tile_cast_dequant


def test_vecout_cast_scratch_probe_expands_tile_cast_flags():
    build_helper = _load_build_helper()
    args = _strategy_args("manual")
    args.experimental_vecout_cast_scratch_probe = True

    build_helper._resolve_experimental_flags(args, argparse.ArgumentParser())

    assert args.experimental_staged_dequant
    assert args.experimental_cube_consumer
    assert args.experimental_mixed_launch
    assert args.experimental_cann9_vector_dequant
    assert args.experimental_vecout_consumer
    assert args.experimental_vecout_runtime_handoff
    assert args.experimental_vecout_local_a
    assert args.experimental_vecout_tile_cast_dequant
    assert args.experimental_vecout_cast_scratch_probe


def test_command_for_python_entrypoint_uses_active_interpreter(tmp_path):
    build_helper = _load_build_helper()
    entrypoint = tmp_path / "msopgen"
    entrypoint.write_text("#!/missing/python\nprint('x')\n")

    command = build_helper._command_for_entrypoint(str(entrypoint))

    assert command[0].endswith("python")
    assert command[1] == str(entrypoint)
