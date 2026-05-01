# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_build_helper():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_komodo_cann_ascendc.py"
    spec = importlib.util.spec_from_file_location("build_komodo_cann_ascendc", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_raw_validator():
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_komodo_cann_ascendc_raw.py"
    spec = importlib.util.spec_from_file_location("validate_komodo_cann_ascendc_raw", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_validator_finds_embedded_json_after_cann_warning():
    validator = _load_raw_validator()
    text = (
        "[Warning]: tiling struct [TopkTiling] "
        '{"device": 7, "rows": 8, "pass": true}\n'
        "is conflict with one in file topk_tilingdata.h, line 21\n"
    )

    assert validator._last_json_object(text) == {"device": 7, "rows": 8, "pass": True}


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

    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_STAGED_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_CUBE_CONSUMER")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_MIXED_LAUNCH")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_MIXED_AIV_BASELINE")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_CANN9_VECTOR_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_VECOUT_CONSUMER")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_VECOUT_LOCAL_A")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_TSCM_CONSUMER")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_MULTIK")

    text = cmake_path.read_text()
    assert text.count("add_ops_compile_options(ALL OPTIONS -DKOMODO_CANN_EXPERIMENTAL_") == 1
    assert (
        "add_ops_compile_options(ALL OPTIONS "
        "-DKOMODO_CANN_EXPERIMENTAL_STAGED_DEQUANT=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_CUBE_CONSUMER=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_MIXED_LAUNCH=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_MIXED_AIV_BASELINE=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_CANN9_VECTOR_DEQUANT=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_VECOUT_CONSUMER=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_VECOUT_LOCAL_A=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_TSCM_CONSUMER=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_DEQUANT=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_MULTIK=1)"
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

    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_STAGED_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_CUBE_CONSUMER")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_MIXED_LAUNCH")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_CANN9_VECTOR_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_VECOUT_CONSUMER")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_VECOUT_LOCAL_A")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_TSCM_CONSUMER")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_DEQUANT")
    build_helper._enable_kernel_define(tmp_path, "KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_MULTIK")

    text = cmake_path.read_text()
    assert text.count("npu_op_kernel_options(ascendc_kernels ALL OPTIONS -DKOMODO_CANN_EXPERIMENTAL_") == 1
    assert (
        "npu_op_kernel_options(ascendc_kernels ALL OPTIONS "
        "-DKOMODO_CANN_EXPERIMENTAL_STAGED_DEQUANT=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_CUBE_CONSUMER=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_MIXED_LAUNCH=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_CANN9_VECTOR_DEQUANT=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_VECOUT_CONSUMER=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_VECOUT_LOCAL_A=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_TSCM_CONSUMER=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_DEQUANT=1 "
        "-DKOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_MULTIK=1)"
    ) in text


def test_command_for_python_entrypoint_uses_active_interpreter(tmp_path):
    build_helper = _load_build_helper()
    entrypoint = tmp_path / "msopgen"
    entrypoint.write_text("#!/missing/python\nprint('x')\n")

    command = build_helper._command_for_entrypoint(str(entrypoint))

    assert command[0].endswith("python")
    assert command[1] == str(entrypoint)
