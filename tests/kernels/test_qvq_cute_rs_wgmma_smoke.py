# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME

from gptqmodel.utils import machete as machete_utils


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qvq_cute_rs_wgmma_fragment_mapping_on_sm90(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None and CUDA_HOME:
        cuda_nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
        if cuda_nvcc.is_file():
            nvcc = str(cuda_nvcc)
    if nvcc is None:
        pytest.skip("nvcc required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 H100/H200 required")

    source = Path(__file__).with_name("qvq_cute_rs_wgmma_smoke.cu")
    cutlass_root = machete_utils._ensure_cutlass_source()
    executable = tmp_path / "qvq_cute_rs_wgmma_smoke"
    command = [
        nvcc,
        "-std=c++20",
        "-O3",
        "-lineinfo",
        "-arch=sm_90a",
        f"-I{cutlass_root / 'include'}",
        str(source),
        "-o",
        str(executable),
    ]
    compile_result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert compile_result.returncode == 0, compile_result.stderr

    env = os.environ.copy()
    run_result = subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert run_result.returncode == 0, run_result.stderr
    assert "cutlass=4.7.1" in run_result.stdout
    assert "A_values_per_lane=8" in run_result.stdout
    assert "A_rows_per_lane=2" in run_result.stdout
    assert "A_values_per_lane_row=4" in run_result.stdout
    assert "lanes_per_A_row=4" in run_result.stdout


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qvq_cute_rs_wgmma_fp8_primitive_on_h200(tmp_path):
    if torch.cuda.device_count() != 1:
        pytest.skip("exclusive H200 validation device required")
    properties = torch.cuda.get_device_properties(0)
    if torch.cuda.get_device_capability() != (9, 0) or "H200" not in properties.name:
        pytest.skip("H200 required")

    nvcc = shutil.which("nvcc")
    if nvcc is None and CUDA_HOME:
        cuda_nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
        if cuda_nvcc.is_file():
            nvcc = str(cuda_nvcc)
    if nvcc is None:
        pytest.skip("nvcc required")

    source = Path(__file__).with_name("qvq_cute_rs_wgmma_fp8_smoke.cu")
    cutlass_root = machete_utils._ensure_cutlass_source()
    executable = tmp_path / "qvq_cute_rs_wgmma_fp8_smoke"
    command = [
        nvcc,
        "-std=c++20",
        "-O3",
        "-lineinfo",
        "-arch=sm_90a",
        f"-I{cutlass_root / 'include'}",
        str(source),
        "-o",
        str(executable),
    ]
    compile_result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    assert run_result.returncode == 0, run_result.stderr
    assert "device=NVIDIA H200" in run_result.stdout
    assert "mma=m64n16k32" in run_result.stdout
    assert "operands=e4m3xe4m3" in run_result.stdout
    assert "accumulator=f32" in run_result.stdout
    assert "A_values_per_lane=16" in run_result.stdout
