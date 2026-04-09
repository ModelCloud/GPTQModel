# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

import gptqmodel.exllamav3.ext as exllamav3_ext_module
from gptqmodel.utils import cpp as cpp_module


class _FakeExtension:
    def __init__(self, ops: dict[str, object], *, available: bool = True, error: str = ""):
        self._ops = ops
        self._available = available
        self._error = error

    def load(self) -> bool:
        return self._available

    def last_error_message(self) -> str:
        return self._error

    def op(self, name: str):
        return self._ops[name]


def test_exllamav3_bc_linear_wrapper_uses_jit_op(monkeypatch):
    calls = {}

    def fake_run(trellis, suh, svh, K, bias, mcg, mul1, xh, x, y):
        calls["bc_linear_exl3_run"] = {
            "trellis_shape": tuple(trellis.shape),
            "suh_shape": tuple(suh.shape),
            "svh_shape": tuple(svh.shape),
            "K": K,
            "bias_is_none": bias is None,
            "mcg": mcg,
            "mul1": mul1,
            "xh_shape": tuple(xh.shape),
            "x_shape": tuple(x.shape),
            "y_shape": tuple(y.shape),
        }

    monkeypatch.setattr(
        exllamav3_ext_module,
        "_EXLLAMAV3_TORCH_OPS_EXTENSION",
        _FakeExtension({"bc_linear_exl3_run": fake_run}),
    )

    wrapper = exllamav3_ext_module.exllamav3_ext.BC_LinearEXL3(
        trellis=torch.zeros((1, 4, 32), dtype=torch.int16),
        suh=torch.zeros((128,), dtype=torch.float16),
        svh=torch.zeros((64,), dtype=torch.float16),
        K=2,
        bias=None,
        mcg=True,
        mul1=False,
        xh=torch.zeros((1, 128), dtype=torch.float16),
    )

    wrapper.run(
        torch.zeros((3, 128), dtype=torch.float16),
        torch.zeros((3, 64), dtype=torch.float16),
    )

    assert calls["bc_linear_exl3_run"] == {
        "trellis_shape": (1, 4, 32),
        "suh_shape": (128,),
        "svh_shape": (64,),
        "K": 2,
        "bias_is_none": True,
        "mcg": True,
        "mul1": False,
        "xh_shape": (1, 128),
        "x_shape": (3, 128),
        "y_shape": (3, 64),
    }


def test_exllamav3_quantize_tiles_uses_jit_op(monkeypatch):
    calls = {}

    def fake_quantize_tiles(input_tiles, output_tiles, output_indices, temp_costs, temp_edges, K, mcg, mul1):
        calls["quantize_tiles"] = {
            "input_shape": tuple(input_tiles.shape),
            "output_shape": tuple(output_tiles.shape),
            "indices_dtype": output_indices.dtype,
            "temp_costs_shape": tuple(temp_costs.shape),
            "temp_edges_shape": tuple(temp_edges.shape),
            "K": K,
            "mcg": mcg,
            "mul1": mul1,
        }
        output_tiles.fill_(1.0)
        output_indices.fill_(7)

    monkeypatch.setattr(
        exllamav3_ext_module,
        "_EXLLAMAV3_TORCH_OPS_EXTENSION",
        _FakeExtension({"quantize_tiles": fake_quantize_tiles}),
    )

    input_tiles = torch.zeros((2, 256), dtype=torch.float32)
    output_tiles = torch.zeros_like(input_tiles)
    output_indices = torch.zeros((2, 256), dtype=torch.int16)
    temp_costs = torch.zeros((2, 2, 16384), dtype=torch.float16)
    temp_edges = torch.zeros((2, 256, 16384), dtype=torch.int16)

    exllamav3_ext_module.exllamav3_ext.quantize_tiles(
        input_tiles,
        output_tiles,
        output_indices,
        temp_costs,
        temp_edges,
        2,
        False,
        True,
    )

    assert calls["quantize_tiles"] == {
        "input_shape": (2, 256),
        "output_shape": (2, 256),
        "indices_dtype": torch.int16,
        "temp_costs_shape": (2, 2, 16384),
        "temp_edges_shape": (2, 256, 16384),
        "K": 2,
        "mcg": False,
        "mul1": True,
    }
    assert torch.all(output_tiles == 1.0)
    assert torch.all(output_indices == 7)


def test_exllamav3_bc_linear_wrapper_surfaces_jit_error(monkeypatch):
    monkeypatch.setattr(
        exllamav3_ext_module,
        "_EXLLAMAV3_TORCH_OPS_EXTENSION",
        _FakeExtension({}, available=False, error="missing exllamav3 jit ops"),
    )

    with pytest.raises(ModuleNotFoundError, match="missing exllamav3 jit ops"):
        exllamav3_ext_module.exllamav3_ext.BC_LinearEXL3(
            trellis=torch.zeros((1, 4, 32), dtype=torch.int16),
            suh=torch.zeros((128,), dtype=torch.float16),
            svh=torch.zeros((64,), dtype=torch.float16),
            K=2,
            bias=None,
            mcg=False,
            mul1=False,
            xh=torch.zeros((1, 128), dtype=torch.float16),
        )


def test_exllamav3_include_paths_use_wheel_headers_when_local_cuda_is_incomplete(monkeypatch, tmp_path):
    source_root = tmp_path / "exllamav3"
    local_cuda_include = tmp_path / "local_cuda_include"
    wheel_cuda_include = tmp_path / "wheel_cuda_include"
    source_root.mkdir()
    local_cuda_include.mkdir()
    wheel_cuda_include.mkdir()
    (wheel_cuda_include / "cusparse.h").write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(exllamav3_ext_module, "_source_root", lambda: source_root)
    monkeypatch.setattr(
        cpp_module,
        "detected_local_cuda_include_paths",
        lambda: [str(local_cuda_include)],
    )
    monkeypatch.setattr(
        cpp_module,
        "detected_cuda_wheel_include_paths",
        lambda: [str(wheel_cuda_include)],
    )

    include_paths = exllamav3_ext_module._exllamav3_include_paths()

    assert include_paths[0] == str(source_root)
    assert str(wheel_cuda_include) in include_paths


def test_exllamav3_include_paths_skip_wheel_headers_when_local_cuda_has_required_headers(monkeypatch, tmp_path):
    source_root = tmp_path / "exllamav3"
    local_cuda_include = tmp_path / "local_cuda_include"
    wheel_cuda_include = tmp_path / "wheel_cuda_include"
    source_root.mkdir()
    local_cuda_include.mkdir()
    wheel_cuda_include.mkdir()
    (local_cuda_include / "cusparse.h").write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(exllamav3_ext_module, "_source_root", lambda: source_root)
    monkeypatch.setattr(
        cpp_module,
        "detected_local_cuda_include_paths",
        lambda: [str(local_cuda_include)],
    )
    monkeypatch.setattr(
        cpp_module,
        "detected_cuda_wheel_include_paths",
        lambda: [str(wheel_cuda_include)],
    )

    include_paths = exllamav3_ext_module._exllamav3_include_paths()

    assert include_paths == [str(source_root)]
