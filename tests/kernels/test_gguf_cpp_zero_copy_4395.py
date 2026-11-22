# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#4395: GGUF_CPP_CPU must read the packed weight in place, not copy it.

The stock CPU path allocated a backend buffer and copied the whole packed
weight on every forward. The zero-copy path wraps the existing storage with
``ggml_backend_cpu_buffer_from_ptr``; an unaligned or rejected layout must
fall back to the copy path and stay bitwise identical.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_cpp import (
    _GGML_CPU_WEIGHT_ALIGNMENT,
    GGUFCppKernel,
    _get_ggml_bridge,
    _GGMLBridge,
)


IN_FEATURES = 256
OUT_FEATURES = 128


def _build_kernels(bits: str = "q4_k_s") -> tuple[GGUFTorchLinear, GGUFCppKernel]:
    torch.manual_seed(4395)
    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.float32).eval()
    torch_kernel = GGUFTorchLinear(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )
    torch_kernel.pack_original(linear, scales=None, zeros=None)
    cpp_kernel = GGUFCppKernel(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
        exact_matmul=False,
    )
    cpp_kernel.load_state_dict(torch_kernel.state_dict(), strict=True)
    cpp_kernel.eval()
    cpp_kernel.post_init()
    return torch_kernel, cpp_kernel


def _require_bridge() -> _GGMLBridge:
    GGUFCppKernel.cached_validate_once.cache_clear()
    ok, err = GGUFCppKernel.cached_validate_once()
    if not ok:
        pytest.skip(f"llama-cpp-python unavailable: {err}")
    return _get_ggml_bridge()


def _force_unaligned_qweight(kernel: GGUFCppKernel) -> torch.Tensor:
    """Replace the packed buffer with a one-byte-offset view of new storage."""

    qweight = kernel.qweight
    storage = torch.empty(qweight.numel() + 1, dtype=torch.uint8)
    storage[1:] = qweight.reshape(-1)
    kernel.qweight = storage[1:].view(qweight.shape)
    assert kernel.qweight.data_ptr() % _GGML_CPU_WEIGHT_ALIGNMENT != 0
    return qweight


def test_gguf_cpp_cpu_reads_packed_weight_in_place(monkeypatch):
    bridge = _require_bridge()
    _torch_kernel, kernel = _build_kernels()
    assert kernel.qweight.data_ptr() % _GGML_CPU_WEIGHT_ALIGNMENT == 0

    flags: list[bool] = []
    original = _GGMLBridge._run_quantized_matmul

    def spy(self, **kwargs):
        flags.append(bool(kwargs.get("zero_copy_weight")))
        return original(self, **kwargs)

    monkeypatch.setattr(_GGMLBridge, "_run_quantized_matmul", spy)

    x = torch.randn(3, IN_FEATURES, dtype=torch.float32)
    with torch.inference_mode():
        aligned = kernel(x)

    assert flags == [True]
    assert torch.isfinite(aligned).all()

    # The copy path must produce exactly the same bytes for the same input.
    with torch.inference_mode():
        copied = bridge.quantized_matmul_cpu(
            qweight=kernel.qweight,
            x=x,
            gguf_tensor_qtype=kernel.gguf_tensor_qtype,
            padded_in_features=kernel.padded_in_features,
            out_features=OUT_FEATURES,
        )
    assert torch.equal(aligned, copied)


def test_gguf_cpp_cpu_unaligned_weight_falls_back_bitwise_equal(monkeypatch):
    _require_bridge()
    _torch_kernel, kernel = _build_kernels()

    x = torch.randn(5, IN_FEATURES, dtype=torch.float32)
    with torch.inference_mode():
        expected = kernel(x)

    _force_unaligned_qweight(kernel)
    flags: list[bool] = []
    original = _GGMLBridge._run_quantized_matmul

    def spy(self, **kwargs):
        flags.append(bool(kwargs.get("zero_copy_weight")))
        return original(self, **kwargs)

    monkeypatch.setattr(_GGMLBridge, "_run_quantized_matmul", spy)
    with torch.inference_mode():
        actual = kernel(x)

    # The unaligned buffer must never be wrapped; only the copy path runs.
    assert flags == [False]
    assert torch.equal(actual, expected)


def test_gguf_cpp_cpu_missing_zero_copy_symbols_falls_back(monkeypatch):
    """An older llama.cpp build without the symbols keeps the copy path."""

    bridge = _require_bridge()
    _torch_kernel, kernel = _build_kernels()
    x = torch.randn(4, IN_FEATURES, dtype=torch.float32)
    with torch.inference_mode():
        expected = kernel(x)

    flags: list[bool] = []
    original = _GGMLBridge._run_quantized_matmul

    def spy(self, **kwargs):
        flags.append(bool(kwargs.get("zero_copy_weight")))
        return original(self, **kwargs)

    monkeypatch.setattr(bridge, "_cpu_buffer_from_ptr", None)
    monkeypatch.setattr(_GGMLBridge, "_run_quantized_matmul", spy)
    with torch.inference_mode():
        actual = kernel(x)

    assert flags == [False]
    assert torch.equal(actual, expected)


def test_gguf_cpp_post_init_aligns_mmap_loaded_weight(tmp_path):
    _require_bridge()
    torch_kernel, _kernel = _build_kernels()
    packed = torch_kernel.state_dict()["qweight"]

    # A safetensors mmap view is not guaranteed to be 32-byte aligned. Publish
    # the packed weight through a file and bind the mmap tensor directly, the
    # way a loader can assign checkpoint storage.
    source = tmp_path / "model.safetensors"
    save_file({"qweight": packed}, str(source))
    from safetensors import safe_open

    with safe_open(str(source), framework="pt", device="cpu") as handle:
        mmap_weight = handle.get_tensor("qweight")
        assert mmap_weight.data_ptr() % _GGML_CPU_WEIGHT_ALIGNMENT != 0
        expected = mmap_weight.clone()

        kernel = GGUFCppKernel(
            bits="q4_k_s",
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            bias=False,
            register_buffers=True,
            exact_matmul=False,
        )
        kernel.qweight = mmap_weight
        kernel.post_init()

        assert kernel.qweight.data_ptr() % _GGML_CPU_WEIGHT_ALIGNMENT == 0
        assert torch.equal(kernel.qweight, expected)
        with torch.inference_mode():
            out = kernel(torch.randn(2, IN_FEATURES, dtype=torch.float32))
        assert torch.isfinite(out).all()
