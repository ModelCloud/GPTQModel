# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import gptqmodel.models.base as base_module
import gptqmodel.quantization.nf12 as nf12_quantization
from gptqmodel.looper import gptq_processor
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.nn_modules.hooked_linear import HookedLinear
from gptqmodel.nn_modules.qlinear.nf12 import NF12Linear
from gptqmodel.quantization import AutoDtypeCast, QuantizeConfig
from gptqmodel.utils import auto_dtype_cast
from gptqmodel.utils.model import move_module_to_cpu_pinned


def _bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return torch.equal(
        actual.contiguous().view(torch.int16),
        expected.contiguous().view(torch.int16),
    )


def test_auto_dtype_cast_defaults_to_auto_nf12_and_roundtrips():
    config = QuantizeConfig(offload_to_disk=False)

    assert config.auto_dtype_cast is AutoDtypeCast.AUTO_NF12
    payload = config.to_dict()
    assert payload["auto_dtype_cast"] == "auto_nf12"
    assert QuantizeConfig.from_quant_config(payload).auto_dtype_cast is AutoDtypeCast.AUTO_NF12

    disabled = QuantizeConfig(auto_dtype_cast="off", offload_to_disk=False)
    assert disabled.auto_dtype_cast is AutoDtypeCast.OFF


def test_auto_dtype_cast_rejects_unknown_policy():
    with pytest.raises(ValueError, match="auto_dtype_cast"):
        QuantizeConfig(auto_dtype_cast="fp8", offload_to_disk=False)


def test_fp16_eligibility_requires_an_exact_bf16_roundtrip(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)

    exact = torch.full((16, 64), 0.5, dtype=torch.float16)
    exact.view(torch.int16)[0, 0] = -32768  # -0.0
    assert auto_dtype_cast.fp16_is_bf16_exact(exact)
    exact_module = nn.Linear(64, 16, bias=False, dtype=torch.float16)
    with torch.no_grad():
        exact_module.weight.copy_(exact)
    assert auto_dtype_cast.eligible_source_module(exact_module, "cpu") is not None

    inexact = torch.full((16, 64), 1.001, dtype=torch.float16)
    assert not auto_dtype_cast.fp16_is_bf16_exact(inexact)
    inexact_module = nn.Linear(64, 16, bias=False, dtype=torch.float16)
    with torch.no_grad():
        inexact_module.weight.copy_(inexact)
    assert auto_dtype_cast.eligible_source_module(inexact_module, "cpu") is None


def test_only_supported_matrix_modules_are_eligible(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)

    assert (
        auto_dtype_cast.eligible_source_module(nn.Linear(64, 16, bias=True, dtype=torch.bfloat16), "cpu").kind
        == "linear"
    )
    assert (
        auto_dtype_cast.eligible_source_module(nn.Embedding(16, 64, dtype=torch.bfloat16), "cpu").kind == "embedding"
    )
    assert auto_dtype_cast.eligible_source_module(nn.LayerNorm(64, dtype=torch.bfloat16), "cpu") is None
    assert auto_dtype_cast.eligible_source_module(nn.Linear(65, 16, bias=False, dtype=torch.bfloat16), "cpu") is None


def test_hooked_linear_cast_keeps_calibration_hook(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    source = HookedLinear(64, 16)
    source.weight = nn.Parameter(torch.randn(16, 64, dtype=torch.bfloat16) * 0.02)
    source.bias = nn.Parameter(torch.randn(16, dtype=torch.bfloat16))
    calls = []
    source.forward_hook = lambda module, args, output: calls.append((module, output))

    spec = auto_dtype_cast.eligible_source_module(source, "cpu")
    packed = auto_dtype_cast.cast_source_module(source, spec)
    actual = packed(torch.randn(2, 64, dtype=torch.bfloat16))

    assert spec.kind == "hooked_linear"
    assert not auto_dtype_cast.AutoNF12HookedLinear.SUPPORTS_BACKEND_SELECTION
    assert len(calls) == 1
    assert calls[0][0] is packed
    assert actual.dtype is torch.bfloat16
    assert actual.shape == (2, 16)


def test_linear_cast_and_restore_preserve_bf16_bits(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    source = nn.Linear(64, 16, bias=True, dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.copy_(torch.randn_like(source.weight) * 0.02)
        source.bias.copy_(torch.randn_like(source.bias))

    spec = auto_dtype_cast.eligible_source_module(source, "cpu")
    packed = auto_dtype_cast.cast_source_module(source, spec)
    restored = auto_dtype_cast.restore_source_module(packed, spec, device="cpu")

    assert _bits_equal(restored.weight, source.weight)
    assert _bits_equal(restored.bias, source.bias)
    assert "weight" not in packed.state_dict()


def test_fp16_exact_cast_and_restore_preserves_source_dtype_and_bits(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    source = nn.Linear(64, 16, bias=False, dtype=torch.float16)
    with torch.no_grad():
        source.weight.copy_(torch.full_like(source.weight, 0.5))
        source.weight.view(torch.int16)[0, 0] = -32768

    spec = auto_dtype_cast.eligible_source_module(source, "cpu")
    packed = auto_dtype_cast.cast_source_module(source, spec)
    restored = auto_dtype_cast.restore_source_module(packed, spec, device="cpu")

    assert restored.weight.dtype is torch.float16
    assert _bits_equal(restored.weight, source.weight)


def test_embedding_cast_and_restore_preserve_bf16_bits(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    source = nn.Embedding(16, 64, padding_idx=0, dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.copy_(torch.randn_like(source.weight) * 0.02)
        source.weight[0].zero_()

    spec = auto_dtype_cast.eligible_source_module(source, "cpu")
    packed = auto_dtype_cast.cast_source_module(source, spec)
    ids = torch.tensor([[0, 1, 15], [2, 3, 4]])
    actual = packed(ids)
    restored = auto_dtype_cast.restore_source_module(packed, spec, device="cpu")

    assert actual.dtype is torch.bfloat16
    assert _bits_equal(actual, source(ids))
    assert restored.padding_idx == 0
    assert _bits_equal(restored.weight, source.weight)


def test_exact_fp16_embedding_keeps_source_dtype(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    source = nn.Embedding(16, 64, dtype=torch.float16)
    with torch.no_grad():
        source.weight.copy_(torch.full_like(source.weight, 0.5))
        source.weight.view(torch.int16)[0, 0] = -32768

    spec = auto_dtype_cast.eligible_source_module(source, "cpu")
    packed = auto_dtype_cast.cast_source_module(source, spec)
    ids = torch.tensor([0, 1, 15])
    actual = packed(ids)

    assert actual.dtype is torch.float16
    assert _bits_equal(actual, source(ids))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None or torch.cuda.get_device_capability(0)[0] < 8,
    reason="requires NVIDIA SM80 or newer",
)
def test_cuda_embedding_uses_only_packed_storage(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)

    import gptqmodel_nf12

    native = gptqmodel_nf12.available()
    source = nn.Embedding(32, 128, dtype=torch.bfloat16, device="cuda:0")
    with torch.no_grad():
        source.weight.copy_(torch.randn_like(source.weight) * 0.02)
        source.weight[7, 0] = -0.0
        source.weight[15, 127] = 2.0**-20
    ids = torch.tensor([[-1, 0, 7], [15, 31, 32]], dtype=torch.int64, device="cuda:0")

    spec = auto_dtype_cast.eligible_source_module(source, "cuda:0")
    packed = auto_dtype_cast.cast_source_module(source, spec)
    actual = packed(ids)
    oracle = torch.zeros_like(actual)
    valid = (ids >= 0) & (ids < source.num_embeddings)
    oracle[valid] = source(ids[valid])

    if native:
        assert packed.row_major
        assert {name for name, _ in packed.named_buffers()} == {
            "nf12_row_packed",
            "nf12_row_escapes",
            "nf12_exp_base",
        }
        assert packed.nf12_row_packed.nbytes + packed.nf12_row_escapes.nbytes < source.weight.nbytes
    else:
        assert not packed.row_major
        assert {name for name, _ in packed.named_buffers()} == {
            "nf12_packed",
            "nf12_escapes",
            "nf12_exp_base",
        }
        assert packed.nf12_packed.nbytes + packed.nf12_escapes.nbytes < source.weight.nbytes
    assert _bits_equal(actual, oracle)

    restored = auto_dtype_cast.restore_source_module(packed, spec, device="cuda:0")
    assert restored.weight.dtype is torch.bfloat16
    assert _bits_equal(restored.weight, source.weight)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None or torch.cuda.get_device_capability(0)[0] < 8,
    reason="requires NVIDIA SM80 or newer",
)
def test_cuda_embedding_falls_back_when_native_nf12_is_unavailable(monkeypatch):
    import gptqmodel_nf12
    import gptqmodel_nf12._library as nf12_library

    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_library, "_native_available", lambda: False)
    assert not gptqmodel_nf12.available()

    source = nn.Embedding(32, 128, dtype=torch.bfloat16, device="cuda:0")
    with torch.no_grad():
        source.weight.copy_(torch.randn_like(source.weight) * 0.02)
        source.weight[5, 3] = -0.0
    ids = torch.tensor([[0, 5, 31]], dtype=torch.int64, device="cuda:0")

    spec = auto_dtype_cast.eligible_source_module(source, "cuda:0")
    packed = auto_dtype_cast.cast_source_module(source, spec)

    assert not packed.row_major
    assert {name for name, _ in packed.named_buffers()} == {
        "nf12_packed",
        "nf12_escapes",
        "nf12_exp_base",
    }
    assert not hasattr(packed, "weight")
    assert _bits_equal(packed(ids), source(ids))

    restored = auto_dtype_cast.restore_source_module(packed, spec, device="cuda:0")
    assert _bits_equal(restored.weight, source.weight)


def test_source_build_available_reports_missing_tools(monkeypatch):
    from gptqmodel_nf12 import _jit

    monkeypatch.setattr(_jit, "_architectures", lambda: [89])
    monkeypatch.setattr(_jit.shutil, "which", lambda _name: None)
    assert not _jit.source_build_available()

    monkeypatch.setattr(_jit.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert _jit.source_build_available()

    monkeypatch.setattr(_jit, "_architectures", lambda: [70])
    assert not _jit.source_build_available()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None or torch.cuda.get_device_capability(0)[0] < 8,
    reason="requires NVIDIA SM80 or newer",
)
def test_cuda_embedding_cast_survives_missing_build_tools(tmp_path):
    """The issue repro: a supported CUDA device with no toolkit must not abort."""
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import torch
from torch import nn
from gptqmodel.utils.auto_dtype_cast import eligible_source_module, cast_source_module

source = nn.Embedding(16, 64, device="cuda", dtype=torch.bfloat16)
spec = eligible_source_module(source, torch.device("cuda"))
assert spec is not None
packed = cast_source_module(source, spec)
ids = torch.tensor([[0, 3, 15]], device="cuda")
out = packed(ids)
assert out.dtype is torch.bfloat16 and tuple(out.shape) == (1, 3, 64)
print("CAST_OK")
"""
    env = dict(os.environ)
    env["PATH"] = "/usr/bin:/bin"
    env["CUDA_HOME"] = "/nonexistent"
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["PYTHONPATH"] = str(repo_root)
    env.pop("CUDACXX", None)

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )

    assert result.returncode == 0, result.stderr
    assert "CAST_OK" in result.stdout


def test_forward_role_packs_and_quant_source_role_restores_dense(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(64, 16, bias=True, dtype=torch.bfloat16)

    model = Wrapper()
    with torch.no_grad():
        model.linear.weight.copy_(torch.randn_like(model.linear.weight) * 0.02)
        model.linear.bias.copy_(torch.randn_like(model.linear.bias))
    expected_weight = model.linear.weight.detach().clone()
    expected_bias = model.linear.bias.detach().clone()

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = model
    harness.quantize_config = QuantizeConfig(offload_to_disk=False)
    harness._turtle_lock = threading.RLock()
    harness.turtle_model = None
    harness.auto_dtype_cast_events = []
    named = NamedModule(model.linear, name="linear", full_name="linear", layer_index=0)

    packed = harness._maybe_auto_dtype_cast_forward_module(
        model.linear, device=torch.device("cpu"), named_module=named
    )
    assert model.linear is packed
    assert "weight" not in packed.state_dict()
    packed_bytes = sum(t.numel() * t.element_size() for t in packed.buffers())

    restored = harness.shell_module_materialize(
        target_submodule=packed,
        device=torch.device("cpu"),
        role="quant_source",
        named_module=named,
    )
    assert model.linear is restored
    assert type(restored) is nn.Linear
    assert _bits_equal(restored.weight, expected_weight)
    assert _bits_equal(restored.bias, expected_bias)
    assert "auto_dtype_cast_source_spec" not in named.state
    assert all(not tensor.numel() for tensor in packed.buffers())
    assert harness.auto_dtype_cast_events == [
        {
            "module": "linear",
            "device": "cpu",
            "kind": "linear",
            "source_dtype": "bfloat16",
            "source_bytes": expected_weight.numel() * 2 + expected_bias.numel() * 2,
            "packed_bytes": packed_bytes,
        }
    ]


@pytest.mark.parametrize("kind", ["linear", "hooked_linear", "embedding"])
def test_quant_source_releases_packed_storage_for_every_source_kind(monkeypatch, kind):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    if kind == "embedding":
        source = nn.Embedding(16, 64, dtype=torch.bfloat16)
    elif kind == "hooked_linear":
        source = HookedLinear.from_linear(
            nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
        )
    else:
        source = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.copy_(torch.randn_like(source.weight) * 0.02)
    expected = source.weight.detach().clone()

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.source = source

    model = Wrapper()
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = model
    harness.quantize_config = QuantizeConfig(offload_to_disk=False)
    harness._turtle_lock = threading.RLock()
    harness.turtle_model = None
    harness.auto_dtype_cast_events = []
    named = NamedModule(model.source, name="source", full_name="source", layer_index=0)

    packed = harness._maybe_auto_dtype_cast_forward_module(
        model.source,
        device=torch.device("cpu"),
        named_module=named,
    )
    assert any(tensor.numel() for tensor in packed.buffers())

    restored = harness.shell_module_materialize(
        target_submodule=packed,
        device=torch.device("cpu"),
        role="quant_source",
        named_module=named,
    )

    assert model.source is restored
    assert _bits_equal(restored.weight, expected)
    assert all(not tensor.numel() for tensor in packed.buffers())


def test_quant_source_keeps_packed_source_when_eora_needs_the_diff(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    source = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.copy_(torch.randn_like(source.weight) * 0.02)

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = source

    model = Wrapper()
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = model
    harness.quantize_config = QuantizeConfig(offload_to_disk=False)
    harness._turtle_lock = threading.RLock()
    harness.turtle_model = None
    harness.auto_dtype_cast_events = []
    harness._needs_w_wq_diff = True
    named = NamedModule(model.proj, name="proj", full_name="proj", layer_index=0)

    packed = harness._maybe_auto_dtype_cast_forward_module(
        model.proj, device=torch.device("cpu"), named_module=named
    )
    harness.shell_module_materialize(
        target_submodule=packed,
        device=torch.device("cpu"),
        role="quant_source",
        named_module=named,
    )

    assert named.state["auto_dtype_cast_packed_source"] is packed
    assert any(tensor.numel() for tensor in packed.buffers())


def test_w_wq_diff_falls_back_and_releases_packed_source(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    source = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.copy_(torch.randn_like(source.weight) * 0.02)
    spec = auto_dtype_cast.eligible_source_module(source, "cpu")
    assert spec is not None
    packed = auto_dtype_cast.cast_source_module(source, spec)
    restored = auto_dtype_cast.restore_source_module(packed, spec, device="cpu")

    named = NamedModule(restored, name="proj", full_name="proj", layer_index=0)
    named.state["auto_dtype_cast_packed_source"] = packed
    wq = torch.randn_like(restored.weight, dtype=torch.float32)
    processor = gptq_processor.GPTQProcessor.__new__(gptq_processor.GPTQProcessor)
    processor.calculate_w_wq_diff = True

    diff = processor._w_wq_diff(named, wq)

    expected = restored.weight.detach().to(torch.float32) - wq
    assert torch.equal(diff, expected)
    assert "auto_dtype_cast_packed_source" not in named.state
    assert all(not tensor.numel() for tensor in packed.buffers())


def test_w_wq_diff_prefers_the_fused_native_path(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    source = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.copy_(torch.randn_like(source.weight) * 0.02)
    spec = auto_dtype_cast.eligible_source_module(source, "cpu")
    assert spec is not None
    packed = auto_dtype_cast.cast_source_module(source, spec)
    restored = auto_dtype_cast.restore_source_module(packed, spec, device="cpu")
    named = NamedModule(restored, name="proj", full_name="proj", layer_index=0)
    named.state["auto_dtype_cast_packed_source"] = packed
    wq = torch.randn_like(restored.weight, dtype=torch.float32)
    sentinel = torch.zeros_like(wq)
    seen = []

    def fake_fused(candidate, candidate_wq):
        seen.append((candidate, candidate_wq))
        return sentinel

    monkeypatch.setattr(auto_dtype_cast, "fused_w_minus_wq", fake_fused)
    processor = gptq_processor.GPTQProcessor.__new__(gptq_processor.GPTQProcessor)
    processor.calculate_w_wq_diff = True

    diff = processor._w_wq_diff(named, wq)

    assert diff is sentinel
    assert seen == [(packed, wq)]
    assert "auto_dtype_cast_packed_source" not in named.state
    assert all(not tensor.numel() for tensor in packed.buffers())


def test_gptq_process_routes_the_eora_diff_through_the_fused_helper():
    """The processor must call the fused helper, not the dense expression."""

    source = inspect.getsource(gptq_processor.GPTQProcessor.process)

    assert "self._w_wq_diff(module, wq)" in source
    assert "module.weight.data.to(dtype=torch.float32) - wq.to(dtype=torch.float32)" not in source


def test_eora_forward_does_not_repack_the_restored_source(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    linear = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.quantize_config = QuantizeConfig(offload_to_disk=False)
    named = NamedModule(linear, name="proj", full_name="proj", layer_index=0)
    named.state["auto_dtype_cast_forward_disabled"] = True

    result = harness._maybe_auto_dtype_cast_forward_module(
        linear, device=torch.device("cpu"), named_module=named
    )

    assert result is linear
    assert "auto_dtype_cast_source_spec" not in named.state


@pytest.mark.skipif(
    not torch.cuda.is_available() or not auto_dtype_cast.nf12_supported_device("cuda:0"),
    reason="requires a native NF12 accelerator",
)
def test_fused_w_minus_wq_matches_the_dense_difference():
    linear = nn.Linear(64, 16, bias=False, device="cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(torch.randn_like(linear.weight) * 0.02)
    spec = auto_dtype_cast.eligible_source_module(linear, "cuda:0")
    assert spec is not None
    packed = auto_dtype_cast.cast_source_module(linear, spec)
    wq = torch.randn((16, 64), device="cuda:0", dtype=torch.float32)

    fused = auto_dtype_cast.fused_w_minus_wq(packed, wq)

    assert fused is not None
    assert fused.dtype is torch.float32
    expected = packed.dequantize_weight(device="cuda:0", dtype=torch.float32) - wq
    assert torch.equal(fused.view(torch.int32), expected.view(torch.int32))


def test_fused_w_minus_wq_declines_when_the_source_is_not_packed():
    linear = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
    wq = torch.randn((16, 64), dtype=torch.float32)

    assert auto_dtype_cast.fused_w_minus_wq(linear, wq) is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not auto_dtype_cast.nf12_supported_device("cuda:0"),
    reason="requires a native NF12 accelerator",
)
def test_native_gpu_encoding_keeps_packed_tensors_on_source_device():
    weight = torch.randn(16, 64, device="cuda:0", dtype=torch.bfloat16)

    result = nf12_quantization.quantize_nf12_weight(weight)

    assert result.packed.device == weight.device
    assert result.escapes.device == weight.device


def test_native_floatx_forward_plan_is_left_untouched(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)

    model = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = model
    harness.quantize_config = QuantizeConfig(offload_to_disk=False)
    harness.quantize_config._native_floatx_forward_plan = {
        "source_format": "fp8",
        "mode": "native",
    }
    named = NamedModule(model, name="", full_name="linear", layer_index=0)

    result = harness._maybe_auto_dtype_cast_forward_module(model, device=torch.device("cpu"), named_module=named)

    assert result is model
    assert "auto_dtype_cast_source_spec" not in named.state


def test_decoded_floatx_forward_plan_uses_nf12(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)

    model = Wrapper()
    source = model.linear
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = model
    harness.quantize_config = QuantizeConfig(offload_to_disk=False)
    harness.quantize_config._native_floatx_forward_plan = {
        "source_format": "fp8",
        "mode": "decode",
    }
    named = NamedModule(source, name="linear", full_name="linear", layer_index=0)

    result = harness._maybe_auto_dtype_cast_forward_module(source, device=torch.device("cpu"), named_module=named)

    assert result is not source
    assert model.linear is result
    assert named.state["auto_dtype_cast_source_spec"].kind == "linear"


def test_processor_task_releases_dense_storage_then_rebinds_exact_source(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    source = nn.Linear(64, 16, bias=True, dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.copy_(torch.randn_like(source.weight) * 0.02)
        source.bias.copy_(torch.randn_like(source.bias))
    expected_weight = source.weight.detach().clone()
    expected_bias = source.bias.detach().clone()
    spec = auto_dtype_cast.eligible_source_module(source, "cpu")
    packed = auto_dtype_cast.cast_source_module(source, spec)
    named = NamedModule(source, name="linear", full_name="linear", layer_index=0)
    named.module = packed
    task = SimpleNamespace(module=source)
    processor = SimpleNamespace(tasks={"linear": task})

    ModuleLooper._release_processor_task_source_storage(processor, named, source)

    assert task.module.weight.is_meta
    assert task.module.bias.is_meta
    restored = auto_dtype_cast.restore_source_module(packed, spec, device="cpu")
    ModuleLooper._rebind_processor_task_module(processor, named, None, restored)
    assert task.module is restored
    assert _bits_equal(task.module.weight, expected_weight)
    assert _bits_equal(task.module.bias, expected_bias)


def test_hooked_linear_cast_preserves_transformed_hook_operand(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))
    monkeypatch.setattr(
        "gptqmodel.nn_modules.hooked_linear.apply_online_hadamard",
        lambda value, **_kwargs: value.flip(-1),
    )
    monkeypatch.setattr(
        NF12Linear,
        "_apply_rotation_to_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("rotation applied twice")),
    )

    source = HookedLinear.from_linear(
        nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
    )
    source.online_full_had = True
    seen = []
    source.forward_hook = lambda _module, args, _output: seen.append(args[0].clone())
    packed = auto_dtype_cast.AutoNF12HookedLinear.from_linear(source)
    x = torch.zeros(1, 64, dtype=torch.bfloat16)
    x[0, 0] = 1

    source(x)
    packed(x)

    assert _bits_equal(seen[0], x.flip(-1))
    assert _bits_equal(seen[1], seen[0])


@pytest.mark.parametrize("mode", ["w4afp8", "w4a_nvfp4"])
def test_hooked_linear_cast_preserves_w4a_replay_semantics(monkeypatch, mode):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))
    monkeypatch.setattr(
        "gptqmodel.nn_modules.hooked_linear.apply_online_hadamard",
        lambda value, **_kwargs: value.flip(-1),
    )

    source = HookedLinear.from_linear(
        nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    )
    with torch.no_grad():
        source.weight.copy_(torch.eye(64, dtype=torch.bfloat16))
    source.online_full_had = True
    source._w4a_stream_replay_mode = mode
    source._w4a_stream_replay_recipe = "least_squares" if mode == "w4a_nvfp4" else None
    source._w4a_replay_model_dtype = torch.bfloat16
    seen = []
    source.forward_hook = lambda _module, args, _output: seen.append(args[0].clone())
    packed = auto_dtype_cast.AutoNF12HookedLinear.from_linear(source)
    x = torch.linspace(-3, 3, 128, dtype=torch.bfloat16).reshape(2, 64)

    expected = source(x)
    actual = packed(x)

    assert len(seen) == 2
    assert _bits_equal(seen[0], seen[1])
    assert _bits_equal(actual, expected)


def test_restore_hooked_linear_preserves_runtime_metadata_and_forward(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))
    monkeypatch.setattr(
        "gptqmodel.nn_modules.hooked_linear.apply_online_hadamard",
        lambda value, **_kwargs: value.flip(-1),
    )

    source = HookedLinear.from_linear(
        nn.Linear(64, 16, bias=True, dtype=torch.bfloat16)
    )
    with torch.no_grad():
        source.weight.copy_(torch.randn_like(source.weight) * 0.02)
        source.bias.copy_(torch.randn_like(source.bias))
    source.online_full_had = True
    source.online_partial_had = False
    source.had_dim = -1
    source.had_K = None
    source.K = 1
    source.groups = 2
    source.module_name = "proj"
    source.full_name = "layer.proj"
    source.name = "proj"
    source.forward_hook_last = False
    source._w4a_replay_disabled = True
    source._w4a_norm_preapplied = False
    source._w4a_stream_replay_mode = "w4afp8"
    source._w4a_stream_replay_recipe = None
    source._w4a_replay_model_dtype = torch.bfloat16
    source._w4a_stream_replay_pre_hook = True
    source._w4a_replay_input_hook_active = True
    source._w4a_rotation_preapplied = False
    source._w4a_activation_global_scale = 0.25
    source._w4a_headroom_probe = True
    custom_hook_seen = []
    source.forward_hook = lambda _module, args, _output: custom_hook_seen.append(args[0].clone())
    registered_hook_seen = []
    source.register_forward_hook(
        lambda _module, _args, _output: registered_hook_seen.append(True)
    )

    spec = auto_dtype_cast.eligible_source_module(source, "cpu")
    packed = auto_dtype_cast.cast_source_module(source, spec)
    restored = auto_dtype_cast.restore_source_module(packed, spec, device="cpu")
    x = torch.randn(3, 64, dtype=torch.bfloat16)

    expected = source(x)
    actual = restored(x)

    assert type(restored) is HookedLinear
    assert _bits_equal(restored.weight, source.weight)
    assert _bits_equal(restored.bias, source.bias)
    for name in (
        "online_full_had",
        "online_partial_had",
        "had_dim",
        "had_K",
        "K",
        "groups",
        "module_name",
        "forward_hook",
        "forward_hook_last",
        "full_name",
        "name",
        "_w4a_replay_disabled",
        "_w4a_norm_preapplied",
        "_w4a_stream_replay_mode",
        "_w4a_stream_replay_recipe",
        "_w4a_replay_model_dtype",
        "_w4a_stream_replay_pre_hook",
        "_w4a_replay_input_hook_active",
        "_w4a_rotation_preapplied",
        "_w4a_activation_global_scale",
        "_w4a_headroom_probe",
    ):
        assert getattr(restored, name) == getattr(source, name)
    assert _bits_equal(actual, expected)
    assert len(custom_hook_seen) == 2
    assert len(registered_hook_seen) == 2


def test_registered_hook_removal_survives_cast_and_restore(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    source = HookedLinear.from_linear(
        nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
    )
    with torch.no_grad():
        source.weight.fill_(0.5)

    pre_calls = []
    forward_calls = []

    def pre_hook(_module, args, kwargs):
        pre_calls.append(1)
        return args[0] + 1, kwargs

    def forward_hook(_module, args, kwargs, output):
        forward_calls.append(1)
        return output

    pre_handle = source.register_forward_pre_hook(pre_hook, with_kwargs=True)
    forward_handle = source.register_forward_hook(
        forward_hook,
        with_kwargs=True,
        always_call=True,
    )
    spec = auto_dtype_cast.eligible_source_module(source, "cpu")
    packed = auto_dtype_cast.cast_source_module(source, spec)
    restored = auto_dtype_cast.restore_source_module(packed, spec, device="cpu")
    modules = (source, packed, restored)

    for name in (
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
    ):
        assert all(getattr(module, name) is getattr(source, name) for module in modules)

    pre_handle.remove()
    forward_handle.remove()
    for name in (
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
    ):
        assert all(not getattr(module, name) for module in modules)

    x = torch.zeros(1, 64, dtype=torch.bfloat16)
    outputs = [module(x) for module in modules]
    assert all(output.count_nonzero().item() == 0 for output in outputs)
    assert not pre_calls
    assert not forward_calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pinned_cpu_offload_preserves_bits_and_pins_large_tensors():
    module = nn.Module()
    module.large = nn.Linear(2048, 2048, bias=False, device="cuda", dtype=torch.bfloat16)
    module.small = nn.Linear(64, 64, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        module.large.weight.copy_(torch.randn_like(module.large.weight))
        module.small.weight.copy_(torch.randn_like(module.small.weight))
    expected_large = module.large.weight.detach().cpu()
    expected_small = module.small.weight.detach().cpu()

    moved = move_module_to_cpu_pinned(module)

    assert moved is module
    assert module.large.weight.device.type == "cpu"
    assert module.small.weight.device.type == "cpu"
    assert module.large.weight.is_pinned()
    assert not module.small.weight.is_pinned()
    assert _bits_equal(module.large.weight, expected_large)
    assert _bits_equal(module.small.weight, expected_small)


def test_pinned_cpu_offload_moves_non_cuda_accelerators_to_cpu():
    """XPU/MPS/NPU/XLA must still be offloaded; only meta may stay put."""

    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        for device in ("xpu:0", "mps", "cpu", "meta"):
            module = nn.Module()
            module.register_buffer("b", torch.empty(4, device=device))

            move_module_to_cpu_pinned(module)

            expected = "meta" if device == "meta" else "cpu"
            assert module.b.device.type == expected, (device, module.b.device)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not auto_dtype_cast.nf12_supported_device("cuda:0"),
    reason="requires a native NF12 accelerator",
)
def test_gptq_process_uses_the_fused_diff_with_a_bf16_reconstruction(monkeypatch):
    """Drive the real process path with a BF16 wq and a packed NF12 source."""

    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    linear = nn.Linear(64, 16, bias=False, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(torch.randn_like(linear.weight) * 0.02)
    spec = auto_dtype_cast.eligible_source_module(linear, device)
    assert spec is not None
    packed = auto_dtype_cast.cast_source_module(linear, spec)
    restored = auto_dtype_cast.restore_source_module(packed, spec, device=device)

    named = NamedModule(restored, name="proj", full_name="proj", layer_index=0)
    named.state["auto_dtype_cast_packed_source"] = packed
    object.__setattr__(
        named,
        "stream_state_payload_to_cpu",
        lambda payload: named.state.update(payload),
    )

    wq = (restored.weight.detach() + 0.001).to(torch.bfloat16)
    source = restored.weight.detach().clone()
    q_scales = torch.ones((1,), dtype=torch.bfloat16)
    q_zeros = torch.zeros((1,), dtype=torch.bfloat16)
    q_g_idx = torch.zeros((64,), dtype=torch.int32)

    class StubTask:
        def __init__(self) -> None:
            self.module = SimpleNamespace(weight=restored.weight, H=None)
            self.gsq_diagnostics = None
            self._borrow_workspace_last_summary = None
            self._borrow_workspace_totals = None

        def quantize(self):
            return wq.clone(), q_scales, q_zeros, q_g_idx, 0.1, 0.5, 0.01, 1

        def free(self) -> None:
            pass

        def log_workspace_stats(self, **_kwargs) -> None:
            pass

    processor = gptq_processor.GPTQProcessor.__new__(gptq_processor.GPTQProcessor)
    processor.tasks = {"proj": StubTask()}
    processor.lock = threading.Lock()
    processor.durations = []
    processor.avg_losses = []
    processor.module_names = []
    processor.log = []
    processor.qcfg = SimpleNamespace(dynamic=None, activation_mode="none")
    processor.calculate_w_wq_diff = True
    processor.draw_progress = lambda *_args, **_kwargs: None
    processor.log_new_row = lambda *_args, **_kwargs: None
    processor.module_feature_summary = lambda *_args, **_kwargs: ""
    processor.module_dtype_size_summary = lambda *_args, **_kwargs: ""
    processor.formatted_fwd_time = lambda *_args, **_kwargs: ""
    processor.device_memory_report = lambda *_args, **_kwargs: ""
    processor.name = lambda: "gptq"

    real_fused = auto_dtype_cast.fused_w_minus_wq
    calls = []

    def spy(candidate, candidate_wq):
        result = real_fused(candidate, candidate_wq)
        calls.append((candidate_wq.dtype, result is not None))
        return result

    monkeypatch.setattr(auto_dtype_cast, "fused_w_minus_wq", spy)

    processor.process(named, device=device)

    diff = named.state["w_wq_diff"]
    expected = source.to(torch.float32) - wq.to(torch.float32)
    assert calls == [(torch.float32, True)]
    assert diff.dtype is torch.float32
    assert torch.equal(diff.view(torch.int32), expected.view(torch.int32))
    assert "auto_dtype_cast_packed_source" not in named.state
    assert all(not tensor.numel() for tensor in packed.buffers())
    assert named.state["wq"].dtype is torch.bfloat16
    assert named.state["wq"].device.type == "cpu"


def test_quantization_pipeline_installs_nf12_modules_for_every_eligible_kind(monkeypatch):
    """Exercise the live module-installation boundary used by the quantizer."""

    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    class PipelineModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(16, 64, dtype=torch.bfloat16)
            self.proj = nn.Linear(64, 16, bias=True, dtype=torch.bfloat16)
            self.hooked = HookedLinear(64, 16)
            self.hooked.weight = nn.Parameter(torch.randn(16, 64, dtype=torch.bfloat16) * 0.02)
            self.hooked.bias = nn.Parameter(torch.randn(16, dtype=torch.bfloat16))
            self.norm = nn.LayerNorm(64, dtype=torch.bfloat16)
            self.unsupported = nn.Linear(63, 16, bias=False, dtype=torch.bfloat16)
            self.max_norm_embedding = nn.Embedding(16, 64, max_norm=1.0, dtype=torch.bfloat16)

    class PipelineProcessor:
        def __init__(self) -> None:
            self.tasks = {}

        def preprocess(self, named_module: NamedModule) -> None:
            self.tasks[named_module.name] = SimpleNamespace(module=named_module.module)

        def is_skipped(self, _named_module: NamedModule) -> bool:
            return False

    model = PipelineModel()
    ids = torch.tensor([0, 1, 15])
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    expected_embedding = model.embed_tokens(ids).clone()
    expected_proj = model.proj(x).clone()
    expected_hooked = model.hooked(x).clone()
    names = [
        "embed_tokens",
        "proj",
        "hooked",
        "norm",
        "unsupported",
        "max_norm_embedding",
    ]
    full = {name: getattr(model, name) for name in names}
    processor = PipelineProcessor()

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = model
    harness.quantize_config = QuantizeConfig(offload_to_disk=False)
    harness._turtle_lock = threading.RLock()
    harness.turtle_model = None
    harness.auto_dtype_cast_events = []

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = harness
    looper.input_embeddings_name = "embed_tokens"
    looper.output_embeddings_name = None
    looper.layer_module_coverage = SimpleNamespace(
        record_hit=lambda *_args: None,
        record_miss=lambda *_args: None,
    )

    subset = looper.create_named_modules(
        module=model,
        full=full,
        is_lm_head_module=False,
        layer_index=0,
        layers_prefix="layer",
        names=names,
        processor=processor,
        fallback=torch.device("cpu"),
    )

    assert isinstance(model.embed_tokens, auto_dtype_cast.AutoNF12Embedding)
    assert isinstance(model.proj, NF12Linear)
    assert isinstance(model.hooked, auto_dtype_cast.AutoNF12HookedLinear)
    assert isinstance(model.norm, nn.LayerNorm)
    assert isinstance(model.unsupported, nn.Linear)
    assert isinstance(model.max_norm_embedding, nn.Embedding)

    assert subset["embed_tokens"].module is model.embed_tokens
    assert subset["proj"].module is model.proj
    assert subset["hooked"].module is model.hooked
    assert subset["norm"].module is model.norm
    assert subset["unsupported"].module is model.unsupported
    assert subset["max_norm_embedding"].module is model.max_norm_embedding

    for name in ("embed_tokens", "proj", "hooked"):
        assert subset[name].state["auto_dtype_cast_was_applied"]
        assert subset[name].state["auto_dtype_cast_source_spec"] is not None
        assert processor.tasks[name].module.weight.is_meta
    assert not processor.tasks["norm"].module.weight.is_meta
    assert not processor.tasks["unsupported"].module.weight.is_meta
    assert not processor.tasks["max_norm_embedding"].module.weight.is_meta

    assert _bits_equal(model.embed_tokens(ids), expected_embedding)
    torch.testing.assert_close(model.proj(x), expected_proj, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(model.hooked(x), expected_hooked, rtol=2e-3, atol=2e-3)
