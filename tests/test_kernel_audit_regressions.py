# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.nn_modules.qlinear import (
    exllamav2,
    exllamav2_awq,
    gemm_awq,
    machete,
)
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear, pack_intweight
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCudaKernel, _GGMLBridge
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils.marlin_scratch import MarlinScratchContext


@pytest.mark.parametrize(
    ("module_name", "class_name", "dense_method"),
    (
        (
            "gptqmodel.nn_modules.qlinear.torch_aten_kernel",
            "TorchAtenLinear",
            "dequantize_weight",
        ),
        (
            "gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq",
            "TorchAtenAwqLinear",
            "awq_weight_dequantize",
        ),
        (
            "gptqmodel.nn_modules.qlinear.torch_fused",
            "TorchFusedLinear",
            "dequantize_weight",
        ),
        (
            "gptqmodel.nn_modules.qlinear.torch_fused_awq",
            "TorchFusedAwqLinear",
            "awq_weight_dequantize",
        ),
    ),
)
def test_fused_cpu_layer_restores_checkpoint_state_before_training(
    monkeypatch,
    module_name,
    class_name,
    dense_method,
):
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    monkeypatch.setattr(module, "TORCH_HAS_FUSED_OPS", True, raising=False)
    monkeypatch.setattr(module, "_has_local_int4pack_cpu_ops", lambda: True, raising=False)

    layer = cls(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=32,
        bias=False,
        register_buffers=True,
    )
    original_shape = tuple(layer.qweight.shape)

    def fake_transform(*_args, **_kwargs):
        layer.qweight = torch.zeros((16, 16), dtype=torch.int32)
        layer.scales = torch.zeros((1, 32), dtype=torch.float16)
        layer.qzeros = torch.zeros((1, 32), dtype=torch.float16)
        layer.scales_and_zeros = torch.zeros((1, 32, 2), dtype=torch.float16)

    monkeypatch.setattr(layer, "transform", fake_transform)
    monkeypatch.setattr(
        layer,
        "_fused_op_forward",
        lambda x: torch.ones((x.shape[0], 32), dtype=x.dtype, device=x.device),
    )
    monkeypatch.setattr(
        layer,
        dense_method,
        lambda *_args, **_kwargs: torch.ones((32, 32), dtype=torch.float16),
    )

    x = torch.randn(2, 32, dtype=torch.float16)
    layer.eval()
    layer(x)
    assert str(getattr(layer.linear_mode, "name", layer.linear_mode)).upper() == "INFERENCE"
    assert not any(
        name.startswith("_fused_train_state_") for name in layer.state_dict()
    )

    layer._suspend_fused_train_restore = True
    layer.clear_weight_cache()
    assert layer._fused_train_state is not None
    layer.__dict__.pop("_suspend_fused_train_restore")

    layer.to(dtype=torch.bfloat16)
    assert tuple(layer.qweight.shape) == original_shape
    assert layer.scales.dtype == torch.bfloat16

    layer.train()
    x.requires_grad_(True)
    out = layer(x)
    assert str(getattr(layer.linear_mode, "name", layer.linear_mode)).upper() == "TRAIN"
    assert tuple(layer.qweight.shape) == original_shape
    assert out.requires_grad
    out.sum().backward()


def test_gemv_fast_rejects_unaligned_shapes_before_packing():
    with pytest.raises(ValueError, match="in_features"):
        pack_intweight(torch.zeros((8, 96), dtype=torch.int32), interleave=4, kstride=64)
    with pytest.raises(ValueError, match="out_features"):
        pack_intweight(torch.zeros((6, 128), dtype=torch.int32), interleave=4, kstride=64)

    ok, error = AwqGEMVFastLinear.validate(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        in_features=96,
        out_features=8,
        pack_dtype=torch.int16,
        dtype=torch.float16,
    )
    assert not ok
    assert isinstance(error, NotImplementedError)

    ok, error = AwqGEMVFastLinear.validate(
        bits=4,
        group_size=64,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=128,
        pack_dtype=torch.int16,
        dtype=torch.float16,
    )
    assert not ok
    assert isinstance(error, NotImplementedError)

    ok, error = AwqGEMVFastLinear.validate(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=64,
        pack_dtype=torch.int16,
        dtype=torch.float16,
    )
    assert not ok
    assert isinstance(error, NotImplementedError)


def test_awq_gemm_validation_matches_native_width_and_group_contract(monkeypatch):
    monkeypatch.setattr(gemm_awq, "awq_runtime_available", lambda: True)
    gemm_awq.AwqGEMMLinear.cached_validate_once.cache_clear()

    ok, _ = gemm_awq.AwqGEMMLinear.validate(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=96,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )
    assert not ok

    ok, _ = gemm_awq.AwqGEMMLinear.validate(
        bits=4,
        group_size=16,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=128,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )
    assert not ok

    ok, _ = gemm_awq.AwqGEMMLinear.validate(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=64,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )
    assert not ok

    ok, _ = gemm_awq.AwqGEMMLinear.validate(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=128,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )
    assert ok

    ok, _ = gemm_awq.AwqGEMMLinear.validate(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=192,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )
    assert not ok


def test_machete_validation_checks_extension_availability(monkeypatch):
    monkeypatch.setattr(machete, "machete_extension_available", lambda: False)
    monkeypatch.setattr(
        machete,
        "machete_extension_error",
        lambda: RuntimeError("forced unavailable"),
    )
    machete.MacheteLinear.cached_validate_once.cache_clear()

    once, _ = machete.MacheteLinear.validate_once()
    validated, error = machete.MacheteLinear.validate(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=128,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )
    assert not once
    assert not validated
    assert isinstance(error, ImportError)


def test_torch_linear_rejects_non_int32_three_bit_packing():
    ok, error = TorchLinear.validate(
        bits=3,
        group_size=32,
        desc_act=False,
        sym=True,
        in_features=32,
        out_features=32,
        pack_dtype=torch.int8,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    assert not ok
    assert isinstance(error, NotImplementedError)


def test_native_pack_block_rejects_partial_output_words():
    from gptqmodel.nn_modules.qlinear.pack_block_ext import pack_block_cpu

    weight = torch.zeros((33, 32), dtype=torch.float32)
    scales = torch.ones((1, 33), dtype=torch.float32)
    zeros = torch.zeros((1, 33), dtype=torch.float32)
    g_idx = torch.zeros(32, dtype=torch.int32)

    with pytest.raises(RuntimeError, match="out_features"):
        pack_block_cpu(weight, scales, zeros, g_idx, 4, 32, 32, 1)


def test_paroquant_runtime_dtype_does_not_mutate_checkpoint_metadata():
    from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear

    module = SimpleNamespace(
        scales=torch.tensor([1.001, 1.003], dtype=torch.float16),
        bias=torch.tensor([0.3333, -0.3333], dtype=torch.float16),
        _runtime_dtype_key=None,
        _runtime_scales=None,
        _runtime_bias=None,
    )
    before_scales = module.scales.clone()
    before_bias = module.bias.clone()

    scales, bias = ParoLinear._runtime_dtype_tensors(
        module,
        torch.device("cpu"),
        torch.bfloat16,
        cache=True,
    )
    assert scales.dtype == torch.bfloat16
    assert bias.dtype == torch.bfloat16
    assert torch.equal(module.scales, before_scales)
    assert torch.equal(module.bias, before_bias)


def test_gguf_cuda_bf16_uses_fp32_without_range_loss():
    bridge = SimpleNamespace(ggml_type_f16=1, ggml_type_f32=2)
    x = torch.tensor([1e10, 1e-10], dtype=torch.bfloat16)

    normalized, ggml_type = _GGMLBridge._normalize_input_cuda(
        bridge,
        x,
        padded_in_features=2,
    )

    assert ggml_type == bridge.ggml_type_f32
    assert normalized.dtype == torch.float32
    assert torch.isfinite(normalized).all()
    assert normalized[0].item() > 1e9


def test_gguf_cuda_plan_key_tracks_inplace_weight_updates():
    module = SimpleNamespace(qweight=torch.zeros((32, 8), dtype=torch.uint8))
    x = torch.empty((1, 32), dtype=torch.float16)

    before = GGUFCudaKernel._cuda_plan_key(module, x)
    module.qweight.add_(1)
    after = GGUFCudaKernel._cuda_plan_key(module, x)

    assert before != after


def test_gguf_cuda_plan_cache_is_bounded_and_closes_evicted_plans():
    class Plan:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    module = SimpleNamespace(_ggml_cuda_plans=OrderedDict())
    plans = []
    for index in range(9):
        plan = Plan()
        plans.append(plan)
        key = (0, index, torch.float16, ())
        GGUFCudaKernel._remember_cuda_plan(
            module,
            key,
            plan,
            cached=False,
        )

    assert len(module._ggml_cuda_plans) == 8
    assert plans[0].closed
    assert not plans[-1].closed


def test_gguf_triton_rejects_wrong_input_width_before_device_dispatch():
    module = SimpleNamespace(in_features=32)
    with pytest.raises(ValueError, match="expected input width 32"):
        GGUFTritonKernel.forward(module, torch.zeros((1, 31)))


def test_marlin_scratch_context_honors_workspace_floor(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=20),
    )
    context = MarlinScratchContext("cuda:0")
    assert context._lock_workspace_bytes() == 128 * 4


@pytest.mark.parametrize(
    ("module", "class_name"),
    (
        (exllamav2, "ExllamaV2Linear"),
        (exllamav2_awq, "AwqExllamaV2Linear"),
    ),
)
def test_exllamav2_clear_releases_native_handle(monkeypatch, module, class_name):
    freed = []
    monkeypatch.setattr(
        module,
        "exllamav2_free_q_matrix",
        lambda handle, **_kwargs: freed.append(handle),
    )
    cls = getattr(module, class_name)
    layer = SimpleNamespace(q_handle=77, q_tensors={"qweight": torch.empty(0)})
    layer._release_q_handle = lambda: cls._release_q_handle(layer)

    cls.clear_weight_cache(layer)

    assert freed == [77]
    assert layer.q_handle is None


def test_exllamav2_handle_tracks_replaced_packed_tensors(monkeypatch):
    freed = []
    made = []

    def make_q_matrix(tensors):
        made.append(tensors["qweight"].clone())
        return len(made)

    monkeypatch.setattr(exllamav2, "exllamav2_free_q_matrix", freed.append)
    layer = SimpleNamespace(
        qweight=torch.zeros((2, 2), dtype=torch.int32),
        qzeros=torch.zeros((1, 2), dtype=torch.int32),
        scales=torch.ones((1, 2), dtype=torch.float16),
        g_idx=torch.zeros(8, dtype=torch.int32),
        q_handle=None,
        q_tensors=None,
        _q_handle_key=None,
    )
    layer.ext_make_q_matrix = make_q_matrix
    layer._release_q_handle = lambda: exllamav2.ExllamaV2Linear._release_q_handle(
        layer
    )

    first = exllamav2.ExllamaV2Linear._ensure_q_handle(layer)
    layer.qweight = torch.ones_like(layer.qweight)
    second = exllamav2.ExllamaV2Linear._ensure_q_handle(layer)

    assert (first, second) == (1, 2)
    assert freed == [1]
    assert made[1].eq(1).all()


def test_exllamav2_gptq_forward_compacts_noncontiguous_activations(monkeypatch):
    seen = {}

    def fake_gemm(x, _handle, output, _force_cuda):
        seen["contiguous"] = x.is_contiguous()
        seen["shape"] = tuple(x.shape)
        output.fill_(1)

    monkeypatch.setattr(exllamav2, "exllamav2_gemm_half_q_half", fake_gemm)
    layer = SimpleNamespace(
        q_handle=11,
        _fusion_native_view=False,
        out_features=3,
        bias=None,
        adapter=None,
    )
    layer._ensure_q_handle = lambda: 11
    layer.input_rows = lambda value: value.shape[0]
    layer.ext_gemm_half_q_half = lambda *args: exllamav2.ExllamaV2Linear.ext_gemm_half_q_half(
        layer,
        *args,
    )
    base = torch.arange(16, dtype=torch.float16).reshape(4, 4)
    x = base[:, ::2]

    out = exllamav2.ExllamaV2Linear.forward(layer, x)

    assert seen == {"contiguous": True, "shape": (4, 2)}
    assert out.shape == (4, 3)
    assert out.dtype == torch.float16


def test_exllamav3_bf16_uses_fp32_native_input_and_output():
    seen = {}

    class Inner:
        def forward(self, x, _params, *, out_dtype=None):
            seen["input_dtype"] = x.dtype
            seen["out_dtype"] = out_dtype
            return torch.ones(x.shape[:-1] + (4,), dtype=out_dtype)

    layer = SimpleNamespace(_ensure_inner=lambda: Inner(), out_features=4)
    x = torch.tensor([[1e10, 1e-10]], dtype=torch.bfloat16)

    out = ExllamaV3Linear.forward(layer, x)

    assert seen == {
        "input_dtype": torch.float32,
        "out_dtype": torch.float32,
    }
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()

    fp16 = torch.ones((1, 2), dtype=torch.float16)
    out = ExllamaV3Linear.forward(layer, fp16)

    assert seen == {
        "input_dtype": torch.float16,
        "out_dtype": torch.float16,
    }
    assert out.dtype == torch.float16
