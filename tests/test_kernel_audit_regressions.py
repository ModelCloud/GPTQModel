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

    def make_q_matrix(tensors, **_kwargs):
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


def test_exllamav3_wide_inputs_route_to_torch_reconstruction():
    """Wide activations must not reach the FP16-only native EXL3 kernel.

    #3717 pinned a native path that upcast wide inputs to FP32. #3731
    (``86a59903e``) replaced it with the bounded Torch reconstruction path,
    which receives the caller dtype as-is and upcasts internally inside
    ``ExllamaV3TorchLinear._forward_eager``. Because that inner takes a single
    ``x`` argument, the native ``(x, _params, out_dtype=...)`` call would still
    be reached here under the stale contract and explode.
    """
    seen = {}

    class TorchInner:
        def forward(self, x):
            seen["input_dtype"] = x.dtype
            return torch.ones(x.shape[:-1] + (4,), dtype=x.dtype)

    class NativeInner:
        def forward(self, x, _params, *, out_dtype=None):
            seen["native"] = (x.dtype, out_dtype)
            return torch.ones(x.shape[:-1] + (4,), dtype=out_dtype or x.dtype)

    layer = SimpleNamespace(
        _ensure_torch_inner=lambda: TorchInner(),
        _ensure_inner=lambda: pytest.fail("wide inputs must not use the native kernel"),
        out_features=4,
    )
    # 1e10 overflows FP16 (max 65504); routing through the FP32 Torch inner
    # keeps the result finite.
    x = torch.tensor([[1e10, 1e-10]], dtype=torch.bfloat16)

    out = ExllamaV3Linear.forward(layer, x)

    # The Torch inner owns the FP32 upcast; the wrapper only sees the caller dtype.
    assert seen == {"input_dtype": torch.bfloat16}
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()

    # FP16 keeps the FP16 native kernel with an explicit out_dtype.
    seen.clear()
    layer._ensure_torch_inner = lambda: pytest.fail("FP16 must keep the native path")
    layer._ensure_inner = lambda: NativeInner()
    out = ExllamaV3Linear.forward(layer, torch.ones((1, 2), dtype=torch.float16))

    assert seen == {"native": (torch.float16, torch.float16)}
    assert out.dtype == torch.float16


@pytest.mark.parametrize(
    ("class_name", "expected_dtype"),
    (
        ("TorchAtenLinear", torch.bfloat16),
        ("TorchAtenAwqLinear", torch.bfloat16),
        ("TorchFusedLinear", torch.bfloat16),
        ("TorchFusedAwqLinear", torch.bfloat16),
        ("TorchCpuAvx512Linear", torch.bfloat16),
        ("TorchCpuAvx512AwqLinear", torch.bfloat16),
    ),
)
def test_cpu_inference_forward_preserves_checkpoint_save_layout(class_name, expected_dtype):
    """A plain inference forward must not corrupt what ``save()`` writes.

    The CPU W4A16 kernels rewrite ``qweight``/``qzeros``/``scales`` into their
    int4pack runtime layout *in place*. The checkpoint layout survives only in
    ``_fused_train_state``, so the export path and ``load_state_dict`` must
    both restore it first. Otherwise ``state_dict()``/``save()`` emit runtime
    shapes under checkpoint keys and the written file cannot be loaded into a
    fresh model.
    """

    from tests.module_tree.test_fusion_kernel_backends import _projection

    module = importlib.import_module(
        {
            "TorchAtenLinear": "gptqmodel.nn_modules.qlinear.torch_aten_kernel",
            "TorchAtenAwqLinear": "gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq",
            "TorchFusedLinear": "gptqmodel.nn_modules.qlinear.torch_fused",
            "TorchFusedAwqLinear": "gptqmodel.nn_modules.qlinear.torch_fused_awq",
            "TorchCpuAvx512Linear": "gptqmodel.nn_modules.qlinear.torch_cpu_avx512",
            "TorchCpuAvx512AwqLinear": "gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq",
        }[class_name]
    )
    cls = getattr(module, class_name)
    if not getattr(cls, "SUPPORTS_DTYPES", [expected_dtype]) or expected_dtype not in cls.SUPPORTS_DTYPES:
        pytest.skip(f"{class_name} does not support {expected_dtype}")

    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel.utils.model import get_state_dict_for_save

    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=64,
        attention_bias=True,
        mlp_bias=True,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).to(dtype=expected_dtype).eval()
    parent = model.model.layers[0].mlp
    for name in ("gate_proj", "up_proj"):
        projection, _weight, _bias = _projection(
            cls, getattr(parent, name), expected_dtype, "cpu"
        )
        setattr(parent, name, projection)

    def exported():
        return {
            name: (tuple(entry.shape), entry.torch_dtype)
            for name, entry in get_state_dict_for_save(model).items()
            if name.endswith(("gate_proj.qweight", "gate_proj.qzeros", "gate_proj.bias"))
        }

    pristine = exported()
    x = torch.randn(2, 5, 256, dtype=expected_dtype)
    with torch.inference_mode():
        baseline = parent(x).clone()

    # One inference forward moves the operands into the int4pack runtime layout.
    assert tuple(parent.gate_proj.qweight.shape) != pristine[
        "model.layers.0.mlp.gate_proj.qweight"
    ][0]

    # The export path restores the checkpoint layout before serializing.
    assert exported() == pristine

    # Loading that checkpoint back into the now-transformed model restores the
    # layout first, so strict loading does not report a size mismatch.
    checkpoint = {
        name: tensor.clone() for name, tensor in model.state_dict().items()
    }
    with torch.inference_mode():
        parent(x)
    model.load_state_dict(checkpoint, strict=True)
    assert exported() == pristine

    # The round trip is numerically lossless.
    with torch.inference_mode():
        restored = parent(x).clone()
    torch.testing.assert_close(restored.float(), baseline.float(), rtol=2e-3, atol=2e-3)


def _transformed_layer(cls, *, seed=17):
    """Build a finalized CPU layer and drive one inference forward."""

    from tests.module_tree.test_fusion_kernel_backends import _projection

    dtype = torch.bfloat16
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=64,
        attention_bias=True,
        mlp_bias=True,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = LlamaForCausalLM(config).to(dtype).eval()
    layer, _weight, _bias = _projection(cls, model.model.layers[0].mlp.gate_proj, dtype, "cpu")
    x = torch.randn(2, 5, 256, dtype=dtype)

    class _Holder(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.proj = module

        def forward(self, x):
            return self.proj(x)

    holder = _Holder(layer).eval()
    with torch.inference_mode():
        baseline = holder(x).clone()
    return holder, x, baseline


@pytest.mark.parametrize(
    ("class_name", "module_path"),
    (
        ("TorchAtenLinear", "gptqmodel.nn_modules.qlinear.torch_aten_kernel"),
        ("TorchAtenAwqLinear", "gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq"),
        ("TorchFusedLinear", "gptqmodel.nn_modules.qlinear.torch_fused"),
        ("TorchFusedAwqLinear", "gptqmodel.nn_modules.qlinear.torch_fused_awq"),
        ("TorchCpuAvx512Linear", "gptqmodel.nn_modules.qlinear.torch_cpu_avx512"),
        ("TorchCpuAvx512AwqLinear", "gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq"),
    ),
)
@pytest.mark.parametrize("assign", (False, True))
def test_transformed_cpu_layer_round_trips_its_own_state_dict(class_name, module_path, assign):
    """A transformed layer must accept its own runtime ``state_dict()`` back.

    After a CPU inference forward the live operands are in the int4pack layout.
    Taking ``state_dict()`` in that state and loading it back is a self
    round-trip: the incoming ``qweight``/``qzeros`` describe the *runtime*
    layout, not the checkpoint held in ``_fused_train_state``.

    ``scales`` keeps its checkpoint shape in both layouts, so a gate that
    accepted any single matching operand mistook this dict for a checkpoint,
    restored the snapshot, and then failed strict loading with a size mismatch
    (``qweight`` 64x64 into 16x64). The load must instead keep the live layout
    and remain numerically lossless.
    """

    from tests.test_torch_weight_storage import state_tensors

    cls = getattr(importlib.import_module(module_path), class_name)
    if torch.bfloat16 not in getattr(cls, "SUPPORTS_DTYPES", [torch.bfloat16]):
        pytest.skip(f"{class_name} does not support {torch.bfloat16}")

    holder, x, baseline = _transformed_layer(cls)
    layer = holder.proj
    assert layer._fused_train_state, "the forward must have captured a snapshot"
    for tensor in state_tensors(layer):
        assert tensor.numel() < layer.in_features * layer.out_features
    # ``scales`` is shape-identical across both layouts; it must never be the
    # operand that decides the load.
    live_qweight = tuple(layer.qweight.shape)
    snapshot_qweight = tuple(layer._buffers[layer._fused_train_state["qweight"]].shape)
    assert live_qweight != snapshot_qweight
    assert tuple(layer.scales.shape) == tuple(
        layer._buffers[layer._fused_train_state["scales"]].shape
    )

    state = {name: value.clone() for name, value in layer.state_dict().items()}
    layer.load_state_dict(state, assign=assign)

    # The load kept the live layout instead of restoring the checkpoint shapes.
    assert tuple(layer.qweight.shape) == live_qweight

    with torch.inference_mode():
        restored = holder(x).clone()
    torch.testing.assert_close(restored.float(), baseline.float(), rtol=2e-3, atol=2e-3)
    for tensor in state_tensors(layer):
        assert tensor.numel() < layer.in_features * layer.out_features

    # A self round-trip still needs its original packed checkpoint layout for save.
    from gptqmodel.utils.model import get_state_dict_for_save

    exported = get_state_dict_for_save(holder)
    assert exported["proj.qweight"].shape == snapshot_qweight


def test_changed_runtime_state_cannot_silently_export_stale_checkpoint():
    from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
    from gptqmodel.utils.model import get_state_dict_for_save

    holder, _x, _baseline = _transformed_layer(TorchAtenLinear)
    state = {name: value.clone() for name, value in holder.proj.state_dict().items()}
    checkpoint = dict(state)
    for name, buffer_name in holder.proj._fused_train_state.items():
        checkpoint[name] = holder.proj._buffers[buffer_name].clone()
    state["scales"].mul_(2)
    holder.proj.load_state_dict(state, assign=True)

    with pytest.raises(RuntimeError, match="modified CPU int4pack runtime state"):
        get_state_dict_for_save(holder)

    holder.proj.load_state_dict(checkpoint)
    assert get_state_dict_for_save(holder)["proj.qweight"].shape == checkpoint["qweight"].shape


def test_runtime_load_materializes_inference_g_idx():
    from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear

    holder, x, baseline = _transformed_layer(TorchAtenLinear)
    with torch.inference_mode():
        holder.proj.g_idx = holder.proj.g_idx.clone()
    assert torch.is_inference(holder.proj.g_idx)
    state = {name: value.clone() for name, value in holder.proj.state_dict().items()}
    holder.proj.load_state_dict(state, assign=False)

    with torch.inference_mode():
        actual = holder(x).clone()
    torch.testing.assert_close(actual.float(), baseline.float(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    ("class_name", "module_path"),
    (
        ("TorchAtenLinear", "gptqmodel.nn_modules.qlinear.torch_aten_kernel"),
        ("TorchAtenAwqLinear", "gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq"),
        ("TorchFusedLinear", "gptqmodel.nn_modules.qlinear.torch_fused"),
        ("TorchFusedAwqLinear", "gptqmodel.nn_modules.qlinear.torch_fused_awq"),
        ("TorchCpuAvx512Linear", "gptqmodel.nn_modules.qlinear.torch_cpu_avx512"),
        ("TorchCpuAvx512AwqLinear", "gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq"),
    ),
)
def test_checkpoint_layout_gate_uses_discriminating_operands(class_name, module_path):
    """The layout gate must not be fooled by a shape the transform preserves.

    This pins the gate itself: given a runtime dict, only the operands whose
    shape actually changed decide. A load that supplies changes only in the
    shape-preserving ``scales`` operand must still be classified as runtime.
    """

    from gptqmodel.nn_modules.qlinear._fused_train_state import (
        state_dict_matches_checkpoint_layout,
        state_dict_matches_runtime_layout,
    )

    cls = getattr(importlib.import_module(module_path), class_name)
    if torch.bfloat16 not in getattr(cls, "SUPPORTS_DTYPES", [torch.bfloat16]):
        pytest.skip(f"{class_name} does not support {torch.bfloat16}")

    holder, _x, _baseline = _transformed_layer(cls)
    layer = holder.proj
    runtime = {name: value.clone() for name, value in layer.state_dict().items()}
    assert state_dict_matches_runtime_layout(layer, runtime)
    assert not state_dict_matches_checkpoint_layout(layer, runtime)

    checkpoint = {
        name: layer._buffers[buffer_name].clone()
        for name, buffer_name in layer._fused_train_state.items()
    }
    assert state_dict_matches_checkpoint_layout(layer, checkpoint)
    assert not state_dict_matches_runtime_layout(layer, checkpoint)


@pytest.mark.parametrize(
    ("class_name", "module_path"),
    (
        ("TorchAtenLinear", "gptqmodel.nn_modules.qlinear.torch_aten_kernel"),
        ("TorchAtenAwqLinear", "gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq"),
        ("TorchFusedLinear", "gptqmodel.nn_modules.qlinear.torch_fused"),
        ("TorchFusedAwqLinear", "gptqmodel.nn_modules.qlinear.torch_fused_awq"),
        ("TorchCpuAvx512Linear", "gptqmodel.nn_modules.qlinear.torch_cpu_avx512"),
        ("TorchCpuAvx512AwqLinear", "gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq"),
    ),
)
@pytest.mark.parametrize("assign", (False, True))
def test_transformed_cpu_layer_accepts_partial_checkpoint_state(class_name, module_path, assign):
    from tests.test_torch_weight_storage import state_tensors

    cls = getattr(importlib.import_module(module_path), class_name)
    if torch.bfloat16 not in getattr(cls, "SUPPORTS_DTYPES", [torch.bfloat16]):
        pytest.skip(f"{class_name} does not support bfloat16")
    holder, x, baseline = _transformed_layer(cls)
    layer = holder.proj
    saved = layer._buffers[layer._fused_train_state["qweight"]].clone()
    missing = holder.load_state_dict({"proj.qweight": saved}, strict=False, assign=assign)
    assert "proj.qzeros" in missing.missing_keys
    assert torch.equal(layer.qweight, saved)
    with torch.inference_mode():
        actual = holder(x)
        holder(x)
    torch.testing.assert_close(actual.float(), baseline.float(), rtol=2e-3, atol=2e-3)
    assert all(tensor.numel() < layer.in_features * layer.out_features for tensor in state_tensors(layer))


@pytest.mark.parametrize("assign", (False, True))
def test_transformed_cpu_layer_keeps_partial_runtime_state(assign):
    from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
    from gptqmodel.utils.model import get_state_dict_for_save
    from tests.test_torch_weight_storage import state_tensors

    holder, x, _ = _transformed_layer(TorchAtenLinear)
    layer = holder.proj
    runtime_shape = tuple(layer.qweight.shape)
    updated = layer.qweight.clone()
    updated.view(-1)[0] ^= 1
    layer.load_state_dict({"qweight": updated}, strict=False, assign=assign)
    assert tuple(layer.qweight.shape) == runtime_shape
    assert torch.equal(layer.qweight, updated)
    with torch.inference_mode():
        assert torch.isfinite(holder(x)).all()
        holder(x)
    assert all(tensor.numel() < layer.in_features * layer.out_features for tensor in state_tensors(layer))
    with pytest.raises(RuntimeError, match="modified CPU int4pack runtime state"):
        get_state_dict_for_save(holder)


@pytest.mark.parametrize("assign", (False, True))
def test_transformed_cpu_layer_keeps_shared_shape_update(assign):
    from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
    from gptqmodel.utils.model import get_state_dict_for_save

    holder, x, baseline = _transformed_layer(TorchAtenLinear)
    layer = holder.proj
    runtime_shape = tuple(layer.qweight.shape)
    updated = layer.scales.clone() * 2
    layer.load_state_dict({"scales": updated}, strict=False, assign=assign)
    assert tuple(layer.qweight.shape) == runtime_shape
    assert torch.equal(layer.scales, updated)
    with torch.inference_mode():
        actual = holder(x)
    assert not torch.equal(actual, baseline)
    with pytest.raises(RuntimeError, match="modified CPU int4pack runtime state"):
        get_state_dict_for_save(holder)


def test_transformed_cpu_layer_rejects_mixed_layout_without_mutation():
    from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear

    holder, _x, _baseline = _transformed_layer(TorchAtenLinear)
    layer = holder.proj
    runtime_qweight = layer.qweight.clone()
    checkpoint_qweight = layer._buffers[layer._fused_train_state["qweight"]].clone()
    with pytest.raises(RuntimeError, match="mixed or invalid CPU int4pack"):
        layer.load_state_dict({"qweight": checkpoint_qweight, "qzeros": layer.qzeros.clone()}, strict=False)
    assert torch.equal(layer.qweight, runtime_qweight)
    assert layer._fused_train_state


def test_empty_partial_load_does_not_restore_transformed_layer():
    from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear

    holder, _x, _baseline = _transformed_layer(TorchAtenLinear)
    layer = holder.proj
    runtime_shape = tuple(layer.qweight.shape)
    layer.load_state_dict({}, strict=False)
    assert tuple(layer.qweight.shape) == runtime_shape
    assert layer._fused_train_state


@pytest.mark.parametrize(
    ("class_name", "module_path"),
    (
        ("TorchAtenLinear", "gptqmodel.nn_modules.qlinear.torch_aten_kernel"),
        ("TorchAtenAwqLinear", "gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq"),
        ("TorchFusedLinear", "gptqmodel.nn_modules.qlinear.torch_fused"),
        ("TorchFusedAwqLinear", "gptqmodel.nn_modules.qlinear.torch_fused_awq"),
        ("TorchCpuAvx512Linear", "gptqmodel.nn_modules.qlinear.torch_cpu_avx512"),
        ("TorchCpuAvx512AwqLinear", "gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq"),
    ),
)
def test_fused_cpu_group_preserves_checkpoint_save_layout(class_name, module_path):
    """Fusing CPU projections must not corrupt what ``save()`` writes.

    ``FusionKernelSpec.prepare()`` finalizes sources outside their own
    ``forward``, so it never hit the ``capture_fused_train_state`` call that a
    plain inference forward makes. The fused group then published int4pack
    shapes under the source checkpoint keys and the export path had no snapshot
    to restore, so a fused model serialized a checkpoint that no fresh model
    could load. ``fuse(False)`` inherited the same gap.
    """

    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel.models.fusion import FusionDecision, _FusionGroup
    from gptqmodel.utils.model import get_state_dict_for_save
    from tests.module_tree.test_fusion_kernel_backends import _projection

    cls = getattr(importlib.import_module(module_path), class_name)
    dtype = torch.bfloat16
    if not getattr(cls, "SUPPORTS_DTYPES", [dtype]) or dtype not in cls.SUPPORTS_DTYPES:
        pytest.skip(f"{class_name} does not support {dtype}")

    torch.manual_seed(31)
    config = LlamaConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=64,
        attention_bias=True,
        mlp_bias=True,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).to(dtype).eval()
    parent = model.model.layers[0].mlp
    modules = []
    for name in ("gate_proj", "up_proj"):
        projection, _weight, _bias = _projection(cls, getattr(parent, name), dtype, "cpu")
        setattr(parent, name, projection)
        modules.append(projection)

    def exported():
        return {
            name: tuple(entry.shape)
            for name, entry in get_state_dict_for_save(model).items()
            if name.endswith(("gate_proj.qweight", "gate_proj.qzeros", "gate_proj.scales"))
        }

    pristine = exported()
    x = torch.randn(2, 7, 512, dtype=dtype)
    with torch.inference_mode():
        baseline = parent(x).clone()

    decision = FusionDecision("gate_up", "mlp", ("mlp.gate_proj", "mlp.up_proj"))
    group = _FusionGroup(decision, parent, modules, activation_dtype=dtype)
    try:
        group.install()
        # Install finalizes the sources and publishes fused views; the export
        # must still describe the checkpoint it replaced.
        assert exported() == pristine
        with torch.inference_mode():
            fused = parent(x).clone()
        torch.testing.assert_close(fused.float(), baseline.float(), rtol=2e-3, atol=2e-3)
        assert exported() == pristine
    finally:
        group.remove()

    # Unfusing without an intervening export must land on checkpoint shapes too.
    assert exported() == pristine
    with torch.inference_mode():
        restored = parent(x).clone()
    torch.testing.assert_close(restored.float(), baseline.float(), rtol=2e-3, atol=2e-3)
