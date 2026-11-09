# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Execute finalized native GPTQ/AWQ layouts on actual CPU, CUDA and HIP.

Every registered GPTQ/AWQ backend is accounted for. Optional dependencies and
architecture exclusions are explicit skips, never portable-kernel substitutions.
"""

import weakref

import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel.models._const import DEVICE
from gptqmodel.models.fusion import FusionDecision, _FusionGroup, _incompatible
from gptqmodel.models.fusion_kernels import kernel_specs
from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear, LLMAwqLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils import importer
from gptqmodel.utils.awq import awq_runtime_available, awq_runtime_error
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.cpu_features import has_avx512_bf16
from gptqmodel.utils.marlin import marlin_runtime_available, marlin_runtime_error
from tests.module_tree.test_fusion_device_inference import _target


SPECS = kernel_specs()
# These layouts are deliberately not admitted by the executor. Keep the
# inventory exhaustive so new kernel registrations require an explicit review.
EXCLUDED = {
    "BitblasLinear": "Optional BitBLAS operator layout has no qualified fusion adapter",
    "AWQBitBlasKernel": "Optional BitBLAS operator layout has no qualified fusion adapter",
    "MacheteLinear": "Machete requires SM90+; this host is SM89",
    "AwqMacheteLinear": "Machete requires SM90+; this host is SM89",
    "SwordfishLinear": "Swordfish requires Blackwell",
    "AwqSwordfishLinear": "Swordfish requires Blackwell",
    "W4ANVFP4Linear": "NVFP4 requires SM121+",
    "TorchInt8AwqLinear": "Retired INT4-to-INT8 expanded-weight backend",
}


def test_every_registered_gptq_awq_kernel_has_an_explicit_fusion_contract():
    for cls in importer._registry_kernel_classes():
        if not set(importer.infer_quant_methods(cls)) & {METHOD.GPTQ, METHOD.AWQ}:
            continue
        if not set(cls.SUPPORTS_DEVICES) & {DEVICE.ALL, DEVICE.CPU, DEVICE.CUDA, DEVICE.ROCM}:
            continue  # Metal kernels require an Apple host, outside this matrix.
        assert cls in SPECS or cls.__name__ in EXCLUDED, cls.__name__


def test_zero_free_kernel_contracts_never_include_a_zero_operand():
    for cls, spec in SPECS.items():
        if spec.zero_name is None:
            # Padding these collections would make source rebinding consume
            # zero tensors that this finalized native layout does not own.
            assert set(dict(spec.axes)) == {spec.weight_name, spec.scale_name, "bias"}, cls.__name__


def _cases():
    for cls in sorted(SPECS, key=lambda cls: cls.__name__):
        for family in (DEVICE.CPU, DEVICE.CUDA, DEVICE.ROCM):
            if DEVICE.ALL not in cls.SUPPORTS_DEVICES and family not in cls.SUPPORTS_DEVICES:
                continue
            for dtype in (torch.float16, torch.bfloat16):
                if dtype in cls.SUPPORTS_DTYPES:
                    yield pytest.param(cls, family, dtype, id=f"{family.value}-{cls.__name__}-{dtype}")


def _contract(cls):
    spec = SPECS[cls]
    bits = 8 if cls.SUPPORTS_BITS == [8] else 4
    return bits, -1 if spec.layout == "allspark" else 128, torch.int16 if spec.layout == "awq_fast" else torch.int32


def _available(cls, family, dtype):
    device = _target(family)
    # A visible NVIDIA device does not imply that optional native extensions
    # can be loaded. Prebuilt ops still run without a local CUDA toolkit.
    if family == DEVICE.CUDA and CUDA_HOME is None:
        if cls in (AwqGEMVLinear, AwqGEMVFastLinear, LLMAwqLinear) and not awq_runtime_available():
            reason = awq_runtime_error().partition(" | ")[0]
            pytest.skip(f"Native AWQ extension unavailable: {reason}")
        if cls in (MarlinLinear, AwqMarlinLinear) and not marlin_runtime_available(dtype):
            reason = marlin_runtime_error(dtype).partition(" | ")[0]
            pytest.skip(f"Native Marlin extension unavailable for {dtype}: {reason}")
    bits, group_size, pack_dtype = _contract(cls)
    ok, error = cls.validate(bits=bits, group_size=group_size, sym=True, desc_act=False,
                             pack_dtype=pack_dtype, dtype=dtype, device=device,
                             in_features=512, out_features=1024)
    if not ok:
        pytest.skip(f"{cls.__name__} native contract unavailable: {error}")
    return device


def _projection(cls, linear, dtype, device, *, uniform_scales=False):
    """Independent logical integer matrices, finalized by the actual backend."""
    spec = SPECS[cls]
    k, n = linear.in_features, linear.out_features
    bits, group, pack_dtype = _contract(cls)
    group_width = k if group == -1 else group
    codes = torch.randint(0, 1 << bits, (k, n))
    zeros = torch.full((k // group_width, n), 1 << (bits - 1), dtype=torch.int32)
    # Vary metadata across both groups and output channels; uniform scales can
    # hide a wrong native stride or tile permutation.
    scales = (torch.full(zeros.shape, 1 / (64 * (1 << bits)), dtype=dtype) if uniform_scales else
              (2.0 ** torch.randint(-2, 1, zeros.shape).float() / (64 * (1 << bits))).to(dtype))
    indices = torch.arange(k, dtype=torch.int32) // group_width
    weight = ((codes - zeros[indices]) * scales[indices]).to(dtype)
    linear.weight.data.copy_(weight.t())
    linear.bias.data.uniform_(-0.01, 0.01)
    kwargs = {"bits": bits, "group_size": group, "pack_dtype": pack_dtype, "sym": True, "desc_act": False,
                  "in_features": k, "out_features": n, "bias": True, "register_buffers": True, "dtype": dtype}
    if spec.gptq:
        kwargs["format"] = FORMAT.GPTQ_V2
    module = cls(**kwargs)
    if spec.gptq or cls is AwqMarlinLinear or not callable(getattr(module, "pack", None)):
        source = (TorchLinear if spec.gptq else AwqTorchLinear)(**kwargs)
        if spec.gptq:
            source.pack_block(linear, scales.t(), zeros.t().float(), indices)
        else:
            source.pack(linear, scales.t(), zeros.t(), indices)
        for name in ("qweight", "qzeros", "scales", "bias", "g_idx"):
            src, dst = getattr(source, name, None), getattr(module, name, None)
            if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor) and src.shape == dst.shape:
                dst.data.copy_(src)
    else:
        module.pack(linear, scales.t(), zeros.t(), indices)
    module.bias = module.bias.detach()
    if spec.gptq:
        module.qzero_format(2)
    module.to(device).eval()
    if spec.layout == "exllamav2":
        module.post_init(None)
    else:
        module.post_init()
    if spec.layout == "int8":
        # This backend requantizes native eight-bit sources before fusion.
        # The inference oracle decodes its resulting native codes independently.
        weight = module.int8_weight_nk.float().t() * module.int8_channel_scale.float()
    return module, weight.to(device), linear.bias.detach().to(device).clone()


def _assert_no_dense_weights(module, spec):
    seen = set()

    def inspect(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.numel() == module.in_features * module.out_features:
                assert not value.is_floating_point()
                assert module.bits == 8 or value.dtype not in (torch.int8, torch.uint8)
        elif isinstance(value, torch.nn.Module):
            inspect(vars(value))
        elif isinstance(value, dict):
            for child in value.values():
                inspect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                inspect(child)

    inspect(module)


def _oracle(x, weights, biases, spec):
    values = x.float()
    if spec.layout == "gptq_fp8":
        maximum = values.abs().amax(-1, keepdim=True)
        scale = torch.where(maximum > 0, maximum / 448, 1.0)
        values = (values / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
    # Compute the mathematical result independently in FP32, then round once
    # to the visible dtype. Mirroring a kernel's intermediate BF16 rounding
    # can make the oracle itself differ by a full BF16 step near a bias tie.
    return torch.cat([(values @ w.float() + b.float()).to(x.dtype) for w, b in zip(weights, biases)], -1)


@pytest.mark.parametrize("cls,family,dtype", list(_cases()))
@pytest.mark.parametrize("profile", ["gate_up", "qkv_gqa", "qkv_mha"])
def test_native_fusion_shapes_oracle_ownership_and_invalidation(cls, family, dtype, profile, record_property):
    _check_native_fusion(cls, family, dtype, profile, record_property)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_uniform_scales_large_m_and_fp32_bias_oracle(dtype, record_property):
    from gptqmodel.nn_modules.qlinear.w4a_floatx import W4AFP8Linear

    _check_native_fusion(W4AFP8Linear, DEVICE.CUDA, dtype, "gate_up", record_property, uniform_scales=True)


def _check_native_fusion(cls, family, dtype, profile, record_property, *, uniform_scales=False):
    device = _available(cls, family, dtype)
    spec = SPECS[cls]
    torch.manual_seed(17)
    config = LlamaConfig(hidden_size=512, intermediate_size=1024, num_hidden_layers=1,
                         num_attention_heads=8, num_key_value_heads=8 if profile == "qkv_mha" else 4,
                         vocab_size=64, attention_bias=True, mlp_bias=True)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).to(dtype).eval()
    parent_name = "mlp" if profile == "gate_up" else "self_attn"
    parent = getattr(model.model.layers[0], parent_name)
    names = ("gate_proj", "up_proj") if profile == "gate_up" else ("q_proj", "k_proj", "v_proj")
    modules, weights, biases = [], [], []
    for name in names:
        module, weight, bias = _projection(cls, getattr(parent, name), dtype, device, uniform_scales=uniform_scales)
        setattr(parent, name, module)
        modules.append(module)
        weights.append(weight)
        biases.append(bias)
    parent.to(device)

    def invoke(x):
        if parent_name == "mlp":
            return parent(x)
        cos = torch.ones(1, x.shape[1], 64, device=device, dtype=dtype)
        return parent(x, position_embeddings=(cos, torch.zeros_like(cos)), attention_mask=None)[0]

    rows = (1, 8, 33, 257, 1025) if profile == "gate_up" else (1, 8, 33, 257)
    xs = [torch.randn(1, size, 512, device=device, dtype=dtype) for size in rows]
    with torch.inference_mode():
        before = [invoke(x) for x in xs]
        repeat = [invoke(x) for x in xs]
    baseline_noise = max((a.float() - b.float()).abs().max().item() for a, b in zip(before, repeat))
    snapshots = [{name: getattr(m, name).clone() for name, _ in spec.axes
                  if isinstance(getattr(m, name, None), torch.Tensor)} for m in modules]
    reason = _incompatible(modules)
    if cls is AwqTorchLinear and profile == "qkv_gqa":
        assert "AWQ GQA" in reason
        record_property("declared_rejection", reason)
        return
    assert not reason, reason
    if spec.layout == "gptq_fp8":
        modules[-1]._require_activation_stream = True
        assert "encoded W4A" in _incompatible(modules)
        del modules[-1]._require_activation_stream
    width = modules[-1].out_features
    modules[-1].out_features += 1
    assert _incompatible(modules), "A non-native output shape must be rejected before a launch"
    modules[-1].out_features = width
    refs = [weakref.ref(getattr(m, spec.weight_name)) for m in modules]
    decision = FusionDecision("gate_up" if profile == "gate_up" else "qkv", parent_name,
                              tuple(f"{parent_name}.{name}" for name in names))
    group = _FusionGroup(decision, parent, modules, activation_dtype=dtype)
    max_oracle = max_drift = 0.0
    try:
        group.install()
        assert type(group.packed) is cls
        assert all(ref() is None for ref in refs), "Original native weight allocations were retained"
        with torch.inference_mode():
            for x, baseline in zip(xs, before):
                actual = invoke(x)
                torch.testing.assert_close(actual, baseline, rtol=2e-3, atol=2e-3)
                combined = group.packed(x)
                expected = _oracle(x, weights, biases, spec)
                assert combined.dtype == dtype
                torch.testing.assert_close(combined, expected, rtol=2e-3, atol=2e-3)
                direct = torch.cat([m(x) for m in modules], -1)
                torch.testing.assert_close(direct, combined, rtol=2e-3, atol=2e-3)
                max_oracle = max(max_oracle, (combined.float() - expected.float()).abs().max().item())
                max_drift = max(max_drift, (actual.float() - baseline.float()).abs().max().item())
            empty = group.packed(torch.empty(1, 0, 512, device=device, dtype=dtype))
            assert empty.shape == (1, 0, sum(m.out_features for m in modules))
        assert group.calls == len(xs)
        for module in modules:
            for name, _ in spec.axes:
                source, packed = getattr(module, name), getattr(group.packed, name)
                if isinstance(source, torch.Tensor) and source.numel():
                    assert source.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
            _assert_no_dense_weights(module, spec)
        _assert_no_dense_weights(group.packed, spec)

        # Rebuild after an in-place write through the final sibling's view.
        previous = group.packed
        scale = getattr(modules[-1], spec.scale_name)
        with torch.no_grad():
            scale.mul_(0.5)
        assert not group.guard()
        with torch.inference_mode():
            invoke(xs[0])
            torch.testing.assert_close(group.packed(xs[0]), torch.cat([m(xs[0]) for m in modules], -1),
                                       rtol=2e-3, atol=2e-3)
        assert group.packed is not previous
        with torch.no_grad():
            getattr(modules[-1], spec.scale_name).mul_(2)
        with torch.inference_mode():
            invoke(xs[0])
        # Explicit clear also covers unversioned inference-tensor mutations.
        modules[0].clear_weight_cache()
        assert group.packed is None
        with torch.inference_mode():
            invoke(xs[0])
        assert group.packed is not None
    finally:
        group.remove()
    with torch.inference_mode():
        for x, baseline in zip(xs, before):
            torch.testing.assert_close(invoke(x), baseline, rtol=2e-3, atol=2e-3)
    for module, snapshot in zip(modules, snapshots):
        for name, expected in snapshot.items():
            actual = getattr(module, name)
            assert actual.is_contiguous()
            assert torch.equal(actual, expected), name
    record_property("family", family.value)
    record_property("kernel", cls.__name__)
    record_property("dtype", str(dtype))
    record_property("profile", profile)
    record_property("bits", modules[0].bits)
    record_property("uniform_scales", uniform_scales)
    record_property("projection_oracle_max_abs", max_oracle)
    record_property("parent_max_abs", max_drift)
    record_property("baseline_repeat_max_abs", baseline_noise)
    record_property("extra_packed_weight_bytes", 0)


@pytest.mark.parametrize("method", ["gptq", "awq"])
@pytest.mark.parametrize("mutation", ["replace", "resize", "dtype"])
def test_marlin_rebuilds_native_workspace_after_binding_changes(method, mutation):
    device = _target(DEVICE.CUDA)
    from tests.module_tree.test_llama_fusion import _wrapper

    model, _ = _wrapper(method, device=device, native=True, automatic=True)
    plan = model.fuse()
    group = plan.groups[0]
    previous = group.packed
    if mutation == "replace":
        group.packed.workspace = torch.empty(0, device=device, dtype=torch.int32)
    elif mutation == "resize":
        group.packed.workspace.resize_(0)
    else:
        group.packed.workspace = group.packed.workspace.half()
    assert not group.guard()
    with torch.inference_mode():
        model.model(torch.tensor([[1, 2]], device=device))
    assert group.packed is not previous
    workspace = group.packed.workspace
    assert workspace.dtype == torch.int32
    assert workspace.numel() >= max(128, torch.cuda.get_device_properties(device).multi_processor_count)
    assert torch.count_nonzero(workspace) == 0
    model.fuse(False)


@pytest.mark.parametrize("family", [DEVICE.CPU, DEVICE.CUDA, DEVICE.ROCM])
@pytest.mark.parametrize("method,fmt", [(METHOD.GPTQ, FORMAT.GPTQ_V2), (METHOD.AWQ, FORMAT.GEMM)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_auto_keeps_the_native_optimized_kernel(family, method, fmt, dtype, record_property):
    device = _target(family)
    selected = importer.select_quant_linear(bits=4, group_size=128, sym=True, desc_act=False,
                                            pack_dtype=torch.int32, dtype=dtype, device=device,
                                            backend=BACKEND.AUTO, format=fmt, quant_method=method)
    cpu_avx512 = has_avx512_bf16()
    expected = {
        (DEVICE.CPU, METHOD.GPTQ): ("TorchAtenLinear", "TorchCpuAvx512Linear" if cpu_avx512 else "TorchAtenLinear"),
        (DEVICE.CPU, METHOD.AWQ): ("TorchAtenAwqLinear", "TorchCpuAvx512AwqLinear" if cpu_avx512 else "TorchAtenAwqLinear"),
        (DEVICE.CUDA, METHOD.GPTQ): ("MarlinLinear", "MarlinLinear"),
        (DEVICE.CUDA, METHOD.AWQ): ("AwqMarlinLinear", "AwqMarlinLinear"),
        (DEVICE.ROCM, METHOD.GPTQ): ("TorchLinear", "TorchLinear"),
        (DEVICE.ROCM, METHOD.AWQ): ("AwqGEMMTritonLinear", "AwqTorchLinear"),
    }[family, method][dtype == torch.bfloat16]
    # AVX512-BF16 is hardware-specific; AUTO must preserve its native selection when available.
    assert selected.__name__ == expected
    assert selected in SPECS
    record_property("auto_kernel", expected)
