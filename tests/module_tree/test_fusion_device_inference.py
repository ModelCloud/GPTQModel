# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Actual CPU/CUDA/ROCm inference, with explicit vendor and executor identities."""

import weakref

import pytest
import torch

from gptqmodel.models._const import DEVICE, normalize_device
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from tests.benchmark_llama_fusion import storage_footprint
from tests.module_tree.test_llama_fusion import _wrapper


EXECUTORS = {
    "cpu_torch": (DEVICE.CPU, False),
    "cuda_torch": (DEVICE.CUDA, False),
    "cuda_marlin": (DEVICE.CUDA, True),
    "rocm_torch": (DEVICE.ROCM, False),
}


def _target(family):
    if family == DEVICE.CPU:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        pytest.skip(f"Actual {family.value} hardware/runtime required")
    device = torch.device("cuda:0")
    if normalize_device(device) != family:
        pytest.skip(f"This Torch runtime does not expose {family.value} devices")
    return device


def _cached_decode(model, ids):
    result = model(ids, use_cache=True)
    logits = [result.logits]
    for token in (4, 6, 42):
        result = model(torch.full_like(ids[:, :1], token), past_key_values=result.past_key_values, use_cache=True)
        logits.append(result.logits)
    return torch.cat(logits, dim=1)


def _assert_no_expanded_weight_cache(module):
    """Inspect owned containers, not just conventionally named cache attributes."""
    seen = set()

    def inspect_value(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            expanded = value.numel() == module.in_features * module.out_features
            assert not (expanded and (value.is_floating_point() or value.dtype in (torch.int8, torch.uint8)))
        elif isinstance(value, dict):
            for child in value.values():
                inspect_value(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                inspect_value(child)

    for value in vars(module).values():
        inspect_value(value)


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("method", ["gptq", "awq"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [2, 4], ids=["gqa", "mha"])
def test_symmetric_int4_fusion_inference_on_actual_device(executor, method, dtype, kv_heads, record_property):
    family, native = EXECUTORS[executor]
    device = _target(family)
    model, weights = _wrapper(method, dtype, device, native=native, canonical=True, kv_heads=kv_heads)
    assert model.quantize_config.bits == 4 and model.quantize_config.group_size == 128
    assert model.quantize_config.sym and not model.quantize_config.desc_act
    cls = ((MarlinLinear if method == "gptq" else AwqMarlinLinear) if native else
           (TorchLinear if method == "gptq" else AwqTorchLinear))
    ids = torch.tensor([[1, 5, 8, 3]], device=device)
    with torch.inference_mode():
        baseline = _cached_decode(model.model, ids)
        tokens = model.model.generate(ids, max_new_tokens=3, do_sample=False)
    original_storage = storage_footprint(model)
    planner = model.fuse()
    expected_groups = 2 if method == "awq" and kv_heads == 2 and not native else 4
    assert len(planner.groups) == expected_groups
    if expected_groups == 2:
        assert all("AWQ GQA" in d.reason for d in planner.decisions if not d.enabled)
    max_oracle_error = 0.0
    packed_paths = set()
    with torch.inference_mode():
        for group in planner.groups:
            assert type(group.packed) is cls
            assert normalize_device(group.packed.qweight.device) == family
            for rows in (1, 8, 32):
                x = torch.randn(1, rows, 128, dtype=dtype, device=device)
                expected = []
                for path in group.decision.paths:
                    key = f"{group.decision.parent}.{path.rsplit('.', 1)[-1]}"
                    weight, bias = weights[key]
                    expected.append((x.float() @ weight.float()).to(dtype) + bias)
                expected = torch.cat(expected, dim=-1)
                actual = group.packed(x)
                assert actual.dtype == dtype
                torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
                max_oracle_error = max(max_oracle_error, (actual.float() - expected.float()).abs().max().item())
                if isinstance(group.packed, TorchLinear):
                    flat = x.reshape(-1, 128)
                    if group.packed._can_use_rocm_gemv(flat):
                        packed_paths.add("triton_packed_gemv")
                    elif group.packed._can_use_rocm_gemm(flat):
                        packed_paths.add("triton_packed_gemm")
                if isinstance(group.packed, AwqTorchLinear) and group.packed._cpu_packed_qweight is not None:
                    packed_paths.add("cpu_avx512_awq")
            for module in group.modules:
                for name in ("qweight", "qzeros", "scales", "bias"):
                    operand, fused = getattr(module, name), getattr(group.packed, name)
                    if operand is not None and operand.numel():
                        assert operand.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()
            _assert_no_expanded_weight_cache(group.packed)
        for _ in range(2):
            fused = _cached_decode(model.model, ids)
            torch.testing.assert_close(fused, baseline, rtol=2e-3, atol=2e-3)
        generated = model.model.generate(ids, max_new_tokens=3, do_sample=False)
        assert torch.equal(generated, tokens)
    if method == "gptq" and not native and family != DEVICE.CPU:
        packed = planner.groups[0].packed
        if packed._triton_dequant_enabled and packed._rocm_arch_is_fused(torch.empty(0, device=device)):
            assert "triton_packed_gemv" in packed_paths
    assert planner.to_dict()["fused_calls"] > 0
    fused_storage = storage_footprint(model, planner)
    extra_packed_bytes = fused_storage["packed_operand_storage_bytes"] - original_storage["packed_operand_storage_bytes"]
    assert extra_packed_bytes == 0
    for group in planner.groups:
        _assert_no_expanded_weight_cache(group.packed)
        for module in group.modules:
            _assert_no_expanded_weight_cache(module)
    # Version guards must rebuild after a write through any source view.
    group = planner.groups[0]
    previous = group.packed
    scales = group.modules[0].scales.detach().clone()
    with torch.no_grad():
        group.modules[0].scales.mul_(0.5)
    with torch.inference_mode():
        model.model(ids)
    assert group.packed is not previous
    with torch.no_grad():
        group.modules[0].scales.copy_(scales)
    with torch.inference_mode():
        restored = _cached_decode(model.model, ids)
    torch.testing.assert_close(restored, baseline, rtol=2e-3, atol=2e-3)
    model.fuse(False)
    assert all(g.packed is None for g in planner.groups)
    assert all(module.qweight.is_contiguous() for g in planner.groups for module in g.modules)
    with torch.inference_mode():
        separate = _cached_decode(model.model, ids)
    torch.testing.assert_close(separate, baseline, rtol=0, atol=0)
    record_property("family", family.value)
    record_property("executor", executor)
    record_property("kernel", cls.__name__)
    record_property("dtype", str(dtype))
    record_property("attention", "gqa" if kv_heads == 2 else "mha")
    record_property("fused_groups", expected_groups)
    record_property("projection_oracle_max_abs", max_oracle_error)
    record_property("prefill_decode_max_abs", (fused.float() - baseline.float()).abs().max().item())
    record_property("generated_ids_equal", True)
    record_property("extra_packed_bytes", extra_packed_bytes)
    record_property("model_storage_delta_bytes", fused_storage["model_tensor_storage_bytes"] - original_storage["model_tensor_storage_bytes"])
    record_property("packed_paths", ",".join(sorted(packed_paths)))
    record_property("hardware", torch.cuda.get_device_name(device) if device.type != "cpu" else "x86_64 CPU")
    record_property("torch", torch.__version__)
    record_property("hip", str(torch.version.hip))


def test_retired_guard_releases_weight_bindings_even_when_function_is_retained():
    model, _ = _wrapper(automatic=True)
    plan = model.fuse()
    group = plan.groups[0]
    guard = group.guard
    weight = weakref.ref(group.packed.qweight)
    assert guard()
    model.fuse(False)
    assert weight() is None
    assert not guard()
    assert not guard.__globals__
