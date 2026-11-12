# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# Credit: int8 kernel sync adapted from Yintong Lu (yintong-lu), vLLM PR #35697.

from __future__ import annotations

import weakref
from concurrent.futures import ThreadPoolExecutor
from time import sleep

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.torch_int8 import (
    INT8_GROUP_OFFSET_BUFFER_NAME,
    INT8_GROUP_SCALE_BUFFER_NAME,
    INT8_INPUT_PERM_BUFFER_NAME,
    INT8_WEIGHT_BUFFER_NAME,
    Int8PackedModule,
    TorchInt8Linear,
    _write_group_buffers,
)
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear
from gptqmodel.utils.model import get_state_dict_for_save
from tests.test_torch_weight_storage import build_case, state_tensors


@pytest.mark.parametrize("bits", [2, 4])
def test_lower_bit_int8_construction_is_rejected(bits):
    with pytest.raises(NotImplementedError, match="8-bit source weights"):
        TorchInt8Linear(bits=bits, group_size=32, sym=True, desc_act=False,
                        in_features=256, out_features=64)


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("backend", [BACKEND.GPTQ_TORCH_INT8, BACKEND.TORCH_INT8])
def test_lower_bit_int8_selection_is_rejected(bits, backend):
    with pytest.raises(ValueError, match="8-bit source weights"):
        select_quant_linear(bits=bits, group_size=32, desc_act=False, sym=True,
                            device=DEVICE.CPU, backend=backend, format=FORMAT.GPTQ_V2,
                            quant_method=METHOD.GPTQ, pack_dtype=torch.int32)


@pytest.mark.parametrize("bits", [2, 4])
def test_requantization_writer_rejects_expansion_before_work(bits, monkeypatch):
    module = torch.nn.Module()
    module.bits = bits

    def forbidden(*args, **kwargs):
        pytest.fail("A rejected low-bit conversion must not requantize weights")

    monkeypatch.setattr("gptqmodel.nn_modules.qlinear.torch_int8._write_group_buffers", forbidden)
    with pytest.raises(NotImplementedError, match="8-bit source weights"):
        _write_group_buffers(
            module,
            int8_weight_gnk=torch.zeros(1, 64, 32, dtype=torch.int8),
            group_scale=torch.ones(1, 64),
            group_offset=None,
            input_perm=None,
            group_lengths=(32,),
            dtype=torch.float32,
        )
    assert list(module.buffers()) == []


@pytest.mark.parametrize("entry", ["post_init", "transform_cpu", "forward"])
def test_low_bit_runtime_entry_points_cannot_bypass_guard(entry, monkeypatch):
    module = TorchInt8Linear(bits=8, group_size=32, sym=True, desc_act=False,
                            in_features=256, out_features=64).eval()
    module.bits = 4
    monkeypatch.setattr(module, "dequantize_weight",
                        lambda **kwargs: pytest.fail("Do not dequantize rejected source weights"))
    with pytest.raises(NotImplementedError, match="8-bit source weights"):
        if entry == "post_init":
            module.post_init()
        elif entry == "transform_cpu":
            module.transform_cpu(torch.float32)
        else:
            module(torch.ones(1, 256))
    assert not hasattr(module, "int8_weight_gnk")


def _checkpoint_weight(state):
    """Independent FP64 decode of a GPTQ checkpoint's ``[K, N]`` matrix."""
    shifts = torch.arange(0, 32, 8, dtype=torch.int64)
    qweight = state["qweight"].to(torch.int64)
    qzeros = state["qzeros"].to(torch.int64)
    codes = ((qweight[:, None, :] >> shifts[None, :, None]) & 255).reshape(-1, qweight.shape[1])
    zeros = ((qzeros[:, :, None] >> shifts) & 255).reshape(-1, qweight.shape[1])
    groups = state["g_idx"].long()
    return (codes - zeros[groups]).double() * state["scales"].double()[groups]


def _group_runtime_oracle(state):
    """Independent group-preserving INT8 layout for the same checkpoint."""
    shifts = torch.arange(0, 32, 8, dtype=torch.int64)
    qweight = state["qweight"].to(torch.int64)
    qzeros = state["qzeros"].to(torch.int64)
    codes = (((qweight[:, None, :] >> shifts[None, :, None]) & 255)
             .reshape(-1, qweight.shape[1])[: state["g_idx"].numel()])
    zeros = ((qzeros[:, :, None] >> shifts) & 255).reshape(-1, qweight.shape[1])
    groups = state["g_idx"].long()
    num_groups = zeros.shape[0]
    counts = torch.bincount(groups, minlength=num_groups).tolist()
    width = max(counts)
    perm = torch.argsort(groups, stable=True)
    ordered = codes.index_select(0, perm)
    packed = torch.zeros((num_groups, width, codes.shape[1]), dtype=torch.int64)
    start = 0
    for group, length in enumerate(counts):
        packed[group, :length] = ordered[start:start + length]
        start += length
    residual = (packed - 128).to(torch.int8).permute(0, 2, 1).contiguous()
    scale = state["scales"].double()
    offset = (128.0 - zeros.double()) * scale
    return residual, scale, (None if bool((offset == 0).all()) else offset), perm


@pytest.mark.parametrize("group_size", [-1, 16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_int8_group_layout_output_and_storage_oracle(group_size, dtype, monkeypatch):
    source, _ = build_case(bits=8, group_size=group_size)
    checkpoint = {name: tensor.clone() for name, tensor in source.state_dict().items()}
    candidate = TorchInt8Linear(bits=8, group_size=group_size, sym=False, desc_act=True,
                                in_features=source.in_features, out_features=source.out_features)
    candidate.load_state_dict(checkpoint, strict=True)
    references = []
    original = candidate._dequantize_gptq_source_weight

    def tracked(*args, **kwargs):
        dense = original(*args, **kwargs)
        references.append(weakref.ref(dense))
        return dense

    monkeypatch.setattr(candidate, "_dequantize_gptq_source_weight", tracked)
    candidate.post_init()
    candidate.eval()
    assert all(reference() is None for reference in references)

    residual, group_scale, group_offset, perm = _group_runtime_oracle(checkpoint)
    weight = _checkpoint_weight(checkpoint)
    torch.testing.assert_close(candidate.int8_weight_gnk, residual, rtol=0, atol=0)
    torch.testing.assert_close(candidate.int8_group_scale, group_scale.float(), rtol=0, atol=0)
    torch.testing.assert_close(candidate.int8_input_perm, perm, rtol=0, atol=0)
    if group_offset is None:
        assert candidate.int8_group_offset is None
    else:
        torch.testing.assert_close(candidate.int8_group_offset, group_offset.float(), rtol=0, atol=0)
    # The derived layout must reconstruct the checkpoint matrix bit-exactly:
    # this is storage repacking, not a second quantization.
    torch.testing.assert_close(
        candidate.dequantize_weight().double(), weight, rtol=0, atol=0
    )
    for name in candidate.GPTQ_BUFFER_NAMES:
        torch.testing.assert_close(getattr(candidate, name), checkpoint[name], rtol=0, atol=0)

    # The runtime scales are the checkpoint's own group scales, one row per
    # group, so the group-wise dynamic range is preserved rather than being
    # flattened into a single per-output-channel scale.
    assert candidate.int8_group_scale.shape == checkpoint["scales"].shape
    torch.testing.assert_close(candidate.int8_group_scale, checkpoint["scales"].float(), rtol=0, atol=0)

    for current_dtype in (dtype, torch.bfloat16, torch.float32, dtype):
        activation = torch.arange(7 * source.in_features).reshape(7, -1).remainder(9).sub(4).to(current_dtype) / 16
        arithmetic = activation.double() @ weight
        expected = arithmetic.to(current_dtype)
        with torch.inference_mode():
            output = candidate(activation)
        assert output.dtype == current_dtype
        torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)
        # The kernel reduces FP32 per group and rounds once at the end, so it
        # must stay inside half an output-dtype ulp plus FP32 accumulation.
        magnitude = activation.abs().double() @ weight.abs()
        bound = 0.5 * torch.finfo(current_dtype).eps * arithmetic.abs() + 8 * torch.finfo(torch.float32).eps * magnitude
        excess = (output.double() - arithmetic).abs() - bound
        assert int((excess > 0).sum()) == 0, (
            f"group_size={group_size} {current_dtype}: {(excess > 0).sum().item()}/{excess.numel()} "
            f"outputs exceed the FP32-reduction bound; max excess={excess.max().item():.3e}"
        )
        torch.testing.assert_close(candidate.int8_module.int8_group_scale, candidate.int8_group_scale,
                                   rtol=0, atol=0)
        for name in (INT8_WEIGHT_BUFFER_NAME, INT8_GROUP_SCALE_BUFFER_NAME,
                     INT8_GROUP_OFFSET_BUFFER_NAME, INT8_INPUT_PERM_BUFFER_NAME):
            buffer = getattr(candidate, name)
            if buffer is not None:
                assert buffer.untyped_storage().data_ptr() in {
                    tensor.untyped_storage().data_ptr() for tensor in state_tensors(candidate)
                }
        assert all(reference() is None for reference in references)


def test_native_int8_layout_preserves_a_small_group_inside_a_large_channel():
    """Regression for per-output-channel requantization collapsing a group.

    One output channel spans two groups whose checkpoint scales differ by 1024x
    (32.0 vs 0.03125). With a single per-channel scale the small group rounds to
    zero and the output becomes 32.0 instead of 32.03125; the group layout must
    reproduce the checkpoint exactly.
    """
    in_features, out_features, group_size = 128, 32, 64
    codes = torch.full((in_features, out_features), 128, dtype=torch.int64)
    codes[0, 0] = 128 + 127  # 32.0 with scale 32/127
    codes[group_size, 0] = 128 + 1  # 0.03125 with scale 32/127/1024
    zeros = torch.full((in_features // group_size, out_features), 128, dtype=torch.int64)
    scales = torch.full((in_features // group_size, out_features),
                        torch.tensor(32.0 / 127.0 / 1024.0, dtype=torch.float16))
    scales[0] = torch.tensor(32.0 / 127.0, dtype=torch.float16)
    groups = (torch.arange(in_features) // group_size).int()
    weight = (codes - zeros[groups]).double() * scales.double()[groups]
    assert float(scales.float()[0, 0].item()) / float(scales.float()[1, 0].item()) == 1024.0

    def pack(words):
        return ((words[..., 0] << 0) | (words[..., 1] << 8) | (words[..., 2] << 16)
                | (words[..., 3] << 24)).to(torch.int32).contiguous()
    module = TorchInt8Linear(bits=8, group_size=group_size, sym=True, desc_act=False,
                             in_features=in_features, out_features=out_features, bias=False)
    module.load_state_dict({
        "qweight": pack(codes.reshape(in_features // 4, 4, out_features).permute(0, 2, 1)),
        "qzeros": pack(zeros.reshape(-1, out_features // 4, 4)),
        "scales": scales,
        "g_idx": groups,
    }, strict=True)
    module.post_init()
    module.eval()

    activation = torch.ones(1, in_features)
    with torch.inference_mode():
        output = module(activation)
    exact = weight.sum(dim=0).expand(1, out_features)
    torch.testing.assert_close(output.double(), exact, rtol=0, atol=2e-6)
    # The historical per-output-channel requantization: the small group entry
    # rounds to zero and the output drops the small group's contribution.
    channel_scale = (weight.abs().amax(0) / 127).clamp_min(1e-10)
    collapsed = (weight / channel_scale).round().clamp(-128, 127) * channel_scale
    assert float(collapsed[group_size, 0].item()) == 0.0
    assert float((activation.double() @ collapsed)[0, 0].item()) == float(weight[0, 0].item())
    torch.testing.assert_close(module.dequantize_weight().double(), weight, rtol=0, atol=0)


def test_native_int8_ragged_last_group_keeps_exact_weights():
    """``in_features`` not divisible by ``group_size`` pads the last group."""
    in_features, out_features, group_size = 100, 64, 32
    generator = torch.Generator().manual_seed(4242)
    codes = torch.randint(0, 256, (in_features, out_features), generator=generator)
    zeros = torch.full((4, out_features), 128, dtype=torch.int64)
    scales = (torch.rand(4, out_features, generator=generator) * 0.01 + 1e-3).half()
    groups = (torch.arange(in_features) // group_size).int()
    weight = (codes - zeros[groups]).double() * scales.double()[groups]

    def pack(words):
        return ((words[..., 0] << 0) | (words[..., 1] << 8) | (words[..., 2] << 16)
                | (words[..., 3] << 24)).to(torch.int32).contiguous()
    module = TorchInt8Linear(bits=8, group_size=group_size, sym=True, desc_act=False,
                             in_features=in_features, out_features=out_features, bias=False)
    module.load_state_dict({
        "qweight": pack(codes.reshape(in_features // 4, 4, out_features).permute(0, 2, 1)),
        "qzeros": pack(zeros.reshape(-1, out_features // 4, 4)),
        "scales": scales,
        "g_idx": groups,
    }, strict=True)
    module.post_init()
    module.eval()

    assert module._int8_group_lengths == (32, 32, 32, 4)
    torch.testing.assert_close(module.dequantize_weight().double(), weight, rtol=0, atol=0)
    activation = torch.randn(3, in_features)
    with torch.inference_mode():
        output = module(activation)
    torch.testing.assert_close(output.double(), activation.double() @ weight, rtol=2e-3, atol=2e-3)
    # Pad columns must stay zero: they may not become an extra bias term.
    with torch.inference_mode():
        zero_output = module(torch.zeros(2, in_features))
    torch.testing.assert_close(zero_output, torch.zeros_like(zero_output), rtol=0, atol=0)


@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("assign", [False, True])
def test_native_int8_checkpoint_save_strict_reload_and_oracle(tmp_path, group_size, dtype, assign):
    source, _ = build_case(bits=8, group_size=group_size)
    constructor = {"bits": 8, "group_size": group_size, "sym": False, "desc_act": True,
                   "in_features": source.in_features, "out_features": source.out_features, "bias": True}
    checkpoint = {name: tensor.clone() for name, tensor in source.state_dict().items()}
    checkpoint["bias"] = torch.linspace(-0.25, 0.25, source.out_features, dtype=torch.float16)
    candidate = TorchInt8Linear(**constructor)
    candidate.load_state_dict(checkpoint, strict=True)
    candidate.post_init()
    candidate.eval()

    saved = candidate.state_dict()
    assert set(saved) == set(checkpoint)
    assert not any(name.startswith("int8_") for name in saved)
    for name, tensor in checkpoint.items():
        torch.testing.assert_close(saved[name], tensor, rtol=0, atol=0)
    owner = torch.nn.Module()
    owner.add_module("projection", candidate)
    writer_sources = get_state_dict_for_save(owner)
    assert set(writer_sources) == {f"projection.{name}" for name in checkpoint}
    path = tmp_path / "model.safetensors"
    save_file({name: entry.source.contiguous() for name, entry in writer_sources.items()}, str(path))
    restored = TorchInt8Linear(**constructor)
    restored_owner = torch.nn.Module()
    restored_owner.add_module("projection", restored)
    restored_owner.load_state_dict(load_file(str(path)), strict=True, assign=assign)
    restored.post_init()
    restored.eval()

    activation = torch.arange(7 * source.in_features).reshape(7, -1).remainder(11).sub(5).to(dtype) / 16
    weight = _checkpoint_weight(checkpoint)
    residual, group_scale, group_offset, perm = _group_runtime_oracle(checkpoint)
    for module in (candidate, restored):
        torch.testing.assert_close(module.int8_weight_gnk, residual, rtol=0, atol=0)
        torch.testing.assert_close(module.int8_group_scale, group_scale.float(), rtol=0, atol=0)
        torch.testing.assert_close(module.int8_input_perm, perm, rtol=0, atol=0)
        assert (module.int8_group_offset is None) == (group_offset is None)
        torch.testing.assert_close(module.dequantize_weight().double(), weight, rtol=0, atol=0)
    expected = (activation.double() @ weight + checkpoint["bias"].double()).to(dtype)
    with torch.inference_mode():
        actual = candidate(activation)
        reloaded = restored(activation)
        repeated = restored(activation)
    assert set(get_state_dict_for_save(owner)) == set(writer_sources)
    assert actual.dtype == reloaded.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(reloaded, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(reloaded, repeated, rtol=0, atol=0)
    assert restored.int8_weight_gnk.dtype == torch.int8
    assert not any(
        tensor.is_floating_point() and tuple(tensor.shape) in ((source.in_features, source.out_features),
                                                                 (source.out_features, source.in_features))
        for tensor in state_tensors(restored)
    )


@pytest.mark.parametrize("change", ["load", "load_assign", "inplace", "replace", "move", "clear"])
def test_native_int8_runtime_refreshes_after_checkpoint_change(change):
    source, _ = build_case(bits=8, group_size=32)
    constructor = {"bits": 8, "group_size": 32, "sym": False, "desc_act": True,
                   "in_features": source.in_features, "out_features": source.out_features}
    candidate = TorchInt8Linear(**constructor)
    candidate.load_state_dict(source.state_dict(), strict=True)
    candidate.post_init()
    candidate.eval()
    activation = torch.ones(3, source.in_features, dtype=torch.float32) / 8
    before = candidate(activation)

    if change in ("load", "load_assign"):
        state = {name: tensor.clone() for name, tensor in candidate.state_dict().items()}
        state["scales"].mul_(2)
        candidate.load_state_dict(state, strict=True, assign=change == "load_assign")
    elif change == "inplace":
        candidate.scales.mul_(2)
    elif change == "replace":
        candidate.scales = candidate.scales.clone() * 2
    elif change == "move":
        candidate.to(torch.float32)
    else:
        candidate.scales.mul_(2)
        candidate.clear_weight_cache()

    weight = _checkpoint_weight(candidate.state_dict())
    expected = (activation.double() @ weight)
    with torch.inference_mode():
        actual = candidate(activation)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-3, atol=2e-3)
    assert candidate.int8_module.int8_weight_gnk.data_ptr() == candidate.int8_weight_gnk.data_ptr()
    if change != "move":
        assert not torch.equal(before, actual)


def test_native_int8_inference_tensor_edit_requires_explicit_clear():
    source, _ = build_case(bits=8, group_size=32)
    candidate = TorchInt8Linear(bits=8, group_size=32, sym=False, desc_act=True,
                                in_features=source.in_features, out_features=source.out_features)
    with torch.inference_mode():
        state = {name: value.clone() for name, value in source.state_dict().items()}
    candidate.load_state_dict(state, strict=True, assign=True)
    candidate.post_init()
    candidate.eval()
    activation = torch.ones(3, source.in_features, dtype=torch.float32) / 8
    before = candidate(activation)
    with torch.inference_mode():
        candidate.scales.mul_(2)
    candidate.clear_weight_cache()
    weight = _checkpoint_weight(candidate.state_dict())
    expected = activation.double() @ weight
    actual = candidate(activation)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-3, atol=2e-3)
    assert not torch.equal(before, actual)


def test_native_int8_cold_refresh_builds_once_across_threads(monkeypatch):
    source, _ = build_case(bits=8, group_size=32)
    candidate = TorchInt8Linear(bits=8, group_size=32, sym=False, desc_act=True,
                                in_features=source.in_features, out_features=source.out_features)
    candidate.load_state_dict(source.state_dict(), strict=True)
    candidate.post_init()
    candidate.clear_weight_cache()
    original = candidate.transform_cpu
    calls = []

    def tracked(dtype):
        calls.append(dtype)
        sleep(0.01)
        original(dtype)

    monkeypatch.setattr(candidate, "transform_cpu", tracked)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: candidate._refresh_int8_runtime(), range(8)))
    assert calls == [torch.float32]


def test_native_int8_scale_cast_does_not_destroy_canonical_precision():
    # Two groups of four channels; the second group's scale is 1024x smaller.
    weight = torch.zeros((2, 8, 4), dtype=torch.int8)
    weight[0] = 1
    weight[1] = -1
    scales = torch.tensor([[1.0] * 8, [2.0 ** -10] * 8])
    module = Int8PackedModule(weight, scales.clone())
    module(torch.ones(1, 8, dtype=torch.bfloat16))
    torch.testing.assert_close(module.int8_group_scale, scales, rtol=0, atol=0)
    expected = torch.full((1, 8), 4.0 * 1.0 - 4.0 * 2.0 ** -10)
    torch.testing.assert_close(module(torch.ones(1, 8)), expected, rtol=1e-6, atol=1e-6)


def test_torch_int8_kernel_is_cpu_only():
    with pytest.raises(NotImplementedError):
        TorchInt8Linear.validate_device(DEVICE.XPU)


def test_torch_int8_supports_native_width_only():
    assert TorchInt8Linear.SUPPORTS_BITS == [8]
    supported, error = TorchInt8Linear.validate(bits=8, group_size=32, pack_dtype=torch.int32,
                                               device=DEVICE.CPU, dynamic={".*": {"bits": 4}})
    assert not supported and error is not None
