# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ must stage and pack a module at the width `dynamic` assigns to it.

`AWQConfig` accepts a per-module width (`{"bits": 8}` on one layer of a 4-bit
run). The width was honoured when the quant module was created, but the staging
pass (`pseudo_quantize_tensor`) and the clip search read `qcfg.bits` directly,
so a W8 module staged 4-bit codes and packed them into an 8-bit container.
"""

import copy
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig


def _stub(bits=4, dynamic=None):
    class Stub:
        qcfg = AWQConfig(bits=bits, group_size=-1, sym=True, dynamic=dynamic)
        _module_bits = AWQProcessor._module_bits
        pseudo_quantize_tensor = AWQProcessor.pseudo_quantize_tensor
        _pseudo_quantize_tensor_into = AWQProcessor._pseudo_quantize_tensor_into

    return Stub()


def test_module_bits_follows_the_dynamic_override():
    stub = _stub(dynamic={"model.layers.0.mlp.down_proj": {"bits": 8}})
    assert stub._module_bits("model.layers.0.mlp.down_proj") == 8
    assert stub._module_bits("model.layers.1.mlp.down_proj") == 4
    assert stub._module_bits(None) == 4


def test_a_non_numeric_override_is_not_width_zero():
    """An excluded module (`bits: false`) or a `format`-only rule is not 2^0."""
    stub = _stub(
        dynamic={
            "-model.layers.0.mlp.up_proj": {"bits": 4},
            "model.layers.1.mlp.up_proj": {"group_size": 64},
        }
    )
    assert stub._module_bits("model.layers.0.mlp.up_proj") == 4
    assert stub._module_bits("model.layers.1.mlp.up_proj") == 4


def test_staging_uses_the_requested_width():
    """W8 staging must reach the 8-bit code range, not the 4-bit one."""
    stub = _stub()
    torch.manual_seed(0)
    weight = torch.randn(16, 64, dtype=torch.float32)

    wq4, scales4, _ = stub.pseudo_quantize_tensor(weight, bits=4)
    wq8, scales8, _ = stub.pseudo_quantize_tensor(weight, bits=8)

    # Codes are recoverable as wq / scale; the widest code has to fill each
    # signed range (7 for 4-bit, 127 for 8-bit).
    codes4 = (wq4 / scales4).round().abs().max().item()
    codes8 = (wq8 / scales8).round().abs().max().item()
    assert codes4 <= 8
    assert codes8 > 8
    assert codes8 <= 128

    # An 8-bit round trip must be far more accurate than a 4-bit one.
    err4 = (wq4 - weight).pow(2).mean().item()
    err8 = (wq8 - weight).pow(2).mean().item()
    assert err8 < err4 / 10


def test_tensor_into_matches_the_allocating_path():
    """The clip search writes into a preallocated buffer; it must agree."""
    stub = _stub()
    torch.manual_seed(1)
    weight = torch.randn(8, 64, dtype=torch.float32)
    for bits in (4, 8):
        expected, _, _ = stub.pseudo_quantize_tensor(weight, bits=bits)
        into = torch.empty_like(weight)
        stub._pseudo_quantize_tensor_into(weight, into, bits=bits)
        assert torch.allclose(into, expected, rtol=1e-6, atol=1e-6), f"bits={bits}"


def _processor(model, **config):
    processor = AWQProcessor(
        tokenizer=None, qcfg=AWQConfig(device="cpu", **config), calibration=None,
        prepare_dataset_func=None, calibration_concat_size=None, calibration_sort=None,
        batch_size=1, gptq_model=SimpleNamespace(
            model=model, qlinear_kernel=AwqTorchLinear, rotary_embedding=None, lm_head="lm_head",
        ), model=model, require_fwd=True, calculate_w_wq_diff=False,
        calibration_concat_separator=None,
    )
    processor.draw_progress = lambda *args: None
    processor.log_new_row = lambda *args: None
    return processor


def _oracle(weight, bits, sym, group_size):
    groups = weight.reshape(-1, group_size if group_size > 0 else weight.shape[-1])
    if sym:
        scale = groups.abs().amax(-1, keepdim=True).clamp(min=1e-5) / (2 ** (bits - 1) - 1)
        zero = torch.full_like(scale, 2 ** (bits - 1))
    else:
        scale = (groups.amax(-1, keepdim=True) - groups.amin(-1, keepdim=True)).clamp(min=1e-5)
        scale = scale / (2 ** bits - 1)
        zero = (-torch.round(groups.amin(-1, keepdim=True) / scale)).clamp(0, 2 ** bits - 1)
    codes = (torch.round(groups / scale) + zero).clamp(0, 2 ** bits - 1)
    reconstructed = ((codes - zero) * scale).reshape_as(weight)
    return reconstructed, scale.reshape(weight.shape[0], -1), zero.reshape(weight.shape[0], -1), codes.reshape_as(weight)


@pytest.mark.parametrize("bits", range(2, 9))
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_quantization_thresholds_match_independent_oracle(bits, sym, dtype):
    # Extrema force scale=1. Include exact half steps and adjacent input values.
    high = 2 ** (bits - 1) - 1 if sym else 2 ** bits - 2
    thresholds = torch.tensor([-0.5, 0.5], dtype=dtype)
    row = torch.cat((
        torch.tensor([-high if sym else -1, high, 0.0, -0.0], dtype=dtype),
        torch.nextafter(thresholds, torch.full_like(thresholds, -torch.inf)),
        thresholds,
        torch.nextafter(thresholds, torch.full_like(thresholds, torch.inf)),
    )).reshape(1, -1)
    stub = _stub()
    stub.qcfg.sym = sym
    expected, scales, zeros, codes = _oracle(row, bits, sym, -1)
    actual, actual_scales, actual_zeros = stub.pseudo_quantize_tensor(row, bits=bits)
    into = torch.empty_like(row)
    stub._pseudo_quantize_tensor_into(row, into, bits=bits)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(into, expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(actual_scales, scales)
    assert torch.equal(actual_zeros, zeros)
    assert torch.equal(torch.round(actual / actual_scales) + actual_zeros, codes)


class _ScalingGroup(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.a = torch.nn.Linear(16, 8, bias=False, device=device)
        self.b = torch.nn.Linear(16, 8, bias=False, device=device)

    def forward(self, x):
        return self.a(x) + self.b(x)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("widths", [(2, 3), (3, 2), (4, 8), (8, 4)])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("sym", [False, True])
@torch.inference_mode()
def test_mixed_width_scale_search_matches_independent_oracle(device, widths, chunked, sym):
    torch.manual_seed(42)
    group = _ScalingGroup(device)
    model = torch.nn.Sequential(group)
    processor = _processor(model, bits=4, group_size=16, sym=sym,
                           dynamic={"0.a": {"bits": widths[0]}, "0.b": {"bits": widths[1]}},
                           scale_search_chunked_activations=chunked)
    x = torch.randn(9, 16, device=device)
    reference = group(x)
    weights = [group.a.weight.clone(), group.b.weight.clone()]
    normalized = torch.cat([w.abs() / (w.abs().amax(-1, keepdim=True) + 1e-6) for w in weights])
    w_mean = normalized.mean(0)
    x_mean = x.abs().mean(0)
    losses, candidates = [], []
    for index in range(20):
        ratio = index / 20
        scale = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
        scale = scale / (scale.max() * scale.min()).sqrt()
        quantized = [_oracle(w * scale, bits, sym, 16)[0] / scale for w, bits in zip(weights, widths)]
        output = sum(torch.nn.functional.linear(x, w) for w in quantized)
        losses.append((reference - output).float().square().mean().item())
        candidates.append(scale)
    best = min(range(20), key=losses.__getitem__)
    _, _, actual, loss = processor._search_best_scale(
        group, group.a, [group.a, group.b], x, module2inspect=group,
    )
    torch.testing.assert_close(actual, candidates[best].cpu(), rtol=1e-6, atol=1e-6)
    assert loss == pytest.approx(losses[best], rel=1e-6, abs=1e-6)
    assert torch.equal(group.a.weight, weights[0])
    assert torch.equal(group.b.weight, weights[1])


def _pack_oracle(codes, bits):
    # Independent little-endian packing, including cross-word fields.
    if bits == 4:
        codes = codes.reshape(codes.shape[0], -1, 8)[..., [0, 2, 4, 6, 1, 3, 5, 7]].flatten(1)
    rows = []
    for row in codes.to(torch.int64).tolist():
        stream = sum(value << (index * bits) for index, value in enumerate(row))
        rows.append([(stream >> shift) & 0xFFFFFFFF for shift in range(0, len(row) * bits, 32)])
    return torch.tensor(rows, dtype=torch.int64).to(torch.int32)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_apply_pack_reload_and_forward_at_module_width(device, bits, sym, dtype):
    # Power-of-two scales make the staged codes exactly representable at both dtypes.
    torch.manual_seed(5)
    high = 2 ** (bits - 1) - 1 if sym else 2 ** bits - 2
    weight = torch.randint(-high if sym else -1, high + 1, (8, 32)).to(dtype)
    weight[:, 0] = -high if sym else -1
    weight[:, 1] = high
    weight.div_(64)
    linear = torch.nn.Linear(32, 8, bias=False, dtype=dtype)
    linear.weight.copy_(weight)
    model = torch.nn.Sequential(linear)
    processor = _processor(model, bits=4, group_size=-1, sym=sym, dynamic={"0": {"bits": bits}})
    named = NamedModule(linear, name="0", full_name="0", layer_index=0)
    expected, scales, zeros, codes = _oracle(weight, bits, sym, -1)
    processor.apply_quant({"0": named}, [])
    processor.pack_module(named)
    packed = model[0]
    assert isinstance(packed, AwqTorchLinear)
    assert packed.bits == bits
    assert torch.equal(packed.qweight, _pack_oracle(codes.T, bits))
    assert torch.equal(packed.qzeros, _pack_oracle(zeros.T, bits))
    assert torch.equal(packed.scales, scales.T)
    reloaded = AwqTorchLinear(bits=bits, group_size=-1, sym=sym, desc_act=False,
                             in_features=32, out_features=8, dtype=dtype, register_buffers=True)
    reloaded.load_state_dict(copy.deepcopy(packed.state_dict()))
    reloaded.to(device)
    reloaded.post_init()
    reloaded.eval()
    x = torch.randn(3, 32, dtype=dtype, device=device)
    oracle = torch.nn.functional.linear(x.float(), expected.to(device).float()).to(dtype)
    buffers_before = {name: tensor.clone() for name, tensor in reloaded.named_buffers()}
    for _ in range(2):
        actual = reloaded(x)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, oracle, rtol=2e-3, atol=2e-3)
        assert set(dict(reloaded.named_buffers())) == set(buffers_before)
        for name, tensor in reloaded.named_buffers():
            assert torch.equal(tensor, buffers_before[name])
        # No unregistered dense or expanded weights may be retained either.
        assert not any(isinstance(value, torch.Tensor) for value in vars(reloaded).values())


@pytest.mark.parametrize("sym", [True, False])
@torch.inference_mode()
def test_clip_search_uses_module_width(sym):
    from test_awq_clip_consistency import _sequential_clip_grid

    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64, bias=False))
    processor = _processor(model, bits=4, group_size=16, sym=sym, dynamic={"0": {"bits": 8}})
    x = torch.randn(11, 32)
    expected = _sequential_clip_grid(AWQConfig(bits=8, group_size=16, sym=sym), model[0].weight, x)
    [(name, actual)] = processor._search_best_clip(model, {"0": model[0]}, {"0": x})
    assert name == "0"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_gemm_kernel_selection_respects_module_width(bits):
    from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear

    processor = _processor(torch.nn.Module(), bits=4, dynamic={"proj": {"bits": bits}})
    processor.gptq_model.qlinear_kernel = AwqGEMMLinear
    expected = AwqGEMMLinear if bits == 4 else AwqTorchLinear
    assert processor._resolve_qlinear_kernel("proj") is expected
