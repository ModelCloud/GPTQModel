# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The portable QQQ kernel must fuse decode and GEMM on the GPU.

``QQQTorchLinear`` is the AUTO kernel on ROCm and for BF16.  Before the fused
path it expanded the packed INT4 codes into an FP32/INT8 weight plane on every
forward, which made a 4096x4096 projection roughly 50-100x slower than dense
FP16.  These tests drive the real module forward on the target device and
compare it with an independent integer oracle; they also pin the packed
storage and the cache invalidation rules.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import gptqmodel.nn_modules.qlinear.qqq as qqq_module
from gptqmodel.quantization.qqq_activation import qqq_activation_quantize
from gptqmodel.utils.arch import tuning_family


pytestmark = pytest.mark.cuda

IN_FEATURES = 512
OUT_FEATURES = 256
GROUP_SIZE = 128
FUSED_FAMILIES = ("ada", "blackwell", "rdna4")


def _require_fused_family() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    family = tuning_family(torch.device("cuda"))
    if family not in FUSED_FAMILIES:
        pytest.skip(f"fused QQQ GEMM is not enabled on the {family!r} family")


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack logical ``[K, N]`` codes into the kernel's ``[K // 8, N]`` words."""

    packed = torch.zeros(
        (codes.shape[0] // 8, codes.shape[1]),
        dtype=torch.int32,
        device=codes.device,
    )
    for lane in range(8):
        packed.bitwise_or_(codes[lane::8].to(torch.int32) << (4 * lane))
    return packed


def _build(
    group_size: int,
    dtype: torch.dtype,
    *,
    bias: bool = False,
    in_features: int = IN_FEATURES,
    out_features: int = OUT_FEATURES,
    seed: int = 20261126,
):
    """Build a packed module plus the independent codes/scales behind it."""

    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
    with torch.no_grad():
        linear.weight.normal_(0, 0.05)
        if bias:
            linear.bias.normal_(0, 0.02)
    weight = linear.weight.data.float()

    if group_size == -1:
        scale = (weight.abs().amax(dim=1, keepdim=True) / 7.0).clamp_min(1e-8)
        scale = scale.to(torch.float32)
        codes = torch.round(weight / scale).clamp(-8, 7).to(torch.int32)
        quantized = (codes.float() * scale).to(dtype)
        s_extra = scale.reshape(-1)
        raw_group_scales = None
        passed_scales = scale
    else:
        groups = in_features // group_size
        passed_scales = torch.rand((out_features, groups), dtype=torch.float16) * 0.02 + 0.005
        s_extra = torch.rand(out_features, dtype=torch.float32) * 0.002 + 0.001
        raw_group_scales = passed_scales.t().float()
        codes = torch.empty(out_features, in_features, dtype=torch.int32)
        for k in range(in_features):
            codes[:, k] = (
                torch.round(weight[:, k] / raw_group_scales[k // group_size]).add(8).clamp(0, 15).to(torch.int32)
            )
        quantized = ((codes.float() - 8) * passed_scales.repeat_interleave(group_size, dim=1)).to(dtype)

    source = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
    source.weight.data.copy_(quantized)
    if bias:
        source.bias.data.copy_(linear.bias.data)
    module = qqq_module.QQQTorchLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        register_buffers=False,
    )
    module.pack(source, passed_scales, s_extra=s_extra)
    module.post_init()
    module = module.to(device="cuda").eval()
    return module, codes.t().contiguous(), raw_group_scales, s_extra


def _oracle(
    module: qqq_module.QQQTorchLinear,
    codes: torch.Tensor,
    raw_group_scales: torch.Tensor | None,
    s_extra: torch.Tensor,
    x: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Independent INT64 GEMM over the documented QQQ weight grid."""

    activation, activation_scale = qqq_activation_quantize(x.reshape(-1, x.shape[-1]).float())
    if raw_group_scales is None:
        weights = codes.long() * 16
        channel_scale = s_extra.double() / 16.0
    else:
        group_size = module.group_size
        # ``pack`` stores the group scale normalised by the channel scale.
        normalised = (raw_group_scales / s_extra.double()[None, :]).to(torch.float16)
        centred = (codes.double() - 8.0).reshape(-1, group_size, module.out_features)
        weights = torch.round(centred * normalised.double()[:, None, :]).reshape_as(centred)
        weights = weights.clamp(-128, 127).reshape(module.in_features, module.out_features).long()
        channel_scale = s_extra.double()
    accum = torch.matmul(activation.cpu().long(), weights.cpu()).double()
    scaled = (accum * activation_scale.cpu().double()) * channel_scale.cpu()
    alternate = (accum * channel_scale.cpu()) * activation_scale.cpu().double()
    result = torch.where(torch.isfinite(scaled), scaled, alternate)
    if module.bias is not None:
        result = result + module.bias.cpu().double()
    return result.to(device=x.device, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [GROUP_SIZE, -1])
@pytest.mark.parametrize("rows", [1, 16, 17])
def test_fused_forward_matches_the_integer_oracle(group_size, dtype, rows):
    _require_fused_family()
    module, codes, raw_scales, s_extra = _build(group_size, dtype)
    x = torch.randn(rows, IN_FEATURES, device="cuda", dtype=dtype)

    with torch.inference_mode():
        actual = module(x)
    expected = _oracle(module, codes, raw_scales, s_extra, x, dtype)

    assert module._torch_gpu_packed_cache is not None
    assert actual.dtype is dtype
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_fused_forward_is_the_selected_kernel(monkeypatch):
    _require_fused_family()
    module, _, _, _ = _build(GROUP_SIZE, torch.float16)
    x = torch.randn(2, IN_FEATURES, device="cuda", dtype=torch.float16)
    real = qqq_module._qqq_gpu_gemm()
    calls = []

    def recorded(*args, **kwargs):
        calls.append((args[0].dtype, args[0].shape, kwargs["group_size"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(qqq_module, "_qqq_gpu_gemm", lambda: recorded)
    with torch.inference_mode():
        module(x)

    assert calls == [(torch.int8, (2, IN_FEATURES), GROUP_SIZE)]


def test_fused_forward_keeps_packed_weight_storage():
    _require_fused_family()
    module, _, _, _ = _build(GROUP_SIZE, torch.float16)
    x = torch.randn(1, IN_FEATURES, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        module(x)

    packed, s_group, s_channel = module._torch_gpu_packed_cache
    assert packed.dtype is torch.int32
    assert packed.shape == (IN_FEATURES // 8, OUT_FEATURES)
    # Four bits per weight: the packed layout is not an expanded copy.
    assert packed.numel() * 4 == IN_FEATURES * OUT_FEATURES // 2
    assert s_group.dtype is torch.float16
    assert s_group.shape == (IN_FEATURES // GROUP_SIZE, OUT_FEATURES)
    assert s_channel.dtype is torch.float32
    assert s_channel.shape == (OUT_FEATURES,)
    assert module._torch_packed_cache is None
    assert module._torch_dropped_checkpoint is None
    assert any(tensor is packed for tensor in module.iter_runtime_weight_tensors())

    for name, tensor in module.named_buffers():
        assert not (
            tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and tensor.numel() >= IN_FEATURES * OUT_FEATURES
        ), f"{name} retains an expanded weight plane"


def test_fused_forward_does_not_build_the_expanded_weight(monkeypatch):
    """The old path's FP32/INT8 expansion must not run behind the fused call."""

    _require_fused_family()
    module, codes, raw_scales, s_extra = _build(GROUP_SIZE, torch.float16)
    x = torch.randn(2, IN_FEATURES, device="cuda", dtype=torch.float16)

    def fail():
        raise AssertionError("QQQTorchLinear materialised the expanded INT8 weight plane")

    monkeypatch.setattr(module, "_int8_weight_for_forward", fail)
    with torch.inference_mode():
        actual = module(x)
    expected = _oracle(module, codes, raw_scales, s_extra, x, torch.float16)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_fused_cache_follows_weight_and_state_changes():
    _require_fused_family()
    module, _, _, _ = _build(GROUP_SIZE, torch.float16)
    x = torch.randn(3, IN_FEATURES, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        module(x)
    first_key = module._torch_gpu_packed_key
    assert first_key is not None

    with torch.no_grad():
        module.B.add_(1)
    with torch.inference_mode():
        module(x)
    assert module._torch_gpu_packed_key != first_key
    rebuilt = _pack_codes(module._unpack_weight_codes())
    assert torch.equal(module._torch_gpu_packed_cache[0], rebuilt)

    state = module.state_dict()
    module.clear_weight_cache()
    assert module._torch_gpu_packed_cache is None
    assert module._torch_gpu_packed_key is None

    with torch.no_grad():
        module.B.zero_()
    module.load_state_dict(state)
    assert module._torch_gpu_packed_cache is None
    with torch.inference_mode():
        module(x)
    assert module._torch_gpu_packed_cache is not None

    module.cpu()
    assert module._torch_gpu_packed_cache is None


def test_fused_cache_build_orders_against_another_stream():
    _require_fused_family()
    module, codes, raw_scales, s_extra = _build(GROUP_SIZE, torch.float16)
    x = torch.randn(4, IN_FEATURES, device="cuda", dtype=torch.float16)

    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        first = module(x)
    # The cache was published on ``side``; this forward runs on the default
    # stream and must wait for the build instead of reading unwritten words.
    second = module(x)
    torch.cuda.synchronize()

    expected = _oracle(module, codes, raw_scales, s_extra, x, torch.float16)
    torch.testing.assert_close(first, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(second, expected, rtol=2e-3, atol=2e-3)


def test_fused_forward_keeps_a_wide_bias_once():
    _require_fused_family()
    module, _, _, _ = _build(GROUP_SIZE, torch.bfloat16, bias=True)
    with torch.no_grad():
        module.bias.fill_(65536.0)
        module.B.zero_()
    module.clear_weight_cache()
    x = torch.zeros(1, IN_FEATURES, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        actual = module(x)
    assert torch.isfinite(actual).all()
    assert actual[0, 0].item() == 65536.0


def test_portable_torch_path_remains_the_control(monkeypatch):
    _require_fused_family()
    module, codes, raw_scales, s_extra = _build(GROUP_SIZE, torch.float16)
    x = torch.randn(5, IN_FEATURES, device="cuda", dtype=torch.float16)
    monkeypatch.setattr(qqq_module, "_qqq_gpu_gemm", lambda: None)

    with torch.inference_mode():
        actual = module(x)
    expected = _oracle(module, codes, raw_scales, s_extra, x, torch.float16)

    assert module._torch_gpu_packed_cache is None
    assert actual.dtype is torch.float16
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
