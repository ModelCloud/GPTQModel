# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.quantization.config import (
    Fallback,
    FallbackStrategy,
    MSEConfig,
    QQQConfig,
    QuantizeConfig,
    SmoothMSE,
    SmoothRowCol,
)
from gptqmodel.quantization.qqq import QQQ, Quantizer, quantize


def _raw_bytes(tensor):
    return tensor.detach().contiguous().view(torch.uint8)


def _legacy_qqq_quantize(x, scale, zero, maxq, sym, groupsize):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    if groupsize != -1 or not sym:
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)
    q = torch.clamp(torch.round(x / scale), -maxq, maxq)
    return scale * q


def _reference_qqq_find_params(
    x,
    *,
    bits,
    sym,
    groupsize,
    mse,
    norm=2.4,
    grid=100,
    maxshrink=0.8,
):
    x = x.flatten(1)
    maxq = 2**bits - 1 if groupsize != -1 or not sym else 2 ** (bits - 1) - 1
    # Tensor division has different rounding from scalar reciprocal-multiply
    # on accelerators. Keep the reference's FP32 division in the same domain
    # as the stored maxq buffer, without calling the production fitter.
    maxq_tensor = torch.tensor(maxq, dtype=torch.int64, device=x.device)
    tmp = torch.zeros(x.shape[0], device=x.device)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)

    if sym:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        # The stored zero point is fixed at ``(maxq + 1) // 2``, so the range
        # must be zero-centered for non-negative groups too.
        xmin = -xmax
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1

    if maxq < 0:
        scale = xmax
        zero = xmin
    elif groupsize != -1 or not sym:
        scale = (xmax - xmin) / maxq_tensor
        zero = torch.full_like(scale, (maxq + 1) / 2) if sym else torch.round(-xmin / scale)
    else:
        scale = xmax / maxq_tensor
        zero = torch.zeros_like(scale)

    if mse:
        best = torch.full([x.shape[0]], float("inf"), device=x.device)
        for index in range(int(maxshrink * grid)):
            p = 1 - index / grid
            xmin1 = p * xmin
            xmax1 = p * xmax
            scale1 = (xmax1 - xmin1) / maxq_tensor if groupsize != -1 or not sym else xmax1 / maxq_tensor
            zero1 = torch.round(-xmin1 / scale1) if not sym else zero
            q = _legacy_qqq_quantize(
                x,
                scale1.unsqueeze(1),
                zero1.unsqueeze(1),
                maxq,
                sym,
                groupsize,
            )
            q -= x
            q.abs_()
            q.pow_(norm)
            err = torch.sum(q, 1)
            better = err < best
            if torch.any(better):
                best[better] = err[better]
                scale[better] = scale1[better]
                zero[better] = zero1[better]

    return scale, zero


def _cases(dtype):
    generator = torch.Generator().manual_seed(0)
    cases = {"randn": torch.randn(16, 64, generator=generator)}
    cases["all_zero"] = torch.zeros(16, 64)
    # Non-negative groups are the regression case for zero-centered fitting.
    cases["all_positive"] = torch.rand(16, 64, generator=generator) + 0.25
    cases["all_negative"] = -(torch.rand(16, 64, generator=generator) + 0.25)
    cases["constant_positive"] = torch.full((16, 64), 2.5)

    mixed = torch.randn(16, 64, generator=generator)
    mixed[::3] = 0.0
    mixed[1::5] = -mixed[1::5].abs()
    mixed[2::7] = mixed[2::7].abs()
    mixed[3::11] = -0.0
    cases["mixed_signs"] = mixed

    wide = torch.randn(16, 64, generator=generator) * 1e4
    wide[5] = 0.0
    cases["wide"] = wide

    centers = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=dtype)
    ulps = torch.nextafter(centers, torch.full_like(centers, float("inf"))) - centers
    thresholds = torch.empty(16, 64, dtype=dtype)
    for row in range(16):
        for column in range(64):
            index = (row + column) % centers.numel()
            thresholds[row, column] = centers[index] + (column % 3 - 1) * ulps[index]
    thresholds[0, 0] = -0.0
    cases["rounding_thresholds"] = thresholds

    return {name: tensor.to(dtype) for name, tensor in cases.items()}


class _DispatchRecorder(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def _assert_sync_free(ops, context):
    banned = ("aten.nonzero", "aten.index_put", "aten._local_scalar_dense", "aten.masked_scatter")
    offenders = [entry for entry in ops if entry.startswith(banned)]
    assert not offenders, (context, offenders)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("sym,groupsize", [(True, -1), (True, 128), (False, -1)])
@pytest.mark.parametrize("mse", [False, True])
def test_qqq_find_params_stays_bit_exact_and_sync_free(dtype, bits, sym, groupsize, mse):
    for name, weights in _cases(dtype).items():
        quantizer = Quantizer()
        quantizer.configure(
            bits=bits,
            perchannel=True,
            sym=sym,
            groupsize=groupsize,
            mse=mse,
            norm=2.4,
        )

        recorder = _DispatchRecorder()
        with recorder:
            quantizer.find_params(weights, weight=True)

        expected_scale, expected_zero = _reference_qqq_find_params(
            weights.clone(),
            bits=bits,
            sym=sym,
            groupsize=groupsize,
            mse=mse,
        )
        assert torch.equal(_raw_bytes(quantizer.scale.reshape(-1)), _raw_bytes(expected_scale)), (name, "scale")
        assert torch.equal(_raw_bytes(quantizer.zero.reshape(-1)), _raw_bytes(expected_zero)), (name, "zero")
        _assert_sync_free(recorder.ops, (name, "find_params"))

        recorder = _DispatchRecorder()
        with recorder:
            actual = quantize(
                weights,
                quantizer.scale,
                quantizer.zero,
                quantizer._maxq_value,
                quantizer.sym,
                quantizer.groupsize,
            )
        expected = _legacy_qqq_quantize(
            weights,
            quantizer.scale,
            quantizer.zero,
            quantizer._maxq_value,
            quantizer.sym,
            quantizer.groupsize,
        )
        assert torch.equal(_raw_bytes(actual), _raw_bytes(expected)), (name, "quantize")
        _assert_sync_free(recorder.ops, (name, "quantize"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("maxq", [-1, 1, 3, 7, 15])
@pytest.mark.parametrize("sym,groupsize", [(True, -1), (True, 128), (False, 128)])
def test_qqq_quantize_host_maxq_matches_tensor_bound(dtype, maxq, sym, groupsize):
    torch.manual_seed(0)
    x = torch.randn(16, 32, dtype=dtype)
    scale = torch.rand(16, 1, dtype=dtype) * 0.1
    zero = torch.randn(16, 1, dtype=dtype) * 2.0

    expected = quantize(
        x,
        scale,
        zero,
        torch.tensor(maxq, dtype=torch.float32),
        sym,
        groupsize,
    )

    recorder = _DispatchRecorder()
    with recorder:
        actual = quantize(x, scale, zero, maxq, sym, groupsize)

    assert torch.equal(_raw_bytes(actual), _raw_bytes(expected))
    _assert_sync_free(recorder.ops, (maxq, sym, groupsize))


def test_qqq_ready_uses_host_flag_without_device_sync():
    """Readiness must not reduce the scale buffer on the device."""

    quantizer = Quantizer()
    quantizer.configure(bits=4, perchannel=True, sym=True, groupsize=128, mse=False)
    x = torch.randn(16, 64)

    assert quantizer.ready() is False

    recorder = _DispatchRecorder()
    with recorder:
        unconfigured = quantizer.quantize(x)
    assert unconfigured is x
    _assert_sync_free(recorder.ops, "unconfigured quantize")

    quantizer.find_params(x, weight=True)
    assert quantizer.ready() is True

    recorder = _DispatchRecorder()
    with recorder:
        quantizer.quantize(x)
    _assert_sync_free(recorder.ops, "configured quantize")


@pytest.mark.parametrize("bits,sym,groupsize", [(4, True, -1), (4, False, -1), (2, True, 128)])
def test_qqq_ready_after_find_params_always_has_nonzero_scale(bits, sym, groupsize):
    """The host flag is only equivalent if ``find_params`` never leaves a zero scale."""

    quantizer = Quantizer()
    quantizer.configure(bits=bits, perchannel=True, sym=sym, groupsize=groupsize, mse=False)
    quantizer.find_params(torch.zeros(8, 32), weight=True)

    assert quantizer.ready() is True
    assert bool(torch.all(quantizer.scale != 0))


def test_qqq_ready_flag_survives_deepcopy():
    import copy

    quantizer = Quantizer()
    quantizer.configure(bits=4, perchannel=True, sym=True, groupsize=128, mse=False)
    quantizer.find_params(torch.randn(16, 64), weight=True)

    clone = copy.deepcopy(quantizer)
    assert clone.ready() is True


def test_qqq_hessian_reduction_uses_the_direction_aware_move(monkeypatch):
    """The per-device partial merge must not issue a raw cross-device copy.

    Partials are keyed by the device that captured them, so the merge can move
    a tensor between accelerators. On an asymmetric ROCm peer topology that
    direct copy segfaults inside ``hipMemcpyPeerAsync``, so the merge goes
    through ``safe_move_to`` even for a plain dtype cast.
    """
    from gptqmodel.quantization import qqq as qqq_module

    quantizer = qqq_module.QQQ(torch.nn.Linear(6, 8, bias=False))

    # A float64 partial forces the cast branch; the real device move is covered
    # by the P2P tests.
    partial = torch.full((6, 6), 8.0, dtype=torch.float64)
    quantizer._device_hessian_partials[torch.device("cpu")] = partial
    quantizer._device_sample_counts[torch.device("cpu")] = 4
    quantizer.nsamples = 4

    calls = []

    def fake_move(tensor, device, *, dtype=None, **kwargs):
        calls.append((tensor.dtype, torch.device(device), dtype))
        return tensor.to(dtype=dtype)

    monkeypatch.setattr(qqq_module, "safe_move_to", fake_move)

    hessian = quantizer.materialize_hessian(torch.device("cpu"))

    assert calls == [(torch.float64, torch.device("cpu"), torch.float32)]
    assert torch.allclose(hessian, torch.full((6, 6), 4.0, dtype=torch.float32))


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable",
    )),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bits,sym,groupsize", [
    (2, False, 32), (3, True, -1), (4, True, 32), (8, False, -1),
])
@pytest.mark.parametrize("budget", [0, 1])
@pytest.mark.parametrize("non_negative", [False, True])
def test_qqq_mse_grid_chunks(device, dtype, bits, sym, groupsize, budget, non_negative):
    weights = torch.randn(128, 32, generator=torch.Generator().manual_seed(7))
    weights = weights.to(device=device, dtype=dtype)[::2, ::2].t()
    if non_negative:
        weights = weights.abs()
    quantizer = Quantizer(
        qcfg=QuantizeConfig(mse=MSEConfig(norm=2.4, grid_budget=budget)),
    )
    quantizer.configure(
        bits=bits, perchannel=True, sym=sym, groupsize=groupsize, mse=True,
    )
    quantizer.find_params(weights, weight=True)
    scale, zero = _reference_qqq_find_params(
        weights, bits=bits, sym=sym, groupsize=groupsize, mse=True,
    )
    torch.testing.assert_close(quantizer.scale.flatten(), scale, rtol=0, atol=0)
    torch.testing.assert_close(quantizer.zero.flatten(), zero, rtol=0, atol=0)


# ``+/-2**127`` is exactly representable in BF16 but the affine span
# ``2**128`` is not representable in the FP32 work dtype, so ``xmax - xmin``
# overflows before the division even though the fitted step is finite. The
# shared ``_affine_scale`` helper fits it by dividing each endpoint first.
_BF16_EXTREME = 2.0**127
_BF16_EXTREME_STEP = torch.tensor(2.0**128 / 15, dtype=torch.float64).float()


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("mse", [False, True])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable",
    )),
])
def test_qqq_find_params_keeps_finite_affine_scale_at_bf16_extreme_endpoints(
    sym, mse, device
):
    group = 128
    weights = torch.zeros(2, group, dtype=torch.bfloat16, device=device)
    weights[0, 0] = _BF16_EXTREME
    weights[0, 2] = -_BF16_EXTREME
    control = torch.linspace(-1.0, 1.0, group, device=device).to(torch.bfloat16)
    weights[1] = control

    quantizer = Quantizer()
    quantizer.configure(bits=4, perchannel=True, sym=sym, groupsize=group, mse=mse)
    quantizer.find_params(weights, weight=True)

    # The overflow lane must take the representable step, not ``inf``.
    assert torch.isfinite(quantizer.scale).all()
    assert quantizer.scale.dtype == torch.float32
    torch.testing.assert_close(
        quantizer.scale[0, 0].cpu(), _BF16_EXTREME_STEP, rtol=0, atol=0
    )

    # The ordinary group keeps the bitwise-identical pre-fix fit.
    control_f32 = control.float()
    control_step = (control_f32.max() - control_f32.min()) / quantizer.maxq
    assert quantizer.scale[1, 0] == control_step

    # A finite scale must produce a finite reconstruction of the endpoints.
    dequant = quantizer.quantize(weights)
    assert torch.isfinite(dequant).all()
    assert dequant[0, 0] > 0
    assert dequant[0, 2] < 0


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize(
    "strategy",
    [FallbackStrategy.MIDPOINT, FallbackStrategy.MEAN, FallbackStrategy.MEDIAN],
)
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable",
    )),
])
def test_qqq_fallback_affine_grid_keeps_finite_scale_at_bf16_extreme_endpoints(
    sym, strategy, device
):
    group = 128
    weight = torch.zeros(4, group, dtype=torch.bfloat16, device=device)
    weight[0, 0] = _BF16_EXTREME
    weight[0, 2] = -_BF16_EXTREME
    weight[1] = torch.linspace(-1.0, 1.0, group, device=device).to(torch.bfloat16)
    layer = torch.nn.Linear(group, 4, bias=False, device=device)
    with torch.no_grad():
        layer.weight.copy_(weight)

    qcfg = QQQConfig(
        bits=4,
        group_size=group,
        sym=sym,
        fallback=Fallback(strategy=strategy, threshold=False),
    )
    task = QQQ(layer, qcfg)

    dequant, scale, zero, _g_idx, _duration, _loss, _damp, _extra, _samples = (
        task._fallback_quantize(strategy)
    )

    assert torch.isfinite(scale).all()
    torch.testing.assert_close(scale[0, 0].cpu(), _BF16_EXTREME_STEP, rtol=0, atol=0)
    assert torch.isfinite(zero).all()
    assert torch.isfinite(dequant).all()
    assert dequant[0, 0] > 0
    assert dequant[0, 2] < 0


# A symmetric INT4 group fits ``scale = 2 * amax / 15`` and stores zero point
# eight, so code zero reconstructs ``-16 * amax / 15``. At ``amax = 61440``
# that is ``-65536``: the FP16 replay weight becomes ``-inf`` even though every
# source value is finite. The grid is valid; the selected codes must stay
# inside the destination dtype's finite range (#6000, #6011, #6023).
_FP16_MAX = float(torch.finfo(torch.float16).max)
_FP16_OVERFLOW_AMAX = 61440.0


def _extreme_fp16_layer(device, *, in_features=128, out_features=4):
    layer = nn.Linear(in_features, out_features, bias=False, dtype=torch.float16, device=device)
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[:, 0] = -_FP16_OVERFLOW_AMAX
        layer.weight[:, 1] = _FP16_OVERFLOW_AMAX
        layer.weight[0, 0] = -_FP16_MAX
        layer.weight[0, 1] = _FP16_MAX
    return layer


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable",
    )),
])
@pytest.mark.parametrize("mse", [False, True])
@pytest.mark.parametrize("sym", [True, False])
def test_qqq_find_params_bounds_codes_to_finite_fp16_range(device, mse, sym):
    weights = torch.zeros(2, 128, dtype=torch.float32, device=device)
    weights[0, 0] = -_FP16_OVERFLOW_AMAX
    weights[0, 2] = _FP16_OVERFLOW_AMAX
    weights[1] = torch.linspace(-1.0, 1.0, 128, device=device)

    quantizer = Quantizer()
    quantizer.configure(
        bits=4, perchannel=True, sym=sym, groupsize=128, mse=mse
    )
    quantizer.find_params(weights, weight=True, value_limit=_FP16_MAX)

    assert quantizer.code_lo is not None
    assert quantizer.code_hi is not None
    assert (quantizer.code_lo >= 0).all()
    assert (quantizer.code_hi <= 15).all()

    dequant = quantizer.quantize(weights)
    assert torch.isfinite(dequant.to(torch.float16)).all()
    assert dequant.min() >= -_FP16_MAX

    if not mse:
        # 61440 / 8192 = 7.5 rounds ties-to-even to code zero, whose
        # reconstruction is -65536. The finite grid excludes it.
        assert quantizer.code_lo.flatten()[0] == 1
        assert quantizer.code_hi.flatten()[0] == 15


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable",
    )),
])
@pytest.mark.parametrize(
    "strategy",
    [
        FallbackStrategy.MIDPOINT,
        FallbackStrategy.MEAN,
        FallbackStrategy.MEDIAN,
        FallbackStrategy.STDCLIP,
        FallbackStrategy.RTN,
    ],
    ids=lambda strategy: strategy.value,
)
@pytest.mark.parametrize("sym", [True, False])
def test_qqq_fallback_bounds_codes_to_finite_fp16_range(device, strategy, sym):
    layer = _extreme_fp16_layer(device)
    qcfg = QQQConfig(
        bits=4,
        group_size=128,
        sym=sym,
        fallback=Fallback(strategy=strategy, threshold=False),
        offload_to_disk=False,
    )
    task = QQQ(layer, qcfg)
    task.quantizer.configure(4, perchannel=True, sym=sym, groupsize=128)

    dequant, scale, zero, *_ = task._fallback_quantize(strategy)

    assert torch.isfinite(dequant).all()
    assert torch.isfinite(scale).all()
    assert torch.isfinite(zero).all()
    assert dequant.min() >= -_FP16_MAX
    assert dequant.max() <= _FP16_MAX


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable",
    )),
])
def test_qqq_fallback_smooth_paths_bound_codes_to_finite_fp16_range(device):
    """Smoothing restores a per-row factor after the fit; the bound follows it."""

    for smooth, strategy in (
        (SmoothMSE(steps=8), FallbackStrategy.RTN),
        (SmoothRowCol(axis="row"), FallbackStrategy.MIDPOINT),
    ):
        layer = _extreme_fp16_layer(device)
        qcfg = QQQConfig(
            bits=4,
            group_size=128,
            sym=True,
            fallback=Fallback(strategy=strategy, threshold=False, smooth=smooth),
            offload_to_disk=False,
        )
        task = QQQ(layer, qcfg)
        task.quantizer.configure(4, perchannel=True, sym=True, groupsize=128)

        dequant, scale, zero, *_ = task._fallback_quantize(strategy)

        assert torch.isfinite(dequant).all(), type(smooth).__name__
        assert torch.isfinite(scale).all(), type(smooth).__name__
        assert torch.isfinite(zero).all(), type(smooth).__name__


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable",
    )),
])
@pytest.mark.parametrize("static_groups", [False, True])
@pytest.mark.parametrize("mse", [False, True])
def test_qqq_main_path_narrows_finite_endpoint_and_packs_checkpoint(
    device, static_groups, mse
):
    """The full QQQ fit, narrowing, and packer round-trip stay finite."""

    torch.manual_seed(6000)
    layer = nn.Linear(256, 64, bias=False, dtype=torch.float16, device=device)
    with torch.no_grad():
        layer.weight.normal_(0, 1)
        layer.weight[0, 0] = -_FP16_OVERFLOW_AMAX
        layer.weight[0, 2] = _FP16_OVERFLOW_AMAX
        layer.weight[1, 0] = -_FP16_MAX
        layer.weight[1, 1] = _FP16_MAX

    qcfg = QQQConfig(
        bits=4,
        group_size=128,
        sym=True,
        static_groups=static_groups,
        desc_act=False,
        offload_to_disk=False,
    )
    task = QQQ(layer, qcfg)
    task.quantizer.configure(
        4, perchannel=True, sym=True, groupsize=128, mse=mse
    )
    task.add_batch(torch.eye(256, dtype=torch.float16, device=device).unsqueeze(0), None)

    (
        weight,
        scales,
        _zeros,
        _g_idx,
        _duration,
        _loss,
        _damp,
        scale_extra,
        _samples,
    ) = task.quantize(blocksize=128)

    assert weight.dtype == torch.float16
    assert torch.isfinite(weight).all()
    assert torch.isfinite(scales).all()
    assert torch.isfinite(scale_extra).all()

    # Pack the narrowed weight exactly as the looper does before saving.
    pack_layer = nn.Linear(256, 64, bias=False, dtype=torch.float16)
    with torch.no_grad():
        pack_layer.weight.copy_(weight.cpu())
    module = QQQTorchLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=256,
        out_features=64,
        bias=False,
    )
    module.pack(pack_layer, scales.cpu(), s_extra=scale_extra.cpu())

    assert torch.isfinite(module.s_group).all()
    assert torch.isfinite(module.s_channel).all()
    # A non-finite channel scale divides the group scales to zero and the
    # forward then computes ``0 * inf`` (#6076).
    assert (module.s_group != 0).all()

    # The packer must recover exactly the codes the fitted grid selected.
    codes = module._unpack_weight_codes().to(torch.int32)
    raw_scale = scales.cpu().t().to(torch.float32)
    group_index = torch.arange(256) // 128
    expected = (
        torch.round(weight.cpu().t().float() / raw_scale[group_index].float()) + 8
    ).clamp(0, 15).to(torch.int32)
    assert torch.equal(codes, expected)

    # The consumer end of the chain: an input whose dense reference is finite
    # must not produce NaN from ``0 * inf`` in the packed forward.
    module = module.to(device)
    probe = torch.nn.functional.one_hot(torch.tensor([0, 1]), 256).to(
        dtype=torch.float16, device=device
    )
    output = module(probe)
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert not torch.isnan(output).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("magnitude", [_FP16_OVERFLOW_AMAX, _FP16_MAX])
def test_qqq_fused_block_update_uses_the_same_finite_codes_as_eager(
    monkeypatch, magnitude
):
    """The fused kernel must select the same codes the eager loop selects."""

    from gptqmodel.quantization import gptq_cuda
    from gptqmodel.utils.kernel import KernelConfig, configure_kernel, reset_kernel_config

    if not gptq_cuda.block_update_available():
        pytest.skip(gptq_cuda._EXTENSION.last_error_message())

    device = torch.device("cuda:0")

    def run(fused):
        configure_kernel(KernelConfig(gptq_fused_block_update=fused))
        launches = []
        original = gptq_cuda.gptq_block_update

        def counted(*args, **kwargs):
            launches.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(gptq_cuda, "gptq_block_update", counted)

        torch.manual_seed(6076)
        layer = nn.Linear(256, 64, bias=False, dtype=torch.float16, device=device)
        with torch.no_grad():
            layer.weight.normal_(0, 1)
            layer.weight[0, 0] = -magnitude
            layer.weight[0, 2] = magnitude
            layer.weight[1, 0] = -_FP16_MAX
            layer.weight[1, 1] = _FP16_MAX
        qcfg = QQQConfig(
            bits=4,
            group_size=128,
            sym=True,
            desc_act=False,
            offload_to_disk=False,
        )
        task = QQQ(layer, qcfg)
        task.quantizer.configure(4, perchannel=True, sym=True, groupsize=128)
        task.add_batch(torch.eye(256, dtype=torch.float16, device=device).unsqueeze(0), None)
        weight, scales, zeros, *_ = task.quantize(blocksize=128)
        torch.cuda.synchronize(device)
        return weight.cpu(), scales.cpu(), zeros.cpu(), len(launches)

    try:
        eager = run(False)
        fused = run(True)
    finally:
        reset_kernel_config()

    # The trigger must actually run: two 128-column blocks.
    assert eager[3] == 0
    assert fused[3] == 2
    assert torch.isfinite(fused[0]).all()
    for eager_value, fused_value in zip(eager[:3], fused[:3]):
        assert torch.equal(eager_value, fused_value)
