# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.quantization.qqq import Quantizer, quantize


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


def _legacy_qqq_find_params(
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
    tmp = torch.zeros(x.shape[0], device=x.device)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)

    if sym:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        tmp = xmin < 0
        if torch.any(tmp):
            xmin[tmp] = -xmax[tmp]
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1

    if maxq < 0:
        scale = xmax
        zero = xmin
    elif groupsize != -1 or not sym:
        scale = (xmax - xmin) / maxq
        zero = torch.full_like(scale, (maxq + 1) / 2) if sym else torch.round(-xmin / scale)
    else:
        scale = xmax / maxq
        zero = torch.zeros_like(scale)

    if mse:
        best = torch.full([x.shape[0]], float("inf"), device=x.device)
        for index in range(int(maxshrink * grid)):
            p = 1 - index / grid
            xmin1 = p * xmin
            xmax1 = p * xmax
            scale1 = (xmax1 - xmin1) / maxq if groupsize != -1 or not sym else xmax1 / maxq
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

        expected_scale, expected_zero = _legacy_qqq_find_params(
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
