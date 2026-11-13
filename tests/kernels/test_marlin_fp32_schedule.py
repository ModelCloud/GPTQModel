# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent arithmetic checks for the narrow-projection FP32 schedule."""

import pytest
import torch
from test_torch_weight_storage import state_tensors

from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or bool(torch.version.hip) or torch.cuda.get_device_capability()[0] < 8,
    reason="AWQ Marlin requires NVIDIA Ampere or newer",
)

SHAPES = [
    (m, 2048, 256, 128, 4) for m in (1, 8, 16, 17, 32, 33, 48, 63, 64, 65, 80, 96, 112, 127, 128, 129)
] + [
    (64, 8192, n, 128, 4) for n in (128, 512, 1024, 2048)
] + [
    (16, 8192, 256, 128, 4), (128, 8192, 256, 128, 4),
    (64, 4096, 256, 128, 4), (128, 4096, 512, 128, 4),
    (64, 1920, 256, 128, 4), (64, 8320, 256, 128, 4),
    (64, 2048, 2112, 128, 4),
    (64, 2048, 256, 32, 4), (64, 2048, 256, 64, 4),
    (64, 2048, 256, -1, 4), (64, 2048, 256, 128, 8),
]


def _pack_columns(values, bits):
    order = (0, 2, 4, 6, 1, 3, 5, 7) if bits == 4 else (0, 2, 1, 3)
    packed = torch.zeros((values.shape[0], values.shape[1] // len(order)), dtype=torch.int32)
    for lane, source in enumerate(order):
        packed |= values[:, source::len(order)] << (lane * bits)
    return packed


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("cancellation", [False, True])
def test_marlin_narrow_fp32_oracle(dtype, shape, cancellation):
    m, k, n, group, bits = shape
    groups = k // (k if group == -1 else group)
    generator = torch.Generator().manual_seed(3520)
    if cancellation:
        codes = torch.ones((k, n), dtype=torch.int32)
        zeros = torch.zeros((groups, n), dtype=torch.int32)
        scales = torch.ones((groups, n), dtype=dtype)
        source = torch.full((m, k), 8192., dtype=dtype)
        source[:, k // 2:] *= -1
        source[:, -1] = torch.nextafter(source[:, -1], torch.zeros(m, dtype=dtype))
    else:
        codes = torch.randint(0, 1 << bits, (k, n), generator=generator, dtype=torch.int32)
        zeros = torch.full((groups, n), 1 << (bits - 1), dtype=torch.int32)
        scales = torch.full((groups, n), 1 / 128, dtype=dtype)
        source = (torch.randint(-8, 9, (m, k), generator=generator).double() / 8).to(dtype)

    dense = (codes.double() - zeros.repeat_interleave(k // groups, 0).double())
    dense *= scales.repeat_interleave(k // groups, 0).double()
    raw = source.double() @ dense
    # Marlin's existing bias epilogue rounds the product before adding bias.
    expected = (raw.to(dtype).double() + .25).to(dtype)
    layer = AwqMarlinLinear(bits=bits, group_size=group, sym=False, desc_act=False,
                           in_features=k, out_features=n, bias=True, dtype=dtype,
                           register_buffers=True).cuda().eval()
    layer.qweight.copy_(_pack_columns(codes, bits).cuda())
    layer.qzeros.copy_(_pack_columns(zeros, bits).cuda())
    layer.scales.copy_(scales.cuda())
    layer.bias.fill_(.25)
    layer.post_init()
    x = source.cuda()
    before = {t.data_ptr() for t in state_tensors(layer)}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        for _ in range(3):
            actual = layer(x)
            assert actual.dtype == dtype and actual.isfinite().all()
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    stream.synchronize()
    assert not any(t.data_ptr() not in before and t.is_floating_point() and t.numel() >= k * n
                   for t in state_tensors(layer))
