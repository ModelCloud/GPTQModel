# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""VPTQ's "deterministic" segment sum must not rely on a float cumsum (#4224).

``torch.cumsum`` over a floating-point CUDA/HIP tensor is nondeterministic, so
the previous fixed-order prefix-sum reduction still produced a different
codebook on every run once the bucket count grew.  The reduction now sums each
sorted bucket with ``torch.segment_reduce`` and must stay bit-identical across
repeats on the device.
"""

import pytest
import torch

from gptqmodel.quantization import vptq as vptq_module


def test_vptq_deterministic_segment_sum_handles_empty_index():
    result = vptq_module._deterministic_segment_sum(
        4, torch.empty(0, dtype=torch.long), torch.empty(0, 3)
    )

    assert torch.equal(result, torch.zeros(4, 3))


def test_vptq_deterministic_segment_sum_is_repeatable_on_cpu():
    generator = torch.Generator().manual_seed(0)
    index = torch.randint(0, 257, (4096,), generator=generator)
    source = torch.randn(4096, 8, generator=generator)
    weights = torch.rand(4096, generator=generator)

    for value in (source, weights):
        oracle = torch.zeros(
            257, *value.shape[1:], dtype=torch.float64
        ).index_add_(0, index, value.double())
        first = vptq_module._deterministic_segment_sum(257, index, value)
        for _ in range(5):
            assert torch.equal(
                first, vptq_module._deterministic_segment_sum(257, index, value)
            )
        relative = (first.double() - oracle).norm() / oracle.norm()
        assert relative.item() < 1e-6


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA/HIP device"
)
def test_vptq_deterministic_segment_sum_is_bitwise_repeatable_on_gpu():
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(0)
    num_buckets = 65536
    num_rows = 1 << 21
    index = torch.randint(
        0, num_buckets, (num_rows,), device=device, generator=generator
    )
    source = torch.rand(num_rows, device=device, generator=generator) * 100

    outputs = [
        vptq_module._deterministic_segment_sum(num_buckets, index, source)
        for _ in range(20)
    ]

    differing = sum(not torch.equal(output, outputs[0]) for output in outputs[1:])
    assert differing == 0, f"{differing}/{len(outputs) - 1} GPU repeats differed"

    oracle = torch.zeros(
        num_buckets, dtype=torch.float64, device=device
    ).index_add_(0, index, source.double())
    relative = (outputs[0].double() - oracle).norm() / oracle.norm()
    assert relative.item() < 1e-6
