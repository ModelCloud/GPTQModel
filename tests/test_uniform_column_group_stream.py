# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The cached zero group must be ordered on the stream that consumes it."""

from __future__ import annotations

import pytest
import torch

from gptqmodel.quantization.gptq import (
    _uniform_column_group,
    _uniform_column_group_cached,
)


def setup_function():
    _uniform_column_group_cached.cache_clear()


def test_cpu_uniform_column_group_is_a_shared_zero_tensor():
    device = torch.device("cpu")
    first = _uniform_column_group(8, device)
    second = _uniform_column_group(8, device)
    assert first is second
    assert torch.equal(first, torch.zeros(8, dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA streams are unavailable")
def test_cuda_uniform_column_group_is_stream_scoped():
    device = torch.device("cuda:0")
    first_stream = torch.cuda.Stream(device=device)
    second_stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(first_stream):
        first = _uniform_column_group(8, device)
    with torch.cuda.stream(second_stream):
        second = _uniform_column_group(8, device)

    assert first is not second
    with torch.cuda.stream(first_stream):
        assert _uniform_column_group(8, device) is first
    with torch.cuda.stream(second_stream):
        assert _uniform_column_group(8, device) is second

    assert torch.equal(first.cpu(), torch.zeros(8, dtype=torch.int32))
    assert torch.equal(second.cpu(), torch.zeros(8, dtype=torch.int32))
