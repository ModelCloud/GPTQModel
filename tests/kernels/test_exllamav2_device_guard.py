# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""ExLlamaV2 Q-matrix construction must restore the caller's CUDA device.

A `device_map` shards a GPTQ/AWQ checkpoint over several GPUs. `make_q_matrix`
constructs a `QMatrix`, whose constructor calls `cudaSetDevice(device)` and
never restores the previous device. Loading a layer that lives on `cuda:1`
therefore left `torch.cuda.current_device()` at 1, and later unqualified
`torch.zeros(1, device="cuda")` allocations landed on the wrong GPU.

This drives the native op with operands on `cuda:1` while `cuda:0` is current,
which is the exact state a multi-GPU load creates.
"""

import pytest
import torch

from gptqmodel.utils.exllamav2 import (
    exllamav2_awq_make_q_matrix,
    exllamav2_free_q_matrix,
    exllamav2_make_q_matrix,
)
from gptqmodel.utils.rocm import IS_ROCM


pytestmark = [
    pytest.mark.skipif(
        IS_ROCM or not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="two CUDA GPUs required",
    ),
    pytest.mark.cuda,
]


def _q_matrix_tensors(device):
    # 4-bit GPTQ/AWQ layout: qweight is [in_features / 8, out_features],
    # qzeros is [groups, out_features / 8] and scales is [groups, out_features].
    in_features, out_features, groups = 128, 64, 4
    qweight = torch.zeros(
        (in_features // 8, out_features), dtype=torch.int32, device=device
    )
    qzeros = torch.zeros(
        (groups, out_features // 8), dtype=torch.int32, device=device
    )
    scales = torch.ones(
        (groups, out_features), dtype=torch.float16, device=device
    )
    return qweight, qzeros, scales


@pytest.mark.parametrize("variant", ["gptq", "awq"])
def test_make_q_matrix_restores_current_device(variant):
    torch.cuda.set_device(0)
    qweight, qzeros, scales = _q_matrix_tensors(torch.device("cuda", 1))

    before = torch.cuda.current_device()
    if variant == "gptq":
        handle = exllamav2_make_q_matrix(
            qweight, None, None, None, None, None, qzeros, scales, None
        )
    else:
        handle = exllamav2_awq_make_q_matrix(
            qweight, None, None, None, None, None, qzeros, scales, None
        )

    try:
        assert torch.cuda.current_device() == before
    finally:
        exllamav2_free_q_matrix(handle, awq=variant == "awq")
