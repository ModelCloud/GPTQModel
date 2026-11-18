# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""AWQ GEMV Triton must launch rows on grid axis 0 and address them in 64 bits.

``awq_gemv_triton`` put the row index on grid axis 1, whose block count is
capped at 65535: on ROCm every row past 65535 was silently left uninitialized,
and on CUDA the launch failed with ``invalid argument`` (#5192). The row
offsets were also ``pid_m * K`` / ``pid_m * N`` in 32 bits, so a prefill whose
``rows * in_features`` crossed 2^31 wrapped and faulted.
"""

import pytest
import torch


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)

GROUP_SIZE = 128
K = 128
N = 128
# Rows per chunk for the reference: comfortably below the old grid-axis cap.
CHUNK = 8192


def _kernel_fixture(rows, seed=5192):
    """A small deterministic GEMV layout: K == N == group_size == 128."""

    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, K, device="cuda", dtype=torch.float16, generator=gen)
    qweight = torch.randint(
        -(2**31),
        2**31 - 1,
        (N, K // 8),
        device="cuda",
        dtype=torch.int32,
        generator=gen,
    )
    scales = (
        torch.rand(N, 1, device="cuda", dtype=torch.float16, generator=gen)
        .abs()
        .mul(0.01)
    )
    qzeros = torch.zeros(N, 1, device="cuda", dtype=torch.int32)
    return x, qweight, scales, qzeros


def _kernel_reference(x, qweight, scales, qzeros):
    from gptqmodel.quantization.awq.modules.triton.gemv import awq_gemv_triton

    return torch.cat(
        [
            awq_gemv_triton(
                x[i : i + CHUNK].contiguous(), qweight, scales, qzeros, GROUP_SIZE
            )
            for i in range(0, x.shape[0], CHUNK)
        ]
    )


@requires_cuda
@pytest.mark.cuda
def test_awq_gemv_triton_rows_above_the_grid_axis_limit():
    """65537 rows used to exceed gridDim.y and drop (ROCm) or fail (CUDA)."""

    from gptqmodel.quantization.awq.modules.triton.gemv import awq_gemv_triton

    rows = 65537
    x, qweight, scales, qzeros = _kernel_fixture(rows)

    out = awq_gemv_triton(x, qweight, scales, qzeros, GROUP_SIZE)
    torch.cuda.synchronize()

    assert out.shape == (rows, N)
    assert torch.isfinite(out).all()
    # Every row is independent, so the chunked call must be bit-identical.
    assert torch.equal(out, _kernel_reference(x, qweight, scales, qzeros))


@requires_cuda
@pytest.mark.cuda
def test_awq_gemv_triton_addresses_rows_past_two_to_the_31_elements():
    """The last row sits past the 32-bit element boundary of ``rows * K``."""

    from gptqmodel.quantization.awq.modules.triton.gemv import awq_gemv_triton

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if properties.total_memory < 12 * 1024**3:
        pytest.skip("the overflow regression needs at least 12 GiB of device memory")

    rows = (2**31 // K) + 1
    x = torch.zeros(rows, K, device="cuda", dtype=torch.float16)
    x[0] = torch.randn(K, device="cuda", dtype=torch.float16)
    x[-8:] = torch.randn(8, K, device="cuda", dtype=torch.float16)
    qweight = torch.randint(
        -(2**31), 2**31 - 1, (N, K // 8), device="cuda", dtype=torch.int32
    )
    scales = (
        torch.rand(N, 1, device="cuda", dtype=torch.float16).abs().mul(0.01)
    )
    qzeros = torch.zeros(N, 1, device="cuda", dtype=torch.int32)

    out = awq_gemv_triton(x, qweight, scales, qzeros, GROUP_SIZE)
    torch.cuda.synchronize()

    # Only the boundary rows carry data; the kernel is row-independent.
    head = awq_gemv_triton(x[:1].contiguous(), qweight, scales, qzeros, GROUP_SIZE)
    tail = awq_gemv_triton(x[-8:].contiguous(), qweight, scales, qzeros, GROUP_SIZE)
    assert torch.equal(out[:1], head)
    assert torch.equal(out[-8:], tail)
    assert torch.isfinite(out[-8:]).all()


@requires_cuda
@pytest.mark.cuda
def test_awq_gemv_triton_module_forward_above_the_grid_axis_limit():
    """The module forwards every row, not just the first 65535."""

    from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVTritonLinear

    torch.manual_seed(5192)
    device = torch.device("cuda:0")
    in_features = out_features = 512
    groups = in_features // GROUP_SIZE
    scales = (
        torch.rand(out_features, groups, device=device, dtype=torch.float16) * 0.1
        + 0.01
    ).contiguous()
    zeros = torch.randint(0, 15, (out_features, groups), device=device, dtype=torch.int32)
    dense = torch.nn.Linear(in_features, out_features, bias=False).to(device=device)

    layer = (
        AwqGEMVTritonLinear(
            bits=4,
            group_size=GROUP_SIZE,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            register_buffers=True,
        )
        .to(device)
        .eval()
    )
    layer.pack(dense, scales, zeros)
    layer.post_init()
    del dense

    rows = 65537
    inputs = torch.randn(1, rows, in_features, device=device, dtype=torch.float16)
    with torch.inference_mode():
        out = layer(inputs)
        ref = torch.cat(
            [layer(inputs[:, i : i + CHUNK]) for i in range(0, rows, CHUNK)], dim=1
        )
    torch.cuda.synchronize(device)

    assert out.shape == (1, rows, out_features)
    assert torch.equal(out, ref)
