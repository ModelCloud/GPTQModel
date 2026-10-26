# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.looper.weight_only_processor import _mean_abs_error


def _oracle(reference: torch.Tensor, dequantized: torch.Tensor, *, transpose: bool) -> float:
    reference = reference.detach().cpu().to(torch.float32)
    dequantized = dequantized.detach().cpu().to(torch.float32)
    if transpose:
        dequantized = dequantized.T
    return (dequantized - reference).abs().mean().item()


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize(
    ("reference_device", "dequantized_device"),
    [
        ("cpu", "cpu"),
        ("cpu", "cuda"),
        ("cuda", "cpu"),
        ("cuda", "cuda"),
    ],
)
def test_mean_abs_error_matches_cpu_oracle(
    reference_device: str,
    dequantized_device: str,
    transpose: bool,
):
    if "cuda" in (reference_device, dequantized_device) and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    generator = torch.Generator().manual_seed(0)
    reference = torch.randn((31, 19), generator=generator, dtype=torch.float16)
    dequantized = torch.randn((19, 31), generator=generator, dtype=torch.float16)
    if not transpose:
        dequantized = dequantized.T
    dequantized = dequantized.contiguous()

    expected = _oracle(reference, dequantized, transpose=transpose)
    reference = reference.to(reference_device)
    dequantized = dequantized.to(dequantized_device)
    dequantized_before = dequantized.clone()

    actual = _mean_abs_error(
        reference,
        dequantized.T if transpose else dequantized,
    )

    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert torch.equal(dequantized, dequantized_before)
