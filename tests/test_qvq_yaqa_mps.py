# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.quantization.qvq import (
    _canonical_qvq_codebook,
    _canonical_qvq_v2b2_pair_stacks,
    yaqa_inner,
)
from gptqmodel.quantization.qvq_codecs import PGC16_CODEBOOK_VERSION


pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")


def _problem(width: int = 32):
    generator = torch.Generator().manual_seed(20260817)
    weight = torch.randn((width, width), generator=generator)
    input_samples = torch.randn((64, width), generator=generator)
    output_samples = torch.randn((64, width), generator=generator)
    input_hessian = input_samples.T @ input_samples / input_samples.shape[0]
    output_hessian = output_samples.T @ output_samples / output_samples.shape[0]
    input_hessian.diagonal().add_(0.5)
    output_hessian.diagonal().add_(0.5)
    return weight, input_hessian, output_hessian


def test_yaqa_mps_hybrid_matches_cpu_artifact_exactly():
    weight, input_hessian, output_hessian = _problem()
    cpu_codebook = _canonical_qvq_codebook(
        device=torch.device("cpu"),
        vector_size=2,
        bits=2,
        codebook_version=PGC16_CODEBOOK_VERSION,
        dtype=torch.float32,
    )
    expected = yaqa_inner(weight, input_hessian, output_hessian, cpu_codebook, bits=2, trellis_batch_size=8)
    actual = yaqa_inner(
        weight.to("mps"),
        input_hessian.to("mps"),
        output_hessian.to("mps"),
        cpu_codebook.to("mps"),
        bits=2,
        trellis_batch_size=8,
    )

    assert all(torch.equal(cpu, mps.cpu()) for cpu, mps in zip(expected, actual, strict=True))
    assert all(tensor.device.type == "mps" for tensor in actual)


def test_yaqa_v2b2_p32_mlx_hybrid_matches_cpu_artifact_exactly():
    weight, input_hessian, output_hessian = _problem()

    def run(device: torch.device):
        pair_stack = _canonical_qvq_v2b2_pair_stacks(
            device=device,
            bits=2,
            codebook_version=PGC16_CODEBOOK_VERSION,
            dtype=torch.float32,
        )[0]
        codebooks = tuple(pair_stack[index] for index in range(2))
        return yaqa_inner(
            weight.to(device),
            input_hessian.to(device),
            output_hessian.to(device),
            codebooks[0],
            bits=2,
            trellis_batch_size=8,
            bank_codebooks=codebooks,
            segmented_bank_stack=pair_stack,
            v2b2_p32=True,
        )

    expected = run(torch.device("cpu"))
    actual = run(torch.device("mps"))

    assert all(torch.equal(cpu, mps.cpu()) for cpu, mps in zip(expected, actual, strict=True))
    assert all(tensor.device.type == "mps" for tensor in actual)
