# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.quantization.qvq import (
    _canonical_qvq_codebook,
    _canonical_qvq_v2b2_pair_stacks,
    yaqa_inner,
)
from gptqmodel.quantization.qvq_codecs import PGC16_CODEBOOK_VERSION
from gptqmodel.quantization.qvq_yaqa import capture_yaqa_sketch_b

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")


class _TinySketchLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(self, hidden):
        return self.proj(hidden)


class _TinySketchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinySketchLayer()])
        self.head = nn.Linear(8, 16, bias=False)

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        hidden = torch.nn.functional.one_hot(input_ids.remainder(8), 8).float()
        hidden = self.model.layers[0](hidden)
        return SimpleNamespace(logits=self.head(hidden))


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


@pytest.mark.parametrize("width", (1, 2, 17, 512))
def test_yaqa_mlx_symmetric_gram_pack_round_trip_is_bit_exact(width):
    from gptqmodel.utils.qvq_mlx import (
        qvq_mlx_pack_symmetric_gram_from_torch_mps,
        qvq_mlx_unpack_symmetric_gram_to_torch_cpu,
    )

    generator = torch.Generator().manual_seed(20260819 + width)
    source = torch.randn((width, width), generator=generator)
    symmetric = (source + source.T).to("mps").contiguous()
    packed = qvq_mlx_pack_symmetric_gram_from_torch_mps(symmetric)
    restored = qvq_mlx_unpack_symmetric_gram_to_torch_cpu(packed.to("cpu"), width=width)

    assert packed.numel() == width * (width + 1) // 2
    assert torch.equal(restored, symmetric.cpu())


def test_yaqa_mps_packed_symmetric_accumulation_matches_full_factors_exactly():
    baseline = _TinySketchModel().eval().to("mps")
    packed_model = copy.deepcopy(baseline)
    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        },
        {
            "input_ids": torch.tensor([[6, 7]]),
            "attention_mask": torch.tensor([[1, 1]]),
        },
    ]

    baseline_input, baseline_output, baseline_stats = capture_yaqa_sketch_b(
        baseline,
        batches,
        {"proj": baseline.model.layers[0].proj},
        device=torch.device("mps"),
        seed=787,
        mps_pack_symmetric_grams=False,
    )
    packed_input, packed_output, packed_stats = capture_yaqa_sketch_b(
        packed_model,
        batches,
        {"proj": packed_model.model.layers[0].proj},
        device=torch.device("mps"),
        seed=787,
    )

    assert torch.equal(packed_input["proj"], baseline_input["proj"])
    assert torch.equal(packed_output["proj"], baseline_output["proj"])
    assert baseline_stats["packed_symmetric_accumulators"] is False
    assert packed_stats["packed_symmetric_accumulators"] is True
    assert packed_stats["accumulator_bytes"] == 8 * 9 * 4
    assert packed_stats["accumulator_bytes"] < baseline_stats["accumulator_bytes"]
