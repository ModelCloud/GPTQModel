# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Native CPU factored-YAQA feedback kernels versus the dense FP32 reference."""

import pytest
import torch

from gptqmodel.utils.qvq_cpu import qvq_cpu_supported, qvq_cpu_yaqa_feedback_update


pytestmark = pytest.mark.skipif(not qvq_cpu_supported(), reason="QVQ CPU kernels unavailable")

TILE = 16


def _reference_update(left, right, input_feedback, output_feedback, reconstructed, blocks):
    expected_left = left.clone()
    expected_right = right.clone()
    for tile, (input_block, output_block) in enumerate(blocks):
        input_start = input_block * TILE
        output_start = output_block * TILE
        expected_left[:, output_start : output_start + TILE] -= (
            input_feedback[input_start : input_start + TILE].T @ reconstructed[tile]
        )
        expected_right[input_start : input_start + TILE] -= (
            reconstructed[tile] @ output_feedback[output_start : output_start + TILE]
        )
    return expected_left, expected_right


@pytest.mark.parametrize("shape", ((32, 64), (64, 32), (48, 48)))
def test_qvq_cpu_factored_yaqa_cache_update_matches_fp32_reference(shape):
    in_features, out_features = shape
    generator = torch.Generator().manual_seed(20260826 + in_features)
    left = torch.randn((in_features, out_features), generator=generator) * 0.01
    right = torch.randn((in_features, out_features), generator=generator) * 0.01
    input_feedback = torch.randn((in_features, in_features), generator=generator) * 0.01
    output_feedback = torch.randn((out_features, out_features), generator=generator) * 0.01

    first_input = 0
    first_output = min(out_features // TILE, in_features // TILE) - 1
    count = first_output + 1
    blocks = [(first_input + tile, first_output - tile) for tile in range(count)]
    reconstructed = (torch.randn((count, TILE, TILE), generator=generator) * 0.05).contiguous()

    expected_left, expected_right = _reference_update(
        left, right, input_feedback, output_feedback, reconstructed, blocks
    )

    actual_left = left.clone()
    actual_right = right.clone()
    qvq_cpu_yaqa_feedback_update(
        actual_left,
        actual_right,
        input_feedback,
        output_feedback,
        reconstructed,
        first_input,
        first_output,
        count,
    )

    torch.testing.assert_close(actual_left, expected_left, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(actual_right, expected_right, rtol=0.0, atol=1e-6)


def test_qvq_cpu_factored_yaqa_cache_update_rejects_invalid_geometry():
    left = torch.zeros((32, 32), dtype=torch.float32)
    right = torch.zeros((32, 32), dtype=torch.float32)
    feedback = torch.zeros((32, 32), dtype=torch.float32)
    reconstructed = torch.zeros((2, TILE, TILE), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="anti-diagonal geometry"):
        qvq_cpu_yaqa_feedback_update(left, right, feedback, feedback, reconstructed, 0, 0, 2)
