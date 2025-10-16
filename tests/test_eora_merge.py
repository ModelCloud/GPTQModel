import math

import pytest
import torch

from gptqmodel.eora.eora import eora_process_input, merge_eora_segments


def _segment_reduce(contributions, scales):
    total = None
    scale_product = 1.0
    for contribution, scale in zip(contributions, scales):
        if total is None:
            total = contribution.clone()
            scale_product = float(scale)
        else:
            total.mul_(float(scale))
            total.add_(contribution)
            scale_product *= float(scale)
    return total, scale_product


@pytest.mark.parametrize("segments", [1, 2, 4])
def test_eora_merge_matches_sequential(segments):
    torch.manual_seed(0)

    sample_size = 96
    cols = 8
    base_contributions = [
        torch.randn(cols, cols, dtype=torch.float32)
        for _ in range(sample_size)
    ]
    def scale_fn(batch):
        return sample_size / (sample_size + batch)
    scales = [scale_fn(1) for _ in range(sample_size)]

    sequential = torch.zeros((cols, cols), dtype=torch.float32)
    for contribution, scale in zip(base_contributions, scales):
        sequential.mul_(float(scale))
        sequential.add_(contribution)

    segment_length = math.ceil(sample_size / segments)
    segment_pairs = []
    for start in range(0, sample_size, segment_length):
        end = min(start + segment_length, sample_size)
        contributions = base_contributions[start:end]
        seg_scales = scales[start:end]
        segment_pairs.append(_segment_reduce(contributions, seg_scales))

    merged = merge_eora_segments(segment_pairs)
    torch.testing.assert_close(merged, sequential, atol=5e-6, rtol=5e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_eora_process_input_preserves_device():
    device = torch.device("cuda", 0)
    sample = torch.randn(1, 4, 6, device=device, dtype=torch.float16)

    batch, contribution, scale = eora_process_input(
        input=(sample,),
        name="test",
        sample_size=32,
        device=device,
    )

    assert batch == 1
    assert contribution.device == device
    assert contribution.dtype == torch.float32
    assert isinstance(scale, float)
