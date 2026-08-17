import pytest
import torch

from gptqmodel.quantization.qvq_codecs.pgc16 import pgc16_codebook_v2_bank
from gptqmodel.utils.qvq_mlx import (
    _qvq_mlx_v2_banked_tail_implicit_launch,
    qvq_mlx_prepare_v2_banked_codebooks_from_torch,
)


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 3, 3.5))
def test_qvq_mlx_512_thread_tail_is_bit_exact_with_256_threads(bits):
    mx = pytest.importorskip("mlx.core")
    generator = torch.Generator().manual_seed(20260817)
    sequences = mx.array(torch.randn((7, 128, 2), generator=generator).numpy())
    pair = torch.stack(
        (pgc16_codebook_v2_bank(0, bits=bits), pgc16_codebook_v2_bank(1, bits=bits))
    ).contiguous()
    prepared = qvq_mlx_prepare_v2_banked_codebooks_from_torch(pair)
    assert prepared.implicit_levels is not None
    assert prepared.implicit_masks is not None
    kwargs = {
        "transition_bits": int(2 * bits),
        "segment_steps": 16,
        "step_weights": mx.zeros((1,), dtype=mx.float32),
        "weighted": False,
    }

    reference = _qvq_mlx_v2_banked_tail_implicit_launch(
        sequences,
        prepared.implicit_levels,
        prepared.implicit_masks,
        thread_count=256,
        **kwargs,
    )
    actual = _qvq_mlx_v2_banked_tail_implicit_launch(
        sequences,
        prepared.implicit_levels,
        prepared.implicit_masks,
        thread_count=512,
        **kwargs,
    )
    mx.eval(*reference, *actual)
    assert all(mx.array_equal(left, right).item() for left, right in zip(reference, actual, strict=True))
