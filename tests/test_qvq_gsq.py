import io

import pytest
import torch

from gptqmodel.quantization.qvq import (
    decode_p32_window_tiles,
    repack_p32_planar_to_window,
    repack_p32_window_to_planar,
)
from gptqmodel.quantization.qvq_gsq import refine_p32_candidates


@pytest.mark.parametrize("bits", [1, 1.5, 2, 2.5, 3, 3.5])
def test_hard_export_and_baseline_guard(bits):
    gen = torch.Generator().manual_seed(12)
    candidates = torch.randint(-(2**31), 2**31 - 1, (3, 1, int(bits * 8)), generator=gen, dtype=torch.int32)
    bank = torch.tensor([0b10101010], dtype=torch.uint8)
    alt = torch.tensor([2], dtype=torch.int32)
    target = decode_p32_window_tiles(candidates[1], bits=bits, bank_ids=bank, bank_alt_id=alt).reshape(16, 16)
    inputs = torch.eye(16)
    original = candidates.clone()
    kwargs = dict(bits=bits, bank_ids=bank, bank_alt_id=alt, target=target, inputs=inputs, steps=40, seed=9)
    result = refine_p32_candidates(candidates, **kwargs)
    repeated = refine_p32_candidates(candidates, **kwargs)
    assert torch.equal(result.window_words, repeated.window_words)
    assert torch.equal(candidates, original)
    assert result.calibration_after <= result.calibration_before
    assert result.calibration_after == 0  # the exact target is a legal candidate
    assert torch.equal(result.window_words, candidates[result.choices, torch.arange(1)])
    assert torch.equal(result.window_words, repack_p32_planar_to_window(
        repack_p32_window_to_planar(result.window_words, bits=bits), bits=bits
    ))
    buffer = io.BytesIO()
    torch.save(result.window_words, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    assert torch.equal(loaded, result.window_words)
    assert loaded.numel() * loaded.element_size() == candidates[0].numel() * 4


def test_reject_nonfinite_and_noop():
    candidates = torch.zeros(2, 1, 16, dtype=torch.int32)
    kwargs = dict(bits=2, bank_ids=torch.zeros(1, dtype=torch.uint8), bank_alt_id=torch.tensor([1]),
                  target=torch.ones(16, 16), inputs=torch.eye(16), steps=0)
    result = refine_p32_candidates(candidates, **kwargs)
    assert result.calibration_before == result.calibration_after
    kwargs["target"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        refine_p32_candidates(candidates, **kwargs)
