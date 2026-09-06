"""ROCm dispatch regressions for the exact banked YAQA trellis reference."""

import pytest
import torch

from gptqmodel.quantization.qvq import batched_v2b2_p32_viterbi_quantize


@pytest.mark.skipif(torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("bits", [2.0, 2.5, 3.0, 3.5])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_banked_rocm_reference_preserves_zero_ties_without_nvidia_dispatch(monkeypatch, bits, dtype):
    sequences = torch.zeros((1, 128, 2), device="cuda", dtype=torch.float32)
    codebooks = torch.zeros((2, 65536, 2), device="cuda", dtype=dtype)

    def reject_nvidia_probe(*args, **kwargs):
        pytest.fail("ROCm must not use NVIDIA compute capability to choose the native CUDA quantizer")

    monkeypatch.setattr(torch.cuda, "get_device_capability", reject_nvidia_probe)
    result = batched_v2b2_p32_viterbi_quantize(sequences, codebooks, bits=bits)

    assert torch.equal(result.states, torch.zeros((1, 128), device="cuda", dtype=torch.long))
    # P32 switches every 32 scalar weights = 16 V2 steps, hence eight segments.
    assert torch.equal(result.segment_bank_ids, torch.zeros((1, 8), device="cuda", dtype=torch.uint8))
    assert torch.equal(result.values, torch.zeros((1, 128, 2), device="cuda", dtype=dtype))
    assert torch.equal(result.squared_error, torch.zeros(1, device="cuda", dtype=torch.float32))


@pytest.mark.skipif(torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("policy", [{"mode": "required"}, {"mode": "auto", "fallback": "error"}])
def test_banked_rocm_preserves_strict_pruning_rejection(policy):
    sequences = torch.zeros((1, 128, 2), device="cuda", dtype=torch.float32)
    codebooks = torch.zeros((2, 65536, 2), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="ROCm"):
        batched_v2b2_p32_viterbi_quantize(sequences, codebooks, bits=2.0, viterbi_pruning=policy)
