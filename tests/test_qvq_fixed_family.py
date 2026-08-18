import pytest
import torch

from gptqmodel.quantization.qvq import quantize_qvq_linear


def _yaqa_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260818)
    weight = torch.randn((16, 16), generator=generator)
    input_rows = torch.randn((64, 16), generator=generator)
    output_rows = torch.randn((64, 16), generator=generator)
    input_hessian = input_rows.T @ input_rows / input_rows.shape[0] + torch.eye(16) * 0.1
    output_hessian = output_rows.T @ output_rows / output_rows.shape[0] + torch.eye(16) * 0.1
    return weight, input_hessian, output_hessian


def _quantize_fixed_family(family_id: int):
    weight, input_hessian, output_hessian = _yaqa_inputs()
    return quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        output_hessian=output_hessian,
        seed=7,
        trellis_batch_size=1,
        rounding="yaqa",
        bank_count=2,
        v2b2_p32=True,
        yaqa_v2b2_family_mode="fixed_block_ldlq",
        yaqa_v2b2_fixed_family_id=family_id,
    )


def test_qvq_yaqa_fixed_family_zero_is_exact_independent_v2_artifact():
    weight, input_hessian, output_hessian = _yaqa_inputs()
    canonical = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        output_hessian=output_hessian,
        seed=7,
        trellis_batch_size=1,
        rounding="yaqa",
    )
    family_zero = _quantize_fixed_family(0)

    assert torch.equal(family_zero.trellis, canonical.trellis)
    assert torch.equal(family_zero.inner_weight, canonical.inner_weight)
    assert torch.equal(family_zero.weight, canonical.weight)
    assert torch.count_nonzero(family_zero.bank_ids) == 0
    assert family_zero.bank_alt_id.tolist() == [1]
    assert family_zero.yaqa_bank_fallback_to_v2 is True
    assert family_zero.yaqa_block_family_id == 0


@pytest.mark.parametrize("family_id", (1, 2, 3))
def test_qvq_yaqa_fixed_family_preserves_requested_alternative_metadata(family_id):
    result = _quantize_fixed_family(family_id)

    assert result.bank_alt_id.tolist() == [family_id]
    assert result.bank_ids is not None
    assert result.bank_ids.dtype == torch.uint8
    assert result.yaqa_block_family_id == family_id
    assert set(result.serialized_tensors()) == {"SU", "SV", "trellis", "bank_ids", "bank_alt_id"}


@pytest.mark.parametrize("family_id", (-1, 4, True, 1.5))
def test_qvq_yaqa_fixed_family_rejects_invalid_ids(family_id):
    weight, input_hessian, output_hessian = _yaqa_inputs()
    with pytest.raises(ValueError, match="fixed family ID must be 0, 1, 2, or 3"):
        quantize_qvq_linear(
            weight,
            input_hessian,
            bits=2,
            output_hessian=output_hessian,
            rounding="yaqa",
            bank_count=2,
            v2b2_p32=True,
            yaqa_v2b2_family_mode="fixed_block_ldlq",
            yaqa_v2b2_fixed_family_id=family_id,
        )


def test_qvq_yaqa_fixed_family_rejects_non_fixed_mode():
    weight, input_hessian, output_hessian = _yaqa_inputs()
    with pytest.raises(ValueError, match="requires V2B2-P32 YAQA"):
        quantize_qvq_linear(
            weight,
            input_hessian,
            bits=2,
            output_hessian=output_hessian,
            rounding="yaqa",
            bank_count=2,
            v2b2_p32=True,
            yaqa_v2b2_fixed_family_id=1,
        )


def test_qvq_yaqa_fixed_family_rejects_subsequent_spectral_refinement():
    weight, input_hessian, output_hessian = _yaqa_inputs()
    with pytest.raises(ValueError, match="no subsequent candidate refinement"):
        quantize_qvq_linear(
            weight,
            input_hessian,
            bits=2,
            output_hessian=output_hessian,
            rounding="yaqa",
            bank_count=2,
            v2b2_p32=True,
            yaqa_v2b2_family_mode="fixed_block_ldlq",
            yaqa_v2b2_fixed_family_id=1,
            yaqa_spectral_refinement=True,
        )
