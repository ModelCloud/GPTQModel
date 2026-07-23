import importlib.util
from pathlib import Path

import pytest
import torch

from gptqmodel.utils.model_dequant import pack_cols


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "gptqmodel-quantization-regressions"
    / "scripts"
    / "analyze_quant_regression.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_quant_regression", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def _pack_rows(codes: torch.Tensor, bits: int) -> torch.Tensor:
    return pack_cols(codes.T.contiguous(), bits, pack_dtype=torch.int32).T.contiguous()


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_count_logical_code_differences_for_even_pack_widths(bits):
    pack_factor = 32 // bits
    reference_codes = (
        torch.arange(pack_factor * 5 * 3, dtype=torch.int32).reshape(pack_factor * 5, 3)
        % (1 << bits)
    )
    candidate_codes = reference_codes.clone()
    candidate_codes[0, 0] = (candidate_codes[0, 0] + 1) % (1 << bits)
    candidate_codes[-1, -1] = (candidate_codes[-1, -1] + 2) % (1 << bits)

    summary = ANALYZER.count_logical_code_differences(
        _pack_rows(reference_codes, bits),
        _pack_rows(candidate_codes, bits),
        bits,
    )

    assert summary["code_count"] == reference_codes.numel()
    assert summary["code_mismatch_count"] == 2
    assert summary["mean_absolute_code_delta"] == pytest.approx(
        (candidate_codes - reference_codes).abs().float().mean().item()
    )
    assert summary["maximum_absolute_code_delta"] == int(
        (candidate_codes - reference_codes).abs().max().item()
    )


def test_count_logical_code_differences_for_three_bit_boundary_codes():
    reference_codes = torch.arange(32 * 4, dtype=torch.int32).reshape(32, 4) % 8
    candidate_codes = reference_codes.clone()
    candidate_codes[10, 1] = 7
    candidate_codes[21, 2] = 0

    summary = ANALYZER.count_logical_code_differences(
        _pack_rows(reference_codes, 3),
        _pack_rows(candidate_codes, 3),
        3,
    )

    expected_delta = (candidate_codes - reference_codes).abs()
    assert summary["code_count"] == reference_codes.numel()
    assert summary["code_mismatch_count"] == int((expected_delta != 0).sum().item())
    assert summary["mean_absolute_code_delta"] == pytest.approx(expected_delta.float().mean().item())
    assert summary["maximum_absolute_code_delta"] == int(expected_delta.max().item())


def test_count_logical_code_differences_reports_exact_packed_parity():
    codes = torch.arange(16 * 7, dtype=torch.int32).reshape(16, 7) % 4
    packed = _pack_rows(codes, 2)

    summary = ANALYZER.count_logical_code_differences(packed, packed.clone(), 2)

    assert summary["packed_word_match_rate"] == 1.0
    assert summary["code_mismatch_count"] == 0
    assert summary["code_mismatch_rate"] == 0.0
