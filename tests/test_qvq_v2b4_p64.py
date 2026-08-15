# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.config import FORMAT, QVQConfig
from gptqmodel.quantization.qvq import (
    QVQ_V2B4_P64_SEGMENTS_PER_TILE,
    pack_qvq_bank_ids,
    pack_trellis_states,
    quantize_qvq_linear,
    reconstruct_qvq_inner_weight,
    tail_biting_v2b4_p64_quantize,
    tail_biting_viterbi_quantize,
    unpack_qvq_bank_ids,
)
from gptqmodel.quantization.qvq_codecs import (
    pgc16_codebook,
    pgc16_codebook_v2_bank,
    pgc16_decode_states,
    pgc16_decode_states_v2_banked,
)
from scripts.compare_qvq_codecs_llama_qkvo import (
    ARM_CONFIG,
    DEFAULT_ARMS,
    _parser,
    _selector_metrics,
)


def test_qvq_v2b4_p64_comparison_harness_defaults_to_matched_v2_control():
    assert DEFAULT_ARMS == ("v2", "v2b2-p32")
    args = _parser().parse_args(("--model", "model", "--dataset", "dataset", "--output", "report.json"))
    assert args.layers == 4
    assert args.rates == (1, 1.5, 2, 2.5)
    assert args.arms == DEFAULT_ARMS
    assert args.calibration_rows == 64
    assert args.evaluation_rows == 64
    assert args.evaluation_row_offset == 64
    assert args.max_length is None
    assert ARM_CONFIG["v2b4-p64"] == {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b4_p64": True,
        "bank_count": 4,
    }
    metrics = _selector_metrics([48, 8, 4, 4])
    assert metrics is not None
    assert metrics["count"] == 64
    assert metrics["histogram"] == [48, 8, 4, 4]
    assert metrics["nonzero_fraction"] == 0.25
    assert metrics["entropy_bits"] == pytest.approx(1.186278124459133)
    assert metrics["selector_bpw"] == 0.03125


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5))
def test_qvq_v2b4_p64_config_round_trip(bits):
    config = QVQConfig(bits=bits, format=FORMAT.QVQ_V2B4_P64, offload_to_disk=False)
    assert config.vector_size == 2
    assert config.trellis_window == 16
    assert config.bank_count == 4
    assert config.calculate_bits_per_weight() is None
    reloaded = QVQConfig.from_quant_config(config.to_dict())
    assert reloaded.format == FORMAT.QVQ_V2B4_P64
    assert reloaded.quant_linear_init_kwargs()["v2b4_p64"] is True


def test_qvq_v2b4_p64_config_rejects_unsupported_math():
    with pytest.raises(ValueError, match="W1 through W2.5"):
        QVQConfig(bits=3, format=FORMAT.QVQ_V2B4_P64, offload_to_disk=False)
    config = QVQConfig(bits=2, format=FORMAT.QVQ_V2B4_P64, rounding="yaqa", offload_to_disk=False)
    assert config.rounding == "yaqa"
    with pytest.raises(ValueError, match="one tail-biting candidate"):
        QVQConfig(
            bits=2,
            format=FORMAT.QVQ_V2B4_P64,
            tail_biting_candidates=2,
            offload_to_disk=False,
        )
    with pytest.raises(ValueError, match="propagation replay"):
        QVQConfig(
            bits=2,
            format=FORMAT.QVQ_V2B4_P64,
            propagated_bank_selection=True,
            offload_to_disk=False,
        )
    with pytest.raises(ValueError, match="bank_count"):
        QVQConfig(bits=2, format=FORMAT.QVQ_V2B4_P64, bank_count=2, offload_to_disk=False)
    with pytest.raises(ValueError, match="qvq_v2b4_p64.*W1 through W2.5"):
        QVQConfig(
            bits=2,
            format=FORMAT.QVQ_V2B4_P64,
            dynamic={r".*q_proj": {"bits": 3}},
            offload_to_disk=False,
        )


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5))
def test_qvq_v2b4_p64_bank_zero_is_exhaustively_bit_exact_v2(bits):
    states = torch.arange(1 << 16, dtype=torch.int64)
    selectors = torch.zeros_like(states, dtype=torch.uint8)
    canonical = pgc16_decode_states(states)
    bank_zero = pgc16_decode_states_v2_banked(states, selectors, bits=bits)
    torch.testing.assert_close(bank_zero, canonical, rtol=0, atol=0)


def test_qvq_v2b4_p64_identical_banks_reproduce_canonical_v2_path():
    generator = torch.Generator().manual_seed(20260815)
    sequences = torch.randn((1, 128, 2), generator=generator)
    canonical = pgc16_codebook(dtype=torch.float32)
    v2 = tail_biting_viterbi_quantize(sequences, canonical, bits=2.5, candidate_count=1)
    v2b4 = tail_biting_v2b4_p64_quantize(
        sequences,
        torch.stack((canonical, canonical, canonical, canonical)),
        bits=2.5,
    )
    assert torch.equal(v2b4.states, v2.states)
    assert torch.equal(v2b4.values, v2.values)
    assert torch.equal(v2b4.squared_error, v2.squared_error)
    assert torch.count_nonzero(v2b4.segment_bank_ids) == 0


def test_qvq_v2b4_p64_selector_round_trip_reconstructs_independent_bank_decoder():
    generator = torch.Generator().manual_seed(47)
    sequence = torch.randn((1, 128, 2), generator=generator)
    banks = torch.stack(tuple(pgc16_codebook_v2_bank(bank, bits=2) for bank in range(4)))
    result = tail_biting_v2b4_p64_quantize(sequence, banks, bits=2)
    trellis = pack_trellis_states(result.states, bits=2)
    packed_selectors = pack_qvq_bank_ids(result.segment_bank_ids.reshape(-1))
    actual = reconstruct_qvq_inner_weight(
        trellis,
        bits=2,
        in_features=16,
        out_features=16,
        bank_ids=packed_selectors,
        v2b4_p64=True,
    )
    assert packed_selectors.numel() == 1
    assert torch.equal(
        unpack_qvq_bank_ids(packed_selectors, QVQ_V2B4_P64_SEGMENTS_PER_TILE),
        result.segment_bank_ids.reshape(-1),
    )
    torch.testing.assert_close(actual, result.values.reshape(16, 16), rtol=0, atol=0)


def test_qvq_v2b4_p64_block_ldlq_pack_reload_and_torch_forward():
    generator = torch.Generator().manual_seed(9)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    result = quantize_qvq_linear(
        weight,
        torch.eye(16),
        bits=2.5,
        bank_count=4,
        v2b4_p64=True,
        trellis_batch_size=1,
    )
    tensors = result.serialized_tensors()
    assert tensors["bank_ids"].dtype == torch.uint8
    assert tensors["bank_ids"].numel() == 1
    decoded = reconstruct_qvq_inner_weight(
        result.trellis,
        bits=2.5,
        in_features=16,
        out_features=16,
        bank_ids=tensors["bank_ids"],
        v2b4_p64=True,
    )
    torch.testing.assert_close(decoded, result.inner_weight, rtol=0, atol=0)

    layer = QVQLinear(
        bits=2.5,
        in_features=16,
        out_features=16,
        bank_count=4,
        v2b4_p64=True,
        tensors=tensors,
    ).eval()
    shell = QVQLinear(bits=2.5, in_features=16, out_features=16, bank_count=4, v2b4_p64=True)
    shell.load_state_dict(layer.state_dict(), strict=True)
    x = torch.randn((3, 16), generator=generator)
    torch.testing.assert_close(layer(x), x @ result.weight.T, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(shell(x), layer(x), rtol=0, atol=0)


def test_qvq_v2b4_p64_full_proxy_cannot_regress_canonical_v2():
    generator = torch.Generator().manual_seed(20260815)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    calibration = torch.randn((97, 16), generator=generator)
    hessian = calibration.T @ calibration / calibration.shape[0]
    canonical = quantize_qvq_linear(weight, hessian, bits=2, trellis_batch_size=1)
    banked = quantize_qvq_linear(
        weight,
        hessian,
        bits=2,
        bank_count=4,
        v2b4_p64=True,
        trellis_batch_size=1,
    )
    assert torch.isfinite(banked.proxy_loss)
    assert banked.proxy_loss <= canonical.proxy_loss
    assert banked.bank_ids is not None
    decoded = reconstruct_qvq_inner_weight(
        banked.trellis,
        bits=2,
        in_features=16,
        out_features=16,
        bank_ids=banked.bank_ids,
        v2b4_p64=True,
    )
    torch.testing.assert_close(decoded, banked.inner_weight, rtol=0, atol=0)


def test_qvq_v2b4_p64_yaqa_pack_reload_and_full_proxy_cannot_regress_v2_yaqa():
    generator = torch.Generator().manual_seed(20260817)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    input_hessian = torch.eye(16)
    output_hessian = torch.eye(16)
    canonical = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        rounding="yaqa",
        output_hessian=output_hessian,
        trellis_batch_size=1,
    )
    banked = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        rounding="yaqa",
        output_hessian=output_hessian,
        bank_count=4,
        v2b4_p64=True,
        trellis_batch_size=1,
    )
    block_control = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        bank_count=4,
        v2b4_p64=True,
        trellis_batch_size=1,
    )
    assert banked.kronecker_proxy_loss <= canonical.kronecker_proxy_loss
    assert banked.bank_ids is not None and banked.bank_ids.numel() == 4
    assert isinstance(banked.yaqa_bank_fallback_to_v2, bool)
    assert banked.yaqa_selector_churn is not None and 0.0 <= banked.yaqa_selector_churn <= 1.0
    assert banked.yaqa_family_changed is None
    assert banked.yaqa_block_family_id is None
    decoded = reconstruct_qvq_inner_weight(
        banked.trellis,
        bits=2,
        in_features=16,
        out_features=16,
        bank_ids=banked.serialized_tensors()["bank_ids"],
        v2b4_p64=True,
    )
    torch.testing.assert_close(decoded, banked.inner_weight, rtol=0, atol=0)
    assert torch.equal(banked.trellis, block_control.trellis)
    assert torch.equal(banked.bank_ids, block_control.bank_ids)


def test_qvq_v2b4_p64_rejects_invalid_selector_payload():
    trellis = torch.zeros((1, 16), dtype=torch.int32)
    with pytest.raises(ValueError, match="invalid tile count"):
        reconstruct_qvq_inner_weight(
            trellis,
            bits=2,
            in_features=16,
            out_features=16,
            bank_ids=torch.zeros(2, dtype=torch.uint8),
            v2b4_p64=True,
        )
