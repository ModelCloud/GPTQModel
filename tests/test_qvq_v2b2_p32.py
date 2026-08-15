# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.config import FORMAT, QVQConfig
from gptqmodel.quantization.qvq import (
    QVQ_V2B2_P32_SEGMENTS_PER_TILE,
    pack_qvq_binary_bank_ids,
    pack_trellis_states,
    quantize_qvq_linear,
    reconstruct_qvq_inner_weight,
    tail_biting_v2b2_p32_quantize,
    tail_biting_viterbi_quantize,
    unpack_qvq_binary_bank_ids,
)
from gptqmodel.quantization.qvq_codecs import pgc16_codebook, pgc16_codebook_v2_bank
from scripts.compare_qvq_codecs_llama_qkvo import ARM_CONFIG, DEFAULT_ARMS, _parser


def test_qvq_v2b2_p32_is_the_default_matched_model_comparison():
    assert DEFAULT_ARMS == ("v2", "v2b2-p32")
    args = _parser().parse_args(("--model", "model", "--dataset", "dataset", "--output", "report.json"))
    assert args.layers == 4
    assert args.calibration_rows == 64
    assert args.evaluation_rows == 64
    assert args.evaluation_row_offset == 64
    assert args.max_length is None
    assert ARM_CONFIG["v2b2-p32"] == {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b2_p32": True,
        "bank_count": 2,
    }


def test_qvq_v2b2_p32_native_mlx_conversion_fails_closed_until_selector_support_exists():
    from gptqmodel.utils.mlx import _qvq_mlx_linear_from_torch

    layer = QVQLinear(bits=2, in_features=16, out_features=16, bank_count=2, v2b2_p32=True)
    with pytest.raises(NotImplementedError, match="binary P32 selector"):
        _qvq_mlx_linear_from_torch(layer)


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5))
def test_qvq_v2b2_p32_config_round_trip(bits):
    config = QVQConfig(bits=bits, format=FORMAT.QVQ_V2B2_P32, offload_to_disk=False)
    assert config.vector_size == 2
    assert config.trellis_window == 16
    assert config.bank_count == 2
    reloaded = QVQConfig.from_quant_config(config.to_dict())
    assert reloaded.format == FORMAT.QVQ_V2B2_P32
    assert reloaded.quant_linear_init_kwargs()["v2b2_p32"] is True


def test_qvq_v2b2_p32_config_rejects_unimplemented_objectives():
    with pytest.raises(ValueError, match="W1 through W2.5"):
        QVQConfig(bits=3, format=FORMAT.QVQ_V2B2_P32, offload_to_disk=False)
    with pytest.raises(ValueError, match="Block-LDLQ"):
        QVQConfig(bits=2, format=FORMAT.QVQ_V2B2_P32, rounding="yaqa", offload_to_disk=False)
    with pytest.raises(ValueError, match="one tail-biting candidate"):
        QVQConfig(
            bits=2,
            format=FORMAT.QVQ_V2B2_P32,
            tail_biting_candidates=2,
            offload_to_disk=False,
        )
    with pytest.raises(ValueError, match="propagation replay"):
        QVQConfig(
            bits=2,
            format=FORMAT.QVQ_V2B2_P32,
            propagated_bank_selection=True,
            offload_to_disk=False,
        )


def test_qvq_v2b2_p32_binary_selector_round_trip_and_validation():
    selectors = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=torch.uint8)
    packed = pack_qvq_binary_bank_ids(selectors)
    assert packed.tolist() == [0b10010110, 0b00000001]
    assert torch.equal(unpack_qvq_binary_bank_ids(packed, selectors.numel()), selectors)
    with pytest.raises(ValueError, match="binary"):
        pack_qvq_binary_bank_ids(torch.tensor([0, 2], dtype=torch.uint8))
    with pytest.raises(ValueError, match="invalid selector count"):
        unpack_qvq_binary_bank_ids(torch.zeros(2, dtype=torch.uint8), 8)


def test_qvq_v2b2_p32_identical_banks_reproduce_canonical_v2_path():
    generator = torch.Generator().manual_seed(20260815)
    sequences = torch.randn((1, 128, 2), generator=generator)
    canonical = pgc16_codebook(dtype=torch.float32)
    v2 = tail_biting_viterbi_quantize(sequences, canonical, bits=2.5, candidate_count=1)
    v2b2 = tail_biting_v2b2_p32_quantize(
        sequences,
        torch.stack((canonical, canonical)),
        bits=2.5,
    )
    assert torch.equal(v2b2.states, v2.states)
    assert torch.equal(v2b2.values, v2.values)
    assert torch.equal(v2b2.squared_error, v2.squared_error)
    assert torch.count_nonzero(v2b2.segment_bank_ids) == 0


def test_qvq_v2b2_p32_selector_round_trip_reconstructs_selected_alternative():
    generator = torch.Generator().manual_seed(47)
    sequence = torch.randn((1, 128, 2), generator=generator)
    alt_id = 3
    banks = torch.stack(
        (
            pgc16_codebook_v2_bank(0, bits=2),
            pgc16_codebook_v2_bank(alt_id, bits=2),
        )
    )
    result = tail_biting_v2b2_p32_quantize(sequence, banks, bits=2)
    trellis = pack_trellis_states(result.states, bits=2)
    packed_selectors = pack_qvq_binary_bank_ids(result.segment_bank_ids.reshape(-1))
    actual = reconstruct_qvq_inner_weight(
        trellis,
        bits=2,
        in_features=16,
        out_features=16,
        bank_ids=packed_selectors,
        v2b2_p32=True,
        bank_alt_id=torch.tensor([alt_id], dtype=torch.uint8),
    )
    assert packed_selectors.numel() == 1
    assert torch.equal(
        unpack_qvq_binary_bank_ids(packed_selectors, QVQ_V2B2_P32_SEGMENTS_PER_TILE),
        result.segment_bank_ids.reshape(-1),
    )
    torch.testing.assert_close(actual, result.values.reshape(16, 16), rtol=0, atol=0)


def test_qvq_v2b2_p32_block_ldlq_pack_reload_and_torch_forward():
    generator = torch.Generator().manual_seed(9)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    result = quantize_qvq_linear(
        weight,
        torch.eye(16),
        bits=2,
        bank_count=2,
        v2b2_p32=True,
        trellis_batch_size=1,
    )
    tensors = result.serialized_tensors()
    assert result.bank_ids is not None and result.bank_ids.numel() == 8
    assert tensors["bank_ids"].dtype == torch.uint8
    assert tensors["bank_ids"].numel() == 1
    assert tensors["bank_alt_id"].shape == (1,)
    assert 1 <= int(tensors["bank_alt_id"].item()) <= 3
    decoded = reconstruct_qvq_inner_weight(
        result.trellis,
        bits=2,
        in_features=16,
        out_features=16,
        bank_ids=tensors["bank_ids"],
        v2b2_p32=True,
        bank_alt_id=tensors["bank_alt_id"],
    )
    torch.testing.assert_close(decoded, result.inner_weight, rtol=0, atol=0)

    layer = QVQLinear(
        bits=2,
        in_features=16,
        out_features=16,
        bank_count=2,
        v2b2_p32=True,
        tensors=tensors,
    ).eval()
    shell = QVQLinear(bits=2, in_features=16, out_features=16, bank_count=2, v2b2_p32=True)
    shell.load_state_dict(layer.state_dict(), strict=True)
    x = torch.randn((3, 16), generator=generator)
    torch.testing.assert_close(layer(x), x @ result.weight.T, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(shell(x), layer(x), rtol=0, atol=0)


def test_qvq_v2b2_p32_full_proxy_cannot_regress_independent_v2_oracle():
    generator = torch.Generator().manual_seed(20260815)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    calibration = torch.randn((97, 16), generator=generator)
    hessian = calibration.T @ calibration / calibration.shape[0]
    canonical = quantize_qvq_linear(weight, hessian, bits=2, trellis_batch_size=1)
    banked = quantize_qvq_linear(
        weight,
        hessian,
        bits=2,
        bank_count=2,
        v2b2_p32=True,
        trellis_batch_size=1,
    )
    assert torch.isfinite(banked.proxy_loss)
    assert banked.proxy_loss <= canonical.proxy_loss
    assert banked.bank_ids is not None
    assert banked.bank_alt_id is not None


@pytest.mark.parametrize("alt_id", (0, 4))
def test_qvq_v2b2_p32_rejects_invalid_alternative_bank(alt_id):
    trellis = torch.zeros((1, 16), dtype=torch.int32)
    with pytest.raises(ValueError, match=r"in \[1, 3\]"):
        reconstruct_qvq_inner_weight(
            trellis,
            bits=2,
            in_features=16,
            out_features=16,
            bank_ids=torch.zeros(1, dtype=torch.uint8),
            v2b2_p32=True,
            bank_alt_id=torch.tensor([alt_id], dtype=torch.uint8),
        )
