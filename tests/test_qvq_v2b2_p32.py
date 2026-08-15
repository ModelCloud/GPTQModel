# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.config import FORMAT, QVQConfig, YaqaConfig
from gptqmodel.quantization.qvq import (
    QVQ_V2B2_P32_SEGMENTS_PER_TILE,
    block_ldlq_inner,
    pack_qvq_binary_bank_ids,
    pack_trellis_states,
    quantize_qvq_linear,
    reconstruct_qvq_inner_weight,
    tail_biting_v2b2_p32_quantize,
    tail_biting_viterbi_quantize,
    unpack_qvq_binary_bank_ids,
    yaqa_inner_v2b2_p32,
)
from gptqmodel.quantization.qvq_codecs import pgc16_codebook, pgc16_codebook_v2_bank
from scripts.compare_qvq_codecs_llama_qkvo import (
    ARM_CONFIG,
    DEFAULT_ARMS,
    _load_yaqa_factor_cache,
    _padded_batch_chunks,
    _parser,
    _save_yaqa_factor_cache,
)


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


def test_qvq_banked_yaqa_sweep_arms_and_disjoint_batch_contract():
    assert ARM_CONFIG["v2-yaqa"]["rounding"] == "yaqa"
    assert ARM_CONFIG["v2b2-p32-yaqa-fixed"]["yaqa_v2b2_family_mode"] == "fixed_block_ldlq"
    assert ARM_CONFIG["v2b2-p32-yaqa"]["yaqa_v2b2_family_mode"] == "reselect"
    args = _parser().parse_args(
        (
            "--model", "model", "--dataset", "dataset", "--output", "report.json",
            "--arms", "v2", "v2-yaqa", "v2b2-p32-yaqa",
            "--calibration-rows", "512", "--evaluation-rows", "512", "--evaluation-row-offset", "512",
            "--yaqa-rows", "512", "--yaqa-row-offset", "1024",
        )
    )
    assert args.yaqa_batch_size == 8
    assert (args.calibration_rows, args.evaluation_row_offset, args.yaqa_row_offset) == (512, 512, 1024)

    encoded = {
        "input_ids": torch.tensor([[0, 0, 11, 12, 13], [0, 21, 22, 23, 24], [0, 0, 0, 31, 32]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [0, 0, 0, 1, 1]]),
    }
    batches = _padded_batch_chunks(encoded, batch_size=2)
    assert [tuple(batch["attention_mask"].shape) for batch in batches] == [(2, 4), (1, 2)]
    assert sum(int(batch["attention_mask"].sum()) for batch in batches) == 9


def test_qvq_banked_yaqa_factor_cache_is_atomic_and_validated(tmp_path):
    path = tmp_path / "sketch_b.pt"
    metadata = {"version": 1, "module_shapes": {"proj": [3, 2]}}
    input_hessians = {"proj": torch.eye(2)}
    output_hessians = {"proj": torch.eye(3)}
    stats = {"independent_sequences": 512}
    _save_yaqa_factor_cache(
        path,
        metadata=metadata,
        input_hessians=input_hessians,
        output_hessians=output_hessians,
        stats=stats,
    )

    loaded_input, loaded_output, loaded_stats = _load_yaqa_factor_cache(path, expected_metadata=metadata)
    assert torch.equal(loaded_input["proj"], input_hessians["proj"])
    assert torch.equal(loaded_output["proj"], output_hessians["proj"])
    assert loaded_stats == stats
    assert not tuple(tmp_path.glob(".*.tmp"))
    with pytest.raises(ValueError, match="metadata does not match"):
        _load_yaqa_factor_cache(path, expected_metadata={**metadata, "version": 2})


def test_qvq_v2b2_p32_native_mlx_conversion_preserves_selector_payload():
    pytest.importorskip("mlx.core")
    from gptqmodel.utils.mlx import _qvq_mlx_linear_from_torch
    from gptqmodel.utils.qvq_mlx import QVQMLXLinear

    layer = QVQLinear(bits=2, in_features=16, out_features=16, bank_count=2, v2b2_p32=True)
    converted = _qvq_mlx_linear_from_torch(layer)

    assert isinstance(converted, QVQMLXLinear)
    assert converted.v2b2_p32 is True
    assert converted.v2b4_p64 is False
    assert converted.bank_ids.shape == (1,)
    assert converted.bank_alt_id.shape == (1,)


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5))
def test_qvq_v2b2_p32_config_round_trip(bits):
    config = QVQConfig(bits=bits, format=FORMAT.QVQ_V2B2_P32, offload_to_disk=False)
    assert config.vector_size == 2
    assert config.trellis_window == 16
    assert config.bank_count == 2
    reloaded = QVQConfig.from_quant_config(config.to_dict())
    assert reloaded.format == FORMAT.QVQ_V2B2_P32
    assert reloaded.quant_linear_init_kwargs()["v2b2_p32"] is True


def test_qvq_v2b2_p32_config_accepts_yaqa_and_rejects_unimplemented_objectives():
    with pytest.raises(ValueError, match="W1 through W2.5"):
        QVQConfig(bits=3, format=FORMAT.QVQ_V2B2_P32, offload_to_disk=False)
    config = QVQConfig(bits=2, format=FORMAT.QVQ_V2B2_P32, rounding="yaqa", offload_to_disk=False)
    assert config.rounding == "yaqa"
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


@pytest.mark.parametrize("family_mode", ("fixed_block_ldlq", "reselect"))
def test_qvq_v2b2_p32_yaqa_family_mode_config_round_trip(family_mode):
    config = QVQConfig(
        bits=2,
        format=FORMAT.QVQ_V2B2_P32,
        rounding="yaqa",
        yaqa=YaqaConfig(v2b2_family_mode=family_mode),
        offload_to_disk=False,
    )
    reloaded = QVQConfig.from_quant_config(config.to_dict())
    assert reloaded.yaqa.v2b2_family_mode == family_mode


@pytest.mark.parametrize("family_mode", (None, "unknown"))
def test_qvq_v2b2_p32_rejects_invalid_yaqa_family_mode(family_mode):
    error = TypeError if family_mode is None else ValueError
    with pytest.raises(error, match="v2b2_family_mode"):
        YaqaConfig(v2b2_family_mode=family_mode)


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


def test_qvq_v2b2_p32_yaqa_pack_reload_and_full_proxy_cannot_regress_v2_yaqa():
    generator = torch.Generator().manual_seed(20260818)
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
        bank_count=2,
        v2b2_p32=True,
        trellis_batch_size=1,
    )
    block_control = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        bank_count=2,
        v2b2_p32=True,
        trellis_batch_size=1,
    )
    assert banked.kronecker_proxy_loss <= canonical.kronecker_proxy_loss
    assert banked.bank_ids is not None and banked.bank_ids.numel() == 8
    assert banked.bank_alt_id is not None
    assert isinstance(banked.yaqa_bank_fallback_to_v2, bool)
    assert banked.yaqa_selector_churn is not None and 0.0 <= banked.yaqa_selector_churn <= 1.0
    assert isinstance(banked.yaqa_family_changed, bool)
    assert banked.yaqa_block_family_id in {1, 2, 3}
    decoded = reconstruct_qvq_inner_weight(
        banked.trellis,
        bits=2,
        in_features=16,
        out_features=16,
        bank_ids=banked.serialized_tensors()["bank_ids"],
        v2b2_p32=True,
        bank_alt_id=banked.bank_alt_id,
    )
    torch.testing.assert_close(decoded, banked.inner_weight, rtol=0, atol=0)
    assert torch.equal(banked.trellis, block_control.trellis)
    assert torch.equal(banked.bank_ids, block_control.bank_ids)
    assert torch.equal(banked.bank_alt_id, block_control.bank_alt_id)


def test_qvq_v2b2_p32_fixed_yaqa_uses_matched_block_ldlq_damping_for_family():
    generator = torch.Generator().manual_seed(20260819)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    input_samples = torch.randn((41, 16), generator=generator)
    output_samples = torch.randn((43, 16), generator=generator)
    input_hessian = input_samples.T @ input_samples / input_samples.shape[0]
    output_hessian = output_samples.T @ output_samples / output_samples.shape[0]
    block = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        bank_count=2,
        v2b2_p32=True,
        trellis_batch_size=1,
    )
    fixed = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        rounding="yaqa",
        output_hessian=output_hessian,
        bank_count=2,
        v2b2_p32=True,
        yaqa_v2b2_family_mode="fixed_block_ldlq",
        trellis_batch_size=1,
    )

    expected_family = int(block.bank_alt_id.item())
    assert fixed.yaqa_block_family_id == expected_family
    assert int(fixed.bank_alt_id.item()) == expected_family


@pytest.mark.parametrize(
    ("family_mode", "expected_family", "expected_changed"),
    (("fixed_block_ldlq", 2, False), ("reselect", 3, True)),
)
def test_qvq_v2b2_p32_yaqa_fixed_and_reselected_family_objectives(
    family_mode,
    expected_family,
    expected_changed,
):
    weight = torch.zeros((16, 16))
    hessian = torch.eye(16)
    states = torch.zeros((1, 128), dtype=torch.long)
    block_selectors = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.uint8)
    codebooks = tuple(torch.full((1,), family_id, dtype=torch.float32) for family_id in range(4))

    def fake_block(*args, **kwargs):
        del args, kwargs
        return weight, states, block_selectors, torch.tensor([2], dtype=torch.uint8)

    def fake_yaqa(*args, bank_codebooks=None, **kwargs):
        del args, kwargs
        if bank_codebooks is None:
            return torch.ones_like(weight), states
        family_id = int(bank_codebooks[1].item())
        candidate = torch.full_like(weight, {1: 0.3, 2: 0.2, 3: 0.1}[family_id])
        selectors = torch.full((8,), family_id & 1, dtype=torch.uint8)
        return candidate, states, selectors

    diagnostics = {}
    with (
        patch("gptqmodel.quantization.qvq.block_ldlq_inner_v2b2_p32", side_effect=fake_block),
        patch("gptqmodel.quantization.qvq.yaqa_inner", side_effect=fake_yaqa),
    ):
        _, _, selectors, family = yaqa_inner_v2b2_p32(
            weight,
            hessian,
            hessian,
            codebooks,
            bits=2,
            family_mode=family_mode,
            diagnostics=diagnostics,
        )

    assert int(family.item()) == expected_family
    assert diagnostics == {
        "fallback_to_v2": False,
        "selector_churn": float((selectors != block_selectors).to(torch.float32).mean()),
        "family_changed": expected_changed,
        "block_family_id": 2,
    }


def test_qvq_v2b2_p32_reuses_one_canonical_block_ldlq_oracle():
    generator = torch.Generator().manual_seed(20260816)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    hessian = torch.eye(16)
    with patch("gptqmodel.quantization.qvq.block_ldlq_inner", wraps=block_ldlq_inner) as canonical:
        quantize_qvq_linear(
            weight,
            hessian,
            bits=2,
            bank_count=2,
            v2b2_p32=True,
            trellis_batch_size=1,
        )
    assert canonical.call_count == 1


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
