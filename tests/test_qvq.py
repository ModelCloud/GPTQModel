# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import gc
import io
import itertools
import weakref
from dataclasses import fields
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import gptqmodel.quantization.qvq as qvq_module
import gptqmodel.utils.qvq_cuda as qvq_cuda_module
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.qvq import (
    QVQLinear,
    QVQReferenceLinear,
    _qvq_compute_dtype,
    qvq_dense_oracle_forward,
)
from gptqmodel.quantization import (
    FORMAT,
    METHOD,
    OutputAlignConfig,
    QuantizeConfig,
    QVQConfig,
)
from gptqmodel.quantization.config import QUANTIZE_BLACK_LIST, YaqaConfig
from gptqmodel.quantization.protocol import compile_protocol_to_quantize_config
from gptqmodel.quantization.qvq import (
    QVQCandidate,
    QVQPropagationRefiner,
    batched_viterbi_quantize,
    bitshift_next_state,
    block_ldl_factor,
    block_ldlq_inner,
    block_ldlq_inner_banked,
    decode_trellis_tiles,
    default_qvq_trellis_batch_size,
    optimize_qvq_module_scale,
    optimize_qvq_output_channel_scales,
    pack_qvq_bank_ids,
    pack_dual_v2_states,
    pack_trellis_states,
    quantize_qvq_linear,
    qvq_proxy_loss,
    reconstruct_qvq_inner_weight,
    rht_preprocess_hessian,
    rht_preprocess_weight,
    rht_reconstruct_weight,
    select_banked_tiles_by_output_error,
    tail_biting_viterbi_quantize,
    unpack_qvq_bank_ids,
    unpack_dual_v2_states,
    unpack_trellis_states,
    viterbi_quantize,
    yaqa_inner,
    yaqa_proxy_loss,
    yaqa_sketch_b,
)
from gptqmodel.quantization.qvq_codecs import (
    PGC16_CODEBOOK_VERSION,
    PGC16_NORMALIZATION_RMS,
    PGC16_SCALE_FACTORS,
    PGC16_STATE_COUNT,
    PGC16_V4_BANK_COUNT,
    PGC16_V4_BANK_XOR_MASKS,
    PGC16_V4_BANK_XOR_MASKS_BY_TRANSITION_BITS,
    canonical_pgc16_levels,
    pgc16_codebook,
    pgc16_codebook_v4_bank,
    pgc16_decode_states,
    pgc16_decode_states_v4,
    pgc16_decode_states_v4_banked,
    pgc16_mix_states,
    pgc18_codebook_v4,
    pgc18_decode_states_v4,
)
from gptqmodel.quantization.qvq_codecs.hyb_reference import (
    canonical_hyb_lut,
    fit_hyb_lut,
    hyb_codebook,
    hyb_decode_states,
)
from gptqmodel.quantization.qvq_codecs.uniform_reference import uniform_codebook
from gptqmodel.quantization.qvq_rates import (
    QVQ_BITS as QVQ_HALF_STEP_BITS,
)
from gptqmodel.quantization.qvq_rates import (
    QVQ_TRANSITION_BITS,
    normalize_qvq_rate,
    qvq_rate_from_transition_bits,
    qvq_transition_bits,
    qvq_words_per_tile,
)
from gptqmodel.quantization.qvq_yaqa import (
    YAQA_PAPER_MINIMUM_SEQUENCES,
    YAQA_DEFAULT_REGULARIZATION,
    YAQA_PAPER_REGULARIZATION,
)
from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear
from gptqmodel.utils.model import make_quant
from gptqmodel.utils.planar_packing import planar_pack_rows, planar_unpack_rows
from gptqmodel.utils.qvq_mps import (
    qvq_hyb_reference_mps_gemv,
    qvq_mps_gemv,
    qvq_mps_supported,
)


@pytest.mark.parametrize("transition_bits", QVQ_TRANSITION_BITS)
def test_qvq_half_step_rate_contract_is_exact(transition_bits):
    rate = qvq_rate_from_transition_bits(transition_bits)

    assert rate == QVQ_HALF_STEP_BITS[transition_bits - 2]
    assert qvq_transition_bits(rate) == transition_bits
    assert qvq_words_per_tile(rate) == 4 * transition_bits


@pytest.mark.parametrize(
    ("rate", "expected"),
    (("w1", 1), ("1.5", 1.5), ("3/2", 1.5), (2.0, 2), (8, 8)),
)
def test_qvq_rate_normalization_accepts_exact_public_forms(rate, expected):
    assert normalize_qvq_rate(rate) == expected


@pytest.mark.parametrize(
    ("rate", "exception"),
    ((True, TypeError), (object(), TypeError), (0.5, ValueError), (8.5, ValueError), (1.25, ValueError),
     (float("inf"), ValueError), ("nope", ValueError)),
)
def test_qvq_rate_normalization_rejects_noncanonical_values(rate, exception):
    with pytest.raises(exception, match="rate|finite|Unsupported"):
        normalize_qvq_rate(rate)


@pytest.mark.parametrize("transition_bits", range(1, 17))
def test_qvq_transition_planar_primitive_round_trips_all_widths(transition_bits):
    generator = torch.Generator().manual_seed(9100 + transition_bits)
    edges = torch.randint(0, 1 << transition_bits, (128, 3), generator=generator, dtype=torch.int32)
    edges[0] = 0
    edges[-1] = (1 << transition_bits) - 1

    packed = planar_pack_rows(edges, transition_bits)

    assert packed.shape == (4 * transition_bits, 3)
    assert torch.equal(planar_unpack_rows(packed, transition_bits), edges)


@pytest.mark.parametrize(("value", "message"), ((-1, "must be in"), (8, "must be in")))
def test_qvq_transition_planar_primitive_rejects_out_of_range_codes(value, message):
    edges = torch.zeros((32, 1), dtype=torch.int32)
    edges[0, 0] = value
    with pytest.raises(ValueError, match=message):
        planar_pack_rows(edges, 3)


@pytest.mark.parametrize(
    ("call", "message"),
    (
        (lambda: qvq_transition_bits(2, vector_size=0), "vector size"),
        (lambda: qvq_transition_bits(1.5, vector_size=3), "integer transition"),
        (lambda: qvq_rate_from_transition_bits(True), "transition width"),
        (lambda: qvq_rate_from_transition_bits(4, vector_size=0), "vector size"),
        (lambda: qvq_rate_from_transition_bits(1), "rate"),
        (lambda: qvq_words_per_tile(2, weight_count=0), "weight count"),
        (lambda: qvq_words_per_tile(2, word_bits=True), "word width"),
        (lambda: qvq_words_per_tile(2, vector_size=0), "vector size"),
        (lambda: qvq_words_per_tile(2, weight_count=255), "divisible"),
        (lambda: qvq_words_per_tile(2, weight_count=2), "whole number"),
    ),
)
def test_qvq_rate_geometry_helpers_reject_invalid_contracts(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_qvq_config_round_trip_preserves_trellis_contract():
    tensor_storage = {
        "model.layers.0.self_attn.q_proj": {
            "trellis": {"shape": [16, 16], "torch_dtype": "int32"},
            "SU": {"shape": [2048], "torch_dtype": "float16"},
            "SV": {"shape": [2048], "torch_dtype": "float16"},
        }
    }
    cfg = QuantizeConfig(
        method="qvq",
        bits=2.5,
        tensor_storage=tensor_storage,
        offload_to_disk=False,
    )

    assert isinstance(cfg, QVQConfig)
    assert cfg.method == METHOD.QVQ
    assert cfg.format == FORMAT.QVQ
    assert cfg.bits == 2.5
    assert cfg.pack_dtype == torch.int32
    assert cfg.group_size == -1
    assert cfg.desc_act is False
    assert cfg.codebook == PGC16_CODEBOOK_VERSION
    assert cfg.module_scale_search is False
    assert cfg.output_channel_scale_optimization is False
    assert cfg.viterbi_objective == "euclidean"
    assert cfg.tail_biting_candidates == 1
    assert cfg.viterbi_minimum_proxy_improvement == 0.0
    assert cfg.output_alignment is None

    payload = cfg.to_dict()
    assert "desc_act" not in payload
    reloaded = QuantizeConfig.from_quant_config(payload)

    assert isinstance(reloaded, QVQConfig)
    assert reloaded.to_dict() == payload
    assert reloaded.tensor_storage == tensor_storage
    assert reloaded.trellis_window == 16
    assert reloaded.vector_size == 2
    assert reloaded.tile_rows == 16
    assert reloaded.tile_cols == 16


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 3, 3.5, 4))
def test_qvq_v4_toggle_round_trips_and_reconstructs(bits):
    cfg = QVQConfig(bits=bits, format="qvq_v4", rounding="block_ldlq", offload_to_disk=False)
    assert cfg.vector_size == 4
    payload = cfg.to_dict()
    reloaded = QuantizeConfig.from_quant_config(payload)
    assert reloaded.format.value == "qvq_v4"
    assert reloaded.vector_size == 4

    vector_shift = qvq_transition_bits(bits, vector_size=4)
    generator = torch.Generator().manual_seed(9200 + vector_shift)
    edges = torch.randint(0, 1 << vector_shift, (1, 64), generator=generator, dtype=torch.int64)
    stream = torch.stack([(edges >> bit) & 1 for bit in range(vector_shift - 1, -1, -1)], dim=-1).reshape(1, -1)
    states = torch.empty((1, 64), dtype=torch.int64)
    for step in range(64):
        offsets = (torch.arange((step + 1) * vector_shift - 16, (step + 1) * vector_shift) % stream.shape[1])
        state = torch.zeros(1, dtype=torch.int64)
        for offset in offsets:
            state = (state << 1) | stream[:, offset]
        states[:, step] = state
    packed = pack_trellis_states(states, bits=bits, vector_size=4)
    assert torch.equal(unpack_trellis_states(packed, bits=bits, vector_size=4), states)
    decoded = decode_trellis_tiles(packed, bits=bits, vector_size=4)
    assert decoded.shape == (1, 256)


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5))
def test_qvq_l18_v4_torch_pack_decode_matches_contextual_codebook(bits):
    transition_bits = qvq_transition_bits(bits, vector_size=4)
    generator = torch.Generator().manual_seed(18200 + transition_bits)
    edges = torch.randint(0, 1 << transition_bits, (1, 64), generator=generator, dtype=torch.int64)
    packed_edges = planar_pack_rows(edges.T.contiguous(), transition_bits).T.contiguous()
    states = unpack_trellis_states(
        packed_edges,
        bits=bits,
        vector_size=4,
        trellis_window=18,
    )
    trellis = pack_trellis_states(
        states,
        bits=bits,
        vector_size=4,
        trellis_window=18,
    )

    assert torch.equal(trellis, packed_edges)
    decoded = decode_trellis_tiles(
        trellis,
        bits=bits,
        vector_size=4,
        trellis_window=18,
    )
    direct = pgc18_decode_states_v4(states, bits=bits).reshape(1, 256).float()
    codebook = pgc18_codebook_v4(bits=bits)

    torch.testing.assert_close(decoded, direct, rtol=0, atol=0)
    torch.testing.assert_close(codebook[states].reshape(1, 256), direct, rtol=0, atol=0)
    assert int(states.max()) < 1 << 18
    assert torch.unique(states >> 16).numel() > 1


def test_qvq_l18_v4_torch_viterbi_recovers_exact_transition_consistent_path():
    bits = 2
    transition_bits = qvq_transition_bits(bits, vector_size=4)
    generator = torch.Generator().manual_seed(18208)
    edges = torch.randint(0, 1 << transition_bits, (1, 32), generator=generator, dtype=torch.int64)
    trellis = planar_pack_rows(edges.T.contiguous(), transition_bits).T.contiguous()
    states = unpack_trellis_states(trellis, bits=bits, vector_size=4, trellis_window=18)
    codebook = pgc18_codebook_v4(bits=bits)
    targets = codebook[states]
    overlap = states[:, 0] >> transition_bits

    result = batched_viterbi_quantize(targets, codebook, bits=bits, overlap=overlap)

    torch.testing.assert_close(result.values, targets, rtol=0, atol=0)
    torch.testing.assert_close(result.squared_error, torch.zeros_like(result.squared_error), rtol=0, atol=1e-6)


def test_qvq_l18_v4_block_quantize_pack_and_torch_linear_are_synchronized():
    bits = 2
    generator = torch.Generator().manual_seed(18218)
    weight = torch.randn((16, 16), generator=generator)
    inputs = torch.randn((5, 16), generator=generator)
    result = quantize_qvq_linear(
        weight,
        torch.eye(16),
        bits=bits,
        vector_size=4,
        trellis_window=18,
        trellis_batch_size=1,
    )
    layer = QVQLinear(
        bits=bits,
        in_features=16,
        out_features=16,
        tensors=result.serialized_tensors(),
        vector_size=4,
        trellis_window=18,
    ).eval()
    reloaded = QVQLinear(
        bits=bits,
        in_features=16,
        out_features=16,
        vector_size=4,
        trellis_window=18,
        register_buffers=True,
    ).eval()
    reloaded.load_state_dict(layer.state_dict(), strict=True)

    reconstructed = layer.get_inner_weight_tensor(dtype=torch.float32)
    actual = layer(inputs)
    reloaded_actual = reloaded(inputs)
    expected = inputs @ result.weight.float().T

    torch.testing.assert_close(reconstructed, result.inner_weight.float(), rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(reloaded_actual, actual, rtol=0, atol=0)
    assert result.trellis.shape == (1, qvq_words_per_tile(bits, vector_size=4))
    assert torch.isfinite(actual).all()


def test_qvq_l18_v4_yaqa_quantize_pack_and_torch_linear_are_synchronized():
    bits = 2
    generator = torch.Generator().manual_seed(18219)
    weight = torch.randn((16, 16), generator=generator)
    inputs = torch.randn((5, 16), generator=generator)
    input_hessian = inputs.T @ inputs / inputs.shape[0]
    output_hessian = torch.eye(16)
    result = quantize_qvq_linear(
        weight,
        input_hessian,
        output_hessian=output_hessian,
        bits=bits,
        rounding="yaqa",
        vector_size=4,
        trellis_window=18,
        trellis_batch_size=1,
    )
    layer = QVQLinear(
        bits=bits,
        in_features=16,
        out_features=16,
        tensors=result.serialized_tensors(),
        vector_size=4,
        trellis_window=18,
    ).eval()

    reconstructed = layer.get_inner_weight_tensor(dtype=torch.float32)
    actual = layer(inputs)
    expected = inputs @ result.weight.float().T

    torch.testing.assert_close(reconstructed, result.inner_weight.float(), rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert result.rounding == "yaqa"
    assert result.kronecker_proxy_loss is not None
    assert torch.isfinite(result.kronecker_proxy_loss)


def test_qvq_v4_toggle_rejects_high_rate_and_manual_vector_size():
    with pytest.raises(ValueError, match="unsupported bits|W1 through W4"):
        QVQConfig(bits=4.5, format="qvq_v4", offload_to_disk=False)
    with pytest.raises(ValueError, match="requires `format=qvq_v4`"):
        QVQConfig(bits=2, vector_size=4, offload_to_disk=False)


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5))
def test_qvq_l18_v4_format_round_trips_with_fixed_geometry(bits):
    cfg = QVQConfig(bits=bits, format="qvq_v4_l18", offload_to_disk=False)
    payload = cfg.to_dict()
    reloaded = QuantizeConfig.from_quant_config(payload)

    assert cfg.format == FORMAT.QVQ_V4_L18
    assert cfg.vector_size == 4
    assert cfg.trellis_window == 18
    assert cfg.bank_count == 1
    assert reloaded.to_dict() == payload
    assert cfg.quant_linear_init_kwargs()["trellis_window"] == 18


def test_qvq_l18_v4_format_rejects_unsupported_rate_and_explicit_banks():
    with pytest.raises(ValueError, match="W1 through W2.5"):
        QVQConfig(bits=3, format="qvq_v4_l18", offload_to_disk=False)
    with pytest.raises(ValueError, match="implicit history banks"):
        QVQConfig(bits=2, format="qvq_v4_l18", bank_count=4, offload_to_disk=False)
    with pytest.raises(NotImplementedError, match=r"supports.*\[1, 1.5, 2, 2.5\]"):
        QVQLinear(bits=3, in_features=16, out_features=16, vector_size=4, trellis_window=18)
    with pytest.raises(ValueError, match="W1 through W2.5"):
        quantize_qvq_linear(
            torch.eye(16),
            torch.eye(16),
            bits=3,
            vector_size=4,
            trellis_window=18,
        )


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 4, 8))
def test_qvq_dual_v2_pack_decode_matches_two_independent_v2_chains(bits):
    shift = qvq_transition_bits(bits, vector_size=2)
    generator = torch.Generator().manual_seed(32000 + shift)
    edge_a = torch.randint(0, 1 << shift, (1, 64), generator=generator, dtype=torch.int64)
    edge_b = torch.randint(0, 1 << shift, (1, 64), generator=generator, dtype=torch.int64)
    trellis_a = planar_pack_rows(edge_a.T.contiguous(), shift).T.contiguous()
    trellis_b = planar_pack_rows(edge_b.T.contiguous(), shift).T.contiguous()
    state_a = unpack_trellis_states(trellis_a, bits=bits)
    state_b = unpack_trellis_states(trellis_b, bits=bits)
    states = torch.empty((1, 128), dtype=torch.int64)
    states[:, 0::2] = state_a
    states[:, 1::2] = state_b

    trellis = pack_dual_v2_states(states, bits=bits)
    recovered = unpack_dual_v2_states(trellis, bits=bits)
    decoded = decode_trellis_tiles(trellis, bits=bits, dual_v2=True)
    direct = pgc16_decode_states(states).reshape(1, 256).float()

    assert torch.equal(recovered, states)
    torch.testing.assert_close(decoded, direct, rtol=0, atol=0)
    assert trellis.shape == (1, qvq_words_per_tile(bits, vector_size=2))


@pytest.mark.parametrize("rounding", ("block_ldlq", "yaqa"))
def test_qvq_dual_v2_quantize_pack_and_torch_linear_are_synchronized(rounding):
    bits = 2
    generator = torch.Generator().manual_seed(32200 + (rounding == "yaqa"))
    weight = torch.randn((16, 16), generator=generator)
    inputs = torch.randn((16, 16), generator=generator)
    kwargs = {}
    if rounding == "yaqa":
        kwargs = {"rounding": "yaqa", "output_hessian": torch.eye(16)}
    result = quantize_qvq_linear(
        weight,
        inputs.T @ inputs / inputs.shape[0],
        bits=bits,
        dual_v2=True,
        trellis_batch_size=1,
        **kwargs,
    )
    layer = QVQLinear(
        bits=bits,
        in_features=16,
        out_features=16,
        tensors=result.serialized_tensors(),
        dual_v2=True,
    ).eval()

    reconstructed = layer.get_inner_weight_tensor(dtype=torch.float32)
    actual = layer(inputs)
    expected = inputs @ result.weight.float().T

    reloaded = QVQLinear(
        bits=bits,
        in_features=16,
        out_features=16,
        dual_v2=True,
        register_buffers=True,
    ).eval()
    reloaded.load_state_dict(layer.state_dict(), strict=True)

    torch.testing.assert_close(reconstructed, result.inner_weight.float(), rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(reloaded(inputs), actual, rtol=0, atol=0)
    assert result.rounding == rounding


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 4, 8))
def test_qvq_dual_v2_format_round_trips(bits):
    cfg = QVQConfig(bits=bits, format="qvq_dual_v2", offload_to_disk=False)
    reloaded = QuantizeConfig.from_quant_config(cfg.to_dict())

    assert cfg.format == FORMAT.QVQ_DUAL_V2
    assert cfg.vector_size == 2
    assert cfg.trellis_window == 16
    assert cfg.quant_linear_init_kwargs()["dual_v2"] is True
    assert reloaded.to_dict() == cfg.to_dict()


def test_qvq_dual_v2_rejects_explicit_v4_banks():
    with pytest.raises(ValueError, match="requires bank_count=1"):
        QVQConfig(bits=2, format="qvq_dual_v2", bank_count=4, offload_to_disk=False)


def test_qvq_v4_four_bank_config_round_trips_and_rejects_legacy_format():
    cfg = QVQConfig(bits=2, format="qvq_v4", bank_count=4, offload_to_disk=False)
    payload = cfg.to_dict()
    assert payload["bank_count"] == 4
    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded, QVQConfig)
    assert reloaded.bank_count == 4
    with pytest.raises(ValueError, match="requires `format=qvq_v4`"):
        QVQConfig(bits=2, bank_count=4, offload_to_disk=False)
    yaqa_cfg = QVQConfig(bits=2, format="qvq_v4", bank_count=4, rounding="yaqa", offload_to_disk=False)
    assert yaqa_cfg.bank_count == 4


def test_qvq_output_alignment_is_explicit_and_round_trips_exactly():
    cfg = QVQConfig(
        bits=1.5,
        output_alignment={
            "learning_rate": 2e-5,
            "epochs": 3,
            "optimizer": "adamw",
            "weight_decay": 1e-4,
            "maximum_train_batches": 12,
            "maximum_validation_batches": 4,
            "validation_fraction": 0.25,
            "minimum_relative_improvement": 0.001,
        },
        offload_to_disk=False,
    )

    assert isinstance(cfg.output_alignment, OutputAlignConfig)
    payload = cfg.to_dict()
    assert payload["output_alignment"] == {
        "learning_rate": 2e-5,
        "epochs": 3,
        "optimizer": "adamw",
        "weight_decay": 1e-4,
        "maximum_train_batches": 12,
        "maximum_validation_batches": 4,
        "validation_fraction": 0.25,
        "minimum_relative_improvement": 0.001,
        "pristine_hessian": True,
    }
    assert QuantizeConfig.from_quant_config(payload).to_dict() == payload


def test_qvq_output_alignment_pristine_hessian_validates_and_round_trips():
    cfg = QVQConfig(
        bits=2,
        output_alignment={"pristine_hessian": False},
        offload_to_disk=False,
    )

    assert cfg.output_alignment.pristine_hessian is False
    assert cfg.to_dict()["output_alignment"]["pristine_hessian"] is False
    with pytest.raises(TypeError, match="pristine_hessian"):
        OutputAlignConfig(pristine_hessian=1)


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"optimizer": "lion"}, ValueError, "optimizer"),
        ({"optimizer": 1}, TypeError, "optimizer"),
        ({"weight_decay": -1e-4}, ValueError, "weight_decay"),
        ({"weight_decay": float("nan")}, ValueError, "weight_decay"),
    ],
)
def test_qvq_output_alignment_rejects_invalid_optimizer_controls(kwargs, exception, message):
    with pytest.raises(exception, match=message):
        QVQConfig(output_alignment=kwargs, offload_to_disk=False)


@pytest.mark.parametrize(
    ("output_alignment", "exception", "message"),
    (
        (True, TypeError, "output_alignment"),
        ({"learning_rate": 0}, ValueError, "learning_rate"),
        ({"learning_rate": float("nan")}, ValueError, "learning_rate"),
        ({"epochs": True}, ValueError, "epochs"),
        ({"maximum_train_batches": 0}, ValueError, "maximum_train_batches"),
        ({"maximum_validation_batches": 1.5}, ValueError, "maximum_validation_batches"),
        ({"validation_fraction": 1.0}, ValueError, "validation_fraction"),
        ({"validation_fraction": 0.0}, ValueError, "validation_fraction"),
        ({"validation_fraction": float("inf")}, ValueError, "validation_fraction"),
        ({"minimum_relative_improvement": -1e-3}, ValueError, "minimum_relative_improvement"),
        ({"minimum_relative_improvement": float("nan")}, ValueError, "minimum_relative_improvement"),
        ({"unknown": 1}, TypeError, "unknown"),
    ),
)
def test_qvq_output_alignment_rejects_invalid_controls(output_alignment, exception, message):
    with pytest.raises(exception, match=message):
        QVQConfig(output_alignment=output_alignment, offload_to_disk=False)


def test_qvq_config_rejects_unknown_tensor_storage_entries():
    tensor_storage = {
        "model.layers.0.self_attn.q_proj": {
            "trellis": {"shape": [16, 16], "torch_dtype": "int32"},
            "extra": {"shape": [1], "torch_dtype": "float16"},
        }
    }

    with pytest.raises(ValueError, match="unexpected tensors.*extra"):
        QVQConfig(tensor_storage=tensor_storage, offload_to_disk=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bits": 0}, "rate"),
        ({"bits": 9}, "rate"),
        ({"group_size": 128}, "group_size"),
        ({"sym": False}, "asymmetric"),
        ({"pack_dtype": torch.int16}, "pack_dtype"),
        ({"codebook": "hyb"}, "codebook"),
        ({"trellis_window": 32}, "trellis_window"),
        ({"vector_size": 4}, "vector_size"),
        ({"tile_rows": 32}, "tile_rows"),
        ({"tile_cols": 32}, "tile_cols"),
        ({"rounding": "gptq"}, "rounding"),
        ({"incoherence": "none"}, "incoherence"),
        ({"output_alignment": {}, "lm_head": True}, "decoder layers"),
    ],
)
def test_qvq_config_rejects_noncanonical_layout(kwargs, message):
    with pytest.raises(ValueError, match=message):
        QVQConfig(offload_to_disk=False, **kwargs)


def test_qvq_config_does_not_expose_gptq_activation_ordering():
    with pytest.raises(TypeError, match="desc_act"):
        QVQConfig(desc_act=True, offload_to_disk=False)

    config = QVQConfig(offload_to_disk=False)
    assert config.rounding == "yaqa"
    assert "desc_act" not in {config_field.name for config_field in fields(config)}
    assert "desc_act" not in vars(config)
    payload = config.to_dict()
    assert "desc_act" not in payload


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    (
        ({"output_channel_scale_optimization": 1}, TypeError, "must be boolean"),
        ({"module_scale_search": 1}, TypeError, "must be boolean"),
        ({"viterbi_objective": 1}, TypeError, "must be a string"),
        (
            {"viterbi_objective": "full_hessian"},
            ValueError,
            "euclidean.*hessian_diagonal",
        ),
        ({"tail_biting_candidates": 0}, ValueError, "positive integer"),
        ({"tail_biting_candidates": True}, ValueError, "positive integer"),
        ({"rounding": 1}, TypeError, "rounding.*string"),
        ({"viterbi_minimum_proxy_improvement": True}, TypeError, "real scalar"),
        (
            {"viterbi_minimum_proxy_improvement": -0.1},
            ValueError,
            "finite and nonnegative",
        ),
        (
            {"viterbi_minimum_proxy_improvement": float("inf")},
            ValueError,
            "finite and nonnegative",
        ),
        (
            {"viterbi_minimum_proxy_improvement": 0.001},
            ValueError,
            "requires.*hessian_diagonal",
        ),
    ),
)
def test_qvq_config_rejects_invalid_accuracy_upgrade_controls(kwargs, exception, message):
    with pytest.raises(exception, match=message):
        QVQConfig(offload_to_disk=False, **kwargs)


def test_qvq_dynamic_bits_allow_only_supported_rates():
    cfg = QVQConfig(
        bits=2,
        dynamic={
            r".*k_proj": {"bits": 1},
            r".*q_proj": {"bits": 2.5},
            r".*v_proj": {"bits": 8},
            r"-:.*lm_head": {},
        },
        offload_to_disk=False,
    )

    assert cfg.dynamic[r".*k_proj"]["bits"] == 1
    assert cfg.dynamic[r".*q_proj"]["bits"] == 2.5
    assert cfg.dynamic[r".*v_proj"]["bits"] == 8

    with pytest.raises(ValueError, match="rate must"):
        QVQConfig(dynamic={r".*q_proj": {"bits": 9}}, offload_to_disk=False)
    with pytest.raises(ValueError, match="half-integer"):
        QVQConfig(dynamic={r".*q_proj": {"bits": 2.25}}, offload_to_disk=False)
    with pytest.raises(ValueError, match="only supports `bits` and `yaqa_regularization` overrides"):
        QVQConfig(dynamic={r".*q_proj": {"group_size": 128}}, offload_to_disk=False)

    with pytest.raises(ValueError, match="format=qvq_v4.*W1 through W4"):
        QVQConfig(
            bits=2,
            format="qvq_v4",
            dynamic={r".*q_proj": {"bits": 4.5}},
            offload_to_disk=False,
        )


@pytest.mark.parametrize("bits", QVQ_HALF_STEP_BITS)
def test_qvq_config_accepts_every_planar_rate(bits):
    cfg = QVQConfig(bits=bits, rounding="block_ldlq", offload_to_disk=False)

    assert cfg.bits == bits
    assert cfg.format == FORMAT.QVQ
    assert cfg.pack_dtype == torch.int32


def test_qvq_protocol_compiles_to_qvq_config():
    cfg = compile_protocol_to_quantize_config(
        {
            "version": 2,
            "stages": [
                {
                    "name": "qvq_ptq",
                    "rules": [
                        {
                            "match": ["*", r"-:.*lm_head"],
                            "weight": {
                                "quantize": {
                                    "method": "qvq",
                                    "bits": 2,
                                    "codebook": "pgc16-v1",
                                    "rounding": "block_ldlq",
                                    "incoherence": "rht",
                                },
                                "export": {"format": "qvq", "variant": "pgc16-v1"},
                            },
                        }
                    ],
                }
            ],
        }
    )

    assert isinstance(cfg, QVQConfig)
    assert cfg.bits == 2
    assert cfg.dynamic == {r"-:.*lm_head": {}}
    assert cfg.method == METHOD.QVQ
    assert cfg.format == FORMAT.QVQ


def test_qvq_protocol_requires_explicit_rounding():
    payload = {
        "version": 2,
        "stages": [
            {
                "name": "qvq_ptq",
                "rules": [
                    {
                        "match": "*",
                        "weight": {
                            "quantize": {"method": "qvq", "bits": 2},
                            "export": {"format": "qvq"},
                        },
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="explicit.*rounding"):
        compile_protocol_to_quantize_config(payload)


def test_qvq_accuracy_upgrade_controls_round_trip_through_config_and_protocol():
    payload = {
        "version": 2,
        "stages": [
            {
                "name": "qvq_ptq",
                "rules": [
                    {
                        "match": "*",
                        "weight": {
                            "quantize": {
                                "method": "qvq",
                                "bits": 2,
                                "rounding": "block_ldlq",
                                "module_scale_search": True,
                                "output_channel_scale_optimization": True,
                                "viterbi_objective": "hessian_diagonal",
                                "tail_biting_candidates": 4,
                                "viterbi_minimum_proxy_improvement": 0.001,
                            },
                            "export": {
                                "format": "qvq",
                                "variant": PGC16_CODEBOOK_VERSION,
                            },
                        },
                    }
                ],
            }
        ],
    }

    cfg = compile_protocol_to_quantize_config(payload)
    serialized = cfg.to_dict()
    reloaded = QuantizeConfig.from_quant_config(serialized)

    assert cfg.module_scale_search is True
    assert cfg.output_channel_scale_optimization is True
    assert cfg.viterbi_objective == "hessian_diagonal"
    assert cfg.tail_biting_candidates == 4
    assert serialized["module_scale_search"] is True
    assert serialized["output_channel_scale_optimization"] is True
    assert serialized["viterbi_objective"] == "hessian_diagonal"
    assert serialized["tail_biting_candidates"] == 4
    assert serialized["viterbi_minimum_proxy_improvement"] == 0.001
    assert reloaded.module_scale_search is True
    assert reloaded.output_channel_scale_optimization is True
    assert reloaded.viterbi_objective == "hessian_diagonal"
    assert reloaded.tail_biting_candidates == 4
    assert cfg.viterbi_minimum_proxy_improvement == 0.001
    assert reloaded.viterbi_minimum_proxy_improvement == 0.001


def test_yaqa_rounding_round_trips_through_config_and_protocol():
    payload = {
        "version": 2,
        "stages": [
            {
                "name": "qvq_yaqa",
                "rules": [
                    {
                        "match": "*",
                        "weight": {
                            "quantize": {
                                "method": "qvq",
                                "bits": 2,
                                "rounding": "yaqa",
                                "yaqa": {"seed": 787, "regularization": 0.01, "minimum_sequences": 4096},
                                "tail_biting_candidates": 4,
                            },
                            "export": {
                                "format": "qvq",
                                "variant": PGC16_CODEBOOK_VERSION,
                            },
                        },
                    }
                ],
            }
        ],
    }

    cfg = compile_protocol_to_quantize_config(payload)
    reloaded = QuantizeConfig.from_quant_config(cfg.to_dict())

    assert cfg.rounding == "yaqa"
    assert cfg.yaqa == YaqaConfig(seed=787, regularization=0.01, minimum_sequences=4096)
    assert cfg.tail_biting_candidates == 4
    assert reloaded.rounding == "yaqa"
    assert reloaded.yaqa == YaqaConfig(seed=787, regularization=0.01, minimum_sequences=4096)
    assert reloaded.tail_biting_candidates == 4


def test_yaqa_config_defaults_to_validated_rate_damping_and_sample_floor():
    config = QVQConfig(rounding="yaqa", yaqa=YaqaConfig(), offload_to_disk=False)

    assert config.yaqa.regularization == pytest.approx(0.05)
    assert config.yaqa.minimum_sequences == YAQA_PAPER_MINIMUM_SEQUENCES == 2_000
    assert config.yaqa.batch_size == 8
    assert config.yaqa.activation_checkpointing is True
    assert config.yaqa.mps_cleanup_interval == 8
    assert config.yaqa.sequence_sort == "desc"
    assert config.yaqa.max_factor_bytes_per_pass is None
    for rate in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        assert config.yaqa.regularization_for_rate(rate) == pytest.approx(0.1)
    for rate in (4.5, 5.0, 6.0, 7.0, 8.0):
        assert config.yaqa.regularization_for_rate(rate) == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    (
        ({"batch_size": 0}, ValueError, "batch_size"),
        ({"batch_size": True}, ValueError, "batch_size"),
        ({"activation_checkpointing": 1}, TypeError, "activation_checkpointing"),
        ({"mps_cleanup_interval": 0}, ValueError, "mps_cleanup_interval"),
        ({"mps_cleanup_interval": True}, ValueError, "mps_cleanup_interval"),
        ({"sequence_sort": 1}, TypeError, "sequence_sort"),
        ({"sequence_sort": "shuffle"}, ValueError, "sequence_sort"),
        ({"max_factor_bytes_per_pass": 0}, ValueError, "max_factor_bytes_per_pass"),
        ({"max_factor_bytes_per_pass": True}, ValueError, "max_factor_bytes_per_pass"),
    ),
)
def test_yaqa_config_rejects_invalid_collection_controls(kwargs, exception, message):
    with pytest.raises(exception, match=message):
        YaqaConfig(**kwargs)


def test_yaqa_config_supports_exact_rate_regularization_overrides():
    config = YaqaConfig(regularization_by_rate=((2.5, 5e-4), (1.0, 1e-2)))
    assert config.regularization_by_rate == ((1.0, 0.01), (2.5, 0.0005))
    assert config.regularization_for_rate(1.0) == pytest.approx(0.01)
    assert config.regularization_for_rate(2.5) == pytest.approx(0.0005)
    assert config.regularization_for_rate(2.0) == pytest.approx(YAQA_DEFAULT_REGULARIZATION)


def test_yaqa_rate_regularization_overrides_round_trip_through_qvq_config():
    config = QVQConfig(
        rounding="yaqa",
        yaqa=YaqaConfig(regularization_by_rate=((2.5, 5e-4), (1.0, 1e-2))),
        offload_to_disk=False,
    )
    reloaded = QuantizeConfig.from_quant_config(config.to_dict())
    assert reloaded.yaqa.regularization_by_rate == ((1.0, 0.01), (2.5, 0.0005))
    assert reloaded.yaqa.regularization_for_rate(1.0) == pytest.approx(0.01)


@pytest.mark.parametrize(
    "overrides",
    [
        ((1.0, 1e-3), (1.0, 2e-3)),
        ((0.0, 1e-3),),
        ((1.0, -1e-3),),
        ((1.0,),),
    ],
)
def test_yaqa_config_rejects_invalid_rate_regularization_overrides(overrides):
    with pytest.raises((TypeError, ValueError)):
        YaqaConfig(regularization_by_rate=overrides)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {"rounding": "yaqa", "output_channel_scale_optimization": True},
            "output-channel",
        ),
        ({"rounding": "yaqa", "module_scale_search": True}, "module-scale"),
        ({"rounding": "yaqa", "viterbi_objective": "hessian_diagonal"}, "euclidean"),
    ),
)
def test_yaqa_config_rejects_incompatible_local_accuracy_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        QVQConfig(offload_to_disk=False, **kwargs)


def test_qvq_config_rejects_propagated_selection_with_yaqa_rounding():
    with pytest.raises(ValueError, match="propagated bank selection.*block_ldlq"):
        QVQConfig(
            bits=2,
            format="qvq_v4",
            vector_size=4,
            bank_count=4,
            propagated_bank_selection=True,
            rounding="yaqa",
            offload_to_disk=False,
        )


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    (
        ({"rounding": "yaqa", "yaqa": {"seed": True}}, TypeError, "YaqaConfig.*seed"),
        ({"rounding": "yaqa", "yaqa": {"seed": 1.5}}, TypeError, "YaqaConfig.*seed"),
        ({"rounding": "yaqa", "yaqa": {"regularization": True}}, TypeError, "YaqaConfig.*regularization"),
        ({"rounding": "yaqa", "yaqa": {"regularization": float("nan")}}, ValueError, "YaqaConfig.*regularization"),
        ({"rounding": "yaqa", "yaqa": {"regularization": -1e-4}}, ValueError, "YaqaConfig.*regularization"),
        ({"rounding": "yaqa", "yaqa": {"minimum_sequences": True}}, ValueError, "YaqaConfig.*minimum_sequences"),
        ({"rounding": "yaqa", "yaqa": {"minimum_sequences": 0}}, ValueError, "YaqaConfig.*minimum_sequences"),
        ({"rounding": "yaqa", "lm_head": True}, ValueError, "language-model head"),
    ),
)
def test_yaqa_config_rejects_invalid_lifecycle_controls(kwargs, exception, message):
    with pytest.raises(exception, match=message):
        QVQConfig(offload_to_disk=False, **kwargs)






@pytest.mark.parametrize(
    ("input_dtype", "device_type", "expected"),
    (
        (torch.float16, "cpu", torch.float32),
        (torch.bfloat16, "cpu", torch.float32),
        (torch.float16, "cuda", torch.float16),
        (torch.bfloat16, "cuda", torch.float16),
        (torch.float32, "cuda", torch.float32),
        (torch.float16, "mps", torch.float16),
    ),
)
def test_qvq_compute_dtype_selects_the_preferred_first_pass_dtype(input_dtype, device_type, expected):
    assert _qvq_compute_dtype(input_dtype, device_type) == expected






def test_qvq_owns_a_quantization_lifecycle_instead_of_an_inference_only_guard():
    assert METHOD.QVQ not in QUANTIZE_BLACK_LIST


def test_bitshift_transition_matches_paper_definition():
    states = torch.tensor([0b00, 0b01, 0b10, 0b11])
    edges = torch.tensor([0, 0, 1, 0])

    actual = bitshift_next_state(
        states,
        edges,
        bits=1,
        vector_size=1,
        trellis_window=2,
    )

    assert torch.equal(actual, torch.tensor([0b00, 0b10, 0b01, 0b10]))


@pytest.mark.parametrize("bits", QVQ_HALF_STEP_BITS)
def test_bitshift_transition_uses_every_exact_half_step_width(bits):
    transition_bits = qvq_transition_bits(bits)
    states = torch.tensor([0xA5A5, 0x5A5A])
    edges = torch.tensor([0, (1 << transition_bits) - 1])

    actual = bitshift_next_state(
        states,
        edges,
        bits=bits,
        vector_size=2,
        trellis_window=16,
    )
    expected = ((states << transition_bits) & 0xFFFF) | edges

    assert torch.equal(actual, expected)


def test_viterbi_reproduces_paper_l2_worked_example():
    sequence = torch.tensor([[0.54], [0.03], [0.72], [0.19], [0.26], [0.89]])
    codebook = torch.tensor([[0.5], [0.1], [0.8], [0.3]])

    result = viterbi_quantize(sequence, codebook, bits=1)

    assert torch.equal(result.states, torch.tensor([0, 1, 2, 1, 3, 2]))
    assert torch.equal(result.values, torch.tensor([[0.5], [0.1], [0.8], [0.1], [0.3], [0.8]]))
    assert torch.allclose(result.squared_error, (sequence - result.values).square().sum())


def test_hyb_decode_hashes_lut_index_and_flips_second_component():
    lut = torch.stack(
        (
            torch.arange(8, dtype=torch.float32),
            torch.arange(8, dtype=torch.float32) + 0.5,
        ),
        dim=-1,
    )
    states = torch.tensor([0, 64, 128, 255], dtype=torch.int64)

    decoded = hyb_decode_states(states, lut, trellis_window=8, lut_bits=3)

    assert torch.equal(
        decoded,
        torch.tensor(
            [
                [0.0, 0.5],
                [1.0, 1.5],
                [4.0, 4.5],
                [7.0, -7.5],
            ]
        ),
    )
    assert torch.equal(hyb_codebook(lut, trellis_window=8, lut_bits=3)[states], decoded)


def test_hyb_lut_fit_is_deterministic_finite_and_rng_isolated():
    torch.manual_seed(1122)
    expected_next_random = torch.randn(3)
    torch.manual_seed(1122)

    first = fit_hyb_lut(lut_bits=3, sample_count=256, iterations=3, seed=7)
    second = fit_hyb_lut(lut_bits=3, sample_count=256, iterations=3, seed=7)

    assert first.shape == (8, 2)
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    assert torch.all(first[:, 1] >= 0)
    assert torch.equal(first, second)
    assert torch.equal(torch.randn(3), expected_next_random)
    assert canonical_hyb_lut() is canonical_hyb_lut()


def test_pgc16_has_frozen_symmetric_levels_and_exactly_65536_unique_vectors():
    levels = canonical_pgc16_levels()
    states = torch.arange(PGC16_STATE_COUNT, dtype=torch.int64)
    mixed = pgc16_mix_states(states)
    codebook = pgc16_codebook(dtype=torch.float16)

    assert levels.shape == (256,)
    assert levels.dtype == torch.float16
    assert torch.isfinite(levels).all()
    assert torch.equal(levels, -levels.flip(0))
    assert torch.unique(levels).numel() == 256
    assert torch.unique(mixed).numel() == PGC16_STATE_COUNT
    assert torch.unique(codebook, dim=0).shape[0] == PGC16_STATE_COUNT
    assert torch.equal(codebook, pgc16_decode_states(states))


def test_pgc16_mixer_matches_versioned_integer_specification():
    states = torch.tensor([0, 1, 0x1234, 0xFFFF], dtype=torch.int64)
    first = states ^ (states >> 8)
    second = (first * 40503 + 17011) & 0xFFFF
    expected = second ^ (second >> 7)

    assert torch.equal(pgc16_mix_states(states), expected)
    decoded = pgc16_decode_states(states)
    levels = canonical_pgc16_levels()
    assert torch.equal(decoded[:, 0], levels[expected >> 8])
    assert torch.equal(decoded[:, 1], levels[expected & 0xFF])


def test_pgc16_v4_four_bank_reference_decoder_preserves_bank_zero_and_changes_geometry():
    states = torch.arange(64, dtype=torch.int64).reshape(4, 16)
    selectors = torch.arange(PGC16_V4_BANK_COUNT, dtype=torch.uint8)
    banked = pgc16_decode_states_v4_banked(states, selectors, bits=2)
    canonical = pgc16_decode_states_v4(states)

    torch.testing.assert_close(banked[0], canonical[0], rtol=0, atol=0)
    assert len(PGC16_V4_BANK_XOR_MASKS) == PGC16_V4_BANK_COUNT
    assert not torch.equal(banked[1], banked[0])
    assert all(
        torch.unique(pgc16_codebook_v4_bank(bank, bits=2), dim=0).shape[0] == PGC16_STATE_COUNT
        for bank in range(4)
    )


@pytest.mark.parametrize(
    ("selectors", "exception", "message"),
    (
        (torch.tensor([[4]], dtype=torch.uint8), ValueError, "selectors"),
        (torch.tensor([[True]]), TypeError, "integer"),
    ),
)
def test_pgc16_v4_banked_decoder_rejects_invalid_selectors(selectors, exception, message):
    states = torch.zeros((1, 1), dtype=torch.int64)
    with pytest.raises(exception, match=message):
        pgc16_decode_states_v4_banked(states, selectors, bits=2)


def test_block_ldlq_v4_banked_reference_returns_valid_selector_and_matches_selected_tile():
    generator = torch.Generator().manual_seed(20260813)
    weight = torch.randn((16, 16), generator=generator)
    hessian = torch.eye(16)
    codebooks = tuple(pgc16_codebook_v4_bank(bank, bits=2) for bank in range(4))
    selected, trellis, bank_ids = block_ldlq_inner_banked(
        weight,
        hessian,
        codebooks,
        bits=2,
        trellis_batch_size=1,
    )
    assert trellis.shape == (1, 16 * 16 // 4)
    assert bank_ids.shape == (1,)
    assert bank_ids.dtype == torch.uint8
    assert int(bank_ids.item()) in range(4)
    decoded = reconstruct_qvq_inner_weight(
        pack_trellis_states(trellis, bits=2, vector_size=4),
        bits=2,
        in_features=16,
        out_features=16,
        vector_size=4,
        bank_ids=bank_ids,
    )
    torch.testing.assert_close(decoded, selected, rtol=0, atol=0)


def test_block_ldlq_v4_banked_reference_exposes_independent_bank_zero_oracle():
    generator = torch.Generator().manual_seed(20260815)
    weight = torch.randn((16, 16), generator=generator)
    hessian_source = torch.randn((16, 16), generator=generator)
    hessian = hessian_source @ hessian_source.T + torch.eye(16) * 0.25
    codebooks = tuple(pgc16_codebook_v4_bank(bank, bits=2) for bank in range(4))
    _, _, _, oracle_weight, oracle_states = block_ldlq_inner_banked(
        weight,
        hessian,
        codebooks,
        bits=2,
        trellis_batch_size=1,
        return_bank0_oracle=True,
    )
    expected_weight, expected_states = block_ldlq_inner(
        weight,
        hessian,
        codebooks[0],
        bits=2,
        trellis_batch_size=1,
    )
    torch.testing.assert_close(oracle_weight, expected_weight, rtol=0, atol=0)
    assert torch.equal(oracle_states, expected_states)


def test_block_ldlq_v4_banked_candidates_preserve_supplied_baseline_on_cpu():
    generator = torch.Generator().manual_seed(20260816)
    weight = torch.randn((16, 16), generator=generator)
    hessian = torch.eye(16)
    codebooks = tuple(pgc16_codebook_v4_bank(bank, bits=2) for bank in range(4))
    baseline_weight, baseline_states = block_ldlq_inner(
        weight,
        hessian,
        codebooks[0],
        bits=2,
        trellis_batch_size=1,
    )
    candidates, states = qvq_module.block_ldlq_inner_banked_candidates(
        weight,
        hessian,
        codebooks,
        bits=2,
        trellis_batch_size=1,
        baseline_weight=baseline_weight,
        baseline_states=baseline_states,
    )
    torch.testing.assert_close(candidates[0], baseline_weight, rtol=0, atol=0)
    assert torch.equal(states[0], baseline_states)


def test_block_ldlq_v4_banked_reference_preserves_multi_output_tile_order():
    generator = torch.Generator().manual_seed(20260814)
    weight = torch.randn((16, 32), generator=generator)
    hessian = torch.eye(16)
    codebooks = tuple(pgc16_codebook_v4_bank(bank, bits=2) for bank in range(4))
    selected, trellis, bank_ids = block_ldlq_inner_banked(
        weight,
        hessian,
        codebooks,
        bits=2,
        trellis_batch_size=1,
    )
    assert trellis.shape == (2, 16 * 16 // 4)
    assert bank_ids.shape == (2,)
    decoded = reconstruct_qvq_inner_weight(
        pack_trellis_states(trellis, bits=2, vector_size=4),
        bits=2,
        in_features=16,
        out_features=32,
        vector_size=4,
        bank_ids=bank_ids,
    )
    torch.testing.assert_close(decoded, selected, rtol=0, atol=0)
    packed_bank_ids = qvq_module.pack_qvq_bank_ids(bank_ids)
    decoded_packed = reconstruct_qvq_inner_weight(
        pack_trellis_states(trellis, bits=2, vector_size=4),
        bits=2,
        in_features=16,
        out_features=32,
        vector_size=4,
        bank_ids=packed_bank_ids,
    )
    torch.testing.assert_close(decoded_packed, selected, rtol=0, atol=0)


def test_qvq_v4_bank_cache_is_rate_keyed_and_thread_safe_contract_is_immutable():
    first = qvq_module._canonical_qvq_v4_banks(
        device=torch.device("cpu"), bits=2, codebook_version=PGC16_CODEBOOK_VERSION, dtype=torch.float32
    )
    second = qvq_module._canonical_qvq_v4_banks(
        device=torch.device("cpu"), bits=2, codebook_version=PGC16_CODEBOOK_VERSION, dtype=torch.float32
    )
    other_rate = qvq_module._canonical_qvq_v4_banks(
        device=torch.device("cpu"), bits=2.5, codebook_version=PGC16_CODEBOOK_VERSION, dtype=torch.float32
    )
    first_stack = qvq_module._canonical_qvq_v4_bank_stack(
        device=torch.device("cpu"), bits=2, codebook_version=PGC16_CODEBOOK_VERSION, dtype=torch.float32
    )
    second_stack = qvq_module._canonical_qvq_v4_bank_stack(
        device=torch.device("cpu"), bits=2, codebook_version=PGC16_CODEBOOK_VERSION, dtype=torch.float32
    )
    assert first is second
    assert PGC16_V4_BANK_XOR_MASKS_BY_TRANSITION_BITS[4] != (
        PGC16_V4_BANK_XOR_MASKS_BY_TRANSITION_BITS[8]
    )
    assert first[0].data_ptr() == second[0].data_ptr()
    assert first_stack is second_stack
    assert first_stack.data_ptr() == first[0].data_ptr()
    assert tuple(first_stack.shape) == (4, 1 << 16, 4)
    assert other_rate is not first
    assert all(not bank.requires_grad for bank in first + other_rate)


@pytest.mark.parametrize(
    ("states", "exception", "message"),
    (
        (torch.tensor([-1]), ValueError, "states"),
        (torch.tensor([65536]), ValueError, "states"),
        (torch.tensor([1.5]), TypeError, "integer"),
        (torch.tensor([True]), TypeError, "integer"),
    ),
)
def test_pgc16_rejects_invalid_states(states, exception, message):
    with pytest.raises(exception, match=message):
        pgc16_decode_states(states)


































@pytest.mark.parametrize("bits", (True, 1.25, 0, 9))
def test_pgc16_scale_factor_rejects_invalid_rates(bits):
    with pytest.raises((TypeError, ValueError), match="rate|half-integer"):
        qvq_module.pgc16_scale_factor(bits)


def test_pgc16_scale_factor_covers_every_qvq_rate():
    assert tuple(qvq_module.pgc16_scale_factor(bits) for bits in QVQ_HALF_STEP_BITS) == PGC16_SCALE_FACTORS


def test_pgc16_normalization_rms_is_exactly_owned_by_the_canonical_fp32_codebook():
    expected = pgc16_codebook(dtype=torch.float32).square().mean().sqrt().item()

    assert PGC16_NORMALIZATION_RMS == expected
















def test_qvq_reference_quantizer_uses_rate_aware_backend_batches():
    assert [default_qvq_trellis_batch_size(bits, "mps") for bits in QVQ_HALF_STEP_BITS] == [
        96,
        96,
        32,
        32,
        32,
        32,
        64,
        64,
        64,
        64,
        64,
        64,
        256,
        256,
        512,
    ]
    assert [default_qvq_trellis_batch_size(bits, "cuda") for bits in QVQ_HALF_STEP_BITS] == [
        496,
        496,
        512,
        3968,
        3968,
        3968,
        3968,
        3968,
        3968,
        3968,
        3968,
        3968,
        3968,
        3968,
        2976,
    ]
    assert default_qvq_trellis_batch_size(8, "cpu") == 16
    assert [
        default_qvq_trellis_batch_size(bits, "mps", trellis_window=18)
        for bits in (1, 1.5, 2, 2.5)
    ] == [24, 24, 8, 8]
    assert [
        default_qvq_trellis_batch_size(bits, "cuda", trellis_window=18)
        for bits in (1, 1.5, 2, 2.5)
    ] == [124, 124, 128, 992]
    assert default_qvq_trellis_batch_size(1, "cpu", trellis_window=18) == 4
    with pytest.raises(ValueError, match="rate"):
        default_qvq_trellis_batch_size(0, "mps")
    with pytest.raises(TypeError, match="integer"):
        default_qvq_trellis_batch_size(1, "mps", trellis_window=True)
    with pytest.raises(ValueError, match="windows 16 and 18"):
        default_qvq_trellis_batch_size(1, "mps", trellis_window=17)
    with pytest.raises(ValueError, match="W1 through W2.5"):
        default_qvq_trellis_batch_size(3, "mps", trellis_window=18)


def test_yaqa_segmented_batch_policy_fills_cuda_without_overriding_explicit_batches():
    policy = qvq_module._yaqa_segmented_batch_size

    assert policy(128, 16, 1, apple_host_feedback=False, cuda_feedback=True) == 32
    assert policy(128, 16, 2.5, apple_host_feedback=False, cuda_feedback=True) == 64
    assert policy(31, 16, 2.5, apple_host_feedback=False, cuda_feedback=True) == 31
    assert policy(128, 8, 2.5, apple_host_feedback=False, cuda_feedback=True) == 8
    assert policy(128, 96, 2.5, apple_host_feedback=False, cuda_feedback=True) == 96
    assert policy(128, 16, 2.5, apple_host_feedback=False, cuda_feedback=False) == 16
    assert policy(128, 16, 2.5, apple_host_feedback=True, cuda_feedback=False) == 128

    canonical_policy = qvq_module._yaqa_viterbi_batch_size
    assert canonical_policy(128, 16, 1, cuda_feedback=True) == 32
    assert canonical_policy(128, 16, 1.5, cuda_feedback=True) == 32
    assert canonical_policy(128, 16, 2, cuda_feedback=True) == 128
    assert canonical_policy(23, 16, 2.5, cuda_feedback=True) == 23
    assert canonical_policy(128, 8, 2.5, cuda_feedback=True) == 8
    assert canonical_policy(128, 16, 2.5, cuda_feedback=False) == 16


def test_qvq_l18_implicit_batch_policy_reaches_quantizer_without_changing_math(monkeypatch):
    observed = []
    original_policy = qvq_module.default_qvq_trellis_batch_size

    def policy(bits, device, *, trellis_window=16):
        observed.append((bits, torch.device(device).type, trellis_window))
        return original_policy(bits, device, trellis_window=trellis_window)

    monkeypatch.setattr(qvq_module, "default_qvq_trellis_batch_size", policy)
    generator = torch.Generator().manual_seed(18241)
    weight = torch.randn((16, 32), generator=generator)
    hessian = torch.eye(32)
    implicit = quantize_qvq_linear(
        weight,
        hessian,
        bits=2,
        vector_size=4,
        trellis_window=18,
    )
    explicit = quantize_qvq_linear(
        weight,
        hessian,
        bits=2,
        vector_size=4,
        trellis_window=18,
        trellis_batch_size=1,
    )

    assert observed == [(2, "cpu", 18)]
    assert torch.equal(implicit.trellis, explicit.trellis)
    torch.testing.assert_close(implicit.weight, explicit.weight, rtol=0, atol=0)


def test_qvq_cuda_capability_probe_is_device_specific(monkeypatch):
    observed = []
    capabilities = {0: (7, 5), 1: (8, 0)}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def capability(device):
        observed.append(device)
        return capabilities[device.index]

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)

    assert qvq_cuda_module.qvq_cuda_device_supported("cpu") is False
    assert qvq_cuda_module.qvq_cuda_device_supported("cuda:0") is False
    assert qvq_cuda_module.qvq_cuda_device_supported("cuda:1") is True
    assert observed == [torch.device("cuda:0"), torch.device("cuda:1")]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", QVQ_HALF_STEP_BITS)
@pytest.mark.parametrize("seed_offset", range(3))
def test_qvq_cuda_rate_aware_batches_preserve_trellis_paths(bits, seed_offset):
    batch_size = default_qvq_trellis_batch_size(bits, "cuda")
    generator = torch.Generator().manual_seed(20260820 + qvq_transition_bits(bits) * 10 + seed_offset)
    sequences = torch.randn((batch_size, 128, 2), generator=generator, dtype=torch.float32, device="cpu").cuda()
    codebook = pgc16_codebook(device="cuda", dtype=torch.float32)

    scalar = tail_biting_viterbi_quantize(sequences[:1], codebook, bits=bits)
    batched = tail_biting_viterbi_quantize(sequences, codebook, bits=bits)
    torch.cuda.synchronize()

    assert torch.equal(batched.states[:1], scalar.states)
    assert torch.equal(batched.values[:1], scalar.values)
    torch.testing.assert_close(batched.squared_error[:1], scalar.squared_error, atol=2e-4, rtol=2e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", (1, 1.5))
def test_qvq_cuda_low_rate_default_batch_matches_small_batch_for_every_tile(bits):
    batch_size = default_qvq_trellis_batch_size(bits, "cuda")
    generator = torch.Generator().manual_seed(20260812 + int(bits * 2))
    sequences = torch.randn((batch_size, 128, 2), generator=generator, dtype=torch.float32).cuda()
    codebook = pgc16_codebook(device="cuda", dtype=torch.float32)

    expected_chunks = [
        tail_biting_viterbi_quantize(chunk, codebook, bits=bits)
        for chunk in sequences.split(16)
    ]
    actual = tail_biting_viterbi_quantize(sequences, codebook, bits=bits)
    torch.cuda.synchronize()

    assert torch.equal(actual.states, torch.cat([chunk.states for chunk in expected_chunks]))
    assert torch.equal(actual.values, torch.cat([chunk.values for chunk in expected_chunks]))
    assert torch.equal(actual.squared_error, torch.cat([chunk.squared_error for chunk in expected_chunks]))


def _noncontiguous_last_dimension(values: torch.Tensor) -> torch.Tensor:
    storage = torch.empty(
        (*values.shape[:-1], values.shape[-1] * 2),
        dtype=values.dtype,
        device=values.device,
    )
    view = storage[..., ::2]
    view.copy_(values)
    assert not view.is_contiguous()
    return view


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("constrained", (False, True))
@pytest.mark.parametrize("seed_offset", range(3))
def test_qvq_cuda_viterbi_matches_same_device_eager_oracle(bits, constrained, seed_offset):
    generator = torch.Generator().manual_seed(20260850 + bits * 10 + seed_offset)
    sequences = torch.randn((3, 9, 2), generator=generator, dtype=torch.float32).cuda()
    codebook = torch.randn((1 << 16, 2), generator=generator, dtype=torch.float32).cuda()
    overlap_bits = 16 - bits * 2
    overlap = None
    if constrained:
        overlap = torch.randint(
            0,
            1 << overlap_bits if overlap_bits else 1,
            (sequences.shape[0],),
            generator=generator,
            dtype=torch.int64,
        ).cuda()

    expected = batched_viterbi_quantize(
        _noncontiguous_last_dimension(sequences),
        _noncontiguous_last_dimension(codebook),
        bits=bits,
        overlap=overlap,
    )
    actual_runs = [batched_viterbi_quantize(sequences, codebook, bits=bits, overlap=overlap) for _ in range(10)]
    torch.cuda.synchronize()

    for actual in actual_runs:
        assert torch.equal(actual.states, expected.states)
        assert torch.equal(actual.values, expected.values)
        assert torch.equal(actual.squared_error, expected.squared_error)
    assert all(torch.equal(actual_runs[0].states, actual.states) for actual in actual_runs[1:])
    assert all(torch.equal(actual_runs[0].squared_error, actual.squared_error) for actual in actual_runs[1:])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("seed_offset", range(3))
def test_qvq_cuda_tail_biting_matches_explicit_same_device_eager_oracle(bits, seed_offset):
    # W8/seed 1 is a regression for a one-ulp emission contraction that used
    # to change two states despite agreeing on the other production seeds.
    generator = torch.Generator().manual_seed(20260900 + bits * 10 + seed_offset)
    sequences = torch.randn((2, 128, 2), generator=generator, dtype=torch.float32).cuda()
    codebook = pgc16_codebook(device="cuda", dtype=torch.float32)
    shift = bits * 2
    overlap_bits = 16 - shift

    if overlap_bits:
        midpoint = sequences.shape[1] // 2
        rotated = torch.roll(sequences, shifts=midpoint, dims=1)
        provisional = batched_viterbi_quantize(
            _noncontiguous_last_dimension(rotated),
            _noncontiguous_last_dimension(codebook),
            bits=bits,
        )
        overlap = provisional.states[:, midpoint - 1] & ((1 << overlap_bits) - 1)
        expected = batched_viterbi_quantize(
            _noncontiguous_last_dimension(sequences),
            _noncontiguous_last_dimension(codebook),
            bits=bits,
            overlap=overlap,
        )
    else:
        expected = batched_viterbi_quantize(
            _noncontiguous_last_dimension(sequences),
            _noncontiguous_last_dimension(codebook),
            bits=bits,
        )
    actual_runs = [tail_biting_viterbi_quantize(sequences, codebook, bits=bits) for _ in range(10)]
    torch.cuda.synchronize()

    for actual in actual_runs:
        assert torch.equal(actual.states, expected.states)
        assert torch.equal(actual.values, expected.values)
        # The native reduction and eager oracle may sum the same FP32
        # emissions in a different order; the packed path and values remain
        # bit-exact while the scalar loss allows one tight ULP.
        torch.testing.assert_close(actual.squared_error, expected.squared_error, rtol=2e-5, atol=2e-5)
    assert all(torch.equal(actual_runs[0].states, actual.states) for actual in actual_runs[1:])
    assert all(torch.equal(actual_runs[0].squared_error, actual.squared_error) for actual in actual_runs[1:])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qvq_cuda_wide_tail_biting_candidates_use_contiguous_overlap_columns():
    generator = torch.Generator().manual_seed(20261108)
    sequences = torch.randn((2, 128, 2), generator=generator, dtype=torch.float32).cuda()
    codebook = pgc16_codebook(device="cuda", dtype=torch.float32)

    baseline = tail_biting_viterbi_quantize(sequences, codebook, bits=2, candidate_count=1)
    widened_runs = [tail_biting_viterbi_quantize(sequences, codebook, bits=2, candidate_count=4) for _ in range(2)]
    torch.cuda.synchronize()

    for widened in widened_runs:
        assert torch.all(widened.squared_error <= baseline.squared_error)
        assert torch.equal(widened.states[:, 1:] >> 4, widened.states[:, :-1] & 4095)
        assert torch.equal(widened.states[:, :1] >> 4, widened.states[:, -1:] & 4095)
    assert torch.equal(widened_runs[0].states, widened_runs[1].states)
    assert torch.equal(widened_runs[0].values, widened_runs[1].values)
    assert torch.equal(widened_runs[0].squared_error, widened_runs[1].squared_error)


def test_uniform_reference_is_full_capacity_but_not_a_loadable_config():
    codebook = uniform_codebook()

    assert codebook.shape == (PGC16_STATE_COUNT, 2)
    assert torch.unique(codebook, dim=0).shape[0] == PGC16_STATE_COUNT
    with pytest.raises(ValueError, match="codebook"):
        QVQConfig(codebook="uniform-reference", offload_to_disk=False)


@pytest.mark.parametrize("bits", [1, 2])
def test_viterbi_matches_exhaustive_path_reference(bits):
    torch.manual_seed(20260811 + bits)
    trellis_window = bits + 2
    state_count = 1 << trellis_window
    sequence = torch.randn(3, 1)
    codebook = torch.randn(state_count, 1)

    result = viterbi_quantize(sequence, codebook, bits=bits)

    state_mask = state_count - 1
    edge_mask = (1 << bits) - 1
    best_error = None
    best_path = None
    for candidate in itertools.product(range(state_count), repeat=sequence.shape[0]):
        transitions_are_valid = all(
            next_state == (((state << bits) & state_mask) | (next_state & edge_mask))
            for state, next_state in itertools.pairwise(candidate)
        )
        if not transitions_are_valid:
            continue
        candidate_values = codebook[torch.tensor(candidate)]
        candidate_error = (sequence - candidate_values).square().sum().item()
        if best_error is None or candidate_error < best_error:
            best_error = candidate_error
            best_path = candidate

    assert best_error is not None
    assert best_path is not None
    assert result.states.tolist() == list(best_path)
    assert result.squared_error.item() == pytest.approx(best_error, abs=1e-6)


@pytest.mark.parametrize("bits", [1, 2])
def test_weighted_viterbi_matches_exhaustive_path_reference(bits):
    generator = torch.Generator().manual_seed(20260821 + bits)
    trellis_window = bits + 2
    state_count = 1 << trellis_window
    sequence = torch.randn((3, 1), generator=generator)
    codebook = torch.randn((state_count, 1), generator=generator)
    step_weights = torch.tensor([7.0, 0.125, 3.0])

    result = viterbi_quantize(sequence, codebook, bits=bits, step_weights=step_weights)

    state_mask = state_count - 1
    edge_mask = (1 << bits) - 1
    candidates = []
    for candidate in itertools.product(range(state_count), repeat=sequence.shape[0]):
        if not all(
            next_state == (((state << bits) & state_mask) | (next_state & edge_mask))
            for state, next_state in itertools.pairwise(candidate)
        ):
            continue
        candidate_values = codebook[torch.tensor(candidate)]
        candidate_error = ((sequence - candidate_values).square().squeeze(1) * step_weights).sum().item()
        candidates.append((candidate_error, candidate))
    expected_error, expected_path = min(candidates)

    assert result.states.tolist() == list(expected_path)
    assert result.squared_error.item() == pytest.approx(expected_error, abs=1e-6)


def test_weighted_viterbi_unit_weights_are_an_exact_disabled_control():
    generator = torch.Generator().manual_seed(20260824)
    sequences = torch.randn((3, 7, 2), generator=generator)
    codebook = torch.randn((64, 2), generator=generator)
    baseline = batched_viterbi_quantize(sequences, codebook, bits=2)
    weighted = batched_viterbi_quantize(
        sequences,
        codebook,
        bits=2,
        step_weights=torch.ones(sequences.shape[:2]),
    )

    assert torch.equal(weighted.states, baseline.states)
    assert torch.equal(weighted.values, baseline.values)
    assert torch.equal(weighted.squared_error, baseline.squared_error)


def test_scalar_weighted_viterbi_rejects_non_vector_step_weights():
    with pytest.raises(ValueError, match="shape.*steps"):
        viterbi_quantize(
            torch.zeros((3, 1)),
            torch.zeros((4, 1)),
            bits=1,
            step_weights=torch.ones((1, 3)),
        )


def test_viterbi_float64_reference_preserves_near_tie_ordering():
    codebook = torch.tensor(
        [[1.0e8, 0.0], [1.0e8 + 1.0, 0.0], [2.0e8, 0.0], [3.0e8, 0.0]], dtype=torch.float64
    )
    sequences = torch.tensor([[[1.0e8 + 0.75, 0.0]]], dtype=torch.float64)

    result = batched_viterbi_quantize(sequences, codebook, bits=1.0)

    assert result.states.tolist() == [[1]]
    torch.testing.assert_close(result.squared_error, torch.tensor([0.0625], dtype=torch.float64), rtol=0, atol=0)


def test_viterbi_rejects_finite_values_that_overflow_distance_arithmetic():
    sequences = torch.tensor([[[1.0e20, -1.0e20]]], dtype=torch.float32)
    codebook = torch.tensor(
        [[1.0e20, -1.0e20], [1.1e20, -1.1e20], [2.0e20, -2.0e20], [3.0e20, -3.0e20]],
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="squared-distance arithmetic"):
        batched_viterbi_quantize(sequences, codebook, bits=1.0)


def test_viterbi_rejects_finite_values_that_overflow_accumulated_distance():
    sequences = torch.full((1, 64, 2), 1e18)
    codebook = torch.zeros((4, 2))

    with pytest.raises(ValueError, match="squared-distance arithmetic"):
        batched_viterbi_quantize(sequences, codebook, bits=1.0)


@pytest.mark.parametrize(
    ("overlap", "exception"),
    ((torch.zeros(2, dtype=torch.int64), ValueError), (torch.zeros(1, dtype=torch.float32), TypeError)),
)
def test_viterbi_rejects_invalid_zero_overlap_metadata(overlap, exception):
    sequences = torch.zeros((1, 1, 2), dtype=torch.float32)
    codebook = torch.zeros((4, 2), dtype=torch.float32)

    with pytest.raises(exception, match="overlap"):
        batched_viterbi_quantize(sequences, codebook, bits=1.0, overlap=overlap)


@pytest.mark.parametrize(
    ("step_weights", "exception", "message"),
    (
        (torch.ones(2), ValueError, "shape"),
        (torch.ones((1, 3), dtype=torch.int64), TypeError, "floating-point"),
        (torch.tensor([[1.0, -1.0, 1.0]]), ValueError, "nonnegative"),
        (torch.tensor([[1.0, float("nan"), 1.0]]), ValueError, "finite"),
    ),
)
def test_weighted_viterbi_rejects_invalid_step_weights(step_weights, exception, message):
    with pytest.raises(exception, match=message):
        batched_viterbi_quantize(
            torch.zeros((1, 3, 1)),
            torch.zeros((4, 1)),
            bits=1,
            step_weights=step_weights,
        )


def test_viterbi_ties_are_deterministic_and_noncontiguous_inputs_work():
    sequence_storage = torch.zeros(4, 2)
    codebook_storage = torch.zeros(8, 2)
    sequence = sequence_storage[:, ::2]
    codebook = codebook_storage[:, ::2]
    assert not sequence.is_contiguous()
    assert not codebook.is_contiguous()

    first = viterbi_quantize(sequence, codebook, bits=1)
    second = viterbi_quantize(sequence, codebook, bits=1)

    assert torch.equal(first.states, second.states)
    assert first.states.tolist() == [0, 0, 0, 0]
    assert first.squared_error.item() == 0.0


def test_batched_viterbi_matches_scalar_oracle_and_compresses_traceback():
    generator = torch.Generator().manual_seed(20260812)
    sequences = torch.randn((3, 7, 2), generator=generator)
    codebook = torch.randn((64, 2), generator=generator)

    batched = batched_viterbi_quantize(sequences, codebook, bits=2)
    scalar = [viterbi_quantize(sequence, codebook, bits=2) for sequence in sequences]

    assert torch.equal(batched.states, torch.stack([result.states for result in scalar]))
    assert torch.equal(batched.values, torch.stack([result.values for result in scalar]))
    torch.testing.assert_close(
        batched.squared_error,
        torch.stack([result.squared_error for result in scalar]),
    )


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("case", ("structured", "ties", "finite-extremes", "noncontiguous"))
def test_contiguous_viterbi_suffix_expansion_exactly_matches_gather(bits, case):
    shift = bits * 2
    state_count = 1 << 16
    suffix_count = state_count >> shift
    prefix_count = 1 << shift
    state_ids = torch.arange(state_count, dtype=torch.long)
    predecessor_suffix = state_ids >> shift

    if case == "ties":
        best_cost = torch.zeros((2, suffix_count), dtype=torch.float32)
    elif case == "finite-extremes":
        pattern = torch.tensor(
            [
                -torch.finfo(torch.float32).max,
                -0.0,
                0.0,
                torch.finfo(torch.float32).max,
            ],
            dtype=torch.float32,
        )
        best_cost = pattern.repeat((2, (suffix_count + pattern.numel() - 1) // pattern.numel()))[:, :suffix_count]
    elif case == "noncontiguous":
        storage = torch.arange(2 * suffix_count * 2, dtype=torch.float32).reshape(2, suffix_count * 2)
        best_cost = storage[:, ::2]
        assert not best_cost.is_contiguous()
    else:
        suffix_ids = torch.arange(suffix_count, dtype=torch.float32)
        best_cost = torch.stack((suffix_ids, -suffix_ids - 0.5))

    gathered = best_cost[:, predecessor_suffix]
    contiguous = best_cost.repeat_interleave(prefix_count, dim=1)

    assert gathered.shape == (2, state_count)
    assert torch.equal(contiguous, gathered)


@pytest.mark.parametrize("bits", range(1, 9))
def test_two_pass_tail_biting_is_transition_consistent_and_planar_packable(bits):
    generator = torch.Generator().manual_seed(20260900 + bits)
    sequences = torch.randn((2, 128, 2), generator=generator)
    codebook = torch.randn((1 << 16, 2), generator=generator)

    result = tail_biting_viterbi_quantize(sequences, codebook, bits=bits)
    packed = pack_trellis_states(result.states, bits=bits)

    assert result.states.shape == (2, 128)
    assert result.values.shape == (2, 128, 2)
    assert result.squared_error.shape == (2,)
    assert packed.shape == (2, qvq_words_per_tile(bits))
    torch.testing.assert_close(
        result.squared_error,
        (sequences - result.values).square().sum(dim=(1, 2)),
        atol=2e-4,
        rtol=2e-5,
    )


def test_weighted_tail_biting_rotates_weights_with_the_provisional_sequence():
    generator = torch.Generator().manual_seed(20260907)
    sequences = torch.randn((2, 9, 2), generator=generator)
    codebook = torch.randn((64, 2), generator=generator)
    step_weights = torch.rand((2, 9), generator=generator) * 4 + 0.1
    midpoint = sequences.shape[1] // 2
    provisional = batched_viterbi_quantize(
        torch.roll(sequences, shifts=midpoint, dims=1),
        codebook,
        bits=2,
        step_weights=torch.roll(step_weights, shifts=midpoint, dims=1),
    )
    overlap = provisional.states[:, midpoint - 1] & 3
    expected = batched_viterbi_quantize(
        sequences,
        codebook,
        bits=2,
        overlap=overlap,
        step_weights=step_weights,
    )

    actual = tail_biting_viterbi_quantize(
        sequences,
        codebook,
        bits=2,
        step_weights=step_weights,
    )

    assert torch.equal(actual.states, expected.states)
    assert torch.equal(actual.values, expected.values)
    assert torch.equal(actual.squared_error, expected.squared_error)


def test_tail_biting_overlap_scores_match_exhaustive_weighted_paths():
    generator = torch.Generator().manual_seed(20260915)
    sequences = torch.randn((1, 3, 2), generator=generator)
    codebook = torch.randn((1 << 6, 2), generator=generator)
    step_weights = torch.tensor([[0.25, 3.0, 1.5]])

    actual = qvq_module._tail_biting_overlap_scores(
        sequences,
        codebook,
        bits=2,
        boundary=1,
        step_weights=step_weights,
    )
    expected = torch.full((4,), torch.inf)
    for first_state in range(1 << 6):
        for second_edge, third_edge in itertools.product(range(16), repeat=2):
            second_state = ((first_state << 4) & 63) | second_edge
            third_state = ((second_state << 4) & 63) | third_edge
            states = torch.tensor([first_state, second_state, third_state])
            error = ((sequences[0] - codebook[states]).square().sum(dim=1) * step_weights[0]).sum()
            overlap = second_state >> 4
            expected[overlap] = torch.minimum(expected[overlap], error)

    torch.testing.assert_close(actual[0], expected, atol=2e-5, rtol=2e-5)


def test_full_tail_biting_candidate_list_matches_exhaustive_circular_oracle():
    generator = torch.Generator().manual_seed(20260916)
    sequences = torch.randn((1, 3, 2), generator=generator)
    codebook = torch.randn((1 << 6, 2), generator=generator)

    actual = tail_biting_viterbi_quantize(
        sequences,
        codebook,
        bits=2,
        candidate_count=4,
    )
    best_error = None
    for edges in itertools.product(range(16), repeat=3):
        state = 0
        states = []
        for edge_index, edge in enumerate(edges * 2):
            state = ((state << 4) & 63) | edge
            if edge_index >= len(edges):
                states.append(state)
        candidate_states = torch.tensor(states)
        assert candidate_states[0].item() == (((candidate_states[-1].item() << 4) & 63) | edges[0])
        error = (sequences[0] - codebook[candidate_states]).square().sum().item()
        if best_error is None or error < best_error:
            best_error = error

    assert best_error is not None
    assert actual.squared_error.item() == pytest.approx(best_error, abs=2e-5)
    assert torch.equal(actual.states[:, 1:] >> 4, actual.states[:, :-1] & 3)
    assert torch.equal(actual.states[:, :1] >> 4, actual.states[:, -1:] & 3)


def test_wide_tail_biting_candidates_strictly_improve_a_w2_regression_case():
    generator = torch.Generator().manual_seed(1000)
    sequences = torch.randn((2, 7, 2), generator=generator)
    codebook = torch.randn((1 << 8, 2), generator=generator)

    baseline = tail_biting_viterbi_quantize(sequences, codebook, bits=2, candidate_count=1)
    widened = tail_biting_viterbi_quantize(sequences, codebook, bits=2, candidate_count=4)

    assert torch.all(widened.squared_error <= baseline.squared_error)
    assert widened.squared_error[0] < baseline.squared_error[0]
    torch.testing.assert_close(
        widened.squared_error,
        (sequences - widened.values).square().sum(dim=(1, 2)),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.equal(widened.states[:, 1:] >> 4, widened.states[:, :-1] & 15)
    assert torch.equal(widened.states[:, :1] >> 4, widened.states[:, -1:] & 15)


def test_tail_biting_candidate_cost_ties_keep_first_candidate_and_packed_words(monkeypatch):
    sequences = torch.zeros((2, 32, 2), dtype=torch.float32)
    codebook = torch.arange(1 << 16, dtype=torch.float32).unsqueeze(1).repeat(1, 2)
    first_states = torch.full((2, 32), 0x1111, dtype=torch.int64)
    second_states = torch.zeros((2, 32), dtype=torch.int64)
    one_ulp_above = torch.nextafter(torch.tensor(1.0), torch.tensor(torch.inf))
    two_ulps_above = torch.nextafter(one_ulp_above, torch.tensor(torch.inf))
    calls = []

    def fake_viterbi(sequence, candidate_codebook, *, overlap=None, **kwargs):
        calls.append(None if overlap is None else overlap.clone())
        if overlap is None:
            states = first_states
            squared_error = torch.zeros(2)
        elif torch.equal(overlap, torch.full((2,), 0x111, dtype=torch.int64)):
            states = first_states
            squared_error = torch.ones(2)
        else:
            states = second_states
            # Batch 0 is an exact tie; batch 1 is only two FP32 ULPs worse.
            squared_error = torch.tensor([1.0, two_ulps_above.item()])
        return qvq_module.TrellisQuantizationResult(
            states=states,
            values=candidate_codebook[states],
            squared_error=squared_error,
        )

    monkeypatch.setattr(qvq_module, "batched_viterbi_quantize", fake_viterbi)
    monkeypatch.setattr(
        qvq_module,
        "_tail_biting_overlap_scores",
        lambda *args, **kwargs: torch.zeros((2, 1 << 12)),
    )

    expected_packed = pack_trellis_states(first_states, bits=2)
    for _ in range(3):
        actual = qvq_module.tail_biting_viterbi_quantize(sequences, codebook, bits=2, candidate_count=2)
        assert torch.equal(actual.states, first_states)
        assert torch.equal(pack_trellis_states(actual.states, bits=2), expected_packed)
        assert torch.equal(actual.squared_error, torch.ones(2))

    assert len(calls) == 9


@pytest.mark.parametrize("candidate_count", (0, -1, True, 1.5))
def test_tail_biting_rejects_invalid_candidate_counts(candidate_count):
    with pytest.raises(ValueError, match="candidate count"):
        tail_biting_viterbi_quantize(
            torch.zeros((1, 2, 2)),
            torch.zeros((1 << 8, 2)),
            bits=4,
            candidate_count=candidate_count,
        )


def test_tail_biting_without_overlap_runs_one_exact_pass(monkeypatch):
    generator = torch.Generator().manual_seed(20260908)
    sequences = torch.randn((2, 9, 2), generator=generator)
    codebook = torch.randn((1 << 8, 2), generator=generator)
    expected = batched_viterbi_quantize(sequences, codebook, bits=4)
    calls = []
    original = qvq_module.batched_viterbi_quantize

    def counted_pass(*args, **kwargs):
        calls.append(kwargs.get("overlap"))
        return original(*args, **kwargs)

    monkeypatch.setattr(qvq_module, "batched_viterbi_quantize", counted_pass)
    actual = qvq_module.tail_biting_viterbi_quantize(
        sequences,
        codebook,
        bits=4,
        candidate_count=8,
    )

    assert calls == [None]
    assert torch.equal(actual.states, expected.states)
    assert torch.equal(actual.values, expected.values)
    assert torch.equal(actual.squared_error, expected.squared_error)


@pytest.mark.parametrize("case", ("ties", "structured", "finite-extremes", "noncontiguous"))
def test_tail_biting_without_overlap_matches_explicit_overlap_oracle_for_corner_cases(
    case,
):
    batch_size = 2
    step_count = 2 if case == "finite-extremes" else 7
    state_count = 1 << 8

    if case == "ties":
        sequences = torch.zeros((batch_size, step_count, 2), dtype=torch.float32)
        codebook = torch.zeros((state_count, 2), dtype=torch.float32)
    elif case == "finite-extremes":
        sequences = torch.tensor([[[-1024.0, 1024.0], [1024.0, -1024.0]], [[-511.0, -257.0], [511.0, 257.0]]])
        state_ids = torch.arange(state_count, dtype=torch.float32)
        codebook = torch.stack((state_ids - 128.0, 127.0 - state_ids), dim=1) * 8.0
    elif case == "noncontiguous":
        generator = torch.Generator().manual_seed(20260909)
        sequence_storage = torch.randn((batch_size, step_count, 4), generator=generator)
        codebook_storage = torch.randn((state_count, 4), generator=generator)
        sequences = sequence_storage[:, :, ::2]
        codebook = codebook_storage[:, ::2]
        assert not sequences.is_contiguous()
        assert not codebook.is_contiguous()
    else:
        state_ids = torch.arange(state_count, dtype=torch.float32)
        codebook = torch.stack(
            (state_ids.remainder(16), torch.div(state_ids, 16, rounding_mode="floor")),
            dim=1,
        )
        sequences = torch.tensor(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                    [7.0, 8.0],
                    [9.0, 10.0],
                    [11.0, 12.0],
                    [13.0, 14.0],
                ],
                [
                    [14.0, 13.0],
                    [12.0, 11.0],
                    [10.0, 9.0],
                    [8.0, 7.0],
                    [6.0, 5.0],
                    [4.0, 3.0],
                    [2.0, 1.0],
                ],
            ]
        )

    expected = batched_viterbi_quantize(
        sequences,
        codebook,
        bits=4,
        overlap=torch.zeros(batch_size, dtype=torch.long),
    )
    actual = tail_biting_viterbi_quantize(sequences, codebook, bits=4)

    assert torch.equal(actual.states, expected.states)
    assert torch.equal(actual.values, expected.values)
    assert torch.equal(actual.squared_error, expected.squared_error)
    if case == "ties":
        assert torch.count_nonzero(actual.states) == 0
    torch.testing.assert_close(
        actual.squared_error,
        (sequences - actual.values).square().sum(dim=(1, 2)),
        atol=1e-4,
        rtol=1e-6,
    )


def test_tail_biting_w8_fast_path_matches_production_state_space_oracle():
    state_ids = torch.arange(1 << 16, dtype=torch.long)
    codebook = torch.stack((state_ids & 255, state_ids >> 8), dim=1).to(torch.float32)
    sequences = torch.tensor(
        [[[1.0, 2.0], [127.0, 129.0]], [[254.0, 253.0], [64.0, 192.0]]],
        dtype=torch.float32,
    )

    expected = batched_viterbi_quantize(
        sequences,
        codebook,
        bits=8,
        overlap=torch.zeros(sequences.shape[0], dtype=torch.long),
    )
    actual = tail_biting_viterbi_quantize(sequences, codebook, bits=8)

    assert torch.equal(actual.states, expected.states)
    assert torch.equal(actual.values, expected.values)
    assert torch.equal(actual.squared_error, expected.squared_error)
    torch.testing.assert_close(
        actual.squared_error,
        (sequences - actual.values).square().sum(dim=(1, 2)),
        atol=0,
        rtol=0,
    )


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("ties", (False, True))
def test_qvq_w8_independent_mps_kernel_matches_cpu_oracle(weighted, ties):
    generator = torch.Generator().manual_seed(20260812 + weighted * 2 + ties)
    codebook = pgc16_codebook(dtype=torch.float32)
    sequences = torch.zeros((3, 17, 2)) if ties else torch.randn((3, 17, 2), generator=generator)
    if ties:
        codebook = torch.zeros_like(codebook)
    step_weights = torch.rand((3, 17), generator=generator).add_(0.1) if weighted else None

    expected = batched_viterbi_quantize(sequences, codebook, bits=8, step_weights=step_weights)
    actual = batched_viterbi_quantize(
        sequences.to("mps"),
        codebook.to("mps"),
        bits=8,
        overlap=torch.zeros(3, dtype=torch.int64, device="mps"),
        step_weights=None if step_weights is None else step_weights.to("mps"),
    )

    assert torch.equal(actual.states.cpu(), expected.states)
    assert torch.equal(actual.values.cpu(), expected.values)
    torch.testing.assert_close(actual.squared_error.cpu(), expected.squared_error, atol=5e-6, rtol=1e-6)
    if ties:
        assert torch.count_nonzero(actual.states) == 0


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("ties", (False, True))
def test_qvq_w7_5_two_context_mps_kernel_matches_cpu_oracle(weighted, ties):
    generator = torch.Generator().manual_seed(20260813 + weighted * 2 + ties)
    codebook = pgc16_codebook(dtype=torch.float32)
    sequences = torch.zeros((3, 17, 2)) if ties else torch.randn((3, 17, 2), generator=generator)
    if ties:
        codebook = torch.zeros_like(codebook)
    step_weights = torch.rand((3, 17), generator=generator).add_(0.1) if weighted else None
    overlap = torch.tensor([0, 1, 0], dtype=torch.int64)

    expected = batched_viterbi_quantize(
        sequences,
        codebook,
        bits=7.5,
        overlap=overlap,
        step_weights=step_weights,
    )
    actual = batched_viterbi_quantize(
        sequences.to("mps"),
        codebook.to("mps"),
        bits=7.5,
        overlap=overlap.to("mps"),
        step_weights=None if step_weights is None else step_weights.to("mps"),
    )

    assert torch.equal(actual.states.cpu(), expected.states)
    assert torch.equal(actual.values.cpu(), expected.values)
    torch.testing.assert_close(actual.squared_error.cpu(), expected.squared_error, atol=5e-6, rtol=1e-6)
    if ties:
        assert torch.equal(actual.states[:, 0].cpu() >> 15, overlap)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("ties", (False, True))
def test_qvq_w7_four_context_mps_kernel_matches_cpu_oracle(weighted, ties):
    generator = torch.Generator().manual_seed(20260814 + weighted * 2 + ties)
    codebook = pgc16_codebook(dtype=torch.float32)
    sequences = torch.zeros((4, 17, 2)) if ties else torch.randn((4, 17, 2), generator=generator)
    if ties:
        codebook = torch.zeros_like(codebook)
    step_weights = torch.rand((4, 17), generator=generator).add_(0.1) if weighted else None
    overlap = torch.arange(4, dtype=torch.int64)

    expected = batched_viterbi_quantize(sequences, codebook, bits=7, overlap=overlap, step_weights=step_weights)
    actual = batched_viterbi_quantize(
        sequences.to("mps"),
        codebook.to("mps"),
        bits=7,
        overlap=overlap.to("mps"),
        step_weights=None if step_weights is None else step_weights.to("mps"),
    )

    assert torch.equal(actual.states.cpu(), expected.states)
    assert torch.equal(actual.values.cpu(), expected.values)
    torch.testing.assert_close(actual.squared_error.cpu(), expected.squared_error, atol=5e-6, rtol=1e-6)
    if ties:
        assert torch.equal(actual.states[:, 0].cpu() >> 14, overlap)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("ties", (False, True))
def test_qvq_w6_5_eight_context_mps_kernel_matches_cpu_oracle(weighted, ties):
    generator = torch.Generator().manual_seed(20260815 + weighted * 2 + ties)
    codebook = pgc16_codebook(dtype=torch.float32)
    sequences = torch.zeros((8, 17, 2)) if ties else torch.randn((8, 17, 2), generator=generator)
    if ties:
        codebook = torch.zeros_like(codebook)
    step_weights = torch.rand((8, 17), generator=generator).add_(0.1) if weighted else None
    overlap = torch.arange(8, dtype=torch.int64)

    expected = batched_viterbi_quantize(sequences, codebook, bits=6.5, overlap=overlap, step_weights=step_weights)
    actual = batched_viterbi_quantize(
        sequences.to("mps"),
        codebook.to("mps"),
        bits=6.5,
        overlap=overlap.to("mps"),
        step_weights=None if step_weights is None else step_weights.to("mps"),
    )

    assert torch.equal(actual.states.cpu(), expected.states)
    assert torch.equal(actual.values.cpu(), expected.values)
    torch.testing.assert_close(actual.squared_error.cpu(), expected.squared_error, atol=5e-6, rtol=1e-6)
    if ties:
        assert torch.equal(actual.states[:, 0].cpu() >> 13, overlap)


@pytest.mark.parametrize(
    ("sequences", "codebook"),
    [
        (torch.full((1, 2, 2), torch.inf), torch.zeros((1 << 8, 2))),
        (torch.zeros((1, 2, 2)), torch.full((1 << 8, 2), torch.nan)),
    ],
)
def test_tail_biting_without_overlap_preserves_nonfinite_validation(sequences, codebook):
    with pytest.raises(ValueError, match="finite"):
        tail_biting_viterbi_quantize(sequences, codebook, bits=4)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("seed_offset", range(3))
def test_qvq_mps_contiguous_suffix_viterbi_matches_exact_cpu_gather(bits, seed_offset):
    state_ids = torch.arange(1 << 16, dtype=torch.long)
    codebook = torch.stack((state_ids & 255, state_ids >> 8), dim=1).to(torch.float32)
    generator = torch.Generator().manual_seed(20260920 + seed_offset)
    sequences = torch.randint(0, 256, (2, 3, 2), generator=generator).to(torch.float32)

    expected = batched_viterbi_quantize(sequences, codebook, bits=bits)
    actual = batched_viterbi_quantize(sequences.to("mps"), codebook.to("mps"), bits=bits)

    assert torch.equal(actual.states.cpu(), expected.states)
    assert torch.equal(actual.values.cpu(), expected.values)
    assert torch.equal(actual.squared_error.cpu(), expected.squared_error)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("bits", tuple(rate / 2 for rate in range(2, 17)))
def test_qvq_low_rate_native_mps_viterbi_matches_weighted_constrained_cpu_oracle(bits):
    generator = torch.Generator().manual_seed(20261010 + int(bits * 2))
    codebook = pgc16_codebook(dtype=torch.float32)
    sequences = torch.randn((2, 128, 2), generator=generator)
    step_weights = torch.rand((2, 128), generator=generator).add_(0.1)
    transition_bits = qvq_transition_bits(bits)
    overlap = torch.randint(0, 1 << (16 - transition_bits), (2,), generator=generator, dtype=torch.int64)

    expected = batched_viterbi_quantize(
        sequences,
        codebook,
        bits=bits,
        overlap=overlap,
        step_weights=step_weights,
    )
    actual = batched_viterbi_quantize(
        sequences.to("mps"),
        codebook.to("mps"),
        bits=bits,
        overlap=overlap.to("mps"),
        step_weights=step_weights.to("mps"),
    )

    assert torch.equal(actual.states.cpu(), expected.states)
    assert torch.equal(actual.values.cpu(), expected.values)
    torch.testing.assert_close(actual.squared_error.cpu(), expected.squared_error, atol=5e-6, rtol=1e-6)

    for candidate_count in (4, 8, 16):
        expected_tail = tail_biting_viterbi_quantize(
            sequences,
            codebook,
            bits=bits,
            step_weights=step_weights,
            candidate_count=candidate_count,
        )
        actual_tail = tail_biting_viterbi_quantize(
            sequences.to("mps"),
            codebook.to("mps"),
            bits=bits,
            step_weights=step_weights.to("mps"),
            candidate_count=candidate_count,
        )
        assert torch.equal(actual_tail.states.cpu(), expected_tail.states)
        assert torch.equal(actual_tail.values.cpu(), expected_tail.values)
        torch.testing.assert_close(
            actual_tail.squared_error.cpu(), expected_tail.squared_error, atol=5e-6, rtol=1e-6
        )


def test_block_ldl_factor_reconstructs_spd_hessian_for_multiple_block_sizes():
    generator = torch.Generator().manual_seed(20261001)
    source = torch.randn((16, 16), generator=generator)
    H = source @ source.T + torch.eye(16) * 0.25

    for block_size in (1, 2, 4, 8, 16):
        L, D = block_ldl_factor(H, block_size=block_size)
        torch.testing.assert_close(L @ D @ L.T, H, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(torch.diagonal(L), torch.ones(16))
        assert torch.count_nonzero(torch.triu(L, diagonal=1)) == 0


def test_yaqa_hessian_stabilization_retries_isotropic_damping_without_clamping():
    hessian = torch.tensor([[1.0, 0.25], [0.25, 0.06245]], dtype=torch.float32)
    off_diagonal = hessian[0, 1].clone()

    with patch("torch.linalg.cholesky_ex", wraps=torch.linalg.cholesky_ex) as cholesky:
        factor = qvq_module.stabilized_block_ldl_factor(
            hessian,
            block_size=1,
            retry_damping=torch.tensor(1e-4),
        )

    assert factor.retry_count == 1
    assert factor.effective_damping == torch.tensor(1e-4)
    assert cholesky.call_count == 2
    torch.testing.assert_close(factor.L @ factor.D @ factor.L.T, factor.hessian)
    assert factor.hessian[0, 1] == off_diagonal
    assert factor.hessian[1, 0] == off_diagonal


@pytest.mark.parametrize("bits", (1, 1.5, 2))
def test_block_ldlq_matches_qtip_author_reference_recurrence_at_low_rates(bits):
    """Keep QVQ's transposed recurrence identical to the authoritative QTIP implementation."""

    generator = torch.Generator().manual_seed(20261040 + int(bits * 2))
    inner = torch.randn((8, 8), generator=generator)
    hessian_source = torch.randn((8, 8), generator=generator)
    hessian = hessian_source @ hessian_source.T + torch.eye(8) * 0.5
    codebook = torch.randn((16, 2), generator=generator)

    actual, actual_states = block_ldlq_inner(
        inner,
        hessian,
        codebook,
        bits=bits,
        tile_rows=4,
        tile_cols=4,
        trellis_batch_size=2,
    )

    # Cornell-RelaxML/qtip lib/utils/math_utils.py::block_LDL normalizes each
    # Cholesky block column by its diagonal block, then Algorithm 5 clears the
    # diagonal blocks before applying feedback from already rounded columns.
    cholesky = torch.linalg.cholesky(hessian)
    reference_feedback = cholesky.clone()
    for start in range(0, 8, 4):
        stop = start + 4
        reference_feedback[:, start:stop] = reference_feedback[:, start:stop] @ torch.linalg.inv(
            cholesky[start:stop, start:stop]
        )
        reference_feedback[start:stop, start:stop] = 0

    expected = torch.zeros_like(inner)
    expected_states = torch.empty((2, 2, 8), dtype=torch.long)
    for block in range(1, -1, -1):
        start = block * 4
        stop = start + 4
        corrected = inner[start:stop] + reference_feedback[start:, start:stop].T @ (
            inner[start:] - expected[start:]
        )
        sequences = corrected.reshape(4, 2, 4).permute(1, 0, 2).reshape(2, 8, 2)
        rounded = tail_biting_viterbi_quantize(sequences, codebook, bits=bits)
        expected[start:stop] = rounded.values.reshape(2, 4, 4).permute(1, 0, 2).reshape(4, 8)
        expected_states[block] = rounded.states

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    assert torch.equal(actual_states, expected_states.reshape(-1, 8))


def test_yaqa_sketch_b_matches_independent_per_sequence_gram_reference():
    generator = torch.Generator().manual_seed(20261020)
    gradient_storage = torch.randn((5, 4, 6), generator=generator)
    gradients = gradient_storage[..., ::2]
    assert not gradients.is_contiguous()

    input_hessian, output_hessian = yaqa_sketch_b(gradients)
    expected_input = sum(
        (gradient.double().T @ gradient.double() for gradient in gradients),
        torch.zeros((3, 3), dtype=torch.float64),
    )
    expected_output = sum(
        (gradient.double() @ gradient.double().T for gradient in gradients),
        torch.zeros((4, 4), dtype=torch.float64),
    )
    expected_input /= gradients.shape[0] * gradients.shape[1]
    expected_output /= gradients.shape[0] * gradients.shape[2]

    torch.testing.assert_close(input_hessian, expected_input)
    torch.testing.assert_close(output_hessian, expected_output)
    torch.testing.assert_close(input_hessian, input_hessian.T)
    torch.testing.assert_close(output_hessian, output_hessian.T)
    assert torch.linalg.eigvalsh(input_hessian).min() >= -1e-12
    assert torch.linalg.eigvalsh(output_hessian).min() >= -1e-12

    averaged_gradient = gradients.double().mean(dim=0)
    invalid_averaged_input = averaged_gradient.T @ averaged_gradient / gradients.shape[1]
    invalid_averaged_output = averaged_gradient @ averaged_gradient.T / gradients.shape[2]
    assert not torch.allclose(input_hessian, invalid_averaged_input)
    assert not torch.allclose(output_hessian, invalid_averaged_output)


@pytest.mark.parametrize(
    ("gradients", "exception", "message"),
    (
        (torch.zeros((2, 3)), ValueError, "shape"),
        (torch.zeros((0, 3, 4)), ValueError, "non-empty"),
        (torch.zeros((2, 3, 4), dtype=torch.int64), TypeError, "floating-point"),
        (torch.full((2, 3, 4), torch.nan), ValueError, "finite"),
    ),
)
def test_yaqa_sketch_b_rejects_invalid_gradient_populations(gradients, exception, message):
    with pytest.raises(exception, match=message):
        yaqa_sketch_b(gradients)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_yaqa_sketch_b_mps_fp32_fallback_matches_cpu_reference():
    generator = torch.Generator().manual_seed(20261024)
    gradients = torch.randn((4, 5, 3), generator=generator)
    expected_input, expected_output = yaqa_sketch_b(gradients)

    actual_input, actual_output = yaqa_sketch_b(gradients.to("mps"))

    assert actual_input.dtype == torch.float32
    assert actual_output.dtype == torch.float32
    torch.testing.assert_close(actual_input.cpu(), expected_input.float(), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_output.cpu(), expected_output.float(), atol=2e-6, rtol=2e-6)


def test_yaqa_proxy_matches_independent_four_index_reference_and_ldlq_identity_case():
    generator = torch.Generator().manual_seed(20261021)
    weight = torch.randn((3, 2), generator=generator)
    reconstructed = torch.randn((3, 2), generator=generator)
    input_source = torch.randn((2, 2), generator=generator)
    output_source = torch.randn((3, 3), generator=generator)
    input_hessian = input_source @ input_source.T + torch.eye(2) * 0.2
    output_hessian = output_source @ output_source.T + torch.eye(3) * 0.2
    error = (reconstructed - weight).double()

    expected = torch.zeros((), dtype=torch.float64)
    for output_a, output_b, input_a, input_b in itertools.product(range(3), range(3), range(2), range(2)):
        expected += (
            error[output_a, input_a]
            * input_hessian[input_a, input_b]
            * error[output_b, input_b]
            * output_hessian[output_b, output_a]
        )

    actual = yaqa_proxy_loss(weight, reconstructed, input_hessian, output_hessian)
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        yaqa_proxy_loss(weight, reconstructed, input_hessian, torch.eye(3)),
        qvq_proxy_loss(weight, reconstructed, input_hessian),
    )


def test_block_ldlq_inner_identity_hessian_matches_independent_tile_rounding():
    generator = torch.Generator().manual_seed(20261002)
    inner = torch.randn((16, 32), generator=generator)
    codebook = torch.randn((64, 2), generator=generator)

    quantized, states = block_ldlq_inner(
        inner,
        torch.eye(16),
        codebook,
        bits=2,
    )
    sequences = inner.reshape(16, 2, 16).permute(1, 0, 2).reshape(2, 128, 2)
    expected = tail_biting_viterbi_quantize(sequences, codebook, bits=2)

    assert torch.equal(states, expected.states)
    assert torch.equal(
        quantized,
        expected.values.reshape(2, 16, 16).permute(1, 0, 2).reshape(16, 32),
    )


def test_block_ldlq_forwards_wide_tail_biting_candidates_without_proxy_regression():
    generator = torch.Generator().manual_seed(1000)
    inner = torch.randn((16, 16), generator=generator)
    codebook = torch.randn((1 << 8, 2), generator=generator)

    baseline, _ = block_ldlq_inner(
        inner,
        torch.eye(16),
        codebook,
        bits=2,
        tail_biting_candidates=1,
    )
    widened, _ = block_ldlq_inner(
        inner,
        torch.eye(16),
        codebook,
        bits=2,
        tail_biting_candidates=4,
    )

    assert (inner - widened).square().sum() <= (inner - baseline).square().sum()


def test_yaqa_identity_output_hessian_is_exact_block_ldlq_control():
    generator = torch.Generator().manual_seed(20261022)
    inner = torch.randn((32, 32), generator=generator)
    source = torch.randn((32, 32), generator=generator)
    input_hessian = source @ source.T + torch.eye(32) * 0.25
    codebook = torch.randn((64, 2), generator=generator)

    expected_weight, expected_states = block_ldlq_inner(
        inner,
        input_hessian,
        codebook,
        bits=2,
        tail_biting_candidates=2,
    )
    actual_weight, actual_states = yaqa_inner(
        inner,
        input_hessian,
        torch.eye(32),
        codebook,
        bits=2,
        tail_biting_candidates=2,
    )

    assert torch.equal(actual_states, expected_states)
    assert torch.equal(actual_weight, expected_weight)


def test_yaqa_antidiagonal_schedule_satisfies_the_two_sided_fixed_point():
    generator = torch.Generator().manual_seed(20261023)
    inner = torch.randn((8, 8), generator=generator)
    input_source = torch.randn((8, 8), generator=generator)
    output_source = torch.randn((8, 8), generator=generator)
    input_hessian = input_source @ input_source.T + torch.eye(8) * 0.5
    output_hessian = output_source @ output_source.T + torch.eye(8) * 0.5
    codebook = torch.randn((64, 2), generator=generator)

    quantized, states = yaqa_inner(
        inner,
        input_hessian,
        output_hessian,
        codebook,
        bits=2,
        tile_rows=4,
        tile_cols=4,
        tail_biting_candidates=4,
    )
    input_L, _ = block_ldl_factor(input_hessian, block_size=4)
    output_L, _ = block_ldl_factor(output_hessian, block_size=4)
    input_feedback = input_L - torch.eye(8)
    output_feedback = output_L - torch.eye(8)
    error = inner - quantized

    for input_block, output_block in itertools.product(range(2), repeat=2):
        input_start = input_block * 4
        output_start = output_block * 4
        left = input_feedback[input_start:, input_start : input_start + 4].T
        right = output_feedback[output_start:, output_start : output_start + 4]
        target = (
            inner[input_start : input_start + 4, output_start : output_start + 4]
            + left @ error[input_start:, output_start:] @ right
            + left @ error[input_start:, output_start : output_start + 4]
            + error[input_start : input_start + 4, output_start:] @ right
        )
        expected = tail_biting_viterbi_quantize(
            target.reshape(1, 8, 2),
            codebook,
            bits=2,
            candidate_count=4,
        )
        tile_index = input_block * 2 + output_block
        assert torch.equal(states[tile_index], expected.states[0])
        assert torch.equal(
            quantized[input_start : input_start + 4, output_start : output_start + 4],
            expected.values.reshape(4, 4),
        )


def test_yaqa_prepared_factorization_is_exact_and_immutable():
    generator = torch.Generator().manual_seed(20261024)
    inner = torch.randn((8, 8), generator=generator)
    input_source = torch.randn((8, 8), generator=generator)
    output_source = torch.randn((8, 8), generator=generator)
    input_hessian = input_source @ input_source.T + torch.eye(8) * 0.5
    output_hessian = output_source @ output_source.T + torch.eye(8) * 0.5
    codebook = torch.randn((64, 2), generator=generator)
    input_factor = qvq_module.stabilized_block_ldl_factor(
        input_hessian,
        block_size=4,
        retry_damping=torch.tensor(1e-4),
    )
    output_factor = qvq_module.stabilized_block_ldl_factor(
        output_hessian,
        block_size=4,
        retry_damping=torch.tensor(1e-4),
    )
    preserved = input_factor.L.clone(), output_factor.L.clone()

    expected = yaqa_inner(
        inner,
        input_hessian,
        output_hessian,
        codebook,
        bits=2,
        tile_rows=4,
        tile_cols=4,
        tail_biting_candidates=4,
    )
    actual = yaqa_inner(
        inner,
        input_hessian,
        output_hessian,
        codebook,
        bits=2,
        tile_rows=4,
        tile_cols=4,
        tail_biting_candidates=4,
        factorization=(input_factor, output_factor),
    )

    assert all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))
    assert torch.equal(input_factor.L, preserved[0])
    assert torch.equal(output_factor.L, preserved[1])

    input_hessian.diagonal().add_(1e-3)
    with pytest.raises(ValueError, match="originating Hessian"):
        yaqa_inner(
            inner,
            input_hessian,
            output_hessian,
            codebook,
            bits=2,
            tile_rows=4,
            tile_cols=4,
            factorization=(input_factor, output_factor),
        )


@pytest.mark.parametrize(
    "format_kwargs",
    (
        {},
        {"bank_count": 2, "v2b2_p32": True, "yaqa_v2b2_family_mode": "fixed_block_ldlq"},
        {"bank_count": 2, "v2b2_p32": True, "yaqa_v2b2_family_mode": "reselect"},
        {
            "bank_count": 2,
            "v2b2_p32": True,
            "yaqa_v2b2_family_mode": "reselect",
            "yaqa_sample_strategy": "64_16x16",
        },
        {"bank_count": 4, "v2b4_p64": True},
    ),
)
def test_yaqa_all_candidates_reuse_one_input_and_output_factorization(format_kwargs):
    generator = torch.Generator().manual_seed(20261025)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    input_source = torch.randn((24, 16), generator=generator)
    output_source = torch.randn((24, 16), generator=generator)
    input_hessian = input_source.T @ input_source / input_source.shape[0]
    output_hessian = output_source.T @ output_source / output_source.shape[0]
    original_stabilized = qvq_module.stabilized_block_ldl_factor

    with patch.object(qvq_module, "stabilized_block_ldl_factor", wraps=original_stabilized) as stabilized:
        result = quantize_qvq_linear(
            weight,
            input_hessian,
            bits=2,
            rounding="yaqa",
            output_hessian=output_hessian,
            trellis_batch_size=1,
            **format_kwargs,
        )

    assert stabilized.call_count == 2
    assert torch.isfinite(result.weight).all()


def test_yaqa_two_sided_feedback_improves_proxy_kld_top1_and_top5_overlap_fixture():
    generator = torch.Generator().manual_seed(2008)
    inner = torch.randn((8, 8), generator=generator) * 0.6
    activations = torch.randn((64, 8), generator=generator)
    downstream = torch.randn((8, 12), generator=generator)
    input_hessian = activations.T @ activations / activations.shape[0] + torch.eye(8) * 0.1
    output_hessian = downstream @ downstream.T / downstream.shape[1] + torch.eye(8) * 0.1
    codebook = torch.randn((64, 2), generator=generator)

    ldlq_weight, _ = block_ldlq_inner(
        inner,
        input_hessian,
        codebook,
        bits=2,
        tile_rows=4,
        tile_cols=4,
        tail_biting_candidates=4,
    )
    yaqa_weight, _ = yaqa_inner(
        inner,
        input_hessian,
        output_hessian,
        codebook,
        bits=2,
        tile_rows=4,
        tile_cols=4,
        tail_biting_candidates=4,
    )

    dense_logits = (activations @ inner) @ downstream
    ldlq_logits = (activations @ ldlq_weight) @ downstream
    yaqa_logits = (activations @ yaqa_weight) @ downstream

    def metrics(logits):
        kld = F.kl_div(
            logits.log_softmax(dim=-1),
            dense_logits.softmax(dim=-1),
            reduction="batchmean",
        )
        top1 = (logits.argmax(dim=-1) == dense_logits.argmax(dim=-1)).float().mean()
        dense_top5 = dense_logits.topk(5, dim=-1).indices
        actual_top5 = logits.topk(5, dim=-1).indices
        top5_overlap = (dense_top5[:, :, None] == actual_top5[:, None, :]).any(dim=-1).float().mean()
        return kld, top1, top5_overlap

    ldlq_metrics = metrics(ldlq_logits)
    yaqa_metrics = metrics(yaqa_logits)
    assert yaqa_proxy_loss(inner.T, yaqa_weight.T, input_hessian, output_hessian) < yaqa_proxy_loss(
        inner.T,
        ldlq_weight.T,
        input_hessian,
        output_hessian,
    )
    assert yaqa_metrics[0] < ldlq_metrics[0]
    assert yaqa_metrics[1] >= ldlq_metrics[1]
    assert yaqa_metrics[2] >= ldlq_metrics[2]


@pytest.mark.parametrize(
    (
        "inner",
        "input_hessian",
        "output_hessian",
        "codebook",
        "kwargs",
        "exception",
        "message",
    ),
    (
        (
            torch.eye(4, dtype=torch.int64),
            torch.eye(4),
            torch.eye(4),
            torch.randn((64, 2)),
            {"tile_rows": 4, "tile_cols": 4},
            ValueError,
            "floating-point matrix",
        ),
        (
            torch.full((4, 4), torch.nan),
            torch.eye(4),
            torch.eye(4),
            torch.randn((64, 2)),
            {"tile_rows": 4, "tile_cols": 4},
            ValueError,
            "inner weight.*finite",
        ),
        (
            torch.empty((0, 4)),
            torch.empty((0, 0)),
            torch.eye(4),
            torch.randn((64, 2)),
            {"tile_rows": 4, "tile_cols": 4},
            ValueError,
            "positive",
        ),
        (
            torch.eye(4),
            torch.eye(4),
            torch.eye(4),
            torch.randn((64, 2)),
            {"tile_rows": 0, "tile_cols": 4},
            ValueError,
            "tile rows",
        ),
        (
            torch.eye(4),
            torch.eye(4),
            torch.eye(4),
            torch.randn((64, 2)),
            {"tile_rows": 4, "tile_cols": False},
            ValueError,
            "tile columns",
        ),
        (
            torch.eye(4),
            torch.eye(4),
            torch.eye(4),
            torch.empty((0, 2)),
            {"tile_rows": 4, "tile_cols": 4},
            ValueError,
            "codebook.*non-empty",
        ),
        (
            torch.eye(4),
            torch.eye(4),
            torch.eye(4),
            torch.full((64, 2), torch.nan),
            {"tile_rows": 4, "tile_cols": 4},
            ValueError,
            "codebook.*finite",
        ),
        (
            torch.eye(4),
            torch.eye(4, device="meta"),
            torch.eye(4),
            torch.randn((64, 2)),
            {"tile_rows": 4, "tile_cols": 4},
            ValueError,
            "share one device",
        ),
        (
            torch.randn((5, 4)),
            torch.eye(5),
            torch.eye(4),
            torch.randn((64, 2)),
            {"tile_rows": 4, "tile_cols": 4},
            ValueError,
            "divisible",
        ),
        (
            torch.eye(4),
            torch.eye(3),
            torch.eye(4),
            torch.randn((64, 2)),
            {"tile_rows": 4, "tile_cols": 4},
            ValueError,
            "input Hessian",
        ),
        (
            torch.eye(4),
            torch.eye(4),
            torch.eye(3),
            torch.randn((64, 2)),
            {"tile_rows": 4, "tile_cols": 4},
            ValueError,
            "output Hessian",
        ),
        (
            torch.eye(4),
            torch.eye(4, dtype=torch.int64),
            torch.eye(4),
            torch.randn((64, 2)),
            {"tile_rows": 4, "tile_cols": 4},
            TypeError,
            "Hessians.*floating-point",
        ),
        (
            torch.eye(4),
            torch.eye(4),
            torch.full((4, 4), torch.nan),
            torch.randn((64, 2)),
            {"tile_rows": 4, "tile_cols": 4},
            ValueError,
            "Hessians.*finite",
        ),
        (
            torch.eye(4),
            torch.eye(4),
            torch.eye(4),
            torch.zeros((64, 2), dtype=torch.int64),
            {"tile_rows": 4, "tile_cols": 4},
            TypeError,
            "codebook.*floating-point",
        ),
        (
            torch.eye(4),
            torch.eye(4),
            torch.eye(4),
            torch.randn((64, 3)),
            {"tile_rows": 4, "tile_cols": 4},
            ValueError,
            "tile size.*divisible",
        ),
        (
            torch.eye(4),
            torch.eye(4),
            torch.eye(4),
            torch.randn((64, 2)),
            {"tile_rows": 4, "tile_cols": 4, "trellis_batch_size": 0},
            ValueError,
            "batch size",
        ),
        (
            torch.eye(4),
            torch.eye(4),
            torch.eye(4),
            torch.randn((64, 2)),
            {"tile_rows": 4, "tile_cols": 4, "tail_biting_candidates": 0},
            ValueError,
            "candidate count",
        ),
    ),
)
def test_yaqa_inner_rejects_malformed_inputs(
    inner,
    input_hessian,
    output_hessian,
    codebook,
    kwargs,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        yaqa_inner(
            inner,
            input_hessian,
            output_hessian,
            codebook,
            bits=2,
            **kwargs,
        )


@pytest.mark.parametrize(
    (
        "weight",
        "reconstructed",
        "input_hessian",
        "output_hessian",
        "exception",
        "message",
    ),
    (
        (
            torch.eye(3),
            torch.eye(3)[:2],
            torch.eye(3),
            torch.eye(3),
            ValueError,
            "matching rank-2",
        ),
        (
            torch.eye(3),
            torch.eye(3),
            torch.eye(2),
            torch.eye(3),
            ValueError,
            "input Hessian",
        ),
        (
            torch.eye(3),
            torch.eye(3),
            torch.eye(3),
            torch.eye(2),
            ValueError,
            "output Hessian",
        ),
        (
            torch.eye(3),
            torch.eye(3),
            torch.eye(3, device="meta"),
            torch.eye(3),
            ValueError,
            "share one device",
        ),
        (
            torch.eye(3, dtype=torch.int64),
            torch.eye(3),
            torch.eye(3),
            torch.eye(3),
            TypeError,
            "floating-point",
        ),
        (
            torch.eye(3),
            torch.full((3, 3), torch.nan),
            torch.eye(3),
            torch.eye(3),
            ValueError,
            "finite",
        ),
    ),
)
def test_yaqa_proxy_rejects_malformed_inputs(
    weight,
    reconstructed,
    input_hessian,
    output_hessian,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        yaqa_proxy_loss(weight, reconstructed, input_hessian, output_hessian)


def test_yaqa_proxy_rejects_fp64_values_that_narrow_to_fp32_inf():
    weight = torch.zeros((2, 2), dtype=torch.float64)
    reconstructed = weight.clone()
    input_hessian = torch.eye(2, dtype=torch.float64)
    output_hessian = torch.eye(2, dtype=torch.float64)
    weight[0, 0] = 1e40

    with pytest.raises(ValueError, match="finite FP32"):
        yaqa_proxy_loss(weight, reconstructed, input_hessian, output_hessian)


def test_block_ldlq_hessian_diagonal_objective_matches_independent_weighted_viterbi():
    generator = torch.Generator().manual_seed(20261012)
    inner = torch.randn((16, 32), generator=generator)
    codebook = torch.randn((64, 2), generator=generator)
    diagonal = torch.linspace(0.1, 4.0, 16)
    H = torch.diag(diagonal)

    quantized, states = block_ldlq_inner(
        inner,
        H,
        codebook,
        bits=2,
        viterbi_objective="hessian_diagonal",
    )
    sequences = inner.reshape(16, 2, 16).permute(1, 0, 2).reshape(2, 128, 2)
    step_weights = (diagonal / diagonal.mean()).repeat_interleave(8).unsqueeze(0).expand(2, -1)
    expected = tail_biting_viterbi_quantize(
        sequences,
        codebook,
        bits=2,
        step_weights=step_weights,
    )

    assert torch.equal(states, expected.states)
    assert torch.equal(
        quantized,
        expected.values.reshape(2, 16, 16).permute(1, 0, 2).reshape(16, 32),
    )


def test_block_ldlq_hessian_diagonal_rejects_unknown_objective():
    with pytest.raises(ValueError, match="objective"):
        block_ldlq_inner(
            torch.eye(16),
            torch.eye(16),
            torch.randn((64, 2)),
            bits=2,
            viterbi_objective="full_hessian",
        )


@pytest.mark.parametrize(
    ("inner", "H", "codebook", "kwargs", "exception", "message"),
    (
        (
            torch.empty((0, 16)),
            torch.empty((0, 0)),
            torch.randn((64, 2)),
            {},
            ValueError,
            "positive",
        ),
        (
            torch.eye(16),
            torch.eye(16),
            torch.randn((64, 2)),
            {"tile_rows": 0},
            ValueError,
            "tile rows",
        ),
        (
            torch.eye(16),
            torch.eye(16),
            torch.randn((64, 2)),
            {"tile_cols": True},
            ValueError,
            "tile columns",
        ),
        (
            torch.eye(16),
            torch.eye(16),
            torch.empty((0, 2)),
            {},
            ValueError,
            "non-empty matrix",
        ),
        (
            torch.eye(16),
            torch.eye(16),
            torch.zeros((64, 2), dtype=torch.int32),
            {},
            TypeError,
            "floating-point",
        ),
        (
            torch.eye(16),
            torch.eye(16),
            torch.full((64, 2), torch.nan),
            {},
            ValueError,
            "finite",
        ),
        (
            torch.eye(16),
            torch.eye(16),
            torch.randn((64, 2)),
            {"tail_biting_candidates": 0},
            ValueError,
            "candidate count",
        ),
    ),
)
def test_block_ldlq_rejects_malformed_boundary_inputs(inner, H, codebook, kwargs, exception, message):
    with pytest.raises(exception, match=message):
        block_ldlq_inner(inner, H, codebook, bits=2, **kwargs)


def test_block_ldlq_hessian_diagonal_rejects_degenerate_conditioned_metric(monkeypatch):
    monkeypatch.setattr(
        qvq_module,
        "block_ldl_factor",
        lambda H, block_size: (torch.eye(H.shape[0]), torch.zeros_like(H)),
    )
    with pytest.raises(RuntimeError, match="positive finite mean"):
        block_ldlq_inner(
            torch.eye(16),
            torch.eye(16),
            torch.randn((64, 2)),
            bits=2,
            viterbi_objective="hessian_diagonal",
        )


def test_randomized_hadamard_preprocess_round_trip_and_hessian_match_explicit_basis():
    generator = torch.Generator().manual_seed(20261003)
    weight = torch.randn((32, 16), generator=generator)
    source = torch.randn((16, 16), generator=generator)
    H = source @ source.T
    SU = torch.randint(0, 2, (16,), generator=generator).mul(2).sub(1).float()
    SV = torch.randint(0, 2, (32,), generator=generator).mul(2).sub(1).float()

    inner = rht_preprocess_weight(weight, SU, SV)
    reconstructed = rht_reconstruct_weight(inner, SU, SV)
    basis = matmul_hadU(torch.eye(16))
    expected_H = basis.T @ (H * SU.unsqueeze(0) * SU.unsqueeze(1)) @ basis

    torch.testing.assert_close(reconstructed, weight, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(rht_preprocess_hessian(H, SU), expected_H, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("bits", [1, 1.5, 2, 2.5, 4, 7.5, 8])
def test_qvq_linear_quantization_produces_pack_ready_runtime_parity(bits):
    generator = torch.Generator().manual_seed(20261100 + qvq_transition_bits(bits))
    weight = torch.randn((16, 16), generator=generator) * 0.1
    activations = torch.randn((64, 16), generator=generator)
    H = activations.T @ activations / activations.shape[0]
    bias = torch.randn((16,), generator=generator)

    result = quantize_qvq_linear(
        weight,
        H,
        bits=bits,
        bias=bias,
        seed=13,
        trellis_batch_size=1,
    )
    layer = QVQReferenceLinear(
        bits=bits,
        in_features=16,
        out_features=16,
        name="proj",
        tensors={
            "trellis": result.trellis,
            "SU": result.SU,
            "SV": result.SV,
            "bias": result.bias,
        },
        out_dtype=torch.float32,
    )
    x = torch.randn((4, 16), generator=generator)

    torch.testing.assert_close(
        result.inner_weight,
        reconstruct_qvq_inner_weight(
            result.trellis,
            bits=bits,
            in_features=16,
            out_features=16,
        ),
    )
    expected = x @ result.weight.T + bias
    layer.train()
    torch.testing.assert_close(layer(x), expected, atol=2e-5, rtol=2e-5)
    layer.eval()
    torch.testing.assert_close(layer(x), expected, atol=2e-5, rtol=2e-5)
    expected_proxy = torch.trace((result.weight - weight) @ H @ (result.weight - weight).T)
    torch.testing.assert_close(result.proxy_loss, expected_proxy)
    torch.testing.assert_close(result.baseline_proxy_loss, expected_proxy)
    assert result.output_scale_optimized_channels == 0
    assert result.module_scale_search_selected is False
    assert result.module_scale_multiplier == 1.0
    assert result.module_scale_reencoded is False
    assert result.rounding == "block_ldlq"
    assert result.kronecker_proxy_loss is None


def test_qvq_v4_four_bank_quantization_persists_rate_keyed_selectors():
    generator = torch.Generator().manual_seed(20261201)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    activations = torch.randn((32, 16), generator=generator)
    H = activations.T @ activations / activations.shape[0]
    result = quantize_qvq_linear(
        weight,
        H,
        bits=2,
        vector_size=4,
        bank_count=4,
        seed=7,
        trellis_batch_size=1,
    )
    assert result.bank_ids is not None
    assert result.bank_ids.shape == (1,)
    assert result.bank_ids.dtype == torch.uint8
    assert int(result.bank_ids.item()) in range(PGC16_V4_BANK_COUNT)
    payload = result.serialized_tensors()
    assert payload["bank_ids"].numel() == 1
    assert torch.equal(unpack_qvq_bank_ids(payload["bank_ids"], 1), result.bank_ids)
    reconstructed = reconstruct_qvq_inner_weight(
        result.trellis,
        bits=2,
        in_features=16,
        out_features=16,
        vector_size=4,
        bank_ids=result.bank_ids,
    )
    torch.testing.assert_close(reconstructed, result.inner_weight)
    layer = QVQLinear(
        bits=2,
        in_features=16,
        out_features=16,
        name="banked",
        tensors=result.serialized_tensors(),
        vector_size=4,
        bank_count=4,
        out_dtype=torch.float32,
    )
    layer.eval()
    x = torch.randn((3, 16), generator=generator)
    torch.testing.assert_close(layer(x), x @ result.weight.T, atol=2e-5, rtol=2e-5)


def test_qvq_v4_four_bank_module_rejects_missing_selectors():
    with pytest.raises(ValueError, match="require serialized bank_ids"):
        QVQLinear(
            bits=2,
            in_features=16,
            out_features=16,
            vector_size=4,
            bank_count=4,
            tensors={
                "trellis": torch.zeros((1, 128), dtype=torch.int32),
                "SU": torch.ones(16),
                "SV": torch.ones(16),
            },
        )


def test_qvq_v4_module_rejects_removed_bank_selectors_after_validation():
    tensors = {
        "trellis": torch.zeros((1, 16), dtype=torch.int32),
        "SU": torch.ones(16),
        "SV": torch.ones(16),
        "bank_ids": torch.zeros(1, dtype=torch.uint8),
    }
    layer = QVQLinear(
        bits=2,
        in_features=16,
        out_features=16,
        vector_size=4,
        bank_count=4,
        tensors=tensors,
    ).eval()
    layer(torch.zeros((1, 16)))
    layer.bank_ids = None
    with pytest.raises(RuntimeError, match="bank_ids"):
        layer(torch.zeros((1, 16)))


def test_qvq_linear_empty_leading_batch_preserves_output_shape_and_dtype():
    layer = QVQLinear(
        bits=2,
        in_features=16,
        out_features=32,
        tensors={
            "trellis": torch.zeros((2, 16), dtype=torch.int32),
            "SU": torch.ones(16),
            "SV": torch.ones(32),
        },
    ).eval()
    output = layer(torch.empty((0, 3, 16), dtype=torch.float16))
    assert output.shape == (0, 3, 32)
    assert output.dtype == torch.float16


def test_qvq_v4_from_tensors_preserves_bank_count_and_allows_loader_shell():
    shell = QVQLinear(bits=2, in_features=16, out_features=16, vector_size=4, bank_count=4, tensors={})
    assert shell.bank_count == 4
    tensors = {
        "trellis": torch.zeros((1, 16), dtype=torch.int32),
        "SU": torch.ones(16),
        "SV": torch.ones(16),
        "bank_ids": torch.zeros(1, dtype=torch.uint8),
    }
    loaded = QVQLinear.from_tensors(
        bits=2,
        in_features=16,
        out_features=16,
        name="loaded",
        tensors=tensors,
        vector_size=4,
        bank_count=4,
    )
    assert loaded.bank_count == 4
    with pytest.raises(RuntimeError, match="bank_ids"):
        shell.load_state_dict({"trellis": tensors["trellis"], "SU": tensors["SU"], "SV": tensors["SV"]}, strict=True)
    assert shell.load_state_dict(tensors, strict=True).missing_keys == []


def test_qvq_v4_banked_yaqa_returns_selectors_and_preserves_reconstruction_shape():
    generator = torch.Generator().manual_seed(20261203)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    input_hessian = torch.eye(16)
    output_hessian = torch.eye(16)
    banks = tuple(pgc16_codebook_v4_bank(bank, bits=2) for bank in range(4))
    reconstructed, states, bank_ids = yaqa_inner(
        weight,
        input_hessian,
        output_hessian,
        banks[0],
        bits=2,
        trellis_batch_size=1,
        bank_codebooks=banks,
    )
    assert reconstructed.shape == weight.shape
    assert states.shape == (1, 64)
    assert bank_ids.shape == (1,)
    assert bank_ids.dtype == torch.uint8
    assert int(bank_ids.item()) in range(4)
    canonical, _ = yaqa_inner(
        weight, input_hessian, output_hessian, banks[0], bits=2, trellis_batch_size=1
    )
    mixed_error = reconstructed - weight
    canonical_error = canonical - weight
    mixed_loss = torch.einsum("ij,ik,kl,lj->", mixed_error, input_hessian, mixed_error, output_hessian)
    canonical_loss = torch.einsum("ij,ik,kl,lj->", canonical_error, input_hessian, canonical_error, output_hessian)
    assert mixed_loss <= canonical_loss


def test_qvq_v4_yaqa_bank_stack_requires_shared_bank_storage():
    weight = torch.zeros((16, 16))
    hessian = torch.eye(16)
    storage = torch.stack(tuple(pgc16_codebook_v4_bank(bank, bits=2) for bank in range(4))).contiguous()
    banks = tuple(storage[bank] for bank in range(4))
    yaqa_inner(
        weight,
        hessian,
        hessian,
        banks[0],
        bits=2,
        bank_codebooks=banks,
        bank_codebook_stack=storage,
    )
    canonical_stack = storage.clone()
    with pytest.raises(ValueError, match="share storage"):
        yaqa_inner(
            weight,
            hessian,
            hessian,
            banks[0],
            bits=2,
            bank_codebooks=banks,
            bank_codebook_stack=canonical_stack,
        )


def test_qvq_v4_block_ldlq_bank_stack_requires_shared_bank_storage():
    weight = torch.zeros((16, 16))
    hessian = torch.eye(16)
    storage = torch.stack(tuple(pgc16_codebook_v4_bank(bank, bits=2) for bank in range(4))).contiguous()
    banks = tuple(storage[bank] for bank in range(4))
    with pytest.raises(ValueError, match="share storage"):
        block_ldlq_inner_banked(
            weight,
            hessian,
            banks,
            bits=2,
            trellis_batch_size=1,
            bank_codebook_stack=storage.clone(),
        )


def test_qvq_v4_banked_yaqa_scores_feedback_corrected_multitile_inputs():
    generator = torch.Generator().manual_seed(20261204)
    weight = torch.randn((32, 32), generator=generator) * 0.1
    input_basis = torch.randn((32, 32), generator=generator)
    output_basis = torch.randn((32, 32), generator=generator)
    input_hessian = input_basis @ input_basis.T + torch.eye(32) * 0.1
    output_hessian = output_basis @ output_basis.T + torch.eye(32) * 0.1
    banks = tuple(pgc16_codebook_v4_bank(bank, bits=2) for bank in range(4))
    reconstructed, states, bank_ids = yaqa_inner(
        weight,
        input_hessian,
        output_hessian,
        banks[0],
        bits=2,
        trellis_batch_size=1,
        bank_codebooks=banks,
    )
    assert reconstructed.shape == weight.shape
    assert states.shape == (4, 64)
    assert bank_ids.shape == (4,)
    assert torch.isfinite(reconstructed).all()
    canonical, _ = yaqa_inner(
        weight,
        input_hessian,
        output_hessian,
        banks[0],
        bits=2,
        trellis_batch_size=1,
    )
    mixed_error = reconstructed.to(torch.float32) - weight
    canonical_error = canonical.to(torch.float32) - weight
    mixed_loss = torch.einsum("ij,ik,kl,lj->", mixed_error, input_hessian, mixed_error, output_hessian)
    canonical_loss = torch.einsum("ij,ik,kl,lj->", canonical_error, input_hessian, canonical_error, output_hessian)
    assert torch.isfinite(mixed_loss)
    assert mixed_loss <= canonical_loss


def test_qvq_v4_banked_yaqa_lifecycle_passes_banks_and_preserves_dtype(monkeypatch):
    captured = []

    def fake_yaqa(inner_weight, input_hessian, output_hessian, codebook, **kwargs):
        captured.append(kwargs["bank_codebooks"])
        assert len(captured[-1]) == 4
        assert all(bank.shape == (1 << 16, 4) for bank in captured[-1])
        steps = inner_weight.numel() // 4
        return (
            torch.zeros_like(inner_weight),
            torch.zeros((steps,), dtype=torch.long, device=inner_weight.device),
            torch.zeros((1,), dtype=torch.uint8, device=inner_weight.device),
        )

    monkeypatch.setattr(qvq_module, "yaqa_inner", fake_yaqa)
    weight = torch.randn((16, 16), dtype=torch.float16)
    input_hessian = torch.eye(16, dtype=torch.float64)
    output_hessian = torch.eye(16, dtype=torch.float64)
    result = quantize_qvq_linear(
        weight,
        input_hessian,
        output_hessian=output_hessian,
        bits=2,
        rounding="yaqa",
        vector_size=4,
        bank_count=4,
        trellis_batch_size=1,
    )
    assert len(captured) == 1
    assert result.bank_ids is not None
    assert result.weight.dtype == weight.dtype
    # RHT/YAQA keeps the inner reconstruction in FP32; the public dense
    # reconstruction is converted back to the source module dtype.
    assert result.inner_weight.dtype == torch.float32


def test_qvq_banked_linear_deepcopy_drops_transient_cuda_selector_cache():
    tensors = {
        "trellis": torch.zeros((1, 16), dtype=torch.int32),
        "SU": torch.ones(16),
        "SV": torch.ones(16),
        "bank_ids": torch.zeros(1, dtype=torch.uint8),
    }
    layer = QVQLinear.from_tensors(
        bits=2,
        in_features=16,
        out_features=16,
        name="copy",
        tensors=tensors,
        vector_size=4,
        bank_count=4,
    )
    layer._qvq_cuda_bank_cache = (layer.bank_ids, 0, torch.device("cpu"), torch.zeros(1))
    cloned = copy.deepcopy(layer)
    assert cloned.bank_count == 4
    assert cloned._qvq_cuda_bank_cache is None
    assert cloned._qvq_cuda_bank_cache_lock is not layer._qvq_cuda_bank_cache_lock
    buffer = io.BytesIO()
    torch.save(layer, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    assert restored.bank_count == 4
    assert restored._qvq_cuda_bank_cache is None


def test_qvq_propagated_bank_gate_is_opt_in_and_accepts_only_heldout_improvement():
    config = QVQConfig(format=FORMAT.QVQ_V4, vector_size=4, bits=2, bank_count=4, rounding="block_ldlq")
    assert config.propagated_bank_selection is None
    config.propagated_bank_selection = False
    config.__post_init__()
    assert config.propagated_bank_selection is False
    config.propagated_bank_selection = True
    config.__post_init__()
    assert config.to_dict()["propagated_bank_selection"] is True
    with pytest.raises(ValueError, match="propagated inputs"):
        quantize_qvq_linear(
            torch.randn((16, 16)),
            torch.eye(16),
            bits=2,
            vector_size=4,
            bank_count=4,
            propagated_inputs=torch.randn((4, 15)),
            propagated_target_output=torch.randn((4, 16)),
        )


def test_qvq_propagated_inner_space_residual_matches_output_space():
    """The fast selector's transformed residual must equal dense output error."""
    generator = torch.Generator().manual_seed(20260813)
    inner = torch.randn((16, 16), generator=generator)
    su = torch.rand(16, generator=generator) + 0.25
    sv = torch.rand(16, generator=generator) + 0.25
    dense = rht_reconstruct_weight(inner, su, sv)
    candidate_inner = inner + torch.randn((16, 16), generator=generator) * 0.1
    candidate_dense = rht_reconstruct_weight(candidate_inner, su, sv)
    inputs = torch.randn((7, 16), generator=generator)
    teacher = inputs @ dense.T
    transformed_inputs = matmul_hadU(inputs * su)
    transformed_target = matmul_hadU(teacher / sv, transpose=True)
    torch.testing.assert_close(
        transformed_target - transformed_inputs @ candidate_inner,
        matmul_hadU((teacher - inputs @ candidate_dense.T) / sv, transpose=True),
        rtol=2e-5,
        atol=2e-5,
    )


def test_qvq_propagated_block_ldlq_starts_from_canonical_bank_zero(monkeypatch):
    generator = torch.Generator().manual_seed(20260822)
    weight = torch.randn((16, 16), generator=generator)
    inputs = torch.randn((8, 16), generator=generator)

    def unexpected_mixed_local_search(*args, **kwargs):
        raise AssertionError("propagated Block-LDLQ must not run the redundant mixed local search")

    monkeypatch.setattr(qvq_module, "block_ldlq_inner_banked", unexpected_mixed_local_search)
    result = quantize_qvq_linear(
        weight,
        torch.eye(16),
        bits=2,
        vector_size=4,
        bank_count=4,
        propagated_inputs=inputs,
        propagated_target_output=inputs @ weight.T,
        propagated_acceptance=lambda proposal, baseline: True,
        trellis_batch_size=1,
    )
    assert torch.isfinite(result.inner_weight).all()
    assert result.bank_ids is not None


def test_qvq_propagation_refiner_evaluates_serialized_candidates_conditionally():
    """Deferred selection must evaluate every exact candidate and roll back rejects."""
    state = {"active": "baseline", "installs": [], "restores": 0}
    scores = {"baseline": 10.0, "local": 12.0, "downstream": 7.0, "worse": 20.0}

    def evaluate():
        return scores[state["active"]]

    def install(candidate):
        state["installs"].append(candidate.module_name)
        serialized = candidate.serialized_tensors()
        assert set(serialized) == {"trellis", "SU", "SV"}
        state["active"] = candidate.local_metrics["key"]

    def snapshot():
        return state["active"]

    def restore(checkpoint):
        state["active"] = checkpoint
        state["restores"] += 1

    base = torch.zeros((1, 1))
    candidates = {
        "layer.0": [
            QVQCandidate("layer.0", base, None, base, base, base, local_metrics={"key": "local"}),
            QVQCandidate("layer.0", base, None, base, base, base, local_metrics={"key": "downstream"}),
            QVQCandidate("layer.0", base, None, base, base, base, local_metrics={"key": "worse"}),
        ]
    }
    score, selected = QVQPropagationRefiner(evaluate, install, snapshot, restore).refine(candidates)
    assert score == 7.0
    assert [candidate.local_metrics["key"] for candidate in selected] == ["downstream"]
    assert state["active"] == "downstream"
    assert state["restores"] == 2

    def failing_evaluate():
        raise RuntimeError("replay failed")

    with pytest.raises(RuntimeError, match="replay failed"):
        QVQPropagationRefiner(failing_evaluate, install, snapshot, restore).refine(candidates)
    assert state["active"] == "downstream"


def test_qvq_propagated_residual_gather_keeps_all_samples_and_tiles():
    """Per-tile bank choices must gather deltas without mixing sample rows."""
    generator = torch.Generator().manual_seed(20260814)
    samples, output_tiles, width = 5, 3, 4
    all_delta = torch.randn((5, samples, output_tiles, width), generator=generator)
    residual = torch.randn((samples, output_tiles, width), generator=generator)
    winners = torch.tensor([0, 3, 1])
    tile_indices = torch.arange(output_tiles)
    selected = all_delta[winners, :, tile_indices, :].permute(1, 0, 2)
    expected = torch.stack([all_delta[winners[t], :, t, :] for t in range(output_tiles)], dim=1)
    torch.testing.assert_close(selected, expected)
    updated = residual - selected
    loop_updated = torch.stack(
        [residual[:, t, :] - all_delta[winners[t], :, t, :] for t in range(output_tiles)], dim=1
    )
    torch.testing.assert_close(updated, loop_updated)


def test_qvq_propagated_zero_delta_ties_always_keep_no_change():
    """The vectorized scorer must preserve exact baseline tie precedence."""
    generator = torch.Generator().manual_seed(20260815)
    residual = torch.randn((1000, 3, 16), generator=generator)
    delta = torch.zeros((4, 1000, 3, 16))
    no_change = torch.zeros_like(delta[:1])
    losses = torch.cat(
        (
            (residual.unsqueeze(0) - no_change).square().sum(dim=(1, 3)),
            (residual.unsqueeze(0) - delta).square().sum(dim=(1, 3)),
        ),
        dim=0,
    )
    assert torch.equal(losses.argmin(dim=0), torch.zeros(3, dtype=torch.long))


def test_qvq_propagated_zero_delta_tie_preserves_serialized_baseline(monkeypatch):
    """Exact bank ties must leave the production trellis and selectors unchanged."""
    torch.manual_seed(20260816)
    weight = torch.randn((32, 16), dtype=torch.float32)
    inputs = torch.randn((8, 16), dtype=torch.float32)
    targets = inputs @ weight.T
    original = qvq_module.block_ldlq_inner_banked_candidates

    def identical_candidates(*args, **kwargs):
        candidate_weight, candidate_states = original(*args, **kwargs)
        return candidate_weight[:1].expand(4, -1, -1).clone(), candidate_states[:1].expand(4, -1, -1).clone()

    monkeypatch.setattr(qvq_module, "block_ldlq_inner_banked_candidates", identical_candidates)
    baseline = quantize_qvq_linear(weight, torch.eye(16), bits=2, vector_size=4, bank_count=4)
    replay_calls = []
    tied = quantize_qvq_linear(
        weight,
        torch.eye(16),
        bits=2,
        vector_size=4,
        bank_count=4,
        propagated_inputs=inputs,
        propagated_target_output=targets,
        propagated_acceptance=lambda proposal, reference: replay_calls.append(True) or True,
        trellis_batch_size=1,
    )
    assert not replay_calls
    torch.testing.assert_close(tied.trellis, baseline.trellis, rtol=0, atol=0)
    torch.testing.assert_close(tied.bank_ids, baseline.bank_ids, rtol=0, atol=0)


def test_qvq_propagated_delta_cache_budget_fallback_is_exact(monkeypatch):
    generator = torch.Generator().manual_seed(20260817)
    weight = torch.randn((32, 16), generator=generator)
    inputs = torch.randn((8, 16), generator=generator)
    targets = inputs @ weight.T
    kwargs = {
        "bits": 2,
        "vector_size": 4,
        "bank_count": 4,
        "propagated_inputs": inputs,
        "propagated_target_output": targets,
        "propagated_acceptance": lambda proposal, reference: True,
        "trellis_batch_size": 1,
    }
    cached = quantize_qvq_linear(weight, torch.eye(16), **kwargs)
    monkeypatch.setattr(qvq_module, "_QVQ_PROPAGATION_DELTA_CACHE_BYTES", 0)
    recomputed = quantize_qvq_linear(weight, torch.eye(16), **kwargs)
    torch.testing.assert_close(recomputed.trellis, cached.trellis, rtol=0, atol=0)
    torch.testing.assert_close(recomputed.bank_ids, cached.bank_ids, rtol=0, atol=0)


def test_qvq_propagated_bank_gate_acceptance_and_rejection_are_serialization_atomic():
    torch.manual_seed(20260813)
    weight = torch.randn((32, 16), dtype=torch.float16)
    inputs = torch.randn((8, 16), dtype=torch.bfloat16)
    targets = (inputs.float() @ weight.float().T).to(torch.bfloat16)
    accepted_calls = []
    accepted = quantize_qvq_linear(
        weight,
        torch.eye(16),
        bits=2,
        vector_size=4,
        bank_count=4,
        propagated_inputs=inputs,
        propagated_target_output=targets,
        propagated_acceptance=lambda proposal, baseline: accepted_calls.append((proposal, baseline)) or True,
        trellis_batch_size=1,
    )
    # This seed produces an unchanged serialized proposal; the expensive
    # downstream callback must not run for a no-op.
    assert not accepted_calls
    decoded = reconstruct_qvq_inner_weight(
        accepted.trellis,
        bits=2,
        vector_size=4,
        in_features=16,
        out_features=32,
        bank_ids=accepted.bank_ids,
    )
    torch.testing.assert_close(decoded, accepted.inner_weight, rtol=0, atol=0)
    rejected_calls = []
    rejected = quantize_qvq_linear(
        weight,
        torch.eye(16),
        bits=2,
        vector_size=4,
        bank_count=4,
        propagated_inputs=inputs,
        propagated_target_output=targets,
        propagated_acceptance=lambda proposal, baseline: rejected_calls.append((proposal, baseline)) or False,
        trellis_batch_size=1,
    )
    assert not rejected_calls
    decoded_rejected = reconstruct_qvq_inner_weight(
        rejected.trellis,
        bits=2,
        vector_size=4,
        in_features=16,
        out_features=32,
        bank_ids=rejected.bank_ids,
    )
    torch.testing.assert_close(decoded_rejected, rejected.inner_weight, rtol=0, atol=0)
    baseline = quantize_qvq_linear(weight, torch.eye(16), bits=2, vector_size=4, bank_count=4, trellis_batch_size=1)
    torch.testing.assert_close(rejected.weight, baseline.weight, rtol=0, atol=0)


@pytest.mark.parametrize("hessian_dtype", (torch.float16, torch.bfloat16, torch.float64))
def test_qvq_v4_banked_yaqa_rolls_back_atomically_to_bank_zero(hessian_dtype):
    generator = torch.Generator().manual_seed(20261205)
    weight = torch.randn((16, 32), generator=generator) * 0.1
    input_hessian = torch.eye(16, dtype=hessian_dtype)
    output_hessian = torch.eye(32, dtype=hessian_dtype)
    bank0 = pgc16_codebook_v4_bank(0, bits=2)
    banks = (bank0, bank0.clone(), bank0.clone(), bank0.clone())
    mixed, mixed_states, bank_ids = yaqa_inner(
        weight,
        input_hessian,
        output_hessian,
        bank0,
        bits=2,
        trellis_batch_size=1,
        bank_codebooks=banks,
    )
    canonical, canonical_states = yaqa_inner(
        weight,
        input_hessian,
        output_hessian,
        bank0,
        bits=2,
        trellis_batch_size=1,
    )
    torch.testing.assert_close(mixed, canonical, rtol=0, atol=0)
    torch.testing.assert_close(mixed_states, canonical_states, rtol=0, atol=0)
    assert torch.equal(bank_ids, torch.zeros_like(bank_ids))


@pytest.mark.parametrize("bits", (1, 1.5, 2, 3, 4, 5, 6, 7, 8))
def test_qvq_yaqa_linear_quantization_preserves_planar_runtime_and_reports_full_proxy(
    bits,
):
    generator = torch.Generator().manual_seed(20261120 + int(bits * 2))
    weight = torch.randn((32, 16), generator=generator) * 0.1
    activations = torch.randn((64, 16), generator=generator)
    output_source = torch.randn((32, 32), generator=generator)
    input_hessian = activations.T @ activations / activations.shape[0]
    output_hessian = output_source @ output_source.T / output_source.shape[0]
    bias = torch.randn((32,), generator=generator)

    result = quantize_qvq_linear(
        weight,
        input_hessian,
        output_hessian=output_hessian,
        bits=bits,
        bias=bias,
        seed=31,
        trellis_batch_size=1,
        tail_biting_candidates=2,
        rounding="yaqa",
    )
    layer = QVQReferenceLinear(
        bits=bits,
        in_features=16,
        out_features=32,
        name="proj",
        tensors={
            "trellis": result.trellis,
            "SU": result.SU,
            "SV": result.SV,
            "bias": result.bias,
        },
        out_dtype=torch.float32,
    )
    x = torch.randn((8, 16), generator=generator)
    expected = x @ result.weight.T + bias
    actual = layer(x)
    forward_kl = F.kl_div(
        actual.log_softmax(dim=-1),
        expected.softmax(dim=-1),
        reduction="batchmean",
    )

    assert result.rounding == "yaqa"
    assert result.kronecker_proxy_loss is not None
    torch.testing.assert_close(
        result.kronecker_proxy_loss,
        yaqa_proxy_loss(weight, result.weight, input_hessian, output_hessian),
    )
    torch.testing.assert_close(
        result.inner_weight,
        reconstruct_qvq_inner_weight(
            result.trellis,
            bits=bits,
            in_features=16,
            out_features=32,
        ),
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert forward_kl < 2e-6
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))
    assert torch.equal(actual.topk(5, dim=-1).indices, expected.topk(5, dim=-1).indices)
    assert result.trellis.numel() * 32 == weight.numel() * qvq_transition_bits(bits) // 2


def test_qvq_yaqa_identity_output_hessian_matches_block_ldlq_after_rht():
    generator = torch.Generator().manual_seed(20261121)
    weight = torch.randn((32, 16), generator=generator) * 0.1
    activations = torch.randn((48, 16), generator=generator)
    input_hessian = activations.T @ activations / activations.shape[0]
    kwargs = {
        "bits": 2,
        "seed": 37,
        "damp_percent": YAQA_PAPER_REGULARIZATION,
        "trellis_batch_size": 1,
        "tail_biting_candidates": 2,
    }

    baseline = quantize_qvq_linear(weight, input_hessian, **kwargs)
    yaqa = quantize_qvq_linear(
        weight,
        input_hessian,
        output_hessian=torch.eye(32),
        rounding="yaqa",
        **kwargs,
    )

    assert torch.equal(yaqa.trellis, baseline.trellis)
    assert torch.equal(yaqa.inner_weight, baseline.inner_weight)
    assert torch.equal(yaqa.weight, baseline.weight)
    torch.testing.assert_close(yaqa.kronecker_proxy_loss, baseline.proxy_loss)


@pytest.mark.parametrize("regularization", (None, 0.01))
def test_qvq_yaqa_uses_default_or_explicit_damping(monkeypatch, regularization):
    weight = torch.arange(256, dtype=torch.float32).reshape(16, 16).div(100)
    input_hessian = torch.diag(torch.linspace(1.0, 2.0, 16))
    output_hessian = torch.diag(torch.linspace(2.0, 4.0, 16))
    captured = []

    def capture_factors(inner_weight, input_factor, output_factor, codebook, **kwargs):
        del codebook, kwargs
        captured.append((input_factor.clone(), output_factor.clone()))
        return torch.zeros_like(inner_weight), torch.zeros((1, 128), dtype=torch.long)

    monkeypatch.setattr(qvq_module, "yaqa_inner", capture_factors)
    kwargs = {} if regularization is None else {"damp_percent": regularization}
    result = quantize_qvq_linear(
        weight,
        input_hessian,
        output_hessian=output_hessian,
        bits=2,
        seed=47,
        trellis_batch_size=1,
        rounding="yaqa",
        **kwargs,
    )

    expected_regularization = 0.05 if regularization is None else regularization
    expected_input = rht_preprocess_hessian(input_hessian, result.SU)
    expected_input = (expected_input + expected_input.T) * 0.5
    output_sign = result.SV.sign()
    expected_output = rht_preprocess_hessian(output_hessian, output_sign)
    expected_output = (expected_output + expected_output.T) * 0.5
    expected_input.diagonal().add_(expected_input.diagonal().mean() * expected_regularization)
    expected_output.diagonal().add_(expected_output.diagonal().mean() * expected_regularization)

    torch.testing.assert_close(captured[0][0], expected_input)
    torch.testing.assert_close(captured[0][1], expected_output)


def test_qvq_output_channel_scale_closed_form_matches_independent_row_solution():
    generator = torch.Generator().manual_seed(20261104)
    inner = torch.randn((16, 16), generator=generator)
    SU = torch.randint(0, 2, (16,), generator=generator).mul(2).sub(1).float()
    SV = torch.randint(0, 2, (16,), generator=generator).mul(2).sub(1).float() * 0.25
    current = rht_reconstruct_weight(inner, SU, SV)
    expected_correction = torch.linspace(0.5, 1.5, 16)
    weight = current * expected_correction.unsqueeze(1)
    source = torch.randn((16, 16), generator=generator)
    H = source.T @ source + torch.eye(16) * 0.25

    optimized_SV, optimized_weight, optimized_loss, optimized_channels = optimize_qvq_output_channel_scales(
        weight,
        inner,
        H,
        SU,
        SV,
    )

    torch.testing.assert_close(optimized_SV, SV * expected_correction, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(optimized_weight, weight, atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(optimized_loss, torch.zeros_like(optimized_loss), atol=2e-9, rtol=0)
    assert optimized_channels == 16


def test_qvq_output_channel_scale_shrinkage_matches_exact_convex_correction():
    generator = torch.Generator().manual_seed(20260828)
    inner = torch.randn((16, 16), generator=generator)
    SU = torch.randint(0, 2, (16,), generator=generator).mul(2).sub(1).float()
    SV = torch.randint(0, 2, (16,), generator=generator).mul(2).sub(1).float() * 0.25
    current = rht_reconstruct_weight(inner, SU, SV)
    full_correction = torch.linspace(0.25, 1.75, 16)
    weight = current * full_correction.unsqueeze(1)
    source = torch.randn((32, 16), generator=generator)
    H = source.T @ source + torch.eye(16) * 0.25

    optimized_SV, optimized_weight, optimized_loss, optimized_channels = optimize_qvq_output_channel_scales(
        weight,
        inner,
        H,
        SU,
        SV,
        correction_strength=0.5,
    )

    expected_correction = 1 + 0.5 * (full_correction - 1)
    torch.testing.assert_close(optimized_SV, SV * expected_correction, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(optimized_weight, current * expected_correction.unsqueeze(1), atol=5e-6, rtol=5e-6)
    assert optimized_loss < qvq_proxy_loss(weight, current, H)
    assert optimized_channels == 16


def test_qvq_output_channel_scale_uses_regularized_fit_but_original_hessian_acceptance():
    generator = torch.Generator().manual_seed(20261110)
    inner = torch.randn((16, 16), generator=generator)
    SU = torch.ones(16)
    SV = torch.ones(16) * 0.2
    current = rht_reconstruct_weight(inner, SU, SV)
    weight = current.clone()
    weight[:, 0] *= 3
    H = torch.diag(torch.tensor([100.0] + [0.0] * 15))
    optimization_H = H + torch.eye(16) * 100

    optimized_SV, optimized_weight, optimized_loss, optimized_channels = optimize_qvq_output_channel_scales(
        weight,
        inner,
        H,
        SU,
        SV,
        optimization_H=optimization_H,
    )

    current_product = current @ optimization_H
    expected_correction = (current_product * weight).sum(1) / (current_product * current).sum(1)
    torch.testing.assert_close(optimized_SV, SV * expected_correction)
    assert torch.all((optimized_SV / SV - 1).abs() < 2)
    assert optimized_loss <= qvq_proxy_loss(weight, current, H)
    torch.testing.assert_close(optimized_loss, qvq_proxy_loss(weight, optimized_weight, H))
    assert optimized_channels == 16


def test_qvq_output_channel_scale_rejects_equal_loss_updates_from_unobserved_hessian():
    generator = torch.Generator().manual_seed(20261112)
    inner = torch.randn((16, 16), generator=generator)
    SU = torch.ones(16)
    SV = torch.ones(16) * 0.2
    current = rht_reconstruct_weight(inner, SU, SV)

    optimized_SV, optimized_weight, optimized_loss, optimized_channels = optimize_qvq_output_channel_scales(
        current * 2,
        inner,
        torch.zeros((16, 16)),
        SU,
        SV,
        optimization_H=torch.eye(16),
    )

    torch.testing.assert_close(optimized_SV, SV, rtol=0, atol=0)
    torch.testing.assert_close(optimized_weight, current, rtol=0, atol=0)
    torch.testing.assert_close(optimized_loss, torch.zeros_like(optimized_loss), rtol=0, atol=0)
    assert optimized_channels == 0


def test_qvq_module_scale_closed_form_matches_independent_fp64_solution():
    generator = torch.Generator().manual_seed(2026081201)
    inner = torch.randn((16, 16), generator=generator)
    SU = torch.randint(0, 2, (16,), generator=generator).mul(2).sub(1).float()
    SV = torch.randint(0, 2, (16,), generator=generator).mul(2).sub(1).float() * 0.25
    current = rht_reconstruct_weight(inner, SU, SV)
    expected_multiplier = 1.375
    weight = current * expected_multiplier
    source = torch.randn((32, 16), generator=generator, dtype=torch.float64)
    H = (source.T @ source + torch.eye(16, dtype=torch.float64) * 0.25).float()

    optimized_SV, optimized_weight, optimized_loss, multiplier, accepted = optimize_qvq_module_scale(
        weight,
        inner,
        H,
        SU,
        SV,
    )

    current_fp64 = current.double()
    expected_fp64 = ((current_fp64 @ H.double()) * weight.double()).sum() / (
        (current_fp64 @ H.double()) * current_fp64
    ).sum()
    assert accepted is True
    assert multiplier == pytest.approx(float(expected_fp64), rel=2e-6, abs=2e-6)
    torch.testing.assert_close(optimized_SV, SV * expected_multiplier, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(optimized_weight, weight, atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(optimized_loss, torch.zeros_like(optimized_loss), atol=2e-9, rtol=0)


@pytest.mark.parametrize("case", ("zero-denominator", "negative-correction", "epsilon-guard"))
def test_qvq_module_scale_invalid_or_unsafe_updates_are_exact_noops(case):
    generator = torch.Generator().manual_seed(2026081202)
    inner = torch.randn((16, 16), generator=generator)
    SU = torch.ones(16)
    SV = torch.ones(16) * 0.2
    current = rht_reconstruct_weight(inner, SU, SV)
    H = torch.eye(16)
    epsilon = None
    if case == "zero-denominator":
        inner.zero_()
        weight = torch.randn((16, 16), generator=generator)
    elif case == "negative-correction":
        weight = -current
    else:
        weight = current * 1.5
        epsilon = 1e30

    optimized_SV, optimized_weight, optimized_loss, multiplier, accepted = optimize_qvq_module_scale(
        weight,
        inner,
        H,
        SU,
        SV,
        denominator_epsilon=epsilon,
    )

    expected_weight = rht_reconstruct_weight(inner, SU, SV)
    torch.testing.assert_close(optimized_SV, SV, rtol=0, atol=0)
    torch.testing.assert_close(optimized_weight, expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(optimized_loss, qvq_proxy_loss(weight, expected_weight, H))
    assert multiplier == 1.0
    assert accepted is False


@pytest.mark.parametrize(
    ("weight", "inner", "H", "optimization_H", "epsilon", "exception", "message"),
    (
        (torch.eye(16), torch.randn((16, 32)), torch.eye(16), None, None, ValueError, "reconstruction"),
        (torch.eye(16), torch.eye(16), torch.eye(15), None, None, ValueError, "input width"),
        (torch.eye(16, dtype=torch.int64), torch.eye(16), torch.eye(16), None, None, TypeError, "floating-point"),
        (
            torch.eye(16).index_put((torch.tensor([0]), torch.tensor([0])), torch.tensor(float("nan"))),
            torch.eye(16),
            torch.eye(16),
            None,
            None,
            ValueError,
            "finite",
        ),
        (torch.eye(16), torch.eye(16), torch.eye(16), torch.eye(15), None, ValueError, "shape"),
        (
            torch.eye(16),
            torch.eye(16),
            torch.eye(16),
            torch.eye(16, dtype=torch.int64),
            None,
            TypeError,
            "floating-point",
        ),
        (
            torch.eye(16),
            torch.eye(16),
            torch.eye(16),
            torch.eye(16).index_put((torch.tensor([0]), torch.tensor([0])), torch.tensor(float("inf"))),
            None,
            ValueError,
            "finite",
        ),
        (torch.eye(16), torch.eye(16), torch.eye(16), None, True, TypeError, "real scalar"),
        (torch.eye(16), torch.eye(16), torch.eye(16), None, -1.0, ValueError, "nonnegative"),
        (torch.eye(16), torch.eye(16), torch.eye(16), None, float("inf"), ValueError, "finite"),
    ),
)
def test_qvq_module_scale_rejects_invalid_inputs(
    weight,
    inner,
    H,
    optimization_H,
    epsilon,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        optimize_qvq_module_scale(
            weight,
            inner,
            H,
            torch.ones(16),
            torch.ones(inner.shape[1]),
            optimization_H=optimization_H,
            denominator_epsilon=epsilon,
        )


@pytest.mark.parametrize("bits", (1, 2))
def test_qvq_disabled_module_scale_search_is_an_exact_control(bits):
    generator = torch.Generator().manual_seed(2026081205 + bits)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    activations = torch.randn((64, 16), generator=generator)
    H = activations.T @ activations / activations.shape[0]

    implicit = quantize_qvq_linear(weight, H, bits=bits, seed=17, trellis_batch_size=1)
    explicit = quantize_qvq_linear(
        weight,
        H,
        bits=bits,
        seed=17,
        trellis_batch_size=1,
        module_scale_search=False,
    )

    for name in ("trellis", "SU", "SV", "inner_weight", "weight", "proxy_loss", "baseline_proxy_loss"):
        torch.testing.assert_close(getattr(explicit, name), getattr(implicit, name), rtol=0, atol=0)
    assert explicit.module_scale_search_selected is False
    assert explicit.module_scale_multiplier == 1.0
    assert explicit.module_scale_reencoded is False


@pytest.mark.parametrize("bits", (1, 2))
def test_qvq_module_scale_search_w1_w2_is_format_preserving_and_proxy_nonregressing(bits):
    generator = torch.Generator().manual_seed(2026081210 + bits)
    weight = torch.randn((16, 16), generator=generator) * torch.linspace(0.03, 0.3, 16).unsqueeze(1)
    activations = torch.randn((96, 16), generator=generator) * torch.linspace(0.25, 3.0, 16)
    H = activations.T @ activations / activations.shape[0]
    baseline = quantize_qvq_linear(weight, H, bits=bits, seed=23, trellis_batch_size=1)
    searched = quantize_qvq_linear(
        weight,
        H,
        bits=bits,
        seed=23,
        trellis_batch_size=1,
        module_scale_search=True,
    )

    assert searched.trellis.shape == baseline.trellis.shape
    assert searched.trellis.dtype == torch.int32
    assert searched.SU.shape == baseline.SU.shape
    assert searched.SV.shape == baseline.SV.shape
    assert searched.proxy_loss <= baseline.proxy_loss
    torch.testing.assert_close(searched.baseline_proxy_loss, baseline.proxy_loss, rtol=0, atol=0)
    torch.testing.assert_close(searched.proxy_loss, qvq_proxy_loss(weight, searched.weight, H))
    assert searched.module_scale_search_selected is True
    assert searched.module_scale_multiplier > 0

    layer = QVQReferenceLinear(
        bits=bits,
        in_features=16,
        out_features=16,
        name="proj",
        tensors={"trellis": searched.trellis, "SU": searched.SU, "SV": searched.SV},
        out_dtype=torch.float32,
    )
    x = torch.randn((8, 16), generator=generator)
    torch.testing.assert_close(layer(x), x @ searched.weight.T, atol=2e-5, rtol=2e-5)
    assert set(layer.state_dict()) == {"trellis", "SU", "SV"}


@pytest.mark.parametrize(
    ("weight", "reconstructed", "H", "exception", "message"),
    (
        (
            torch.eye(16),
            torch.eye(16)[:15],
            torch.eye(16),
            ValueError,
            "matching rank-2",
        ),
        (torch.eye(16), torch.eye(16), torch.eye(15), ValueError, "input width"),
        (
            torch.eye(16, dtype=torch.int64),
            torch.eye(16),
            torch.eye(16),
            TypeError,
            "floating-point",
        ),
        (
            torch.eye(16),
            torch.eye(16).index_put((torch.tensor([0]), torch.tensor([0])), torch.tensor(float("nan"))),
            torch.eye(16),
            ValueError,
            "finite",
        ),
    ),
)
def test_qvq_proxy_loss_rejects_invalid_inputs(weight, reconstructed, H, exception, message):
    with pytest.raises(exception, match=message):
        qvq_proxy_loss(weight, reconstructed, H)


@pytest.mark.parametrize(
    ("weight", "inner", "H", "SU", "SV", "exception", "message"),
    (
        (
            torch.eye(16),
            torch.randn((16, 32)),
            torch.eye(16),
            torch.ones(16),
            torch.ones(32),
            ValueError,
            "reconstruction",
        ),
        (
            torch.eye(16),
            torch.eye(16),
            torch.eye(15),
            torch.ones(16),
            torch.ones(16),
            ValueError,
            "input width",
        ),
        (
            torch.eye(16, dtype=torch.int64),
            torch.eye(16),
            torch.eye(16),
            torch.ones(16),
            torch.ones(16),
            TypeError,
            "floating-point",
        ),
        (
            torch.eye(16).index_put((torch.tensor([0]), torch.tensor([0])), torch.tensor(float("nan"))),
            torch.eye(16),
            torch.eye(16),
            torch.ones(16),
            torch.ones(16),
            ValueError,
            "finite",
        ),
    ),
)
def test_qvq_output_channel_scale_rejects_invalid_inputs(weight, inner, H, SU, SV, exception, message):
    with pytest.raises(exception, match=message):
        optimize_qvq_output_channel_scales(weight, inner, H, SU, SV)


@pytest.mark.parametrize(
    ("optimization_H", "exception", "message"),
    (
        (torch.eye(15), ValueError, "shape"),
        (torch.eye(16, dtype=torch.int64), TypeError, "floating-point"),
        (
            torch.eye(16).index_put((torch.tensor([0]), torch.tensor([0])), torch.tensor(float("nan"))),
            ValueError,
            "finite",
        ),
    ),
)
def test_qvq_output_channel_scale_rejects_invalid_optimization_hessian(optimization_H, exception, message):
    with pytest.raises(exception, match=message):
        optimize_qvq_output_channel_scales(
            torch.eye(16),
            torch.eye(16),
            torch.eye(16),
            torch.ones(16),
            torch.ones(16),
            optimization_H=optimization_H,
        )


@pytest.mark.parametrize("case", ("zero-denominator", "negative-correction", "epsilon-guard"))
def test_qvq_output_channel_scale_invalid_or_unsafe_updates_are_exact_noops(case):
    generator = torch.Generator().manual_seed(20261105)
    inner = torch.randn((16, 16), generator=generator)
    SU = torch.ones(16)
    SV = torch.ones(16) * 0.2
    current = rht_reconstruct_weight(inner, SU, SV)
    H = torch.eye(16)
    epsilon = None
    if case == "zero-denominator":
        inner.zero_()
        weight = torch.randn((16, 16), generator=generator)
    elif case == "negative-correction":
        weight = -current
    else:
        weight = current * 1.5
        epsilon = 1e30

    optimized_SV, optimized_weight, optimized_loss, optimized_channels = optimize_qvq_output_channel_scales(
        weight,
        inner,
        H,
        SU,
        SV,
        denominator_epsilon=epsilon,
    )

    expected_weight = rht_reconstruct_weight(inner, SU, SV)
    torch.testing.assert_close(optimized_SV, SV, rtol=0, atol=0)
    torch.testing.assert_close(optimized_weight, expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(optimized_loss, qvq_proxy_loss(weight, expected_weight, H))
    assert optimized_channels == 0


@pytest.mark.parametrize(
    ("epsilon", "exception", "message"),
    (
        (True, TypeError, "real scalar"),
        (-1.0, ValueError, "nonnegative"),
        (float("inf"), ValueError, "finite"),
    ),
)
def test_qvq_output_channel_scale_rejects_invalid_denominator_epsilon(epsilon, exception, message):
    with pytest.raises(exception, match=message):
        optimize_qvq_output_channel_scales(
            torch.eye(16),
            torch.eye(16),
            torch.eye(16),
            torch.ones(16),
            torch.ones(16),
            denominator_epsilon=epsilon,
        )


@pytest.mark.parametrize(
    ("strength", "exception", "message"),
    (
        (True, TypeError, "real scalar"),
        (0.0, ValueError, r"\(0, 1\]"),
        (-0.5, ValueError, r"\(0, 1\]"),
        (1.01, ValueError, r"\(0, 1\]"),
        (float("nan"), ValueError, "finite"),
    ),
)
def test_qvq_output_channel_scale_rejects_invalid_correction_strength(strength, exception, message):
    with pytest.raises(exception, match=message):
        optimize_qvq_output_channel_scales(
            torch.eye(16),
            torch.eye(16),
            torch.eye(16),
            torch.ones(16),
            torch.ones(16),
            correction_strength=strength,
        )


def test_qvq_output_channel_scale_quantization_is_fixed_trellis_and_proxy_nonregressing():
    generator = torch.Generator().manual_seed(20261106)
    weight = torch.randn((16, 16), generator=generator) * torch.linspace(0.03, 0.3, 16).unsqueeze(1)
    activations = torch.randn((96, 16), generator=generator) * torch.linspace(0.25, 3.0, 16)
    H = activations.T @ activations / activations.shape[0]
    baseline = quantize_qvq_linear(weight, H, bits=2, seed=23, trellis_batch_size=1)
    optimized = quantize_qvq_linear(
        weight,
        H,
        bits=2,
        seed=23,
        trellis_batch_size=1,
        output_channel_scale_optimization=True,
    )

    assert torch.equal(optimized.trellis, baseline.trellis)
    assert torch.equal(optimized.inner_weight, baseline.inner_weight)
    torch.testing.assert_close(optimized.baseline_proxy_loss, baseline.proxy_loss, rtol=0, atol=0)
    assert optimized.proxy_loss <= optimized.baseline_proxy_loss
    assert optimized.output_scale_optimized_channels > 0
    assert torch.unique(optimized.SV.abs()).numel() > 1
    torch.testing.assert_close(optimized.proxy_loss, qvq_proxy_loss(weight, optimized.weight, H))


@pytest.mark.parametrize("bits", (1, 1.5, 2))
def test_qvq_output_channel_scale_disabled_control_is_bitwise_exact(bits):
    generator = torch.Generator().manual_seed(20260830 + int(bits * 2))
    weight = torch.randn((16, 16), generator=generator) * 0.1
    activations = torch.randn((32, 16), generator=generator)
    H = activations.T @ activations / activations.shape[0]

    default = quantize_qvq_linear(weight, H, bits=bits, seed=41, trellis_batch_size=1)
    disabled = quantize_qvq_linear(
        weight,
        H,
        bits=bits,
        seed=41,
        trellis_batch_size=1,
        output_channel_scale_optimization=False,
    )

    for field in ("trellis", "SU", "SV", "inner_weight", "weight", "proxy_loss", "baseline_proxy_loss"):
        assert torch.equal(getattr(default, field), getattr(disabled, field))
    assert disabled.output_scale_optimized_channels == 0


@pytest.mark.parametrize(("bits", "expected_strength"), ((1, 0.5), (1.5, 0.5), (2, 1.0)))
def test_qvq_quantization_uses_shrunk_output_scale_only_for_low_rates(monkeypatch, bits, expected_strength):
    generator = torch.Generator().manual_seed(20260829 + int(bits * 2))
    weight = torch.randn((16, 16), generator=generator) * 0.1
    activations = torch.randn((32, 16), generator=generator)
    H = activations.T @ activations / activations.shape[0]
    original = qvq_module.optimize_qvq_output_channel_scales
    strengths = []

    def wrapped(*args, correction_strength=1.0, **kwargs):
        strengths.append(correction_strength)
        return original(*args, correction_strength=correction_strength, **kwargs)

    monkeypatch.setattr(qvq_module, "optimize_qvq_output_channel_scales", wrapped)
    result = quantize_qvq_linear(
        weight,
        H,
        bits=bits,
        seed=31,
        trellis_batch_size=1,
        output_channel_scale_optimization=True,
    )

    assert strengths == [expected_strength]
    assert result.proxy_loss <= result.baseline_proxy_loss


def test_qvq_hessian_diagonal_viterbi_keeps_euclidean_full_hessian_control():
    generator = torch.Generator().manual_seed(20261107)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    activations = torch.randn((96, 16), generator=generator) * torch.linspace(0.1, 5.0, 16)
    activations[:, 1:] += activations[:, :-1] * 0.35
    H = activations.T @ activations / activations.shape[0]
    baseline = quantize_qvq_linear(weight, H, bits=2, seed=29, trellis_batch_size=1)
    candidate = quantize_qvq_linear(
        weight,
        H,
        bits=2,
        seed=29,
        trellis_batch_size=1,
        viterbi_objective="hessian_diagonal",
    )
    assert candidate.hessian_viterbi_candidate_relative_improvement is not None
    rejected = quantize_qvq_linear(
        weight,
        H,
        bits=2,
        seed=29,
        trellis_batch_size=1,
        viterbi_objective="hessian_diagonal",
        viterbi_minimum_proxy_improvement=(candidate.hessian_viterbi_candidate_relative_improvement + 1e-6),
    )

    torch.testing.assert_close(candidate.baseline_proxy_loss, baseline.proxy_loss, rtol=0, atol=0)
    assert candidate.proxy_loss <= baseline.proxy_loss
    torch.testing.assert_close(candidate.proxy_loss, qvq_proxy_loss(weight, candidate.weight, H))
    assert baseline.hessian_viterbi_candidate_relative_improvement is None
    assert candidate.hessian_viterbi_selected is True
    assert rejected.hessian_viterbi_selected is False
    assert rejected.hessian_viterbi_candidate_relative_improvement == pytest.approx(
        candidate.hessian_viterbi_candidate_relative_improvement,
        rel=0,
        abs=0,
    )
    assert torch.equal(rejected.trellis, baseline.trellis)
    assert torch.equal(rejected.weight, baseline.weight)


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    (
        ({"output_channel_scale_optimization": 1}, TypeError, "must be boolean"),
        ({"module_scale_search": 1}, TypeError, "must be boolean"),
        (
            {"viterbi_objective": "full_hessian"},
            ValueError,
            "euclidean.*hessian_diagonal",
        ),
        ({"tail_biting_candidates": 0}, ValueError, "positive integer"),
        ({"rounding": 1}, TypeError, "rounding.*string"),
        ({"rounding": "gptq"}, ValueError, "block_ldlq.*yaqa"),
        ({"rounding": "yaqa"}, ValueError, "requires an output Hessian"),
        (
            {
                "rounding": "yaqa",
                "output_hessian": torch.eye(16),
                "output_channel_scale_optimization": True,
            },
            ValueError,
            "output-channel scale",
        ),
        (
            {
                "rounding": "yaqa",
                "output_hessian": torch.eye(16),
                "module_scale_search": True,
            },
            ValueError,
            "module-scale",
        ),
        (
            {
                "rounding": "yaqa",
                "output_hessian": torch.eye(16),
                "viterbi_objective": "hessian_diagonal",
            },
            ValueError,
            "Euclidean",
        ),
        ({"viterbi_minimum_proxy_improvement": True}, TypeError, "real scalar"),
        (
            {"viterbi_minimum_proxy_improvement": -0.1},
            ValueError,
            "finite and nonnegative",
        ),
        (
            {"viterbi_minimum_proxy_improvement": float("inf")},
            ValueError,
            "finite and nonnegative",
        ),
        (
            {"viterbi_minimum_proxy_improvement": 0.001},
            ValueError,
            "requires.*hessian_diagonal",
        ),
    ),
)
def test_qvq_linear_quantization_rejects_invalid_accuracy_upgrade_controls(kwargs, exception, message):
    with pytest.raises(exception, match=message):
        quantize_qvq_linear(torch.eye(16), torch.eye(16), bits=2, **kwargs)


@pytest.mark.parametrize(
    ("weight", "H", "kwargs", "exception", "message"),
    (
        (torch.empty((0, 16)), torch.eye(16), {}, ValueError, "positive"),
        (torch.empty((16, 0)), torch.empty((0, 0)), {}, ValueError, "positive"),
        (
            torch.eye(16),
            torch.eye(16, dtype=torch.int32),
            {},
            TypeError,
            "Hessian.*floating-point",
        ),
        (
            torch.eye(16),
            torch.full((16, 16), torch.nan),
            {},
            ValueError,
            "weight and Hessian.*finite",
        ),
        (
            torch.eye(16),
            torch.eye(16),
            {"bias": torch.zeros(16, dtype=torch.int32)},
            TypeError,
            "bias.*floating",
        ),
        (
            torch.eye(16),
            torch.eye(16),
            {"bias": torch.full((16,), torch.inf)},
            ValueError,
            "bias.*finite",
        ),
        (torch.eye(16), torch.eye(16), {"seed": 1.5}, TypeError, "seed.*integer"),
        (
            torch.eye(16),
            torch.eye(16),
            {"damp_percent": True},
            TypeError,
            "damping percent.*real scalar",
        ),
        (
            torch.eye(16),
            torch.eye(16),
            {"output_hessian": torch.eye(15)},
            ValueError,
            "output Hessian shape",
        ),
        (
            torch.eye(16),
            torch.eye(16),
            {"output_hessian": torch.eye(16, dtype=torch.int64)},
            TypeError,
            "output Hessian.*floating-point",
        ),
        (
            torch.eye(16),
            torch.eye(16),
            {"output_hessian": torch.full((16, 16), torch.nan)},
            ValueError,
            "output Hessian.*finite",
        ),
    ),
)
def test_qvq_linear_quantization_rejects_malformed_boundary_inputs(weight, H, kwargs, exception, message):
    with pytest.raises(exception, match=message):
        quantize_qvq_linear(weight, H, bits=2, **kwargs)


@pytest.mark.parametrize("name", ("inner", "SU", "SV"))
def test_rht_reconstruction_rejects_nonfinite_inputs(name):
    tensors = {
        "inner": torch.eye(16),
        "SU": torch.ones(16),
        "SV": torch.ones(16),
    }
    tensors[name].reshape(-1)[0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        rht_reconstruct_weight(tensors["inner"], tensors["SU"], tensors["SV"])






@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"bits": 0, "vector_size": 1, "trellis_window": 2}, ValueError, "rate"),
        ({"bits": True, "vector_size": 1, "trellis_window": 2}, TypeError, "rate"),
        ({"bits": 1.25, "vector_size": 1, "trellis_window": 2}, ValueError, "half-integer"),
        ({"bits": 1, "vector_size": 0, "trellis_window": 2}, ValueError, "vector_size"),
        ({"bits": 1, "vector_size": True, "trellis_window": 2}, ValueError, "vector_size"),
        ({"bits": 1, "vector_size": 1.0, "trellis_window": 2}, ValueError, "vector_size"),
        ({"bits": 1, "vector_size": 1, "trellis_window": 0}, ValueError, "trellis_window"),
        ({"bits": 1, "vector_size": 1, "trellis_window": True}, ValueError, "trellis_window"),
        ({"bits": 1, "vector_size": 1, "trellis_window": 2.0}, ValueError, "trellis_window"),
        ({"bits": 3, "vector_size": 2, "trellis_window": 4}, ValueError, r"rate \* vector_size"),
    ],
)
def test_bitshift_rejects_invalid_trellis_geometry(kwargs, exception, message):
    with pytest.raises(exception, match=message):
        bitshift_next_state(torch.tensor(0), torch.tensor(0), **kwargs)


def test_qvq_transition_bits_rejects_bools_despite_lru_cache_hash_aliasing():
    """Deterministic guard for the lru_cache bool-aliasing regression.

    ``qvq_transition_bits`` was once ``lru_cache``-decorated directly; because
    ``True == 1`` and ``hash(True) == hash(1)``, a cache entry populated by an
    integer rate was returned verbatim for a boolean rate, silently skipping
    the ``TypeError`` — but only when suite ordering had already warmed the
    cache. Reproduce that ordering explicitly: reset the cache, populate it
    with the aliased integer keys FIRST, then assert booleans still raise and
    the cached integer results remain intact.
    """

    from gptqmodel.quantization import qvq_rates as qvq_rates_module

    qvq_rates_module._qvq_transition_bits_cached.cache_clear()
    assert qvq_transition_bits(1, vector_size=1) == 1
    with pytest.raises(TypeError, match="rate"):
        qvq_transition_bits(True, vector_size=1)
    # rate=0 is rejected before caching (ValueError), so False cannot alias a
    # cached zero entry — but assert the boolean still raises TypeError first.
    with pytest.raises(ValueError, match="rate"):
        qvq_transition_bits(0, vector_size=1)
    with pytest.raises(TypeError, match="rate"):
        qvq_transition_bits(False, vector_size=1)
    # vector_size shares the aliasing hazard through the same cache key.
    with pytest.raises(ValueError, match="vector size"):
        qvq_transition_bits(1, vector_size=True)
    assert qvq_transition_bits(1, vector_size=1) == 1


def test_bitshift_rejects_out_of_range_states_and_edges():
    with pytest.raises(ValueError, match="states"):
        bitshift_next_state(
            torch.tensor([-1, 0]),
            torch.tensor([0, 0]),
            bits=1,
            vector_size=1,
            trellis_window=2,
        )
    with pytest.raises(ValueError, match="states"):
        bitshift_next_state(
            torch.tensor([0, 4]),
            torch.tensor([0, 0]),
            bits=1,
            vector_size=1,
            trellis_window=2,
        )
    with pytest.raises(ValueError, match="edge"):
        bitshift_next_state(
            torch.tensor([0, 0]),
            torch.tensor([-1, 2]),
            bits=1,
            vector_size=1,
            trellis_window=2,
        )


@pytest.mark.parametrize(
    ("states", "lut", "kwargs", "error", "message"),
    [
        (
            torch.tensor([0]),
            torch.zeros(8, 2),
            {"trellis_window": 0, "lut_bits": 3},
            ValueError,
            "trellis_window",
        ),
        (
            torch.tensor([0]),
            torch.zeros(8, 2),
            {"trellis_window": 33, "lut_bits": 3},
            ValueError,
            "trellis_window",
        ),
        (
            torch.tensor([0]),
            torch.zeros(8, 2),
            {"trellis_window": 8, "lut_bits": 0},
            ValueError,
            "lut_bits",
        ),
        (
            torch.tensor([0]),
            torch.zeros(8, 2),
            {"trellis_window": 8, "lut_bits": 16},
            ValueError,
            "lut_bits",
        ),
        (
            torch.tensor([0]),
            torch.zeros(8),
            {"trellis_window": 8, "lut_bits": 3},
            ValueError,
            "shape",
        ),
        (
            torch.tensor([0]),
            torch.zeros(7, 2),
            {"trellis_window": 8, "lut_bits": 3},
            ValueError,
            "shape",
        ),
        (
            torch.tensor([0]),
            torch.zeros(8, 2, dtype=torch.int32),
            {"trellis_window": 8, "lut_bits": 3},
            TypeError,
            "floating",
        ),
        (
            torch.tensor([0]),
            torch.full((8, 2), torch.nan),
            {"trellis_window": 8, "lut_bits": 3},
            ValueError,
            "finite",
        ),
        (
            torch.tensor([-1]),
            torch.zeros(8, 2),
            {"trellis_window": 8, "lut_bits": 3},
            ValueError,
            "states",
        ),
        (
            torch.tensor([256]),
            torch.zeros(8, 2),
            {"trellis_window": 8, "lut_bits": 3},
            ValueError,
            "states",
        ),
    ],
)
def test_hyb_decode_rejects_invalid_inputs(states, lut, kwargs, error, message):
    with pytest.raises(error, match=message):
        hyb_decode_states(states, lut, **kwargs)


@pytest.mark.parametrize(
    ("sequence", "codebook", "error", "message"),
    [
        (torch.zeros(2), torch.zeros(4, 1), ValueError, "sequence"),
        (torch.zeros(2, 1), torch.zeros(4), ValueError, "codebook"),
        (torch.zeros(0, 1), torch.zeros(4, 1), ValueError, "at least one"),
        (torch.zeros(2, 2), torch.zeros(4, 1), ValueError, "vector sizes"),
        (
            torch.zeros(2, 1, dtype=torch.int32),
            torch.zeros(4, 1),
            TypeError,
            "floating",
        ),
        (
            torch.zeros(2, 1),
            torch.zeros(4, 1, dtype=torch.int32),
            TypeError,
            "floating",
        ),
        (torch.full((2, 1), torch.inf), torch.zeros(4, 1), ValueError, "finite"),
        (torch.zeros(2, 1), torch.full((4, 1), torch.nan), ValueError, "finite"),
        (torch.zeros(2, 1), torch.zeros(3, 1), ValueError, "power of two"),
        (torch.zeros(2, 2), torch.zeros(2, 2), ValueError, r"rate \* vector_size"),
    ],
)
def test_viterbi_rejects_invalid_inputs(sequence, codebook, error, message):
    with pytest.raises(error, match=message):
        viterbi_quantize(sequence, codebook, bits=1)


def test_viterbi_rejects_mixed_devices():
    with pytest.raises(ValueError, match="same device"):
        viterbi_quantize(torch.zeros(2, 1), torch.zeros(4, 1, device="meta"), bits=1)


def test_viterbi_single_step_selects_nearest_code():
    result = viterbi_quantize(
        torch.tensor([[0.2]]),
        torch.tensor([[0.5], [0.1], [0.8], [0.3]]),
        bits=1,
    )

    assert result.states.tolist() == [1]
    assert torch.allclose(result.values, torch.tensor([[0.1]]))


def test_native_viterbi_single_step_overlap_applies_only_initial_constraint():
    from gptqmodel.utils.qvq_cpu import qvq_cpu_supported, qvq_cpu_viterbi

    if not qvq_cpu_supported():
        pytest.skip("native QVQ CPU kernel unavailable")

    codebook = torch.full((16, 2), 100.0, dtype=torch.float32)
    codebook[5] = torch.tensor([1.0, 0.0])
    codebook[6] = torch.tensor([0.0, 0.0])
    sequences = torch.zeros((1, 1, 2), dtype=torch.float32)
    overlap = torch.tensor([1], dtype=torch.int64)

    states, squared_error = qvq_cpu_viterbi(
        sequences,
        codebook,
        transition_bits=2,
        overlap=overlap,
    )

    # The old step-0 early-continue semantics constrain only the high bits:
    # valid states are 4..7, so state 6 is the unique exact match. Applying a
    # final low-bit constraint as well would incorrectly force state 5.
    assert states.tolist() == [[6]]
    assert squared_error.tolist() == [0.0]

    def legacy_eager_oracle(sequence, candidate_codebook, transition_bits, required_overlap, step_weights=None):
        batch_size, step_count, _ = sequence.shape
        state_count = candidate_codebook.shape[0]
        suffix_count = state_count >> transition_bits
        state_ids = torch.arange(state_count, dtype=torch.int64)
        codebook_norm = candidate_codebook.square().sum(dim=-1)

        def emission(step):
            target = sequence[:, step]
            distance = (
                target.square().sum(dim=-1, keepdim=True)
                + codebook_norm.unsqueeze(0)
                - 2 * target @ candidate_codebook.T
            ).clamp_min_(0)
            if step_weights is not None:
                distance *= step_weights[:, step, None]
            return distance

        costs = emission(0)
        costs.masked_fill_((state_ids >> transition_bits) != required_overlap[:, None], torch.inf)
        backpointers = []
        for step in range(1, step_count):
            best_cost, best_prefix = costs.reshape(batch_size, 1 << transition_bits, suffix_count).min(dim=1)
            costs = best_cost[:, state_ids >> transition_bits] + emission(step)
            backpointers.append(best_prefix)

        # This is intentionally conditional on a transition having occurred:
        # the old native step-0 early continue skipped its final mask.
        if step_count > 1:
            costs.masked_fill_((state_ids & (suffix_count - 1)) != required_overlap[:, None], torch.inf)

        end_state = costs.argmin(dim=1)
        path = torch.empty((batch_size, step_count), dtype=torch.int64)
        path[:, -1] = end_state
        batch_ids = torch.arange(batch_size)
        for step in range(step_count - 1, 0, -1):
            suffix = path[:, step] >> transition_bits
            prefix = backpointers[step - 1][batch_ids, suffix]
            path[:, step - 1] = prefix * suffix_count + suffix
        return path, costs[batch_ids, end_state]

    generator = torch.Generator().manual_seed(20260824)
    for step_count in (1, 2):
        for transition_bits in range(1, 5):
            suffix_count = 16 >> transition_bits
            for iteration in range(8):
                random_codebook = torch.randint(-3, 4, (16, 2), generator=generator).float()
                random_sequences = torch.randint(-3, 4, (3, step_count, 2), generator=generator).float()
                random_overlap = torch.randint(0, suffix_count, (3,), generator=generator)
                random_weights = (
                    torch.randint(1, 4, (3, step_count), generator=generator).float()
                    if iteration % 2
                    else None
                )
                expected_states, expected_error = legacy_eager_oracle(
                    random_sequences,
                    random_codebook,
                    transition_bits,
                    random_overlap,
                    random_weights,
                )
                actual_states, actual_error = qvq_cpu_viterbi(
                    random_sequences,
                    random_codebook,
                    transition_bits=transition_bits,
                    overlap=random_overlap,
                    step_weights=random_weights,
                )
                assert torch.equal(actual_states, expected_states)
                assert torch.equal(actual_error, expected_error)


def test_native_viterbi_v2_is_thread_count_invariant():
    from gptqmodel.utils.qvq_cpu import qvq_cpu_supported, qvq_cpu_viterbi

    if not qvq_cpu_supported():
        pytest.skip("native QVQ CPU kernel unavailable")

    generator = torch.Generator().manual_seed(149)
    codebook = torch.randn((1 << 16, 2), generator=generator, dtype=torch.float32)
    sequences = torch.randn((128, 128, 2), generator=generator, dtype=torch.float32)[101:102].contiguous()
    overlap = torch.randint(0, 512, (128,), generator=generator, dtype=torch.int64)[101:102].contiguous()
    original_threads = torch.get_num_threads()

    def run(thread_count):
        torch.set_num_threads(thread_count)
        if torch.get_num_threads() != thread_count:
            # Without this the test would pass vacuously on a runtime that
            # clamps the request, comparing three identical 16-thread runs.
            pytest.skip(f"cannot set torch thread count to {thread_count}")
        states, squared_error = qvq_cpu_viterbi(
            sequences, codebook, transition_bits=7, overlap=overlap
        )
        return states, pack_trellis_states(states, bits=3.5), squared_error

    # suffix_count is 512, so at::parallel_for uses grain 16 and chunk
    # divup(512, threads): 16 -> 32 and 32 -> 16 leave no scalar remainder,
    # while 24 -> 22 sends the trailing 6 columns of every chunk through
    # fused_candidate_scalar.  All three must agree.
    try:
        outputs = {threads: run(threads) for threads in (16, 24, 32)}
    finally:
        torch.set_num_threads(original_threads)

    reference = outputs[16]
    for threads, candidate in outputs.items():
        for expected, actual in zip(reference, candidate):
            assert torch.equal(expected, actual), f"thread count {threads} diverged"


def test_native_viterbi_rejects_empty_steps():
    from gptqmodel.utils.qvq_cpu import qvq_cpu_supported, qvq_cpu_viterbi

    if not qvq_cpu_supported():
        pytest.skip("native QVQ CPU kernel unavailable")

    sequences = torch.empty((1, 0, 2), dtype=torch.float32)
    codebook = torch.zeros((1 << 16, 2), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="step_count must be positive"):
        qvq_cpu_viterbi(sequences, codebook, transition_bits=7)


def test_native_viterbi_two_step_overlap_applies_final_constraint():
    from gptqmodel.utils.qvq_cpu import qvq_cpu_supported, qvq_cpu_viterbi

    if not qvq_cpu_supported():
        pytest.skip("native QVQ CPU kernel unavailable")

    codebook = torch.full((16, 2), 100.0, dtype=torch.float32)
    codebook[6] = torch.tensor([0.0, 0.0])
    codebook[9] = torch.tensor([3.0, 3.0])
    codebook[10] = torch.tensor([1.0, 1.0])
    sequences = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])

    states, squared_error = qvq_cpu_viterbi(
        sequences,
        codebook,
        transition_bits=2,
        overlap=torch.tensor([1], dtype=torch.int64),
    )
    unconstrained_states, _ = qvq_cpu_viterbi(sequences, codebook, transition_bits=2)

    # The unconstrained unique optimum ends at state 10 (suffix 2). The final
    # overlap mask must instead choose state 9 (suffix 1), tracing back to 6.
    assert unconstrained_states.tolist() == [[6, 10]]
    assert states.tolist() == [[6, 9]]
    assert squared_error.tolist() == [8.0]


@pytest.mark.parametrize("step_count", [1, 2, 3])
def test_native_viterbi_invalid_overlap_is_all_infinity(step_count):
    from gptqmodel.utils.qvq_cpu import qvq_cpu_supported, qvq_cpu_viterbi

    if not qvq_cpu_supported():
        pytest.skip("native QVQ CPU kernel unavailable")

    generator = torch.Generator().manual_seed(20260825 + step_count)
    codebook = torch.randn((64, 2), generator=generator)
    sequences = torch.randn((7, step_count, 2), generator=generator)
    overlaps = torch.tensor(
        [
            2,
            -1,
            16,
            (1 << 32) + 2,
            torch.iinfo(torch.int64).min,
            torch.iinfo(torch.int64).max,
            17,
        ],
        dtype=torch.int64,
    )

    states, squared_error = qvq_cpu_viterbi(
        sequences,
        codebook,
        transition_bits=2,
        overlap=overlaps,
    )
    valid_states, valid_error = qvq_cpu_viterbi(
        sequences[:1],
        codebook,
        transition_bits=2,
        overlap=overlaps[:1],
    )

    # A mixed batch must preserve the valid row while both kinds of invalid
    # overlap deterministically produce the native op's documented sentinel.
    assert torch.equal(states[:1], valid_states)
    assert torch.equal(squared_error[:1], valid_error)
    assert states[1:].tolist() == [[0] * step_count] * 6
    assert torch.isinf(squared_error[1:]).all()


def test_native_banked_viterbi_adjacent_steps_invalid_sentinels_and_boundary_traceback():
    from gptqmodel.utils.qvq_cpu import qvq_cpu_viterbi_banked

    codebooks = torch.full((2, 4, 2), 100.0, dtype=torch.float32)
    codebooks[0, 1] = 0.0
    codebooks[1, 2] = 10.0

    with pytest.raises(RuntimeError, match="positive segment_steps"):
        qvq_cpu_viterbi_banked(torch.zeros((1, 1, 2)), codebooks, 1, 0)

    one_step = torch.zeros((5, 1, 2), dtype=torch.float32)
    invalid = torch.tensor(
        [-1, 2, torch.iinfo(torch.int64).max, torch.iinfo(torch.int64).min, 2**32], dtype=torch.int64
    )
    states, squared_error, bank_ids = qvq_cpu_viterbi_banked(one_step, codebooks, 1, 1, overlap=invalid)
    assert torch.equal(states, torch.zeros_like(states))
    assert torch.isinf(squared_error).all()
    assert torch.equal(bank_ids, torch.zeros_like(bank_ids))

    two_steps = torch.tensor([[[0.0, 0.0], [10.0, 10.0]]], dtype=torch.float32)
    states, squared_error, bank_ids = qvq_cpu_viterbi_banked(two_steps, codebooks, 1, 1)
    assert torch.equal(states, torch.tensor([[1, 2]], dtype=torch.int64))
    assert torch.equal(bank_ids, torch.tensor([[0, 1]], dtype=torch.uint8))
    assert torch.equal(squared_error, torch.zeros_like(squared_error))

    wide_codebooks = torch.full((4, 65536, 2), 100.0, dtype=torch.float32)
    wide_codebooks[3, 65535] = 0.0
    wide_codebooks[0, 32768] = 10.0
    states, squared_error, bank_ids = qvq_cpu_viterbi_banked(
        two_steps, wide_codebooks, transition_bits=15, segment_steps=1
    )
    assert torch.equal(states, torch.tensor([[65535, 32768]], dtype=torch.int64))
    assert torch.equal(bank_ids, torch.tensor([[3, 0]], dtype=torch.uint8))
    assert torch.equal(squared_error, torch.zeros_like(squared_error))


def test_native_banked_viterbi_v2_is_thread_count_invariant():
    from gptqmodel.utils.qvq_cpu import qvq_cpu_viterbi_banked

    generator = torch.Generator().manual_seed(2)
    codebooks = torch.randn((2, 65536, 2), generator=generator, dtype=torch.float32)
    sequences = torch.randn((1, 32, 2), generator=generator, dtype=torch.float32)
    overlap = torch.tensor([2], dtype=torch.int64)
    original_threads = torch.get_num_threads()

    def run(thread_count):
        torch.set_num_threads(thread_count)
        states, _, segment_bank_ids = qvq_cpu_viterbi_banked(
            sequences,
            codebooks,
            transition_bits=7,
            segment_steps=16,
            overlap=overlap,
        )
        packed_words = pack_trellis_states(states, bits=3.5)
        return states, segment_bank_ids, packed_words

    try:
        aligned = run(16)  # suffix chunks are 32 columns: no scalar remainder.
        unaligned = run(24)  # suffix chunks are 22 columns: scalar remainder is reachable.
    finally:
        torch.set_num_threads(original_threads)

    for aligned_output, unaligned_output in zip(aligned, unaligned):
        assert torch.equal(aligned_output, unaligned_output)


@pytest.mark.parametrize("transition_bits", (7, 16))
def test_native_banked_viterbi_suffix_partition_is_thread_count_invariant(transition_bits):
    from gptqmodel.utils.qvq_cpu import qvq_cpu_viterbi_banked

    state_count = 1 << 16
    suffix_count = 1 << (16 - transition_bits)
    prefix_count = state_count // suffix_count
    suffix = torch.arange(suffix_count, dtype=torch.float32)
    bank_zero = torch.stack((suffix.remainder(31) / 8, suffix.remainder(29) / 8), dim=-1)
    bank_one = bank_zero + torch.tensor((0.75, -0.5), dtype=torch.float32)
    codebooks = torch.stack((bank_zero.repeat((prefix_count, 1)), bank_one.repeat((prefix_count, 1))))

    distinct_codewords = torch.unique(codebooks[0], dim=0).shape[0]
    tie_density = 1.0 - distinct_codewords / state_count
    assert tie_density >= 0.99

    selected_suffix = min(37, suffix_count - 1)
    sequences = torch.empty((1, 32, 2), dtype=torch.float32)
    sequences[:, :16] = codebooks[0, selected_suffix]
    sequences[:, 16:] = codebooks[1, selected_suffix]
    original_threads = torch.get_num_threads()
    outputs = {}

    try:
        for thread_count in (8, 16, 24, 32):
            torch.set_num_threads(thread_count)
            if torch.get_num_threads() != thread_count:
                pytest.skip(f"cannot set torch thread count to {thread_count}")
            states, squared_error, segment_bank_ids = qvq_cpu_viterbi_banked(
                sequences,
                codebooks,
                transition_bits=transition_bits,
                segment_steps=16,
                overlap=torch.tensor([selected_suffix], dtype=torch.int64),
            )
            outputs[thread_count] = (
                states,
                squared_error,
                segment_bank_ids,
                pack_trellis_states(states, bits=transition_bits / 2),
            )
    finally:
        torch.set_num_threads(original_threads)

    reference = outputs[8]
    for thread_count, actual in outputs.items():
        for expected, observed in zip(reference, actual):
            assert torch.equal(expected, observed), (
                f"transition_bits={transition_bits} thread_count={thread_count} "
                f"tie_density={tie_density:.6f} output diverged"
            )

    # Legacy's row-parallel / tiled split must be bit-identical.
    #
    # qvq_viterbi_banked_cpu_legacy picks between a row-parallel leg and an
    # outer-serial / inner-parallel tiled leg on
    # `batch_size >= min(8, at::get_num_threads())`. Both legs must produce the
    # same answer, and this is the only test that crosses that branch. Batch 4
    # does cross it: 2 threads takes the row-parallel leg (4 >= min(8, 2) == 2)
    # and 16 threads takes the tiled leg (4 < 8). Batch >= 8 is row-parallel at
    # every thread count and would NOT exercise the split.
    #
    # The codebooks are quantised onto a coarse grid so the 65536 codewords
    # collapse onto a couple of hundred distinct vectors. Every argmin in the
    # recurrence is then an exact tie broken purely by scan order, which is
    # what a leg that reordered a reduction would corrupt. The tie-density
    # assertion below keeps this gate from passing vacuously on an input that
    # happens to have no ties.
    #
    # Transition width 16 is a latent path: both CPU call sites reject
    # shift > 7 (qvq.py), so nothing in production reaches it today.
    from gptqmodel.quantization.qvq import pack_qvq_bank_ids

    def run_legacy_t16(thread_count, t16_sequences, t16_codebooks, segment_steps):
        torch.set_num_threads(thread_count)
        if torch.get_num_threads() != thread_count:
            pytest.skip(f"cannot set torch thread count to {thread_count}")
        states, squared_error, segment_bank_ids = qvq_cpu_viterbi_banked(
            t16_sequences, t16_codebooks, transition_bits=16, segment_steps=segment_steps
        )
        return (
            states,
            squared_error,
            segment_bank_ids,
            pack_trellis_states(states, bits=8),
            pack_qvq_bank_ids(segment_bank_ids.reshape(-1)),
        )

    try:
        for seed in (11, 202, 3033, 40404):
            for banks, segment_steps in ((1, 16), (2, 16), (2, 8)):
                generator = torch.Generator().manual_seed(seed)
                t16_sequences = torch.round(
                    torch.randn((4, 32, 2), generator=generator, dtype=torch.float32) * 2.0
                ) / 2.0
                t16_codebooks = torch.round(
                    torch.randn((banks, 1 << 16, 2), generator=generator, dtype=torch.float32) * 2.0
                ) / 2.0
                distinct = torch.unique(t16_codebooks[0], dim=0).shape[0]
                assert distinct < 1024, (
                    "tie-density fixture broken: "
                    f"{distinct} distinct codewords means the gate no longer tests tie order"
                )
                row_parallel = run_legacy_t16(2, t16_sequences, t16_codebooks, segment_steps)
                tiled = run_legacy_t16(16, t16_sequences, t16_codebooks, segment_steps)
                for expected, actual in zip(row_parallel, tiled):
                    assert torch.equal(expected, actual), (
                        "legacy t16 tiling legs diverged: "
                        f"seed={seed} banks={banks} segment_steps={segment_steps}"
                    )
    finally:
        torch.set_num_threads(original_threads)


def test_native_banked_viterbi_force_overrides_are_parsed_and_scoped(monkeypatch):
    """QVQ_TEST_FORCE_BANKED_* must parse their value and stay in scope."""

    from gptqmodel.utils.qvq_cpu import qvq_cpu_viterbi_banked

    monkeypatch.delenv("QVQ_TEST_FORCE_BANKED_LEGACY", raising=False)
    monkeypatch.delenv("QVQ_TEST_FORCE_BANKED_G_ONLY", raising=False)

    # 1. Value parsing. The two overrides are mutually exclusive, so setting one
    #    truthy and the other to a falsey value must NOT trip the exclusion
    #    check. Under presence-only `getenv(...) != nullptr` semantics every one
    #    of these would raise, which is what makes this assertion non-vacuous.
    generator = torch.Generator().manual_seed(7)
    sequences = torch.randn((4, 16, 2), generator=generator, dtype=torch.float32)
    codebooks = torch.randn((2, 1 << 16, 2), generator=generator, dtype=torch.float32)

    monkeypatch.setenv("QVQ_TEST_FORCE_BANKED_LEGACY", "1")
    for falsey in ("0", "false", "FALSE", "off", "Off", ""):
        monkeypatch.setenv("QVQ_TEST_FORCE_BANKED_G_ONLY", falsey)
        qvq_cpu_viterbi_banked(sequences, codebooks, transition_bits=16, segment_steps=16)

    # ...and two genuinely truthy values still do trip it.
    monkeypatch.setenv("QVQ_TEST_FORCE_BANKED_G_ONLY", "1")
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        qvq_cpu_viterbi_banked(sequences, codebooks, transition_bits=16, segment_steps=16)

    monkeypatch.delenv("QVQ_TEST_FORCE_BANKED_LEGACY", raising=False)
    monkeypatch.delenv("QVQ_TEST_FORCE_BANKED_G_ONLY", raising=False)

    # 2. Scope. The overrides only select between the two recurrences for the
    #    shapes the dispatcher actually considers. They must not divert shapes
    #    it deliberately excludes -- here bank_count 3, which the t16 predicate
    #    excludes. Legacy implements bank_count 3, so an out-of-scope
    #    QVQ_TEST_FORCE_BANKED_LEGACY does not crash; it silently returns a
    #    different answer. This fixture is not vacuous: with the override
    #    unscoped it changes one selected state and all four squared errors,
    #    reproducibly at 4, 16 and 32 threads.
    def run_banks3():
        gen = torch.Generator().manual_seed(2)
        seq = torch.randn((4, 32, 2), generator=gen, dtype=torch.float32)
        cbs = torch.randn((3, 1 << 16, 2), generator=gen, dtype=torch.float32)
        return qvq_cpu_viterbi_banked(seq, cbs, transition_bits=16, segment_steps=16)

    unforced = run_banks3()
    for name in ("QVQ_TEST_FORCE_BANKED_LEGACY", "QVQ_TEST_FORCE_BANKED_G_ONLY"):
        monkeypatch.setenv(name, "1")
        forced = run_banks3()
        monkeypatch.delenv(name, raising=False)
        for expected, actual in zip(unforced, forced):
            assert torch.equal(expected, actual), f"{name} escaped its dispatcher scope"

    # V=4 is accepted by the public wrapper but is not a legacy-t16 dispatch
    # shape. The G-only override must not divert this call either.
    generator = torch.Generator().manual_seed(2)
    v4_sequences = torch.randn((4, 32, 4), generator=generator, dtype=torch.float32)
    v4_codebooks = torch.randn((1, 1 << 16, 4), generator=generator, dtype=torch.float32)
    v4_unforced = qvq_cpu_viterbi_banked(v4_sequences, v4_codebooks, transition_bits=16, segment_steps=16)
    monkeypatch.setenv("QVQ_TEST_FORCE_BANKED_G_ONLY", "1")
    v4_forced = qvq_cpu_viterbi_banked(v4_sequences, v4_codebooks, transition_bits=16, segment_steps=16)
    for expected, actual in zip(v4_unforced, v4_forced):
        assert torch.equal(expected, actual), "QVQ_TEST_FORCE_BANKED_G_ONLY escaped the V=2 dispatcher scope"


def _tail_biting_states(bits: float, *, tiles: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shift = qvq_transition_bits(bits)
    edges = torch.randint(0, 1 << shift, (tiles, 128), generator=generator, dtype=torch.int64)
    stream = []
    for bit in range(shift - 1, -1, -1):
        stream.append((edges >> bit) & 1)
    stream = torch.stack(stream, dim=-1).reshape(tiles, -1)
    states = torch.empty((tiles, 128), dtype=torch.int64)
    for step in range(128):
        end = (step + 1) * shift
        offsets = torch.arange(end - 16, end) % stream.shape[1]
        state = torch.zeros(tiles, dtype=torch.int64)
        for offset in offsets:
            state = (state << 1) | stream[:, offset]
        states[:, step] = state
    return states


@pytest.mark.parametrize("bits", QVQ_HALF_STEP_BITS)
def test_planar_trellis_round_trip_all_bits(bits):
    transition_bits = qvq_transition_bits(bits)
    states = _tail_biting_states(bits, tiles=3, seed=6100 + transition_bits)

    packed = pack_trellis_states(states, bits=bits)
    unpacked = unpack_trellis_states(packed, bits=bits)

    assert packed.dtype == torch.int32
    assert packed.shape == (3, qvq_words_per_tile(bits))
    assert packed.numel() * 32 == states.numel() * transition_bits
    assert torch.equal(unpacked, states)


@pytest.mark.parametrize("bits", QVQ_HALF_STEP_BITS)
def test_planar_trellis_decode_and_reconstruction_agree(bits):
    states = _tail_biting_states(bits, tiles=4, seed=7100 + qvq_transition_bits(bits))
    trellis = pack_trellis_states(states, bits=bits)

    tiles = decode_trellis_tiles(trellis, bits=bits)
    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=bits,
        in_features=32,
        out_features=32,
    )
    expected = tiles.view(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)

    assert torch.equal(inner, expected)




@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: pack_trellis_states(torch.empty(0), bits=2), "non-empty"),
        (
            lambda: pack_trellis_states(torch.zeros(15, dtype=torch.int64), bits=2),
            "32-edge",
        ),
        (lambda: pack_trellis_states(torch.full((32,), 65536), bits=2), "states"),
        (
            lambda: pack_trellis_states(torch.arange(32, dtype=torch.int64), bits=2),
            "tail-biting",
        ),
        (
            lambda: unpack_trellis_states(torch.empty(0, dtype=torch.int32), bits=2),
            "non-empty",
        ),
        (
            lambda: unpack_trellis_states(torch.zeros(3, dtype=torch.int32), bits=2),
            "word count",
        ),
        (
            lambda: unpack_trellis_states(torch.zeros(6, dtype=torch.int32), bits=1.5, vector_size=3),
            "integer transition width",
        ),
        (
            lambda: decode_trellis_tiles(
                torch.zeros(2, dtype=torch.int32),
                bits=2,
                vector_size=1,
            ),
            "PGC16 requires",
        ),
        (
            lambda: reconstruct_qvq_inner_weight(
                torch.zeros((1, 16), dtype=torch.int32),
                bits=2,
                in_features=15,
                out_features=16,
            ),
            "divisible",
        ),
        (
            lambda: reconstruct_qvq_inner_weight(
                torch.zeros((2, 16), dtype=torch.int32),
                bits=2,
                in_features=16,
                out_features=16,
            ),
            "shape",
        ),
    ],
)
def test_planar_trellis_rejects_invalid_layouts(call, message):
    with pytest.raises((TypeError, ValueError), match=message):
        call()


@pytest.mark.parametrize("bits", QVQ_HALF_STEP_BITS)
def test_qvq_torch_linear_matches_explicit_reference(bits):
    transition_bits = qvq_transition_bits(bits)
    generator = torch.Generator().manual_seed(10100 + transition_bits)
    states = _tail_biting_states(bits, tiles=1, seed=10100 + transition_bits)
    tensors = {
        "trellis": pack_trellis_states(states, bits=bits),
        "SU": torch.randn(16, generator=generator),
        "SV": torch.randn(16, generator=generator),
        "bias": torch.randn(16, generator=generator),
    }
    layer = QVQReferenceLinear(
        bits=bits,
        in_features=16,
        out_features=16,
        name="proj",
        tensors=tensors,
        out_dtype=torch.float32,
    )
    x = torch.randn((2, 3, 16), generator=generator)

    inner = reconstruct_qvq_inner_weight(
        tensors["trellis"],
        bits=bits,
        in_features=16,
        out_features=16,
    )
    expected = matmul_hadU(matmul_hadU(x.reshape(-1, 16) * tensors["SU"]) @ inner)
    expected = (expected * tensors["SV"] + tensors["bias"]).reshape(2, 3, 16)

    torch.testing.assert_close(layer(x), expected)


def _qvq_oracle_layer(*, bias: bool = True) -> QVQLinear:
    generator = torch.Generator().manual_seed(20260824)
    states = _tail_biting_states(2, tiles=1, seed=20260824)
    tensors = {
        "trellis": pack_trellis_states(states, bits=2),
        "SU": torch.randn(16, generator=generator),
        "SV": torch.randn(16, generator=generator),
    }
    if bias:
        tensors["bias"] = torch.randn(16, generator=generator)
    return QVQLinear.from_tensors(
        bits=2,
        in_features=16,
        out_features=16,
        name="oracle_proj",
        tensors=tensors,
    )


def test_qvq_dense_oracle_is_full_layer_fp32_finite_and_does_not_cache():
    layer = _qvq_oracle_layer()
    x = torch.randn((2, 3, 16), generator=torch.Generator().manual_seed(7), dtype=torch.float16)
    attributes_before = set(vars(layer))
    assert layer._dtype_cache == {}
    assert layer._qvq_cuda_bank_cache is None
    assert layer._qvq_mps_bank_ids_cache is None
    assert not hasattr(layer, "_qvq_cpu_dense_inner_cache")

    actual = qvq_dense_oracle_forward(layer, x, device="cpu")
    inner = layer.get_inner_weight_tensor()
    x_fp32 = x.float().reshape(-1, 16)
    expected = matmul_hadU(matmul_hadU(x_fp32 * layer.SU.float()) @ inner)
    expected = (expected * layer.SV.float() + layer.bias.float()).reshape(2, 3, 16)

    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"
    assert not actual.requires_grad
    assert torch.isfinite(actual).all()
    assert (actual - expected).abs().max().item() <= 2e-3
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-3)
    assert set(vars(layer)) == attributes_before
    assert layer._dtype_cache == {}
    assert layer._qvq_cuda_bank_cache is None
    assert layer._qvq_mps_bank_ids_cache is None
    assert not hasattr(layer, "_qvq_cpu_dense_inner_cache")


def test_qvq_dense_oracle_does_not_inherit_input_dtype():
    layer = _qvq_oracle_layer()
    rounded = torch.randn((4, 16), generator=torch.Generator().manual_seed(8)).half()

    from_fp16 = qvq_dense_oracle_forward(layer, rounded)
    from_same_values_fp32 = qvq_dense_oracle_forward(layer, rounded.float())

    torch.testing.assert_close(from_fp16, from_same_values_fp32, rtol=0, atol=0)


def test_qvq_dense_oracle_releases_reconstruction_after_return_and_repeated_calls():
    layer = _qvq_oracle_layer()
    x = torch.randn((2, 16), generator=torch.Generator().manual_seed(9))
    references = []
    original = reconstruct_qvq_inner_weight

    def tracked_reconstruction(*args, **kwargs):
        assert torch.is_inference_mode_enabled()
        inner = original(*args, **kwargs)
        references.append(weakref.ref(inner))
        return inner

    with patch("gptqmodel.nn_modules.qlinear.qvq.reconstruct_qvq_inner_weight", tracked_reconstruction):
        for _ in range(4):
            result = qvq_dense_oracle_forward(layer, x)
            del result
            gc.collect()
            assert all(reference() is None for reference in references)


def test_qvq_dense_oracle_releases_reconstruction_on_exception():
    layer = _qvq_oracle_layer()
    x = torch.randn((2, 16), generator=torch.Generator().manual_seed(10))
    references = []
    original = reconstruct_qvq_inner_weight

    def tracked_reconstruction(*args, **kwargs):
        inner = original(*args, **kwargs)
        references.append(weakref.ref(inner))
        return inner

    with (
        patch("gptqmodel.nn_modules.qlinear.qvq.reconstruct_qvq_inner_weight", tracked_reconstruction),
        patch("gptqmodel.nn_modules.qlinear.qvq.matmul_hadU", side_effect=RuntimeError("injected failure")),
        pytest.raises(RuntimeError, match="injected failure"),
    ):
        qvq_dense_oracle_forward(layer, x)

    gc.collect()
    assert len(references) == 1
    assert references[0]() is None


def test_qvq_dense_oracle_rejects_grad_enabled_input_before_reconstruction():
    layer = _qvq_oracle_layer()
    x = torch.randn((2, 16), requires_grad=True)

    with (
        patch("gptqmodel.nn_modules.qlinear.qvq.reconstruct_qvq_inner_weight") as reconstruct,
        pytest.raises(RuntimeError, match="requires gradients"),
    ):
        qvq_dense_oracle_forward(layer, x)

    reconstruct.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device parity requires a CUDA-enabled Torch host")
def test_qvq_dense_oracle_cpu_and_cuda_device_parity():
    layer = _qvq_oracle_layer()
    x = torch.randn((2, 16), generator=torch.Generator().manual_seed(11))

    cpu = qvq_dense_oracle_forward(layer, x, device="cpu")
    cuda = qvq_dense_oracle_forward(layer, x, device="cuda").cpu()

    assert torch.isfinite(cuda).all()
    torch.testing.assert_close(cuda, cpu, rtol=0, atol=2e-3)


def _qvq_linear_tensors(*, bits: float = 2):
    return {
        "trellis": torch.zeros((1, qvq_words_per_tile(bits)), dtype=torch.int32),
        "SU": torch.ones(16),
        "SV": torch.ones(16),
    }


@pytest.mark.parametrize(
    ("bits", "in_features", "mutator", "error", "message"),
    [
        (9, 16, lambda tensors: tensors, NotImplementedError, "supports"),
        (2, 15, lambda tensors: tensors, NotImplementedError, "divisible"),
        (
            2,
            16,
            lambda tensors: {k: v for k, v in tensors.items() if k != "SU"},
            ValueError,
            "missing",
        ),
        (
            2,
            16,
            lambda tensors: {**tensors, "tlut": torch.zeros((512, 2))},
            ValueError,
            "rejects serialized",
        ),
        (
            2,
            16,
            lambda tensors: {**tensors, "trellis": tensors["trellis"].to(torch.int16)},
            TypeError,
            "int32",
        ),
        (
            2,
            16,
            lambda tensors: {**tensors, "SU": tensors["SU"].to(torch.int32)},
            TypeError,
            "floating",
        ),
        (
            2,
            16,
            lambda tensors: {**tensors, "SV": torch.ones(15)},
            ValueError,
            "shape",
        ),
        (
            2,
            16,
            lambda tensors: {**tensors, "bias": torch.ones(15)},
            ValueError,
            "bias",
        ),
        (
            2,
            16,
            lambda tensors: {**tensors, "bias": torch.ones(16, dtype=torch.int32)},
            TypeError,
            "floating-point",
        ),
        (
            2,
            16,
            lambda tensors: {**tensors, "SU": torch.full((16,), torch.nan)},
            ValueError,
            "finite",
        ),
        (
            2,
            16,
            lambda tensors: {**tensors, "SV": torch.full((16,), torch.inf)},
            ValueError,
            "finite",
        ),
        (
            2,
            16,
            lambda tensors: {**tensors, "bias": torch.full((16,), torch.nan)},
            ValueError,
            "finite",
        ),
    ],
)
def test_qvq_reference_linear_rejects_invalid_contract(bits, in_features, mutator, error, message):
    with pytest.raises(error, match=message):
        QVQReferenceLinear(
            bits=bits,
            in_features=in_features,
            out_features=16,
            name="proj",
            tensors=mutator(_qvq_linear_tensors(bits=bits if bits in range(1, 9) else 2)),
        )


def test_qvq_linear_base_lifecycle_factory_post_init_and_cpu_fallback():
    tensors = _qvq_linear_tensors()
    layer = QVQLinear.from_tensors(
        bits=2,
        in_features=16,
        out_features=16,
        name="proj",
        tensors=tensors,
    )
    layer.post_init()
    assert set(layer.state_dict()) == {"trellis", "SU", "SV"}
    assert layer.get_inner_weight_tensor().dtype == torch.float32
    assert layer(torch.ones((1, 16))).shape == (1, 16)
    with pytest.raises(ValueError, match="input width"):
        layer(torch.ones((1, 15)))


def test_qvq_linear_declares_valid_base_quant_linear_contract():
    assert issubclass(QVQLinear, BaseQuantLinear)
    QVQLinear.verify_supports_params()
    valid, error = QVQLinear.validate(
        bits=7.5,
        in_features=16,
        out_features=16,
        dtype=torch.float16,
        adapter=None,
        group_size=-1,
        desc_act=False,
        sym=True,
        pack_dtype=torch.int32,
        format=FORMAT.QVQ,
    )
    assert valid is True
    assert error is None


@pytest.mark.parametrize("backend", (BACKEND.QVQ, BACKEND.AUTO))
@pytest.mark.parametrize("bits", (1, 1.5, 8))
def test_qvq_linear_is_selected_for_qvq_lifecycle(backend, bits):
    selected = select_quant_linear(
        bits=bits,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=backend,
        format=FORMAT.QVQ,
        quant_method=METHOD.QVQ,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )

    assert selected is QVQLinear


@pytest.mark.parametrize("bits", (1.5, 2.5, 7.5))
def test_qvq_generic_replacement_preserves_fractional_rate(bits):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(16, 16, bias=False, dtype=torch.float16)

    cfg = QVQConfig(bits=bits, rounding="block_ldlq", offload_to_disk=False)
    model = TinyModel()

    selected = make_quant(
        model,
        cfg,
        {"proj": {}},
        BACKEND.QVQ,
        "lm_head",
        device=DEVICE.CPU,
        dtype=torch.float16,
    )

    assert selected is QVQLinear
    assert model.proj.bits == bits


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"group_size": 128}, "group_size=-1"),
        ({"desc_act": True}, "desc_act=False"),
        ({"sym": False}, "sym=True"),
        ({"format": FORMAT.GPTQ}, "format=qvq"),
    ),
)
def test_qvq_linear_rejects_noncanonical_lifecycle_contract(override, message):
    kwargs = {
        "bits": 2,
        "in_features": 16,
        "out_features": 16,
        "dtype": torch.float16,
        "adapter": None,
        "group_size": -1,
        "desc_act": False,
        "sym": True,
        "pack_dtype": torch.int32,
        "format": FORMAT.QVQ,
    }
    kwargs.update(override)

    valid, error = QVQLinear.validate(**kwargs)

    assert valid is False
    assert message in str(error)


def test_qvq_linear_preallocates_exact_checkpoint_buffers_and_invalidates_runtime_handles():
    layer = QVQLinear(
        bits=3.5,
        in_features=32,
        out_features=48,
        bias=True,
        name="proj",
    )
    assert layer.trellis.shape == (6, 28)
    assert layer.trellis.dtype == torch.int32
    assert layer.SU.shape == (32,)
    assert layer.SV.shape == (48,)
    assert layer.bias.shape == (48,)
    assert set(layer.state_dict()) == {"trellis", "SU", "SV", "bias"}

    layer._qvq_mps_compander = object()
    layer.to(dtype=torch.float32)
    assert layer._qvq_mps_compander is None
    assert layer.trellis.dtype == torch.int32
    assert layer.SU.dtype == torch.float32


def test_qvq_linear_accepts_rate_keyed_v4_bank_selectors_and_reconstructs_exactly():
    trellis = torch.zeros((1, qvq_words_per_tile(2, vector_size=4)), dtype=torch.int32)
    bank_ids = torch.tensor([2], dtype=torch.uint8)
    layer = QVQLinear(
        bits=2,
        in_features=16,
        out_features=16,
        vector_size=4,
        bank_count=4,
        name="banked",
        tensors={"trellis": trellis, "SU": torch.ones(16), "SV": torch.ones(16), "bank_ids": bank_ids},
    )
    expected = reconstruct_qvq_inner_weight(
        trellis,
        bits=2,
        in_features=16,
        out_features=16,
        vector_size=4,
        bank_ids=bank_ids,
    )
    torch.testing.assert_close(layer.get_inner_weight_tensor(dtype=torch.float32), expected, rtol=0, atol=0)
    assert torch.equal(layer.bank_ids, bank_ids)


def test_qvq_linear_canonicalizes_dense_bank_selectors_for_state_dict_reload():
    in_features, out_features = 32, 64
    tile_count = (in_features // 16) * (out_features // 16)
    trellis = torch.zeros((tile_count, qvq_words_per_tile(2, vector_size=4)), dtype=torch.int32)
    dense_bank_ids = torch.arange(tile_count, dtype=torch.uint8).remainder_(4)
    layer = QVQLinear(
        bits=2,
        in_features=in_features,
        out_features=out_features,
        vector_size=4,
        bank_count=4,
        name="dense_banked",
        tensors={
            "trellis": trellis,
            "SU": torch.ones(in_features),
            "SV": torch.ones(out_features),
            "bank_ids": dense_bank_ids,
        },
    )

    assert layer.bank_ids.shape == ((tile_count + 3) // 4,)
    reloaded = QVQLinear(
        bits=2,
        in_features=in_features,
        out_features=out_features,
        vector_size=4,
        bank_count=4,
        name="dense_banked",
    )
    reloaded.load_state_dict(layer.state_dict(), strict=True)

    assert torch.equal(reloaded.bank_ids, pack_qvq_bank_ids(dense_bank_ids))
    torch.testing.assert_close(
        reloaded.get_inner_weight_tensor(dtype=torch.float32),
        layer.get_inner_weight_tensor(dtype=torch.float32),
        rtol=0,
        atol=0,
    )


def test_banked_tile_selection_uses_heldout_outputs_and_updates_residual_in_order():
    generator = torch.Generator().manual_seed(20260813)
    candidates = torch.randn((4, 16, 16), generator=generator)
    inputs = torch.randn((8, 16), generator=generator)
    target = inputs @ candidates[2]
    selected, bank_ids, residual = select_banked_tiles_by_output_error(candidates, inputs, target)
    assert bank_ids.shape == (1,)
    assert int(bank_ids.item()) == 2
    torch.testing.assert_close(selected, candidates[2], rtol=0, atol=0)
    torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=1e-5, atol=1e-5)


def test_qvq_linear_rejects_unknown_tensor_entries():
    tensors = _qvq_linear_tensors()
    tensors["extra"] = torch.zeros(1)

    with pytest.raises(ValueError, match="unexpected tensors.*extra"):
        QVQLinear(bits=2, in_features=16, out_features=16, name="proj", tensors=tensors)


def test_qvq_linear_meta_preallocation_defers_value_validation_until_materialization():
    with torch.device("meta"):
        layer = QVQLinear(
            bits=2,
            in_features=16,
            out_features=16,
            bias=True,
            name="proj",
        )

    assert layer.trellis.device.type == "meta"
    assert layer.SU.device.type == "meta"
    assert layer.SV.device.type == "meta"
    assert layer.bias.device.type == "meta"


def _qvq_inner_gemv_case(
    bits: float,
    *,
    m: int,
    k: int = 32,
    n: int = 32,
    seed_offset: int = 0,
    codebook_version: str = PGC16_CODEBOOK_VERSION,
):
    transition_bits = qvq_transition_bits(bits)
    generator = torch.Generator().manual_seed(20261200 + transition_bits * 10 + m + seed_offset)
    states = _tail_biting_states(
        bits,
        tiles=(k // 16) * (n // 16),
        seed=20261300 + transition_bits + seed_offset,
    )
    trellis = pack_trellis_states(states, bits=bits)
    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=bits,
        in_features=k,
        out_features=n,
        codebook_version=codebook_version,
    )
    x = torch.randn((m, k), generator=generator).to(torch.float16)
    reference = (x.float() @ inner.float()).to(torch.float16)
    return x, trellis, reference


def test_qvq_cpu_production_forward_never_creates_dense_cache(monkeypatch):
    import gptqmodel.utils.qvq_cpu as qvq_cpu_module

    monkeypatch.setenv("QVQ_CPU_GEMV_DENSE_CACHE", "1")

    def fail_dense_materialization():
        raise AssertionError("production CPU inference must call the native packed GEMV")

    monkeypatch.setattr(qvq_cpu_module, "_qvq_cpu_inner_weight_op", fail_dense_materialization)
    x, trellis, _ = _qvq_inner_gemv_case(2, m=8, k=32, n=32)
    layer = QVQLinear(
        bits=2,
        in_features=32,
        out_features=32,
        tensors={"trellis": trellis, "SU": torch.ones(32), "SV": torch.ones(32)},
    )
    layer.train()
    reference = layer(x.float())
    layer.eval()

    actual = layer(x.float())

    torch.testing.assert_close(actual, reference, rtol=0, atol=2e-3)
    assert not hasattr(layer, "_qvq_cpu_dense_inner_cache")


@pytest.mark.parametrize(("k", "n"), ((2048, 2048), (2048, 8192), (8192, 2048), (2048, 256)))
@pytest.mark.parametrize("bits", (2, 3.5, 4))
@pytest.mark.parametrize("m", (1, 8, 32))
def test_qvq_cpu_native_packed_gemv_accuracy_matrix(k, n, bits, m):
    from gptqmodel.utils.qvq_cpu import qvq_cpu_gemv

    transition_bits = qvq_transition_bits(bits)
    generator = torch.Generator().manual_seed(20260824 + transition_bits * 100 + m + n)
    edges = torch.randint(
        0,
        1 << transition_bits,
        (128, (k // 16) * (n // 16)),
        generator=generator,
        dtype=torch.int32,
    )
    trellis = planar_pack_rows(edges, transition_bits).T.contiguous()
    x = torch.randn((m, k), generator=generator)
    inner = reconstruct_qvq_inner_weight(trellis, bits=bits, in_features=k, out_features=n)
    reference = x @ inner

    actual = qvq_cpu_gemv(x, trellis, bits, out_features=n, use_dense_cache=False)

    torch.testing.assert_close(actual, reference, rtol=0, atol=2e-3)




def _assert_qvq_logit_metrics(reference: torch.Tensor, actual: torch.Tensor) -> None:
    reference_f32 = reference.float()
    actual_f32 = actual.float()
    forward_kl = F.kl_div(
        actual_f32.log_softmax(dim=-1),
        reference_f32.softmax(dim=-1),
        reduction="batchmean",
    )
    assert forward_kl.item() < 2e-5
    assert torch.equal(actual_f32.argmax(dim=-1), reference_f32.argmax(dim=-1))
    assert torch.equal(actual_f32.topk(5, dim=-1).indices, reference_f32.topk(5, dim=-1).indices)




def _hyb_inner_gemv_case(bits: int, *, m: int, k: int = 32, n: int = 32):
    generator = torch.Generator().manual_seed(20261400 + bits * 10 + m)
    states = _tail_biting_states(bits, tiles=(k // 16) * (n // 16), seed=20261500 + bits)
    trellis = pack_trellis_states(states, bits=bits)
    lut = torch.randn((512, 2), generator=generator).to(torch.float16)
    decoded = hyb_decode_states(states, lut).reshape(-1, 16, 16)
    inner = decoded.view(k // 16, n // 16, 16, 16).permute(0, 2, 1, 3).reshape(k, n).contiguous()
    x = torch.randn((m, k), generator=generator).to(torch.float16)
    reference = (x.float() @ inner.float()).to(torch.float16)
    return x, trellis, lut, reference


@pytest.mark.mps
@pytest.mark.skipif(not qvq_mps_supported(), reason="requires runtime Metal shaders")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("m", (1, 4))
def test_qvq_pgc16_mps_matches_torch_reconstruction(bits, m):
    x, trellis, reference = _qvq_inner_gemv_case(bits, m=m)

    result = qvq_mps_gemv(
        x.to("mps"),
        trellis.to("mps"),
        bits,
        out_features=reference.shape[1],
    ).cpu()

    torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)


@pytest.mark.mps
@pytest.mark.skipif(not qvq_mps_supported(), reason="requires runtime Metal shaders")
@pytest.mark.parametrize("bits", (1, 2, 5, 8))
@pytest.mark.parametrize("m", (16, 17, 32, 33))
def test_qvq_pgc16_mps_multirow_boundaries_match_torch(bits, m):
    x, trellis, reference = _qvq_inner_gemv_case(bits, m=m, k=16, n=32)

    result = qvq_mps_gemv(
        x.to("mps"),
        trellis.to("mps"),
        bits,
        out_features=reference.shape[1],
    ).cpu()

    torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)




@pytest.mark.mps
@pytest.mark.skipif(not qvq_mps_supported(), reason="requires runtime Metal shaders")
@pytest.mark.parametrize("bits", (1, 2, 5, 8))
def test_qvq_hyb_mps_reference_remains_a_correct_nonloadable_oracle(bits):
    x, trellis, lut, reference = _hyb_inner_gemv_case(bits, m=1)

    result = qvq_hyb_reference_mps_gemv(
        x.to("mps"),
        trellis.to("mps"),
        lut.to("mps"),
        bits,
        out_features=reference.shape[1],
    ).cpu()

    torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("m", (1, 4))
def test_qvq_pgc16_mlx_matches_torch_reconstruction(bits, m):
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")
    from gptqmodel.utils.qvq_mlx import qvq_mlx_gemv

    x, trellis, reference = _qvq_inner_gemv_case(bits, m=m)
    result = qvq_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(trellis.numpy()),
        bits,
        out_features=reference.shape[1],
    )

    np.testing.assert_allclose(np.asarray(result), reference.numpy(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("bits", (1, 2, 5, 8))
@pytest.mark.parametrize("m", (16, 17, 32, 33))
def test_qvq_pgc16_mlx_multirow_boundaries_match_torch(bits, m):
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")
    from gptqmodel.utils.qvq_mlx import qvq_mlx_gemv

    x, trellis, reference = _qvq_inner_gemv_case(bits, m=m, k=16, n=32)
    result = qvq_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(trellis.numpy()),
        bits,
        out_features=reference.shape[1],
    )

    np.testing.assert_allclose(np.asarray(result), reference.numpy(), rtol=1e-3, atol=1e-3)




@pytest.mark.parametrize("bits", (2, 5, 8))
def test_qvq_hyb_mlx_reference_remains_a_correct_nonloadable_oracle(bits):
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")
    from gptqmodel.utils.qvq_mlx import qvq_hyb_reference_mlx_gemv

    x, trellis, lut, reference = _hyb_inner_gemv_case(bits, m=1)
    result = qvq_hyb_reference_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(trellis.numpy()),
        mx.array(lut.numpy()),
        bits,
        out_features=reference.shape[1],
    )

    np.testing.assert_allclose(np.asarray(result), reference.numpy(), rtol=1e-3, atol=1e-3)
