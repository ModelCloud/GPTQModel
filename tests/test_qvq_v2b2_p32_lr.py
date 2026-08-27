# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Reference and MLX coverage for the GPU-oriented QVQ LR32 layout."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from gptqmodel.nn_modules.qlinear.qvq import (
    QVQLinear,
    qvq_local_ring_dense_oracle_forward,
)
from gptqmodel.quantization.config import FORMAT, QVQConfig
from gptqmodel.quantization.qvq import (
    QVQ_V2B2_P32_LR_RING_STEPS,
    QVQ_V2B2_P32_LR_RINGS_PER_TILE,
    decode_local_ring_states,
    decode_local_ring_tiles,
    local_ring_states_from_edges,
    pack_local_ring_states,
    pack_qvq_binary_bank_ids,
    reconstruct_local_ring_inner_weight,
    unpack_local_ring_edges,
    unpack_local_ring_states,
)
from gptqmodel.quantization.qvq_codecs import (
    pgc16_decode_states_v2_banked,
    pgc16_levels_for_version,
)
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.model import hf_gptqmodel_prepare_model_for_load, make_quant

LR_RATES = (1, 1.5, 2, 2.5, 3, 3.5)


@pytest.mark.parametrize(
    "m,k,n,output_fp32,expected",
    (
        (1, 2048, 256, False, 1),
        (1, 2048, 256, True, 16),
        (1, 2304, 256, True, 8),
        (1, 2048, 8192, True, 4),
        (2, 2048, 8192, True, 4),
        (1, 8192, 2048, True, 16),
        (3, 64, 16, False, 1),
        (4, 2048, 8192, False, 1),
        (4, 2048, 8192, True, 1),
        (8, 2048, 8192, True, 1),
        (16, 8192, 2048, True, 8),
        (16, 8192, 8192, True, 4),
    ),
)
def test_lr32_multirow_split_policy(m, k, n, output_fp32, expected):
    from gptqmodel.utils.qvq_mlx import _local_ring_multirow_split_k

    assert _local_ring_multirow_split_k(m, k, n, output_fp32=output_fp32) == expected


@pytest.mark.parametrize("n,expected", ((256, 16), (2048, 16), (8192, 16)))
def test_lr32_small_row_output_width_policy(n, expected):
    from gptqmodel.utils.qvq_mlx import _local_ring_small_output_width

    assert _local_ring_small_output_width(n) == expected


@pytest.mark.parametrize("k,expected", ((256, 1), (512, 1), (1024, 1), (2048, 2)))
def test_lr32_m1_n64_split_policy(k, expected):
    from gptqmodel.utils.qvq_mlx import _local_ring_m1_n64_split_k

    assert _local_ring_m1_n64_split_k(k) == expected


@pytest.mark.parametrize(
    "in_features, out_features, kernel_name, expected_split_count",
    (
        (2048, 240, "_local_ring_m1_fused_split16_kernel", None),
        (2048, 256, "_local_ring_m1_fused_split16_kernel", None),
        (2048, 2048, "_local_ring_m1_n64_n32pair_split8_kernel", None),
        (2304, 256, "_local_ring_m1_fused_split_kernel", None),
        (8192, 2048, "_local_ring_m1_n32_fused_split_kernel", 32),
    ),
)
def test_lr32_m1_fused_split_route_matches_torch_oracle(
    monkeypatch, in_features, out_features, kernel_name, expected_split_count
):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils import qvq_mlx

    monkeypatch.setattr(qvq_mlx, "_USE_LR_M1_N64_SPLIT2", False)
    monkeypatch.setattr(qvq_mlx, "_USE_LR_M1_N64_N32PAIR_SPLIT8", True)
    _, _, trellis, selectors = _random_lr_payload(
        2,
        tiles=(in_features // 32) * (out_features // 8),
    )
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([1], dtype=torch.uint8)
    x = torch.randn(1, in_features, dtype=torch.float32)
    expected = x @ reconstruct_local_ring_inner_weight(
        trellis,
        bits=2,
        in_features=in_features,
        out_features=out_features,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )
    selected = {"called": False}
    original = getattr(qvq_mlx, kernel_name)

    def observed(**kwargs):
        selected["called"] = True
        selected["split_count"] = kwargs.get("split_count")
        return original(**kwargs)

    monkeypatch.setattr(qvq_mlx, kernel_name, observed)
    actual = qvq_mlx.qvq_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(trellis.numpy()),
        2,
        out_features=out_features,
        bank_ids=mx.array(packed_selectors.numpy()),
        bank_alt_id=mx.array(bank_alt_id.numpy()),
        v2b2_p32_lr=True,
        output_fp32=True,
        _bank_alt_id_value=1,
    )
    mx.eval(actual)
    assert selected["called"]
    if expected_split_count is not None:
        assert selected["split_count"] == expected_split_count
    torch.testing.assert_close(torch.from_numpy(np.asarray(actual)), expected, rtol=0, atol=2e-2)


def _random_lr_payload(bits: float, tiles: int = 2):
    transition_bits = int(bits * 2)
    edges = torch.randint(
        0,
        1 << transition_bits,
        (tiles, QVQ_V2B2_P32_LR_RINGS_PER_TILE, QVQ_V2B2_P32_LR_RING_STEPS),
        dtype=torch.int64,
    )
    states = local_ring_states_from_edges(edges, bits=bits)
    trellis = pack_local_ring_states(states, bits=bits)
    selectors = torch.randint(
        0,
        2,
        (tiles * QVQ_V2B2_P32_LR_RINGS_PER_TILE,),
        dtype=torch.uint8,
    )
    return edges, states, trellis, selectors


@pytest.mark.parametrize("bits", LR_RATES)
def test_lr32_planar_pack_roundtrip_all_rates(bits):
    edges, states, trellis, _ = _random_lr_payload(bits)

    assert trellis.dtype == torch.int32
    assert torch.equal(unpack_local_ring_edges(trellis, bits=bits), edges)
    assert torch.equal(unpack_local_ring_states(trellis, bits=bits), states)
    assert torch.equal(decode_local_ring_states(trellis, bits=bits), states)


def test_lr32_rings_are_independent():
    edges, states, _, _ = _random_lr_payload(2, tiles=1)
    changed = edges.clone()
    changed[..., 1, :] ^= 3

    changed_states = local_ring_states_from_edges(changed, bits=2)
    assert torch.equal(changed_states[..., 0, :], states[..., 0, :])
    assert not torch.equal(changed_states[..., 1, :], states[..., 1, :])


@pytest.mark.parametrize("bits", (1, 2, 3, 3.5))
def test_lr32_decode_matches_banked_codec(bits):
    _, states, trellis, selectors = _random_lr_payload(bits, tiles=3)
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([2], dtype=torch.uint8)

    actual = decode_local_ring_tiles(
        trellis,
        bits=bits,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )
    bank_ids = selectors.reshape(3, 8).repeat_interleave(16, dim=1).mul(2)
    expected = pgc16_decode_states_v2_banked(
        states.reshape(3, 128),
        bank_ids,
        bits=bits,
        levels=pgc16_levels_for_version("pgc16-v1"),
    ).reshape(3, 8, 16, 2).to(torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_lr32_reconstruction_uses_k32_n8_logical_tiles():
    _, _, trellis, selectors = _random_lr_payload(2, tiles=4)
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([1], dtype=torch.uint8)
    decoded = decode_local_ring_tiles(
        trellis,
        bits=2,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )
    expected = (
        decoded.reshape(2, 2, 8, 16, 2)
        .reshape(2, 2, 8, 32)
        .permute(0, 3, 1, 2)
        .reshape(64, 16)
    )
    actual = reconstruct_local_ring_inner_weight(
        trellis,
        bits=2,
        in_features=64,
        out_features=16,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _make_torch_lr_layer(bits=2, in_features=64, out_features=16):
    _, _, trellis, selectors = _random_lr_payload(
        bits,
        tiles=(in_features // 32) * (out_features // 8),
    )
    return QVQLinear(
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_count=2,
        v2b2_p32_lr=True,
        dtype=torch.float32,
        tensors={
            "trellis": trellis,
            "SU": torch.rand(in_features, dtype=torch.float32) + 0.5,
            "SV": torch.rand(out_features, dtype=torch.float32) + 0.5,
            "bank_ids": pack_qvq_binary_bank_ids(selectors),
            "bank_alt_id": torch.tensor([2], dtype=torch.uint8),
        },
    ).eval()


def test_lr32_qvq_linear_oracle_uses_torch_reference_path():
    layer = _make_torch_lr_layer()
    x = torch.randn(5, layer.in_features, dtype=torch.float32)
    oracle = qvq_local_ring_dense_oracle_forward(layer, x)

    assert layer.v2b2_p32_lr
    assert layer.trellis.shape == (4, 16)
    assert oracle.shape == (5, layer.out_features)
    torch.testing.assert_close(layer(x), oracle, rtol=0, atol=0)


def test_lr32_qvq_linear_state_dict_round_trip_preserves_oracle():
    layer = _make_torch_lr_layer(bits=2, in_features=64, out_features=8)
    reloaded = QVQLinear(
        bits=2,
        in_features=64,
        out_features=8,
        bank_count=2,
        v2b2_p32_lr=True,
        dtype=torch.float32,
    ).eval()
    reloaded.load_state_dict(layer.state_dict(), strict=True)
    x = torch.randn(3, 64, dtype=torch.float32)

    for name, tensor in layer.state_dict().items():
        torch.testing.assert_close(tensor, reloaded.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(
        qvq_local_ring_dense_oracle_forward(reloaded, x),
        qvq_local_ring_dense_oracle_forward(layer, x),
        rtol=0,
        atol=0,
    )


def test_lr32_rejects_legacy_or_multiple_format_flags():
    with pytest.raises(ValueError, match="mutually exclusive"):
        QVQLinear(
            bits=2,
            in_features=32,
            out_features=8,
            bank_count=2,
            v2b2_p32=True,
            v2b2_p32_lr=True,
        )


def test_lr32_config_and_format_metadata():
    config = QVQConfig(bits=2, format=FORMAT.QVQ_V2B2_P32_LR, bank_count=2)
    assert config.format == FORMAT.QVQ_V2B2_P32_LR
    assert config.quant_linear_init_kwargs()["v2b2_p32_lr"] is True
    assert config.quant_linear_init_kwargs()["v2b2_p32"] is False
    assert QVQLinear.supported_bits(FORMAT.QVQ_V2B2_P32_LR) == (1, 1.5, 2, 2.5, 3, 3.5)


def test_lr32_make_quant_preserves_layout_format_for_n8_modules():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(32, 8, bias=False, dtype=torch.float16)

    model = TinyModel()
    config = QVQConfig(
        bits=2,
        format=FORMAT.QVQ_V2B2_P32_LR,
        bank_count=2,
        rounding="block_ldlq",
        offload_to_disk=False,
    )

    selected = make_quant(
        model,
        config,
        {"proj": {}},
        BACKEND.QVQ,
        "lm_head",
        device="cpu",
        dtype=torch.float16,
    )

    assert selected is QVQLinear
    assert model.proj.v2b2_p32_lr is True
    assert model.proj.trellis.shape == (1, 16)


def test_lr32_reconstruction_pins_k32_n8_abi_coordinates():
    # Every tile/ring gets a distinct constant edge pattern. The expected
    # values below are scalar decoder lookups, not a second reshape/permute
    # implementation, so this test pins the serialized K32 x N8 ABI.
    edge_count = 4
    edges = torch.empty((4, 8, 16), dtype=torch.int64)
    for tile in range(4):
        for ring in range(8):
            edges[tile, ring] = (tile * 8 + ring + 1) % (1 << edge_count)
    states = local_ring_states_from_edges(edges, bits=2)
    trellis = pack_local_ring_states(states, bits=2)
    selectors = torch.zeros(32, dtype=torch.uint8)
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    decoded = pgc16_decode_states_v2_banked(
        states.reshape(4, 128),
        selectors.reshape(4, 8).repeat_interleave(16, dim=1),
        bits=2,
        levels=pgc16_levels_for_version("pgc16-v1"),
    ).reshape(4, 8, 16, 2)
    actual = reconstruct_local_ring_inner_weight(
        trellis,
        bits=2,
        in_features=64,
        out_features=16,
        bank_ids=packed_selectors,
        bank_alt_id=torch.tensor([1], dtype=torch.uint8),
    )

    torch.testing.assert_close(actual[0, 0], decoded[0, 0, 0, 0].to(torch.float32))
    torch.testing.assert_close(actual[31, 0], decoded[0, 0, 15, 1].to(torch.float32))
    torch.testing.assert_close(actual[0, 7], decoded[0, 7, 0, 0].to(torch.float32))
    torch.testing.assert_close(actual[32, 0], decoded[2, 0, 0, 0].to(torch.float32))
    torch.testing.assert_close(actual[0, 8], decoded[1, 0, 0, 0].to(torch.float32))


def test_legacy_v2_cpu_dispatch_does_not_pass_lr_keyword(monkeypatch):
    from gptqmodel.utils import qvq_cpu

    calls = {}

    def fake_gemv(x, trellis, bits, **kwargs):
        calls.update(kwargs)
        return torch.zeros((x.shape[0], kwargs["out_features"]), dtype=torch.float32)

    monkeypatch.setattr(qvq_cpu, "qvq_cpu_supported", lambda: True)
    monkeypatch.setattr(qvq_cpu, "qvq_cpu_gemv", fake_gemv)
    _, _, trellis, selectors = _random_lr_payload(2, tiles=2)
    layer = QVQLinear(
        bits=2,
        in_features=32,
        out_features=16,
        bank_count=2,
        v2b2_p32=True,
        dtype=torch.float32,
        tensors={
            "trellis": trellis,
            "SU": torch.ones(32),
            "SV": torch.ones(16),
            "bank_ids": pack_qvq_binary_bank_ids(selectors),
            "bank_alt_id": torch.tensor([1], dtype=torch.uint8),
        },
    ).eval()

    output = layer._inner_forward(torch.ones(1, 32))

    assert output.shape == (1, 16)
    assert "v2b2_p32_lr" not in calls
    assert calls["v2b2_p32"] is True


def test_lr32_checkpoint_shell_loads_n8_payload_and_converts_to_mlx(tmp_path):
    mx = pytest.importorskip("mlx.core")

    class TinyCheckpointModel(nn.Module):
        def __init__(self, checkpoint_path):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="llama",
                _name_or_path=str(checkpoint_path),
                dtype=torch.float16,
            )
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module()])
            self.model.layers[0].proj = nn.Linear(32, 8, bias=False, device="meta")

    (tmp_path / "quantize_config.json").write_text(
        '{"bits": 2, "group_size": -1, "desc_act": false, "sym": true, '
        '"format": "qvq_v2b2_p32_lr", "method": "qvq", "bank_count": 2, '
        '"codebook": "pgc16-v1", "trellis_window": 16, "vector_size": 2}',
        encoding="utf-8",
    )
    prefix = "model.layers.0.proj"
    save_file(
        {
            f"{prefix}.trellis": torch.zeros((1, 16), dtype=torch.int32),
            f"{prefix}.SU": torch.ones(32, dtype=torch.float32),
            f"{prefix}.SV": torch.ones(8, dtype=torch.float32),
            f"{prefix}.bank_ids": torch.zeros(1, dtype=torch.uint8),
            f"{prefix}.bank_alt_id": torch.ones(1, dtype=torch.uint8),
        },
        tmp_path / "model.safetensors",
    )

    model = TinyCheckpointModel(tmp_path)
    context = hf_gptqmodel_prepare_model_for_load(
        model,
        checkpoint_files=[tmp_path / "model.safetensors"],
        device_map={"": "cpu"},
        backend=BACKEND.QVQ,
        dtype=torch.float16,
    )

    layer = model.model.layers[0].proj
    assert context is not None
    assert context.quantize_config.format == FORMAT.QVQ_V2B2_P32_LR
    assert isinstance(layer, QVQLinear)
    assert layer.v2b2_p32_lr is True
    assert layer.trellis.device.type == "meta"

    loaded = load_file(tmp_path / "model.safetensors")
    model.load_state_dict(loaded, strict=True, assign=True)
    layer = model.model.layers[0].proj.eval()
    x = torch.randn(2, 32)
    expected = qvq_local_ring_dense_oracle_forward(layer, x)
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=2e-3)

    from gptqmodel.utils.mlx import _qvq_mlx_linear_from_torch

    mlx_layer = _qvq_mlx_linear_from_torch(layer)
    actual = mlx_layer(mx.array(x.numpy()))
    mx.eval(actual)
    torch.testing.assert_close(
        torch.from_numpy(np.asarray(actual)), expected, rtol=0, atol=2e-2
    )


@pytest.mark.parametrize("output_fp32", (False, True))
@pytest.mark.parametrize("out_features", (8, 16, 40))
@pytest.mark.parametrize("bits", LR_RATES)
def test_lr32_mlx_gpu_kernel_matches_torch_reconstruction(
    output_fp32,
    out_features,
    bits,
):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils.qvq_mlx import qvq_mlx_gemv

    in_features = 64
    tile_count = (in_features // 32) * (out_features // 8)
    _, _, trellis, selectors = _random_lr_payload(bits, tiles=tile_count)
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([2], dtype=torch.uint8)
    x = torch.randn(5, in_features, dtype=torch.float16)
    inner = reconstruct_local_ring_inner_weight(
        trellis,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )
    expected = x.to(torch.float32) @ inner

    actual = qvq_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(trellis.numpy()),
        bits,
        out_features=out_features,
        bank_ids=mx.array(packed_selectors.numpy()),
        bank_alt_id=mx.array(bank_alt_id.numpy()),
        v2b2_p32_lr=True,
        output_fp32=output_fp32,
    )
    mx.eval(actual)
    actual_torch = torch.from_numpy(np.asarray(actual))
    expected = expected if output_fp32 else expected.to(torch.float16)
    torch.testing.assert_close(actual_torch, expected, rtol=0, atol=8e-3)


def test_lr32_mlx_gpu_kernel_accepts_fp32_activation():
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils.qvq_mlx import qvq_mlx_gemv

    bits = 2
    in_features = 64
    out_features = 8
    _, _, trellis, selectors = _random_lr_payload(bits, tiles=(in_features // 32) * (out_features // 8))
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([2], dtype=torch.uint8)
    x = torch.randn(3, in_features, dtype=torch.float32)
    expected = x @ reconstruct_local_ring_inner_weight(
        trellis,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )

    actual = qvq_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(trellis.numpy()),
        bits,
        out_features=out_features,
        bank_ids=mx.array(packed_selectors.numpy()),
        bank_alt_id=mx.array(bank_alt_id.numpy()),
        v2b2_p32_lr=True,
        output_fp32=True,
    )
    mx.eval(actual)
    torch.testing.assert_close(torch.from_numpy(np.asarray(actual)), expected, rtol=0, atol=8e-3)


@pytest.mark.parametrize("bank_alt_value", (1, 2, 3))
def test_lr32_mlx_small_w2_literal_alt_bank_masks_match_torch(bank_alt_value):
    """Exercise the small-row W2 literal-mask specialization for every bank."""

    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils.qvq_mlx import qvq_mlx_gemv

    bits = 2
    in_features = 64
    out_features = 16
    _, _, trellis, selectors = _random_lr_payload(
        bits,
        tiles=(in_features // 32) * (out_features // 8),
    )
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([bank_alt_value], dtype=torch.uint8)
    x = torch.randn(1, in_features, dtype=torch.float32)
    expected = x @ reconstruct_local_ring_inner_weight(
        trellis,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )

    actual = qvq_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(trellis.numpy()),
        bits,
        out_features=out_features,
        bank_ids=mx.array(packed_selectors.numpy()),
        bank_alt_id=mx.array(bank_alt_id.numpy()),
        v2b2_p32_lr=True,
        output_fp32=True,
        _bank_alt_id_value=bank_alt_value,
    )
    mx.eval(actual)
    torch.testing.assert_close(torch.from_numpy(np.asarray(actual)), expected, rtol=0, atol=2e-2)


def test_lr32_mlx_linear_validates_missing_alt_bank_before_scalar_access():
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils.qvq_mlx import QVQMLXLinear

    with pytest.raises(ValueError, match="bank_alt_id"):
        QVQMLXLinear(
            bits=2,
            in_features=32,
            out_features=8,
            trellis=mx.zeros((1, 16), dtype=mx.int32),
            SU=mx.ones((32,), dtype=mx.float32),
            SV=mx.ones((8,), dtype=mx.float32),
            bank_ids=mx.zeros((1,), dtype=mx.uint8),
            v2b2_p32_lr=True,
        )


@pytest.mark.parametrize("output_fp32", (False, True))
def test_lr32_mlx_small_row_kernel_handles_two_rows(output_fp32):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils.qvq_mlx import qvq_mlx_gemv

    bits = 2
    in_features = 64
    out_features = 16
    torch_layer = _make_torch_lr_layer(bits=bits, in_features=in_features, out_features=out_features)
    x = torch.randn(2, in_features, dtype=torch.float16)
    expected = x.to(torch.float32) @ reconstruct_local_ring_inner_weight(
        torch_layer.trellis,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=torch_layer.bank_ids,
        bank_alt_id=torch_layer.bank_alt_id,
    )
    actual = qvq_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(torch_layer.trellis.numpy()),
        bits,
        out_features=out_features,
        bank_ids=mx.array(torch_layer.bank_ids.numpy()),
        bank_alt_id=mx.array(torch_layer.bank_alt_id.numpy()),
        v2b2_p32_lr=True,
        output_fp32=output_fp32,
    )
    mx.eval(actual)
    expected = expected if output_fp32 else expected.to(torch.float16)
    torch.testing.assert_close(torch.from_numpy(np.asarray(actual)), expected, rtol=0, atol=8e-3)


@pytest.mark.parametrize("output_fp32", (False, True))
@pytest.mark.parametrize("bits", LR_RATES)
def test_lr32_mlx_mma_m8_matches_torch_reconstruction(bits, output_fp32):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils import qvq_mlx

    in_features = 32
    out_features = 8192
    torch_layer = _make_torch_lr_layer(
        bits=bits,
        in_features=in_features,
        out_features=out_features,
    )
    x = torch.randn(8, in_features, dtype=torch.float16)
    expected = x.to(torch.float32) @ reconstruct_local_ring_inner_weight(
        torch_layer.trellis,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=torch_layer.bank_ids,
        bank_alt_id=torch_layer.bank_alt_id,
    )
    actual = qvq_mlx._local_ring_mma_kernel(
        output_fp32=output_fp32,
        w2=int(bits * 2) == 4,
        alt_bank_id=2,
    )(
        inputs=[
            mx.array(x.numpy()),
            mx.array(torch_layer.trellis.numpy()),
            mx.array(torch_layer.bank_ids.numpy()),
            qvq_mlx._dims_array(8, in_features, out_features, int(bits * 2), 8),
        ],
        template=[("EdgeBits", int(bits * 2)), ("AltBank", 2)],
        grid=(out_features // 8 * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(8, out_features)],
        output_dtypes=[mx.float32 if output_fp32 else mx.float16],
    )[0]
    mx.eval(actual)
    expected = expected if output_fp32 else expected.to(torch.float16)
    torch.testing.assert_close(torch.from_numpy(np.asarray(actual)), expected, rtol=0, atol=8e-3)


def test_lr32_mlx_mma_is_selected_for_production_m8_dispatch(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils import qvq_mlx

    in_features = 32
    out_features = 8192
    _, _, trellis, selectors = _random_lr_payload(2, tiles=out_features // 8)
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([2], dtype=torch.uint8)
    x = torch.randn(8, in_features, dtype=torch.float32)
    selected = {"called": False}
    original = qvq_mlx._local_ring_mma_kernel

    def observed(**kwargs):
        selected["called"] = True
        return original(**kwargs)

    monkeypatch.setattr(qvq_mlx, "_USE_LR_MMA", True)
    monkeypatch.setattr(qvq_mlx, "_local_ring_mma_kernel", observed)
    actual = qvq_mlx.qvq_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(trellis.numpy()),
        2,
        out_features=out_features,
        bank_ids=mx.array(packed_selectors.numpy()),
        bank_alt_id=mx.array(bank_alt_id.numpy()),
        v2b2_p32_lr=True,
        output_fp32=True,
        _bank_alt_id_value=2,
    )
    mx.eval(actual)
    assert selected["called"]


@pytest.mark.parametrize("output_fp32", (False, True))
@pytest.mark.parametrize("bits", LR_RATES)
def test_lr32_mlx_m4_row_tile_matches_torch_reconstruction(bits, output_fp32):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils.qvq_mlx import qvq_mlx_gemv

    in_features = 64
    out_features = 16
    torch_layer = _make_torch_lr_layer(
        bits=bits,
        in_features=in_features,
        out_features=out_features,
    )
    x = torch.randn(4, in_features, dtype=torch.float16)
    expected = x.to(torch.float32) @ reconstruct_local_ring_inner_weight(
        torch_layer.trellis,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=torch_layer.bank_ids,
        bank_alt_id=torch_layer.bank_alt_id,
    )
    actual = qvq_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(torch_layer.trellis.numpy()),
        bits,
        out_features=out_features,
        bank_ids=mx.array(torch_layer.bank_ids.numpy()),
        bank_alt_id=mx.array(torch_layer.bank_alt_id.numpy()),
        v2b2_p32_lr=True,
        output_fp32=output_fp32,
    )
    mx.eval(actual)
    actual_torch = torch.from_numpy(np.asarray(actual))
    expected = expected if output_fp32 else expected.to(torch.float16)
    torch.testing.assert_close(actual_torch, expected, rtol=0, atol=8e-3)


def test_lr32_m4_cooperative_decode_falls_back_for_split_k(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils import qvq_mlx

    monkeypatch.setattr(qvq_mlx, "_USE_LR_M4_COOPERATIVE_DECODE", True)
    monkeypatch.setattr(qvq_mlx, "_lr_m4_cooperative_supported", lambda: True)
    original = qvq_mlx._local_ring_multirow_kernel
    observed = {}

    def selected(**kwargs):
        observed["split_k"] = kwargs["split_k"]
        observed["m4_cooperative_decode"] = kwargs["m4_cooperative_decode"]
        return original(**kwargs)

    monkeypatch.setattr(qvq_mlx, "_local_ring_multirow_kernel", selected)
    m, k, n = 4, 8192, 2048
    tile_count = (k // 32) * (n // 8)
    rng = np.random.default_rng(20260827)
    actual = qvq_mlx.qvq_mlx_gemv(
        mx.array(rng.standard_normal((m, k)).astype(np.float32)),
        mx.array(rng.integers(-2**31, 2**31 - 1, (tile_count, 16), dtype=np.int64).astype(np.int32)),
        2,
        out_features=n,
        bank_ids=mx.array(rng.integers(0, 4, (tile_count,), dtype=np.uint8)),
        bank_alt_id=mx.array(np.array([2], dtype=np.uint8)),
        v2b2_p32_lr=True,
        output_fp32=True,
        _bank_alt_id_value=2,
    )
    mx.eval(actual)
    assert observed == {"split_k": 8, "m4_cooperative_decode": False}


@pytest.mark.parametrize("output_fp32", (False, True))
@pytest.mark.parametrize("rows", (4, 8, 16))
@pytest.mark.parametrize("bits", LR_RATES)
def test_lr32_mlx_cooperative_decode_matches_torch_reconstruction(rows, bits, output_fp32, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils import qvq_mlx

    monkeypatch.setattr(qvq_mlx, "_USE_LR_COOPERATIVE_DECODE", True)

    in_features = 64
    out_features = 16
    torch_layer = _make_torch_lr_layer(
        bits=bits,
        in_features=in_features,
        out_features=out_features,
    )
    x = torch.randn(rows, in_features, dtype=torch.float16)
    expected = x.to(torch.float32) @ reconstruct_local_ring_inner_weight(
        torch_layer.trellis,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=torch_layer.bank_ids,
        bank_alt_id=torch_layer.bank_alt_id,
    )
    actual = qvq_mlx.qvq_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(torch_layer.trellis.numpy()),
        bits,
        out_features=out_features,
        bank_ids=mx.array(torch_layer.bank_ids.numpy()),
        bank_alt_id=mx.array(torch_layer.bank_alt_id.numpy()),
        v2b2_p32_lr=True,
        output_fp32=output_fp32,
    )
    mx.eval(actual)
    expected = expected if output_fp32 else expected.to(torch.float16)
    torch.testing.assert_close(torch.from_numpy(np.asarray(actual)), expected, rtol=0, atol=8e-3)


@pytest.mark.parametrize("out_features", (8, 16, 40))
def test_lr32_mlx_linear_inference_matches_torch_oracle(out_features):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils.qvq_mlx import QVQMLXLinear

    torch_layer = _make_torch_lr_layer(
        bits=2,
        in_features=64,
        out_features=out_features,
    )
    mlx_layer = QVQMLXLinear(
        bits=torch_layer.bits,
        in_features=torch_layer.in_features,
        out_features=torch_layer.out_features,
        trellis=mx.array(torch_layer.trellis.numpy()),
        SU=mx.array(torch_layer.SU.numpy()),
        SV=mx.array(torch_layer.SV.numpy()),
        codebook_version=torch_layer.codebook_version,
        vector_size=torch_layer.vector_size,
        trellis_window=torch_layer.trellis_window,
        bank_ids=mx.array(torch_layer.bank_ids.numpy()),
        v2b2_p32_lr=True,
        bank_alt_id=mx.array(torch_layer.bank_alt_id.numpy()),
    )
    x = torch.randn(3, 64, dtype=torch.float16)
    expected = qvq_local_ring_dense_oracle_forward(torch_layer, x)
    actual = mlx_layer(mx.array(x.numpy()))
    mx.eval(actual)
    torch.testing.assert_close(
        torch.from_numpy(np.asarray(actual)).to(torch.float32),
        expected,
        rtol=0,
        atol=2e-2,
    )


@pytest.mark.parametrize("bits", LR_RATES)
def test_lr32_mlx_single_row_split_inference_matches_torch_oracle(bits):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils.qvq_mlx import qvq_mlx_gemv

    torch_layer = _make_torch_lr_layer(bits=bits, in_features=64, out_features=16)
    x = torch.randn(1, 64, dtype=torch.float16)
    expected = x.to(torch.float32) @ reconstruct_local_ring_inner_weight(
        torch_layer.trellis,
        bits=bits,
        in_features=64,
        out_features=16,
        bank_ids=torch_layer.bank_ids,
        bank_alt_id=torch_layer.bank_alt_id,
    )
    actual = qvq_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(torch_layer.trellis.numpy()),
        bits,
        out_features=16,
        bank_ids=mx.array(torch_layer.bank_ids.numpy()),
        bank_alt_id=mx.array(torch_layer.bank_alt_id.numpy()),
        v2b2_p32_lr=True,
        output_fp32=True,
    )
    mx.eval(actual)
    torch.testing.assert_close(torch.from_numpy(np.asarray(actual)), expected, rtol=0, atol=2e-2)


@pytest.mark.parametrize("out_features", (64, 128))
def test_lr32_mlx_m1_n64_split2_kernel_matches_torch_oracle(monkeypatch, out_features):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils import qvq_mlx

    # This is an opt-in grouped-kernel oracle test.  Production dispatch keeps
    # the barrier-free M1/N16 route for ordinary narrow-N projections after
    # the M4 Max A/B showed it is faster there.
    monkeypatch.setattr(qvq_mlx, "_USE_LR_M1_N64_SPLIT2", True)
    # The fused two-way path requires each split to contain at least one
    # complete K32 pair tile.
    in_features = 128
    _, _, trellis, selectors = _random_lr_payload(
        2,
        tiles=(in_features // 32) * (out_features // 8),
    )
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([2], dtype=torch.uint8)
    x = torch.randn(1, in_features, dtype=torch.float32)
    expected = x @ reconstruct_local_ring_inner_weight(
        trellis,
        bits=2,
        in_features=in_features,
        out_features=out_features,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )
    selected = {"called": False}
    original = qvq_mlx._local_ring_m1_n64_split2_kernel

    def observed(**kwargs):
        selected["called"] = True
        return original(**kwargs)

    monkeypatch.setattr(qvq_mlx, "_local_ring_m1_n64_split2_kernel", observed)
    actual = qvq_mlx.qvq_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(trellis.numpy()),
        2,
        out_features=out_features,
        bank_ids=mx.array(packed_selectors.numpy()),
        bank_alt_id=mx.array(bank_alt_id.numpy()),
        v2b2_p32_lr=True,
        output_fp32=True,
        _bank_alt_id_value=2,
    )
    mx.eval(actual)
    assert selected["called"]
    torch.testing.assert_close(torch.from_numpy(np.asarray(actual)), expected, rtol=0, atol=2e-2)


def test_lr32_mlx_m1_n64_shared_split2_kernel_matches_torch_oracle(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils import qvq_mlx

    monkeypatch.setattr(qvq_mlx, "_USE_LR_M1_N64_SHARED_SPLIT2", True)
    in_features = 4096
    out_features = 8192
    _, _, trellis, selectors = _random_lr_payload(
        2,
        tiles=(in_features // 32) * (out_features // 8),
    )
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([3], dtype=torch.uint8)
    x = torch.randn(1, in_features, dtype=torch.float32)
    expected = x @ reconstruct_local_ring_inner_weight(
        trellis,
        bits=2,
        in_features=in_features,
        out_features=out_features,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )
    selected = {"called": False}
    original = qvq_mlx._local_ring_m1_n64_shared_split2_kernel

    def observed(**kwargs):
        selected["called"] = True
        return original(**kwargs)

    monkeypatch.setattr(qvq_mlx, "_local_ring_m1_n64_shared_split2_kernel", observed)
    actual = qvq_mlx.qvq_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(trellis.numpy()),
        2,
        out_features=out_features,
        bank_ids=mx.array(packed_selectors.numpy()),
        bank_alt_id=mx.array(bank_alt_id.numpy()),
        v2b2_p32_lr=True,
        output_fp32=True,
        _bank_alt_id_value=3,
    )
    mx.eval(actual)
    assert selected["called"]
    torch.testing.assert_close(torch.from_numpy(np.asarray(actual)), expected, rtol=0, atol=2e-2)


def test_lr32_mlx_single_row_specialization_matches_two_row_kernel(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils import qvq_mlx

    torch_layer = _make_torch_lr_layer(bits=2, in_features=64, out_features=16)
    x = torch.randn(1, 64, dtype=torch.float16)
    inputs = (
        mx.array(x.numpy()),
        mx.array(torch_layer.trellis.numpy()),
        mx.array(torch_layer.bank_ids.numpy()),
        mx.array(torch_layer.bank_alt_id.numpy()),
    )
    original = qvq_mlx._local_ring_small_kernel

    def run(single_row):
        def selected(**kwargs):
            kwargs["single_row"] = single_row
            kwargs["vector_activation"] = single_row
            return original(**kwargs)

        monkeypatch.setattr(qvq_mlx, "_local_ring_small_kernel", selected)
        actual = qvq_mlx.qvq_mlx_gemv(
            inputs[0],
            inputs[1],
            2,
            out_features=16,
            bank_ids=inputs[2],
            bank_alt_id=inputs[3],
            v2b2_p32_lr=True,
            output_fp32=True,
            _bank_alt_id_value=2,
        )
        mx.eval(actual)
        return np.asarray(actual)

    specialized = run(True)
    fallback = run(False)
    np.testing.assert_allclose(specialized, fallback, rtol=0, atol=2e-2)


@pytest.mark.parametrize("bank_alt_value", [1, 2, 3])
def test_lr32_m1_w2_n64_grouped_kernel_matches_torch_oracle(monkeypatch, bank_alt_value):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils import qvq_mlx

    in_features = 64
    out_features = 8192
    _, _, trellis, selectors = _random_lr_payload(
        2,
        tiles=(in_features // 32) * (out_features // 8),
    )
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([bank_alt_value], dtype=torch.uint8)
    x = torch.randn(1, in_features, dtype=torch.float32)
    expected = x @ reconstruct_local_ring_inner_weight(
        trellis,
        bits=2,
        in_features=in_features,
        out_features=out_features,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )
    selected = {"called": False}
    original = qvq_mlx._local_ring_m1_n64_kernel

    def observed(**kwargs):
        selected["called"] = True
        return original(**kwargs)

    monkeypatch.setattr(qvq_mlx, "_local_ring_m1_n64_kernel", observed)
    actual = qvq_mlx.qvq_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(trellis.numpy()),
        2,
        out_features=out_features,
        bank_ids=mx.array(packed_selectors.numpy()),
        bank_alt_id=mx.array(bank_alt_id.numpy()),
        v2b2_p32_lr=True,
        output_fp32=True,
        _bank_alt_id_value=bank_alt_value,
    )
    mx.eval(actual)
    assert selected["called"]
    torch.testing.assert_close(torch.from_numpy(np.asarray(actual)), expected, rtol=0, atol=2e-2)


def test_lr32_m1_n64_grouped_kernel_is_reserved_for_wide_shapes(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils import qvq_mlx

    in_features = 64
    out_features = 2048
    _, _, trellis, selectors = _random_lr_payload(
        2,
        tiles=(in_features // 32) * (out_features // 8),
    )
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([1], dtype=torch.uint8)
    x = torch.randn(1, in_features, dtype=torch.float32)
    expected = x @ reconstruct_local_ring_inner_weight(
        trellis,
        bits=2,
        in_features=in_features,
        out_features=out_features,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )

    def unexpected_grouped_kernel(**kwargs):
        pytest.fail(f"N64 M1 route selected for non-wide shape: {kwargs}")

    monkeypatch.setattr(qvq_mlx, "_local_ring_m1_n64_kernel", unexpected_grouped_kernel)
    actual = qvq_mlx.qvq_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(trellis.numpy()),
        2,
        out_features=out_features,
        bank_ids=mx.array(packed_selectors.numpy()),
        bank_alt_id=mx.array(bank_alt_id.numpy()),
        v2b2_p32_lr=True,
        output_fp32=True,
        _bank_alt_id_value=1,
    )
    mx.eval(actual)
    torch.testing.assert_close(torch.from_numpy(np.asarray(actual)), expected, rtol=0, atol=2e-2)


def test_lr32_m1_w2_n64_grouped_kernel_handles_strided_input():
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils import qvq_mlx

    in_features = 64
    out_features = 8192
    _, _, trellis, selectors = _random_lr_payload(
        2,
        tiles=(in_features // 32) * (out_features // 8),
    )
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([1], dtype=torch.uint8)
    x = torch.randn(1, in_features, dtype=torch.float32)
    expected = x @ reconstruct_local_ring_inner_weight(
        trellis,
        bits=2,
        in_features=in_features,
        out_features=out_features,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )
    contiguous = mx.array(x.numpy())
    flat = contiguous.reshape(-1)
    strided = mx.stack((flat, flat), axis=0)[::2]
    actual = qvq_mlx.qvq_mlx_gemv(
        strided,
        mx.array(trellis.numpy()),
        2,
        out_features=out_features,
        bank_ids=mx.array(packed_selectors.numpy()),
        bank_alt_id=mx.array(bank_alt_id.numpy()),
        v2b2_p32_lr=True,
        output_fp32=True,
        _bank_alt_id_value=1,
    )
    mx.eval(actual)
    torch.testing.assert_close(torch.from_numpy(np.asarray(actual)), expected, rtol=0, atol=2e-2)


def test_lr32_m1_w2_n64_k128_grouped_kernel_matches_torch_oracle(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils import qvq_mlx

    in_features = 128
    out_features = 8192
    _, _, trellis, selectors = _random_lr_payload(
        2,
        tiles=(in_features // 32) * (out_features // 8),
    )
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([2], dtype=torch.uint8)
    x = torch.randn(1, in_features, dtype=torch.float32)
    expected = x @ reconstruct_local_ring_inner_weight(
        trellis,
        bits=2,
        in_features=in_features,
        out_features=out_features,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )
    selected = {}
    original = qvq_mlx._local_ring_m1_n64_kernel

    def observed(**kwargs):
        selected.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(qvq_mlx, "_local_ring_m1_n64_kernel", observed)
    actual = qvq_mlx.qvq_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(trellis.numpy()),
        2,
        out_features=out_features,
        bank_ids=mx.array(packed_selectors.numpy()),
        bank_alt_id=mx.array(bank_alt_id.numpy()),
        v2b2_p32_lr=True,
        output_fp32=True,
        _bank_alt_id_value=2,
    )
    mx.eval(actual)
    assert selected["k_tile"] == 128
    torch.testing.assert_close(torch.from_numpy(np.asarray(actual)), expected, rtol=0, atol=2e-2)


def test_lr32_m1_w2_n64_k128_and_split2_match_torch_oracle():
    """Exercise K128 reuse and the production split-2 path against Torch."""

    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils import qvq_mlx

    in_features = 2048
    out_features = 8192
    torch.manual_seed(2048)
    tile_count = (in_features // 32) * (out_features // 8)
    trellis = torch.randint(
        -(1 << 31),
        1 << 31,
        (tile_count, 16),
        dtype=torch.int32,
    )
    selectors = torch.randint(0, 2, (tile_count * 8,), dtype=torch.uint8)
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([2], dtype=torch.uint8)
    x = torch.randn(1, in_features, dtype=torch.float32)
    expected = x @ reconstruct_local_ring_inner_weight(
        trellis,
        bits=2,
        in_features=in_features,
        out_features=out_features,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )

    # Exercise the multi-batch K128 source directly. Production dispatch uses
    # K64 for K>128 because K128 needs the reuse hand-off barrier.
    kernel = qvq_mlx._local_ring_m1_n64_kernel(alt_bank_id=2, k=in_features, n=out_features, k_tile=128)
    actual_k128 = kernel(
        inputs=[mx.array(x.numpy()), mx.array(trellis.numpy()), mx.array(packed_selectors.numpy())],
        template=[("EdgeBits", 4), ("AltBank", 2), ("SplitK", 1)],
        grid=(out_features // 64 * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, out_features)],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(actual_k128)
    torch.testing.assert_close(torch.from_numpy(np.asarray(actual_k128)), expected, rtol=0, atol=2e-2)

    # The public path now uses two K64 slices for sufficiently large wide M1
    # shapes and reduces the interleaved partial output on the MLX side.
    for _ in range(3):
        actual_split2 = qvq_mlx.qvq_mlx_gemv(
            mx.array(x.numpy()),
            mx.array(trellis.numpy()),
            2,
            out_features=out_features,
            bank_ids=mx.array(packed_selectors.numpy()),
            bank_alt_id=mx.array(bank_alt_id.numpy()),
            v2b2_p32_lr=True,
            output_fp32=True,
            _bank_alt_id_value=2,
        )
        mx.eval(actual_split2)
        torch.testing.assert_close(torch.from_numpy(np.asarray(actual_split2)), expected, rtol=0, atol=2e-2)
