# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# NF12 reference: AlpinDale and the nf12 contributors, Apache-2.0, https://github.com/dphnAI/nf12
"""Packed NF12 MLX decode checked against the lossless Torch codec."""

import numpy as np
import pytest
import torch

from tests import accelerator_gates


mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

_METAL_SKIP_REASON = accelerator_gates.mlx_metal_skip_reason()
requires_metal = pytest.mark.skipif(
    _METAL_SKIP_REASON is not None,
    reason=_METAL_SKIP_REASON or "MLX Metal backend is unavailable",
)


def _source(in_features=256, out_features=48, *, escapes=False):
    from gptqmodel.nn_modules.qlinear.nf12 import NF12Linear

    torch.manual_seed(23)
    linear = torch.nn.Linear(in_features, out_features, bias=True)
    if escapes:
        # Force escaped groups: one group spans the whole BF16 exponent range.
        linear.weight.data[0, :8] = torch.tensor(
            [65504.0, 1e-30, -3.0, 2.0, 1e30, -0.0, 7.0, 9.0], dtype=torch.bfloat16,
        )
    source = NF12Linear(
        bits=12, in_features=in_features, out_features=out_features,
        bias=True, dtype=torch.bfloat16,
    )
    source.pack_original(linear)
    return source


def _transfer(monkeypatch, source):
    from gptqmodel.utils import mlx as mlx_utils

    class Args:
        @classmethod
        def from_dict(cls, _config):
            return cls()

    class Tiny(nn.Module):
        def __init__(self, _args):
            super().__init__()
            self.proj = nn.Linear(source.in_features, source.out_features, bias=True)

        def __call__(self, x):
            return self.proj(x)

    monkeypatch.setattr(mlx_utils, "_get_classes", lambda config: (Tiny, Args))
    holder = torch.nn.Module()
    holder.proj = source
    return mlx_utils._packed_mlx_weights(holder, {}, "lm_head")


@requires_metal
def test_nf12_holder_validates_shape_and_bits():
    from gptqmodel.models._const import DEVICE
    from gptqmodel.nn_modules.qlinear.mlx import NF12MlxQuantLinear

    params = {
        "group_size": -1, "desc_act": False, "sym": True,
        "in_features": 128, "out_features": 32, "pack_dtype": torch.int32,
        "dtype": torch.float16, "device": DEVICE.MPS,
    }
    assert NF12MlxQuantLinear.validate(bits=12, **params)[0]
    assert not NF12MlxQuantLinear.validate(bits=8, **params)[0]
    assert not NF12MlxQuantLinear.validate(bits=12, **{**params, "in_features": 96})[0]


@pytest.mark.parametrize("escapes", [False, True])
@requires_metal
def test_nf12_mlx_decode_is_bit_exact(monkeypatch, escapes):
    from gptqmodel.nn_modules.qlinear.mlx_nf12 import MlxNF12Linear
    from gptqmodel.utils import nf12_codec

    source = _source(escapes=escapes)
    model, config = _transfer(monkeypatch, source)
    assert isinstance(model.proj, MlxNF12Linear)
    assert config["_gptqmodel_custom_mlx_runtime"]

    decoded = model.proj._decode()
    mx.eval(decoded)
    reference = nf12_codec.decode(
        source.nf12_packed, source.nf12_escapes, int(source.nf12_exp_base),
        source.out_features, source.in_features,
    )
    assert torch.equal(
        torch.from_numpy(np.array(decoded.astype(mx.float32))).to(torch.bfloat16), reference,
    )


@pytest.mark.parametrize("escapes", [False, True])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 12, 16, 33])
@requires_metal
def test_nf12_mlx_forward_matches_torch(monkeypatch, escapes, rows):
    source = _source(escapes=escapes)
    model, _ = _transfer(monkeypatch, source)
    rng = np.random.default_rng(29)
    x = rng.normal(0, 0.2, (rows, source.in_features)).astype(np.float32)
    weight = source.dequantize_weight(dtype=torch.bfloat16)
    expected = (
        torch.from_numpy(x).to(torch.bfloat16).float() @ weight.float().T
        + source.bias.float()
    ).to(torch.bfloat16).float().numpy()
    actual = model(mx.array(x))
    mx.eval(actual)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), expected, rtol=3e-2, atol=2e-3,
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@requires_metal
def test_nf12_mlx_preserves_activation_dtype(monkeypatch, dtype):
    source = _source()
    model, _ = _transfer(monkeypatch, source)
    rng = np.random.default_rng(31)
    x = mx.array(rng.normal(0, 0.2, (2, 3, source.in_features)).astype(np.float32)).astype(dtype)
    output = model(x)
    mx.eval(output)
    assert output.dtype == dtype
    assert output.shape == (2, 3, source.out_features)


def test_nf12_mlx_rejects_a_foreign_packed_layout(monkeypatch):
    from gptqmodel.nn_modules.qlinear.mlx import NF12MlxQuantLinear

    source = _source()
    assert NF12MlxQuantLinear.source_compatible(source)
    source.nf12_exp_base = torch.zeros((2,), dtype=torch.uint8)
    assert not NF12MlxQuantLinear.source_compatible(source)
    with pytest.raises(ValueError):
        NF12MlxQuantLinear.native_payload(source)


@requires_metal
def test_nf12_mlx_converts_a_llama_layer_end_to_end(tmp_path):
    """A 12-bit checkpoint must pass the conversion's width gate and load."""
    import json

    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel.utils.mlx import convert_gptq_to_mlx_weights

    torch.manual_seed(3)
    config = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config)
    source = _source(in_features=64, out_features=64)
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj = source
    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))

    mlx_model, mlx_config = convert_gptq_to_mlx_weights(
        str(tmp_path), model,
        {"bits": 12, "group_size": -1, "desc_act": False, "sym": True,
         "format": "nf12", "method": "nf12", "dynamic": None},
        "lm_head",
    )
    assert mlx_config["_gptqmodel_custom_mlx_runtime"]
    assert type(mlx_model.model.layers[0].self_attn.q_proj).__name__ == "MlxNF12Linear"
    logits = mlx_model(mx.array(np.array([[1, 2, 3, 4]], dtype=np.int32)))
    mx.eval(logits)
    assert logits.shape == (1, 4, 128)


@pytest.mark.parametrize("in_features,out_features", [(64, 16), (192, 48), (512, 128)])
@requires_metal
def test_nf12_mlx_fused_tile_matches_the_codec_oracle(monkeypatch, in_features, out_features):
    """The 2..16 token tile must equal the exact decode followed by a matmul."""
    source = _source(in_features=in_features, out_features=out_features, escapes=True)
    model, _ = _transfer(monkeypatch, source)
    weight = source.dequantize_weight(dtype=torch.bfloat16).double().numpy()
    bias = source.bias.double().numpy()
    for rows in (2, 3, 4, 8, 15, 16):
        x = np.random.default_rng(rows).normal(0, 0.2, (rows, in_features)).astype(np.float32)
        expected = x.astype(np.float64) @ weight.T + bias
        actual = model(mx.array(x))
        mx.eval(actual)
        assert actual.shape == (rows, out_features)
        scale = np.abs(expected).max() + 1e-30
        np.testing.assert_allclose(
            np.array(actual.astype(mx.float32)), expected, rtol=0, atol=5e-3 * scale,
        )


@requires_metal
def test_nf12_mlx_row_count_dispatch(monkeypatch):
    """1..15 rows use the decode-plus-dot kernel, 16 the tile, 17+ one decode."""
    from gptqmodel.nn_modules.qlinear import mlx_nf12
    from gptqmodel.nn_modules.qlinear.mlx_nf12 import MlxNF12Linear

    source = _source(in_features=256, out_features=48)
    model, _ = _transfer(monkeypatch, source)
    proj = model.proj
    assert isinstance(proj, MlxNF12Linear)

    rng = np.random.default_rng(37)
    for rows in (1, 2, 4, 8):
        x = rng.normal(0, 0.2, (rows, proj.in_features)).astype(np.float32)
        direct = proj._rows(mx.array(x), rows)
        actual = model(mx.array(x))
        mx.eval(direct, actual)
        np.testing.assert_array_equal(
            np.array(actual.astype(mx.float32)), np.array(direct.astype(mx.float32)),
        )

    # The decode-plus-dot kernel owns everything below the full 16-row tile,
    # whatever the output width: the tile's per-super-atom barriers only pay
    # for themselves at exactly 16 rows.
    for rows in (9, 12, 16):
        x = rng.normal(0, 0.2, (rows, proj.in_features)).astype(np.float32)
        direct = proj._rows(mx.array(x), rows)
        actual = model(mx.array(x))
        mx.eval(direct, actual)
        np.testing.assert_array_equal(
            np.array(actual.astype(mx.float32)), np.array(direct.astype(mx.float32)),
        )

    # A 16-row tile on a wide output takes the fused matrix-unit path.
    monkeypatch.setattr(mlx_nf12, "_TILE_MIN_OUT_FEATURES", 0)
    rows = 16
    x = rng.normal(0, 0.2, (rows, proj.in_features)).astype(np.float32)
    # The tile epilogue applies the bias (and the split-K reduce does too).
    tiled = proj._prefill(mx.array(x), rows)
    actual = model(mx.array(x))
    mx.eval(tiled, actual)
    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)), np.array(tiled.astype(mx.float32)),
    )

    rows = 17
    x = mx.array(rng.normal(0, 0.2, (rows, proj.in_features)).astype(np.float32)).astype(mx.bfloat16)
    split = (mx.matmul(x, proj._decode().T) + proj.bias).astype(x.dtype)
    actual = model(x)
    mx.eval(split, actual)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), np.array(split.astype(mx.float32)),
        rtol=0, atol=0,
    )


@requires_metal
def test_nf12_mlx_wide_output_shares_activation_loads(monkeypatch):
    """A wide output decodes several column blocks from one activation load."""
    from gptqmodel.nn_modules.qlinear.mlx_nf12 import MlxNF12Linear, _row_blocks

    # Narrow outputs keep one block so the grid still covers the GPU; wide
    # outputs take four blocks for the short token counts and two for 5..12.
    assert _row_blocks(1, 48) == 1
    assert _row_blocks(1, 512) == 1
    assert _row_blocks(1, 1024) == 2
    assert _row_blocks(1, 2048) == 4
    assert _row_blocks(4, 2048) == 4
    assert _row_blocks(8, 2048) == 2
    assert _row_blocks(12, 2048) == 2
    assert _row_blocks(16, 2048) == 1
    assert _row_blocks(1, 128256) == 4

    source = _source(in_features=256, out_features=1024)
    model, _ = _transfer(monkeypatch, source)
    assert isinstance(model.proj, MlxNF12Linear)
    weight = source.dequantize_weight(dtype=torch.bfloat16)
    rng = np.random.default_rng(41)
    for rows in (1, 4, 8, 12):
        x = rng.normal(0, 0.2, (rows, source.in_features)).astype(np.float32)
        expected = (
            torch.from_numpy(x).to(torch.bfloat16).float() @ weight.float().T
            + source.bias.float()
        ).to(torch.bfloat16).float().numpy()
        actual = model(mx.array(x))
        mx.eval(actual)
        np.testing.assert_allclose(
            np.array(actual.astype(mx.float32)), expected, rtol=3e-2, atol=2e-3,
        )


@requires_metal
def test_nf12_mlx_embedding_matches_the_table():
    """Packed rows gather bit-exactly; out-of-range IDs give +0.0 rows."""
    from gptqmodel.nn_modules.qlinear.mlx_nf12 import MlxNF12Embedding

    torch.manual_seed(5)
    source = _source(in_features=256, out_features=48)
    table = MlxNF12Embedding(
        num_embeddings=source.out_features,
        embedding_dim=source.in_features,
        num_escapes=source.nf12_escapes.shape[0],
    )
    table.packed = mx.array(source.nf12_packed.numpy())
    table.escapes = mx.array(source.nf12_escapes.numpy())
    table.exp_base = mx.array(source.nf12_exp_base.numpy())
    table.freeze()

    weight = source.dequantize_weight(dtype=torch.bfloat16).float().numpy()
    zero = np.zeros(source.in_features, dtype=np.float32)
    for ids in ([0, 5, 17, 47], [3], list(range(48)), [-1, 48, 2], [47, 0, 47]):
        actual = table(mx.array(np.array(ids, dtype=np.int32)))
        mx.eval(actual)
        expected = np.stack([
            weight[index] if 0 <= index < source.out_features else zero for index in ids
        ])
        np.testing.assert_array_equal(np.array(actual.astype(mx.float32)), expected)

    # Arbitrary leading dimensions are preserved.
    grid = table(mx.array(np.arange(6, dtype=np.int32).reshape(2, 3)))
    mx.eval(grid)
    assert grid.shape == (2, 3, source.in_features)
