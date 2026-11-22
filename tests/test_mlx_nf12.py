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
    assert model.proj.bias.dtype == mx.float32

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
    actual = model(mx.array(x).astype(mx.bfloat16))
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

    from gptqmodel.utils.mlx import _convert_gptq_to_mlx_weights

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

    mlx_model, mlx_config = _convert_gptq_to_mlx_weights(
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


@pytest.mark.parametrize(("width", "count", "id_dtype"), [
    (320, 9, mx.int64), (4096, 1025, mx.int32),
])
@requires_metal
def test_nf12_embedding_preserves_escapes_and_full_width_ids(width, count, id_dtype):
    from gptqmodel.nn_modules.qlinear.mlx_nf12 import MlxNF12Embedding
    from gptqmodel.utils import nf12_codec

    weight = torch.full((48, width), -0.5, dtype=torch.bfloat16)
    weight[:2, :4].view(torch.int16)[:] = torch.tensor([-32768, 1, 0x7F80, 0x7FC1], dtype=torch.int16)
    packed, escapes, exp_base = nf12_codec.encode(weight)
    table = MlxNF12Embedding(num_embeddings=48, embedding_dim=width, num_escapes=len(escapes))
    table.packed, table.escapes = mx.array(packed.numpy()), mx.array(escapes.numpy())
    table.exp_base = mx.array([exp_base], dtype=mx.uint8)
    indices = torch.arange(count) % 48
    indices[-3:] = torch.tensor([-1, 48, 2**32 + 1 if id_dtype == mx.int64 else 47])
    valid = (indices >= 0) & (indices < 48)
    expected = torch.zeros((count, width), dtype=torch.int16)
    expected[valid] = weight.view(torch.int16)[indices[valid]]

    actual = table(mx.array(indices.numpy(), dtype=id_dtype))
    assert actual.dtype == mx.bfloat16
    np.testing.assert_array_equal(np.array(actual.view(mx.int16)), expected.numpy())
    assert table(mx.array([], dtype=id_dtype)).shape == (0, width)


@pytest.mark.parametrize(("rows", "n", "k", "dtype"), [
    (32, 128, 2048, mx.bfloat16), (32, 528, 2048, mx.bfloat16),
    (31, 4112, 4160, mx.float16), (65, 4096, 4160, mx.bfloat16),
])
@requires_metal
def test_nf12_tensor_prefill_matches_torch(monkeypatch, rows, n, k, dtype):
    from gptqmodel.nn_modules.qlinear import mlx_nf12
    from gptqmodel.utils import nf12_codec

    if not mlx_nf12._has_accelerated_tensorops():
        pytest.skip("Accelerated Metal 4.1 TensorOps requires macOS 27 and Apple10")
    # Powers of two give an exact FP64 oracle even with split K. The last
    # group escapes and the FP32 bias exposes rounding before the epilogue.
    generator = torch.Generator().manual_seed(11)
    weight = torch.randint(-16, 17, (n, k), generator=generator).to(torch.bfloat16) / 256
    weight[-1, -1] = 16
    packed, escapes, exp_base = nf12_codec.encode(weight)
    layer = mlx_nf12.MlxNF12Linear(in_features=k, out_features=n, num_escapes=len(escapes))
    layer.packed, layer.escapes = mx.array(packed.numpy()), mx.array(escapes.numpy())
    layer.exp_base = mx.array([exp_base], dtype=mx.uint8)
    bias = (torch.arange(n) % 33 - 16).float() / 2**17
    layer.bias = mx.array(bias.numpy())
    x = torch.randint(-16, 17, (rows, k + 1), generator=generator).float() / 64
    torch_dtype = torch.float16 if dtype == mx.float16 else torch.bfloat16
    expected = (x[:, 1:].double() @ weight.double().T + bias.double()).to(torch_dtype)
    activations = mx.array(x.numpy()).astype(dtype)[:, 1:]

    # Observe the public forward's launch without substituting an implementation.
    called = []
    launch = layer._tensor_prefill

    def record(x, m, b):
        called.append(m)
        return launch(x, m, b)

    monkeypatch.setattr(layer, "_tensor_prefill", record)
    for _ in range(2):
        actual = layer(activations)
        assert actual.dtype == dtype
        np.testing.assert_array_equal(np.array(actual.astype(mx.float32)), expected.float().numpy())
    assert called == [rows, rows]
    assert set(layer.parameters()) == {"packed", "escapes", "exp_base", "bias"}


@requires_metal
def test_nf12_fp32_router_preserves_mantissa_bits():
    from gptqmodel.nn_modules.qlinear.mlx_nf12 import MlxNF12Linear
    from gptqmodel.utils import nf12_codec

    weight = torch.ones((128, 2048), dtype=torch.bfloat16)
    weight[-1, -1] = 2**20
    packed, escapes, exp_base = nf12_codec.encode(weight)
    assert len(escapes) > 0
    layer = MlxNF12Linear(in_features=2048, out_features=128, num_escapes=len(escapes))
    layer.packed, layer.escapes = mx.array(packed.numpy()), mx.array(escapes.numpy())
    layer.exp_base = mx.array([exp_base], dtype=mx.uint8)
    x = torch.zeros((32, 2048), dtype=torch.float32)
    x[:, :2] = torch.tensor([1 + 2**-16, -1])
    expected = (x.double() @ weight.double().T).float().numpy()
    actual = layer(mx.array(x.numpy()))
    assert actual.dtype == mx.float32
    np.testing.assert_array_equal(np.array(actual), expected)


@requires_metal
def test_nf12_tied_embedding_projects_from_packed_table():
    from gptqmodel.nn_modules.qlinear.mlx_nf12 import MlxNF12Embedding
    from gptqmodel.utils import nf12_codec

    # Row lookup and the tied projection consume one packed representation.
    weight = torch.full((528, 2048), 0.5, dtype=torch.bfloat16)
    weight[3, 1] = -16
    packed, escapes, exp_base = nf12_codec.encode(weight)
    table = MlxNF12Embedding(num_embeddings=528, embedding_dim=2048, num_escapes=len(escapes))
    # Empty escape buffers are represented by the harmless placeholder.
    table.packed = mx.array(packed.numpy())
    if len(escapes):
        table.escapes = mx.array(escapes.numpy())
    table.exp_base = mx.array([exp_base], dtype=mx.uint8)
    x = torch.zeros((2, 16, 2048), dtype=torch.bfloat16)
    x[..., :2] = 1
    expected = (x.double() @ weight.double().T).to(torch.bfloat16).float().numpy()
    activations = mx.array(x.float().numpy()).astype(mx.bfloat16)
    for project in (table.as_linear, mx.compile(table.as_linear)):
        actual = project(activations)
        assert actual.dtype == mx.bfloat16
        np.testing.assert_array_equal(np.array(actual.astype(mx.float32)), expected)
    np.testing.assert_array_equal(np.array(table(mx.array([3])).astype(mx.float32)), weight[3:4].float().numpy())
    assert set(table.parameters()) == {"packed", "escapes", "exp_base"}


@pytest.mark.parametrize(("dtype", "bias_dtype"), [
    (mx.float16, mx.float32), (mx.bfloat16, mx.float32),
    (mx.float32, mx.float32), (mx.bfloat16, mx.bfloat16),
])
@pytest.mark.parametrize("rows", [1, 7, 16, 33])
@requires_metal
def test_nf12_linear_bias_precedes_output_rounding(dtype, bias_dtype, rows):
    from gptqmodel.nn_modules.qlinear.mlx_nf12 import MlxNF12Linear
    from gptqmodel.utils import nf12_codec

    # Subtracting a wide bias from values around BF16 midpoints exposes
    # rounding of either the activation or the product before the bias add.
    weight = torch.zeros((48, 192), dtype=torch.bfloat16)
    weight[:, 0] = 1
    weight[:, 1] = torch.arange(1, 49, dtype=torch.float32) / 256
    packed, escapes, exp_base = nf12_codec.encode(weight)
    layer = MlxNF12Linear(in_features=192, out_features=48, num_escapes=len(escapes))
    layer.packed = mx.array(packed.numpy())
    if len(escapes):
        layer.escapes = mx.array(escapes.numpy())
    layer.exp_base = mx.array([int(exp_base)], dtype=mx.uint8)
    bias = -1.0 if bias_dtype == mx.bfloat16 else -1.0 - 2**-15
    layer.bias = mx.full((48,), bias, dtype=bias_dtype)

    activations = torch.zeros((rows, 192))
    activations[:, :2] = torch.tensor([1.0, 1.0 - 2**-10])
    torch_dtype = {mx.float16: torch.float16, mx.bfloat16: torch.bfloat16, mx.float32: torch.float32}[dtype]
    expected = (
        activations.to(torch_dtype).double() @ weight.double().T + bias
    ).to(torch_dtype).float().numpy()
    x = mx.array(activations.numpy()).astype(dtype)
    for _ in range(2):
        actual = layer(x)
        assert actual.dtype == dtype
        np.testing.assert_array_equal(np.array(actual.astype(mx.float32)), expected)
    assert set(layer.parameters()) == {"packed", "escapes", "exp_base", "bias"}



@pytest.mark.parametrize("rows", [1, 4, 8, 16, 33])
@requires_metal
def test_nf12_mlx_linear_tails_and_native_bias(monkeypatch, rows):
    from gptqmodel.nn_modules.qlinear import mlx_nf12

    source = _source(in_features=512, out_features=2064)
    source.bias = source.bias.to(torch.bfloat16)
    model, _ = _transfer(monkeypatch, source)
    assert model.proj.bias.dtype == mx.bfloat16
    # Force multiple decoded slices with a final 16-row tail in prefill.
    monkeypatch.setattr(mlx_nf12, "_DECODE_SLICE_BYTES", 512 * 2048 * 2)

    x = torch.randn(2, rows, 512, dtype=torch.bfloat16) * 0.125
    weight = source.dequantize_weight(dtype=torch.bfloat16)
    expected = (x[1].double() @ weight.double().T + source.bias.double()).to(torch.bfloat16)
    # Test activation views as well as a grid with an uneven output width.
    actual = model(mx.array(x.float().numpy()).astype(mx.bfloat16)[1])
    np.testing.assert_allclose(np.array(actual.astype(mx.float32)), expected.float().numpy(), rtol=2e-3, atol=2e-3)
