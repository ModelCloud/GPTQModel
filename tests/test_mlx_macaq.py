# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# MACaQ reference: Huawei CSL contributors, Apache-2.0, https://github.com/huawei-csl/MACaQ
"""Packed MACaQ MXFP4 MLX transfer checked against Torch and MLX oracles."""

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


def _quantized_layer(in_features=256, out_features=64, activation_precision="w4a4", bias=True,
                     **quant_kwargs):
    from gptqmodel.nn_modules.qlinear.macaq import MacaqLinear
    from gptqmodel.quantization import macaq as macaq_core

    torch.manual_seed(11)
    linear = torch.nn.Linear(in_features, out_features, bias=bias)
    calibration = torch.randn(16, in_features) * 0.2
    hessian = (calibration.T @ calibration).float()
    result = macaq_core.macaq_quantize_weight(
        linear.weight.detach().float(), hessian, calibration.shape[0],
        do_flip=True, n_flip_iters=8, **quant_kwargs,
    )
    source = MacaqLinear(
        bits=4, group_size=32, in_features=in_features, out_features=out_features,
        bias=bias, dtype=torch.bfloat16, activation_precision=activation_precision,
    )
    source.pack(linear, macaq_result=result)
    return source, result


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
def test_macaq_holder_validates_without_affine_group_limits():
    from gptqmodel.models._const import DEVICE
    from gptqmodel.nn_modules.qlinear.mlx import MacaqMlxQuantLinear

    params = {
        "group_size": 32, "desc_act": False, "sym": True,
        "in_features": 96, "out_features": 64, "pack_dtype": torch.int32,
        "dtype": torch.float16, "device": DEVICE.MPS,
    }
    assert MacaqMlxQuantLinear.validate(bits=4, **params)[0]
    assert not MacaqMlxQuantLinear.validate(bits=8, **params)[0]
    assert not MacaqMlxQuantLinear.validate(
        bits=4, **{**params, "in_features": 48}
    )[0]


def test_macaq_checkpoint_bytes_are_native_mxfp4():
    """MACaQ nibbles must be the OCP MXFP4 words MLX's mxfp4 mode reads."""
    from gptqmodel.nn_modules.qlinear.mlx import MacaqMlxQuantLinear

    source, result = _quantized_layer()
    assert MacaqMlxQuantLinear.source_compatible(source)
    words = np.ascontiguousarray(result.W_packed.numpy()).view(np.uint32)
    reference = mx.array(words.reshape(source.out_features, source.in_features // 8))
    scales = mx.array(result.W_scales.view(torch.uint8).numpy())
    decoded = mx.dequantize(
        reference, scales, group_size=32, bits=4, mode="mxfp4",
    )
    mx.eval(decoded)
    expected = result.dequantized_weight
    if expected is None:
        from gptqmodel.quantization import macaq as macaq_core

        expected = macaq_core.dequantize_mxfp4(
            result.W_packed, result.W_scales, source.in_features, dtype=torch.float32,
        )
    np.testing.assert_allclose(
        np.array(decoded.astype(mx.float32)), expected.float().numpy(), rtol=0, atol=1e-6,
    )


@requires_metal
def test_macaq_mlx_transfer_matches_torch_w4a16(monkeypatch):
    from gptqmodel.nn_modules.qlinear.mlx_macaq import MlxMACaQLinear

    source, _ = _quantized_layer(activation_precision="w4a16")
    model, config = _transfer(monkeypatch, source)
    assert isinstance(model.proj, MlxMACaQLinear)
    assert config["_gptqmodel_custom_mlx_runtime"]
    assert model.proj.activation_precision == "w4a16"

    rng = np.random.default_rng(7)
    x = rng.normal(0, 0.2, (5, source.in_features)).astype(np.float32)
    expected = source._forward_torch(
        torch.from_numpy(x).to(torch.bfloat16)
    ).float().numpy()
    actual = model(mx.array(x))
    mx.eval(actual)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), expected, rtol=2e-2, atol=5e-3,
    )


@requires_metal
def test_macaq_mlx_w4a4_matches_its_own_activation_quantization(monkeypatch):
    """The fused path must be the MXFP4 activation quant MACaQ specifies.

    MLX picks its own E8M0 exponent and tie handling for the activation
    rounding, so the parity oracle re-runs the same MLX quantization and
    multiplies it against the checkpoint's decoded weight.
    """
    from gptqmodel.nn_modules.qlinear.mlx_macaq import MlxMACaQLinear

    source, result = _quantized_layer(activation_precision="w4a4")
    model, _ = _transfer(monkeypatch, source)
    assert isinstance(model.proj, MlxMACaQLinear)

    rng = np.random.default_rng(13)
    x = rng.normal(0, 0.2, (3, source.in_features)).astype(np.float32)
    rotated = model.proj._rotate(mx.array(x).astype(mx.bfloat16))
    quantized, scales = mx.quantize(rotated, group_size=32, bits=4, mode="mxfp4")
    rounded = mx.dequantize(quantized, scales, group_size=32, bits=4, mode="mxfp4")
    weight = mx.dequantize(
        model.proj.weight, model.proj.scales, group_size=32, bits=4, mode="mxfp4",
    )
    expected = mx.matmul(rounded, weight.T) + model.proj.bias
    actual = model(mx.array(x))
    mx.eval(expected, actual)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32)),
        rtol=2e-2, atol=5e-3,
    )

    # The Torch reference quantizes activations with its own rounding, so it
    # tracks the same layer only within the 4-bit activation grid.
    reference = source._forward_torch(torch.from_numpy(x).to(torch.bfloat16)).float().numpy()
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), reference,
        atol=0.12 * max(float(np.abs(reference).max()), 1e-6),
    )


def test_macaq_mlx_pre_scale_only_layer_skips_the_rotation(monkeypatch):
    """`hadamard=False` keeps D1 but needs no rotate dispatch."""
    from gptqmodel.nn_modules.qlinear.mlx_macaq import MlxMACaQLinear

    source, _ = _quantized_layer(activation_precision="w4a16", hadamard=False)
    model, config = _transfer(monkeypatch, source)
    assert isinstance(model.proj, MlxMACaQLinear)
    assert config["_gptqmodel_custom_mlx_runtime"]
    assert model.proj.apply_rotation is False
    assert model.proj.apply_pre_scale is True

    rng = np.random.default_rng(17)
    x = rng.normal(0, 0.2, (5, source.in_features)).astype(np.float32)
    actual = model(mx.array(x))
    mx.eval(actual)
    reference = source._forward_torch(torch.from_numpy(x).to(torch.bfloat16)).float().numpy()
    np.testing.assert_allclose(np.array(actual.astype(mx.float32)), reference, rtol=2e-2, atol=5e-3)


def _plain_mxfp4_layer(in_features=256, out_features=64, bias=True):
    return _quantized_layer(
        in_features=in_features, out_features=out_features, bias=bias,
        activation_precision="w4a16", hadamard=False, pre_mode="none",
    )


def test_macaq_mlx_plain_mxfp4_layer_is_a_stock_quantized_linear(monkeypatch):
    """No transform + W4A16 is exactly MLX's own MXFP4 QuantizedLinear."""
    source, _ = _plain_mxfp4_layer()
    model, config = _transfer(monkeypatch, source)
    assert isinstance(model.proj, nn.QuantizedLinear)
    assert model.proj.mode == "mxfp4"
    assert "_gptqmodel_custom_mlx_runtime" not in config
    assert config["quantization"] == {"group_size": 32, "bits": 4, "mode": "mxfp4"}

    # Bytes pass through unchanged: no requantization by MLX's own RTN.
    words = source.W_packed.numpy().view(np.uint32).reshape(source.out_features, -1)
    np.testing.assert_array_equal(np.array(model.proj.weight), words)
    np.testing.assert_array_equal(np.array(model.proj.scales), source.W_scales.numpy())

    rng = np.random.default_rng(19)
    x = rng.normal(0, 0.2, (5, source.in_features)).astype(np.float32)
    actual = model(mx.array(x).astype(mx.bfloat16))
    mx.eval(actual)
    reference = source._forward_torch(torch.from_numpy(x).to(torch.bfloat16)).float().numpy()
    np.testing.assert_allclose(np.array(actual.astype(mx.float32)), reference, rtol=2e-2, atol=1e-2)


def test_macaq_plain_mxfp4_exports_to_stock_mlx_lm(tmp_path):
    """The stock export path must write a checkpoint stock `mlx_lm` loads."""
    import json

    from mlx_lm.utils import load_model, save_config, save_model
    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel.utils.mlx import convert_gptq_to_mlx_weights

    torch.manual_seed(5)
    config = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config)
    # Llama projections are bias-free; the stock layer must match that.
    source, _ = _plain_mxfp4_layer(in_features=64, out_features=64, bias=False)
    model.model.layers[0].self_attn.q_proj = source
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(json.dumps(model.config.to_dict()))
    gptq_config = {"bits": 4, "group_size": 32, "desc_act": False, "sym": True,
                   "format": "mxfp4_macaq", "method": "macaq", "dynamic": None}

    mlx_model, mlx_config = convert_gptq_to_mlx_weights(
        str(source_dir), model, dict(gptq_config), "lm_head", stock_compatible=True,
    )
    assert "_gptqmodel_custom_mlx_runtime" not in mlx_config
    q_proj = mlx_model.model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, nn.QuantizedLinear) and q_proj.mode == "mxfp4"

    export_dir = tmp_path / "mlx"
    save_model(str(export_dir), mlx_model)
    save_config(mlx_config, config_path=str(export_dir / "config.json"))
    reloaded, _ = load_model(export_dir)
    reloaded_q = reloaded.model.layers[0].self_attn.q_proj
    assert isinstance(reloaded_q, nn.QuantizedLinear) and reloaded_q.mode == "mxfp4"
    np.testing.assert_array_equal(np.array(reloaded_q.weight), np.array(q_proj.weight))
    np.testing.assert_array_equal(np.array(reloaded_q.scales), np.array(q_proj.scales))

    tokens = mx.array(np.array([[1, 2, 3, 4]], dtype=np.int32))
    expected, actual = mlx_model(tokens), reloaded(tokens)
    mx.eval(expected, actual)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


def test_macaq_rotated_layer_is_not_stock_exportable(tmp_path):
    import json

    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel.utils.mlx import convert_gptq_to_mlx_weights

    config = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config)
    model.model.layers[0].self_attn.q_proj, _ = _quantized_layer(
        in_features=64, out_features=64, activation_precision="w4a16",
    )
    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
    with pytest.raises(ValueError, match="requires transferable packed weights"):
        convert_gptq_to_mlx_weights(
            str(tmp_path), model,
            {"bits": 4, "group_size": 32, "desc_act": False, "sym": True,
             "format": "mxfp4_macaq", "method": "macaq", "dynamic": None},
            "lm_head", stock_compatible=True,
        )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@requires_metal
def test_macaq_mlx_preserves_activation_dtype(monkeypatch, dtype):
    source, _ = _quantized_layer()
    model, _ = _transfer(monkeypatch, source)
    rng = np.random.default_rng(5)
    x = mx.array(rng.normal(0, 0.2, (2, 4, source.in_features)).astype(np.float32)).astype(dtype)
    output = model(x)
    mx.eval(output)
    assert output.dtype == dtype
    assert output.shape == (2, 4, source.out_features)


@requires_metal
def test_macaq_mlx_wide_block_uses_the_split_activation_path(monkeypatch):
    """The >16-row block must skip MLX's fused qqmm (5x slower there)."""
    from gptqmodel.nn_modules.qlinear.mlx_macaq import MlxMACaQLinear

    source, _ = _quantized_layer()
    model, _ = _transfer(monkeypatch, source)
    rng = np.random.default_rng(17)
    x = mx.array(rng.normal(0, 0.2, (17, source.in_features)).astype(np.float32)).astype(mx.bfloat16)
    rotated = model.proj._rotate(x)
    quantized, scales = mx.quantize(rotated, group_size=32, bits=4, mode="mxfp4")
    rounded = mx.dequantize(quantized, scales, group_size=32, bits=4, mode="mxfp4")
    expected = mx.quantized_matmul(
        rounded, model.proj.weight, model.proj.scales,
        transpose=True, group_size=32, bits=4, mode="mxfp4",
    ) + model.proj.bias
    actual = model(x)
    mx.eval(expected, actual)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32)),
        rtol=2e-2, atol=5e-3,
    )
    assert isinstance(model.proj, MlxMACaQLinear)


@requires_metal
def test_macaq_mlx_converts_a_llama_layer_end_to_end(tmp_path):
    """The full conversion must land the holder on a real model's paths."""
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
    source, _ = _quantized_layer(in_features=64, out_features=64)
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj = source
    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))

    mlx_model, mlx_config = convert_gptq_to_mlx_weights(
        str(tmp_path), model,
        {"bits": 4, "group_size": 32, "desc_act": False, "sym": True,
         "format": "mxfp4_macaq", "method": "macaq", "dynamic": None},
        "lm_head",
    )
    assert mlx_config["_gptqmodel_custom_mlx_runtime"]
    assert type(mlx_model.model.layers[0].self_attn.q_proj).__name__ == "MlxMACaQLinear"
    logits = mlx_model(mx.array(np.array([[1, 2, 3, 4]], dtype=np.int32)))
    mx.eval(logits)
    assert logits.shape == (1, 4, 128)


@pytest.mark.parametrize("had_block", [32, 64])
@requires_metal
def test_macaq_mlx_pre_rotate_folds_the_diagonal_scale(monkeypatch, had_block):
    """The fused pre-scale plus block Hadamard must equal the reference product."""
    from gptqmodel.quantization import macaq as macaq_core

    source, result = _quantized_layer(in_features=256, out_features=64)
    source.had_block = had_block
    source.BH = macaq_core.make_hadamard(had_block, dtype=torch.bfloat16)
    model, _ = _transfer(monkeypatch, source)

    rng = np.random.default_rng(41)
    x = mx.array(rng.normal(0, 0.2, (4, source.in_features)).astype(np.float32)).astype(mx.bfloat16)
    actual = model.proj._rotate(x)
    scaled = x.astype(mx.float32) * mx.array(result.pre_scale.float().numpy())
    expected = (
        scaled.reshape(4, source.in_features // had_block, had_block)
        @ mx.array(source.BH.float().numpy())
    ).reshape(4, source.in_features)
    mx.eval(actual, expected)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), np.array(expected),
        rtol=2e-3, atol=2e-3,
    )
