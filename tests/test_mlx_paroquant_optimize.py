# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Torch-oracle checks for module-scope ParoQuant optimization on MLX."""

import sys

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


if sys.platform != "darwin":
    pytest.skip("MLX optimization requires macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.looper import paroquant_processor as processor_module  # noqa: E402
from gptqmodel.looper.named_module import NamedModule  # noqa: E402
from gptqmodel.looper.paroquant_processor import ParoQuantProcessor  # noqa: E402
from gptqmodel.quantization.config import ParoConfig  # noqa: E402
from gptqmodel.quantization.mlx_paroquant_optimize import (  # noqa: E402
    _differentiable_rotation_metadata,
    _forward,
    _pseudo_quantize,
    _smooth_l1,
    optimize_paroquant_linear_mlx_to_torch,
)
from gptqmodel.quantization.paroquant.optimization import (  # noqa: E402
    ParoQuantOptimizationResult,
    _ParoQuantOptimLinear,
    build_random_rotation_buffers,
    optimize_paroquant_linear,
    pseudo_quantize_dequant,
)


def _inputs(seed=3235, *, rows=20, features=128, outputs=128, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    weight = (torch.randn((outputs, features), generator=generator) * 0.1).to(dtype)
    inputs = (torch.randn((rows, features), generator=generator) * 0.1).to(dtype)
    bias = (torch.randn((outputs,), generator=generator) * 0.01).to(dtype)
    return weight, bias, inputs


def _pairs(features=128, *, group_size=128, krot=2):
    return build_random_rotation_buffers(
        in_features=features,
        group_size=group_size,
        krot=krot,
        pair_ratio=0.5,
        seed=7331,
        device=torch.device("cpu"),
    )


def _kwargs(*, epochs=0, group_size=128, optimizer="adamw"):
    return {
        "bits": 4,
        "group_size": group_size,
        "train_rows": 12,
        "val_rows": 4,
        "batch_size": 4,
        "rotation_epochs": epochs,
        "finetune_epochs": epochs,
        "rotation_lr": 0.005,
        "weight_lr": 1e-5,
        "quantizer_lr": 1e-6,
        "optimizer_name": optimizer,
        "optimizer_weight_decay": 0.01,
        "optimizer_betas": (0.9, 0.95),
        "optimizer_eps": 1e-10,
        "optimizer_amsgrad": False,
        "sgd_momentum": 0.1 if optimizer == "sgd" else 0.0,
        "sgd_dampening": 0.0,
        "sgd_nesterov": optimizer == "sgd",
        "best_state_dtype": "fp32",
        "scale_clamp_min": 0.01,
        "scale_clamp_max": 100.0,
    }


def _mlx_result(weight, bias, inputs, *, symmetric, epochs=0, optimizer="adamw"):
    pairs, mask = _pairs(weight.shape[1])
    return optimize_paroquant_linear_mlx_to_torch(
        weight=weight.float(),
        bias=bias.float(),
        inputs=inputs.float(),
        pairs=pairs,
        theta_mask=mask,
        symmetric=symmetric,
        **_kwargs(epochs=epochs, optimizer=optimizer),
    )


def _torch_result(weight, bias, inputs, *, symmetric, epochs=0, optimizer="adamw"):
    kwargs = _kwargs(epochs=epochs, optimizer=optimizer)
    return optimize_paroquant_linear(
        weight=weight.float(),
        bias=bias.float(),
        inputs=inputs.float(),
        sym=True,
        krot=2,
        pair_ratio=0.5,
        seed=7331,
        fused_rotation=False,
        gradient_checkpointing=False,
        stage_cudagraph=False,
        stage_impl="fast",
        pair_impl="fast",
        quantizer_impl="fast" if symmetric else "reference",
        **kwargs,
    )


@pytest.mark.parametrize("symmetric", (False, True), ids=("affine", "symmetric"))
def test_paroquant_mlx_zero_epoch_matches_torch_oracle(symmetric):
    weight, bias, inputs = _inputs()
    pairs, _ = _pairs()
    actual = _mlx_result(weight, bias, inputs, symmetric=symmetric)
    expected = optimize_paroquant_linear(
        weight=weight,
        bias=bias,
        inputs=inputs,
        bits=4,
        group_size=128,
        sym=True,
        krot=2,
        pair_ratio=0.5,
        train_rows=12,
        val_rows=4,
        batch_size=4,
        rotation_epochs=0,
        finetune_epochs=0,
        rotation_lr=0.005,
        weight_lr=1e-5,
        quantizer_lr=1e-6,
        seed=7331,
        optimizer_name="adamw",
        optimizer_weight_decay=0.01,
        optimizer_betas=(0.9, 0.95),
        optimizer_eps=1e-10,
        fused_rotation=False,
        stage_cudagraph=False,
        stage_impl="fast",
        pair_impl="fast",
        quantizer_impl="fast" if symmetric else "reference",
        best_state_dtype="fp32",
    )

    torch.testing.assert_close(actual.pack_weight, expected.pack_weight, rtol=0, atol=0)
    torch.testing.assert_close(actual.pseudo_weight, expected.pseudo_weight, rtol=0, atol=0)
    torch.testing.assert_close(actual.q_scales, expected.q_scales, rtol=0, atol=0)
    torch.testing.assert_close(actual.q_zeros, expected.q_zeros, rtol=0, atol=0)
    torch.testing.assert_close(actual.theta, expected.theta, rtol=0, atol=0)
    torch.testing.assert_close(actual.channel_scales, expected.channel_scales, rtol=0, atol=0)
    torch.testing.assert_close(actual.pairs, pairs, rtol=0, atol=0)
    assert abs(actual.train_loss - expected.train_loss) <= 1e-6
    assert abs(actual.val_loss - expected.val_loss) <= 1e-6


def test_paroquant_mlx_forward_and_gradients_match_torch_oracle():
    weight, bias, inputs = _inputs(rows=4)
    pairs, mask = _pairs()
    oracle = _ParoQuantOptimLinear(
        weight,
        bias,
        bits=4,
        group_size=128,
        quantizer_sym=False,
        pairs=pairs,
        theta_mask=mask,
        fused_rotation=False,
    )
    target = torch.nn.functional.linear(inputs, weight, bias)
    oracle_output = oracle(inputs)
    oracle_loss = torch.nn.functional.smooth_l1_loss(oracle_output, target)
    oracle_loss.backward()

    metadata = tuple(
        mx.array(value)
        for value in _differentiable_rotation_metadata(
            pairs.numpy(),
            columns=128,
            group_size=128,
        )
    )
    parameters = {
        "weight": mx.array(weight.numpy()),
        "theta": mx.zeros((2, 64), dtype=mx.float32),
        "theta_mask": mx.array(mask.numpy()),
        "channel_scales": mx.ones((128,), dtype=mx.float32),
    }

    def loss(values):
        output = _forward(
            values,
            mx.array(inputs.numpy()),
            mx.array(bias.numpy()),
            metadata,
            bits=4,
            group_size=128,
            symmetric=False,
            scale_clamp_min=0.01,
            scale_clamp_max=100.0,
            initialized_quantizer=False,
        )
        return _smooth_l1(output, mx.array(target.numpy())), output

    (actual_loss, actual_output), gradients = mx.value_and_grad(loss)(parameters)
    mx.eval(actual_loss, actual_output, gradients)
    np.testing.assert_allclose(np.asarray(actual_output), oracle_output.detach().numpy(), rtol=1e-6, atol=1e-6)
    assert abs(float(actual_loss.item()) - float(oracle_loss.item())) <= 1e-6
    for name, expected in (
        ("weight", oracle.weight.grad),
        ("theta", oracle.theta.grad),
        ("channel_scales", oracle.channel_scales_opt.grad),
    ):
        np.testing.assert_allclose(np.asarray(gradients[name]), expected.numpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("symmetric", (False, True), ids=("affine", "symmetric"))
def test_paroquant_mlx_rounding_boundaries_match_torch(symmetric):
    scale_value = 0.25
    half_steps = torch.tensor([-8.5, -0.5, 0.5, 6.5, 7.5, 15.5], dtype=torch.float32) * scale_value
    values = torch.cat(
        (
            torch.nextafter(half_steps, torch.full_like(half_steps, -torch.inf)),
            half_steps,
            torch.nextafter(half_steps, torch.full_like(half_steps, torch.inf)),
            torch.tensor([-0.0, 0.0, -100.0, 100.0]),
        )
    )
    weight = values.repeat(3)[:64].reshape(2, 32)
    group_size = weight.shape[1]
    scales = torch.full((2, 1), scale_value, dtype=torch.float32)
    zero = None if symmetric else torch.full_like(scales, -8.0)
    expected = pseudo_quantize_dequant(
        weight,
        bits=4,
        group_size=group_size,
        sym=symmetric,
        scale=scales,
        zero_point_float=zero,
        use_ste=False,
    )
    actual = _pseudo_quantize(
        mx.array(weight.numpy()),
        bits=4,
        group_size=group_size,
        symmetric=symmetric,
        scale=mx.array(scales.numpy()),
        zero_point=None if zero is None else mx.array(zero.numpy()),
        ste=False,
    )
    np.testing.assert_array_equal(np.asarray(actual), expected.numpy())


@pytest.mark.parametrize("optimizer", ("adamw", "adam", "sgd"))
def test_paroquant_mlx_trained_quality_tracks_torch(optimizer):
    weight, bias, inputs = _inputs(seed=4235)
    actual = _mlx_result(weight, bias, inputs, symmetric=False, epochs=1, optimizer=optimizer)
    expected = _torch_result(weight, bias, inputs, symmetric=False, epochs=1, optimizer=optimizer)
    assert torch.isfinite(actual.pack_weight).all()
    assert torch.isfinite(actual.pseudo_weight).all()
    assert actual.train_loss >= 0 and actual.val_loss >= 0
    source_output = torch.nn.functional.linear(inputs, weight, bias)
    actual_output = torch.nn.functional.linear(inputs, actual.pseudo_weight, bias)
    expected_output = torch.nn.functional.linear(inputs, expected.pseudo_weight, bias)
    actual_error = torch.nn.functional.smooth_l1_loss(actual_output, source_output)
    expected_error = torch.nn.functional.smooth_l1_loss(expected_output, source_output)
    assert float(actual_error) <= 1e-3
    assert abs(float(actual_error) - float(expected_error)) <= 1e-6
    assert float((actual_output - expected_output).abs().max()) <= 2e-3


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def test_paroquant_mlx_bridge_accepts_checkpoint_dtypes(dtype):
    weight, bias, inputs = _inputs(dtype=dtype)
    result = _mlx_result(weight, bias, inputs, symmetric=False)
    assert result.pack_weight.dtype == torch.float32
    assert result.pseudo_weight.dtype == torch.float32
    assert torch.isfinite(result.pseudo_weight).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_paroquant_mlx_bridge_preserves_mps_results():
    weight, bias, inputs = _inputs()
    pairs, mask = _pairs()
    expected = optimize_paroquant_linear_mlx_to_torch(
        weight=weight,
        bias=bias,
        inputs=inputs,
        pairs=pairs,
        theta_mask=mask,
        symmetric=False,
        **_kwargs(),
    )
    actual = optimize_paroquant_linear_mlx_to_torch(
        weight=weight.to("mps"),
        bias=bias.to("mps"),
        inputs=inputs.to("mps"),
        pairs=pairs,
        theta_mask=mask,
        symmetric=False,
        **_kwargs(),
    )
    for name in (
        "pseudo_weight",
        "pack_weight",
        "q_scales",
        "q_zeros",
        "theta",
        "channel_scales",
    ):
        torch.testing.assert_close(getattr(actual, name).cpu(), getattr(expected, name), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_paroquant_mlx_qwen38_export_matches_torch_codes(
    name,
    out_features,
    in_features,
    dtype,
):
    rows = (torch.arange(out_features, dtype=torch.float32).remainder(31) - 15) / 31
    columns = (torch.arange(in_features, dtype=torch.float32).remainder(127) - 63) / 127
    source = (rows[:, None] * columns[None, :] * 0.25 + columns[None, :] * 0.01).to(dtype).float()
    inputs = torch.zeros((1, in_features), dtype=torch.float32)
    bias = torch.zeros((out_features,), dtype=torch.float32)
    pairs, mask = _pairs(in_features, group_size=128, krot=1)
    actual = optimize_paroquant_linear_mlx_to_torch(
        weight=source,
        bias=bias,
        inputs=inputs,
        pairs=pairs,
        theta_mask=mask,
        symmetric=False,
        **_kwargs(epochs=0),
    )
    grouped = source.reshape(-1, 128)
    minimum = grouped.amin(dim=1, keepdim=True)
    maximum = grouped.amax(dim=1, keepdim=True)
    scales = (maximum - minimum).clamp(min=1e-5) / 15
    zero = minimum / scales
    expected_weight = pseudo_quantize_dequant(
        source,
        bits=4,
        group_size=128,
        sym=False,
        scale=scales,
        zero_point_float=zero,
        use_ste=False,
    )
    expected_scales = scales.reshape(out_features, in_features // 128)
    expected_zeros = (-zero.round()).clamp(0, 15).reshape_as(expected_scales)
    actual_codes = torch.round(
        (
            actual.pack_weight.reshape(out_features, -1, 128)
            + actual.q_zeros[:, :, None] * actual.q_scales[:, :, None]
        )
        / actual.q_scales[:, :, None]
    ).to(torch.int8)
    expected_codes = torch.round(
        (
            expected_weight.reshape(out_features, -1, 128)
            + expected_zeros[:, :, None] * expected_scales[:, :, None]
        )
        / expected_scales[:, :, None]
    ).to(torch.int8)
    torch.testing.assert_close(actual_codes, expected_codes, rtol=0, atol=0, msg=name)
    torch.testing.assert_close(actual.pack_weight, expected_weight, rtol=0, atol=0, msg=name)
    torch.testing.assert_close(actual.q_scales, expected_scales, rtol=0, atol=0, msg=name)
    torch.testing.assert_close(actual.q_zeros, expected_zeros, rtol=0, atol=0, msg=name)
    torch.testing.assert_close(actual.pseudo_weight, expected_weight, rtol=1e-6, atol=1e-6, msg=name)

    del source, inputs, bias, pairs, mask, actual, grouped, minimum, maximum
    del scales, zero, expected_weight, expected_scales, expected_zeros
    del actual_codes, expected_codes
    mx.clear_cache()


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_paroquant_mlx_qwen38_trained_quality(
    name,
    out_features,
    in_features,
    dtype,
):
    generator = torch.Generator().manual_seed(4235 + out_features + in_features)
    source = (torch.randn((out_features, in_features), generator=generator) * 0.02).to(dtype).float()
    inputs = (torch.randn((4, in_features), generator=generator) * 0.02).to(dtype).float()
    pairs, mask = _pairs(in_features, group_size=128, krot=1)
    result = optimize_paroquant_linear_mlx_to_torch(
        weight=source,
        bias=None,
        inputs=inputs,
        pairs=pairs,
        theta_mask=mask,
        symmetric=False,
        **_kwargs(epochs=1),
    )
    expected = torch.nn.functional.linear(inputs, source)
    actual = torch.nn.functional.linear(inputs, result.pseudo_weight)
    error = torch.nn.functional.smooth_l1_loss(actual, expected)
    assert float(error) <= 1e-3, name
    assert torch.isfinite(result.pseudo_weight).all()

    del source, inputs, pairs, mask, result, expected, actual, error
    mx.clear_cache()


def test_paroquant_processor_routes_module_scope_through_mlx(monkeypatch):
    linear = torch.nn.Linear(128, 128, bias=False)
    named = NamedModule(linear, "proj", "layers.0.proj", 0)
    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = ParoConfig(
        bits=4,
        group_size=128,
        krot=1,
        opt_rotation_epochs=0,
        opt_finetune_epochs=0,
        opt_train_samples=4,
        opt_validation_samples=2,
        opt_batch_size=2,
    )
    processor.calculate_w_wq_diff = False
    processor.lock = __import__("threading").Lock()
    calls = []

    def fake_mlx(**kwargs):
        calls.append(kwargs)
        pairs = kwargs["pairs"]
        return ParoQuantOptimizationResult(
            pseudo_weight=kwargs["weight"].float(),
            pack_weight=kwargs["weight"].float(),
            q_scales=torch.ones((128, 1)),
            q_zeros=torch.full((128, 1), 8.0),
            pairs=pairs,
            theta=torch.zeros((1, 64)),
            channel_scales=torch.ones(128),
            train_loss=0.0,
            val_loss=0.0,
            used_identity=False,
        )

    monkeypatch.setattr(processor_module, "paroquant_mlx_optimization_available", lambda: True)
    monkeypatch.setattr(processor_module, "optimize_paroquant_linear_mlx_to_torch", fake_mlx)
    processor._quantize_one_module(named, torch.zeros((6, 128)))
    assert len(calls) == 1
    assert calls[0]["symmetric"] is False
    assert {"pack_weight", "q_scales", "q_zeros", "pairs", "theta", "channel_scales"} <= named.state.keys()


def test_paroquant_processor_keeps_grouped_scope_on_torch(monkeypatch):
    linear = torch.nn.Linear(128, 128, bias=False)
    named = NamedModule(linear, "proj", "layers.0.proj", 0)
    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = ParoConfig(
        bits=4,
        group_size=128,
        krot=1,
        opt_scope="compute_block",
        opt_rotation_epochs=0,
        opt_finetune_epochs=0,
        opt_train_samples=4,
        opt_validation_samples=2,
        opt_batch_size=2,
    )
    processor.calculate_w_wq_diff = False
    processor.lock = __import__("threading").Lock()
    called = []
    original = processor_module.optimize_paroquant_linear

    def spy(**kwargs):
        called.append(True)
        return original(**kwargs)

    monkeypatch.setattr(processor_module, "paroquant_mlx_optimization_available", lambda: True)
    monkeypatch.setattr(processor_module, "optimize_paroquant_linear", spy)
    processor._quantize_one_module(named, torch.zeros((6, 128)))
    assert called == [True]
