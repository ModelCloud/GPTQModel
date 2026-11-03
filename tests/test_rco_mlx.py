# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Native MLX regression coverage for the optional RCO search backend."""

import platform
import sys

import numpy as np
import pytest
import torch

from gptqmodel.quantization import rco, rco_mlx


pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="MLX RCO requires Apple silicon",
)


def mlx_modules():
    mlx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx.nn")
    llama = pytest.importorskip("mlx_lm.models.llama")
    return mlx, llama


def tiny_mlx_model():
    mx, llama = mlx_modules()
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=32,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = llama.Model(args)
    mx.eval(model.parameters())
    return model


def tiny_qwen3_mlx_model():
    mx, _ = mlx_modules()
    from mlx_lm.models import qwen3

    args = qwen3.ModelArgs(
        model_type="qwen3",
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-5,
        vocab_size=32,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        rope_theta=1_000_000,
    )
    model = qwen3.Model(args)
    mx.eval(model.parameters())
    return model


def to_torch(array):
    import mlx.core as mx

    mx.eval(array)
    return torch.from_dlpack(array)


def test_mlx_supports_qwen3_dense_decoder_layout():
    assert rco_mlx.rco_mlx_supports_model("llama")
    assert rco_mlx.rco_mlx_supports_model("qwen3")
    assert not rco_mlx.rco_mlx_supports_model("qwen3_moe")


def test_qwen3_mlx_teacher_kl_and_interpolation():
    mx, _ = mlx_modules()
    model = tiny_qwen3_mlx_model()
    name = "model.layers.0.mlp.down_proj"
    reference = to_torch(rco_mlx._child(model, name).weight).detach().cpu()
    ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
    objective = rco_mlx.MLXTeacherKL(
        model,
        [ids],
        "/tmp/rco-mlx-qwen3-test",
        chunk_tokens=2,
        resident_cache=True,
    )
    interpolated = rco_mlx.MLXInterpolatedLlamaLayers(
        model,
        {name: [reference * 0.75, reference]},
        torch.tensor([reference.numel()]),
        torch.tensor([2.0, 3.0]),
    )
    try:
        interpolated.retract(2.5)
        probabilities, history = rco_mlx.projected_gumbel_search_mlx(
            interpolated,
            [0],
            2.5,
            steps=1,
            gumbel_samples=1,
            objective=objective,
            log_every=0,
        )
        mx.eval(probabilities)
        assert probabilities.shape == (1, 2)
        assert history[-1]["drift"] < 1e-6
    finally:
        interpolated.cleanup()
        objective.cleanup()


def test_mlx_budget_argmax_matches_torch_bit_for_bit():
    mx, _ = mlx_modules()
    generator = torch.Generator().manual_seed(91)
    for target in (2.0, 2.5, 3.5, 4.0):
        for _ in range(8):
            scores = torch.randn(5, 3, generator=generator, dtype=torch.float64) * 200
            weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
            costs = torch.tensor([2.0, 3.0, 4.0])
            expected = rco.budget_constrained_argmax(scores, weights, costs, target)
            actual = rco_mlx.budget_constrained_argmax_mlx(
                mx.array(scores.numpy()), mx.array(weights.numpy()), mx.array(costs.numpy()), target
            )
            assert to_torch(actual).tolist() == expected.tolist()


def test_mlx_retraction_reaches_the_torch_budget():
    mx, _ = mlx_modules()
    fraction = mx.array([0.15, 0.35, 0.5], dtype=mx.float32)
    costs = mx.array([2.0, 3.0, 4.0])
    alpha = mx.array(
        [[-0.3, 0.2, 0.1], [0.4, -0.1, 0.3], [-0.2, 0.6, -0.4]],
        dtype=mx.float32,
    )
    # The scalar bisection reads control arrays once rather than
    # synchronizing Metal for every candidate shift.
    shift = rco_mlx._retraction(alpha, costs, 2.65, fraction)
    retracted = alpha + shift * costs
    actual = float(rco_mlx._expected_cost(retracted, costs, fraction).item())
    torch_alpha = torch.tensor(np.asarray(alpha), requires_grad=True)
    torch_cost = rco.retraction(
        torch_alpha, torch.tensor([2.0, 3.0, 4.0]),
        target=2.65, weights=torch.tensor([0.15, 0.35, 0.5])
    )
    assert actual == pytest.approx(torch_cost, abs=1e-6)
    np.testing.assert_allclose(
        np.asarray(retracted), torch_alpha.detach().numpy(), rtol=2e-6, atol=1e-6
    )


@pytest.mark.parametrize("resident_cache", [False, True])
def test_mlx_teacher_kl_value_gradient_and_cache_parity(tmp_path, resident_cache):
    import mlx.nn as nn

    mx, _ = mlx_modules()
    model = tiny_mlx_model()
    ids = np.array([[1, 2, 3, 4], [4, 5, 6, 7]], dtype=np.int64)
    teacher_logits = model.model(mx.array(ids))
    objective = rco_mlx.MLXTeacherKL(
        model, [ids], tmp_path, chunk_tokens=3, resident_cache=resident_cache
    )
    parameter = model.model.layers[0].mlp.down_proj.weight
    changed = parameter + mx.random.normal(parameter.shape) * 0.02
    model.model.layers[0].mlp.down_proj.weight = changed
    actual_value, gradients = nn.value_and_grad(model, lambda _: objective(0))(model)
    gradient = gradients['model']['layers'][0]['mlp']['down_proj']['weight']

    def oracle_loss(_):
        def oracle(hidden):
            logits = model.lm_head(hidden).astype(mx.float32)
            return nn.log_softmax(logits.reshape(-1, logits.shape[-1]), axis=-1)

        teacher_logp = oracle(teacher_logits)
        student_logp = oracle(model.model(mx.array(ids)))
        return mx.mean(
            mx.sum(mx.exp(teacher_logp) * (teacher_logp - student_logp), axis=-1)
        )

    expected_value, expected_gradients = nn.value_and_grad(model, oracle_loss)(model)
    expected_gradient = expected_gradients['model']['layers'][0]['mlp']['down_proj']['weight']
    assert float(actual_value.item()) == pytest.approx(float(expected_value.item()), abs=2e-6)
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(expected_gradient), rtol=2e-5, atol=2e-6
    )
    assert bool((tmp_path / "00000.safetensors").exists()) is not resident_cache
    objective.cleanup()
    assert not objective._resident and not objective.paths


def test_mlx_ste_gradient_projection_adam_retraction_and_transport_match_torch():
    """Exercise the actual search recurrence with identical fixed draws."""
    mx, _ = mlx_modules()
    torch.manual_seed(101)
    torch_alpha = torch.tensor(
        [[-0.7, 0.2, 0.5], [0.4, -0.3, -0.1], [0.1, 0.6, -0.8]],
        dtype=torch.float32,
        requires_grad=True,
    )
    noise = torch.tensor(
        [[0.2, -0.1, 0.05], [-0.4, 0.3, 0.1], [0.7, -0.2, -0.5]],
        dtype=torch.float32,
    )
    onehot = torch.tensor(
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        dtype=torch.float32,
    )
    costs = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
    weights = torch.tensor([3.0, 2.0, 5.0], dtype=torch.float32)
    fraction = weights / weights.sum()
    target = 2.8
    learning_rate = 0.13

    # Independent scalar objective. It deliberately uses a nonlinear loss so
    # the test observes an actual matmul/SOFTMAX graph; the STE identity comes
    # from assigning the hard candidate as an additive zero-gradient offset.
    torch_soft = torch.softmax(torch_alpha + noise, dim=-1)
    torch_probs = onehot - torch_soft.detach() + torch_soft
    torch_loss_value = (torch_probs * torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    )).sum() + torch_probs.square().sum() * 0.07
    torch_loss_value.backward()
    torch_raw_gradient = torch_alpha.grad.clone()
    normal = rco.budget_normal(torch_alpha, costs, fraction)
    normal_flat = normal.flatten()
    coefficient = (torch_raw_gradient.flatten() @ normal_flat) / (
        normal_flat @ normal_flat + 1e-12
    )
    torch_gradient = torch_raw_gradient - coefficient * normal

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    torch_momentum = torch.zeros_like(torch_alpha)
    torch_velocity = torch.zeros_like(torch_alpha)
    torch_momentum = beta1 * torch_momentum + (1 - beta1) * torch_gradient
    torch_velocity = beta2 * torch_velocity + (1 - beta2) * (torch_gradient * torch_gradient)
    torch_updated = torch_alpha.detach() - learning_rate * (
        torch_momentum / (1 - beta1)
    ) / (torch.sqrt(torch_velocity / (1 - beta2)) + epsilon)
    torch_retracted = torch_updated.clone().requires_grad_(True)
    rco.retraction(torch_retracted, costs, target, fraction)
    normal_after = rco.budget_normal(torch_retracted, costs, fraction)
    transport_coefficient = (
        torch_momentum.flatten() @ normal_after.flatten()
    ) / (normal_after.flatten() @ normal_after.flatten() + 1e-12)
    torch_momentum = torch_momentum - transport_coefficient * normal_after

    mx_alpha = mx.array(torch_alpha.detach().numpy())
    mx_noise = mx.array(noise.numpy())
    mx_onehot = mx.array(onehot.numpy())
    mx_costs = mx.array(costs.numpy())
    mx_fraction = mx.array(fraction.numpy())

    def loss(alpha):
        soft = rco_mlx._softmax(alpha + mx_noise)
        probs = mx_onehot - mx.stop_gradient(soft) + soft
        return mx.sum(probs * mx.array([
            [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]
        ])) + mx.sum(probs * probs) * 0.07

    loss_value, raw_gradient = mx.value_and_grad(loss)(mx_alpha)
    mx_gradient = rco_mlx._tangent_gradient(mx_alpha, raw_gradient, mx_costs, mx_fraction)
    mx_momentum = mx.zeros_like(mx_alpha)
    mx_velocity = mx.zeros_like(mx_alpha)
    mx_updated, mx_momentum, mx_velocity = rco_mlx._adam_update(
        mx_alpha, mx_gradient, mx_momentum, mx_velocity, 1, learning_rate
    )
    shift = rco_mlx._retraction(mx_updated, mx_costs, target, mx_fraction)
    mx_retracted = mx_updated + shift * mx_costs
    normal_after = rco_mlx._budget_normal(mx_retracted, mx_costs, mx_fraction)
    transport_coefficient = mx.sum(mx_momentum.reshape(-1) * normal_after.reshape(-1)) / (
        mx.sum(normal_after.reshape(-1) * normal_after.reshape(-1)) + 1e-12
    )
    mx_momentum = mx_momentum - transport_coefficient * normal_after
    mx.eval(loss_value, mx_gradient, mx_updated, mx_retracted, mx_momentum, mx_velocity)

    np.testing.assert_allclose(float(loss_value.item()), float(torch_loss_value.item()), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        np.asarray(mx_gradient), torch_gradient.detach().numpy(), rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        np.asarray(mx_updated), torch_updated.detach().numpy(), rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        np.asarray(mx_retracted), torch_retracted.detach().numpy(), rtol=2e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(mx_momentum), torch_momentum.detach().numpy(), rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        np.asarray(mx_velocity),
        (torch_velocity).detach().numpy(),
        rtol=2e-6,
        atol=2e-6,
    )
    expected_bits = float((torch.softmax(torch_retracted.detach(), dim=-1) * costs).sum(-1).mul(fraction).sum())
    actual_bits = float(rco_mlx._expected_cost(mx_retracted, mx_costs, mx_fraction).item())
    assert actual_bits == pytest.approx(expected_bits, abs=1e-6)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_mlx_projection_matches_torch_oracle_in_low_precision(dtype_name):
    mx, _ = mlx_modules()
    model = tiny_mlx_model()
    dtype = getattr(mx, dtype_name)
    model.set_dtype(dtype)
    mx.eval(model.parameters())

    name = "model.layers.0.mlp.down_proj"
    source = to_torch(rco_mlx._child(model, name).weight).detach().cpu()
    torch_dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    source = source.to(torch_dtype).float()
    torch.manual_seed(73)
    low = source * 0.25 + torch.randn(source.shape) * 0.01
    reference = source + torch.randn(source.shape) * 0.01
    probabilities = torch.tensor([0.35, 0.65], dtype=torch.float32)
    interpolated = rco_mlx.MLXInterpolatedLlamaLayers(
        model,
        {name: [low, reference]},
        torch.tensor([source.numel()]),
        torch.tensor([2.0, 3.0]),
    )
    interpolated.set_probs(mx.array(probabilities[None].numpy()))
    torch.manual_seed(79)
    values = torch.randn(3, source.shape[1], dtype=torch.float32, device="cpu")
    values = values.to(torch_dtype).float()
    module = rco_mlx._child(model, name)
    try:
        actual = module(mx.array(values.numpy()).astype(dtype))
        assert actual.dtype == dtype
        raw = to_torch(actual.astype(mx.float32)).detach().cpu().float()
        visible = to_torch(actual).detach().cpu().float()

        # Reconstruct the exact candidate storage and model-dtype interpolation
        # used by both backends, then evaluate the projection in float64. This
        # separates kernel arithmetic error from the required output cast.
        stored_low = mx.array(low.numpy()).astype(dtype)
        stored_reference = mx.array(reference.numpy()).astype(dtype)
        mixed_delta = stored_low - stored_reference
        mixed = stored_reference + mx.array(probabilities[0].item()).astype(dtype) * mixed_delta
        mx.eval(mixed)
        mixed_oracle = to_torch(mixed).detach().cpu().float()
        oracle = values.double() @ mixed_oracle.double().T
        rounded = oracle.to(torch_dtype).float()

        np.testing.assert_allclose(
            raw.numpy(), oracle.float().numpy(), rtol=2e-3, atol=2e-3
        )
        visible_drift = float((visible - rounded).abs().max())
        tolerance = 2e-3 + 2e-3 * rounded.abs()
        outside_tolerance = (visible - rounded).abs() > tolerance
        if dtype_name == "bfloat16" and bool(outside_tolerance.any()):
            # A BF16 output step is larger than 2e-3 around zero. At those
            # values, require the visible result to be exactly the adjacent
            # representable BF16 value, not an arbitrary exception window.
            rounded_bf16 = rounded.to(torch.bfloat16)
            lower = torch.nextafter(
                rounded_bf16, torch.full_like(rounded_bf16, float("-inf"))
            ).float()
            upper = torch.nextafter(
                rounded_bf16, torch.full_like(rounded_bf16, float("inf"))
            ).float()
            one_step = outside_tolerance & (
                (visible == lower) | (visible == upper)
            )
            assert bool(one_step[outside_tolerance].all())
        else:
            assert not bool(outside_tolerance.any())
        assert visible_drift <= 0.00390625
    finally:
        interpolated.cleanup()


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_mlx_interpolation_matches_torch_backend_precision(dtype_name):
    """MLX must interpolate in the same dtype as the Torch backend.

    ``rco._WeightInterpolation`` multiplies a zero-dimensional float32
    probability by a model-dtype delta, and Torch's scalar type promotion keeps
    that product in the model dtype. MLX promotes a float32 array against a
    low-precision array to float32 instead, which would widen every
    interpolated projection and search a different objective than Torch.
    """
    mx, _ = mlx_modules()
    mx.random.seed(20260518)
    torch.manual_seed(20260518)
    model = tiny_mlx_model()
    dtype = getattr(mx, dtype_name)
    model.set_dtype(dtype)
    mx.eval(model.parameters())

    name = "model.layers.0.mlp.down_proj"
    source = to_torch(rco_mlx._child(model, name).weight).detach().cpu()
    torch_dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    source = source.to(torch_dtype)
    low = (source.float() * 0.7).to(torch_dtype)
    mid = (source.float() * 0.85).to(torch_dtype)
    probabilities = torch.tensor([0.35, 0.42, 0.23], dtype=torch.float32)

    class _Owner:
        def __init__(self):
            self._deltas = [(low - source).to(torch_dtype), (mid - source).to(torch_dtype)]
            self._probs = probabilities

        def deltas(self, _name):
            return self._deltas

        def probabilities(self, _index):
            return self._probs

        def index_of(self, _name):
            return 0

        def reference_index(self):
            return len(self._deltas)

    with torch.no_grad():
        torch_mixed = rco._WeightInterpolation(_Owner(), name)(source).to(torch_dtype)
        values = torch.randn(64, source.shape[1], dtype=torch.float32).to(torch_dtype)
        torch_out = (values @ torch_mixed.T).to(torch_dtype).float()

    interpolated = rco_mlx.MLXInterpolatedLlamaLayers(
        model,
        {name: [low.float(), mid.float(), source.float()]},
        torch.tensor([source.numel()]),
        torch.tensor([2.0, 3.0, 4.0]),
    )
    interpolated.set_probs(mx.array(probabilities[None].numpy()))
    try:
        actual = rco_mlx._child(model, name)(
            mx.array(values.float().numpy()).astype(dtype)
        )
        assert actual.dtype == dtype
        visible = to_torch(actual).detach().cpu().float()
        drift = (visible - torch_out).abs()
        tolerance = 2e-3 + 2e-3 * torch_out.abs()
        outside_tolerance = drift > tolerance
        if dtype_name == "bfloat16" and bool(outside_tolerance.any()):
            # A BF16 output step is larger than 2e-3 around zero. Require any
            # out-of-tolerance element to be exactly the adjacent representable
            # BF16 value, so kernel error stays distinguishable from the
            # mandatory output rounding both backends perform.
            rounded = torch_out.to(torch.bfloat16)
            lower = torch.nextafter(
                rounded, torch.full_like(rounded, float("-inf"))
            ).float()
            upper = torch.nextafter(
                rounded, torch.full_like(rounded, float("inf"))
            ).float()
            one_step = outside_tolerance & ((visible == lower) | (visible == upper))
            assert bool(one_step[outside_tolerance].all())
        else:
            assert not bool(outside_tolerance.any())
        # Both backends must run the same arithmetic, so only mandatory output
        # rounding may separate the two visible results. Interpolating in
        # float32 instead changes most of the output bits.
        differing = visible.to(torch_dtype) != torch_out.to(torch_dtype)
        assert int(differing.sum()) <= max(2, int(0.01 * differing.numel()))
    finally:
        interpolated.cleanup()


def test_mlx_search_restores_original_linear_and_exact_budget():
    mx, _ = mlx_modules()
    model = tiny_mlx_model()
    name = "model.layers.0.mlp.down_proj"
    original = rco_mlx._child(model, name)
    source = to_torch(original.weight)
    interpolated = rco_mlx.MLXInterpolatedLlamaLayers(
        model,
        {name: [source * 0.2, source * 0.8]},
        torch.tensor([source.numel()]),
        torch.tensor([2.0, 3.0]),
    )
    interpolated.alpha = mx.array([[-0.4, 0.4]], dtype=mx.float32)
    interpolated.retract(2.5)
    objective = rco_mlx.MLXTeacherKL(model, [np.array([[1, 2, 3]])], "/tmp/rco-mlx-test", resident_cache=True)
    try:
        probabilities, history = rco_mlx.projected_gumbel_search_mlx(
            interpolated, [0], 2.5, steps=2, gumbel_samples=2,
            objective=objective, log_every=0,
        )
        assert probabilities.shape == (1, 2)
        assert abs(interpolated.current_bits() - 2.5) < 1e-6
        assert max(record["drift"] for record in history) < 1e-6
    finally:
        objective.cleanup()
        interpolated.cleanup()
    assert rco_mlx._child(model, name) is original
    assert not objective._resident and not objective.paths and not objective._entropy


def test_mlx_search_honors_batch_ids_weights_and_clears_override_on_error():
    mx, _ = mlx_modules()
    model = tiny_mlx_model()
    name = "model.layers.0.mlp.down_proj"
    source = to_torch(rco_mlx._child(model, name).weight)
    search = rco_mlx.MLXInterpolatedLlamaLayers(
        model, {name: [source * 0.2, source]}, [source.numel()], [2., 3.]
    )
    calls = []

    def objective(batch):
        calls.append(batch)
        return mx.sum(search.probabilities(0)) * batch

    try:
        _, history = rco_mlx.projected_gumbel_search_mlx(
            search, [7, 11], 2.5, steps=1, gumbel_samples=2,
            batch_weights=[1, 3], objective=objective, log_every=0,
        )
        assert calls == [7, 11, 7, 11]
        assert history[0]["loss"] == pytest.approx(10., abs=1e-6)

        def failed(_):
            raise RuntimeError("objective failed")

        with pytest.raises(RuntimeError, match="objective failed"):
            rco_mlx.projected_gumbel_search_mlx(
                search, [7], 2.5, steps=1, objective=failed, log_every=0,
            )
        assert search._probs_override is None
    finally:
        search.cleanup()
