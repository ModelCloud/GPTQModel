"""QQQ candidate-grid prerequisites; these fixtures make no quality claim."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.quantization.gsq_qqq import qqq_candidate_values


@pytest.mark.parametrize("ratio", [0.5, 1.5, 17.0])
def test_grouped_candidate_grid_matches_packed_reload(ratio):
    # Every stored nibble, including ties and INT8 saturation, in two groups.
    codes = torch.arange(16).repeat(16).reshape(1, 256).expand(64, -1)
    channel = torch.full((64,), 0.25, dtype=torch.float32)
    scales = torch.full((64, 2), ratio * 0.25, dtype=torch.float16)
    linear = torch.nn.Linear(256, 64, bias=False, dtype=torch.float16)
    linear.weight.data.copy_((codes - 8) * scales.repeat_interleave(128, dim=1))

    def make_module():
        return QQQTorchLinear(bits=4, group_size=128, sym=True, desc_act=False,
                              in_features=256, out_features=64, bias=False)

    packed = make_module()
    packed.pack(linear, scales, channel)
    loaded = make_module()
    loaded.load_state_dict(packed.state_dict(), strict=True)
    assert torch.equal(loaded._unpack_weight_codes(), codes.T)
    integer_weight, stored_channel = loaded._dequantize_weight_for_torch()
    expected_integer = ((codes.T - 8).float() * ratio).round().clamp(-128, 127)
    assert torch.equal(integer_weight, expected_integer)
    assert torch.equal(stored_channel, channel.reshape(1, -1))
    deployed = integer_weight * stored_channel
    affine = (codes.T - 8).float() * (ratio * 0.25)
    assert not torch.equal(deployed, affine)

    decoded = qqq_candidate_values(codes, scales, group_size=128, in_features=256,
                                   channel_scales=channel)
    assert torch.equal(decoded.T, deployed)
    candidates = codes.unsqueeze(-1).expand(-1, -1, 3)
    candidate_values = qqq_candidate_values(candidates, scales, group_size=128, in_features=256,
                                            channel_scales=channel)
    assert torch.equal(candidate_values, decoded.unsqueeze(-1).expand_as(candidate_values))


def test_channelwise_candidate_grid_matches_packed_reload():
    signed = torch.arange(-7, 8).repeat(18)[:256].reshape(1, 256).expand(64, -1)
    scales = torch.full((64, 1), 0.03125, dtype=torch.float16)
    linear = torch.nn.Linear(256, 64, bias=False, dtype=torch.float16)
    linear.weight.data.copy_(signed * scales)
    packed = QQQTorchLinear(bits=4, group_size=-1, sym=True, desc_act=False,
                            in_features=256, out_features=64, bias=False)
    packed.pack(linear, scales)
    loaded = QQQTorchLinear(bits=4, group_size=-1, sym=True, desc_act=False,
                            in_features=256, out_features=64, bias=False)
    loaded.load_state_dict(packed.state_dict(), strict=True)
    packed = loaded
    integer_weight, channel = packed._dequantize_weight_for_torch()
    codes = packed._unpack_weight_codes().T
    decoded = qqq_candidate_values(codes, scales, group_size=-1, in_features=256)
    assert torch.equal(decoded.T, integer_weight * channel)
    assert torch.equal(decoded, linear.weight.float())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 16.0, 1.5])
def test_candidate_decoder_rejects_invalid_codes(bad):
    with pytest.raises(ValueError, match="nibbles"):
        qqq_candidate_values(torch.full((1, 128), bad), torch.ones(1, 1),
                             group_size=-1, in_features=128)


@pytest.mark.parametrize("group_size", [-1, 128])
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_candidate_decoder_rejects_invalid_scales(group_size, bad):
    groups = 1 if group_size == -1 else 2
    with pytest.raises(ValueError, match="scales"):
        qqq_candidate_values(torch.zeros(1, 256), torch.full((1, groups), bad),
                             group_size=group_size, in_features=256, channel_scales=torch.ones(1))


def test_candidate_mixture_has_gradient_over_decoded_values():
    codes = torch.tensor([8, 9, 10]).expand(1, 256, 3)
    values = qqq_candidate_values(codes, torch.full((1, 2), 0.125),
                                  group_size=128, in_features=256, channel_scales=torch.tensor([0.25]))
    assert torch.equal(values[0, 0], torch.tensor([0.0, 0.0, 0.25]))
    logits = torch.zeros_like(values, requires_grad=True)
    mixture = (logits.softmax(-1) * values).sum(-1)
    mixture.sum().backward()
    expected = (values - values.mean(-1, keepdim=True)) / 3
    torch.testing.assert_close(logits.grad, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_calibration_moments_match_runtime_and_explicit_loss(dtype):
    from gptqmodel.quantization.gsq_qqq import qqq_calibration_moments
    from gptqmodel.quantization.gsq_scalar import asymmetric_error_term

    rng = torch.Generator().manual_seed(7)
    inputs = torch.randn(19, 128, generator=rng).to(dtype)
    native = inputs.float() + torch.randn(19, 128, generator=rng) * 0.03
    hessian, cross, count = qqq_calibration_moments(inputs, teacher_inputs=native)
    codes, scale = QQQTorchLinear.dynamic_quant(None, inputs.half())
    deployed = codes.float() * scale
    assert count == 19
    assert torch.equal(hessian, deployed.T @ deployed)
    assert torch.equal(cross, (native - deployed).T @ deployed)
    weight = torch.randn(4, 128, generator=rng).double()
    error = torch.randn(4, 128, generator=rng).double() * 0.1
    exact = ((weight + error) @ deployed.double().T - weight @ native.double().T).square().sum()
    constant = (weight @ (deployed.double() - native.double()).T).square().sum()
    quadratic = (error @ hessian.double() * error).sum()
    quadratic += asymmetric_error_term(error, weight, cross.double())
    torch.testing.assert_close(quadratic, exact - constant, atol=1e-4, rtol=1e-5)
    h1, d1, n1 = qqq_calibration_moments(inputs[:7], teacher_inputs=native[:7])
    h2, d2, n2 = qqq_calibration_moments(inputs[7:], teacher_inputs=native[7:])
    torch.testing.assert_close(h1 + h2, hessian)
    torch.testing.assert_close(d1 + d2, cross)
    assert n1 + n2 == count


def test_calibration_zero_rows_and_runtime_underflow():
    from gptqmodel.quantization.gsq_qqq import qqq_calibration_moments

    hessian, cross, count = qqq_calibration_moments(torch.zeros(3, 128))
    assert count == 3
    assert torch.count_nonzero(hessian) == torch.count_nonzero(cross) == 0
    with pytest.raises(ValueError, match="underflows"):
        qqq_calibration_moments(torch.full((1, 128), 2**-24))
    with pytest.raises(ValueError, match="overflow"):
        qqq_calibration_moments(torch.full((1, 128), 1e10))


@pytest.mark.parametrize("group_size", [-1, 128])
@pytest.mark.parametrize("candidates", [3, 16])
def test_qqq_optimizer_guard_and_determinism(group_size, candidates):
    from gptqmodel.quantization.config import GSQConfig
    from gptqmodel.quantization.gsq_qqq import refine_qqq_codes

    codes = torch.full((2, 256), 9, dtype=torch.int64)
    scales = torch.full((2, 1 if group_size == -1 else 2), 0.125)
    channel = None if group_size == -1 else torch.full((2,), 0.25)
    target = torch.full((2, 256), 0.25)
    kwargs = dict(target=target, group_size=group_size, hessian=torch.eye(256),
                  cross_moment=torch.zeros(256, 256), channel_scales=channel,
                  config=GSQConfig(enabled=True, steps=8, candidates=candidates))
    rng = torch.random.get_rng_state().clone()
    first = refine_qqq_codes(codes, scales, **kwargs)
    second = refine_qqq_codes(codes, scales, **kwargs)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(first[0], second[0])
    assert first[1:] == second[1:]
    assert first[2] <= first[1]
    assert len(first[3]) == 9
    assert torch.equal(codes, torch.full_like(codes, 9))
    decoded = qqq_candidate_values(first[0], scales, group_size=group_size,
                                   in_features=256, channel_scales=channel)
    expected = (decoded - target).square().sum() / target.square().sum()
    assert first[2] == pytest.approx(float(expected))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("group_size", [-1, 128])
def test_code_transport_survives_actual_packer(dtype, group_size):
    from gptqmodel.quantization.gsq_qqq import qqq_codes_to_packer_weight

    codes = torch.arange(16).repeat(16).reshape(1, 256).expand(64, -1)
    scales = torch.full((64, 1 if group_size == -1 else 2), 0.03125, dtype=torch.float16)
    weight = qqq_codes_to_packer_weight(codes, scales, group_size=group_size, dtype=dtype)
    linear = torch.nn.Linear(256, 64, bias=False, dtype=dtype)
    linear.weight.data.copy_(weight)
    packed = QQQTorchLinear(bits=4, group_size=group_size, sym=True, desc_act=False,
                            in_features=256, out_features=64, bias=False)
    packed.pack(linear, scales, torch.full((64,), 0.25))
    assert torch.equal(packed._unpack_weight_codes().T, codes)


@pytest.mark.parametrize("group_size", [-1, 128])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("desc_act", [False, True])
@pytest.mark.parametrize("static_groups", [False, True])
def test_real_qqq_quantizer_hook_and_packing(group_size, enabled, desc_act, static_groups):
    from gptqmodel.quantization.config import GSQConfig, QQQConfig
    from gptqmodel.quantization.qqq import QQQ

    rng = torch.Generator().manual_seed(7)
    layer = torch.nn.Linear(256, 64, bias=False, dtype=torch.float16)
    layer.weight.data.copy_(torch.randn(64, 256, generator=rng) * 0.03)
    config = QQQConfig(bits=4, group_size=group_size, desc_act=desc_act, static_groups=static_groups,
                       gsq=GSQConfig(enabled=enabled, steps=2))
    quantizer = QQQ(layer, config)
    quantizer.quantizer.configure(4, perchannel=True, sym=True, mse=False, groupsize=group_size)
    inputs = torch.randn(320, 256, generator=rng).half()
    quantizer.add_batch(inputs[:160], None)
    quantizer.add_batch(inputs[160:], None)
    assert bool(quantizer._gsq_moments) == enabled
    teacher = layer.weight.detach().float().clone()
    result = quantizer.quantize()
    if enabled:
        assert quantizer.gsq_diagnostics['after'] <= quantizer.gsq_diagnostics['before']
    assert not quantizer._gsq_moments
    layer.weight.data.copy_(result[0])
    packed = QQQTorchLinear(bits=4, group_size=group_size, sym=True, desc_act=False,
                            in_features=256, out_features=64, bias=False)
    packed.pack(layer, result[1], result[7])
    assert torch.isfinite(packed(inputs[:2])).all()
    if enabled:
        integer_weight, channel = packed._dequantize_weight_for_torch()
        deployed_weight = (integer_weight * channel).T
        input_codes, input_scale = packed.dynamic_quant(inputs)
        deployed_inputs = input_codes.float() * input_scale
        teacher_output = inputs.float() @ teacher.T
        loss = (deployed_inputs @ deployed_weight.T - teacher_output).square().sum()
        constant = ((deployed_inputs - inputs.float()) @ teacher.T).square().sum()
        denominator = (deployed_inputs @ teacher.T).square().sum()
        expected = float((loss - constant) / denominator)
        assert quantizer.gsq_diagnostics["after"] == pytest.approx(expected, rel=2e-5, abs=1e-7)
    quantizer.free()


@pytest.mark.parametrize("group_size", [-1, 128])
@pytest.mark.parametrize("desc_act", [False, True])
def test_qqq_disabled_and_unmatched_match_original_initializer(group_size, desc_act):
    from gptqmodel.quantization.config import GSQConfig, QQQConfig, QuantizeConfig
    from gptqmodel.quantization.qqq import QQQ

    rng = torch.Generator().manual_seed(17)
    initial = torch.randn(64, 256, generator=rng).half() * 0.03
    inputs = torch.randn(320, 256, generator=rng).half()
    outputs = []
    for control in (None, GSQConfig(enabled=False), GSQConfig(enabled=True, modules=("never_match",))):
        config = QQQConfig(bits=4, group_size=group_size, desc_act=desc_act, gsq=control)
        restored = QuantizeConfig.from_quant_config(config.to_dict())
        assert isinstance(restored, QQQConfig)
        assert restored.gsq == config.gsq
        layer = torch.nn.Linear(256, 64, bias=False, dtype=torch.float16)
        layer.weight.data.copy_(initial)
        quantizer = QQQ(layer, restored)
        quantizer.quantizer.configure(4, perchannel=True, sym=True, mse=False, groupsize=group_size)
        quantizer.add_batch(inputs, None)
        assert not quantizer._gsq_moments
        state = torch.random.get_rng_state().clone()
        result = quantizer.quantize()
        assert torch.equal(state, torch.random.get_rng_state())
        assert not hasattr(quantizer, "gsq_diagnostics")
        outputs.append(result[:4] + (result[7],))
        quantizer.free()
    layer = torch.nn.Linear(256, 64, bias=False, dtype=torch.float16)
    layer.weight.data.copy_(initial)
    original = QQQ(layer, QQQConfig(bits=4, group_size=group_size, desc_act=desc_act))
    original.quantizer.configure(4, perchannel=True, sym=True, mse=False, groupsize=group_size)
    original.add_batch(inputs, None)
    result = original._quantize_impl()
    expected = result[:4] + (result[7],)
    for output in outputs:
        for actual, reference in zip(output, expected, strict=True):
            assert actual is None if reference is None else torch.equal(actual, reference)
    original.free()


@pytest.mark.parametrize("damp_percent", [None, 0.02])
def test_qqq_effective_damping_roundtrip(damp_percent):
    from gptqmodel.quantization.config import GSQConfig, QQQConfig, QuantizeConfig

    config = QQQConfig(bits=4, group_size=128, damp_percent=damp_percent,
                       gsq=GSQConfig(enabled=True, steps=2))
    restored = QuantizeConfig.from_quant_config(config.to_dict())
    assert config.damp.min == (0.005 if damp_percent is None else damp_percent)
    assert restored.damp == config.damp
    assert restored.damp_percent == config.damp_percent == config.damp.min
    assert restored.damp_auto_increment == config.damp_auto_increment == config.damp.step


@pytest.mark.parametrize("group_size", [-1, 128])
@pytest.mark.parametrize("tokens", [1, 17])
def test_gsq_qqq_native_reload_matches_torch(group_size, tokens):
    import os
    if not os.environ.get("GPU_ALLOCATOR_LEASE_ID"):
        pytest.skip("requires an exclusive GPU lease")
    from gptqmodel.nn_modules.qlinear.qqq import QQQLinear
    from gptqmodel.quantization.config import GSQConfig, QQQConfig
    from gptqmodel.quantization.qqq import QQQ

    rng = torch.Generator().manual_seed(7)
    layer = torch.nn.Linear(256, 128, bias=False, dtype=torch.float16)
    layer.weight.data.copy_(torch.randn(128, 256, generator=rng) * 0.03)
    quantizer = QQQ(layer, QQQConfig(bits=4, group_size=group_size, desc_act=False,
                                    gsq=GSQConfig(enabled=True, steps=10)))
    quantizer.quantizer.configure(4, perchannel=True, sym=True, mse=False, groupsize=group_size)
    calibration = torch.randn(320, 256, generator=rng).half()
    quantizer.add_batch(calibration, None)
    weight, scales, _, _, _, _, _, extra, _ = quantizer.quantize()
    layer.weight.data.copy_(weight)
    kwargs = dict(bits=4, group_size=group_size, sym=True, desc_act=False,
                  in_features=256, out_features=128, bias=False)
    reference = QQQTorchLinear(**kwargs)
    reference.pack(layer, scales, extra)
    native = QQQLinear(**kwargs)
    native.load_state_dict(reference.state_dict(), strict=True)
    native = native.cuda().eval()
    native.post_init()
    inputs = torch.randn(tokens, 256, generator=rng).half()
    actual = native(inputs.cuda()).cpu().float()
    expected = reference(inputs).float()
    delta = (actual - expected).abs()
    assert torch.isfinite(actual).all()
    assert delta.mean() <= 0.002
    assert delta.max() <= 0.046875
    print("QQQ_GSQ_NATIVE", group_size, tokens, float(delta.mean()), float(delta.max()))

    # Exercise the public backend on a non-default stream. Keep the input,
    # module workspace and graph-owned output alive across changed-input replays.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    static_input = inputs.cuda()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            native(static_input)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured_output = native(static_input)
    for _ in range(3):
        replay_input = torch.randn(tokens, 256, generator=rng).half()
        with torch.cuda.stream(stream):
            static_input.copy_(replay_input)
            graph.replay()
            replay_output = captured_output.clone()
            eager_output = native(static_input)
        stream.synchronize()
        torch.testing.assert_close(replay_output, eager_output, rtol=0, atol=0)
        replay_delta = (replay_output.cpu().float() - reference(replay_input).float()).abs()
        assert torch.isfinite(replay_output).all()
        assert replay_delta.mean() <= 0.002
        assert replay_delta.max() <= 0.046875
    print("QQQ_GSQ_GRAPH", group_size, tokens, "3 changed-input replays passed")
    quantizer.free()
