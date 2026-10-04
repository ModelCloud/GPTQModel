# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel.nn_modules.qlinear.w4a_activation import W4AActivation
from gptqmodel.nn_modules.qlinear.w4a_boundary import NVFP4BoundaryQuantizer, llama_nvfp4_boundaries
from gptqmodel.nn_modules.qlinear.w4a_llama_stream import install_w4a_llama_stream
from gptqmodel.nn_modules.qlinear.w4a_nvfp4 import W4ANVFP4Linear
from gptqmodel.quantization.activation_calibration import (
    BoundaryMaximum,
    calibrate_nvfp4_producers,
    measure_nvfp4_producers,
)
from gptqmodel.quantization.config import QuantizeConfig
from tests.kernels.test_w4a_stream import _independent_nvfp4_qdq


def test_maximum_scale_fp32_and_zero():
    maximum = BoundaryMaximum()
    with pytest.raises(ValueError, match="unobserved"):
        maximum.scale("nvidia")
    maximum.observe(torch.zeros(2, 128))
    assert maximum.scale("nvidia") == 1
    maximum.observe(torch.full((3, 128), 6 * 448, dtype=torch.float32))
    assert maximum.scale("nvidia") == 1
    assert maximum.scale("least_squares") == 1.5
    assert maximum.tokens == 5
    with pytest.raises(ValueError, match="finite"):
        maximum.observe(torch.full((1, 128), float("nan")))


def test_move_moves_the_reference_with_the_carrier():
    """A cached carrier must not retain its full residual on the source device."""
    from gptqmodel.quantization.activation_calibration import _move

    device = "cuda" if torch.cuda.is_available() else "cpu"
    value = W4AActivation(
        "w4a_nvfp4",
        torch.zeros((2, 8), dtype=torch.uint8, device=device),
        torch.zeros((1, 16), dtype=torch.float8_e4m3fn, device=device),
        (2, 128), torch.bfloat16,
        global_scale=torch.ones((), device=device),
        reference=torch.zeros((2, 128), device=device),
    )
    moved = _move(value, "cpu")
    assert moved.codes.device.type == "cpu"
    assert moved.reference.device.type == "cpu"
    assert moved.reference is not value.reference
    assert moved.exact().device.type == "cpu"


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), 1e-60, 1e60])
def test_config_rejects_invalid_global_scale(value):
    with pytest.raises(ValueError, match="scale"):
        QuantizeConfig(bits=4, group_size=128, desc_act=False, sym=True, rotation="hadamard",
                       activation={"version": 4, "mode": "w4a_nvfp4",
                                   "global_scales": {"model.layers.0.input": value}})


def test_fp32_global_scales_serialize_without_bf16_rounding(tmp_path):
    value = torch.tensor(0.01234567).item()
    config = QuantizeConfig(bits=4, group_size=128, desc_act=False, sym=True, rotation="hadamard",
                            activation={"version": 4, "mode": "w4a_nvfp4",
                                        "global_scales": {"model.layers.0.input": value}})
    config.save_pretrained(str(tmp_path))
    restored = QuantizeConfig.from_pretrained(str(tmp_path))
    assert restored.activation_global_scales == {"model.layers.0.input": value}
    quantizer = NVFP4BoundaryQuantizer("model.layers.0.input", "cpu", value).bfloat16()
    assert quantizer.global_scale.item() == value
    assert quantizer.global_scale.dtype == torch.float32
    assert not quantizer.state_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows,width", [(1, 128), (33, 256), (129, 128)])
def test_fixed_producer_scale_matches_independent_oracle(rows, width):
    generator = torch.Generator(device="cuda").manual_seed(840 + rows)
    x = torch.randn((rows, width), generator=generator, device="cuda", dtype=torch.bfloat16)
    x[0, :16] = 0
    module = NVFP4BoundaryQuantizer("model.layers.0.input", "cuda", 0.01234567).bfloat16()
    actual = module(x, "w4a_nvfp4", recipe="least_squares")
    assert actual.global_scale.data_ptr() == module.global_scale.data_ptr()
    expected = _independent_nvfp4_qdq(x, torch.tensor(0.01234567, device="cuda"))
    torch.testing.assert_close(actual.decode(torch.float32), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fixed_producer_scale_preserves_exact_representable_codes():
    palette = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -.5, -1., -1.5, -2., -3., -4., -6.],
                           device="cuda")
    codebook_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
                                    device="cuda", dtype=torch.uint8)
    indices = torch.arange(33 * 128, device="cuda").reshape(33, 128) % len(palette)
    x = (palette[indices] / 32).bfloat16()
    quantizer = NVFP4BoundaryQuantizer("model.layers.0.input", "cuda", 1 / 32)
    encoded = quantizer(x, "w4a_nvfp4", recipe="least_squares")
    expected_codes = codebook_indices[indices]
    expected_packed = expected_codes[:, ::2] | (expected_codes[:, 1::2] << 4)
    assert torch.equal(encoded.codes.view(torch.uint8), expected_packed)
    # Invert the hardware scale layout independently and inspect active rows.
    local = encoded.scales.float().reshape(1, 1, 2, 32, 4, 4).permute(1, 4, 3, 0, 2, 5).reshape(128, 8)[:33]
    assert torch.equal(local, torch.ones_like(local))
    torch.testing.assert_close(encoded.decode(torch.float32), x.float(), rtol=0, atol=0)


def _tiny_packed_nvfp4(dtype=torch.bfloat16):
    """Build a packed NVFP4 Llama without installing a runtime stream."""
    torch.manual_seed(840)
    config = LlamaConfig(vocab_size=32, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                         max_position_embeddings=128)
    core = LlamaForCausalLM(config).to(device="cuda", dtype=dtype).eval()
    for layer in core.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, dense in list(parent.named_children()):
                if not isinstance(dense, torch.nn.Linear):
                    continue
                linear = W4ANVFP4Linear(bits=4, group_size=128, sym=True, desc_act=False,
                                       in_features=dense.in_features, out_features=dense.out_features,
                                       bias=False).to(device="cuda", dtype=dtype)
                linear.qweight.random_(-(2**31), 2**31 - 1)
                linear.qzeros.fill_(0x77777777)
                linear.scales.fill_(0.002)
                linear.activation_global_scale.fill_(0.01)
                linear.post_init()
                setattr(parent, name, linear)
    return core


def _tiny_stream(scales=None, dtype=torch.bfloat16):
    core = _tiny_packed_nvfp4(dtype=dtype)
    install_w4a_llama_stream(core, "w4a_nvfp4", "least_squares", version=4, global_scales=scales)
    return core


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_headroom_norm_scale_follows_attention_and_mlp_policies(dtype):
    """An FP8 MLP promotion must not drop the attention headroom norm scale.

    The headroom setup previously keyed off the MLP policy alone, so a
    version-3 stream with NVFP4 headroom attention and a promoted FP8 MLP left
    the input norm without its global scale and raised on the first norm call.
    """
    norm_scale = "_w4a_output_global_" + "scale"
    packed = _tiny_packed_nvfp4(dtype=dtype)
    install_w4a_llama_stream(
        packed, "w4a_nvfp4", "nvidia_headroom", version=3,
        attention_mode="w4a_nvfp4", attention_recipe="nvidia_headroom",
        mlp_fp8_layers=(0,),
    )
    layer0, layer1 = packed.model.layers
    assert getattr(layer0.input_layernorm, norm_scale, None) is not None
    assert getattr(layer1.input_layernorm, norm_scale, None) is not None
    # Layer 0's MLP is FP8, so its post-attention norm owns no NVFP4 scale.
    assert not hasattr(layer0.post_attention_layernorm, norm_scale)
    assert getattr(layer1.post_attention_layernorm, norm_scale, None) is not None
    with torch.inference_mode():
        logits = packed(input_ids=torch.arange(17, device="cuda")[None], use_cache=False).logits
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp8_attention_does_not_demand_qkv_headroom_scales(dtype):
    """An FP8 attention side must not validate scales it never consumes."""
    norm_scale = "_w4a_output_global_" + "scale"
    packed = _tiny_packed_nvfp4(dtype=dtype)
    layer0 = packed.model.layers[0]
    # Distinct Q/K/V buffers would fail a shared-carrier check if it ran.
    layer0.self_attn.q_proj.activation_global_scale.fill_(0.01)
    layer0.self_attn.k_proj.activation_global_scale.fill_(0.02)
    layer0.self_attn.v_proj.activation_global_scale.fill_(0.03)
    install_w4a_llama_stream(
        packed, "w4a_nvfp4", "nvidia_headroom", version=3,
        attention_mode="w4afp8", attention_recipe=None,
        mlp_fp8_layers=(),
    )
    assert not hasattr(layer0.input_layernorm, norm_scale)
    assert getattr(layer0.post_attention_layernorm, norm_scale, None) is not None
    with torch.inference_mode():
        logits = packed(input_ids=torch.arange(17, device="cuda")[None], use_cache=False).logits
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_attention_headroom_with_dynamic_mlp_keeps_input_norm_scale(dtype):
    """NVFP4 headroom attention with a dynamic MLP still owns its norm scale."""
    norm_scale = "_w4a_output_global_" + "scale"
    packed = _tiny_packed_nvfp4(dtype=dtype)
    install_w4a_llama_stream(
        packed, "w4a_nvfp4", "least_squares", version=3,
        attention_mode="w4a_nvfp4", attention_recipe="nvidia_headroom",
        mlp_fp8_layers=(),
    )
    layer0 = packed.model.layers[0]
    assert getattr(layer0.input_layernorm, norm_scale, None) is not None
    assert not hasattr(layer0.post_attention_layernorm, norm_scale)
    with torch.inference_mode():
        logits = packed(input_ids=torch.arange(17, device="cuda")[None], use_cache=False).logits
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_v4_replay_norm_operand_matches_actual_encoded_runtime_and_independent_oracle(dtype):
    from types import SimpleNamespace

    from gptqmodel.nn_modules.qlinear.w4a_llama_replay import install_w4a_llama_replay

    runtime = _tiny_stream(dtype=dtype)
    scales = {}
    for module in runtime.modules():
        if isinstance(module, NVFP4BoundaryQuantizer):
            module.set_scale(.01234567)
            scales[module.key] = .01234567
    replay = LlamaForCausalLM(runtime.config).to(device="cuda", dtype=dtype).eval()
    replay.model.embed_tokens.weight.data.copy_(runtime.model.embed_tokens.weight)
    qcfg = SimpleNamespace(activation_mode="w4a_nvfp4", activation_recipe="least_squares",
                           activation_version=4, activation_global_scales=scales,
                           dynamic_get=lambda **_kwargs: None)
    install_w4a_llama_replay(replay, qcfg)
    ids = torch.arange(17, device="cuda")[None]
    source = runtime.model.embed_tokens(ids).detach()
    independent = _independent_nvfp4_qdq(source, torch.tensor(.01234567, device="cuda"))
    # The deployed stream rescales the rounded carrier by the inverse RMS of the
    # pristine residual, so the oracle keeps the unquantized denominator.
    expected = independent * torch.rsqrt(source.float().square().mean(-1, keepdim=True)
                                         + runtime.config.rms_norm_eps)
    captured = []
    class Captured(Exception):
        pass
    def capture(_module, args):
        value = args[0]
        captured.append(value.decode(torch.float32) if isinstance(value, W4AActivation) else value)
        raise Captured
    for model in (runtime, replay):
        handle = model.model.layers[0].self_attn.q_proj.register_forward_pre_hook(capture)
        try:
            with torch.inference_mode(), pytest.raises(Captured):
                model(input_ids=ids, use_cache=False)
        finally:
            handle.remove()
    torch.cuda.synchronize()
    for actual in captured:
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(captured[0], captured[1], rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fixed_scales", [False, True])
def test_v4_weight_adaptation_replay_matches_encoded_decoder_outputs(dtype, fixed_scales, monkeypatch):
    """Exercise the zero-update training function through two complete layers."""
    from types import SimpleNamespace

    from gptqmodel.nn_modules.qlinear import w4a_llama_replay
    from gptqmodel.nn_modules.qlinear.w4a_llama_replay import install_w4a_llama_replay
    from tests.models.w4a_nvfp4_weight_qad import _install_trainable_gptq_codes, straight_through_activation_round

    monkeypatch.setattr(w4a_llama_replay, "_round", straight_through_activation_round)

    runtime = _tiny_stream(dtype=dtype)
    scales = None
    if fixed_scales:
        scales = {}
        for module in runtime.modules():
            if isinstance(module, NVFP4BoundaryQuantizer):
                module.set_scale(.01234567)
                scales[module.key] = .01234567
    replay = LlamaForCausalLM(runtime.config).to(device="cuda", dtype=dtype).eval()
    runtime_modules = dict(runtime.named_modules())
    with torch.no_grad():
        for name, parameter in replay.named_parameters():
            owner, leaf = name.rsplit(".", 1)
            if not isinstance(runtime_modules[owner], W4ANVFP4Linear):
                parameter.copy_(getattr(runtime_modules[owner], leaf))
    # Read the serialized INT4 layout independently, without prepared FP4 planes.
    codes, weight_scales = {}, {}
    for name, module in runtime_modules.items():
        if isinstance(module, W4ANVFP4Linear):
            shifts = (torch.arange(8, device="cuda", dtype=torch.int64) * 4)[None, :, None]
            codes[name] = (((module.qweight.long()[:, None, :] >> shifts) & 15)
                           .reshape(module.in_features, module.out_features) - 8).to(torch.int8)
            weight_scales[name] = module.scales.float().clone()
    _install_trainable_gptq_codes(replay, codes, weight_scales)
    qcfg = SimpleNamespace(activation_mode="w4a_nvfp4", activation_recipe="least_squares",
                           activation_version=4, activation_global_scales=scales,
                           dynamic_get=lambda **_kwargs: None)
    install_w4a_llama_replay(replay, qcfg)
    observed = [{}, {}]
    handles = []
    for model_index, model in enumerate((runtime, replay)):
        for index, layer in enumerate(model.model.layers):
            def capture(_module, _args, value, *, model_index=model_index, index=index):
                # Layer boundaries expose the compute-precision residual; the
                # decoded operand only exists inside the GEMM consumers.
                value = value.exact(torch.float32) if isinstance(value, W4AActivation) else value
                observed[model_index][index] = value.detach().clone()
            handles.append(layer.register_forward_hook(capture))
    try:
        ids = torch.arange(17, device="cuda")[None]
        with torch.inference_mode():
            expected = runtime(input_ids=ids, use_cache=False).logits
            actual = replay(input_ids=ids, use_cache=False).logits
        torch.cuda.synchronize()
        for index in range(2):
            torch.testing.assert_close(observed[1][index], observed[0][index], rtol=2e-3, atol=2e-3,
                                       msg=lambda message: f"decoder layer {index}: {message}")
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    finally:
        for handle in handles:
            handle.remove()


def _mixed_replay(dtype, attention_mode, mlp_fp8_layers, scales, packed):
    """Build a dense replay model under the requested mixed activation policy."""
    from types import SimpleNamespace

    from gptqmodel.nn_modules.qlinear.w4a_boundary import nvfp4_producer_specs
    from gptqmodel.nn_modules.qlinear.w4a_llama_replay import install_w4a_llama_replay

    model = LlamaForCausalLM(packed.config).to(device="cuda", dtype=dtype).eval()
    subset = {key: scales[key] for _, _, key in nvfp4_producer_specs(
        model.model.layers, [True, True], attention_mode=attention_mode or "w4a_nvfp4",
        mode="w4a_nvfp4", recipe="least_squares", mlp_fp8_layers=tuple(mlp_fp8_layers))}
    qcfg = SimpleNamespace(
        activation_mode="w4a_nvfp4", activation_recipe="least_squares", activation_version=4,
        activation_global_scales=subset, activation_attention_mode=attention_mode,
        activation_attention_recipe=None,
        activation_mlp_fp8_layers=tuple(mlp_fp8_layers) or None,
        dynamic_get=lambda **_kwargs: None,
    )
    install_w4a_llama_replay(model, qcfg)
    return model


def _capture_first_operand(model, path, *, decode):
    parent, leaf = path.rsplit(".", 1)
    module = getattr(model.model.layers[0].get_submodule(parent), leaf)
    seen = {}

    def hook(_module, args):
        value = args[0]
        seen["x"] = value.decode(torch.float32) if decode and isinstance(value, W4AActivation) else value

    handle = module.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=torch.arange(17, device="cuda")[None], use_cache=False)
    finally:
        handle.remove()
    return seen["x"]


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_v4_mixed_replay_applies_attention_and_mlp_policies(dtype):
    """Attention FP8 and per-layer MLP promotion must reach the replay operands."""
    from gptqmodel.quantization.activation_floatx import fp8_token_qdq

    packed = _tiny_packed_nvfp4(dtype=dtype)
    scales = {key: .01234567 for _, _, key in llama_nvfp4_boundaries(packed.model.layers, [True, True])}

    uniform_q = _capture_first_operand(_mixed_replay(dtype, None, (), scales, packed),
                                       "self_attn.q_proj", decode=False)
    mixed_q = _capture_first_operand(_mixed_replay(dtype, "w4afp8", (), scales, packed),
                                     "self_attn.q_proj", decode=False)
    assert not torch.equal(uniform_q, mixed_q)

    mixed = _mixed_replay(dtype, "w4afp8", (), scales, packed)
    q_fp8 = _capture_first_operand(mixed, "self_attn.q_proj", decode=False)
    source = mixed.model.embed_tokens(torch.arange(17, device="cuda")[None]).detach()
    inv_rms = torch.rsqrt(source.float().square().mean(-1, keepdim=True) + mixed.config.rms_norm_eps)
    expected = fp8_token_qdq(source.float() * inv_rms)
    torch.testing.assert_close(q_fp8, expected, rtol=1e-6, atol=1e-6)

    uniform_gate = _capture_first_operand(_mixed_replay(dtype, None, (), scales, packed),
                                          "mlp.gate_proj", decode=False)
    promoted_gate = _capture_first_operand(_mixed_replay(dtype, None, (0,), scales, packed),
                                           "mlp.gate_proj", decode=False)
    assert not torch.equal(uniform_gate, promoted_gate)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_v4_mixed_replay_matches_runtime_operands(dtype):
    """Mixed-policy replay must consume the same operands as the deployed stream."""
    from types import SimpleNamespace

    from gptqmodel.nn_modules.qlinear.w4a_boundary import nvfp4_producer_specs
    from gptqmodel.nn_modules.qlinear.w4a_llama_replay import install_w4a_llama_replay
    from tests.models.w4a_nvfp4_weight_qad import _install_trainable_gptq_codes

    packed = _tiny_packed_nvfp4(dtype=dtype)
    keys = [key for _, _, key in llama_nvfp4_boundaries(packed.model.layers, [True, True])]
    scales = dict.fromkeys(keys, .01234567)
    attention_mode, mlp_fp8_layers = "w4afp8", (0,)
    subset = {key: scales[key] for _, _, key in nvfp4_producer_specs(
        packed.model.layers, [True, True], attention_mode=attention_mode, mode="w4a_nvfp4",
        recipe="least_squares", mlp_fp8_layers=mlp_fp8_layers)}
    install_w4a_llama_stream(packed, "w4a_nvfp4", "least_squares", version=4, global_scales=subset,
                             attention_mode=attention_mode, mlp_fp8_layers=mlp_fp8_layers)

    replay = LlamaForCausalLM(packed.config).to(device="cuda", dtype=dtype).eval()
    packed_modules = dict(packed.named_modules())
    with torch.no_grad():
        for name, parameter in replay.named_parameters():
            owner, leaf = name.rsplit(".", 1)
            if not isinstance(packed_modules[owner], W4ANVFP4Linear):
                parameter.copy_(getattr(packed_modules[owner], leaf))
    codes, weight_scales = {}, {}
    for name, module in packed_modules.items():
        if isinstance(module, W4ANVFP4Linear):
            shifts = (torch.arange(8, device="cuda", dtype=torch.int64) * 4)[None, :, None]
            codes[name] = (((module.qweight.long()[:, None, :] >> shifts) & 15)
                           .reshape(module.in_features, module.out_features) - 8).to(torch.int8)
            weight_scales[name] = module.scales.float().clone()
    _install_trainable_gptq_codes(replay, codes, weight_scales)
    qcfg = SimpleNamespace(
        activation_mode="w4a_nvfp4", activation_recipe="least_squares", activation_version=4,
        activation_global_scales=subset, activation_attention_mode=attention_mode,
        activation_attention_recipe=None, activation_mlp_fp8_layers=mlp_fp8_layers,
        dynamic_get=lambda **_kwargs: None,
    )
    install_w4a_llama_replay(replay, qcfg)

    # The layer-0 attention operand depends only on the embedding and norm, so
    # it must match the deployed carrier to quantization precision.
    torch.testing.assert_close(
        _capture_first_operand(replay, "self_attn.q_proj", decode=False),
        _capture_first_operand(packed, "self_attn.q_proj", decode=True),
        rtol=1e-6, atol=1e-6, msg=lambda message: f"attention operand: {message}",
    )
    # The runtime must stage the promoted MLP boundary as an FP8 carrier. The
    # replay's own MLP operand is asserted against the uniform policy in
    # test_v4_mixed_replay_applies_attention_and_mlp_policies; here the operand
    # would depend on the divergent attention GEMM path.
    runtime_gate = _capture_first_operand(packed, "mlp.gate_proj", decode=False)
    assert isinstance(runtime_gate, W4AActivation) and runtime_gate.mode == "w4afp8"


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_v4_fp8_boundary_operands_keep_fp32(dtype):
    """o_proj/down_proj operands must not round-trip through the model dtype.

    The deployed FP8 GEMMs consume E4M3 codes plus FP32 token scales. The
    pre-fix replay cast the decoded operand back to BF16/FP16, so the attention
    output feeding ``o_proj`` and the promoted MLP product feeding ``down_proj``
    described a different function than inference.
    """
    from types import SimpleNamespace

    from gptqmodel.nn_modules.qlinear.w4a_boundary import nvfp4_producer_specs
    from gptqmodel.nn_modules.qlinear.w4a_llama_replay import install_w4a_llama_replay

    packed = _tiny_packed_nvfp4(dtype=dtype)
    keys = [key for _, _, key in llama_nvfp4_boundaries(packed.model.layers, [True, True])]
    scales = dict.fromkeys(keys, .01234567)
    attention_mode, mlp_fp8_layers = "w4afp8", (0,)
    subset = {key: scales[key] for _, _, key in nvfp4_producer_specs(
        packed.model.layers, [True, True], attention_mode=attention_mode, mode="w4a_nvfp4",
        recipe="least_squares", mlp_fp8_layers=mlp_fp8_layers)}
    model = LlamaForCausalLM(packed.config).to(device="cuda", dtype=dtype).eval()
    layer = model.model.layers[0]
    # Pristine operands: the o_proj pre-hook registered before installation
    # sees the unrounded attention output; the MLP product is rebuilt from the
    # gate/up outputs that feed the down-projection rounding.
    pristine = {"o": []}
    layer.self_attn.o_proj.register_forward_pre_hook(
        lambda _module, args: pristine["o"].append(args[0].detach().clone()))
    gate_out, up_out = [], []
    layer.mlp.gate_proj.register_forward_hook(
        lambda _module, _args, out: gate_out.append(out.detach().clone()))
    layer.mlp.up_proj.register_forward_hook(
        lambda _module, _args, out: up_out.append(out.detach().clone()))
    qcfg = SimpleNamespace(
        activation_mode="w4a_nvfp4", activation_recipe="least_squares", activation_version=4,
        activation_global_scales=subset, activation_attention_mode=attention_mode,
        activation_attention_recipe=None, activation_mlp_fp8_layers=mlp_fp8_layers,
        dynamic_get=lambda **_kwargs: None,
    )
    install_w4a_llama_replay(model, qcfg)
    rounded = {"o": [], "down": []}
    layer.self_attn.o_proj.register_forward_pre_hook(
        lambda _module, args: rounded["o"].append(args[0].detach().clone()))
    layer.mlp.down_proj.register_forward_pre_hook(
        lambda _module, args: rounded["down"].append(args[0].detach().clone()))
    ids = torch.arange(17, device="cuda")[None]
    with torch.inference_mode():
        model(input_ids=ids, use_cache=False)

    def fp8_fp32_oracle(x):
        x32 = x.float()
        amax = x32.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
        codes = (x32 / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        return codes.float() * scale

    assert len(pristine["o"]) == len(rounded["o"]) == 1
    assert len(gate_out) == len(up_out) == len(rounded["down"]) == 1
    operands = {
        "o": (pristine["o"][0], rounded["o"][0]),
        "down": (layer.mlp.act_fn(gate_out[0]) * up_out[0], rounded["down"][0]),
    }
    for key, (raw, actual) in operands.items():
        assert actual.dtype == torch.float32, key
        expected = fp8_fp32_oracle(raw)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0,
                                   msg=lambda message, key=key: f"{key}: {message}")
        assert not torch.equal(actual, expected.to(dtype).float()), key


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_token_global_packing_matches_oracle_and_is_independent_of_other_rows(dtype):
    from tests.models.w4a_token_global import pack_token_global

    generator = torch.Generator(device="cuda").manual_seed(9207)
    source = torch.randn((17, 256), device="cuda", dtype=dtype, generator=generator)
    source[0] = 0
    source[-1] *= 1000
    encoded = pack_token_global(source, model_dtype=dtype, rotation_applied=True)
    # Derive each row's outer scale independently and quantize each normalized
    # row with the separate codebook-search oracle.
    expected = []
    expected_scales = []
    for row in source:
        maximum = float(row.double().abs().max())
        scale = torch.tensor(maximum / 1792. if maximum else 1., device="cuda", dtype=torch.float32)
        expected_scales.append(scale)
        normalized = row.float()[None] / scale
        expected.append(_independent_nvfp4_qdq(normalized, torch.ones((), device="cuda"))[0] * scale)
    expected = torch.stack(expected)
    torch.cuda.synchronize()
    torch.testing.assert_close(encoded.token_scale, torch.stack(expected_scales), rtol=0, atol=0)
    torch.testing.assert_close(encoded.decode(torch.float32), expected, rtol=1e-6, atol=1e-6)
    single = pack_token_global(source[1:2], model_dtype=dtype, rotation_applied=True)
    assert torch.equal(single.codes.view(torch.uint8), encoded.codes[1:2].view(torch.uint8))
    torch.testing.assert_close(single.decode(torch.float32), encoded.decode(torch.float32)[1:2], rtol=0, atol=0)
    assert encoded.rotation_applied and encoded.model_dtype == dtype
    assert encoded.global_scale.item() == 1 and encoded.token_scale.shape == (17,)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
def test_token_global_diagnostic_keeps_encoded_handoffs_and_native_weights():
    from tests.models.w4a_token_global import install_token_global_diagnostic

    core = _tiny_stream()
    original = {name: tensor.clone() for name, tensor in core.state_dict().items()}
    handles, stats = install_token_global_diagnostic(core)
    received = []
    handles.append(core.model.layers[1].register_forward_pre_hook(lambda _m, args: received.append(args[0])))
    try:
        with torch.inference_mode():
            result = core(input_ids=torch.arange(17, device="cuda")[None], use_cache=False)
        assert torch.isfinite(result.logits).all()
        assert len(stats) == 9
        assert received and all(isinstance(value, W4AActivation) and value.token_scale is not None for value in received)
        for name, tensor in core.state_dict().items():
            torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
def test_calibrates_in_order_propagates_carriers_and_preserves_weights():
    core = _tiny_stream()
    initial = {name: value.clone() for name, value in core.state_dict().items()}
    spec = list(llama_nvfp4_boundaries(core.model.layers, [True, True]))
    ordered = [getattr(owner, f"_w4a_{name}_quantizer") for owner, name, _ in spec]
    reached = []
    second_inputs = []
    handle = core.model.layers[1].register_forward_pre_hook(lambda _m, args: second_inputs.append(args[0]))
    def progress(key, row):
        index = len(reached)
        assert key == ordered[index].key
        assert all(module.calibrated for module in ordered[:index + 1])
        assert not any(module.calibrated for module in ordered[index + 1:])
        assert row["samples"] == 2 and row["tokens"] == 40
        reached.append(key)
    samples = [torch.arange(17) % 32, torch.arange(23) % 32]
    try:
        report = calibrate_nvfp4_producers(core, samples, progress=progress)
    finally:
        handle.remove()
    assert len(report["global_scales"]) == 9
    assert second_inputs and all(isinstance(value, W4AActivation) for value in second_inputs)
    assert all(module.observer is None for module in ordered)
    for name, value in core.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    statistics = measure_nvfp4_producers(core, [samples[0]])
    assert set(statistics) == set(report["global_scales"])
    assert all(row["calls"] == 1 and row["relative_rmse"] >= 0 for row in statistics.values())
    restored = _tiny_stream(report["global_scales"])
    with torch.inference_mode():
        expected = core(input_ids=samples[0][None].cuda(), use_cache=False).logits
        actual = restored(input_ids=samples[0][None].cuda(), use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
def test_calibration_failure_rolls_back_scales_and_observers():
    core = _tiny_stream()
    def abort(key, _row):
        if key.endswith("o_proj.input"):
            raise RuntimeError("deliberate calibration failure")
    with pytest.raises(RuntimeError, match="deliberate"):
        calibrate_nvfp4_producers(core, [torch.arange(12)], progress=abort)
    assert core._w4a_stream_global_scales is None
    for module in core.modules():
        if isinstance(module, NVFP4BoundaryQuantizer):
            assert not module.calibrated and module.observer is None
            assert module.global_scale.item() == 1


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
def test_missing_producer_scales_rejected():
    with pytest.raises(ValueError, match="Incomplete"):
        _tiny_stream({"model.layers.0.input": 1.0})


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
def test_producer_calibration_supports_mixed_policy_stream():
    """Calibration must target only the boundaries the installed policy keeps NVFP4."""
    from gptqmodel.nn_modules.qlinear.w4a_boundary import nvfp4_producer_specs

    core = _tiny_packed_nvfp4()
    attention_mode, mlp_fp8_layers = "w4afp8", (0,)
    install_w4a_llama_stream(core, "w4a_nvfp4", "least_squares", version=4, global_scales=None,
                             attention_mode=attention_mode, mlp_fp8_layers=mlp_fp8_layers)
    expected = {key for _, _, key in nvfp4_producer_specs(
        core.model.layers, [True, True], attention_mode=attention_mode, mode="w4a_nvfp4",
        recipe="least_squares", mlp_fp8_layers=mlp_fp8_layers)}
    assert expected == {"model.layers.1.post_attention_residual", "model.layers.1.mlp.down_proj.input"}
    samples = [torch.arange(17) % 32, torch.arange(23) % 32]
    report = calibrate_nvfp4_producers(core, samples)
    assert set(report["global_scales"]) == expected
    assert all(value > 0 for value in report["global_scales"].values())
    assert not any(".self_attn." in key for key in report["global_scales"])
    statistics = measure_nvfp4_producers(core, [samples[0]])
    assert set(statistics) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,width", [(1, 128), (33, 2048), (3, 8192)])
def test_experimental_token_energy_matches_independent_nvfp4_oracle(dtype, rows, width):
    from tests.models.w4a_token_energy import install_token_energy_diagnostic

    generator = torch.Generator(device="cuda").manual_seed(926 + rows)
    source = torch.randn((rows, width), generator=generator, device="cuda", dtype=dtype)
    if rows > 1:
        source[0] = 0
    quantizer = NVFP4BoundaryQuantizer("model.layers.0.input", "cuda", 1 / 32)
    with torch.inference_mode():
        uncorrected = quantizer(source, "w4a_nvfp4", recipe="least_squares")
        handles, _ = install_token_energy_diagnostic(quantizer, "all")
        try:
            corrected = quantizer(source, "w4a_nvfp4", recipe="least_squares")
        finally:
            for handle in handles:
                handle.remove()
        expected = _independent_nvfp4_qdq(source, torch.tensor(1 / 32, device="cuda")).double()
        source_energy = source.double().square().sum(-1)
        decoded_energy = expected.square().sum(-1)
        gain = torch.ones(rows, device="cuda", dtype=torch.float64)
        nonzero = decoded_energy > 0
        gain[nonzero] = (source_energy[nonzero] / decoded_energy[nonzero]).sqrt()
        expected *= gain[:, None]
        torch.cuda.synchronize()
        assert torch.equal(corrected.codes.view(torch.uint8), uncorrected.codes.view(torch.uint8))
        assert torch.equal(corrected.scales.view(torch.uint8), uncorrected.scales.view(torch.uint8))
        torch.testing.assert_close(corrected.token_scale.double(), gain, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(corrected.decode(torch.float32).double(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
def test_token_energy_experiment_preserves_native_state_and_encoded_handoffs():
    from tests.models.w4a_token_energy import install_token_energy_diagnostic

    core = _tiny_stream()
    native = {name: value.clone() for name, value in core.state_dict().items()}
    handles, stats = install_token_energy_diagnostic(core, "all")
    inputs = []
    handles.append(core.model.layers[1].register_forward_pre_hook(lambda _m, args: inputs.append(args[0])))
    try:
        with torch.inference_mode():
            result = core(input_ids=torch.arange(19, device="cuda")[None], use_cache=False)
        assert torch.isfinite(result.logits).all()
        assert len(stats) == 9
        assert inputs and all(isinstance(value, W4AActivation) for value in inputs)
        for name, value in core.state_dict().items():
            torch.testing.assert_close(value, native[name], rtol=0, atol=0)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
@pytest.mark.parametrize("rows", [1, 33])
def test_energy_corrected_carrier_gemm_matches_independent_native_int4_oracle(monkeypatch, rows):
    from gptqmodel.nn_modules.qlinear import w4a_activation
    from tests.models.w4a_token_energy import install_token_energy_diagnostic

    width, columns = 256, 128
    linear = W4ANVFP4Linear(bits=4, group_size=128, sym=True, desc_act=False,
                            in_features=width, out_features=columns, bias=True).cuda()
    raw = (torch.arange(width, device="cuda")[:, None] * 3 + torch.arange(columns, device="cuda")) % 16
    shifts = (torch.arange(8, device="cuda", dtype=torch.int32) * 4)[None, :, None]
    linear.qweight.copy_((raw.reshape(width // 8, 8, columns).int() << shifts).sum(1))
    linear.qzeros.fill_(0x77777777)
    linear.scales[0].fill_(.125)
    linear.scales[1].fill_(.03125)
    linear.bias.copy_(torch.linspace(-.25, .25, columns, device="cuda"))
    native = {name: value.clone() for name, value in linear.state_dict().items()}
    linear.post_init()
    generator = torch.Generator(device="cuda").manual_seed(408 + rows)
    source = torch.randn((rows, width), generator=generator, device="cuda", dtype=torch.bfloat16)
    quantizer = NVFP4BoundaryQuantizer("model.layers.0.input", "cuda", 1 / 32)
    handles, _ = install_token_energy_diagnostic(quantizer, "all")
    try:
        with torch.inference_mode():
            encoded = quantizer(source, "w4a_nvfp4", recipe="least_squares")
    finally:
        for handle in handles:
            handle.remove()
    # Observe the raw GEMM result before the next producer rounds it. The
    # input carrier and the native scaled_mm operands remain real NVFP4.
    observed = []
    def capture_output(value, mode, **kwargs):
        observed.append((mode, value.dtype))
        return value
    monkeypatch.setattr(w4a_activation, "pack_activation", capture_output)
    with torch.inference_mode():
        actual = linear(encoded)
    assert observed == [("w4a_nvfp4", torch.float32)]
    # Independent activation codebook search, FP64 norm fitting and native
    # INT4 unpacking; no prepared weight planes or runtime decoder are used.
    quantized = _independent_nvfp4_qdq(source, torch.tensor(1 / 32, device="cuda")).double()
    gain = source.double().norm(dim=-1) / quantized.norm(dim=-1)
    packed = linear.qweight.long()
    unpacked = ((packed[:, None, :] >> shifts.long()) & 15).reshape(width, columns).double() - 8
    reference = torch.zeros((rows, columns), device="cuda", dtype=torch.float64)
    for group in range(2):
        section = slice(group * 128, (group + 1) * 128)
        reference += (quantized[:, section] @ unpacked[section]) * linear.scales[group].double()
    reference = reference * gain[:, None] + linear.bias.double()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.double(), reference, rtol=2e-3, atol=2e-3)
    for name, value in linear.state_dict().items():
        torch.testing.assert_close(value, native[name], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
@pytest.mark.parametrize("abort", [False, True])
def test_producer_reconstruction_propagates_carriers_preserves_weights_and_rolls_back(abort):
    from tests.models.w4a_producer_reconstruct import reconstruct

    core = _tiny_stream()
    fit = [torch.arange(8), torch.arange(11)]
    calibrate_nvfp4_producers(core, fit)
    teacher = LlamaForCausalLM(core.config).to(device="cuda", dtype=torch.bfloat16).eval()

    class NativeLinear(torch.nn.Linear):
        def forward(self, x):
            return torch.nn.functional.linear(x.float(), self.weight, None).to(x.dtype)

    # Same native integer codes and stored group scales, independently unpacked
    # into an FP32 linear teacher. This is a tiny lifecycle fixture, not an
    # alternative implementation of the real same-checkpoint A16 quality gate.
    teacher.model.embed_tokens.weight.data.copy_(core.model.embed_tokens.weight)
    teacher.lm_head.weight.data.copy_(core.lm_head.weight)
    teacher.model.norm.weight.data.copy_(core.model.norm.weight)
    teacher_modules = dict(teacher.named_modules())
    for name, module in core.named_modules():
        if isinstance(module, W4ANVFP4Linear):
            k, n = module.in_features, module.out_features
            shifts = (torch.arange(8, device="cuda") * 4)[None, :, None]
            values = ((module.qweight.long()[:, None, :] >> shifts) & 15).reshape(k, n).float() - 8
            weight = values * module.scales.float().repeat_interleave(128, dim=0)
            dense = NativeLinear(k, n, bias=False, device="cuda", dtype=torch.float32)
            dense.weight.data.copy_(weight.T)
            parent, child = name.rsplit(".", 1)
            setattr(teacher_modules[parent], child, dense)
    native = {name: value.clone() for name, value in core.state_dict().items()}
    old_policy = core._w4a_stream_global_scales
    buffers = {name: module.scale_bits.clone() for name, module in core.named_modules()
               if isinstance(module, NVFP4BoundaryQuantizer)}
    reached = []
    def progress(key, _row):
        reached.append(key)
        if abort:
            raise RuntimeError("deliberate reconstruction failure")
    inputs = []
    handle = core.model.layers[1].register_forward_pre_hook(lambda _m, args: inputs.append(args[0]))
    try:
        if abort:
            with pytest.raises(RuntimeError, match="deliberate"):
                reconstruct(core, teacher, fit, ratios=(.875, 1., 1.25), progress=progress)
            assert core._w4a_stream_global_scales is old_policy
            for name, module in core.named_modules():
                if name in buffers:
                    assert torch.equal(module.scale_bits, buffers[name])
        else:
            report = reconstruct(core, teacher, fit, ratios=(.875, 1., 1.25), progress=progress)
            assert len(reached) == len(report["global_scales"]) == 9
            assert all(row["selected_loss"] <= row["initial_loss"] for row in report["layers"])
            assert report["weight_updates"] == 0
            assert inputs and all(isinstance(value, W4AActivation) for value in inputs)
            assert core._w4a_stream_global_scales == report["global_scales"]
        for name, value in core.state_dict().items():
            torch.testing.assert_close(value, native[name], rtol=0, atol=0)
    finally:
        handle.remove()
