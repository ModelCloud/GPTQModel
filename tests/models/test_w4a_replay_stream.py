# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Check GPTQ capture sees rounded inputs without a second input rounding."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.nn_modules.hooked_linear import HookedLinear
from gptqmodel.nn_modules.qlinear.w4a_llama_replay import install_w4a_llama_replay
from gptqmodel.quantization.activation_floatx import nvfp4_block_qdq
from gptqmodel.quantization.config import QuantizeConfig


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hooked", [False, True])
def test_replay_preserves_fp32_norm_operand_and_model_dtype_exits(dtype, hooked):
    from gptqmodel.nn_modules.qlinear.w4a_boundary import llama_nvfp4_boundaries
    from tests.kernels.test_w4a_stream import _independent_nvfp4_qdq

    torch.manual_seed(9721)
    config = LlamaConfig(vocab_size=32, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    model = LlamaForCausalLM(config).to(dtype).eval()
    scales = {key: .01234567 for _, _, key in llama_nvfp4_boundaries(model.model.layers, [True, True])}
    qcfg = SimpleNamespace(activation_mode="w4a_nvfp4", activation_recipe="least_squares",
                           rotation="hadamard",
                           dynamic_get=lambda **_kwargs: None,
                           activation_global_scales=scales)
    ids = torch.arange(17)[None]
    source = model.model.embed_tokens(ids).detach()
    decoded = _independent_nvfp4_qdq(source, torch.tensor(.01234567))
    # The deployed stream rescales the rounded carrier by the inverse RMS of
    # the pristine residual, so the denominator stays unquantized.
    expected_norm = decoded * torch.rsqrt(source.float().square().mean(-1, keepdim=True)
                                          + config.rms_norm_eps)
    install_w4a_llama_replay(model, qcfg)
    if hooked:
        for layer in model.model.layers:
            for parent in (layer.self_attn, layer.mlp):
                for name, child in list(parent.named_children()):
                    if isinstance(child, torch.nn.Linear):
                        setattr(parent, name, HookedLinear.from_linear(child))
    captured = {}
    model.model.layers[0].self_attn.q_proj.register_forward_pre_hook(
        lambda _m, args: captured.update(norm=args[0].detach().clone()))
    model.model.layers[0].self_attn.q_proj.register_forward_hook(
        lambda _m, _args, value: captured.update(projection_dtype=value.dtype))
    model.model.layers[1].register_forward_pre_hook(
        lambda _m, args: captured.update(layer_input_dtype=args[0].dtype))
    model.model.norm.register_forward_pre_hook(
        lambda _m, args: captured.update(exit_dtype=args[0].dtype))
    with torch.no_grad():
        output = model(input_ids=ids, use_cache=False)
    assert captured["norm"].dtype == torch.float32
    torch.testing.assert_close(captured["norm"], expected_norm, rtol=1e-6, atol=1e-6)
    assert captured["projection_dtype"] == captured["exit_dtype"] == dtype
    assert captured["layer_input_dtype"] == torch.float32
    assert torch.isfinite(output.logits).all()

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_disabled_replay_preserves_native_model_precision(dtype):
    import copy

    from gptqmodel.nn_modules.qlinear.w4a_llama_replay import set_w4a_replay_enabled

    torch.manual_seed(9732)
    config = LlamaConfig(vocab_size=32, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    model = LlamaForCausalLM(config).to(dtype).eval()
    native = copy.deepcopy(model)
    qcfg = SimpleNamespace(activation_mode="w4a_nvfp4", activation_recipe="least_squares",
                           dynamic_get=lambda **_kwargs: None,
                           activation_global_scales=None)
    install_w4a_llama_replay(model, qcfg)
    set_w4a_replay_enabled(model, False)
    with torch.no_grad():
        actual = model(input_ids=torch.arange(11)[None], use_cache=False).logits
        expected = native(input_ids=torch.arange(11)[None], use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp8_replay_operand_stays_fp32(dtype):
    """An FP8 replay operand must skip the model-dtype round-trip.

    The deployed FP8 GEMM consumes E4M3 codes plus FP32 token scales, so
    casting the decoded value back to BF16/FP16 before the GEMM (the pre-fix
    replay behavior) teaches a different function than inference.
    """
    from gptqmodel.nn_modules.qlinear.w4a_llama_replay import round_w4a_replay_operand

    torch.manual_seed(17)
    x = (torch.randn(17, 128) * 3).to(dtype)
    actual = round_w4a_replay_operand(x, "w4afp8", None, None)
    assert actual.dtype == torch.float32
    # Independent oracle: per-token FP8 QDQ in FP32 with no cast back.
    x32 = x.float()
    amax = x32.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    codes = (x32 / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    expected = codes.float() * scale
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The model-dtype round-trip the fix removed is observably different.
    assert not torch.equal(actual, expected.to(dtype).float())

@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 / SM121 required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_runtime_install_removes_the_replay_exit_hook(dtype):
    """Inference must work on the freshly quantized object before save/reload."""
    from gptqmodel.nn_modules.qlinear.w4a_llama_stream import install_w4a_llama_stream
    from gptqmodel.nn_modules.qlinear.w4a_nvfp4 import W4ANVFP4Linear
    from tests.models.test_w4a_producer_calibration import _tiny_packed_nvfp4

    packed = _tiny_packed_nvfp4(dtype=dtype)
    model = LlamaForCausalLM(packed.config).to(device="cuda", dtype=dtype).eval()
    qcfg = SimpleNamespace(
        activation_mode="w4a_nvfp4", activation_recipe="least_squares",
        activation_global_scales=None, activation_attention_mode=None,
        activation_attention_recipe=None, activation_mlp_fp8_layers=None,
        dynamic_get=lambda **_kwargs: None,
    )
    install_w4a_llama_replay(model, qcfg)
    # The replay dtype-exit hook is installed on the final norm.
    assert model.model.norm._forward_pre_hooks

    # The looper replaces dense projections with packed modules in place; the
    # replay hook on the final norm is not touched by that replacement.
    for name, module in dict(packed.named_modules()).items():
        if isinstance(module, W4ANVFP4Linear):
            parent_name, leaf = name.rsplit(".", 1)
            setattr(model.get_submodule(parent_name), leaf, module)

    install_w4a_llama_stream(model, "w4a_nvfp4", "least_squares", global_scales=None)
    assert not model.model.norm._forward_pre_hooks
    ids = torch.arange(17, device="cuda")[None]
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits
    assert torch.isfinite(logits).all()

def test_nvfp4_replay_decodes_at_an_unselected_layer():
    config = LlamaConfig(vocab_size=32, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    model = LlamaForCausalLM(config).bfloat16().eval()
    qcfg = SimpleNamespace(activation_mode="w4a_nvfp4", activation_recipe="least_squares",

                           dynamic_get=lambda layer_name: False if layer_name.startswith("model.layers.1.") else None,
                           activation_global_scales=None)
    install_w4a_llama_replay(model, qcfg)
    captured = []
    model.model.layers[1].register_forward_pre_hook(lambda _m, args: captured.append(args[0].dtype))
    with torch.no_grad():
        model(input_ids=torch.arange(7)[None], use_cache=False)
    assert captured == [torch.bfloat16]

@pytest.mark.parametrize("mode", ["w4afp8", "w4a_nvfp4"])
def test_replay_decodes_at_an_unselected_layer(mode):
    """The compute-precision residual is cast back to the model dtype at an edge."""
    config = LlamaConfig(vocab_size=32, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    model = LlamaForCausalLM(config).bfloat16().eval()
    qcfg = SimpleNamespace(
        activation_mode=mode,
        activation_recipe="least_squares" if mode == "w4a_nvfp4" else None,
        dynamic_get=lambda layer_name: False if layer_name.startswith("model.layers.1.") else None,
        activation_global_scales=None,
    )
    install_w4a_llama_replay(model, qcfg)
    captured = []
    model.model.layers[1].register_forward_pre_hook(lambda _m, args: captured.append(args[0].dtype))
    with torch.no_grad():
        model(input_ids=torch.arange(7)[None], use_cache=False)
    assert captured == [torch.bfloat16]

def test_install_preserves_existing_hooked_linear_capture():
    config = LlamaConfig(vocab_size=32, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    model = LlamaForCausalLM(config).bfloat16().eval()
    linear = HookedLinear.from_linear(model.model.layers[0].self_attn.q_proj)
    model.model.layers[0].self_attn.q_proj = linear
    captured = []
    linear.forward_hook = lambda _m, args, result: captured.append((args[0].dtype, result.dtype))
    qcfg = SimpleNamespace(activation_mode="w4a_nvfp4", activation_recipe="least_squares",
                           dynamic_get=lambda **_kwargs: None,
                           activation_global_scales=None)
    install_w4a_llama_replay(model, qcfg)
    with torch.no_grad():
        model(input_ids=torch.arange(7)[None], use_cache=False)
    assert captured == [(torch.float32, torch.bfloat16)]

def test_existing_hooked_linears_round_each_producer_once(monkeypatch):
    from gptqmodel.nn_modules.qlinear import w4a_llama_replay as replay

    config = LlamaConfig(vocab_size=32, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    model = LlamaForCausalLM(config).bfloat16().eval()
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, torch.nn.Linear):
                    setattr(parent, name, HookedLinear.from_linear(child))
    calls = []
    def rounded(value, *args):
        calls.append(value.shape)
        return value
    monkeypatch.setattr(replay, "_round", rounded)
    monkeypatch.setattr(replay, "round_w4a_activation", rounded)
    qcfg = SimpleNamespace(activation_mode="w4a_nvfp4", activation_recipe="least_squares",
                           rotation="hadamard",
                           dynamic_get=lambda **_kwargs: None,
                           activation_global_scales=None)
    replay.install_w4a_llama_replay(model, qcfg)
    with torch.no_grad():
        model(input_ids=torch.arange(7)[None], use_cache=False)
    # Four producers per layer: the stream entry (or rebuilt predecessor
    # output), the attention output, the post-attention residual, and the MLP
    # product. The terminal layer output feeds only the pristine final norm.
    assert len(calls) == 8

def test_replay_preserves_custom_int4_training_forward_and_gradients(monkeypatch):
    from types import MethodType

    from gptqmodel.nn_modules.qlinear import w4a_llama_replay as replay

    config = LlamaConfig(vocab_size=32, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    model = LlamaForCausalLM(config).bfloat16().eval()
    linear = model.model.layers[0].self_attn.q_proj
    latent = torch.nn.Parameter(linear.weight.detach().float().clone())
    linear.register_parameter("weight", None)
    linear.register_parameter("latent", latent)
    def custom(self, x):
        return F.linear(x.float(), self.latent)
    linear.forward = MethodType(custom, linear)
    qcfg = SimpleNamespace(activation_mode="w4a_nvfp4", activation_recipe="least_squares",
                           dynamic_get=lambda **_kwargs: None,
                           activation_global_scales=None)
    replay.install_w4a_llama_replay(model, qcfg)
    def straight_through(x, mode, recipe=None, scale=None):
        value = replay.round_w4a_activation(x.detach(), mode, recipe, scale)
        return x + (value - x).detach()
    monkeypatch.setattr(replay, "_round", straight_through)
    model(input_ids=torch.arange(7)[None], use_cache=False).logits.float().square().mean().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert latent.grad.abs().max() > 0

def test_replay_requires_complete_producer_scale_coverage():
    config = LlamaConfig(vocab_size=32, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    model = LlamaForCausalLM(config)
    qcfg = SimpleNamespace(activation_mode="w4a_nvfp4", activation_recipe="least_squares",
                           dynamic_get=lambda **_kwargs: None,
                           activation_global_scales={"model.layers.0.input": .01})
    with pytest.raises(ValueError, match="Incomplete NVFP4 producer scales"):
        install_w4a_llama_replay(model, qcfg)
    assert not hasattr(model.model.layers[0], "_w4a_replay_mode")

def test_replay_uses_frozen_producer_scale_at_each_boundary(monkeypatch):
    from gptqmodel.nn_modules.qlinear import w4a_llama_replay as replay
    from gptqmodel.nn_modules.qlinear.w4a_boundary import llama_nvfp4_boundaries

    config = LlamaConfig(vocab_size=32, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    model = LlamaForCausalLM(config).eval()
    specs = list(llama_nvfp4_boundaries(model.model.layers, [True, True]))
    scales = {key: .001 * (index + 1) for index, (_, _, key) in enumerate(specs)}
    qcfg = SimpleNamespace(activation_mode="w4a_nvfp4", activation_recipe="least_squares",
                           rotation="hadamard",
                           dynamic_get=lambda **_kwargs: None,
                           activation_global_scales=scales)
    calls = []
    def rounded(value, _mode, _recipe=None, global_scale=None):
        calls.append(global_scale)
        return value
    monkeypatch.setattr(replay, "_round", rounded)
    install_w4a_llama_replay(model, qcfg)
    # A down projection without Hadamard still receives the already rounded
    # MLP product; it must not be quantized again by its pre-hook.
    with torch.no_grad():
        model(input_ids=torch.arange(8)[None], use_cache=False)
    # Interior layers reuse the predecessor's frozen output scale when they
    # rebuild their input operand, and the terminal output boundary never
    # feeds a GEMM, so the last producer scale is intentionally unused.
    assert calls == list(scales.values())[:-1]

def _independent_round(x: torch.Tensor, mode: str) -> torch.Tensor:
    blocks = x.float()
    if mode == "w4afp8":
        maxima = blocks.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(maxima > 0, maxima / 448.0, torch.ones_like(maxima))
        return ((blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)
    global_max = blocks.abs().amax()
    global_scale = torch.where(global_max > 0, global_max / 1792.0, torch.ones_like(global_max))
    global_scale = global_scale.to(x.dtype).float()
    values = blocks.reshape(*x.shape[:-1], x.shape[-1] // 16, 16)
    maxima = values.abs().amax(dim=-1)
    codebook = torch.tensor((0., -1., 1., -2., 2., -4., 4., -.5, .5, -1.5, 1.5, -3., 3., -6., 6.),
                            device=x.device)
    best_error = torch.full_like(maxima, torch.inf)
    best_reconstructed = torch.zeros_like(values)
    best_values = torch.zeros_like(values)
    seeds = []

    def evaluate(local):
        scale = local[..., None] * global_scale
        nearest = (values[..., None] / scale[..., None] - codebook).abs().argmin(dim=-1)
        quantized = codebook[nearest]
        reconstructed = quantized * scale
        return quantized, reconstructed, (reconstructed - values).square().sum(dim=-1)

    def refine(quantized):
        denominator = quantized.square().sum(dim=-1)
        optimal = (values * quantized).sum(dim=-1) / denominator.clamp_min(1.0)
        inverse = optimal / global_scale
        return torch.where(
            denominator > 0,
            inverse.clamp(min=2.0**-9, max=448.0),
            torch.ones_like(inverse),
        ).to(torch.float8_e4m3fn).float()

    for bound in (4.0, 6.0):
        local = torch.where(
            maxima > 0,
            (maxima / (bound * global_scale)).clamp(min=2.0**-9, max=448.0),
            torch.ones_like(maxima),
        ).to(torch.float8_e4m3fn).float()
        quantized, reconstructed, error = evaluate(local)
        seeds.append(quantized)
        use = error < best_error
        best_error = torch.where(use, error, best_error)
        best_reconstructed = torch.where(use[..., None], reconstructed, best_reconstructed)
        best_values = torch.where(use[..., None], quantized, best_values)
    for quantized in seeds:
        quantized, reconstructed, error = evaluate(refine(quantized))
        use = error < best_error
        best_error = torch.where(use, error, best_error)
        best_reconstructed = torch.where(use[..., None], reconstructed, best_reconstructed)
        best_values = torch.where(use[..., None], quantized, best_values)
    _quantized, reconstructed, error = evaluate(refine(best_values))
    result = torch.where((error < best_error)[..., None], reconstructed, best_reconstructed)
    return result.reshape_as(x).to(x.dtype)

@pytest.mark.parametrize("mode", ["w4afp8", "w4a_nvfp4"])
@pytest.mark.parametrize("skip_first", [False, True])
def test_replay_hessian_input_matches_encoded_stream(mode, skip_first):
    torch.manual_seed(703)
    config = LlamaConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        activation={
            "mode": mode,
            **({"recipe": "least_squares"} if mode == "w4a_nvfp4" else {}),
        },
        offload_to_disk=False,
        dynamic={"-:^model.layers\\.0\\.": {}} if skip_first else None,
    )
    layer = model.model.layers[1 if skip_first else 0]
    target = layer.self_attn.q_proj
    raw = []
    rounded = []
    raw_output = []
    rounded_output = []
    layer_input = []
    layer.register_forward_pre_hook(lambda _module, args: layer_input.append(args[0].clone()))
    target.register_forward_pre_hook(lambda _module, args: raw.append(args[0].clone()))
    target.register_forward_hook(lambda _module, _args, out: raw_output.append(out.clone()))
    install_w4a_llama_replay(model, qcfg)
    target.register_forward_pre_hook(lambda _module, args: rounded.append(args[0].clone()))
    target.register_forward_hook(lambda _module, _args, out: rounded_output.append(out.clone()))
    ids = torch.tensor([[1, 7, 3]], dtype=torch.long)
    with torch.inference_mode():
        result = model(input_ids=ids, use_cache=False)
    assert torch.isfinite(result.logits).all()
    assert len(raw) == len(rounded) == 1
    assert torch.equal(raw[0], rounded[0])
    # The replay rounds the pristine FP32 norm operand, matching runtime's
    # ``_norm_forward``. The independent NVFP4 oracle picks a neighbouring
    # hardware scale on exact FP4 ties, so it can differ from the production
    # kernel by one E4M3 ulp; FP8 remains bit-exact.
    residual = layer_input[0].float()
    variance = residual.square().mean(dim=-1, keepdim=True)
    pristine = residual * torch.rsqrt(variance + config.rms_norm_eps)
    pristine = pristine * layer.input_layernorm.weight.float()
    operand_atol = 0 if mode == "w4afp8" else 1e-6
    torch.testing.assert_close(rounded[0], _independent_round(pristine, mode),
                               rtol=0, atol=operand_atol)
    assert len(raw_output) == len(rounded_output) == 1
    # Only GEMM operands are encoded; a projection output stays in the model
    # dtype until the next actual carrier consumer.
    torch.testing.assert_close(rounded_output[0], raw_output[0], rtol=0, atol=0)
    assert getattr(model.model.layers[0], "_w4a_replay_round_input", None) is (None if skip_first else True)
    assert getattr(model.model.layers[1], "_w4a_replay_round_input", None) is (True if skip_first else False)

@pytest.mark.parametrize("mode", ["w4afp8", "w4a_nvfp4"])
def test_replay_preserves_pristine_residual_with_zero_branches(mode):
    """Runtime adds residuals in compute precision; older replay must too.

    The deployed ``_add_stream`` decodes each carrier to FP32 before summing,
    so a decoder layer whose attention and MLP branches are exactly zero must
    return its input unchanged.
    """
    torch.manual_seed(705)
    config = LlamaConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        activation={
            "mode": mode,
            **({"recipe": "least_squares"} if mode == "w4a_nvfp4" else {}),
        },
        offload_to_disk=False,
    )
    layer = model.model.layers[0]
    install_w4a_llama_replay(model, qcfg)
    # Zero both branches after installation so the layer output is the residual.
    layer.self_attn.forward = lambda hidden_states, **_kwargs: (
        torch.zeros_like(hidden_states), None,
    )
    layer.mlp.forward = torch.zeros_like
    captured = []
    layer.register_forward_hook(lambda _module, _args, out: captured.append(out))
    ids = torch.tensor([[1, 7, 3]], dtype=torch.long)
    source = model.model.embed_tokens(ids).detach()
    with torch.inference_mode():
        model(input_ids=ids, use_cache=False)
    assert len(captured) == 1
    assert captured[0].dtype == torch.float32
    torch.testing.assert_close(captured[0], source.float(), rtol=0, atol=0)

@pytest.mark.parametrize("mode", ["w4afp8", "w4a_nvfp4"])
def test_replay_norm_consumes_pristine_residual(mode):
    """The RMSNorm operand must be built from the unrounded residual value."""
    torch.manual_seed(706)
    config = LlamaConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        activation={
            "mode": mode,
            **({"recipe": "least_squares"} if mode == "w4a_nvfp4" else {}),
        },
        offload_to_disk=False,
    )
    layer = model.model.layers[0]
    captured = []
    layer.self_attn.q_proj.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].clone()))
    install_w4a_llama_replay(model, qcfg)
    ids = torch.tensor([[1, 7, 3]], dtype=torch.long)
    source = model.model.embed_tokens(ids).detach()
    with torch.inference_mode():
        model(input_ids=ids, use_cache=False)
    assert len(captured) == 1
    # The capture hook is registered before the replay pre-hook, so it sees the
    # normed operand the GEMM consumes. It must equal the rounded norm of the
    # FP32 pristine residual, not the norm of a rounded carrier.
    pristine = layer.input_layernorm(source.float())
    expected = _independent_round(pristine, mode)
    atol = 1e-6 if mode == "w4a_nvfp4" else 0
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=atol)
    assert captured[0].dtype == torch.float32

@pytest.mark.parametrize("mode", ["w4afp8", "w4a_nvfp4"])
@pytest.mark.parametrize("rotated", [False, True])
def test_hooked_linear_preserves_w4a_replay_policy(monkeypatch, mode, rotated):
    """The GPTQ stage replacement must retain the replay policy."""
    torch.manual_seed(704)
    dense = torch.nn.Linear(128, 128, bias=True, dtype=torch.float32).eval()
    dense._w4a_stream_replay_mode = mode
    dense._w4a_stream_replay_recipe = "least_squares" if mode == "w4a_nvfp4" else None
    dense._w4a_stream_replay_pre_hook = True
    dense.online_full_had = rotated
    dense.online_partial_had = False
    dense.had_dim = -1
    dense.had_K = None
    dense.K = 1
    hooked = HookedLinear.from_linear(dense).eval()

    if rotated:
        # A deterministic orthogonal permutation keeps this check independent
        # of the optional CUDA Hadamard extension.
        monkeypatch.setattr(
            "gptqmodel.nn_modules.hooked_linear.apply_online_hadamard",
            lambda value, **_kwargs: value.flip(-1),
        )

    seen = []
    hooked.forward_hook = lambda _module, args, _output: seen.append(args[0].clone())
    x = torch.randn(2, 3, 128, dtype=torch.float32)
    first = _independent_round(x, mode)
    expected_input = _independent_round(first.flip(-1), mode) if rotated else first
    expected_output = F.linear(expected_input, dense.weight, dense.bias)

    with torch.inference_mode():
        actual = hooked(x)

    assert hooked._w4a_stream_replay_mode == mode
    assert hooked._w4a_stream_replay_pre_hook is True
    assert len(seen) == 1
    atol = 6e-7 if mode == "w4a_nvfp4" else 0
    torch.testing.assert_close(seen[0], expected_input, rtol=0, atol=atol)
    torch.testing.assert_close(actual, expected_output, rtol=0, atol=atol)

@pytest.mark.parametrize("mode", ["w4afp8", "w4a_nvfp4"])
def test_hooked_linear_consumes_pre_rotated_replay_operand_once(monkeypatch, mode):
    dense = torch.nn.Linear(128, 128, bias=False, dtype=torch.float32).eval()
    dense._w4a_stream_replay_mode = mode
    dense._w4a_stream_replay_recipe = "least_squares" if mode == "w4a_nvfp4" else None
    dense._w4a_stream_replay_pre_hook = True
    dense._w4a_rotation_preapplied = True
    dense.online_full_had = True
    dense.online_partial_had = False
    dense.had_dim = -1
    dense.had_K = None
    dense.K = 1
    hooked = HookedLinear.from_linear(dense).eval()
    monkeypatch.setattr(
        "gptqmodel.nn_modules.hooked_linear.apply_online_hadamard",
        lambda _value, **_kwargs: (_ for _ in ()).throw(AssertionError("rotation repeated")),
    )
    seen = []
    hooked.forward_hook = lambda _module, args, _output: seen.append(args[0].clone())
    operand = _independent_round(torch.randn(2, 3, 128), mode)
    expected = F.linear(operand, dense.weight)

    with torch.inference_mode():
        actual = hooked(operand)

    assert hooked._w4a_rotation_preapplied is True
    torch.testing.assert_close(seen[0], operand, rtol=0, atol=0)
    atol = 3e-7 if mode == "w4a_nvfp4" else 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=atol)

@pytest.mark.parametrize("mode", ["w4afp8", "w4a_nvfp4"])
def test_installed_replay_rounds_pre_rotated_down_operand_once(monkeypatch, mode):
    """The MLP wrapper and generic Linear pre-hook must not both QDQ down input."""
    config = LlamaConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    down = model.model.layers[0].mlp.down_proj
    down.online_full_had = True
    down.online_partial_had = False
    down.had_dim = -1
    down.had_K = None
    down.K = 1
    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        activation={
            "mode": mode,
            **({"recipe": "least_squares"} if mode == "w4a_nvfp4" else {}),
        },
        offload_to_disk=False,
    )
    calls = []

    def count_round(value, *_args, **_kwargs):
        calls.append(tuple(value.shape))
        return value

    monkeypatch.setattr(
        "gptqmodel.nn_modules.qlinear.w4a_llama_replay._round", count_round
    )
    monkeypatch.setattr(
        "gptqmodel.quantization.rotation.hadamard_utils.apply_online_hadamard",
        lambda value, **_kwargs: value,
    )
    install_w4a_llama_replay(model, qcfg)
    with torch.inference_mode():
        model.model.layers[0].mlp(torch.randn(2, 3, 128))

    # The layer forward rounds the normed gate/up operand once; calling the MLP
    # directly bypasses that, so only the MLP product is rounded here. The down
    # pre-hook must not round the already-encoded product a second time.
    assert len(calls) == 1
    assert calls[0][-1] == 256

def test_nvidia_headroom_probe_freezes_scale_before_hessian_capture():
    processor = GPTQProcessor.__new__(GPTQProcessor)
    processor.qcfg = SimpleNamespace(
        activation_mode="w4a_nvfp4", activation_recipe="nvidia_headroom"
    )
    processor._activation_amax = {}
    processor._activation_headroom = {}
    processor._activation_global_scales = {}
    processor._activation_headroom_probe = False

    dense = HookedLinear(128, 128)
    dense.weight = torch.nn.Parameter(torch.eye(128))
    dense.bias = None
    named = NamedModule(
        dense, "self_attn.q_proj", "model.layers.0.self_attn.q_proj", 0
    )
    subset = {"self_attn.q_proj": named}

    assert processor.begin_activation_scale_probe(subset)
    source = torch.full((8, 128), 0.5)
    processor._record_activation_amax("self_attn.q_proj", source, dense)
    processor.end_activation_scale_probe(subset)

    scale = processor._activation_global_scales[named.full_name]
    assert scale == pytest.approx(8192.0 / (6.0 * 448.0), rel=0.06)
    assert dense._w4a_activation_global_scale == scale
    assert dense._w4a_headroom_probe is False

def _headroom_probe_processor(activation):
    import threading

    from gptqmodel.quantization.config import QuantizeConfig

    processor = GPTQProcessor.__new__(GPTQProcessor)
    processor.qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False, activation=activation,
    )
    processor._activation_amax = {}
    processor._activation_headroom = {}
    processor._activation_global_scales = {}
    processor._activation_headroom_probe = False
    processor.lock = threading.Lock()
    return processor

def _projection(leaf):
    dense = HookedLinear(128, 128)
    dense.weight = torch.nn.Parameter(torch.eye(128))
    dense.bias = None
    name = leaf
    full_name = f"model.layers.0.{leaf}"
    named = NamedModule(dense, name, full_name, 0)
    return dense, named, {name: named}

@pytest.mark.parametrize("top_recipe,attention_recipe,headroom", [
    ("nvidia_headroom", "nvidia_headroom", True),
    ("least_squares", "nvidia_headroom", True),
    ("nvidia_headroom", "least_squares", False),
])
def test_headroom_probe_uses_attention_recipe_override(top_recipe, attention_recipe, headroom):
    """The attention override must drive Q-projection probe selection/freezing."""
    nvfp4 = "w4a_nvfp" + "4"
    processor = _headroom_probe_processor({
        "mode": nvfp4, "recipe": top_recipe,
        "attention": {"mode": nvfp4, "recipe": attention_recipe},
    })
    attn_leaf = "self" + "_attn.q_proj"
    dense, named, subset = _projection(attn_leaf)
    assert processor.begin_activation_scale_probe(subset) is headroom
    processor._record_activation_amax(attn_leaf, torch.full((8, 128), 0.5), dense)
    processor.end_activation_scale_probe(subset)
    if headroom:
        assert processor._activation_global_scales[named.full_name] > 0
        assert dense._w4a_activation_global_scale > 0
        assert dense._w4a_headroom_probe is False
    else:
        assert named.full_name not in processor._activation_global_scales
        assert processor._activation_amax[named.full_name] == pytest.approx(0.5)

@pytest.mark.parametrize("mlp_layers,headroom", [((), True), ((0,), False)])
def test_headroom_probe_uses_per_layer_mlp_override(mlp_layers, headroom):
    """A promoted FP8 MLP layer must not run or freeze an NVFP4 headroom probe."""
    nvfp4 = "w4a_nvfp" + "4"
    activation = {"mode": nvfp4, "recipe": "nvidia_headroom"}
    if mlp_layers:
        activation["mlp"] = {"mode": "w4afp8", "layers": list(mlp_layers)}
    processor = _headroom_probe_processor(activation)
    dense, named, subset = _projection("mlp.gate_proj")
    assert processor.begin_activation_scale_probe(subset) is headroom
    processor._record_activation_amax("mlp.gate_proj", torch.full((8, 128), 0.5), dense)
    processor.end_activation_scale_probe(subset)
    if headroom:
        assert processor._activation_global_scales[named.full_name] > 0
    else:
        assert named.full_name not in processor._activation_global_scales
        assert named.full_name not in processor._activation_amax


def _headroom_replay_model(activation, dtype=torch.bfloat16):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(vocab_size=64, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                         max_position_embeddings=64)
    model = LlamaForCausalLM(config).to(dtype).eval()
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, torch.nn.Linear):
                    setattr(parent, name, HookedLinear.from_linear(child))
    install_w4a_llama_replay(
        model,
        QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False, activation=activation),
    )
    return model


@pytest.mark.parametrize("activation,leaves,target_leaf,norm_name", [
    ({"mode": "w4a_nvfp4", "recipe": "nvidia_headroom"},
     ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
     "self_attn.q_proj", "input_layernorm"),
    ({"mode": "w4a_nvfp4", "recipe": "nvidia_headroom"},
     ("mlp.gate_proj", "mlp.up_proj"), "mlp.gate_proj", "post_attention_layernorm"),
    ({"mode": "w4a_nvfp4", "recipe": "least_squares",
      "attention": {"mode": "w4a_nvfp4", "recipe": "nvidia_headroom"}},
     ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
     "self_attn.q_proj", "input_layernorm"),
    ({"mode": "w4a_nvfp4", "recipe": "nvidia_headroom", "attention": {"mode": "w4afp8"}},
     ("mlp.gate_proj", "mlp.up_proj"), "mlp.gate_proj", "post_attention_layernorm"),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_installed_norm_replay_uses_frozen_headroom_scale(monkeypatch, activation, leaves,
                                                          target_leaf, norm_name, dtype):
    """The norm producer must match the deployed headroom operand.

    A headroom probe freezes its scale on the consuming projections. The norm
    that feeds those projections must bypass rounding while probing and then
    pack with the same frozen scale during capture, or the GPTQ Hessian learns
    a different operand than deployment executes.
    """
    import gptqmodel.nn_modules.qlinear.w4a_llama_replay as replay

    model = _headroom_replay_model(activation, dtype)
    processor = _headroom_probe_processor(activation)
    layer = model.model.layers[0]
    subset = {}
    for leaf in leaves:
        parent, child = leaf.split(".")
        module = getattr(getattr(layer, parent), child)
        subset[leaf] = NamedModule(module, leaf, f"model.layers.0.{leaf}", 0)
    parent, child = target_leaf.split(".")
    target = getattr(getattr(layer, parent), child)
    seen = []
    target.register_forward_pre_hook(lambda _m, args: seen.append(args[0].clone()))
    ids = torch.arange(16)[None]

    assert processor.begin_activation_scale_probe(subset) is True
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    probe_input = seen[-1]
    for leaf, named in subset.items():
        processor._record_activation_amax(leaf, seen[-1], named.module)
    processor.end_activation_scale_probe(subset)
    scale = processor._activation_global_scales[subset[target_leaf].full_name]
    assert scale > 0
    assert getattr(target, "_w4a_headroom_probe") is False

    pristine = {}
    norm_module = getattr(layer, norm_name)
    original_norm = replay._replay_norm

    def spy(norm, pristine_value, rounded, mode, recipe, *, preserve_codes, consumers=()):
        pristine[norm] = pristine_value.detach().clone()
        return original_norm(norm, pristine_value, rounded, mode, recipe,
                             preserve_codes=preserve_codes, consumers=consumers)

    monkeypatch.setattr(replay, "_replay_norm", spy)
    seen.clear()
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    capture_input = seen[-1]

    x = pristine[norm_module].float()
    variance = x.square().mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(variance + norm_module.variance_epsilon) * norm_module.weight.float()
    recipe = processor._target_recipe(subset[target_leaf].full_name, target)
    reference = nvfp4_block_qdq(y, scale, recipe)
    torch.testing.assert_close(capture_input.float(), reference, rtol=0, atol=0)
    # The probe must have collected the true normed operand, not a dynamically
    # rounded stand-in that happens to share the capture's numerics.
    assert not torch.equal(probe_input.float(), capture_input.float())


def test_hooked_linear_uses_frozen_nvidia_headroom_input_scale():
    dense = torch.nn.Linear(128, 128, bias=False, dtype=torch.float32).eval()
    dense.weight.data.copy_(torch.eye(128))
    dense._w4a_stream_replay_mode = "w4a_nvfp4"
    dense._w4a_stream_replay_recipe = "nvidia_headroom"
    dense._w4a_activation_global_scale = 0.25
    hooked = HookedLinear.from_linear(dense).eval()
    seen = []
    hooked.forward_hook = lambda _module, args, _output: seen.append(args[0].clone())
    x = torch.linspace(-20, 20, 256).reshape(2, 128)

    with torch.inference_mode():
        hooked(x)

    expected = nvfp4_block_qdq(x, 0.25, "nvidia_headroom")
    torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)

def test_hooked_linear_preserves_norm_input_without_requantizing(monkeypatch):
    dense = torch.nn.Linear(128, 32, bias=False)
    dense._w4a_stream_replay_mode = "w4a_nvfp4"
    dense._w4a_norm_preapplied = True
    hooked = HookedLinear.from_linear(dense)
    monkeypatch.setattr(
        "gptqmodel.nn_modules.qlinear.w4a_llama_replay.round_w4a_activation",
        lambda *_: (_ for _ in ()).throw(AssertionError("normalization operand rounded again")),
    )
    generator = torch.Generator().manual_seed(9772)
    x = torch.randn((2, 128), generator=generator)
    torch.testing.assert_close(hooked(x), dense(x), rtol=0, atol=0)

def test_native_capture_bypasses_activation_rounding_before_gptaq(monkeypatch):
    import copy

    from gptqmodel.nn_modules.qlinear.w4a_llama_replay import set_w4a_replay_enabled

    torch.manual_seed(9780)
    config = LlamaConfig(vocab_size=128, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                         max_position_embeddings=128)
    model = LlamaForCausalLM(config).float().eval()
    reference = copy.deepcopy(model)
    qcfg = QuantizeConfig(bits=4, group_size=128, rotation="hadamard",
                          activation={"mode": "w4a_nvfp4", "recipe": "least_squares"})
    install_w4a_llama_replay(model, qcfg)
    for layer in model.model.layers:
        set_w4a_replay_enabled(layer, False)
    def reject_round(*_args, **_kwargs):
        raise AssertionError("Native reference inputs were activation-quantized")
    monkeypatch.setattr("gptqmodel.nn_modules.qlinear.w4a_llama_replay._round", reject_round)
    ids = torch.tensor([[1, 7, 3]])
    with torch.inference_mode():
        expected = reference(input_ids=ids, use_cache=False).logits
        actual = model(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    for layer in model.model.layers:
        set_w4a_replay_enabled(layer, True)
    with pytest.raises(AssertionError, match="Native reference inputs"):
        model(input_ids=ids, use_cache=False)

def test_hooked_linear_preserves_native_capture_switch(monkeypatch):
    from gptqmodel.nn_modules.qlinear.w4a_llama_replay import set_w4a_replay_enabled

    dense = torch.nn.Linear(128, 32)
    dense._w4a_stream_replay_mode = "w4a_nvfp4"
    set_w4a_replay_enabled(dense, False)
    hooked = HookedLinear.from_linear(dense)
    def reject_round(*_args):
        raise AssertionError("Native capture rounded a HookedLinear operand")
    monkeypatch.setattr("gptqmodel.nn_modules.qlinear.w4a_llama_replay.round_w4a_activation", reject_round)
    x = torch.randn((2, 128), generator=torch.Generator().manual_seed(9781))
    torch.testing.assert_close(hooked(x), dense(x), rtol=0, atol=0)
    set_w4a_replay_enabled(hooked, True)
    with pytest.raises(AssertionError, match="Native capture rounded"):
        hooked(x)
