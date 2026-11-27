# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""SenseNova-Loop is package-owned: no checkpoint Python, native Qwen3.5 layers, independent oracles.

The oracles below never call port helpers for the quantity under test: the
execution plan is written out by hand, Sinkhorn/mHC routing is recomputed in
float64 in probability space, and the Loop itself is checked against a
*native* Qwen3.5 model whose layer list is the unrolled plan.
"""

import builtins
import hashlib
import json
import re

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral
from eager_ssm_kernels import eager_torch_ssm_kernels
from safetensors.torch import load_file, save_file
from transformers import CONFIG_MAPPING, DynamicCache

# Transformers' own Qwen3.5 is the numerical oracle only.
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config as NativeQwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration as NativeQwen3_5ForConditionalGeneration,
)

from gptqmodel import CalibrationConfig, GPTQModel
from gptqmodel.models.definitions.sensenova_loop import SenseNovaLoopQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.native.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from gptqmodel.models.ports.native.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5ForConditionalGeneration
from gptqmodel.models.ports.native.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from gptqmodel.models.ports.sensenova_loop import (
    MHCRouter,
    Qwen2Tokenizer,
    Qwen3VLProcessor,
    SenseNovaLoopConfig,
    SenseNovaLoopForConditionalGeneration,
    SenseNovaLoopMHCDecoderLayer,
)
from gptqmodel.models.ports.sensenova_loop.modeling_sensenova_loop import sinkhorn
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code


@pytest.fixture(autouse=True)
def _pin_eager_torch_ssm_kernels():
    """Run this module's CPU forwards on the Torch SSM/DeltaNet reference.

    ``transformers`` binds ``causal_conv1d``/``mamba_ssm``/``fla`` at import time
    when the distribution is installed, and those implementations are CUDA-only.
    The invariants here feed CPU tensors, so pin the reference kept on
    ``__wrapped__`` (upstream precedent #5537) instead of failing on the device.
    """

    with eager_torch_ssm_kernels():
        yield

LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2
# Hand-written plan for loop {start: 2, end: 6, eval_count: 2}: 0 1 | 2 3 4 5 | 2 3 4 5 | 6 7.
PLAN = (0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7)
MHC = {"start": 2, "end": 6, "streams": 2, "iterations": 20, "fixed_identity": False, "pre_scale": 1.0}

# sensenova/SenseNova-Loop-9B@fa381f7c: sha256 of "\n".join(sorted(weight_map)) of
# model.safetensors.index.json. The index has no mtp.* entries (mtp.safetensors is
# separate) and lists one tensor the fixed-count model never reads.
RELEASED_INDEX_SHA256 = "a5afd8550789da4af90281cf1fd8988250e6c57a1d2a839da8b1e60c32c1ceca"
RELEASED_INDEX_COUNT = 857
RELEASED_IGNORED_KEYS = {"model.language_model.loop_gate.logits"}


def raw_config(*, loop=(2, 6, 2), mhc=MHC, kv_heads=2):
    start, end, count = loop
    return {
        "text_config": {
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 8,
            "num_attention_heads": 4, "num_key_value_heads": kv_heads, "head_dim": 8, "layer_types": LAYER_TYPES,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 8,
            "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4, "pad_token_id": 0, "eos_token_id": 3,
            "attn_output_gate": True,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                                "mrope_section": [0, 0, 1], "mrope_interleaved": True},
        },
        "vision_config": {
            "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4, "out_hidden_size": 32,
            "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2, "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
        "image_token_id": 60, "video_token_id": 61, "vision_start_token_id": 62, "vision_end_token_id": 63,
        "loop_settings": {"start": start, "end": end, "eval_count": count, "counts": [count]},
        "mhc": None if mhc is None else dict(mhc),
    }


def tiny_model(seed=0, **kwargs):
    torch.manual_seed(seed)
    model = SenseNovaLoopForConditionalGeneration(SenseNovaLoopConfig(**raw_config(**kwargs))).eval()
    # Random tiny weights: routers far from their init so routing is non-trivial.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "mhc_" in name:
                parameter.add_(torch.randn_like(parameter) * (0.3 if name.endswith("weight") else 0.5))
            elif "A_log" not in name and "dt_bias" not in name:
                parameter.add_(torch.randn_like(parameter) * 0.05)
    return model


def media_inputs(dtype=torch.float32):
    return {
        "input_ids": torch.tensor([[1, 62, 60, 63, 5, 9, 12]]),
        "attention_mask": torch.ones(1, 7, dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[0, 0, 1, 0, 0, 0, 0]]),
        "pixel_values": torch.linspace(-1.7, 2.3, 96, dtype=dtype).reshape(4, 24),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }


def unrolled_native(model):
    """Native Qwen3.5 whose layer *list* is the execution plan, with copied (not shared) weights."""

    raw = raw_config(mhc=None, kv_heads=model.config.text_config.num_key_value_heads)
    plan = model.model.language_model.loop_plan
    raw["text_config"].update(num_hidden_layers=len(plan), layer_types=[LAYER_TYPES[i] for i in plan])
    raw.pop("loop_settings")
    raw.pop("mhc")
    native = NativeQwen3_5ForConditionalGeneration(NativeQwen3_5Config(**raw)).eval()
    state = {}
    prefix = "model.language_model.layers."
    source = model.state_dict()
    for key in native.state_dict():
        if key.startswith(prefix):
            visit, rest = key[len(prefix):].split(".", 1)
            state[key] = source[f"{prefix}{plan[int(visit)]}.{rest}"].clone()
        else:
            state[key] = source[key].clone()
    native.load_state_dict(state, strict=True)
    return native


def test_definition_owns_loader_and_needs_no_trust():
    loader = SenseNovaLoopQModel.loader
    assert isinstance(loader, LocalModelLoader)
    assert loader.config_class == "gptqmodel.models.ports.sensenova_loop.SenseNovaLoopConfig"
    assert loader.model_class == "gptqmodel.models.ports.sensenova_loop.SenseNovaLoopForConditionalGeneration"
    assert loader.processor_class == "gptqmodel.models.ports.sensenova_loop.Qwen3VLProcessor"
    assert loader.tokenizer_class == "gptqmodel.models.ports.sensenova_loop.Qwen2Tokenizer"
    assert SenseNovaLoopQModel.require_trust_remote_code is False
    # No global registration: Transformers itself still does not know the type.
    assert "sensenova_loop" not in CONFIG_MAPPING
    assert SenseNovaLoopForConditionalGeneration.config_class is SenseNovaLoopConfig
    assert issubclass(SenseNovaLoopForConditionalGeneration, Qwen3_5ForConditionalGeneration)
    assert issubclass(SenseNovaLoopMHCDecoderLayer, Qwen3_5DecoderLayer)
    model = SenseNovaLoopForConditionalGeneration(SenseNovaLoopConfig(**raw_config()))
    assert model._keep_in_fp32_modules == model._keep_in_fp32_modules_strict == {"mhc_attn", "mhc_ffn"}
    assert "SenseNovaLoopMHCDecoderLayer" in model._no_split_modules


@pytest.mark.parametrize("bad", [
    {"loop_settings": {"start": 2, "end": 9, "eval_count": 2, "counts": [2]}},
    {"loop_settings": {"start": 2, "end": 6, "eval_count": 2, "counts": [1, 2]}},
    {"loop_settings": {"start": 2, "end": 6, "eval_count": 0, "counts": [0]}},
    {"mhc": dict(MHC, start=1)},
    {"mhc": dict(MHC, streams=3)},
    {"mhc": dict(MHC, fixed_identity=1)},
    {"mhc": dict(MHC, pre_scale=float("nan"))},
    {"mhc": dict(MHC, extra=1)},
])
def test_invalid_released_settings_fail_at_config(bad):
    raw = raw_config()
    raw.update(bad)
    with pytest.raises(ValueError):
        SenseNovaLoopConfig(**raw)


def test_default_config_constructs_but_model_requires_loop_settings():
    config = SenseNovaLoopConfig()
    assert config.loop_settings is None
    with pytest.raises(ValueError, match="loop_settings"):
        SenseNovaLoopForConditionalGeneration(config)


def test_execution_plan_runs_shared_physical_layers_in_hand_written_order():
    model = tiny_model()
    layers = model.model.language_model.layers
    assert model.model.language_model.loop_plan == PLAN
    order = []
    handles = [layer.register_forward_pre_hook(lambda module, args, index=i: order.append(index))
               for i, layer in enumerate(layers)]
    try:
        with torch.inference_mode():
            model(torch.tensor([[1, 2, 3]]), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    assert tuple(order) == PLAN
    assert {i: order.count(i) for i in range(8)} == {0: 1, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1}
    # One physical copy: 8 layers, mHC only on 2..5, routers FP32.
    assert len(layers) == 8
    assert [isinstance(layer, SenseNovaLoopMHCDecoderLayer) for layer in layers] == [i in range(2, 6) for i in range(8)]
    for name, parameter in model.named_parameters():
        if "mhc_" in name:
            assert parameter.dtype == torch.float32


def _sinkhorn64(logits, iterations):
    matrix = logits.double().exp()
    for _ in range(iterations):
        matrix = matrix / matrix.sum(-1, keepdim=True)
        matrix = matrix / matrix.sum(-2, keepdim=True)
    return matrix


def _router64(router, residual):
    streams = router.streams
    flat = residual.double().flatten(-2)
    normalized = flat / torch.sqrt(flat.pow(2).mean(-1, keepdim=True) + 1e-6)
    logits = normalized @ router.weight.double().T + router.bias.double()
    pre = 1 / (1 + torch.exp(-logits[..., :streams]))
    post = 2 / (1 + torch.exp(-logits[..., streams:2 * streams]))
    if router.fixed_identity:
        mix = torch.eye(streams, dtype=torch.float64).expand(*flat.shape[:-1], streams, streams)
    else:
        mix = _sinkhorn64(logits[..., 2 * streams:].reshape(*flat.shape[:-1], streams, streams), router.iterations)
    branch = torch.einsum("bts,btsh->bth", pre, residual.double()) * router.pre_scale
    return branch, post, mix


def _merge64(residual, branch, post, mix):
    return torch.einsum("btij,btjh->btih", mix, residual.double()) + post[..., None] * branch[..., None, :]


def test_sinkhorn_matches_probability_space_float64():
    torch.manual_seed(3)
    for streams in (2, 4):
        logits = torch.randn(3, 5, streams, streams) * 2
        actual = sinkhorn(logits, 20)
        expected = _sinkhorn64(logits, 20)
        torch.testing.assert_close(actual.double(), expected, rtol=0, atol=2e-6)
        torch.testing.assert_close(expected.sum(-2), torch.ones(3, 5, streams, dtype=torch.float64))


@pytest.mark.parametrize("settings", [
    MHC, dict(MHC, streams=4), dict(MHC, fixed_identity=True), dict(MHC, pre_scale=0.5, iterations=3),
])
def test_mhc_router_equations_float64(settings):
    torch.manual_seed(4)
    router = MHCRouter(16, settings)
    with torch.no_grad():
        router.weight.normal_(0, 0.3)
        router.bias.copy_(router.initial_bias() + torch.randn_like(router.bias) * 0.5)
    residual = torch.randn(2, 3, settings["streams"], 16)
    branch, post, mix = router(residual)
    expected = _router64(router, residual)
    for actual, oracle in zip((branch, post, mix), expected):
        torch.testing.assert_close(actual.double(), oracle, rtol=0, atol=2e-5)
    branch_out = torch.randn(2, 3, 16)
    torch.testing.assert_close(
        MHCRouter.merge(residual, branch_out, post, mix).double(),
        _merge64(residual, branch_out, *expected[1:]), rtol=0, atol=2e-5,
    )
    # A model-wide cast never demotes the router.
    assert router.to(torch.bfloat16).weight.dtype == torch.float32
    with torch.no_grad():
        router.weight.data = router.weight.data.half()
    with pytest.raises(RuntimeError, match="FP32"):
        router(residual)


def test_router_initialization_matches_released_formula():
    model = SenseNovaLoopForConditionalGeneration(SenseNovaLoopConfig(**raw_config()))
    router = model.model.language_model.layers[2].mhc_attn
    p = torch.tensor([0.25, 0.75])
    diagonal = torch.log(torch.tensor((1 - 1 / 16) * 16))
    expected = torch.cat([torch.log(p / (1 - p)), torch.zeros(2), torch.tensor([diagonal, 0, 0, diagonal])])
    torch.testing.assert_close(router.bias.detach(), expected, rtol=0, atol=1e-6)
    assert torch.count_nonzero(router.weight) == 0


@pytest.mark.parametrize("physical", [2, 3, 5])
def test_mhc_layer_matches_native_sublayers_with_float64_routing(physical):
    model = tiny_model(seed=physical)
    layer = model.model.language_model.layers[physical]
    captured = []

    def capture(module, args, kwargs):
        captured.append((args[0].detach().clone(), dict(kwargs)))

    handle = layer.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.inference_mode():
            model(torch.tensor([[4, 9, 17, 2, 33]]), use_cache=False)
    finally:
        handle.remove()
    assert len(captured) == 2  # both loop visits
    for hidden, kwargs in captured:
        with torch.inference_mode():
            actual = layer(hidden, **kwargs)
            if physical == MHC["start"]:
                residual = hidden.double().unsqueeze(-2).expand(*hidden.shape[:-1], 2, hidden.shape[-1])
            else:
                residual = hidden.double().reshape(*hidden.shape[:-1], 2, -1)
            branch, post, mix = _router64(layer.mhc_attn, residual)
            normed = layer.input_layernorm(branch.float())
            if layer.block_type == "linear_attention":
                out = layer.linear_attn(hidden_states=normed, cache_params=None, attention_mask=kwargs["attention_mask"])
            else:
                out, _ = layer.self_attn(hidden_states=normed, position_embeddings=kwargs["position_embeddings"],
                                         attention_mask=kwargs["attention_mask"])
            residual = _merge64(residual, out.double(), post, mix)
            branch, post, mix = _router64(layer.mhc_ffn, residual)
            residual = _merge64(residual, layer.mlp(layer.post_attention_layernorm(branch.float())).double(), post, mix)
            expected = residual.mean(-2) if physical == MHC["end"] - 1 else residual.flatten(-2)
        assert actual.shape[-1] == (32 if physical == MHC["end"] - 1 else 64)
        torch.testing.assert_close(actual.double(), expected, rtol=1e-4, atol=1e-5)


def test_loop_and_mhc_disabled_equals_native_exactly():
    model = tiny_model(loop=(0, 8, 1), mhc=None)
    assert model.model.language_model.loop_plan == tuple(range(8))
    raw = raw_config(mhc=None)
    raw.pop("loop_settings")
    raw.pop("mhc")
    native = NativeQwen3_5ForConditionalGeneration(NativeQwen3_5Config(**raw)).eval()
    native.load_state_dict(model.state_dict(), strict=True)
    ids = torch.tensor([[1, 7, 11, 13, 8, 9]])
    with torch.inference_mode():
        for inputs in ({"input_ids": ids}, media_inputs()):
            torch.testing.assert_close(model(**inputs, use_cache=False).logits,
                                       native(**inputs, use_cache=False).logits, rtol=0, atol=0)
        options = {"max_new_tokens": 4, "do_sample": False, "eos_token_id": None, "pad_token_id": 0}
        torch.testing.assert_close(model.generate(ids, **options), native.generate(ids, **options))


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_loop_without_mhc_equals_unrolled_native_with_independent_caches(kv_heads):
    model = tiny_model(mhc=None, kv_heads=kv_heads)
    native = unrolled_native(model)
    ids = torch.tensor([[1, 7, 11, 13, 8, 9]])
    with torch.inference_mode():
        for inputs in ({"input_ids": ids}, media_inputs()):
            torch.testing.assert_close(model(**inputs, use_cache=False).logits,
                                       native(**inputs, use_cache=False).logits, rtol=0, atol=0)
        cache_a = cache_b = None
        for start, end in [(0, 1), (1, 5), (5, 6)]:
            a = model(ids[:, start:end], past_key_values=cache_a, use_cache=True)
            b = native(ids[:, start:end], past_key_values=cache_b, use_cache=True)
            cache_a, cache_b = a.past_key_values, b.past_key_values
            torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
        assert len(cache_a.layers) == len(PLAN) == len(cache_b.layers)
        for beams in (1, 2):
            options = {"max_new_tokens": 4, "do_sample": False, "num_beams": beams, "eos_token_id": None,
                       "pad_token_id": 0}
            torch.testing.assert_close(model.generate(ids, **options), native.generate(ids, **options))
            torch.testing.assert_close(model.generate(**media_inputs(), **options),
                                       native.generate(**media_inputs(), **options))


def test_mhc_cache_matches_full_forward_and_keeps_visits_separate():
    model = tiny_model()
    ids = torch.tensor([[1, 7, 11, 13, 8, 9, 21, 5]])
    with torch.inference_mode():
        full = model(ids, use_cache=False).logits
        cache = None
        pieces = []
        for start, end in [(0, 3), (3, 4), (4, 7), (7, 8)]:
            output = model(ids[:, start:end], past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            pieces.append(output.logits)
        torch.testing.assert_close(torch.cat(pieces, 1), full, rtol=1e-4, atol=1e-5)
        assert len(cache.layers) == len(PLAN)
        # Visits 2 and 6 run physical layer 2 (linear), 3 and 7 run layer 3 (full).
        assert not torch.equal(cache.layers[2].recurrent_states[0], cache.layers[6].recurrent_states[0])
        assert cache.layers[3].keys.shape == cache.layers[7].keys.shape == (1, 2, 8, 8)
        assert not torch.equal(cache.layers[3].keys, cache.layers[7].keys)
        assert cache.get_seq_length() == 8

        generated = model.generate(ids, max_new_tokens=4, do_sample=False, eos_token_id=None, pad_token_id=0)
        greedy = ids
        for _ in range(4):
            token = model(greedy, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
            greedy = torch.cat([greedy, token], 1)
        torch.testing.assert_close(generated, greedy)
        beam = model.generate(ids, max_new_tokens=3, num_beams=2, do_sample=False, eos_token_id=None, pad_token_id=0)
        assert beam.shape == (1, 11)


def test_visit_cache_routes_state_quant_and_cached_generate():
    from gptqmodel.models.ports.sensenova_loop.modeling_sensenova_loop import _VisitCache
    from gptqmodel.utils.leapquant import LeapQuantConfig, make_leapquant_cache

    model = tiny_model()
    plan = model.model.language_model.loop_plan
    physical = 2
    visit = max(index for index, layer in enumerate(plan) if layer == physical)
    cache = make_leapquant_cache(
        model.config.text_config, LeapQuantConfig(bits=8, window=4, rank=2)
    )
    view = _VisitCache(cache, physical, visit)
    # The shared Qwen3.5 forward probes ``cache_params.layers[layer_idx]`` for
    # LeapQuant state, so a visit must expose its slot under the physical index.
    assert view.layers[physical] is cache.layers[visit]
    assert view.layers[physical].state_quant is not None

    ids = torch.tensor([[1, 7, 11]])
    with torch.inference_mode():
        cached = model(ids, use_cache=True)
        generated = model.generate(
            ids, max_new_tokens=1, do_sample=False, eos_token_id=None, pad_token_id=0
        )
    assert cached.past_key_values is not None
    assert generated.shape == (1, ids.shape[1] + 1)


def test_foreign_caches_are_replaced_only_when_empty():
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        physical = DynamicCache(config=model.config.text_config)
        output = model(ids, past_key_values=physical, use_cache=True)
        assert output.past_key_values is not physical and len(output.past_key_values.layers) == len(PLAN)
        populated = DynamicCache(config=model.config.text_config)
        populated.update(torch.zeros(1, 2, 3, 8), torch.zeros(1, 2, 3, 8), 3)
        with pytest.raises(ValueError, match="execution plan"):
            model(ids, past_key_values=populated, use_cache=True)


def test_full_size_meta_model_matches_released_key_inventory():
    raw = raw_config()
    raw["text_config"].update(
        vocab_size=248320, hidden_size=4096, intermediate_size=12288, num_hidden_layers=32, num_attention_heads=16,
        num_key_value_heads=4, head_dim=256, layer_types=LAYER_TYPES * 4, linear_num_key_heads=16,
        linear_num_value_heads=32, linear_key_head_dim=128, linear_value_head_dim=128, mtp_num_hidden_layers=1,
    )
    raw["vision_config"].update(depth=27, hidden_size=1152, intermediate_size=4304, num_heads=16,
                                out_hidden_size=4096, patch_size=16, num_position_embeddings=2304)
    raw["loop_settings"] = {"counts": [2], "end": 28, "eval_count": 2, "start": 4}
    raw["mhc"] = {"end": 28, "fixed_identity": False, "iterations": 20, "pre_scale": 1.0, "start": 4, "streams": 2}
    with torch.device("meta"):
        model = SenseNovaLoopForConditionalGeneration(SenseNovaLoopConfig(**raw))
    keys = set(model.state_dict())
    released = sorted(keys | RELEASED_IGNORED_KEYS)
    assert len(released) == RELEASED_INDEX_COUNT
    assert hashlib.sha256("\n".join(released).encode()).hexdigest() == RELEASED_INDEX_SHA256
    assert not keys & RELEASED_IGNORED_KEYS
    for key in RELEASED_IGNORED_KEYS:
        assert any(re.search(pattern, key) for pattern in model._keys_to_ignore_on_load_unexpected)
    plan = model.model.language_model.loop_plan
    assert plan == tuple(range(28)) + tuple(range(4, 28)) + tuple(range(28, 32))
    routers = [name for name in keys if ".mhc_" in name]
    assert len(routers) == 24 * 4
    assert model.model.language_model.layers[4].mhc_attn.weight.shape == (8, 8192)


def test_bf16_load_keeps_routers_fp32_and_ignores_aux_tensors(tmp_path):
    model = tiny_model()
    model.save_pretrained(tmp_path)
    payload = tmp_path / "model.safetensors"
    state = load_file(payload)
    state["model.language_model.loop_gate.logits"] = torch.zeros(2)
    state["mtp.fc.weight"] = torch.zeros(2, 2)
    save_file(state, payload, metadata={"format": "pt"})
    loaded = SenseNovaLoopQModel.loader.from_pretrained(tmp_path, dtype=torch.bfloat16).eval()
    assert type(loaded) is SenseNovaLoopForConditionalGeneration
    for name, parameter in loaded.named_parameters():
        assert parameter.dtype == (torch.float32 if ".mhc_" in name else torch.bfloat16), name
        if ".mhc_" in name:
            torch.testing.assert_close(parameter, model.state_dict()[name], rtol=0, atol=0)
    assert "model.language_model.loop_gate.logits" not in loaded.state_dict()
    with torch.inference_mode():
        assert torch.isfinite(loaded(torch.tensor([[1, 2, 3]]), use_cache=False).logits).all()


def write_checkpoint(path):
    model = tiny_model()
    model.save_pretrained(path)
    tokenizer = Qwen2Tokenizer(vocab={"a": 0, "b": 1, "c": 2, "d": 3, "ab": 4, "<|endoftext|>": 5, "<|im_end|>": 6},
                               merges=[("a", "b")], unk_token=None)
    tokenizer.add_special_tokens({"additional_special_tokens": [
        "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]})
    Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=2, merge_size=2),
        video_processor=Qwen3VLVideoProcessor(patch_size=2, temporal_patch_size=2, merge_size=2),
        tokenizer=tokenizer,
    ).save_pretrained(path)
    # Auxiliary tensors of the release that the model does not own.
    save_file({"mtp.fc.weight": torch.ones(2, 2)}, path / "mtp.safetensors", metadata={"format": "pt"})
    state = load_file(path / "model.safetensors")
    state["model.language_model.loop_gate.logits"] = torch.tensor([0.5, -0.5])
    save_file(state, path / "model.safetensors", metadata={"format": "pt"})
    marker = path / "EXECUTED"
    sentinel = f"from pathlib import Path\nPath({str(marker)!r}).write_text('UNSAFE')\nraise RuntimeError('checkpoint')\n"
    for filename in ("modeling_sensenova_loop.py", "tokenization_hostile.py"):
        (path / filename).write_text(sentinel)
    config = json.loads((path / "config.json").read_text())
    config["auto_map"] = {
        "AutoConfig": "modeling_sensenova_loop.SenseNovaLoopConfig",
        "AutoModel": "modeling_sensenova_loop.SenseNovaLoopModel",
        "AutoModelForCausalLM": "modeling_sensenova_loop.SenseNovaLoopForConditionalGeneration",
        "AutoModelForImageTextToText": "modeling_sensenova_loop.SenseNovaLoopForConditionalGeneration",
    }
    (path / "config.json").write_text(json.dumps(config))
    tokenizer_config = json.loads((path / "tokenizer_config.json").read_text())
    tokenizer_config["auto_map"] = {"AutoTokenizer": ["tokenization_hostile.HostileTokenizer", None]}
    (path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    return model, marker


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0"])
def test_public_load_ignores_hostile_auto_map_without_trust_or_bypass(tmp_path, monkeypatch, trust, bypass):
    expected, marker = write_checkpoint(tmp_path)
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    original_import = builtins.__import__

    def deny_checkpoint(name, *args, **kwargs):
        if name.split(".")[0] in {"modeling_sensenova_loop", "tokenization_hostile"}:
            raise AssertionError("checkpoint Python must never be imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_checkpoint)
    assert type(load_model_config(tmp_path, trust_remote_code=trust)) is SenseNovaLoopConfig
    assert resolve_trust_remote_code(tmp_path, trust_remote_code=trust) is False
    loaded = GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert type(loaded.model) is SenseNovaLoopForConditionalGeneration
    assert loaded.trust_remote_code is False
    assert type(loaded.processor) is Qwen3VLProcessor
    with torch.inference_mode():
        torch.testing.assert_close(loaded(**media_inputs(), use_cache=False).logits,
                                   expected(**media_inputs(), use_cache=False).logits, rtol=0, atol=0)
        assert loaded.generate(**media_inputs(), max_new_tokens=2, min_new_tokens=2, do_sample=False).shape == (1, 9)
    assert not marker.exists()


def test_public_awq_quantizes_shared_linears_keeps_routers_and_aux_tensors(tmp_path):
    source = tmp_path / "source"
    original, marker = write_checkpoint(source)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None)
    loaded = GPTQModel.load(str(source), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16)
    calibration = [{"input_ids": torch.tensor([[4, 5, 6, 7, 6, 5, 4, 7, 9, 11]]),
                    "attention_mask": torch.ones(1, 10, dtype=torch.long)}]
    loaded.quantize(CalibrationConfig(dataset=calibration, batch_size=1, min_length=1), backend=BACKEND.TORCH)
    expected = {
        f"model.language_model.layers.{index}.{projection}"
        for index, kind in enumerate(LAYER_TYPES)
        for projection in (["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
                           if kind == "full_attention" else
                           ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"])
        + ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    }
    assert {name for name, module in loaded.model.named_modules() if isinstance(module, AwqTorchLinear)} == expected
    for name, parameter in loaded.model.named_parameters():
        if ".mhc_" in name:
            assert parameter.dtype == torch.float32
            torch.testing.assert_close(parameter, original.state_dict()[name], rtol=0, atol=0)
    with torch.inference_mode():
        logits = loaded(**calibration[0], use_cache=False).logits
    assert torch.isfinite(logits).all()
    destination = tmp_path / "packed"
    loaded.save(str(destination))
    state = {}
    for payload in destination.glob("*.safetensors"):
        state.update(load_file(payload))
    for name in expected:
        assert {f"{name}.qweight", f"{name}.qzeros", f"{name}.scales"} <= state.keys()
    assert all(state[key].dtype == torch.float32 for key in state if ".mhc_" in key)
    torch.testing.assert_close(state["model.language_model.loop_gate.logits"], torch.tensor([0.5, -0.5]))
    assert "mtp.fc.weight" in state
    reloaded = GPTQModel.load(str(destination), backend=BACKEND.TORCH, dtype=torch.float16, device="cpu")
    assert type(reloaded.model) is SenseNovaLoopForConditionalGeneration
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(**calibration[0], use_cache=False).logits, logits, rtol=0, atol=0)
    assert not marker.exists()


@pytest.mark.parametrize("kv_heads", [2, 4])
def test_awq_scale_folds_neutral_on_looped_and_plain_layers(kv_heads):
    model = tiny_model(kv_heads=kv_heads)
    inputs = media_inputs()

    def forward():
        return model(**inputs, use_cache=False).logits

    mlp = {("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")), ("mlp.up_proj", ("mlp.down_proj",))}
    # 0/1/6: plain linear, 7: plain full; 2/4/5: looped mHC linear, 3: looped mHC full.
    for index, layer in enumerate(model.model.language_model.layers):
        checked = assert_awq_folds_neutral(SenseNovaLoopQModel, model, layer, forward)
        groups = {(source, tuple(targets)) for source, targets, _ in checked}
        expected = set(mlp)
        if LAYER_TYPES[index] == "full_attention":
            expected.add(("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")))
            if kv_heads == 4:
                expected.add(("self_attn.v_proj", ("self_attn.o_proj",)))
        assert groups == expected, index
