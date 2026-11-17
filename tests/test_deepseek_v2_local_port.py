# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real tiny native MLA, legacy variants, loading and quantization consumers."""

import copy
import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import DeepseekV2ForCausalLM as NativeModel
from transformers import PreTrainedTokenizerFast
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2MLP

from gptqmodel import GPTQModel
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.definitions.deepseek_v2 import DeepSeekV2QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from gptqmodel.models.ports.deepseek_v2.modeling_deepseek_v2 import DeepseekV2ForCausalLM
from gptqmodel.quantization.config import AWQConfig, QuantizeConfig
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code
from gptqmodel.utils.security import BYPASS_ENV
from tests.module_tree.test_deepseek_v2_local_port import tiny_config


def config(**overrides):
    values = tiny_config().to_dict()
    values.update(overrides)
    return DeepseekV2Config(**values)


def model(**overrides):
    torch.manual_seed(271)
    return DeepseekV2ForCausalLM(config(**overrides)).eval()


def save_checkpoint(path, hf_model, malicious=False):
    path.mkdir(exist_ok=True)
    hf_model.save_pretrained(path, safe_serialization=True)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 3, "[PAD]": 0, "[BOS]": 1, "[EOS]": 2}, unk_token="[UNK]")),
        pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]",
    )
    tokenizer.save_pretrained(path)
    if malicious:
        payload = json.loads((path / "config.json").read_text())
        payload["auto_map"] = {"AutoConfig": "evil.EvilConfig", "AutoModelForCausalLM": "evil.EvilModel"}
        (path / "config.json").write_text(json.dumps(payload))
        (path / "evil.py").write_text("raise AssertionError('checkpoint Python executed')\n")


@pytest.mark.parametrize("q_rank", [None, 8], ids=["lite", "full"])
@pytest.mark.parametrize("method", ["greedy", "group_limited_greedy"])
@torch.inference_mode()
def test_native_numerics_conversion_and_compressed_cache(q_rank, method):
    actual = model(q_lora_rank=q_rank, topk_method=method, routed_scaling_factor=2.3)
    native = NativeModel(tiny_config(q_lora_rank=q_rank, topk_method=method, routed_scaling_factor=2.3)).eval()
    assert native.load_state_dict(actual.state_dict()).missing_keys == []
    ids = torch.tensor([[1, 4, 7, 9, 12], [0, 1, 8, 6, 5]])
    mask = (ids != 0).long()
    expected = native(ids, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(actual(ids, attention_mask=mask, use_cache=False).logits, expected, rtol=2e-5, atol=2e-6)
    fused = copy.deepcopy(actual.state_dict())
    assert _convert_model_with_defuser(DeepSeekV2QModel, actual, cleanup_original=True)
    # Every fused expert is split at the declared gate/up boundary, without
    # transpose, loss, or an extra retained fused parameter.
    for i in range(actual.config.n_routed_experts):
        for name, tensor in (
            ("gate_proj", fused["model.layers.1.mlp.experts.gate_up_proj"][i, :12]),
            ("up_proj", fused["model.layers.1.mlp.experts.gate_up_proj"][i, 12:]),
            ("down_proj", fused["model.layers.1.mlp.experts.down_proj"][i]),
        ):
            torch.testing.assert_close(actual.state_dict()[f"model.layers.1.mlp.experts.{i}.{name}.weight"], tensor, rtol=0, atol=0)
    assert not any(k.endswith(("experts.gate_up_proj", "experts.down_proj")) for k in actual.state_dict())
    torch.testing.assert_close(actual(ids, attention_mask=mask, use_cache=False).logits, expected, rtol=2e-5, atol=2e-6)
    prefix = actual(ids[:, :3], attention_mask=mask[:, :3], use_cache=True)
    cache = prefix.past_key_values
    assert cache.get_seq_length() == 3
    assert cache.layers[0].keys.shape[-1] == actual.config.kv_lora_rank
    assert cache.layers[0].values.shape[-1] == actual.config.qk_rope_head_dim
    continuation = actual(ids[:, 3:], attention_mask=mask, past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == 5
    torch.testing.assert_close(continuation.logits, expected[:, 3:], rtol=2e-5, atol=2e-6)
    changed = ids.clone()
    changed[:, -1] = 19
    torch.testing.assert_close(actual(changed, attention_mask=mask, use_cache=False).logits[:, :-1], expected[:, :-1], rtol=2e-5, atol=2e-6)
    generated = actual.generate(ids[:1, :3], max_new_tokens=3, do_sample=False)
    no_cache = actual.generate(ids[:1, :3], max_new_tokens=3, do_sample=False, use_cache=False)
    assert torch.equal(generated, no_cache)


@pytest.mark.parametrize("method", ["greedy", "group_limited_greedy"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("top_k", [1, 2])
@torch.inference_mode()
def test_routing_matches_independent_legacy_oracle(method, normalize, top_k):
    hf_model = model(topk_method=method, norm_topk_prob=normalize, num_experts_per_tok=top_k, routed_scaling_factor=3.7)
    router = hf_model.model.layers[1].mlp.gate
    x = torch.randn(2, 5, 24)
    logits = x.reshape(-1, 24).float() @ router.weight.float().T
    scores = logits.softmax(-1)
    candidate = scores.clone()
    if method == "group_limited_greedy":
        group_max = scores.reshape(-1, 2, 2).amax(-1)
        chosen = group_max.argmax(-1)
        allowed = torch.arange(4).div(2, rounding_mode="floor")[None] == chosen[:, None]
        candidate[~allowed] = 0
    indices = candidate.argsort(dim=-1, descending=True)[:, :top_k]
    expected = scores.gather(1, indices)
    if top_k > 1 and normalize:
        expected = expected / (expected.sum(-1, keepdim=True) + 1e-20)
    else:
        expected *= 3.7
    got_logits, weights, got_indices = router(x)
    order = weights.argsort(-1, descending=True)
    torch.testing.assert_close(got_logits, logits, rtol=0, atol=0)
    assert torch.equal(got_indices.gather(1, order), indices)
    torch.testing.assert_close(weights.gather(1, order), expected)


@pytest.mark.parametrize("routed,shared,freq,expected_moe", [
    (None, None, 1, []), (4, None, 2, [2, 4]), (4, 2, 2, [2, 4]), (4, 2, 1, [1, 2, 3, 4]),
])
@torch.inference_mode()
def test_dense_periodic_and_optional_shared_layout_parity(routed, shared, freq, expected_moe):
    actual = model(num_hidden_layers=5, n_routed_experts=routed, n_shared_experts=shared, moe_layer_freq=freq)
    assert [i for i, layer in enumerate(actual.model.layers) if not isinstance(layer.mlp, DeepseekV2MLP)] == expected_moe
    # Assemble the independently specified layout from native dense/MLA blocks.
    native = NativeModel(tiny_config(num_hidden_layers=5, first_k_dense_replace=5)).eval()
    for i in expected_moe:
        source = NativeModel(tiny_config(num_hidden_layers=1, first_k_dense_replace=0)).eval().model.layers[0].mlp
        if shared is None:
            # No-shared variant's oracle sums actual per-expert native MLPs below.
            source.shared_experts = torch.nn.Identity()
        native.model.layers[i].mlp = source
    # For the no-shared case use zero shared contribution, leaving the routed
    # calculation and complete decoder output as the oracle.
    handles = []
    if shared is None:
        for i in expected_moe:
            handles.append(native.model.layers[i].mlp.shared_experts.register_forward_hook(lambda _m, _a, out: torch.zeros_like(out)))
    native.load_state_dict(actual.state_dict(), strict=True)
    ids = torch.tensor([[1, 9, 8, 6], [1, 3, 4, 7]])
    try:
        torch.testing.assert_close(actual(ids).logits, native(ids).logits, rtol=0, atol=0)
    finally:
        for handle in handles:
            handle.remove()
    _convert_model_with_defuser(DeepSeekV2QModel, actual, cleanup_original=True)
    wrapper = DeepSeekV2QModel(actual, quantized=False, quantize_config=QuantizeConfig())
    simple = {
        name.split(":", 1)[0]
        for subset in wrapper.simple_layer_modules(actual.config, wrapper.quantize_config)
        for name in subset
    }
    for layer in actual.model.layers:
        for key in layer.state_dict():
            assert wrapper.module_plan().find(key.rsplit(".", 1)[0]) is not None
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                assert name in simple


@pytest.mark.parametrize("q_rank", [None, 8])
@pytest.mark.parametrize("defused", [False, True], ids=["fused", "checkpoint-names"])
@torch.inference_mode()
def test_safetensors_save_reload_and_expert_conversion(tmp_path, q_rank, defused):
    original = model(q_lora_rank=q_rank, norm_topk_prob=True, moe_layer_freq=2, num_hidden_layers=3)
    if defused:
        _convert_model_with_defuser(DeepSeekV2QModel, original, cleanup_original=True)
    ids = torch.tensor([[1, 8, 6, 4]])
    expected = original(ids).logits
    path = tmp_path / "first"
    save_checkpoint(path, original, malicious=True)
    assert list(path.glob("*.safetensors")) and not list(path.glob("*.bin"))
    loaded, info = DeepSeekV2QModel.loader.from_pretrained(path, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    loaded.eval()
    _convert_model_with_defuser(DeepSeekV2QModel, loaded, cleanup_original=True)
    _convert_model_with_defuser(DeepSeekV2QModel, original, cleanup_original=True)
    assert loaded.state_dict().keys() == original.state_dict().keys()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(loaded(ids).logits, expected, rtol=2e-5, atol=2e-6)
    save_checkpoint(tmp_path / "second", loaded)
    again = DeepSeekV2QModel.loader.from_pretrained(tmp_path / "second").eval()
    torch.testing.assert_close(again(ids).logits, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@torch.inference_mode()
def test_public_load_owns_config_and_model_despite_auto_map(tmp_path, monkeypatch, trust, bypass):
    original = model()
    path = tmp_path / "model"
    save_checkpoint(path, original, malicious=True)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")  # Existing security bypass is the tested input.
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    assert resolve_trust_remote_code(path, trust_remote_code=trust) is False
    selected = load_model_config(path, trust_remote_code=trust)
    assert type(selected) is DeepseekV2Config
    loaded = GPTQModel.load(str(path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert type(loaded) is DeepSeekV2QModel
    assert type(loaded.model) is DeepseekV2ForCausalLM
    ids = torch.tensor([[1, 6, 3, 7]])
    torch.testing.assert_close(loaded.model(ids).logits, original(ids).logits, rtol=2e-5, atol=2e-6)


@torch.inference_mode()
def test_failed_checkpoint_load_then_corrected_retry(tmp_path):
    original = model()
    path = tmp_path / "retry"
    save_checkpoint(path, original)
    weights = load_file(path / "model.safetensors")
    good = weights["model.layers.0.self_attn.q_b_proj.weight"].clone()
    weights["model.layers.0.self_attn.q_b_proj.weight"] = good[:1]
    save_file(weights, path / "model.safetensors")
    with pytest.raises((ValueError, RuntimeError), match="(?i)(size|shape|mismatch)"):
        DeepSeekV2QModel.loader.from_pretrained(path)
    weights["model.layers.0.self_attn.q_b_proj.weight"] = good
    save_file(weights, path / "model.safetensors")
    retry = DeepSeekV2QModel.loader.from_pretrained(path).eval()
    ids = torch.tensor([[1, 3, 9]])
    torch.testing.assert_close(retry(ids).logits, original(ids).logits, rtol=0, atol=0)


@pytest.mark.parametrize("defused", [False, True], ids=["fused", "checkpoint-names"])
@pytest.mark.parametrize("routed,shared,freq", [(4, 2, 1), (None, None, 1), (4, None, 2), (4, 2, 2)])
@torch.inference_mode()
def test_public_quantization_shell_materializes_exact_checkpoint_weights(tmp_path, defused, routed, shared, freq):
    original = model(num_hidden_layers=3, n_routed_experts=routed, n_shared_experts=shared, moe_layer_freq=freq)
    if defused:
        _convert_model_with_defuser(DeepSeekV2QModel, original, cleanup_original=True)
    path = tmp_path / "lazy"
    save_checkpoint(path, original, malicious=True)
    wrapped = GPTQModel.load(
        str(path), quantize_config=QuantizeConfig(device="cpu", group_size=-1, offload_to_disk=True),
        dtype=torch.float32,
    )
    assert wrapped.turtle_model is not None
    assert DeepSeekV2QModel.resolve_hf_conversion_map_reversed(wrapped.model) is not None
    wrapped.shell_module_materialize(wrapped.model, torch.device("cpu"), module_path="")
    _convert_model_with_defuser(DeepSeekV2QModel, original, cleanup_original=True)
    assert set(wrapped.model.state_dict()) == set(original.state_dict())
    for key, value in original.state_dict().items():
        actual = wrapped.model.state_dict()[key]
        assert actual.device.type == "cpu"
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    ids = torch.tensor([[1, 8, 5, 7]])
    torch.testing.assert_close(wrapped.model(ids).logits, original(ids).logits, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("q_rank", [None, 8], ids=["lite-yarn", "full-yarn"])
@torch.inference_mode()
def test_legacy_yarn_metadata_and_native_cache_roundtrip(tmp_path, q_rank):
    # Same YaRN fields as pinned DeepSeek-Coder-V2-Lite-Instruct, with tiny
    # original context so the test exercises positions beyond that context.
    rope = {
        "type": "yarn", "factor": 40.0, "beta_fast": 32.0, "beta_slow": 1.0,
        "mscale": 0.707, "mscale_all_dim": 0.707, "original_max_position_embeddings": 4,
    }
    hf_model = model(q_lora_rank=q_rank, max_position_embeddings=160, rope_scaling=rope, rope_theta=10000.0)
    save_checkpoint(tmp_path, hf_model, malicious=True)
    selected = load_model_config(tmp_path, trust_remote_code=True)
    assert selected.rope_parameters["rope_type"] == "yarn"
    assert selected.rope_parameters["factor"] == 40.0
    assert selected.rope_parameters["mscale_all_dim"] == 0.707
    loaded = DeepSeekV2QModel.loader.from_pretrained(tmp_path).eval()
    ids = torch.tensor([[1, 9, 5, 6, 7, 8]])
    expected = hf_model(ids, use_cache=False).logits
    torch.testing.assert_close(loaded(ids, use_cache=False).logits, expected, rtol=0, atol=0)
    prefix = loaded(ids[:, :4], use_cache=True)
    suffix = loaded(ids[:, 4:], past_key_values=prefix.past_key_values, use_cache=True)
    torch.testing.assert_close(suffix.logits, expected[:, 4:], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("overrides,match", [
    ({"moe_layer_freq": 0}, "moe_layer_freq"), ({"ep_size": 2}, "ep_size"),
    ({"scoring_func": "sigmoid"}, "scoring_func"), ({"topk_method": "noaux_tc"}, "topk_method"),
    ({"n_routed_experts": 3, "topk_method": "group_limited_greedy"}, "group-limited"),
    ({"num_experts_per_tok": 8}, "num_experts_per_tok"),
])
def test_ambiguous_or_unsupported_configuration_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        config(**overrides)


@pytest.mark.parametrize("layer_index", [0, 1])
@torch.inference_mode()
def test_actual_awq_plain_quantization_reaches_all_decoder_linears(layer_index):
    hf_model = model()
    _convert_model_with_defuser(DeepSeekV2QModel, hf_model, cleanup_original=True)
    wrapper = DeepSeekV2QModel(hf_model, quantized=False, quantize_config=QuantizeConfig())
    processor = AWQProcessor(tokenizer=None, qcfg=AWQConfig(device="cpu", group_size=-1),
                             calibration=CalibrationConfig(), gptq_model=wrapper, model=hf_model)
    processor.draw_progress = lambda *a, **kw: None
    processor.log_new_row = lambda *a, **kw: None
    state = processor._get_layer_state(layer_index)
    state.layer_module = hf_model.model.layers[layer_index]
    linears = {
        name: NamedModule(module, name, f"model.layers.{layer_index}.{name}", layer_index)
        for name, module in state.layer_module.named_modules() if isinstance(module, torch.nn.Linear)
    }
    state.modules.update(linears)
    before = {name: item.weight.clone() for name, item in linears.items()}
    dense = {key: value.clone() for key, value in state.layer_module.state_dict().items() if key.rsplit(".", 1)[0] not in linears}
    processor._quantize_layer(layer_index, state)
    assert state.quantized and not state.modules
    for name, item in linears.items():
        assert "q_scales" in item.state and "q_zeros" in item.state
        assert not torch.equal(item.weight, before[name]), name
    for key, value in dense.items():
        torch.testing.assert_close(hf_model.model.layers[layer_index].state_dict()[key], value, rtol=0, atol=0)
    assert torch.isfinite(hf_model(torch.tensor([[1, 3, 6]])).logits).all()
