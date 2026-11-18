# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import itertools
import json

import pytest
import torch
from safetensors import SafetensorError
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache, StaticCache

from gptqmodel import CalibrationConfig
from gptqmodel.models.definitions.mobilellm import MobileLLMQModel
from gptqmodel.models.local import LocalModelLoader


def test_definition_owns_model_without_checkpoint_python():
    assert isinstance(MobileLLMQModel.loader, LocalModelLoader)
    assert not MobileLLMQModel.require_trust_remote_code


@pytest.fixture
def tiny_config():
    from gptqmodel.models.ports.mobilellm import MobileLLMConfig

    return MobileLLMConfig(vocab_size=48, hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                           max_position_embeddings=32, pad_token_id=0)


def _model(config, **changes):
    from gptqmodel.models.ports.mobilellm import MobileLLMForCausalLM

    config = copy.deepcopy(config)
    for name, value in changes.items():
        setattr(config, name, value)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        model = MobileLLMForCausalLM(config).eval()
    return model


def _reference(model, ids, mask=None, positions=None, cache=None, rope_state=None):
    """Independent source equations; no port forward/cache/rotary helpers."""
    config, state = model.config, model.state_dict()
    x = F.embedding(ids, state["model.embed_tokens.weight"])
    batch, length = ids.shape
    offset = 0 if not cache else cache[0][0].shape[1]
    positions = torch.arange(offset, offset + length)[None] if positions is None else positions

    def norm(x, weight):
        wide = x.float()
        return (wide / (wide.square().mean(-1, keepdim=True) + config.rms_norm_eps).sqrt()).to(x.dtype) * weight

    def project(x, prefix):
        bias = state.get(prefix + ".bias")
        if config.pretraining_tp > 1 and ("mlp." in prefix or config._attn_implementation == "eager"):
            bias = None
        return F.linear(x, state[prefix + ".weight"], bias)

    base = config.rope_theta if rope_state is None else rope_state.get("base", config.rope_theta)
    kind = config.rope_parameters["rope_type"]
    factor = config.rope_parameters.get("factor", 1)
    if kind == "linear":
        positions = positions.float() / factor
    if kind == "dynamic" and positions.max() + 1 > config.max_position_embeddings:
        base = config.rope_theta * (factor * (positions.max() + 1) / config.max_position_embeddings - factor + 1) ** (
            config.head_dim / (config.head_dim - 2))
        if rope_state is not None:
            rope_state["base"] = base
    angles = positions.float()[..., None] / base ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim)
    angles = torch.cat((angles, angles), -1).unsqueeze(1)

    def rotary(x):
        a, b = x.split(config.head_dim // 2, -1)
        return x * angles.cos().to(x.dtype) + torch.cat((-b, a), -1) * angles.sin().to(x.dtype)

    all_hidden = []
    for layer in range(config.num_hidden_layers):
        root = f"model.layers.{layer}."
        for repeat in range(2 if config.layer_sharing else 1):
            all_hidden.append(x)
            normalized = norm(x, state[root + "input_layernorm.weight"])
            q = project(normalized, root + "self_attn.q_proj").view(batch, length, config.num_attention_heads, -1)
            k = project(normalized, root + "self_attn.k_proj").view(batch, length, config.num_key_value_heads, -1)
            v = project(normalized, root + "self_attn.v_proj").view(batch, length, config.num_key_value_heads, -1)
            q = rotary(q.transpose(1, 2)).transpose(1, 2)
            k = rotary(k.transpose(1, 2)).transpose(1, 2)
            slot = layer * (2 if config.layer_sharing else 1) + repeat
            if cache is not None:
                if slot in cache:
                    k, v = torch.cat((cache[slot][0], k), 1), torch.cat((cache[slot][1], v), 1)
                cache[slot] = k, v
            heads = []
            for head in range(config.num_attention_heads):
                kv = head // (config.num_attention_heads // config.num_key_value_heads)
                scores = torch.einsum("bqd,bkd->bqk", q[:, :, head], k[:, :, kv])
                scores /= config.head_dim ** 0.5
                allowed = (torch.arange(offset, offset + length)[:, None] >= torch.arange(k.shape[1])[None])[None]
                if mask is not None:
                    allowed = allowed & mask[:, None].bool()
                scores = scores.masked_fill(~allowed, torch.finfo(x.dtype).min)
                probabilities = scores.float().softmax(-1).to(x.dtype)
                heads.append(probabilities @ v[:, :, kv])
            attended = torch.stack(heads, dim=2).flatten(2)
            x = x + project(attended, root + "self_attn.o_proj")
            normalized = norm(x, state[root + "post_attention_layernorm.weight"])
            assert config.hidden_act == "silu"
            activated = F.silu(project(normalized, root + "mlp.gate_proj"))
            x = x + project(activated * project(normalized, root + "mlp.up_proj"), root + "mlp.down_proj")
    x = norm(x, state["model.norm.weight"])
    all_hidden.append(x)
    head = "model.embed_tokens.weight" if config.share_embedding else "lm_head.weight"
    return F.linear(x, state[head]).float(), all_hidden


@pytest.mark.parametrize("kv,bias,sharing,shared_embedding", list(itertools.product([1, 2, 4], [False, True], [False, True], [False, True])))
@torch.inference_mode()
def test_source_equations_all_layouts(tiny_config, kv, bias, sharing, shared_embedding):
    model = _model(tiny_config, num_key_value_heads=kv, attention_bias=bias, mlp_bias=bias,
                   layer_sharing=sharing, share_embedding=shared_embedding, _attn_implementation="eager")
    # Nonuniform gains/biases make the oracle sensitive to omitted norms and branches.
    for name, parameter in model.named_parameters():
        if "layernorm.weight" in name:
            parameter.copy_(torch.linspace(0.4, 1.6, parameter.numel()))
        elif name.endswith("bias"):
            parameter.copy_(torch.linspace(-0.1, 0.1, parameter.numel()))
    ids = torch.tensor([[0, 1, 7, 9, 11], [1, 3, 4, 0, 0]])
    mask = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    expected, hidden = _reference(model, ids, mask)
    actual = model(ids, attention_mask=mask, use_cache=False, output_attentions=True, output_hidden_states=True)
    # All-padding query rows are convention-dependent; valid query rows are not.
    torch.testing.assert_close(actual.logits[mask.bool()], expected[mask.bool()], rtol=2e-5, atol=2e-6)
    assert len(actual.hidden_states) == len(hidden) == tiny_config.num_hidden_layers * (1 + sharing) + 1
    assert len(actual.attentions) == len(hidden) - 1
    for weights in actual.attentions:
        assert torch.count_nonzero(weights[:, :, 1:, :].triu(2)) == 0


@pytest.mark.parametrize("kind", ["default", "linear", "dynamic"])
@pytest.mark.parametrize("sharing", [False, True])
@torch.inference_mode()
def test_rope_source_equations_and_save_reload(tiny_config, tmp_path, kind, sharing):
    from gptqmodel.models.ports.mobilellm import MobileLLMConfig, MobileLLMForCausalLM

    data = tiny_config.to_dict()
    data.pop("rope_parameters", None)
    data.update(rope_scaling=None if kind == "default" else {"type": kind, "factor": 2.0},
                max_position_embeddings=4, layer_sharing=sharing)
    model = _model(MobileLLMConfig(**data))
    ids = torch.tensor([[1, 3, 8, 7, 4, 9]])
    expected, _ = _reference(model, ids)
    torch.testing.assert_close(model(ids, use_cache=False).logits, expected, atol=2e-6, rtol=2e-5)
    model.save_pretrained(tmp_path)
    assert list(tmp_path.glob("*.safetensors")) and not list(tmp_path.glob("*.bin"))
    reloaded = MobileLLMForCausalLM.from_pretrained(tmp_path)
    torch.testing.assert_close(reloaded(ids, use_cache=False).logits, expected, atol=2e-6, rtol=2e-5)
    assert reloaded.config.rope_parameters == model.config.rope_parameters


@pytest.mark.parametrize("sharing,shared_embedding,implementation,dtype", list(itertools.product(
    [False, True], [False, True], ["eager", "sdpa"], [torch.float32, torch.bfloat16])))
@torch.inference_mode()
def test_cache_generation_causality_roundtrip(tiny_config, tmp_path, sharing, shared_embedding, implementation, dtype):
    from gptqmodel.models.ports.mobilellm import MobileLLMForCausalLM

    model = _model(tiny_config, layer_sharing=sharing, share_embedding=shared_embedding,
                   _attn_implementation=implementation).to(dtype)
    ids = torch.tensor([[1, 3, 8, 9, 7, 4], [1, 6, 2, 9, 11, 5]])
    whole = model(ids, use_cache=False).logits
    cache = None
    chunks = []
    offset = 0
    for count in [1, 4, 1]:
        out = model(ids[:, offset:offset + count], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        chunks.append(out.logits)
        offset += count
        assert all(layer.get_seq_length() == offset for layer in cache.layers)
        assert len(cache.layers) == tiny_config.num_hidden_layers * (1 + sharing)
    tolerance = 2e-3 if dtype == torch.bfloat16 else 2e-4 if dtype == torch.float16 else 2e-6
    torch.testing.assert_close(torch.cat(chunks, 1), whole, atol=tolerance, rtol=0)
    changed = ids.clone()
    changed[:, -1] = 17
    torch.testing.assert_close(model(changed, use_cache=False).logits[:, :-1], whole[:, :-1], atol=0, rtol=0)
    manual = ids[:, :3]
    for _ in range(3):
        manual = torch.cat((manual, model(manual, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)), 1)
    actual = model.generate(ids[:, :3], max_new_tokens=3, do_sample=False, eos_token_id=None)
    assert torch.equal(actual, manual)
    beam = model.generate(ids[:1, :3], max_new_tokens=2, num_beams=2, eos_token_id=None)
    uncached_beam = model.generate(ids[:1, :3], max_new_tokens=2, num_beams=2, eos_token_id=None, use_cache=False)
    assert beam.shape == (1, 5) and torch.equal(beam, uncached_beam)
    model.save_pretrained(tmp_path)
    for _ in range(2):
        reloaded = MobileLLMForCausalLM.from_pretrained(tmp_path, dtype=dtype, attn_implementation=implementation)
        assert set(reloaded.state_dict()) == set(model.state_dict())
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(reloaded.state_dict()[key], tensor, atol=0, rtol=0)
        torch.testing.assert_close(reloaded(ids, use_cache=False).logits, whole, atol=0, rtol=0)
        reloaded.save_pretrained(tmp_path)


def _tokenizer(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                                    **{f"t{i}": i for i in range(4, 48)}}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", bos_token="[BOS]",
                           eos_token="[EOS]", unk_token="[UNK]").save_pretrained(path)


@pytest.mark.parametrize("trust,bypass", [(False, None), (True, "0"), (True, "1")])
@pytest.mark.parametrize("sharing", [False, True])
def test_public_malicious_checkpoint_never_selects_python(tiny_config, tmp_path, monkeypatch, trust, bypass, sharing):
    from gptqmodel import GPTQModel
    from gptqmodel.models.ports.mobilellm import MobileLLMConfig, MobileLLMForCausalLM
    from gptqmodel.quantization.config import AWQConfig
    from gptqmodel.utils.hf import load_model_config

    model = _model(tiny_config, layer_sharing=sharing)
    model.save_pretrained(tmp_path)
    _tokenizer(tmp_path)
    marker = tmp_path / "EXECUTED"
    (tmp_path / "hostile.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('hostile')\n")
    path = tmp_path / "config.json"
    data = json.loads(path.read_text())
    data["auto_map"] = {"AutoConfig": "hostile.Config", "AutoModelForCausalLM": "hostile.Model"}
    path.write_text(json.dumps(data))
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config) is MobileLLMConfig
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=AWQConfig(device="cpu", group_size=16, offload_to_disk=False),
                            dtype=torch.float32, trust_remote_code=trust)
    assert type(wrapper.model) is MobileLLMForCausalLM
    ids = torch.tensor([[1, 4, 5]])
    with torch.inference_mode():
        torch.testing.assert_close(wrapper.model(ids).logits, model(ids).logits, rtol=0, atol=0)
    assert not marker.exists()


@torch.inference_mode()
@pytest.mark.parametrize("failure", ["decoder", "base_decoder", "head", "loss"])
@pytest.mark.parametrize("sharing", [False, True])
@pytest.mark.parametrize("warm", [False, True])
def test_failure_after_cache_write_restores_state_and_retry(tiny_config, failure, sharing, warm):
    model = _model(tiny_config, layer_sharing=sharing, share_embedding=False)
    ids = torch.tensor([[1, 3, 4, 6]])
    cache = model(ids[:, :2]).past_key_values if warm else DynamicCache()
    prefix_length = 2 if warm else 0
    layer_list = cache.layers
    original_layers = list(cache.layers)
    original_state = [vars(layer).copy() for layer in cache.layers]
    original_tensors = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    observed_lengths = []

    def after_write(module, args, output):
        observed_lengths.append([layer.get_seq_length() for layer in cache.layers])
        if failure != "loss":
            raise RuntimeError("injected after cache write")

    target = model.model if failure == "base_decoder" else model
    module = (model.model.norm if failure == "loss" else model.lm_head if failure == "head"
              else model.model.layers[0])
    handle = module.register_forward_hook(after_write)
    try:
        with pytest.raises(IndexError if failure == "loss" else RuntimeError,
                           match="out of bounds" if failure == "loss" else "injected after cache write"):
            target(ids[:, 2:], past_key_values=cache,
                   **({"labels": torch.tensor([[4, 99]])} if failure == "loss" else {}))
    finally:
        handle.remove()
    uses = tiny_config.num_hidden_layers * (2 if sharing else 1)
    expected_lengths = ([prefix_length + 2] * uses if failure in {"head", "loss"}
                        else [prefix_length + 2] + ([prefix_length] * (uses - 1) if warm else []))
    assert observed_lengths == [expected_lengths], "Failure must follow a completed cache write"
    assert cache.layers is layer_list
    assert len(cache.layers) == len(original_layers)
    for layer, original, state, (keys, values) in zip(
            cache.layers, original_layers, original_state, original_tensors):
        assert layer is original
        assert vars(layer).keys() == state.keys()
        assert all(vars(layer)[name] is value for name, value in state.items())
        torch.testing.assert_close(layer.keys, keys, atol=0, rtol=0)
        torch.testing.assert_close(layer.values, values, atol=0, rtol=0)
    retry = model(ids[:, 2:], past_key_values=cache)
    expected_ids = ids if warm else ids[:, 2:]
    expected, _ = _reference(model, expected_ids)
    torch.testing.assert_close(retry.logits, expected[:, -2:], atol=2e-6, rtol=1e-5)
    assert retry.past_key_values is cache
    assert [layer.get_seq_length() for layer in cache.layers] == [prefix_length + 2] * uses


@torch.inference_mode()
@pytest.mark.parametrize("failure", ["decoder", "loss"])
@pytest.mark.parametrize("sharing", [False, True])
def test_failed_cache_extension_restores_dynamic_ntk_before_short_retry(tiny_config, failure, sharing):
    from gptqmodel.models.ports.mobilellm import MobileLLMConfig

    data = tiny_config.to_dict()
    data.pop("rope_parameters", None)
    data.update(rope_scaling={"type": "dynamic", "factor": 2.0}, max_position_embeddings=3,
                layer_sharing=sharing)
    model = _model(MobileLLMConfig(**data))
    ids = torch.tensor([[1, 3, 4, 6, 7, 8]])
    cache = model(ids[:, :2]).past_key_values
    rotary_state = [(layer.self_attn.rotary_emb._base, layer.self_attn.rotary_emb.inv_freq.clone())
                    for layer in model.model.layers]
    observed_bases = []

    def after_write(module, args, output):
        observed_bases.append(model.model.layers[0].self_attn.rotary_emb._base)
        if failure == "decoder":
            raise RuntimeError("injected after NTK cache write")

    module = model.model.layers[0] if failure == "decoder" else model.model.norm
    handle = module.register_forward_hook(after_write)
    try:
        with pytest.raises(IndexError if failure == "loss" else RuntimeError,
                           match="out of bounds" if failure == "loss" else "injected after NTK cache write"):
            model(ids[:, 2:], past_key_values=cache,
                  **({"labels": torch.tensor([[4, 99, 7, 8]])} if failure == "loss" else {}))
    finally:
        handle.remove()
    assert len(observed_bases) == 1 and observed_bases[0] > tiny_config.rope_theta
    for layer, (base, frequencies) in zip(model.model.layers, rotary_state):
        assert layer.self_attn.rotary_emb._base == base
        torch.testing.assert_close(layer.self_attn.rotary_emb.inv_freq, frequencies, atol=0, rtol=0)
    assert [layer.get_seq_length() for layer in cache.layers] == [2] * len(cache.layers)
    expected, _ = _reference(model, ids[:, :3])
    actual = model(ids[:, 2:3], past_key_values=cache)
    torch.testing.assert_close(actual.logits, expected[:, -1:], atol=2e-6, rtol=1e-5)


@torch.inference_mode()
def test_failure_before_cache_mutation_then_repair_retry(tiny_config, tmp_path):
    from gptqmodel.models.ports.mobilellm import MobileLLMForCausalLM

    model = _model(tiny_config, layer_sharing=True)
    ids = torch.tensor([[1, 7, 9]])
    output = model(ids[:, :1])
    cache = output.past_key_values
    before = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    with pytest.raises(ValueError, match="attention_mask"):
        model(ids[:, 1:], past_key_values=cache, attention_mask=torch.ones(1, 2))
    for layer, (keys, values) in zip(cache.layers, before):
        torch.testing.assert_close(layer.keys, keys, atol=0, rtol=0)
        torch.testing.assert_close(layer.values, values, atol=0, rtol=0)
    actual = model(ids[:, 1:], past_key_values=cache, attention_mask=torch.ones(1, 3))
    torch.testing.assert_close(actual.logits, model(ids, use_cache=False).logits[:, 1:], atol=2e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="DynamicCache"):
        model(ids, past_key_values=StaticCache(config=model.config, max_cache_len=16))
    model.save_pretrained(tmp_path)
    original = (tmp_path / "model.safetensors").read_bytes()
    (tmp_path / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(SafetensorError):
        MobileLLMForCausalLM.from_pretrained(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(original)
    retry = MobileLLMForCausalLM.from_pretrained(tmp_path)
    torch.testing.assert_close(retry(ids).logits, model(ids).logits, atol=0, rtol=0)


@pytest.mark.parametrize("assign", [False, True])
def test_state_assignment_into_warm_and_fresh_owners(tiny_config, assign):
    source = _model(tiny_config, layer_sharing=True)
    target = _model(tiny_config, layer_sharing=True)
    ids = torch.tensor([[1, 5, 7]])
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.mul_(2)
        target(ids)
        target.load_state_dict(source.state_dict(), assign=assign)
        torch.testing.assert_close(target(ids).logits, source(ids).logits, atol=0, rtol=0)


@pytest.mark.parametrize("bad", [{"num_key_value_heads": 3}, {"head_dim": 3},
                                 {"rope_scaling": {"type": "unknown", "factor": 2}},
                                 {"rope_scaling": {"type": "linear", "factor": 0.5}},
                                 {"pretraining_tp": 7}, {"num_key_value_heads": 0},
                                 {"num_attention_heads": 0}, {"hidden_size": 0},
                                 {"num_hidden_layers": -1}, {"head_dim": 0}, {"rope_theta": 0},
                                 {"attention_dropout": 1.1}])
def test_invalid_config_fails_closed(bad):
    from gptqmodel.models.ports.mobilellm import MobileLLMConfig

    options = {"hidden_size": 32, "num_attention_heads": 4, "num_key_value_heads": 2}
    options.update(bad)
    with pytest.raises(ValueError):
        MobileLLMConfig(**options)


@pytest.mark.parametrize("sharing", [False, True])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("kind", ["default", "linear", "dynamic"])
@torch.inference_mode()
def test_independent_rotated_cache_oracle_crosses_original_context(tiny_config, sharing, implementation, kind):
    from gptqmodel.models.ports.mobilellm import MobileLLMConfig

    data = tiny_config.to_dict()
    data.pop("rope_parameters", None)
    data.update(rope_scaling=None if kind == "default" else {"type": kind, "factor": 2.0},
                max_position_embeddings=3, layer_sharing=sharing, _attn_implementation=implementation)
    model = _model(MobileLLMConfig(**data))
    ids = torch.tensor([[0, 1, 9, 4, 7, 3, 8], [1, 5, 11, 13, 9, 22, 4]])
    mask = (ids != 0).long()
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 1)
    expected_cache, rope_state = {}, {}
    cache = None
    offset = 0
    for count in [1, 4, 1, 1]:
        end = offset + count
        expected, _ = _reference(model, ids[:, offset:end], mask[:, :end], positions[:, offset:end],
                                 cache=expected_cache, rope_state=rope_state)
        actual = model(ids[:, offset:end], attention_mask=mask[:, :end], position_ids=positions[:, offset:end],
                       past_key_values=cache, use_cache=True)
        cache = actual.past_key_values
        valid = mask[:, offset:end].bool()
        torch.testing.assert_close(actual.logits[valid], expected[valid], rtol=2e-5, atol=2e-6)
        assert len(cache.layers) == model.config.num_hidden_layers * (1 + sharing)
        for slot, (keys, values) in expected_cache.items():
            torch.testing.assert_close(cache.layers[slot].keys, keys.transpose(1, 2), rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(cache.layers[slot].values, values.transpose(1, 2), rtol=2e-5, atol=2e-6)
        offset = end
    # The source retains its last dynamic NTK frequencies on a later short request.
    expected, _ = _reference(model, ids[:, 1:3], rope_state=rope_state)
    torch.testing.assert_close(model(ids[:, 1:3], use_cache=False).logits, expected, rtol=2e-5, atol=2e-6)
    if kind == "dynamic":
        assert rope_state["base"] > model.config.rope_theta
        assert model.model.layers[0].self_attn.rotary_emb._base == rope_state["base"].item()


@pytest.mark.parametrize("implementation,bias,shared_embedding", list(itertools.product(
    ["eager", "sdpa"], [False, True], [False, True])))
@torch.inference_mode()
def test_pretraining_tp_source_slices_and_bias_semantics(tiny_config, tmp_path, implementation, bias, shared_embedding):
    model = _model(tiny_config, pretraining_tp=2, layer_sharing=True, share_embedding=shared_embedding,
                   attention_bias=bias, mlp_bias=bias, _attn_implementation=implementation)
    for name, parameter in model.named_parameters():
        if name.endswith(".bias"):
            parameter.copy_(torch.linspace(-0.2, 0.3, parameter.numel()))
    ids = torch.tensor([[1, 7, 13, 8]])
    expected, _ = _reference(model, ids)
    torch.testing.assert_close(model(ids, use_cache=False).logits, expected, atol=2e-6, rtol=2e-5)
    prefix = model(ids[:, :2])
    torch.testing.assert_close(model(ids[:, 2:], past_key_values=prefix.past_key_values).logits,
                               expected[:, 2:], atol=2e-6, rtol=2e-5)
    model.save_pretrained(tmp_path)
    reloaded = MobileLLMQModel.loader.from_pretrained(tmp_path, attn_implementation=implementation)
    torch.testing.assert_close(reloaded(ids, use_cache=False).logits, expected, atol=2e-6, rtol=2e-5)


def test_shared_physical_cache_rejected_before_any_mutation_then_retry(tiny_config):
    model = _model(tiny_config, layer_sharing=True)
    cache = DynamicCache(config=model.config)
    with pytest.raises(ValueError, match="one slot for each decoder use"):
        model(torch.tensor([[1, 4]]), past_key_values=cache)
    assert all(layer.get_seq_length() == 0 for layer in cache.layers)
    out = model(torch.tensor([[1, 4]]), past_key_values=DynamicCache())
    assert len(out.past_key_values.layers) == 4


@pytest.mark.parametrize("shared_embedding", [False, True])
def test_loss_embeds_output_selection_and_head_replacement(tiny_config, shared_embedding):
    model = _model(tiny_config, share_embedding=shared_embedding, layer_sharing=True)
    ids = torch.tensor([[1, 7, 13, 8]])
    expected, _ = _reference(model, ids)
    result = model(ids, labels=ids, output_hidden_states=True, output_attentions=True)
    loss = F.cross_entropy(expected[:, :-1].reshape(-1, 48), ids[:, 1:].reshape(-1))
    torch.testing.assert_close(result.loss, loss)
    assert len(result.hidden_states) == 5 and len(result.attentions) == 4
    torch.testing.assert_close(model(inputs_embeds=model.get_input_embeddings()(ids)).logits, expected,
                               atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(model(ids, logits_to_keep=2).logits, expected[:, -2:], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(model(ids, return_dict=False)[0], expected, atol=2e-6, rtol=2e-5)
    replacement = torch.nn.Embedding(48, 32) if shared_embedding else torch.nn.Linear(32, 48, bias=False)
    model.set_output_embeddings(replacement)
    assert model.get_output_embeddings() is replacement
    assert shared_embedding == (model.get_input_embeddings() is replacement)


def test_public_pretraining_tp_quantization_rejects_dense_slice_path_before_packing(tiny_config, tmp_path):
    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import AWQConfig
    from gptqmodel.utils.backend import BACKEND

    model = _model(tiny_config, pretraining_tp=2).half()
    model.save_pretrained(tmp_path)
    _tokenizer(tmp_path)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None),
                             backend=BACKEND.TORCH, dtype=torch.float16)
    with pytest.raises(ValueError, match="dense weight slices"):
        wrapper.quantize(
            CalibrationConfig(dataset=[{"input_ids": torch.tensor([[1, 4, 9, 13, 7, 8, 4, 5]])}], min_length=1),
            backend=BACKEND.TORCH,
        )
    assert all(isinstance(layer.mlp.down_proj, torch.nn.Linear) for layer in wrapper.model.model.layers)


@pytest.mark.parametrize("shared_embedding", [False, True])
def test_optional_standard_embedding_tie_safetensors(tiny_config, tmp_path, shared_embedding):
    model = _model(tiny_config, share_embedding=shared_embedding, tie_word_embeddings=True)
    if shared_embedding:
        assert not hasattr(model, "lm_head")
    else:
        assert model.lm_head.weight is model.model.embed_tokens.weight
    ids = torch.tensor([[1, 4, 9]])
    expected = model(ids, use_cache=False).logits
    model.save_pretrained(tmp_path)
    reloaded = MobileLLMQModel.loader.from_pretrained(tmp_path)
    assert reloaded.get_output_embeddings().weight is reloaded.model.embed_tokens.weight
    assert set(reloaded.state_dict()) == set(model.state_dict())
    torch.testing.assert_close(reloaded(ids, use_cache=False).logits, expected, rtol=0, atol=0)


def test_partial_checkpoint_keeps_loaded_weights_of_partially_missing_module(tiny_config, tmp_path):
    """Missing-key init must go through transformers.initialization guards."""
    from safetensors.torch import load_file, save_file

    model = _model(tiny_config, attention_bias=True)
    model.save_pretrained(tmp_path)
    state = load_file(tmp_path / "model.safetensors")
    missing = state.pop("model.layers.0.self_attn.q_proj.bias")
    save_file(state, tmp_path / "model.safetensors", metadata={"format": "pt"})
    reloaded = MobileLLMQModel.loader.from_pretrained(tmp_path)
    attention = reloaded.model.layers[0].self_attn
    torch.testing.assert_close(attention.q_proj.weight, state["model.layers.0.self_attn.q_proj.weight"], rtol=0, atol=0)
    assert torch.count_nonzero(attention.q_proj.bias) == 0 and missing.shape == attention.q_proj.bias.shape
    torch.testing.assert_close(reloaded.model.norm.weight, state["model.norm.weight"], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_rotary_frequency_stays_fp32_after_move_and_reload(tiny_config, tmp_path, dtype):
    model = _model(tiny_config).to(dtype)
    expected = 1 / 10000 ** (torch.arange(0, 8, 2).float() / 8)
    for layer in model.model.layers:
        assert layer.self_attn.rotary_emb.inv_freq.dtype == torch.float32
        torch.testing.assert_close(layer.self_attn.rotary_emb.inv_freq, expected, rtol=0, atol=0)
    ids = torch.tensor([[1, 4, 8, 9, 13, 2]])
    logits = model(ids).logits
    model.save_pretrained(tmp_path)
    reloaded = MobileLLMQModel.loader.from_pretrained(tmp_path, dtype=dtype)
    for layer in reloaded.model.layers:
        assert layer.self_attn.rotary_emb.inv_freq.dtype == torch.float32
        torch.testing.assert_close(layer.self_attn.rotary_emb.inv_freq, expected, rtol=0, atol=0)
    torch.testing.assert_close(reloaded(ids).logits, logits, rtol=0, atol=0)


@pytest.mark.parametrize("sharing,shared_embedding,offload,kv,bias", [
    (False, False, False, 2, False), (True, False, True, 1, True),
    (False, True, True, 4, True), (True, True, False, 2, False)])
def test_public_awq_scale_search_pack_infer_two_safetensors_cycles(tiny_config, tmp_path, sharing, shared_embedding, offload, kv, bias):
    from gptqmodel import GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig
    from gptqmodel.utils.backend import BACKEND

    source = tmp_path / "source"
    native = _model(tiny_config, layer_sharing=sharing, share_embedding=shared_embedding,
                    num_key_value_heads=kv, attention_bias=bias, mlp_bias=bias).half()
    native.save_pretrained(source)
    _tokenizer(source)
    native_state = {key: value.clone() for key, value in native.state_dict().items()}
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=offload)
    wrapper = GPTQModel.load(str(source), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16)
    rows = [{"input_ids": torch.tensor([[1, 4, 9, 13, 7, 5, 19, 8]]),
             "attention_mask": torch.ones(1, 8, dtype=torch.long)},
            {"input_ids": torch.tensor([[1, 17, 7, 6, 11, 22, 8, 4]]),
             "attention_mask": torch.ones(1, 8, dtype=torch.long)}]
    wrapper.quantize(
        CalibrationConfig(dataset=rows, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    packed = {name: module for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert len(packed) == 14
    expected_linears = {f"model.layers.{index}.{name}" for index in range(2) for name in
                        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]}
    assert set(packed) == expected_linears
    for name, module in packed.items():
        assert module.qweight.numel() > 0 and torch.count_nonzero(module.qweight) > 0, name
        assert "weight" not in module._parameters
    with torch.inference_mode():
        expected = wrapper.model(**rows[0], use_cache=False).logits
        assert torch.isfinite(expected).all()
        baseline = native(**rows[0], use_cache=False).logits
        assert (expected - baseline).abs().max() < 0.08
        assert wrapper.model.generate(rows[0]["input_ids"][:, :3], max_new_tokens=2, eos_token_id=None).shape == (1, 5)
        prefix = wrapper.model(rows[0]["input_ids"][:, :3])
        continued = wrapper.model(rows[0]["input_ids"][:, 3:], past_key_values=prefix.past_key_values)
        torch.testing.assert_close(continued.logits, expected[:, 3:], rtol=0.002, atol=0.002)
    snapshot = {key: value.clone() for key, value in wrapper.model.state_dict().items()}
    dense_keys = {key for key in native_state if not any(key.startswith(name + ".") for name in packed)}
    # Scale search folds the q/k/v and gate/up input scales into the decoder norms.
    folded = {key for key in dense_keys if key.startswith("model.layers.") and "layernorm" in key}
    assert any(not torch.equal(snapshot[key], native_state[key]) for key in folded)
    for key in dense_keys - folded:
        torch.testing.assert_close(snapshot[key], native_state[key], atol=0, rtol=0)
    for cycle in range(2):
        destination = tmp_path / f"quantized-{cycle}"
        wrapper.save(str(destination))
        assert list(destination.glob("*.safetensors")) and not list(destination.glob("*.bin"))
        wrapper = GPTQModel.load(str(destination), backend=BACKEND.TORCH, dtype=torch.float16)
        assert set(wrapper.model.state_dict()) == set(snapshot)
        for key, value in snapshot.items():
            torch.testing.assert_close(wrapper.model.state_dict()[key], value, rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(wrapper.model(**rows[0], use_cache=False).logits, expected, rtol=0, atol=0)
