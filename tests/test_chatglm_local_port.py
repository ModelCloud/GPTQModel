# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""ChatGLM checkpoint ownership, independent numerics, cache and real loading."""

import copy
import json

import pytest
import torch
from torch.nn import functional as F

from gptqmodel.models.definitions.chatglm import ChatGLMQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.utils.hf import load_model_config


@pytest.fixture(autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config(**kwargs):
    from gptqmodel.models.ports.chatglm.configuration_chatglm import ChatGLMConfig
    values = {"num_layers": 2, "hidden_size": 32, "ffn_hidden_size": 48, "kv_channels": 8,
              "num_attention_heads": 4, "padded_vocab_size": 64, "seq_length": 32,
              "multi_query_attention": True, "multi_query_group_num": 2,
              "add_qkv_bias": True, "pad_token_id": 0, "eos_token_id": 2, "tie_word_embeddings": False}
    values.update(kwargs)
    return ChatGLMConfig(**values)


def _model(**kwargs):
    from gptqmodel.models.ports.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(382)
        return ChatGLMForConditionalGeneration(_config(**kwargs)).eval()


def _poison(path, payload=None):
    marker = path / "CHECKPOINT_PYTHON_EXECUTED"
    if payload is None:
        payload = json.loads((path / "config.json").read_text())
    payload["auto_map"] = {"AutoConfig": "checkpoint_code.UntrustedConfig",
                           "AutoModelForCausalLM": "checkpoint_code.UntrustedModel"}
    (path / "config.json").write_text(json.dumps(payload))
    (path / "checkpoint_code.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python executed')\n"
    )
    return marker


def _bypass(monkeypatch, enabled):
    # This is the existing security access flag being tested, not a new switch.
    monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    if enabled:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_chatglm_definition_owns_config_and_model(tmp_path, monkeypatch, trust, bypass):
    assert isinstance(ChatGLMQModel.loader, LocalModelLoader)
    _bypass(monkeypatch, bypass)
    marker = _poison(tmp_path, _config().to_dict())
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config).__module__ == "gptqmodel.models.ports.chatglm.configuration_chatglm"
    with torch.device("meta"):
        model = ChatGLMQModel.loader.from_config(config, trust_remote_code=trust)
    assert type(model).__module__ == "gptqmodel.models.ports.chatglm.modeling_chatglm"
    assert model.transformer.encoder.layers[0].self_attention.query_key_value.weight.is_meta
    assert not marker.exists()


def _oracle(weights, config, ids, positions, padding):
    """Independent functional forward from the pinned upstream equations.

    All tensors use batch/head/sequence coordinates here; RoPE uses complex
    multiplication, rather than either port's paired-real helper. No port
    modules, attention, masking or cache utilities supply expected values.
    """
    batch, length = ids.shape
    heads, dim, groups = config.num_attention_heads, config.kv_channels, config.num_key_value_heads

    def linear(x, name):
        return F.linear(x, weights[name + ".weight"], weights.get(name + ".bias"))

    def norm(x, name):
        if not config.rmsnorm:
            return F.layer_norm(x, (config.hidden_size,), weights[name + ".weight"],
                                weights[name + ".bias"], config.layernorm_epsilon)
        value = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + config.layernorm_epsilon)
        return (value * weights[name + ".weight"]).to(x.dtype)

    def rotate(x):
        rotary_dim = dim // 2
        frequency = (10000.0 * (config.rope_ratio or 1)) ** (
            -torch.arange(0, rotary_dim, 2).float() / rotary_dim
        )
        angles = positions.float()[..., None] * frequency
        # Upstream rounds the trig cache to the model dtype before multiplication.
        cosine = angles.cos().to(x.dtype).float()
        sine = angles.sin().to(x.dtype).float()
        factor = torch.complex(cosine, sine)[:, None]
        paired = torch.view_as_complex(x[..., :rotary_dim].float().reshape(batch, groups if x.shape[1] == groups else heads,
                                                                            length, rotary_dim // 2, 2))
        rotated = torch.view_as_real(paired * factor).flatten(-2).to(x.dtype)
        return torch.cat((rotated, x[..., rotary_dim:]), -1)

    hidden = F.embedding(ids, weights["transformer.embedding.word_embeddings.weight"])
    cache = []
    allowed = torch.arange(length)[None, :] <= torch.arange(length)[:, None]
    allowed = allowed[None, None] & padding[:, None, None, :].bool()
    allowed = allowed | ~padding[:, None, :, None].bool()
    for index in range(config.num_layers):
        prefix = f"transformer.encoder.layers.{index}."
        normalized = norm(hidden, prefix + "input_layernorm")
        packed = linear(normalized, prefix + "self_attention.query_key_value")
        if config.multi_query_attention:
            q, k, v = packed.split((heads * dim, groups * dim, groups * dim), -1)
            q = q.reshape(batch, length, heads, dim).transpose(1, 2)
            k, v = [t.reshape(batch, length, groups, dim).transpose(1, 2) for t in (k, v)]
        else:
            q, k, v = [t.transpose(1, 2) for t in packed.reshape(batch, length, heads, 3 * dim).chunk(3, -1)]
        q, k = rotate(q), rotate(k)
        cache.append((k, v))
        k, v = [t.repeat_interleave(heads // groups, dim=1) for t in (k, v)]
        scores = (q.float() @ k.float().transpose(-1, -2)) / dim ** 0.5
        probabilities = scores.masked_fill(~allowed, float("-inf")).softmax(-1).to(v.dtype)
        context = (probabilities @ v).transpose(1, 2).reshape(batch, length, heads * dim)
        attention = linear(context, prefix + "self_attention.dense")
        hidden = attention + (normalized if config.apply_residual_connection_post_layernorm else hidden)
        normalized = norm(hidden, prefix + "post_attention_layernorm")
        gate, up = linear(normalized, prefix + "mlp.dense_h_to_4h").chunk(2, -1)
        output = linear(F.silu(gate) * up, prefix + "mlp.dense_4h_to_h")
        hidden = output + (normalized if config.apply_residual_connection_post_layernorm else hidden)
    if config.post_layer_norm:
        hidden = norm(hidden, "transformer.encoder.final_layernorm")
    return linear(hidden, "transformer.output_layer"), cache


@pytest.mark.parametrize("rope_ratio", [None, 500])
@pytest.mark.parametrize("multi_query", [False, True])
@pytest.mark.parametrize("post_residual", [False, True])
@pytest.mark.parametrize("rmsnorm", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_chatglm_native_equations_and_cache_axes(rope_ratio, multi_query, post_residual, rmsnorm, dtype):
    model = _model(rope_ratio=rope_ratio, multi_query_attention=multi_query,
                   apply_residual_connection_post_layernorm=post_residual, rmsnorm=rmsnorm).to(dtype)
    ids = torch.tensor([[0, 4, 8, 11, 6, 9], [3, 7, 5, 6, 8, 4]])
    padding = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    positions = torch.tensor([[0, 2, 4, 5, 7, 9], [1, 3, 5, 7, 8, 10]])
    tolerance = 4e-6 if dtype == torch.float32 else (0.002 if dtype == torch.float16 else 0.012)
    with torch.no_grad():
        expected, expected_cache = _oracle(model.state_dict(), model.config, ids, positions, padding)
        actual = model(ids, position_ids=positions, attention_mask=padding)
        torch.testing.assert_close(actual.logits, expected, atol=tolerance, rtol=tolerance)
        for layer, (keys, values) in zip(actual.past_key_values.layers, expected_cache):
            torch.testing.assert_close(layer.keys, keys, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(layer.values, values, atol=tolerance, rtol=tolerance)
        # Test both one-token and multi-token continuation with left padding.
        cache = None
        for start, end in ((0, 3), (3, 4), (4, 6)):
            continuation = model(ids[:, start:end], position_ids=positions[:, start:end],
                                 attention_mask=padding[:, :end], past_key_values=cache)
            cache = continuation.past_key_values
            assert cache.get_seq_length() == end
            if start:
                torch.testing.assert_close(continuation.logits, actual.logits[:, start:end],
                                           atol=tolerance, rtol=tolerance)
        assert actual.past_key_values.layers[0].keys.shape == (
            2, model.config.num_key_value_heads, 6, 8
        )


@pytest.mark.parametrize("rope_ratio", [None, 500])
@pytest.mark.parametrize("rmsnorm", [False, True])
def test_chatglm_generation_safetensors_roundtrip(tmp_path, rope_ratio, rmsnorm):
    model = _model(rope_ratio=rope_ratio, rmsnorm=rmsnorm, add_bias_linear=True, post_layer_norm=False)
    ids = torch.tensor([[4, 7, 9], [5, 6, 11]])
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        cached = model.generate(ids, max_new_tokens=3, do_sample=False, eos_token_id=None)
        uncached = model.generate(ids, max_new_tokens=3, do_sample=False, use_cache=False, eos_token_id=None)
        assert torch.equal(cached, uncached)
        beam = model.generate(ids, max_new_tokens=3, num_beams=3, do_sample=False, eos_token_id=None)
        assert beam.shape == (2, 6)
        embeds = model.get_input_embeddings()(ids)
        torch.testing.assert_close(model(inputs_embeds=embeds).logits, expected)
    model.save_pretrained(tmp_path)
    assert list(tmp_path.glob("*.safetensors")) and not list(tmp_path.glob("*.bin"))
    loaded = ChatGLMQModel.loader.from_pretrained(tmp_path).eval()
    assert set(loaded.state_dict()) == set(model.state_dict())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids).logits, expected, rtol=0, atol=0)
    loaded.save_pretrained(tmp_path / "second")
    again = ChatGLMQModel.loader.from_pretrained(tmp_path / "second").eval()
    with torch.no_grad():
        torch.testing.assert_close(again(ids).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rope_ratio", [None, 500])
def test_chatglm_failed_continuation_preserves_cache_for_retry(monkeypatch, rope_ratio):
    model = _model(rope_ratio=rope_ratio)
    ids = torch.tensor([[4, 8, 11, 7, 9]])
    with torch.no_grad():
        cache = model(ids[:, :3]).past_key_values
        original = copy.deepcopy(cache)
        layer = model.transformer.encoder.layers[1]
        forward = layer.forward
        def fail_after_first_layer(*args, **kwargs):
            raise RuntimeError("injected second-layer failure")
        monkeypatch.setattr(layer, "forward", fail_after_first_layer)
        with pytest.raises(RuntimeError, match="injected second-layer"):
            model(ids[:, 3:], past_key_values=cache)
        monkeypatch.setattr(layer, "forward", forward)
        assert cache.get_seq_length() == 3
        for current, before in zip(cache.layers, original.layers):
            torch.testing.assert_close(current.keys, before.keys, rtol=0, atol=0)
            torch.testing.assert_close(current.values, before.values, rtol=0, atol=0)
        with pytest.raises(ValueError, match="cached and current"):
            model(ids[:, 3:], past_key_values=cache, attention_mask=torch.ones(1, 2))
        retry = model(ids[:, 3:], past_key_values=cache)
        torch.testing.assert_close(retry.logits, model(ids).logits[:, 3:], atol=2e-7, rtol=2e-5)
        model(ids[:, 3:], past_key_values=cache, use_cache=False)
        assert cache.get_seq_length() == 3


@pytest.mark.parametrize("rope_ratio", [None, 500])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_chatglm_cache_generation_continuation_and_native_layout(rope_ratio, implementation):
    model = _model(rope_ratio=rope_ratio, attn_implementation=implementation)
    ids = torch.tensor([[4, 8, 11, 9], [5, 7, 3, 6]])
    observed = []
    handle = model.transformer.encoder.layers[0].register_forward_pre_hook(
        lambda module, args, kwargs: observed.append((args[0].shape, dict(kwargs))), with_kwargs=True
    )
    try:
        with torch.no_grad():
            cache = model(ids[:, :2]).past_key_values
            expected = model.generate(ids, max_new_tokens=3, eos_token_id=None)
            continued = model.generate(ids[:, 2:], past_key_values=cache,
                                       attention_mask=torch.ones(2, 4, dtype=torch.long),
                                       max_new_tokens=3, eos_token_id=None)
            assert torch.equal(continued[:, 2:], expected[:, 4:])
            cache = model(ids[:, :2]).past_key_values
            cache.reorder_cache(torch.tensor([1, 0, 1]))
            assert cache.layers[0].keys.shape[0] == 3
            reordered = model(ids[[1, 0, 1], 2:], past_key_values=cache)
            torch.testing.assert_close(reordered.logits, model(ids[[1, 0, 1]]).logits[:, 2:],
                                       atol=2e-7, rtol=2e-5)
    finally:
        handle.remove()
    shape, kwargs = observed[0]
    assert shape == (2, 2, 32)
    assert "rotary_pos_emb" in kwargs and "attention_mask" in kwargs
    # The next prefill is deliberately non-square in batch/sequence dimensions.
    assert observed[1][0] == ((2, 4, 32) if rope_ratio else (4, 2, 32))


def test_chatglm_config_aliases_and_unsupported_native_modes():
    from gptqmodel.models.ports.chatglm.configuration_chatglm import ChatGLMConfig
    assert ChatGLMConfig(num_hidden_layers=2).num_layers == 2
    with pytest.raises(ValueError, match="conflicting ChatGLM"):
        ChatGLMConfig(num_layers=2, num_hidden_layers=3)
    with pytest.raises(ValueError, match="quantization_bit"):
        ChatGLMConfig(quantization_bit=4)
    with pytest.raises(ValueError, match="divide"):
        _config(multi_query_group_num=3)


@pytest.mark.parametrize("rope_ratio", [None, 500])
def test_chatglm_checkpointed_training_disables_cache_and_backpropagates(rope_ratio):
    model = _model(rope_ratio=rope_ratio).train()
    model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
    ids = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 4]])
    output = model(ids, labels=ids, use_cache=True)
    assert output.past_key_values is None
    output.loss.backward()
    qkv = model.transformer.encoder.layers[0].self_attention.query_key_value
    assert torch.isfinite(qkv.weight.grad).all() and qkv.weight.grad.norm() > 0


def _write_tokenizer(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2,
                                   **{str(i): i for i in range(3, 64)}}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                           eos_token="[EOS]").save_pretrained(path)


@pytest.mark.parametrize("rope_ratio", [None, 500])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_chatglm_public_load_ignores_checkpoint_python(tmp_path, monkeypatch, rope_ratio, trust, bypass):
    from gptqmodel import GPTQModel
    model = _model(rope_ratio=rope_ratio)
    model.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    marker = _poison(tmp_path)
    _bypass(monkeypatch, bypass)
    loaded = GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert isinstance(loaded, ChatGLMQModel)
    assert type(loaded.model) is type(model)
    assert loaded.trust_remote_code is False
    assert not marker.exists()
    ids = torch.tensor([[4, 7, 9]])
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(ids).logits, model(ids).logits, atol=0, rtol=0)


@pytest.mark.parametrize("entry", ["load", "from_pretrained", "from_quantized"])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_chatglm1_public_load_rejects_incompatible_layout_without_remote_fallback(
    tmp_path, monkeypatch, entry, trust, bypass,
):
    from gptqmodel import GPTQModel
    marker = _poison(tmp_path, {"model_type": "chatglm", "position_encoding_2d": True,
                               "inner_hidden_size": 16384, "vocab_size": 130528})
    _bypass(monkeypatch, bypass)
    kwargs = {"quantize_config": None} if entry == "from_pretrained" else {}
    with pytest.raises(ValueError, match="Original ChatGLM1"):
        getattr(GPTQModel, entry)(str(tmp_path), trust_remote_code=trust, device="cpu", **kwargs)
    assert not marker.exists()


@pytest.mark.parametrize("rope_ratio", [None, 500])
@pytest.mark.parametrize("prefix_projection", [False, True])
@pytest.mark.parametrize("mqa", [False, True])
def test_chatglm_prefix_tuning_cache_continuation(rope_ratio, prefix_projection, mqa):
    model = _model(rope_ratio=rope_ratio, pre_seq_len=3, prefix_projection=prefix_projection,
                   multi_query_attention=mqa)
    ids = torch.tensor([[4, 7, 11, 8]])
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        first = model(ids[:, :2]).past_key_values
        assert first.get_seq_length() == 5
        actual = model(ids[:, 2:], past_key_values=first).logits
        torch.testing.assert_close(actual, expected[:, 2:], atol=2e-7, rtol=2e-5)
        cached = model.generate(ids, max_new_tokens=3, eos_token_id=None)
        uncached = model.generate(ids, max_new_tokens=3, use_cache=False, eos_token_id=None)
        assert torch.equal(cached, uncached)


@pytest.mark.cpu
@pytest.mark.parametrize("rope_ratio", [None, 500])
@pytest.mark.parametrize("post_residual", [False, True])
@pytest.mark.parametrize("mqa", [False, True])
def test_chatglm_plain_awq_quantizes_all_linears_and_reloads(tmp_path, rope_ratio, post_residual, mqa):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig
    model = _model(rope_ratio=rope_ratio, apply_residual_connection_post_layernorm=post_residual,
                   multi_query_attention=mqa)
    model.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None),
                            backend=BACKEND.TORCH, dtype=torch.float16)
    assert wrapper.awq_get_modules_for_scaling(wrapper.model.transformer.encoder.layers[0], {}, {}) == []
    calibration = [{"input_ids": torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]]),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    wrapper.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    assert sum(isinstance(module, AwqTorchLinear) for module in wrapper.model.modules()) == 8
    for name, value in wrapper.model.state_dict().items():
        if "layernorm" in name:
            torch.testing.assert_close(value, model.state_dict()[name].half(), atol=0, rtol=0)
    with torch.inference_mode():
        expected = wrapper.model(**calibration[0], use_cache=False).logits
    assert torch.isfinite(expected).all()
    saved = tmp_path / "quantized"
    wrapper.save_quantized(str(saved))
    assert list(saved.glob("*.safetensors")) and not list(saved.glob("*.bin"))
    reloaded = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert sum(isinstance(module, AwqTorchLinear) for module in reloaded.model.modules()) == 8
    with torch.inference_mode():
        torch.testing.assert_close(reloaded.model(**calibration[0], use_cache=False).logits, expected,
                                   atol=2e-4, rtol=2e-3)
