# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""ChatGLM checkpoint ownership, independent numerics, cache and real loading."""

import copy
import hashlib
import json
import shutil
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from gptqmodel import CalibrationConfig
from gptqmodel.models.definitions.chatglm import ChatGLMQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.utils.hf import load_model_config


TOKENIZER_CONTRACTS = json.loads((Path(__file__).resolve().parents[1] /
    'gptqmodel/models/ports/chatglm/tokenizer_contracts.json').read_text())


@lru_cache(maxsize=3)
def _released_assets(release):
    from huggingface_hub import hf_hub_download
    contract = TOKENIZER_CONTRACTS[release]
    paths = {}
    for filename, digest in contract['files'].items():
        path = Path(hf_hub_download(contract['repo'], filename, revision=contract['revision']))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        paths[filename] = path
    return paths


def _write_released_tokenizer(path, release):
    for filename, source in _released_assets(release).items():
        shutil.copyfile(source, path / filename)


def _assert_tokenizer_contract(tokenizer, release):
    contract = TOKENIZER_CONTRACTS[release]
    for sample in contract['texts']:
        actual = tokenizer(sample['text'], return_tensors='pt')
        for key, value in sample['input'].items():
            assert actual[key].tolist() == [value]
        assert tokenizer.encode(sample['text'], add_special_tokens=False) == sample['plain_ids']
        assert tokenizer.decode(sample['plain_ids'], skip_special_tokens=True) == sample['decoded']
    assert tokenizer.get_prefix_tokens() == contract['prefix']
    for field in ['eos_token', 'eos_token_id', 'pad_token', 'pad_token_id', 'unk_token', 'unk_token_id']:
        assert getattr(tokenizer, field) == contract['specials'][field]
    assert dict(tokenizer([contract['texts'][0]['text'], contract['texts'][2]['text']],
                          padding=True, pad_to_multiple_of=8)) == contract['batch']
    if 'pair' in contract:
        assert dict(tokenizer(contract['texts'][0]['text'], text_pair=contract['texts'][1]['text'])) == contract['pair']
    if release == 'glm2':
        assert tokenizer.build_prompt('你好！', [('hi', 'hello')]) == contract['prompt']
        assert tokenizer.encode(contract['prompt']) == contract['prompt_ids']
    else:
        assert tokenizer.apply_chat_template(contract['messages'], tokenize=False,
                                             add_generation_prompt=True) == contract['rendered']
        chat = tokenizer.apply_chat_template(contract['messages'], tokenize=True,
                                             add_generation_prompt=True)
        # TF5 returns BatchEncoding by default; TF4's oracle returned input_ids.
        assert chat['input_ids'] == contract['chat_ids']
        assert tokenizer.build_single_message('assistant', 'python', 'print(2 + 3)') == contract['single_message']
    if release == 'glm3':
        built = tokenizer.build_chat_input(contract['messages'][-1]['content'], history=contract['history'])
        assert {key: value.tolist() for key, value in built.items()} == contract['build_chat']
        # EOS/UNK registration must not overwrite fixed assistant/observation IDs.
        assert tokenizer.convert_tokens_to_ids('<|assistant|>') == 64796
        assert tokenizer.convert_tokens_to_ids('<|observation|>') == 64797


@pytest.mark.parametrize('release', ['glm2', 'glm3', 'glm4'])
@pytest.mark.parametrize('trust', [None, False, True])
@pytest.mark.parametrize('bypass', [False, True])
def test_chatglm_released_tokenizer_public_load(tmp_path, monkeypatch, release, trust, bypass):
    from tokenicer import Tokenicer

    from gptqmodel import GPTQModel
    contract = TOKENIZER_CONTRACTS[release]
    model = _model(rope_ratio=500 if release == 'glm4' else None,
                   padded_vocab_size=151552 if release == 'glm4' else 65024,
                   seq_length=128, pad_token_id=contract['specials']['pad_token_id'],
                   eos_token_id=contract['specials']['eos_token_id'])
    model.save_pretrained(tmp_path)
    _write_released_tokenizer(tmp_path, release)
    marker = _poison(tmp_path)
    (tmp_path / 'tokenization_chatglm.py').write_text(
        (tmp_path / 'checkpoint_code.py').read_text())
    _bypass(monkeypatch, bypass)
    loaded = GPTQModel.load(str(tmp_path), trust_remote_code=trust, device='cpu', dtype=torch.float32)
    assert isinstance(loaded.tokenizer, Tokenicer)
    assert not isinstance(loaded.tokenizer.tokenizer, Tokenicer)
    assert type(loaded.tokenizer.tokenizer).__module__.startswith('gptqmodel.models.ports.chatglm.')
    assert not marker.exists()
    _assert_tokenizer_contract(loaded.tokenizer, release)
    tokens = loaded.tokenizer(contract['texts'][0]['text'], return_tensors='pt')
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(**tokens).logits, model(**tokens).logits, atol=0, rtol=0)
        generated = loaded.generate(**tokens, max_new_tokens=2, eos_token_id=None)
        assert generated.shape[-1] == tokens['input_ids'].shape[-1] + 2
    assert not marker.exists()


@pytest.mark.parametrize('release', ['glm2', 'glm3', 'glm4'])
def test_chatglm_released_tokenizer_roundtrip_retry(tmp_path, monkeypatch, release):
    from tokenicer import Tokenicer

    from gptqmodel import GPTQModel

    contract = TOKENIZER_CONTRACTS[release]
    source = tmp_path / 'source'
    source.mkdir()
    model = _model(rope_ratio=500 if release == 'glm4' else None,
                   padded_vocab_size=151552 if release == 'glm4' else 65024,
                   seq_length=128, pad_token_id=contract['specials']['pad_token_id'],
                   eos_token_id=contract['specials']['eos_token_id'])
    model.save_pretrained(source)
    _write_released_tokenizer(source, release)
    marker = _poison(source)
    _bypass(monkeypatch, True)
    # A failed load must neither publish a stale tokenizer nor execute Python.
    asset = source / 'tokenizer.model'
    asset.write_bytes(b'not a SentencePiece or TikToken model')
    with pytest.raises((ValueError, RuntimeError, OSError, TypeError)):
        GPTQModel.load(str(source), trust_remote_code=True, device='cpu', dtype=torch.float32)
    assert not marker.exists()
    shutil.copyfile(_released_assets(release)['tokenizer.model'], asset)
    original_load = Tokenicer.load.__func__
    calls = []

    def counted_load(cls, tokenizer, **kwargs):
        calls.append(tokenizer)
        return original_load(cls, tokenizer, **kwargs)

    monkeypatch.setattr(Tokenicer, 'load', classmethod(counted_load))
    loaded = GPTQModel.load(str(source), trust_remote_code=True, device='cpu', dtype=torch.float32)
    assert len(calls) == 1, 'Only the shared adapter may normalize an owned tokenizer'
    assert calls[0] is loaded.tokenizer.tokenizer
    tokens = loaded.tokenizer(contract['texts'][1]['text'], return_tensors='pt')
    with torch.no_grad():
        expected = loaded.model(**tokens).logits
    _assert_tokenizer_contract(loaded.tokenizer, release)
    # Repeat both serialization boundaries; original asset hashes remain exact.
    for index in range(2):
        saved = tmp_path / f'saved-{index}'
        loaded.model.save_pretrained(saved, safe_serialization=True)
        loaded.tokenizer.save_pretrained(saved)
        assert hashlib.sha256((saved / 'tokenizer.model').read_bytes()).hexdigest() == contract['files']['tokenizer.model']
        assert list(saved.glob('*.safetensors'))
        assert not list(saved.glob('*.bin'))
        reloaded_marker = _poison(saved)
        calls.clear()
        loaded = GPTQModel.load(str(saved), trust_remote_code=True, device='cpu', dtype=torch.float32)
        assert len(calls) == 1
        assert calls[0] is loaded.tokenizer.tokenizer
        _assert_tokenizer_contract(loaded.tokenizer, release)
        with torch.no_grad():
            torch.testing.assert_close(loaded.model(**tokens).logits, expected, atol=0, rtol=0)
        assert not reloaded_marker.exists()
    assert not marker.exists()


def test_chatglm_released_literal_special_tokens(tmp_path):
    _write_released_tokenizer(tmp_path, 'glm3')
    tokenizer = ChatGLMQModel.loader.load_tokenizer(tmp_path, encode_special_tokens=True)
    contract = TOKENIZER_CONTRACTS['glm3']['special_literal']
    assert tokenizer.encode(contract['text']) == contract['ids']
    saved = tmp_path / 'saved'
    tokenizer.save_pretrained(saved)
    reloaded = ChatGLMQModel.loader.load_tokenizer(saved)
    assert reloaded.encode(contract['text']) == contract['ids']


@pytest.mark.cpu
@pytest.mark.parametrize('release', ['glm2', 'glm3', 'glm4'])
def test_chatglm_released_tokenizer_plain_awq_roundtrips(tmp_path, monkeypatch, release):
    from tokenicer import Tokenicer

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    contract = TOKENIZER_CONTRACTS[release]
    model = _model(rope_ratio=500 if release == 'glm4' else None,
                   padded_vocab_size=151552 if release == 'glm4' else 65024,
                   seq_length=128, pad_token_id=contract['specials']['pad_token_id'],
                   eos_token_id=contract['specials']['eos_token_id'])
    model.save_pretrained(tmp_path)
    _write_released_tokenizer(tmp_path, release)
    marker = _poison(tmp_path)
    _bypass(monkeypatch, True)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=AWQConfig(device='cpu', group_size=16, fallback=None),
                            trust_remote_code=True, backend=BACKEND.TORCH, dtype=torch.float16)
    assert wrapper.awq_get_modules_for_scaling(wrapper.model.transformer.encoder.layers[0], {}, {}) == []
    # Calibrate actual released token IDs produced by the public tokenizer.
    calibration = [wrapper.tokenizer(sample['text'], return_tensors='pt') for sample in contract['texts']]
    wrapper.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    assert sum(isinstance(module, AwqTorchLinear) for module in wrapper.model.modules()) == 8
    state = {key: value.clone() for key, value in wrapper.model.state_dict().items()}
    dense = model.state_dict()
    for key, value in state.items():
        if key.startswith(('transformer.embedding.', 'transformer.output_layer.', 'transformer.rotary_pos_emb.')) or 'layernorm' in key:
            torch.testing.assert_close(value, dense[key].to(value.dtype), atol=0, rtol=0)
    with torch.inference_mode():
        expected = wrapper.model(**calibration[0], use_cache=False).logits
    original_load = Tokenicer.load.__func__
    calls = []

    def counted_load(cls, tokenizer, **kwargs):
        calls.append(tokenizer)
        return original_load(cls, tokenizer, **kwargs)

    monkeypatch.setattr(Tokenicer, 'load', classmethod(counted_load))
    for index in range(2):
        saved = tmp_path / f'awq-{index}'
        wrapper.save_quantized(str(saved))
        assert list(saved.glob('*.safetensors')) and not list(saved.glob('*.bin'))
        assert hashlib.sha256((saved / 'tokenizer.model').read_bytes()).hexdigest() == contract['files']['tokenizer.model']
        calls.clear()
        wrapper = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device='cpu', dtype=torch.float16,
                                trust_remote_code=True)
        assert len(calls) == 1
        assert calls[0] is wrapper.tokenizer.tokenizer
        assert state.keys() == wrapper.model.state_dict().keys()
        for key, value in wrapper.model.state_dict().items():
            torch.testing.assert_close(value, state[key], atol=0, rtol=0)
        _assert_tokenizer_contract(wrapper.tokenizer, release)
        with torch.inference_mode():
            torch.testing.assert_close(wrapper.model(**calibration[0], use_cache=False).logits, expected,
                                       atol=2e-4, rtol=2e-3)
    assert not marker.exists()


@pytest.mark.parametrize('name', ['ForeignTokenizer', '../checkpoint_code.ChatGLMTokenizer', None])
def test_chatglm_tokenizer_unknown_metadata_fails_closed(tmp_path, name):
    (tmp_path / 'tokenizer_config.json').write_text(json.dumps({
        'tokenizer_class': name,
        'auto_map': {'AutoTokenizer': ['checkpoint_code.UntrustedTokenizer', None]},
    }))
    with pytest.raises(ValueError, match='Unsupported ChatGLM tokenizer metadata'):
        ChatGLMQModel.loader.load_tokenizer(tmp_path, trust_remote_code=True, code_revision='malicious')


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
        if config.fp32_residual_connection:
            x = x.to(weights[name + ".weight"].dtype)
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
    if config.fp32_residual_connection:
        hidden = hidden.float()
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
        residual = normalized if config.apply_residual_connection_post_layernorm else hidden
        hidden = attention + (residual.float() if config.fp32_residual_connection else residual)
        normalized = norm(hidden, prefix + "post_attention_layernorm")
        gate, up = linear(normalized, prefix + "mlp.dense_h_to_4h").chunk(2, -1)
        output = linear(F.silu(gate) * up, prefix + "mlp.dense_4h_to_h")
        residual = normalized if config.apply_residual_connection_post_layernorm else hidden
        hidden = output + (residual.float() if config.fp32_residual_connection else residual)
    if config.post_layer_norm:
        hidden = norm(hidden, "transformer.encoder.final_layernorm")
    return linear(hidden.to(weights["transformer.output_layer.weight"].dtype), "transformer.output_layer"), cache


@pytest.mark.parametrize("rope_ratio", [None, 500])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("new_size,padding", [(37, None), (71, None), (37, 8)])
def test_chatglm_public_resize_safetensors_reload_resave(tmp_path, rope_ratio, tied, new_size, padding):
    from gptqmodel import GPTQModel
    model = _model(rope_ratio=rope_ratio, tie_word_embeddings=tied)
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    model.resize_token_embeddings(new_size, pad_to_multiple_of=padding, mean_resizing=False)
    rows = new_size if padding is None else ((new_size + padding - 1) // padding) * padding
    ids = torch.tensor([[3, 4, rows - 1, 6], [7, 8, 9, 4]])
    with torch.no_grad():
        expected = model(ids).logits
    model.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    marker = _poison(tmp_path)
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=False).model
    for current in (model, loaded):
        assert current.config.vocab_size == current.config.padded_vocab_size == rows
        assert current.get_input_embeddings().weight.shape[0] == current.get_output_embeddings().weight.shape[0] == rows
        assert (current.get_input_embeddings().weight is current.get_output_embeddings().weight) == tied
        kept = min(rows, 64)
        for name in ("transformer.embedding.word_embeddings.weight", "transformer.output_layer.weight"):
            torch.testing.assert_close(current.state_dict()[name][:kept], original[name][:kept], atol=0, rtol=0)
        with torch.no_grad():
            torch.testing.assert_close(current(ids).logits, expected, atol=0, rtol=0)
            cache = current(ids[:, :2]).past_key_values
            torch.testing.assert_close(current(ids[:, 2:], past_key_values=cache).logits,
                                       expected[:, 2:], atol=2e-7, rtol=2e-5)
    assert not marker.exists()
    second = tmp_path / "second"
    loaded.save_pretrained(second)
    again = ChatGLMQModel.loader.from_pretrained(second).eval()
    with torch.no_grad():
        torch.testing.assert_close(again(ids).logits, expected, atol=0, rtol=0)
    assert again.config.vocab_size == again.config.padded_vocab_size == rows
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(again.state_dict()[name], value, atol=0, rtol=0)
    assert list(second.glob("*.safetensors")) and not list(second.glob("*.bin"))


@pytest.mark.parametrize("rope_ratio", [None, 500])
def test_chatglm_legacy_logical_vocab_does_not_shrink_padded_checkpoint(tmp_path, rope_ratio):
    from gptqmodel import GPTQModel
    original = _model(rope_ratio=rope_ratio)
    original.save_pretrained(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text())
    payload["vocab_size"] = 31
    (tmp_path / "config.json").write_text(json.dumps(payload))
    _write_tokenizer(tmp_path)
    model = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32).model
    assert model.get_input_embeddings().weight.shape[0] == 64
    before = model.get_input_embeddings()
    assert model.resize_token_embeddings() is before
    assert model.config.vocab_size == model.config.padded_vocab_size == 64
    for name, value in original.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)


@pytest.mark.parametrize("rope_ratio", [None, 500])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rmsnorm", [False, True])
@pytest.mark.parametrize("post_residual", [False, True])
@pytest.mark.parametrize("post_norm", [False, True])
@pytest.mark.parametrize("fp32_residual", [False, True])
def test_chatglm_fp32_residual_compute_dtype_oracle_and_cache(
    rope_ratio, dtype, rmsnorm, post_residual, post_norm, fp32_residual,
):
    model = _model(rope_ratio=rope_ratio, hidden_size=20, ffn_hidden_size=28, kv_channels=8,
                   num_attention_heads=3, padded_vocab_size=31, seq_length=48, multi_query_group_num=1,
                   rmsnorm=rmsnorm, multi_query_attention=rmsnorm, post_layer_norm=post_norm,
                   apply_residual_connection_post_layernorm=post_residual,
                   fp32_residual_connection=fp32_residual).to(dtype)
    ids = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 4]])
    positions = torch.arange(4)[None].expand(2, -1)
    residual_dtypes, projection_dtypes = [], []
    handles = []
    for layer in model.transformer.encoder.layers:
        handles.append(layer.register_forward_hook(
            lambda module, args, output: residual_dtypes.append(output[0].dtype)))
        handles.append(layer.self_attention.query_key_value.register_forward_pre_hook(
            lambda module, args: projection_dtypes.append(args[0].dtype)))
    tolerance = 4e-6 if dtype == torch.float32 else 2e-3
    try:
        with torch.no_grad():
            expected, _ = _oracle(model.state_dict(), model.config, ids, positions, torch.ones_like(ids))
            output = model(ids, labels=ids, output_hidden_states=True)
            assert output.logits.dtype == dtype and torch.isfinite(output.logits).all()
            torch.testing.assert_close(output.logits, expected, atol=tolerance, rtol=tolerance)
            expected_loss = F.cross_entropy(expected[:, :-1].float().reshape(-1, 31), ids[:, 1:].reshape(-1)).to(dtype)
            torch.testing.assert_close(output.loss, expected_loss, atol=tolerance, rtol=tolerance)
            cache = model(ids[:, :2]).past_key_values
            torch.testing.assert_close(model(ids[:, 2:], past_key_values=cache).logits,
                                       output.logits[:, 2:], atol=tolerance, rtol=tolerance)
            embedded = model.get_input_embeddings()(ids)
            torch.testing.assert_close(model(inputs_embeds=embedded).logits, output.logits, atol=0, rtol=0)
        assert set(projection_dtypes) == {dtype}
        assert set(residual_dtypes) == {torch.float32 if fp32_residual else dtype}
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("rope_ratio", [None, 500])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rmsnorm", [False, True])
def test_chatglm_public_load_fp32_residual_dtype_roundtrip(tmp_path, rope_ratio, dtype, rmsnorm):
    from gptqmodel import GPTQModel
    original = _model(rope_ratio=rope_ratio, rmsnorm=rmsnorm, fp32_residual_connection=True,
                      post_layer_norm=False)
    original.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    marker = _poison(tmp_path)
    reference = copy.deepcopy(original).to(dtype)
    loaded = GPTQModel.load(str(tmp_path), dtype=dtype, device="cpu", trust_remote_code=False).model
    ids = torch.tensor([[3, 7, 8, 9], [4, 6, 5, 8]])
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, atol=0, rtol=0)
    with torch.no_grad():
        expected = reference(ids).logits
        torch.testing.assert_close(loaded(ids).logits, expected, atol=0, rtol=0)
        generated = loaded.generate(ids, max_new_tokens=3, eos_token_id=None)
        assert torch.equal(generated, loaded.generate(ids, max_new_tokens=3,
                                                      eos_token_id=None, use_cache=False))
    assert not marker.exists()
    output = loaded(ids, labels=ids)
    output.loss.backward()
    gradient = loaded.transformer.encoder.layers[0].self_attention.query_key_value.weight.grad
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    loaded.zero_grad(set_to_none=True)
    second = tmp_path / "second"
    loaded.save_pretrained(second)
    again = ChatGLMQModel.loader.from_pretrained(second, dtype=dtype).eval()
    with torch.no_grad():
        torch.testing.assert_close(again(ids).logits, expected, atol=0, rtol=0)
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(again.state_dict()[name], value, atol=0, rtol=0)
    assert list(second.glob("*.safetensors")) and not list(second.glob("*.bin"))


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
    # Small-vocabulary resize/dtype fixtures only consume explicit integer IDs.
    # Use SP's real pad=0 there; the released-asset tests use the full matching
    # GLM2/3/4 vocabularies and assert text/chat/model-consumer parity separately.
    _write_released_tokenizer(path, 'glm3')


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
    wrapper.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
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
