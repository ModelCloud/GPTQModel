# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""MOSS Moon-003 local selection and source-equation qualification."""

import copy
import json

import pytest
import torch
from torch.nn import functional as F

from gptqmodel.models.definitions.moss import MossQModel
from gptqmodel.models.ports.moss.configuration_moss import MossConfig
from gptqmodel.models.ports.moss.modeling_moss import MossCache, MossForCausalLM
from gptqmodel.utils.hf import load_model_config


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_moss_local_config_ignores_checkpoint_python(tmp_path, monkeypatch, trust, bypass):
    monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    if bypass is not None:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    marker = tmp_path / "EXECUTED"
    (tmp_path / "configuration_moss.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint code ran')\n"
    )
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "moss", "n_embd": 32, "n_head": 4, "n_layer": 2,
               "n_positions": 32, "rotary_dim": 4, "vocab_size": 37,
               "bos_token_id": 1, "eos_token_id": 2,
        "auto_map": {"AutoConfig": "configuration_moss.MossConfig",
                     "AutoModelForCausalLM": "configuration_moss.MossForCausalLM"},
    }))
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config).__module__ == "gptqmodel.models.ports.moss.configuration_moss"
    assert not marker.exists()


def _config(**kwargs):
    values = {"vocab_size": 320, "n_embd": 32, "n_layer": 2, "n_head": 4, "n_positions": 32,
              "rotary_dim": 4, "n_inner": 64, "bos_token_id": 256, "eos_token_id": 257,
              "pad_token_id": 258, "initializer_range": 0.08, "attn_implementation": "eager"}
    values.update(kwargs)
    return MossConfig(**values)


def _model(**kwargs):
    torch.manual_seed(407)
    model = MossForCausalLM(_config(**kwargs)).eval()
    with torch.no_grad():
        for layer in model.transformer.h:
            layer.ln_1.weight.copy_(torch.linspace(0.7, 1.4, model.config.n_embd))
            layer.ln_1.bias.copy_(torch.linspace(-0.1, 0.2, model.config.n_embd))
    return model


def _source_equations(weights, config, ids, positions, padding=None, past=None, token_types=None, head_mask=None):
    """Independent pinned MOSS equations: project individual source row ranges.

    Uses no port forward, rotary, masking, QKV reshape, or cache helper. The
    original source is inspected, never imported by the tests. FP32 rotation and
    QK, compute-dtype probabilities/V, tanh GELU, parallel LN branches are exact
    source conventions, including the Q,V,K order within EACH of four shards.
    """
    hidden = F.embedding(ids, weights["transformer.wte.weight"])
    if token_types is not None:
        hidden = hidden + F.embedding(token_types, weights["transformer.wte.weight"])
    batch, length = ids.shape
    dim, heads = config.n_embd // config.n_head, config.n_head
    states, attentions, cache = [hidden], [], []

    def norm(x, name):
        return F.layer_norm(x, (config.n_embd,), weights[name + ".weight"],
                            weights[name + ".bias"], config.layer_norm_epsilon)

    def linear(x, name):
        return F.linear(x, weights[name + ".weight"], weights.get(name + ".bias"))

    def rotate(x):
        # Explicit adjacent-pair equations, avoiding the port's vector helper.
        components = []
        for d in range(0, config.rotary_dim, 2):
            theta = positions.float() / (10000 ** (d / config.rotary_dim))
            a, b = x[..., d], x[..., d + 1]
            components.extend([a * theta.cos() - b * theta.sin(), b * theta.cos() + a * theta.sin()])
        components.extend(x[..., d] for d in range(config.rotary_dim, dim))
        return torch.stack(components, -1)

    for index in range(config.n_layer):
        prefix = f"transformer.h.{index}."
        normalized = norm(hidden, prefix + "ln_1")
        packed_weight = weights[prefix + "attn.qkv_proj.weight"]
        queries, keys, values = [], [], []
        heads_per_shard = heads // 4
        shard_width = config.n_embd // 4
        for head in range(heads):
            shard, local_head = divmod(head, heads_per_shard)
            start = shard * shard_width * 3 + local_head * dim
            q = F.linear(normalized, packed_weight[start:start + dim])
            v = F.linear(normalized, packed_weight[start + shard_width:start + shard_width + dim])
            k = F.linear(normalized, packed_weight[start + 2 * shard_width:start + 2 * shard_width + dim])
            queries.append(rotate(q))
            keys.append(rotate(k))
            values.append(v)
        q, k, v = [torch.stack(parts, 1) for parts in (queries, keys, values)]
        if past is not None:
            k, v = torch.cat((past[index][0], k), -2), torch.cat((past[index][1], v), -2)
        cache.append((k, v))
        key_length = k.shape[-2]
        scores = (q.float() @ k.float().transpose(-1, -2)) / dim ** 0.5
        # Scalars compare physical token slots, not per-batch rotary positions.
        allowed = torch.tensor([[col <= key_length - length + row for col in range(key_length)]
                                for row in range(length)], device=ids.device)
        scores = torch.where(allowed, scores, torch.finfo(torch.float32).min)
        if padding is not None:
            scores = scores + (1 - padding[:, None, None, :].to(hidden.dtype)) * torch.finfo(hidden.dtype).min
        probabilities = scores.softmax(-1).to(v.dtype)
        if head_mask is not None:
            probabilities = probabilities * head_mask[index][None, :, None, None].to(hidden.dtype)
        attentions.append(probabilities)
        attended = (probabilities @ v).transpose(1, 2).reshape(batch, length, config.n_embd)
        a = linear(attended, prefix + "attn.out_proj")
        activated = linear(normalized, prefix + "mlp.fc_in")
        if config.activation_function == "gelu_new":
            activated = 0.5 * activated * (1 + torch.tanh((2 / torch.pi) ** 0.5 *
                                                         (activated + 0.044715 * activated.pow(3))))
        elif config.activation_function == "relu":
            activated = activated.clamp_min(0)
        else:
            raise AssertionError("Add an independent activation equation")
        hidden = a + linear(activated, prefix + "mlp.fc_out") + hidden
        states.append(hidden)
    hidden = norm(hidden, "transformer.ln_f")
    states[-1] = hidden
    return linear(hidden, "lm_head").float(), cache, states, attentions


@pytest.mark.parametrize("heads,width", [(4, 32), (8, 64)])
@pytest.mark.parametrize("rotary", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_moss_source_numerics_and_mixed_dtype_cache(heads, width, rotary, dtype):
    model = _model(n_head=heads, n_embd=width, rotary_dim=rotary).to(dtype)
    ids = torch.tensor([[3, 7, 5, 9, 11, 4, 8], [4, 6, 3, 9, 10, 5, 7]])
    positions = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]])
    padding = torch.tensor([[1, 1, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1]])
    head_mask = torch.ones(model.config.n_layer, heads)
    head_mask[0, 1] = 0.25
    reference, reference_cache, _, expected_attention = _source_equations(
        model.state_dict(), model.config, ids, positions, padding, head_mask=head_mask)
    actual = model(ids, position_ids=positions, attention_mask=padding, head_mask=head_mask,
                   output_attentions=True, output_hidden_states=True)
    tolerance = {torch.float32: 3e-6, torch.float16: 0.002, torch.bfloat16: 0.02}[dtype]
    torch.testing.assert_close(actual.logits, reference, atol=tolerance, rtol=tolerance)
    assert len(actual.hidden_states) == model.config.n_layer + 1
    for a, b in zip(actual.attentions, expected_attention):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
    cache, source_cache = None, None
    for start, end in [(0, 1), (1, 4), (4, 6), (6, 7)]:
        expected, source_cache, _, _ = _source_equations(
            model.state_dict(), model.config, ids[:, start:end], positions[:, start:end],
            padding[:, :end], past=source_cache, head_mask=head_mask)
        output = model(ids[:, start:end], position_ids=positions[:, start:end], attention_mask=padding[:, :end],
                       past_key_values=cache, head_mask=head_mask)
        cache = output.past_key_values
        assert isinstance(cache, MossCache) and cache.get_seq_length() == end
        torch.testing.assert_close(output.logits, expected, atol=tolerance, rtol=tolerance)
        for layer, (key, value) in zip(cache.layers, source_cache):
            assert layer.keys.dtype == torch.float32
            assert layer.values.dtype == dtype
            torch.testing.assert_close(layer.keys, key, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(layer.values, value, atol=tolerance, rtol=tolerance)
    for layer, (key, value) in zip(cache.layers, reference_cache):
        # Source uses finite-min masking; fully padded QUERY rows can differ
        # between full prefill and chunked calls. Only live token cache entries
        # should agree; each call's padded rows already matched its source oracle.
        live = padding[:, None, :, None].bool().expand_as(key)
        torch.testing.assert_close(layer.keys[live], key[live], atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(layer.values[live], value[live], atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_moss_released_rotary_fraction_near_original_position_limit(dtype):
    # Released 64/256 rotary fraction, without allocating a 16B model.
    model = _model(n_embd=64, n_head=4, rotary_dim=4, n_positions=2048).to(dtype)
    ids = torch.tensor([[3, 7, 5, 11, 9]])
    positions = torch.tensor([[2043, 2044, 2045, 2046, 2047]])
    expected, expected_cache, _, _ = _source_equations(model.state_dict(), model.config, ids, positions)
    actual = model(ids, position_ids=positions)
    tolerance = {torch.float32: 3e-6, torch.float16: 0.002, torch.bfloat16: 0.02}[dtype]
    torch.testing.assert_close(actual.logits, expected, atol=tolerance, rtol=tolerance)
    for layer, (keys, values) in zip(actual.past_key_values.layers, expected_cache):
        assert layer.keys.dtype == torch.float32 and layer.values.dtype == dtype
        torch.testing.assert_close(layer.keys, keys, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(layer.values, values, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("beams", [1, 3])
@torch.inference_mode()
def test_moss_generation_and_two_safetensors_roundtrips(tmp_path, dtype, beams):
    model = _model().to(dtype)
    ids = torch.tensor([[1, 3, 7], [0, 5, 9]])
    padding = torch.tensor([[1, 1, 1], [0, 1, 1]])
    options = {"attention_mask": padding, "max_new_tokens": 3, "num_beams": beams,
               "do_sample": False, "eos_token_id": None, "pad_token_id": 258}
    cached = model.generate(ids, **options)
    assert torch.equal(cached, model.generate(ids, use_cache=False, **options))
    if beams == 1:
        generated, mask = ids.clone(), padding.clone()
        for _ in range(3):
            positions = mask.cumsum(-1) - 1
            positions.masked_fill_(mask == 0, 0)
            logits, _, _, _ = _source_equations(model.state_dict(), model.config, generated, positions, mask)
            generated = torch.cat((generated, logits[:, -1].argmax(-1, keepdim=True)), -1)
            mask = torch.cat((mask, mask.new_ones(mask.shape[0], 1)), -1)
        assert torch.equal(cached, generated)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    logits = model(ids, use_cache=False).logits
    for index in range(2):
        saved = tmp_path / str(index)
        model.save_pretrained(saved)
        assert list(saved.glob("*.safetensors")) and not list(saved.glob("*.bin"))
        model = MossQModel.loader.from_pretrained(saved, dtype=dtype)
        assert set(model.state_dict()) == set(original)
        for key, tensor in model.state_dict().items():
            assert torch.equal(tensor, original[key]), key
        torch.testing.assert_close(model(ids, use_cache=False).logits, logits, atol=0, rtol=0)
        assert torch.equal(cached, model.generate(ids, **options))


@torch.inference_mode()
def test_moss_cache_lifecycle_and_failed_input_retry():
    model = _model().half()
    ids = torch.tensor([[1, 3, 7, 9], [4, 6, 8, 11]])
    cache = model(ids[:, :3]).past_key_values
    original = copy.deepcopy(cache)
    expected = model(ids, use_cache=False).logits[:, -1:]
    for kwargs, message in [({"attention_mask": torch.ones(2, 3)}, "full cached sequence"),
                            ({"position_ids": torch.tensor([[32], [5]])}, "n_positions")]:
        with pytest.raises(ValueError, match=message):
            model(ids[:, -1:], past_key_values=cache, **kwargs)
        for a, b in zip(cache.layers, original.layers):
            assert torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
    read_only = model(ids[:, -1:], past_key_values=cache, use_cache=False)
    assert read_only.past_key_values is None and cache.get_seq_length() == 3
    torch.testing.assert_close(read_only.logits, expected, atol=0.002, rtol=0.002)
    torch.testing.assert_close(model(ids[:, -1:], past_key_values=cache).logits, expected, atol=0.002, rtol=0.002)
    cache.reorder_cache(torch.tensor([1, 0, 1]))
    cache.batch_repeat_interleave(2)
    cache.batch_select_indices(torch.tensor([0, 3]))
    cache.crop(-1)
    for layer, before in zip(cache.layers, original.layers):
        assert torch.equal(layer.keys, before.keys[[1, 0]])
        assert torch.equal(layer.values, before.values[[1, 0]])
    actual = model(ids[[1, 0], -1:], past_key_values=cache).logits
    torch.testing.assert_close(actual, expected[[1, 0]], atol=0.002, rtol=0.002)
    cache.reset()
    assert all(layer.get_seq_length() == 0 for layer in cache.layers)
    torch.testing.assert_close(model(ids, past_key_values=cache).logits, model(ids).logits)


@pytest.mark.parametrize("assign", [False, True])
@torch.inference_mode()
def test_moss_state_assignment_token_types_loss_and_causality(assign):
    model, other = _model(), _model(activation_function="relu")
    other.load_state_dict(model.state_dict(), assign=assign)
    ids = torch.tensor([[1, 4, 6, 8]])
    token_types = torch.tensor([[3, 5, 5, 3]])
    positions = torch.tensor([[1, 3, 4, 7]])
    actual = other(ids, position_ids=positions, token_type_ids=token_types, labels=ids,
                   output_hidden_states=True, output_attentions=True)
    expected, _, hidden, _ = _source_equations(model.state_dict(), other.config, ids, positions,
                                               token_types=token_types)
    torch.testing.assert_close(actual.logits, expected, atol=3e-6, rtol=3e-6)
    for a, b in zip(actual.hidden_states, hidden):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)
    expected_loss = F.cross_entropy(expected[:, :-1].reshape(-1, 320), ids[:, 1:].reshape(-1))
    torch.testing.assert_close(actual.loss, expected_loss)
    embedded = other.transformer.wte(ids)
    torch.testing.assert_close(other(inputs_embeds=embedded).logits, other(ids).logits)
    changed = ids.clone()
    changed[:, -1] = 10
    torch.testing.assert_close(other(changed).logits[:, :-1], other(ids).logits[:, :-1], atol=0, rtol=0)
    assert isinstance(other(ids, return_dict=False), tuple)


def _write_tokenizer(path):
    from tokenizers.pre_tokenizers import ByteLevel

    from gptqmodel.models.ports.moss.tokenization_moss import MossTokenizer
    vocab = {char: i for i, char in enumerate(sorted(ByteLevel.alphabet()))}
    vocab.update({"<|endoftext|>": 256, "<eom>": 257, "<pad>": 258})
    tokenizer = MossTokenizer(vocab=vocab, merges=[], bos_token="<|endoftext|>",
                              eos_token="<eom>", pad_token="<pad>")
    tokenizer.save_pretrained(path)
    return tokenizer


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
@torch.inference_mode()
def test_moss_public_model_and_tokenizer_ignore_hostile_maps(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel import GPTQModel
    model = _model()
    path = tmp_path / "nested"
    model.save_pretrained(path)
    tokenizer = _write_tokenizer(path)
    marker = path / "EXECUTED"
    (path / "evil.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('executed')\n")
    for name, auto_map in [("config.json", {"AutoConfig": "evil.Config", "AutoModelForCausalLM": "evil.Model"}),
                           ("tokenizer_config.json", {"AutoTokenizer": ["evil.Tokenizer", None]})]:
        payload = json.loads((path / name).read_text())
        payload["auto_map"] = auto_map
        if name == "tokenizer_config.json":
            payload["tokenizer_class"] = "RemoteMossTokenizer"
        (path / name).write_text(json.dumps(payload))
    monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    if bypass is not None:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    loaded = GPTQModel.load(str(tmp_path), subfolder="nested", device="cpu", dtype=torch.float32,
                            trust_remote_code=trust)
    assert isinstance(loaded, MossQModel) and isinstance(loaded.model, MossForCausalLM)
    assert loaded.trust_remote_code is False and not marker.exists()
    assert loaded.tokenizer.encode("Hello 中文 123") == tokenizer.encode("Hello 中文 123")
    ids = torch.tensor([[3, 4, 7]])
    torch.testing.assert_close(loaded.model(ids).logits, model(ids).logits, atol=0, rtol=0)


@pytest.mark.parametrize("rotary", [4, 8])
@pytest.mark.parametrize("offload", [False, True])
def test_moss_public_plain_awq_all_linears_two_roundtrips(tmp_path, rotary, offload):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig
    model = _model(rotary_dim=rotary)
    model.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=offload)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16)
    assert wrapper.awq_get_modules_for_scaling(wrapper.model.transformer.h[0], {}, {}) == []
    calibration = [{"input_ids": torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]]),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    wrapper.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    linears = {name for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert linears == {f"transformer.h.{i}.{leaf}" for i in range(2)
                       for leaf in ("attn.qkv_proj", "attn.out_proj", "mlp.fc_in", "mlp.fc_out")}
    ids = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        logits = wrapper.model(ids, use_cache=False).logits
    assert torch.isfinite(logits).all()
    # Independently retained dense owners must survive LazyTurtle and packing.
    for key, tensor in wrapper.model.state_dict().items():
        if key in model.state_dict() and not any(key.startswith(path + ".") for path in linears):
            assert torch.equal(tensor, model.state_dict()[key].to(tensor.dtype)), key
    packed_state = {key: tensor.detach().clone() for key, tensor in wrapper.model.state_dict().items()}
    for cycle in range(2):
        saved = tmp_path / f"quantized-{cycle}"
        wrapper.save(str(saved))
        assert list(saved.glob("*.safetensors")) and not list(saved.glob("*.bin"))
        reloaded = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
        snapshot = reloaded.model.state_dict()
        assert set(snapshot) == set(packed_state)
        for key, tensor in snapshot.items():
            assert torch.equal(tensor, packed_state[key]), key
        with torch.no_grad():
            torch.testing.assert_close(reloaded.model(ids, use_cache=False).logits, logits, atol=0, rtol=0)
            reloaded.model.generate(ids, max_new_tokens=2, eos_token_id=None)
        wrapper = reloaded


@pytest.mark.parametrize("kwargs,match", [({"n_head": 2}, "divisible by four"),
                                         ({"num_key_value_heads": 1}, "MHA"),
                                         ({"rotary_dim": 3}, "rotary_dim"),
                                         ({"rotary_dim": 16}, "rotary_dim"),
                                         ({"n_positions": 0}, "positive"),
                                         ({"wbits": 4}, "original converter")])
def test_moss_rejects_incompatible_layouts(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _config(**kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_moss_public_safetensors_failure_repair_retry(tmp_path, dtype):
    from gptqmodel import GPTQModel

    source = _model()
    source.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    weights = tmp_path / "model.safetensors"
    saved = weights.read_bytes()
    weights.write_bytes(b"invalid header")
    try:
        with pytest.raises(Exception, match="(?i)safetensor|header"):
            GPTQModel.load(str(tmp_path), device="cpu", dtype=dtype)
    finally:
        weights.write_bytes(saved)
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=dtype)
    assert isinstance(loaded.model, MossForCausalLM)
    ids = torch.tensor([[3, 5, 7]])
    source = source.to(dtype)
    torch.testing.assert_close(loaded.model(ids).logits, source(ids).logits, atol=0, rtol=0)
    assert loaded.model.transformer.wte.weight is not loaded.model.lm_head.weight
    assert loaded.model.lm_head.bias is not None
    assert loaded.model.transformer.h[0].attn.causal_mask.dtype == torch.bool


@torch.inference_mode()
def test_moss_standard_aliases_full_rotary_context_boundary_and_cache_types():
    from transformers.cache_utils import DynamicCache, StaticCache

    config = MossConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                        max_position_embeddings=32, rotary_dim=None,
                        vocab_size=320, bos_token_id=256, eos_token_id=257)
    assert (config.n_embd, config.n_layer, config.n_head, config.n_positions, config.rotary_dim) == (32, 2, 4, 32, 8)
    model = _model(rotary_dim=8)
    ids = torch.tensor([[3, 7, 5]])
    output = model(ids, past_key_values=DynamicCache(config=config))
    torch.testing.assert_close(output.logits, model(ids).logits)
    with pytest.raises(ValueError, match="MossCache"):
        model(ids, past_key_values=StaticCache(config=config, max_cache_len=32))
    prefix = model(torch.tensor([[3] * 31]))
    final = model(ids[:, :1], past_key_values=prefix.past_key_values)
    assert final.past_key_values.get_seq_length() == 32
    before = copy.deepcopy(final.past_key_values)
    with pytest.raises(ValueError, match="exceeds n_positions"):
        model(ids[:, :1], past_key_values=final.past_key_values)
    for old, new in zip(before.layers, final.past_key_values.layers):
        assert torch.equal(old.keys, new.keys) and torch.equal(old.values, new.values)


def _source_bpe_ids(text, vocab, merges, added=None, prefix=False):
    """Inspected source regex/byte mapping/greedy ranked BPE, no HF backend."""
    import regex

    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = byte_values.copy()
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + len(codepoints) - 188)
    mapping = dict(zip(byte_values, map(chr, codepoints)))
    ranks = {tuple(pair): index for index, pair in enumerate(merges)}
    pattern = regex.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
    added = added or {}
    pieces = regex.split("(" + "|".join(regex.escape(t) for t in sorted(added, key=len, reverse=True)) + ")", text) if added else [text]
    result = []
    for piece in pieces:
        if piece in added:
            result.append(added[piece])
            continue
        if prefix:
            piece = " " + piece
            prefix = False
        for word in pattern.findall(piece):
            symbols = [mapping[b] for b in word.encode()]
            while len(symbols) > 1:
                pairs = list(zip(symbols, symbols[1:]))
                best = min(pairs, key=lambda pair: ranks.get(pair, float("inf")))
                if best not in ranks:
                    break
                merged, index = [], 0
                while index < len(symbols):
                    if index + 1 < len(symbols) and tuple(symbols[index:index + 2]) == best:
                        merged.append("".join(best))
                        index += 2
                    else:
                        merged.append(symbols[index])
                        index += 1
                symbols = merged
            result.extend(vocab[s] for s in symbols)
    return result


@pytest.mark.parametrize("bos", [False, True])
@pytest.mark.parametrize("prefix", [False, True])
def test_moss_tokenizer_bpe_bos_pair_padding_truncation_and_save(tmp_path, bos, prefix):
    from tokenizers.pre_tokenizers import ByteLevel

    from gptqmodel.models.ports.moss.tokenization_moss import MossTokenizer
    vocab = {char: i for i, char in enumerate(sorted(ByteLevel.alphabet()))}
    merges = [("H", "e"), ("He", "l"), ("Hel", "l"), ("Hell", "o"), ("1", "2"), ("12", "3")]
    for left, right in merges:
        vocab[left + right] = len(vocab)
    for token in ["<|endoftext|>", "<eom>", "<pad>"]:
        vocab[token] = len(vocab)
    tok = MossTokenizer(vocab=vocab, merges=merges, add_bos_token=bos, add_prefix_space=prefix,
                        pad_token="<pad>")
    texts = ["Hello 中文 1234567890", "We're I'M", "🙂 café\r\n", "def foo():\n    return 123"]
    for text in texts:
        expected = _source_bpe_ids(text, vocab, merges, prefix=prefix)
        assert tok.encode(text, add_special_tokens=False) == expected
        assert tok.encode(text) == ([tok.bos_token_id] if bos else []) + expected
        assert tok.decode(expected) == (" " if prefix else "") + text
        pair = tok(text, text)["input_ids"]
        assert pair == tok.encode(text) * 2
    batch = tok(texts, padding=True)
    assert len({len(ids) for ids in batch["input_ids"]}) == 1
    assert tok.eos_token_id == vocab["<eom>"]
    raw = "\nprint(1)\nprint(2)\nSTOP\n"
    assert tok.decode(tok.encode(raw, add_special_tokens=False), truncate_before_pattern=["^STOP"]).endswith("print(1)\n")
    # Roundtrip a serializer and a legacy vocab/merges representation.
    for legacy in (False, True):
        saved = tmp_path / str(legacy)
        tok.save_pretrained(saved)
        if legacy:
            (saved / "vocab.json").write_text(json.dumps(vocab))
            (saved / "merges.txt").write_text("#version: 0.2\n" + "\n".join(" ".join(p) for p in merges) + "\n")
            (saved / "tokenizer.json").unlink()
            # Original Moon-003 vocab/merges config explicitly carries this
            # source flag. Native TF5 saves it only in the serialized template.
            payload = json.loads((saved / "tokenizer_config.json").read_text())
            payload["add_bos_token"] = bos
            (saved / "tokenizer_config.json").write_text(json.dumps(payload))
        loaded = MossTokenizer.from_pretrained(saved)
        for text in texts:
            assert loaded.encode(text) == tok.encode(text)
            assert loaded.decode(loaded.encode(text)) == tok.decode(tok.encode(text))
