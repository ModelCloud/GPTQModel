# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real, offline Transformers 5 qualification for the four Baichuan layouts."""

import json
import math

import pytest
import torch
from torch.nn import functional as F
from transformers.cache_utils import StaticCache

from gptqmodel.models.definitions.baichuan import BaiChuanQModel
from gptqmodel.models.local import LocalModelLoader, local_model_loader
from gptqmodel.models.ports.baichuan.configuration_baichuan import BaiChuanConfig, BaichuanConfig
from gptqmodel.models.ports.baichuan.modeling_baichuan import (
    BaiChuanForCausalLM,
    BaichuanForCausalLM,
    NormHead,
)
from gptqmodel.utils.hf import load_model_config


VARIANTS = [(position, normalized) for position in ("rope", "alibi") for normalized in (False, True)]


def tiny_config(position="rope", normalized=True, **kwargs):
    return BaichuanConfig(vocab_size=37, hidden_size=24, intermediate_size=40,
                          num_hidden_layers=2, num_attention_heads=3,
                          max_position_embeddings=4, model_max_length=4,
                          position_embedding_type=position, normalize_lm_head=normalized, **kwargs)


def tiny_model(position="rope", normalized=True, dtype=torch.float32):
    torch.manual_seed(1807)
    model = BaichuanForCausalLM(tiny_config(position, normalized)).to(dtype).eval()
    # Non-degenerate magnitudes expose wrong QKV ordering/head normalization.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" not in name:
                param.normal_(std=0.13)
    return model


def upstream_equations(model, ids, mask=None, positions=None):
    """Independent full-forward oracle transcribed from pinned upstream code.

    Uses checkpoint tensors, not port layer/attention/norm/rotary forwards.
    See port PROVENANCE.md for sources. No checkpoint Python is executed.
    """
    state, config = model.state_dict(), model.config
    x = F.embedding(ids, state["model.embed_tokens.weight"])
    b, s = ids.shape
    d = config.hidden_size // config.num_attention_heads
    positions = torch.arange(s)[None] if positions is None else positions

    def norm(t, key):
        t = t * torch.rsqrt(t.float().square().mean(-1, keepdim=True) + config.rms_norm_eps)
        return t.to(state[key].dtype) * state[key]

    def linear(t, key):
        return F.linear(t, state[key + ".weight"])

    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}."
        a = norm(x, prefix + "input_layernorm.weight")
        packed = linear(a, prefix + "self_attn.W_pack")
        # Upstream unflatten/transpose form, independent of port chunking.
        packed = packed.unflatten(-1, (3, config.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)
        q, k, v = [t.reshape(b, s, config.num_attention_heads, d).transpose(1, 2) for t in packed]
        if config.position_embedding_type == "rope":
            inv = (1 / (10000 ** (torch.arange(0, d, 2).float() / d)))
            if not config.normalize_lm_head and s > config.max_position_embeddings:
                inv = state[prefix + "self_attn.rotary_emb.inv_freq"]
            table = torch.outer(torch.arange(int(positions.max()) + 1).to(inv.dtype), inv)
            table = torch.cat((table, table), dim=-1)
            cos, sin = table.cos()[positions, None, :].transpose(1, 2), table.sin()[positions, None, :].transpose(1, 2)
            if config.normalize_lm_head:
                q, k = q.float(), k.float()
            else:
                cos, sin = cos.to(q.dtype), sin.to(q.dtype)
            def rotate(t):
                return torch.cat((-t[..., d // 2:], t[..., :d // 2]), dim=-1)
            q, k = (q * cos + rotate(q) * sin).to(v.dtype), (k * cos + rotate(k) * sin).to(v.dtype)
        scores = q @ k.transpose(-1, -2) / math.sqrt(d)
        allowed = torch.ones((s, s), dtype=torch.bool).tril()[None, None].expand(b, 1, s, s)
        if mask is not None:
            allowed = allowed & mask[:, None, None, :].bool()
            if config.position_embedding_type == "alibi":
                allowed = allowed & mask[:, None, :, None].bool()
                allowed = allowed & (mask[:, None, :, None] == mask[:, None, None, :])
        bias = torch.zeros((b, 1, s, s), dtype=x.dtype)
        if config.position_embedding_type == "alibi":
            # Three heads exercise upstream's non-power-of-two interleave:
            # slopes(2) followed by slopes(4)[::2].
            slopes = x.new_tensor([1 / 16, 1 / 256, 1 / 4])
            offsets = torch.arange(s)[None, :]
            if model.training:
                offsets = offsets - torch.arange(s)[:, None]
            bias = bias + slopes[:, None, None] * offsets
        bias = bias.masked_fill(~allowed, torch.finfo(x.dtype).min)
        scores = (scores + bias).clamp_min(torch.finfo(x.dtype).min)
        weights = F.softmax(scores, dim=-1, dtype=torch.float32 if config.position_embedding_type == "rope" else None).to(v.dtype)
        a = (weights @ v).transpose(1, 2).reshape(b, s, config.hidden_size)
        x = x + linear(a, prefix + "self_attn.o_proj")
        a = norm(x, prefix + "post_attention_layernorm.weight")
        x = x + linear(F.silu(linear(a, prefix + "mlp.gate_proj")) * linear(a, prefix + "mlp.up_proj"), prefix + "mlp.down_proj")
    x = norm(x, "model.norm.weight")
    head = state["lm_head.weight"]
    return F.linear(x, F.normalize(head) if config.normalize_lm_head else head)


@pytest.mark.parametrize("position,normalized", VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_full_forward_matches_pinned_upstream_equations(position, normalized, dtype):
    model = tiny_model(position, normalized, dtype)
    ids = torch.tensor([[0, 1, 4, 5, 6, 7], [1, 8, 9, 10, 11, 0]])
    mask = ids.ne(0).long()
    positions = torch.tensor([[0, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 0]])
    with torch.no_grad():
        result = model(ids, attention_mask=mask, position_ids=positions, output_attentions=True,
                       output_hidden_states=True, use_cache=False)
        expected = upstream_equations(model, ids, mask, positions)
    assert result.logits.dtype == dtype
    torch.testing.assert_close(result.logits, expected, rtol=2e-5, atol=2e-6)
    assert len(result.hidden_states) == 3
    assert len(result.attentions) == 2
    assert torch.isfinite(result.logits).all()
    # Causality: a changed final token cannot affect previous logits.
    changed = ids.clone()
    changed[:, -1] = 14
    with torch.no_grad():
        other = model(changed, attention_mask=mask, position_ids=positions, use_cache=False).logits
    # Upstream gives a fully masked padding query uniform attention after its
    # finite-min clamp. Only real tokens carry a causal-output contract.
    valid = mask[:, :-1].bool()
    torch.testing.assert_close(other[:, :-1][valid], result.logits[:, :-1][valid], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [3, 4, 5])
def test_baichuan1_original_rope_table_boundary(dtype, length):
    # Initial FP32 table -> cast, versus extension in inv_freq's cast dtype.
    # Head dimension 8 has non-exact frequencies and exposes the difference.
    model = tiny_model("rope", False, dtype)
    ids = torch.tensor([[1, 4, 7, 8, 9]])[:, :length]
    with torch.no_grad():
        expected = upstream_equations(model, ids)
        actual = model(ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("position,normalized", VARIANTS)
def test_generation_cache_continuation_padding_and_beam_reorder(position, normalized):
    model = tiny_model(position, normalized)
    ids = torch.tensor([[0, 1, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11]])
    mask = ids.ne(0).long()
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    with torch.no_grad():
        full = model(ids, attention_mask=mask, position_ids=positions).logits
        pre = model(ids[:, :3], attention_mask=mask[:, :3], position_ids=positions[:, :3])
        before_head = model.lm_head.weight.detach().clone()
        # Invalid input must not poison the already-populated cache; retry it.
        with pytest.raises(ValueError, match="attention_mask"):
            model(ids[:, 3:], attention_mask=mask[:, :2], past_key_values=pre.past_key_values)
        assert pre.past_key_values.get_seq_length() == 3
        continued = model(ids[:, 3:], attention_mask=mask, position_ids=positions[:, 3:], past_key_values=pre.past_key_values)
        torch.testing.assert_close(continued.logits, full[:, 3:], rtol=2e-5, atol=2e-6)
        assert continued.past_key_values.get_seq_length() == 6
        continued.past_key_values.reorder_cache(torch.tensor([1, 0]))
        tail = torch.tensor([[12], [13]])
        continued = model(tail, attention_mask=torch.cat((mask.flip(0), torch.ones(2, 1, dtype=mask.dtype)), -1),
                          position_ids=mask.flip(0).sum(-1, keepdim=True), past_key_values=continued.past_key_values)
        full_tail = model(torch.cat((ids.flip(0), tail), -1),
                          attention_mask=torch.cat((mask.flip(0), torch.ones(2, 1, dtype=mask.dtype)), -1),
                          position_ids=torch.cat((positions.flip(0), mask.flip(0).sum(-1, keepdim=True)), -1)).logits[:, -1:]
        torch.testing.assert_close(continued.logits, full_tail, rtol=2e-5, atol=2e-6)
        # Reuse the same model for another generation after a different cache.
        generated = model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False, eos_token_id=None)
        expected = ids
        expected_mask = mask
        for _ in range(3):
            pos = (expected_mask.cumsum(-1) - 1).clamp_min(0)
            logits = model(expected, attention_mask=expected_mask, position_ids=pos, use_cache=False).logits
            expected = torch.cat((expected, logits[:, -1].argmax(-1, keepdim=True)), -1)
            expected_mask = torch.cat((expected_mask, torch.ones(2, 1, dtype=mask.dtype)), -1)
        assert torch.equal(generated, expected)
        assert model.generate(ids, attention_mask=mask, max_new_tokens=2, num_beams=2).shape[1] > ids.shape[1]
        torch.testing.assert_close(model.lm_head.weight, before_head, rtol=0, atol=0)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("position,normalized", VARIANTS)
def test_local_selection_ignores_malicious_auto_map_and_security_toggles(tmp_path, monkeypatch, trust, bypass, position, normalized):
    # The existing security bypass is explicitly part of this regression;
    # neither its value nor API trust may override definition-owned modeling.
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1" if bypass else "0")
    model = tiny_model(position, normalized)
    model.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    payload = json.loads(config_path.read_text())
    payload["auto_map"] = {"AutoConfig": "evil.EvilConfig", "AutoModelForCausalLM": "evil.EvilModel"}
    payload["architectures"] = ["EvilModel"]
    config_path.write_text(json.dumps(payload))
    (tmp_path / "evil.py").write_text("raise RuntimeError('CHECKPOINT PYTHON EXECUTED')\n")
    loader = local_model_loader("baichuan")
    assert isinstance(loader, LocalModelLoader)
    config = load_model_config(tmp_path, trust_remote_code=trust, local_files_only=True)
    assert type(config) is BaichuanConfig
    shell = loader.from_config(config, trust_remote_code=trust, code_revision="untrusted")
    assert type(shell) is BaichuanForCausalLM
    loaded = loader.from_pretrained(tmp_path, trust_remote_code=trust, local_files_only=True)
    assert type(loaded) is BaichuanForCausalLM
    ids = torch.tensor([[1, 4, 5]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids).logits, model(ids).logits, rtol=0, atol=0)
    assert not (tmp_path / "pytorch_model.bin").exists()


@pytest.mark.parametrize("position,normalized", VARIANTS)
def test_safetensors_roundtrip_exact_ownership_and_failed_load_retry(tmp_path, position, normalized):
    model = tiny_model(position, normalized)
    # First use must not mutate the Baichuan 2 head or its saved checkpoint.
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        expected = model(torch.tensor([[1, 4, 5, 6, 7]])).logits
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    model.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").exists()
    assert not list(tmp_path.glob("*.bin"))
    # A config/checkpoint mismatch fails cleanly; corrected config can retry.
    config_path = tmp_path / "config.json"
    original = config_path.read_text()
    bad = json.loads(original)
    bad["vocab_size"] += 1
    config_path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError):
        BaiChuanQModel.loader.from_pretrained(tmp_path, local_files_only=True)
    config_path.write_text(original)
    loaded, info = BaichuanForCausalLM.from_pretrained(tmp_path, local_files_only=True, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    assert loaded.config.position_embedding_type == position
    assert loaded.config.normalize_lm_head == normalized
    assert set(before) == set(loaded.state_dict())
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(loaded(torch.tensor([[1, 4, 5, 6, 7]])).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("hidden,vocab,position,normalized", [
    (4096, 64000, "rope", False), (5120, 64000, "alibi", False),
    (4096, 125696, "rope", True), (5120, 125696, "alibi", True),
])
def test_released_config_discriminators_and_original_class_aliases(hidden, vocab, position, normalized):
    config = BaichuanConfig(hidden_size=hidden, num_attention_heads=32 if hidden == 4096 else 40, vocab_size=vocab)
    assert config.position_embedding_type == position
    assert config.normalize_lm_head == normalized
    assert BaiChuanConfig is BaichuanConfig
    assert BaiChuanForCausalLM is BaichuanForCausalLM


def test_unsupported_layouts_fail_explicitly():
    with pytest.raises(ValueError, match="full multi-head"):
        tiny_config(num_key_value_heads=1)
    with pytest.raises(ValueError, match="rope or alibi"):
        BaichuanConfig(position_embedding_type="llama")
    model = tiny_model()
    with pytest.raises(TypeError, match="legacy tuple"):
        model(torch.tensor([[1]]), past_key_values=())
    with pytest.raises(TypeError, match="DynamicCache only"):
        model(torch.tensor([[1]]), past_key_values=StaticCache(config=model.config, max_cache_len=8))


@pytest.mark.parametrize("position,normalized", VARIANTS)
def test_training_loss_backward_and_packed_document_mask(position, normalized):
    model = tiny_model(position, normalized).train()
    model.config.z_loss_weight = 0.07 if normalized else 0.0
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    mask = torch.tensor([[1, 1, 1, 2, 2, 2]]) if position == "alibi" else None
    output = model(ids, attention_mask=mask, labels=ids, use_cache=False)
    expected = upstream_equations(model, ids, mask)
    torch.testing.assert_close(output.logits, expected, rtol=2e-5, atol=2e-6)
    shifted = expected[:, :-1].reshape(-1, 37)
    expected_loss = F.cross_entropy(shifted, ids[:, 1:].reshape(-1))
    expected_loss += model.config.z_loss_weight * shifted.max(-1).values.square().mean()
    torch.testing.assert_close(output.loss, expected_loss)
    output.loss.backward()
    assert torch.isfinite(model.model.layers[0].self_attn.W_pack.weight.grad).all()
    assert isinstance(model.lm_head, NormHead) == normalized
