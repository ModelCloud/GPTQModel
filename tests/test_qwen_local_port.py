# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Original Qwen real equations, checkpoint transitions, and public consumers."""

import hashlib
import json
import math
import unicodedata
from pathlib import Path

import pytest
import tiktoken
import torch
from safetensors import SafetensorError
from torch.nn import functional as F
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

from gptqmodel import CalibrationConfig, GPTQModel
from gptqmodel.models.definitions.qwen import QwenQModel
from gptqmodel.models.ports.qwen.configuration_qwen import QWenConfig
from gptqmodel.models.ports.qwen.modeling_qwen import (
    QWenLegacyCache,
    QWenLMHeadModel,
    StopWordsLogitsProcessor,
    apply_rotary_pos_emb,
)
from gptqmodel.models.ports.qwen.tokenization_qwen import PAT_STR, SPECIAL_TOKENS, QWenTokenizer, make_context
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig
from gptqmodel.utils.backend import BACKEND
from tests.port_assets import read_port_asset


ASSETS = Path(__file__).resolve().parents[1] / "gptqmodel/models/ports/qwen"
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def tiny_config(**kwargs):
    values = {
        "vocab_size": 64,
        "hidden_size": 32,
        "kv_channels": 8,
        "num_attention_heads": 4,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "seq_length": 8,
        "max_position_embeddings": 32,
        "use_dynamic_ntk": False,
        "use_logn_attn": False,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "use_flash_attn": False,
        "attn_implementation": "eager",
    }
    values.update(kwargs)
    return QWenConfig(**values)


def tiny_model(**kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = QWenLMHeadModel(tiny_config(**kwargs)).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "ln_" in name:
                    parameter.uniform_(0.6, 1.4)
                else:
                    parameter.normal_(0.03 if name.endswith("bias") else 0.0, 0.13)
    return model


def oracle(model, ids, alpha=None, mask=None, attention_bias=None):
    """Independent float64 original decoder equations; never calls port modules."""
    c, s = model.config, {k: v.double() for k, v in model.state_dict().items()}
    batch, length = ids.shape
    if alpha is None:
        sizes = mask.sum(-1).double() if mask is not None and length > c.seq_length else torch.full((batch,), length)
        alpha = (2 ** torch.ceil(torch.log2(sizes / c.seq_length) + 1) - 1).clamp_min(1)
        if not c.use_dynamic_ntk:
            alpha.fill_(1)
    x = s["transformer.wte.weight"][ids]

    def norm(x, name):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + c.layer_norm_epsilon) * s[name + ".weight"]

    def linear(x, name):
        return F.linear(x, s[name + ".weight"], s.get(name + ".bias"))

    dim = int(c.kv_channels * c.rotary_pct)
    positions = torch.arange(length, dtype=torch.float64)
    bases = c.rotary_emb_base * alpha.double() ** (dim / (dim - 2))
    angles = positions[None, :, None] / bases[:, None, None] ** (torch.arange(0, dim, 2).double() / dim)
    cos, sin = angles.cos()[:, :, None], angles.sin()[:, :, None]

    def rotate(t):
        a, b = t[..., : dim // 2], t[..., dim // 2 : dim]
        return torch.cat((a * cos - b * sin, b * cos + a * sin, t[..., dim:]), -1)

    caches = []
    for layer in range(c.num_hidden_layers):
        p = f"transformer.h.{layer}"
        q, k, v = [
            t.view(batch, length, c.num_attention_heads, c.kv_channels)
            for t in linear(norm(x, p + ".ln_1"), p + ".attn.c_attn").chunk(3, -1)
        ]
        q, k = rotate(q), rotate(k)
        caches.append((k.transpose(1, 2), v.transpose(1, 2)))
        if c.use_logn_attn and length > c.seq_length:
            gain = (torch.log(positions + 1) / math.log(c.seq_length)).clamp_min(1)
            q = q * gain[None, :, None, None]
        scores = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / math.sqrt(c.kv_channels)
        allowed = torch.ones(length, length, dtype=torch.bool).tril()[None, None]
        if mask is not None:
            allowed = allowed & mask[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, -1e30)
        if attention_bias is not None:
            scores = scores + attention_bias.double()
        attn = (scores.softmax(-1) @ v.transpose(1, 2)).transpose(1, 2).reshape(batch, length, c.hidden_size)
        x = x + linear(attn, p + ".attn.c_proj")
        n = norm(x, p + ".ln_2")
        x = x + linear(linear(n, p + ".mlp.w1") * F.silu(linear(n, p + ".mlp.w2")), p + ".mlp.c_proj")
    return linear(norm(x, "transformer.ln_f"), "lm_head"), caches


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_additive_attention_bias_preserved(implementation):
    model = tiny_model(attn_implementation=implementation)
    assert model.config._attn_implementation == implementation
    ids = torch.tensor([[2, 3, 4, 5]])
    bias = torch.zeros(1, 1, 4, 4)
    bias[:, :, :, 0] = 0.25
    bias[:, :, :, 1] = -1.0
    expected, _ = oracle(model, ids, attention_bias=bias)
    with torch.inference_mode():
        actual = model(ids, attention_mask=bias).logits
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("no_bias", [False, True])
@pytest.mark.parametrize("rotary_pct", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("dynamic", [False, True])
def test_source_equations_partial_rotary_ntk_logn_and_exact_cache(no_bias, rotary_pct, dynamic):
    model = tiny_model(
        no_bias=no_bias,
        rotary_pct=rotary_pct,
        use_dynamic_ntk=dynamic,
        use_logn_attn=True,
        seq_length=4,
        scale_attn_weights=False,
    )
    ids = torch.tensor([[2, 3, 5, 8, 7, 10, 11, 13, 14], [3, 6, 9, 2, 4, 5, 12, 13, 14]])
    mask = torch.ones_like(ids)
    mask[1, :5] = 0
    expected, keys = oracle(model, ids, mask=mask)
    with torch.inference_mode():
        out = model(ids, attention_mask=mask, output_attentions=True, output_hidden_states=True)
    torch.testing.assert_close(out.logits[mask.bool()].double(), expected[mask.bool()], rtol=2e-5, atol=2e-6)
    assert len(out.attentions) == 2 and len(out.hidden_states) == 3
    for layer, (key, value) in zip(out.past_key_values.layers, keys):
        # Padding rows of later layers differ only at all-masked softmax rows.
        valid = mask[:, None, :, None].expand_as(key).bool()
        torch.testing.assert_close(layer.keys.double()[valid], key[valid], rtol=3e-5, atol=2e-6)
        torch.testing.assert_close(layer.values.double()[valid], value[valid], rtol=3e-5, atol=2e-6)
    assert torch.equal(out.past_key_values.ntk_alpha, torch.tensor([7.0, 1.0]) if dynamic else torch.ones(2))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("rotary_pct", [0.5, 1.0])
def test_dtype_causal_multitoken_cache_continuation_greedy_beam(dtype, implementation, rotary_pct):
    model = tiny_model(rotary_pct=rotary_pct, no_bias=False, attn_implementation=implementation).to(dtype)
    assert model.config._attn_implementation == implementation
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8]])
    tol = {torch.float32: 2e-6, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    with torch.inference_mode():
        full = model(ids).logits
        cache = None
        for start, stop in [(0, 1), (1, 5), (5, 6), (6, 7)]:
            out = model(ids[:, start:stop], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            torch.testing.assert_close(out.logits, full[:, start:stop], rtol=tol, atol=tol)
            assert cache.get_seq_length() == stop
        changed = ids.clone()
        changed[:, -1] = 20
        torch.testing.assert_close(model(changed).logits[:, :-1], full[:, :-1], rtol=0, atol=0)
        for beams in [1, 3]:
            cached = model.generate(ids, max_new_tokens=3, num_beams=beams, do_sample=False)
            fresh = model.generate(ids, max_new_tokens=3, num_beams=beams, do_sample=False, use_cache=False)
            assert torch.equal(cached, fresh)


def test_ntk_state_owned_by_cache_interleaving_reorder_and_failure_retry():
    model = tiny_model(use_dynamic_ntk=True, use_logn_attn=True, seq_length=4)
    a = torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
    with torch.inference_mode():
        cache = model(a, attention_mask=mask).past_key_values
        assert cache.ntk_alpha.tolist() == [3, 1]
        model(torch.tensor([[4, 5, 6]]))  # A second request must not replace A's bases.
        before = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
        with pytest.raises(ValueError, match="cached prefix"):
            model(torch.tensor([[14], [15]]), attention_mask=torch.ones(2, 1), past_key_values=cache)
        assert cache.get_seq_length() == 6
        for layer, (k, v) in zip(cache.layers, before):
            assert torch.equal(layer.keys, k) and torch.equal(layer.values, v)
        ext = torch.tensor([[14, 15], [15, 16]])
        combined = torch.cat((a, ext), -1)
        extmask = torch.cat((mask, torch.ones(2, 2, dtype=torch.long)), -1)
        expected, _ = oracle(model, combined, alpha=cache.ntk_alpha, mask=extmask)
        output = model(ext, attention_mask=extmask, past_key_values=cache)
        torch.testing.assert_close(output.logits.double(), expected[:, -2:], rtol=2e-5, atol=2e-6)
        cache.reorder_cache(torch.tensor([1, 0, 1]))
        assert cache.ntk_alpha.tolist() == [1, 3, 1]
        cache.batch_repeat_interleave(2)
        assert cache.ntk_alpha.tolist() == [1, 1, 3, 3, 1, 1]
        cache.batch_select_indices(torch.tensor([2, 0]))
        assert cache.ntk_alpha.tolist() == [3, 1]
        cache.reset()
        assert cache.get_seq_length() == 0 and cache.ntk_alpha is None


def test_source_legacy_cache_and_inputs_embeds_tuple_loss():
    model = tiny_model()
    ids = torch.tensor([[2, 3, 4, 5]])
    with torch.inference_mode():
        first = model(ids[:, :2])
        cache = QWenLegacyCache(
            [(l.keys.transpose(1, 2), l.values.transpose(1, 2)) for l in first.past_key_values.layers],
            first.past_key_values.ntk_alpha,
        )
        out = model(ids[:, 2:], past_key_values=cache)
        assert isinstance(out.past_key_values, QWenLegacyCache)
        torch.testing.assert_close(out.logits, model(ids).logits[:, 2:], rtol=2e-6, atol=2e-6)
        a = model(inputs_embeds=model.get_input_embeddings()(ids), labels=ids)
        b = model(ids, return_dict=False, labels=ids, output_hidden_states=True)
        torch.testing.assert_close(a.logits, b[1], rtol=0, atol=0)
        torch.testing.assert_close(a.loss, F.cross_entropy(a.logits[:, :-1].reshape(-1, 64), ids[:, 1:].reshape(-1)))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tied", [False, True])
def test_two_safetensors_cycles_corrupt_load_repair_and_assign(tmp_path, dtype, tied):
    model = tiny_model(tie_word_embeddings=tied).to(dtype)
    ids = torch.tensor([[2, 5, 3, 7]])
    with torch.inference_mode():
        expected = model(ids).logits
    for cycle in range(2):
        dst = tmp_path / str(cycle)
        model.save_pretrained(dst)
        assert list(dst.glob("*.safetensors")) and not list(dst.glob("*.bin"))
        weight = dst / "model.safetensors"
        data = weight.read_bytes()
        weight.write_bytes(b"broken")
        with pytest.raises(SafetensorError):
            QwenQModel.loader.from_pretrained(dst, dtype=dtype)
        weight.write_bytes(data)
        loaded = QwenQModel.loader.from_pretrained(dst, dtype=dtype, attn_implementation="eager").eval()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(loaded(ids).logits, expected, rtol=0, atol=0)
        assert (loaded.lm_head.weight is loaded.transformer.wte.weight) == tied
        model = loaded
    with torch.device("meta"):
        shell = QWenLMHeadModel(model.config)
    shell.load_state_dict(model.state_dict(), assign=True)
    with torch.inference_mode():
        torch.testing.assert_close(shell.eval()(ids).logits, expected, rtol=0, atol=0)


def checkpoint(path, release="Qwen-7B", **kwargs):
    path.mkdir(parents=True, exist_ok=True)
    model = tiny_model(vocab_size=151936, eos_token_id=151643, pad_token_id=151643, **kwargs)
    metadata = json.loads((ASSETS / "fixtures/released_metadata.json").read_text())[release]
    model.generation_config = GenerationConfig.from_dict(metadata["generation_config.json"])
    model.save_pretrained(path)
    (path / "qwen.tiktoken").write_bytes(read_port_asset(ASSETS / "fixtures/qwen.tiktoken"))
    marker = path / "EXECUTED"
    payload = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\nraise RuntimeError('checkpoint Python')\n"
    (path / "hostile.py").write_text(payload)
    custom = path / "custom_generate"
    custom.mkdir(exist_ok=True)
    (custom / "generate.py").write_text(payload)
    cfg = json.loads((path / "config.json").read_text())
    cfg["auto_map"] = {"AutoConfig": "hostile.Config", "AutoModelForCausalLM": "hostile.Model"}
    (path / "config.json").write_text(json.dumps(cfg))
    tokenizer_cfg = metadata["tokenizer_config.json"]
    tokenizer_cfg["auto_map"] = {"AutoTokenizer": ["hostile.Tokenizer", None]}
    (path / "tokenizer_config.json").write_text(json.dumps(tokenizer_cfg))
    return model, marker


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
@pytest.mark.parametrize("release", ["Qwen-7B", "Qwen-7B-Chat"])
def test_public_hostile_load_tokenizer_generate_and_two_saves(tmp_path, monkeypatch, trust, bypass, release):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    path = tmp_path / "source"
    source, marker = checkpoint(path, release=release)
    for cycle in range(3):
        loaded = GPTQModel.load(
            str(path), device="cpu", dtype=torch.float32, trust_remote_code=trust, attn_implementation="eager"
        )
        assert type(loaded.model) is QWenLMHeadModel and type(loaded.model.config) is QWenConfig
        tokenizer = loaded.tokenizer
        assert tokenizer.encode("hello 你好 e\u0301<|im_start|>") == [14990, 220, 108386, 3958, 151644]
        assert tokenizer.eod_id == tokenizer.pad_token_id == 151643
        assert tokenizer.eos_token_id is None  # The actual source class declares eod_id, not an EOS alias.
        assert loaded.model.generation_config.chat_format == ("chatml" if release.endswith("Chat") else "raw")
        assert loaded.model.generation_config.do_sample and loaded.model.generation_config.top_p == 0.8
        assert loaded.model.generation_config.eos_token_id == loaded.model.generation_config.pad_token_id == 151643
        with torch.inference_mode():
            ids = torch.tensor([[2, 5, 3, 7]])
            torch.testing.assert_close(loaded.model(ids).logits, source(ids).logits, rtol=0, atol=0)
            assert loaded.model.generate(ids, do_sample=False, max_new_tokens=2).shape == (1, 6)
        assert not marker.exists()
        if cycle < 2:
            path = tmp_path / f"save{cycle}"
            loaded.model.save_pretrained(path)
            loaded.tokenizer.save_pretrained(path)
    del source, loaded


@pytest.mark.parametrize("no_bias", [False, True])
@pytest.mark.parametrize("offload", [False, True])
def test_public_plain_awq_all_linears_dense_exclusions_two_reloads(tmp_path, no_bias, offload):
    path = tmp_path / "source"
    source, marker = checkpoint(path, no_bias=no_bias)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=offload)
    model = GPTQModel.load(
        str(path), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16, attn_implementation="eager"
    )
    assert model.awq_get_modules_for_scaling(model.model.transformer.h[0], {}, {}) == []
    linears = {
        f"transformer.h.{i}.{name}"
        for i in range(2)
        for name in ["attn.c_attn", "attn.c_proj", "mlp.w1", "mlp.w2", "mlp.c_proj"]
    }
    dense = {
        k: v.to(torch.float16).clone()
        for k, v in source.state_dict().items()
        if k.rsplit(".", 1)[0] not in linears or k.endswith("bias")
    }
    rows = [
        {"input_ids": torch.tensor([[2, 3, 5, 8, 7, 4, 6, 9]]), "attention_mask": torch.ones(1, 8, dtype=torch.long)},
        {"input_ids": torch.tensor([[8, 3, 2, 4, 5, 7, 9, 6]]), "attention_mask": torch.ones(1, 8, dtype=torch.long)},
    ]
    model.quantize(
        CalibrationConfig(dataset=rows, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    assert {name for name, module in model.model.named_modules() if isinstance(module, AwqTorchLinear)} == linears
    for name in linears:
        assert model.model.get_submodule(name).qweight.dtype == torch.int32
        assert not hasattr(model.model.get_submodule(name), "weight")
    packed = {k: v.clone() for k, v in model.model.state_dict().items()}
    for key, tensor in dense.items():
        torch.testing.assert_close(packed[key], tensor, rtol=0, atol=0)
    with torch.inference_mode():
        logits = model.model(**rows[0], use_cache=False).logits
        greedy = model.model.generate(**rows[0], max_new_tokens=2, do_sample=False)
    assert torch.isfinite(logits).all()
    for cycle in range(2):
        dst = tmp_path / f"packed{cycle}"
        model.save(str(dst))
        assert list(dst.glob("*.safetensors")) and not list(dst.glob("*.bin"))
        model = GPTQModel.load(
            str(dst), device="cpu", backend=BACKEND.TORCH, dtype=torch.float16, attn_implementation="eager"
        )
        assert set(model.model.state_dict()) == set(packed)
        for key, tensor in packed.items():
            torch.testing.assert_close(model.model.state_dict()[key], tensor, rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(model.model(**rows[0], use_cache=False).logits, logits, rtol=0, atol=0)
            assert torch.equal(model.model.generate(**rows[0], max_new_tokens=2, do_sample=False), greedy)
        assert model.tokenizer.encode("hello 你好 e\u0301<|im_start|>") == [14990, 220, 108386, 3958, 151644]
    assert not marker.exists()


def test_real_tokenizer_source_ids_decode_prompts_repair_retry(tmp_path):
    path = tmp_path / "qwen.tiktoken"
    data = read_port_asset(ASSETS / "fixtures/qwen.tiktoken")
    path.write_bytes(data)
    tok = QWenTokenizer(str(path), eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    # Independent backend from actual pinned release data, not tokenizer helpers.
    import base64

    ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in data.splitlines())}
    ref = tiktoken.Encoding("source", pat_str=PAT_STR, mergeable_ranks=ranks, special_tokens=SPECIAL_TOKENS)
    golden = json.loads((ASSETS / "fixtures/tokenizer_oracles.json").read_text())
    for item in golden["texts"]:
        assert tok.encode(item["text"]) == item["ids"]
        assert tok.decode(item["ids"]) == item["decode"]
    for item in golden["contexts"]:
        raw, tokens = make_context(tok, **item["kwargs"])
        assert raw == item["raw"] and tokens == item["ids"]
    for text in ["hello 你好 e\u0301<|im_start|>", "\x00\t😀\n  spaces", "привет العربية\n12345", "<|extra_204|>"]:
        expected = ref.encode(unicodedata.normalize("NFC", text), allowed_special="all")
        assert tok.encode(text) == expected
        assert tok.decode(expected) == unicodedata.normalize("NFC", text)
    raw, context = make_context(tok, "你好", [("hello", "world")], system="system")
    assert context == ref.encode(raw, allowed_special="all")
    raw2, context2 = make_context(tok, "你好", [("hello", "world")], system="system", max_window_size=1)
    assert "hello" not in raw2 and context2 == ref.encode(raw2, allowed_special="all")
    tok.save_pretrained(tmp_path / "saved")
    path.write_bytes(b"invalid")
    with pytest.raises(RuntimeError, match="vocabulary"):
        QWenTokenizer(str(path))
    path.write_bytes(data)
    assert QWenTokenizer(str(path)).encode(raw) == context
    for _ in range(2):
        tok = QWenTokenizer.from_pretrained(tmp_path / "saved")
        assert tok.encode(raw) == context
        assert tok.decode([151643, 14990, 151644], skip_special_tokens=True) == "hello"
        tok.save_pretrained(tmp_path / "saved")


def test_stop_suffix_source_policy_and_native_qwen_siblings():
    scores = torch.zeros(2, 10)
    processor = StopWordsLogitsProcessor([[2, 3], [1]], 1)
    result = processor(torch.tensor([[2, 3], [3, 4]]), scores)
    assert result[0, 1] == 2**15 and torch.equal(result[1], torch.zeros(10))
    model = tiny_model()
    ids = torch.tensor([[2, 3]])
    assert model.generate(ids, stop_words_ids=[[2, 3]], max_new_tokens=3, do_sample=False).tolist() == [[2, 3, 1]]
    for cls, config in [(Qwen2ForCausalLM, Qwen2Config), (Qwen3ForCausalLM, Qwen3Config)]:
        c = config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
        native = cls(c).eval()
        assert type(native).__module__.startswith("transformers.models.")
        assert native(ids).logits.shape == (1, 2, 64)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rotary_pct", [0.5, 1.0])
def test_rotary_source_fp32_calculation_near_source_table_boundary(dtype, rotary_pct):
    config = tiny_config(rotary_pct=rotary_pct)
    torch.manual_seed(31)
    inputs = torch.randn(2, 4, 3, 8).to(dtype)
    positions = torch.tensor([32764, 32765, 32766])
    alpha = torch.tensor([1.0, 7.0])
    dim = int(config.kv_channels * rotary_pct)
    reference = inputs.double().clone()
    for batch in range(2):
        base = config.rotary_emb_base * float(alpha[batch]) ** (dim / (dim - 2))
        for seq, pos in enumerate(positions.tolist()):
            for channel in range(dim // 2):
                angle = pos / base ** (2 * channel / dim)
                a = inputs[batch, :, seq, channel].double()
                b = inputs[batch, :, seq, channel + dim // 2].double()
                reference[batch, :, seq, channel] = a * math.cos(angle) - b * math.sin(angle)
                reference[batch, :, seq, channel + dim // 2] = b * math.cos(angle) + a * math.sin(angle)
    actual = apply_rotary_pos_emb(inputs, positions, alpha, config)
    tolerance = {torch.float32: 5e-4, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(actual.double(), reference, rtol=tolerance, atol=tolerance)
    assert torch.equal(actual[..., dim:], inputs[..., dim:])


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_missing_corrupt_tokenizer_repair_retry(tmp_path, monkeypatch, trust, bypass):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    source, marker = checkpoint(tmp_path)
    vocab = tmp_path / "qwen.tiktoken"
    data = vocab.read_bytes()
    for invalid in [None, b"invalid base64 rank\n"]:
        if invalid is None:
            vocab.unlink()
        else:
            vocab.write_bytes(invalid)
        with pytest.raises((RuntimeError, OSError), match="vocabulary|qwen.tiktoken|vocab_file"):
            GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
        assert not marker.exists()
        vocab.write_bytes(data)
        loaded = GPTQModel.load(
            str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust, attn_implementation="eager"
        )
        assert loaded.tokenizer.encode("hello 你好") == [14990, 220, 108386]
        with torch.inference_mode():
            torch.testing.assert_close(
                loaded.model(torch.tensor([[2, 3]])).logits, source(torch.tensor([[2, 3]])).logits, rtol=0, atol=0
            )
    assert not marker.exists()


def test_extra_vocabulary_source_tokens_roundtrip_and_collision_policy(tmp_path):
    vocab = tmp_path / "qwen.tiktoken"
    vocab.write_bytes(read_port_asset(ASSETS / "fixtures/qwen.tiktoken"))
    extra = tmp_path / "extra"
    import base64

    extra.write_bytes(
        base64.b64encode(b"unique_extra") + b" 151851\n" + base64.b64encode(b"unique_extra_collision") + b" 151643\n"
    )
    tok = QWenTokenizer(str(vocab), extra_vocab_file=str(extra))
    assert tok._convert_token_to_id(b"unique_extra") == 151851
    assert b"unique_extra_collision" not in tok.mergeable_ranks
    tok.save_pretrained(tmp_path / "saved")
    loaded = QWenTokenizer.from_pretrained(tmp_path / "saved")
    assert loaded._convert_token_to_id(b"unique_extra") == 151851
    assert tok.encode("hello 你好") == loaded.encode("hello 你好") == [14990, 220, 108386]


def test_pinned_release_vocab_and_fixture_license_integrity():
    manifest = json.loads((ASSETS / "provenance.json").read_text())
    vocab = read_port_asset(ASSETS / "fixtures/qwen.tiktoken")
    digest = hashlib.sha256(vocab).hexdigest()
    assert digest == "b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186"
    for release in manifest["releases"].values():
        assert release["files"]["qwen.tiktoken"] == digest
    for group in ("fixture_files", "license_files"):
        for path, expected in manifest[group].items():
            source = ASSETS / path
            data = source.read_bytes() if source.is_file() else read_port_asset(source)
            assert hashlib.sha256(data).hexdigest() == expected


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"num_key_value_heads": 1}, "MHA"),
        ({"kv_channels": 16}, "hidden_size"),
        ({"rotary_pct": 0.25}, "rotary"),
        ({"use_cache_kernel": True}, "cache kernels"),
        ({"use_cache_quantization": True}, "uint8 Qwen cache"),
        ({"use_flash_attn": True}, "FlashAttention"),
    ],
)
def test_unqualified_modes_fail_explicitly_and_ordinary_config_retries(kwargs, match):
    with pytest.raises(ValueError, match=match):
        tiny_model(**kwargs)
    assert tiny_model()(torch.tensor([[2, 3]])).logits.shape == (1, 2, 64)
