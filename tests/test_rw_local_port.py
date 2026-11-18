# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Native Falcon, plus definition-owned translation of historical RW metadata."""

import json
import math

import pytest
import torch
from torch.nn import functional as F
from transformers import FalconConfig, FalconForCausalLM

from gptqmodel.models.auto import MODEL_MAP_CASEFOLD
from gptqmodel.models.definitions.rw import FalconQModel
from gptqmodel.models.ports.rw import RWConfig
from gptqmodel.utils.hf import load_model_config


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def legacy_payload(model_type):
    grouped = model_type.lower() == "refinedweb"
    payload = {
        "model_type": model_type, "architectures": ["RWForCausalLM"], "vocab_size": 64,
        "hidden_size": 32, "n_layer": 2, "n_head": 4, "bias": True, "alibi": False,
        "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
    }
    if grouped:
        payload["n_head_kv"] = 2
    return payload


def tiny_config(layout="mqa", **overrides):
    values = {
        "vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_kv_heads": 2 if layout == "gqa" else 4, "multi_query": layout == "mqa",
        "new_decoder_architecture": layout == "gqa", "parallel_attn": layout != "mha",
        "bias": True, "alibi": layout == "mha", "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
        "tie_word_embeddings": False,
    }
    values.update(overrides)
    config = FalconConfig(**values)
    config._attn_implementation = "eager"
    return config


def source_oracle(model, ids, mask=None, positions=None, return_attentions=False):
    """Independent source equations; never call a model attention/RoPE helper."""
    c = model.config
    state = {key: tensor.double() for key, tensor in model.state_dict().items()}
    batch, length = ids.shape
    heads, dim = c.num_attention_heads, c.head_dim
    if mask is None:
        mask = torch.ones_like(ids)
    if positions is None:
        positions = torch.arange(length)[None].expand(batch, -1)

    def norm(x, name):
        return F.layer_norm(x, (c.hidden_size,), state[name + ".weight"], state[name + ".bias"], c.layer_norm_epsilon)

    def linear(x, name):
        return x @ state[name + ".weight"].T + state.get(name + ".bias", 0)

    def rotate(x):
        frequencies = 10000.0 ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)
        angle = positions.double()[..., None] * frequencies
        angle = torch.cat([angle, angle], dim=-1)[:, None]
        half = torch.cat([-x[..., dim // 2:], x[..., :dim // 2]], dim=-1)
        return x * angle.cos() + half * angle.sin()

    hidden = F.embedding(ids, state["transformer.word_embeddings.weight"])
    attention_weights = []
    for layer in range(c.num_hidden_layers):
        prefix = f"transformer.h.{layer}."
        if c.new_decoder_architecture and c.num_ln_in_parallel_attn == 2:
            attn_input = norm(hidden, prefix + "ln_attn")
            mlp_input = norm(hidden, prefix + "ln_mlp")
        else:
            attn_input = norm(hidden, prefix + "input_layernorm")
            mlp_input = attn_input
        packed = linear(attn_input, prefix + "self_attention.query_key_value")
        queries, keys, vals = [], [], []
        # Scalar-head indexing states the released packed row layouts without
        # reusing native _split_heads or the adapter under test.
        for head in range(heads):
            if c.new_decoder_architecture:
                per_group = heads // c.num_kv_heads
                start = (head // per_group) * (per_group + 2)
                qrow, krow, vrow = start + head % per_group, start + per_group, start + per_group + 1
            elif c.multi_query:
                qrow, krow, vrow = head, heads, heads + 1
            else:
                qrow, krow, vrow = head * 3, head * 3 + 1, head * 3 + 2
            for target, row in ((queries, qrow), (keys, krow), (vals, vrow)):
                target.append(packed[..., row * dim:(row + 1) * dim])
        q, k, v = (torch.stack(x, dim=1) for x in (queries, keys, vals))
        scores = q @ k.transpose(-1, -2)
        if c.alibi:
            power = 2 ** int(math.log2(heads))
            base = 2 ** (-(2 ** -(math.log2(power) - 3)))
            slopes = [base ** exponent for exponent in range(1, power + 1)]
            if heads != power:
                extra_base = 2 ** (-(2 ** -(math.log2(2 * power) - 3)))
                slopes += [extra_base ** exponent for exponent in range(1, 2 * (heads - power), 2)]
            relative = (mask.cumsum(-1) - 1) * mask
            scores += torch.tensor(slopes, dtype=torch.float64)[None, :, None, None] * relative[:, None, None, :]
        else:
            q, k = rotate(q), rotate(k)
            scores = q @ k.transpose(-1, -2)
        allowed = torch.ones(length, length, dtype=torch.bool).tril()[None, None] & mask[:, None, None, :].bool()
        probabilities = (scores / math.sqrt(dim)).masked_fill(~allowed, torch.finfo(torch.float64).min).softmax(-1)
        attention_weights.append(probabilities)
        attention = (probabilities @ v).transpose(1, 2).reshape(batch, length, c.hidden_size)
        attention = linear(attention, prefix + "self_attention.dense")
        residual = hidden
        if not c.new_decoder_architecture and not c.parallel_attn:
            residual = residual + attention
            mlp_input = norm(residual, prefix + "post_attention_layernorm")
        mlp = linear(F.gelu(linear(mlp_input, prefix + "mlp.dense_h_to_4h")), prefix + "mlp.dense_4h_to_h")
        hidden = residual + mlp
        if c.new_decoder_architecture or c.parallel_attn:
            hidden = hidden + attention
    logits = linear(norm(hidden, "transformer.ln_f"), "lm_head")
    return (logits, attention_weights) if return_attentions else logits


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_alibi_source_equations_apply_positional_bias_once(backend):
    torch.manual_seed(940)
    config = tiny_config("mha")
    config._attn_implementation = backend
    model = FalconQModel.loader.from_config(config).double().eval()
    ids = torch.tensor([[1, 4, 6, 9, 3], [1, 3, 8, 5, 7]])
    expected = source_oracle(model, ids)
    actual = model(ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("layout", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_source_equations_dtype_cache_generation_and_safetensors(tmp_path, layout, backend, dtype):
    torch.manual_seed(941)
    bias = layout != "mqa"
    config = tiny_config(layout, bias=bias)
    config._attn_implementation = backend
    # Match HF loading: construct compute dtype without downcasting the
    # nonpersistent FP32 RoPE frequency buffer via blanket Module.to().
    model = FalconQModel.loader.from_config(config, dtype=dtype).eval()
    if not config.alibi:
        assert type(model) is FalconForCausalLM
    ids = torch.tensor([[1, 4, 6, 9, 3, 8, 5], [0, 0, 1, 5, 7, 6, 4]])
    mask = (ids != 0).long()
    reference = source_oracle(model, ids, mask)
    atol, rtol = {torch.float32: (2e-6, 2e-5), torch.float16: (8e-4, 0.02), torch.bfloat16: (0.008, 0.12)}[dtype]
    with torch.no_grad():
        full = model(ids, attention_mask=mask, use_cache=False).logits
        torch.testing.assert_close(full[mask.bool()].double(), reference[mask.bool()], atol=atol, rtol=rtol)
        cached = None
        start = 0
        for chunk in (1, 4, 1, 1):
            end = start + chunk
            out = model(ids[:, start:end], attention_mask=mask[:, :end], past_key_values=cached, use_cache=True)
            cached = out.past_key_values
            valid = mask[:, start:end].bool()
            torch.testing.assert_close(out.logits[valid], full[:, start:end][valid], atol=atol, rtol=rtol)
            assert cached.get_seq_length() == end
            start = end
        greedy = model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False)
        beam = model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False, num_beams=3)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        for cycle in range(2):
            dest = tmp_path / str(cycle)
            model.save_pretrained(dest, safe_serialization=True)
            assert list(dest.glob("*.safetensors"))
            assert not list(dest.glob("*.bin"))
            model = FalconQModel.loader.from_pretrained(dest, dtype=dtype, attn_implementation=backend).eval()
            for name, tensor in model.state_dict().items():
                assert tensor.dtype == state[name].dtype
                torch.testing.assert_close(tensor, state[name], atol=0, rtol=0)
            torch.testing.assert_close(model(ids, attention_mask=mask, use_cache=False).logits, full, atol=0, rtol=0)
            assert torch.equal(model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False), greedy)
            assert torch.equal(model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False, num_beams=3), beam)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("heads", [4, 6])
@pytest.mark.parametrize("output_attentions", [False, True])
def test_alibi_non_power_two_heads_padding_and_attention_outputs(backend, heads, output_attentions):
    torch.manual_seed(942)
    config = tiny_config("mha", hidden_size=heads * 8, num_attention_heads=heads, num_kv_heads=heads)
    config._attn_implementation = backend
    model = FalconQModel.loader.from_config(config).double().eval()
    ids = torch.tensor([[1, 4, 6, 9, 3, 8], [0, 0, 1, 5, 7, 6]])
    mask = (ids != 0).long()
    expected, expected_weights = source_oracle(model, ids, mask, return_attentions=True)
    with torch.no_grad():
        out = model(ids, attention_mask=mask, use_cache=True, output_attentions=output_attentions)
        torch.testing.assert_close(out.logits[mask.bool()], expected[mask.bool()], atol=3e-8, rtol=3e-8)
        if output_attentions:
            assert len(out.attentions) == config.num_hidden_layers
            for probs, expected_probs in zip(out.attentions, expected_weights, strict=True):
                assert probs.shape == (2, heads, 6, 6)
                assert torch.all(probs[..., 0, 1:][0] == 0)
                # Only valid queries: implementations may choose any output
                # for an entirely masked padding query.
                torch.testing.assert_close(probs.transpose(1, 2)[mask.bool()],
                                           expected_probs.transpose(1, 2)[mask.bool()], atol=3e-8, rtol=3e-8)
        past = None
        for start, end in ((0, 1), (1, 5), (5, 6)):
            step = model(ids[:, start:end], attention_mask=mask[:, :end], past_key_values=past,
                         use_cache=True, output_attentions=output_attentions)
            past = step.past_key_values
            valid = mask[:, start:end].bool()
            torch.testing.assert_close(step.logits[valid], expected[:, start:end][valid], atol=3e-8, rtol=3e-8)


def save_tiny_checkpoint(path, layout, model_type=None, hostile=False):
    """Real tiny safetensors; deliberately synthetic tokenizer, no source ID claim."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    torch.manual_seed(943)
    config = tiny_config(layout)
    model = FalconQModel.loader.from_config(config).eval()
    model.save_pretrained(path, safe_serialization=True)
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                                    **{f"t{i}": i for i in range(4, 64)}}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]",
                                       eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.save_pretrained(path)
    payload = json.loads((path / "config.json").read_text())
    if model_type and model_type != "falcon":
        payload["model_type"] = model_type
        for new, old in (("num_attention_heads", "n_head"), ("num_hidden_layers", "n_layer")):
            payload[old] = payload.pop(new)
        if model_type.lower() == "refinedweb":
            payload["n_head_kv"] = payload.pop("num_kv_heads")
        else:
            payload.pop("num_kv_heads", None)
        payload.pop("new_decoder_architecture", None)
        payload.pop("num_ln_in_parallel_attn", None)
        payload["architectures"] = ["RWForCausalLM"]
    marker = path / "EXECUTED"
    if hostile:
        payload["auto_map"] = {"AutoConfig": "untrusted.RemoteConfig", "AutoModelForCausalLM": "untrusted.RemoteModel"}
        (path / "untrusted.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\nraise RuntimeError('checkpoint Python')\n")
    (path / "config.json").write_text(json.dumps(payload))
    return model, marker


@pytest.mark.parametrize("layout,model_type", [("mha", "refinedWebModel"), ("mqa", "RefinedWebModel"),
                                               ("gqa", "refinedWeb"), ("gqa", "RefinedWeb"),
                                               ("mha", "falcon"), ("mqa", "falcon"), ("gqa", "falcon")])
@pytest.mark.parametrize("trust,bypass", [(False, None), (True, "0"), (True, "1")])
def test_public_hostile_load_model_and_tokenizer_policy(tmp_path, monkeypatch, layout, model_type, trust, bypass):
    from gptqmodel import GPTQModel

    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    reference, marker = save_tiny_checkpoint(tmp_path, layout, model_type, hostile=True)
    wrapped = GPTQModel.load(str(tmp_path), device="cpu", trust_remote_code=trust, attn_implementation="eager", dtype=torch.float32)
    assert not marker.exists()
    assert isinstance(wrapped.model, FalconForCausalLM)
    if not reference.config.alibi:
        assert type(wrapped.model) is FalconForCausalLM
    if model_type == "falcon":
        assert type(wrapped.model.config) is FalconConfig
    ids = torch.tensor([[1, 4, 9, 6]])
    with torch.no_grad():
        torch.testing.assert_close(wrapped.model(ids).logits, reference(ids).logits, atol=0, rtol=0)
    assert wrapped.tokenizer.encode("t4 t9") == [4, 9]
    assert wrapped.generate(input_ids=ids, max_new_tokens=2, do_sample=False).shape[1] >= 5
    assert not marker.exists()


@pytest.mark.parametrize("layout,alias", [("mha", "RefinedWebModel"), ("mqa", "RefinedWebModel"), ("gqa", "RefinedWeb")])
def test_failure_repair_retry_subfolder_and_assign(tmp_path, layout, alias):
    path = tmp_path / "nested"
    reference, marker = save_tiny_checkpoint(path, layout, alias, hostile=True)
    definition = MODEL_MAP_CASEFOLD[alias.lower()]
    config_path = path / "config.json"
    payload = config_path.read_text()
    broken = json.loads(payload)
    broken["num_attention_heads"] = broken["n_head"] + 1
    config_path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="Conflicting"):
        load_model_config(tmp_path, subfolder="nested")
    config_path.write_text(payload)
    config = load_model_config(tmp_path, subfolder="nested")
    weights_path = path / "model.safetensors"
    weights = weights_path.read_bytes()
    weights_path.write_bytes(b"broken")
    with pytest.raises(Exception, match="[Hh]eader|[Ss]afetensor"):
        definition.loader.from_pretrained(tmp_path, subfolder="nested", config=config)
    weights_path.write_bytes(weights)
    restored = definition.loader.from_pretrained(tmp_path, subfolder="nested", config=config).eval()
    ids = torch.tensor([[1, 4, 7]])
    with torch.no_grad():
        torch.testing.assert_close(restored(ids).logits, reference(ids).logits, atol=2e-7, rtol=1e-5)
    for assign in (False, True):
        clone = definition.loader.from_config(config)
        clone.load_state_dict(reference.state_dict(), strict=True, assign=assign)
        for key, value in clone.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[key], atol=0, rtol=0)
    assert not marker.exists()


@pytest.mark.parametrize("layout,alias,offload", [
    ("mha", "refinedWebModel", False), ("mqa", "refinedWebModel", True), ("gqa", "refinedWeb", False),
    ("mha", "falcon", True), ("mqa", "falcon", False), ("gqa", "falcon", True)])
def test_public_plain_awq_packs_every_projection_and_two_exact_reloads(tmp_path, layout, alias, offload):
    from gptqmodel import BACKEND, CalibrationConfig, GPTQModel, QuantizeConfig
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear
    from gptqmodel.quantization import FORMAT, METHOD

    original_dir = tmp_path / "original"
    reference, _ = save_tiny_checkpoint(original_dir, layout, alias)
    dense = {name: value.half().clone() for name, value in reference.state_dict().items()}
    qcfg = QuantizeConfig(bits=4, group_size=32, desc_act=False, device="cpu", method=METHOD.AWQ,
                          format=FORMAT.GEMM, fallback=None, offload_to_disk=offload,
                          offload_to_disk_path=str(tmp_path / "offload"))
    wrapped = GPTQModel.load(str(original_dir), quantize_config=qcfg, backend=BACKEND.TORCH,
                            attn_implementation="eager", dtype=torch.float16)
    layer = wrapped.model.transformer.h[0]
    assert wrapped.awq_get_modules_for_scaling(layer, {}, {}) == []
    # Actual public calibration, search-free plain quantization, and packing.
    calibration = [{"input_ids": [1, 4, 6, 9, 3, 8, 5, 7] * 4, "attention_mask": [1] * 32},
                   {"input_ids": [1, 7, 5, 9, 8, 6, 4, 3] * 4, "attention_mask": [1] * 32}]
    wrapped.quantize(CalibrationConfig(dataset=calibration, batch_size=1, min_length=1, sort="desc", concat_size=None),
                     backend=BACKEND.TORCH)
    expected = {f"transformer.h.{index}.{name}" for index in range(2) for name in (
        "self_attention.query_key_value", "self_attention.dense", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h")}
    modules = dict(wrapped.model.named_modules())
    packed = {name for name, module in modules.items() if isinstance(module, BaseQuantLinear)}
    assert packed == expected
    state = {name: value.clone() for name, value in wrapped.model.state_dict().items()}
    for key, tensor in dense.items():
        owner, leaf = key.rsplit(".", 1)
        if owner not in expected or leaf == "bias":
            torch.testing.assert_close(state[key], tensor, atol=0, rtol=0)
    ids = torch.tensor([[1, 4, 8, 7]])
    with torch.no_grad():
        logits = wrapped.model(ids, use_cache=False).logits
        assert torch.isfinite(logits).all()
    for cycle in range(2):
        dest = tmp_path / f"quantized-{cycle}"
        wrapped.save(str(dest))
        assert list(dest.glob("*.safetensors"))
        assert not list(dest.glob("*.bin"))
        wrapped = GPTQModel.load(str(dest), device="cpu", backend=BACKEND.TORCH,
                                attn_implementation="eager", dtype=torch.float16)
        restored = wrapped.model.state_dict()
        assert restored.keys() == state.keys()
        for key, tensor in state.items():
            torch.testing.assert_close(restored[key], tensor, atol=0, rtol=0)
        with torch.no_grad():
            torch.testing.assert_close(wrapped.model(ids, use_cache=False).logits, logits, atol=0, rtol=0)


@pytest.mark.parametrize("model_type", ["refinedWeb", "RefinedWeb", "refinedWebModel", "RefinedWebModel"])
def test_historical_config_selects_exact_native_geometry(tmp_path, model_type):
    (tmp_path / "config.json").write_text(json.dumps(legacy_payload(model_type)))
    config = load_model_config(tmp_path, trust_remote_code=False)
    assert isinstance(config, FalconConfig)
    assert (config.num_hidden_layers, config.num_attention_heads) == (2, 4)
    assert config.new_decoder_architecture is (model_type.lower() == "refinedweb")
    assert config.multi_query is False
    assert config.parallel_attn is (model_type.lower() == "refinedweb")
    model = MODEL_MAP_CASEFOLD[model_type.lower()].loader.from_config(config).eval()
    assert type(model) is FalconForCausalLM
    assert model.transformer.h[0].self_attention.query_key_value.weight.shape == (
        64 if model_type.lower() == "refinedweb" else 96, 32,
    )
    assert model(torch.tensor([[1, 3, 4]]), use_cache=True).logits.shape == (1, 3, 64)


@pytest.mark.parametrize("payload,message", [
    ({"model_type": "foreign"}, "Unqualified"),
    ({"model_type": "RefinedWeb", "n_head": 4, "n_head_kv": 3}, "divisor"),
    ({"model_type": "RefinedWebModel", "n_head": 4, "n_head_kv": 2}, "MHA/MQA"),
    ({"model_type": "RefinedWeb", "new_decoder_architecture": False}, "requires"),
    ({"model_type": "RefinedWebModel", "num_ln_in_parallel_attn": 2}, "requires"),
    ({"model_type": "RefinedWebModel", "n_head": 0}, "positive"),
    ({"model_type": "RefinedWebModel", "hidden_size": 32, "ffn_hidden_size": 48}, "ffn_hidden_size"),
])
def test_unqualified_or_conflicting_legacy_geometry_fails_closed(payload, message):
    with pytest.raises(ValueError, match=message):
        RWConfig(**payload)


@pytest.mark.parametrize("alias", ["RefinedWebModel", "RefinedWeb"])
def test_source_defaults_old_embed_and_canonical_config_save(tmp_path, alias):
    config = RWConfig(model_type=alias, n_embed=48, n_head=6)
    assert config.hidden_size == 48
    assert config.num_hidden_layers == 2
    assert config.multi_query is False
    assert config.vocab_size == 250880
    assert config.num_kv_heads == 6
    assert config.bos_token_id == 1 and config.eos_token_id == 2
    config.save_pretrained(tmp_path)
    restored = load_model_config(tmp_path)
    assert type(restored) is FalconConfig
    assert restored.model_type == "falcon"
    assert restored.num_ln_in_parallel_attn == (2 if alias == "RefinedWeb" else 1)
    assert restored.new_decoder_architecture is (alias == "RefinedWeb")


@pytest.mark.parametrize("kv", [1, 4])
def test_legacy_mqa_accepts_only_equivalent_explicit_kv_metadata(kv):
    config = RWConfig(model_type="RefinedWebModel", hidden_size=32, n_head=4, n_layer=1,
                      multi_query=True, n_head_kv=kv)
    model = MODEL_MAP_CASEFOLD["refinedwebmodel"].loader.from_config(config)
    assert model.transformer.h[0].self_attention.query_key_value.weight.shape == (48, 32)
    with pytest.raises(ValueError, match="MHA/MQA"):
        RWConfig(model_type="RefinedWebModel", hidden_size=32, n_head=4, multi_query=True, n_head_kv=2)
