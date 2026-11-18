# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real TeleChat2 ownership, numerical and lifecycle regressions."""

import json
import math
import shutil
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from gptqmodel.models.ports.telechat2.configuration_telechat2 import Telechat2Config
from gptqmodel.models.ports.telechat2.modeling_telechat2 import Telechat2ForCausalLM
from gptqmodel.models.ports.telechat2.tokenization_telechat2 import Telechat2Tokenizer
from gptqmodel.utils.hf import load_model_config
from tests.port_assets import read_port_asset


FIXTURES = Path(__file__).parents[1] / "gptqmodel/models/ports/telechat2/fixtures"


def _config(**kwargs):
    values = {"vocab_size": 41, "hidden_size": 32, "n_head": 4, "n_layer": 2, "ffn_hidden_size": 64,
              "training_seqlen": 8, "base_seqlen": 8, "attn_implementation": "eager"}
    values.update(kwargs)
    return Telechat2Config(**values)


def _model(**kwargs):
    torch.manual_seed(713)
    return Telechat2ForCausalLM(_config(**kwargs)).eval()


def _assets(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "tokenizer.model").write_bytes(read_port_asset(FIXTURES / "tokenizer.model"))
    shutil.copyfile(FIXTURES / "tokenizer_config.json", path / "tokenizer_config.json")


@pytest.mark.parametrize("variant", ["modern", "115B"])
@pytest.mark.parametrize("bos,eos", [(False, False), (True, False), (False, True), (True, True)])
def test_real_source_tokenizer_ids_chat_masks_and_deleted_source_saves(tmp_path, variant, bos, eos):
    oracle = json.loads((FIXTURES / "tokenizer_oracle.json").read_text())[variant]
    _assets(tmp_path)
    if variant == "115B":
        (tmp_path / "tokenizer.model").write_bytes(read_port_asset(FIXTURES / "115B-tokenizer.model"))
        shutil.copyfile(FIXTURES / "115B-tokenizer_config.json", tmp_path / "tokenizer_config.json")
    original = (tmp_path / "tokenizer.model").read_bytes()
    tokenizer = Telechat2Tokenizer.from_pretrained(tmp_path, add_bos_token=bos, add_eos_token=eos)
    assert len(tokenizer) == oracle["vocab_size"]
    for case in oracle["cases"]:
        assert tokenizer.encode(case["text"], add_special_tokens=False) == case["ids"]
        assert tokenizer.decode(case["ids"]) == case["decoded"]
        expected = ([tokenizer.bos_token_id] if bos else []) + case["ids"] + ([tokenizer.eos_token_id] if eos else [])
        assert tokenizer.encode(case["text"]) == expected
    if "chat" in oracle:
        assert tokenizer.apply_chat_template(oracle["messages"], tokenize=False, add_generation_prompt=True) == oracle["chat"]
        assert tokenizer.apply_chat_template(oracle["messages"], tokenize=True, add_generation_prompt=True,
                                             return_dict=False) == oracle["chat_ids"]
    pair = tokenizer.build_inputs_with_special_tokens([50, 51], [52])
    assert tokenizer.get_special_tokens_mask([50, 51], [52]) == ([1] * bos + [0, 0] + [1] * eos + [1] * bos + [0] + [1] * eos)
    assert tokenizer.create_token_type_ids_from_sequences([50, 51], [52]) == [0] * (2 + bos + eos) + [1] * (1 + bos + eos)
    assert len(pair) == 3 + 2 * (bos + eos)
    (tmp_path / "tokenizer.model").unlink()
    for cycle in range(2):
        dest = tmp_path / str(cycle)
        tokenizer.save_pretrained(dest)
        assert (dest / "tokenizer.model").read_bytes() == original
        tokenizer = Telechat2Tokenizer.from_pretrained(dest)
        assert (tokenizer.add_bos_token, tokenizer.add_eos_token) == (bos, eos)
        for case in oracle["cases"]:
            assert tokenizer.encode(case["text"], add_special_tokens=False) == case["ids"]


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("variant", ["modern", "115B", "telechat1"])
def test_real_tokenizer_asset_failure_repair_and_subfolder(tmp_path, variant, trust):
    from gptqmodel.utils.hf import load_hf_tokenizer
    folder = tmp_path / "nested"
    _assets(folder)
    if variant == "115B":
        (folder / "tokenizer.model").write_bytes(read_port_asset(FIXTURES / "115B-tokenizer.model"))
        shutil.copyfile(FIXTURES / "115B-tokenizer_config.json", folder / "tokenizer_config.json")
    elif variant == "telechat1":
        (folder / "tokenizer.model").unlink()
        for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
            (folder / name).write_bytes(read_port_asset(FIXTURES / ("telechat1-" + name)))
    source = folder / ("tokenizer.json" if variant == "telechat1" else "tokenizer.model")
    original = source.read_bytes()
    source.write_bytes(b"corrupt")
    with pytest.raises((RuntimeError, ValueError)):
        load_hf_tokenizer(str(tmp_path), subfolder="nested", model_config=_config(), trust_remote_code=trust)
    source.write_bytes(original)
    tokenizer = load_hf_tokenizer(str(tmp_path), subfolder="nested", model_config=_config(), trust_remote_code=trust)
    oracle = json.loads((FIXTURES / "tokenizer_oracle.json").read_text())[variant]
    for case in oracle["cases"]:
        assert tokenizer.encode(case["text"], add_special_tokens=False) == case["ids"]
    assert source.read_bytes() == original


def _source_forward(state, config, ids, past=None, padding=None):
    """Independent equations from pinned Tele-AI eager code; no port helpers."""
    batch, length = ids.shape
    offset = 0 if past is None else past[0][0].shape[2]
    x = F.embedding(ids, state["transformer.word_embeddings.weight"])
    def norm(value, name):
        fp32 = value.float()
        return state[name + ".weight"] * (fp32 * torch.rsqrt(
            fp32.pow(2).mean(-1, keepdim=True) + config.layer_norm_epsilon)).to(value.dtype)
    if config.embed_layernorm:
        x = norm(x, "transformer.word_embeddings_layernorm")
    result, all_states, all_probs = [], [x], []
    for index in range(config.n_layer):
        root = f"transformer.h.{index}."
        def linear(value, suffix):
            return F.linear(value, state[root + suffix + ".weight"], state.get(root + suffix + ".bias"))
        normalized = norm(x, root + "input_layernorm")
        residual = normalized if config.apply_residual_connection_post_layernorm else x
        heads, kvheads, dim = config.n_head, config.num_key_value_heads, config.hidden_size // config.n_head
        q = linear(normalized, "self_attention.query").reshape(batch, length, heads, dim).transpose(1, 2)
        combined = linear(normalized, "self_attention.key_value").reshape(batch, length, kvheads, 2, dim)
        k, v = combined[:, :, :, 0].transpose(1, 2), combined[:, :, :, 1].transpose(1, 2)
        seq = max(length + offset, config.training_seqlen)
        alpha = max(2 ** math.ceil(math.log(seq / config.base_seqlen, 2) + 1) - 1, 1)
        base = config.rope_theta * alpha ** (dim / (dim - 2))
        frequencies = 1 / base ** (torch.arange(0, dim, 2).float() / dim)
        angles = torch.arange(offset, offset + length).float()[:, None] * frequencies[None, :]
        angles = angles.repeat(1, 2)
        gain = 1 if seq <= config.training_seqlen else 1 + 0.1 * math.log(seq / config.training_seqlen)
        cs, sn = gain * angles.cos().half(), gain * angles.sin().half()
        if config.telechat_rope_precision == "bfloat16":
            cs, sn = cs.bfloat16(), sn.bfloat16()
        def rotate(value):
            a, b = torch.split(value, dim // 2, dim=-1)
            return (value * cs + torch.cat([-b, a], dim=-1) * sn).to(value.dtype)
        q, k = rotate(q), rotate(k)
        if past is not None:
            k, v = torch.cat([past[index][0], k], 2), torch.cat([past[index][1], v], 2)
        result.append((k, v))
        k, v = k.repeat_interleave(heads // kvheads, 1), v.repeat_interleave(heads // kvheads, 1)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * (1 / math.sqrt(dim))
        dtype = scores.dtype
        if dtype == torch.float16:
            scores = scores.float()
        mask = torch.arange(offset + length)[None, :] > torch.arange(offset, offset + length)[:, None]
        mask = mask[None, None].expand(batch, 1, length, offset + length)
        if padding is not None:
            mask = mask | ~padding[:, None, None].bool()
        probs = scores.masked_fill(mask, torch.finfo(scores.dtype).min).softmax(-1).to(dtype)
        all_probs.append(probs)
        context = torch.einsum("bhqk,bhkd->bhqd", probs, v).transpose(1, 2).reshape(batch, length, -1)
        x = residual + linear(context, "self_attention.dense")
        normalized = norm(x, root + "post_attention_layernorm")
        residual = normalized if config.apply_residual_connection_post_layernorm else x
        x = residual + linear(F.silu(linear(normalized, "mlp.gate_proj")) * linear(normalized, "mlp.up_proj"), "mlp.down_proj")
        if index + 1 != config.n_layer:
            all_states.append(x)
    x = norm(x, "transformer.ln_f")
    return F.linear(x, state["lm_head.weight"]), result, all_states + [x], all_probs


@pytest.mark.parametrize("kvheads", [1, 2, 4])
@pytest.mark.parametrize("postnorm", [False, True])
@pytest.mark.parametrize("policy", ["legacy", "modern"])
def test_independent_decoder_equations_and_multitoken_cache(kvheads, postnorm, policy):
    model = _model(num_key_value_heads=kvheads, apply_residual_connection_post_layernorm=postnorm,
                   embed_layernorm=True, rope_theta=1e6 if policy == "modern" else None,
                   telechat_rope_policy=policy).double()
    ids = torch.tensor([[4, 7, 9, 8, 6, 3, 5, 2, 1, 10, 11, 4], [3, 8, 2, 6, 4, 5, 6, 7, 8, 9, 12, 8]])
    cache, reference = None, None
    for start, end in [(0, 3), (3, 7), (7, 8), (8, 12)]:
        expected, reference, states, probs = _source_forward(model.state_dict(), model.config, ids[:, start:end], reference)
        actual = model(ids[:, start:end], past_key_values=cache, output_hidden_states=True, output_attentions=True)
        cache = actual.past_key_values
        torch.testing.assert_close(actual.logits, expected, atol=2e-8, rtol=2e-7)
        for a, b in zip(actual.hidden_states, states, strict=True):
            torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-7)
        for index, layer in enumerate(cache.layers):
            torch.testing.assert_close(layer.keys, reference[index][0])
            torch.testing.assert_close(layer.values, reference[index][1])
            torch.testing.assert_close(actual.attentions[index], probs[index])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("policy", ["legacy", "modern"])
def test_dtype_cache_generation_and_two_safetensors_cycles(tmp_path, dtype, implementation, policy):
    model = _model(attn_implementation=implementation, num_key_value_heads=2,
                   telechat_rope_policy=policy, rope_theta=1e6, torch_dtype=dtype).to(dtype)
    ids = torch.tensor([[1, 5, 7, 6, 8]])
    with torch.inference_mode():
        expected, _, _, _ = _source_forward(model.state_dict(), model.config, ids)
        actual = model(ids).logits
        torch.testing.assert_close(actual, expected, atol=0.006 if dtype == torch.bfloat16 else 0.0003,
                                   rtol=0.04 if dtype == torch.bfloat16 else 0.003)
        prefill = model(ids[:, :2])
        continued = model(ids[:, 2:], past_key_values=prefill.past_key_values)
        torch.testing.assert_close(continued.logits, actual[:, 2:], atol=0.003, rtol=0.03)
        for beams in [1, 3]:
            generated = model.generate(ids, max_new_tokens=4, do_sample=False, eos_token_id=None, num_beams=beams)
            assert generated.shape == (1, 9)
            if beams == 1:
                source_ids, source_past, current = ids.clone(), None, ids
                for _ in range(4):
                    source_logits, source_past, _, _ = _source_forward(model.state_dict(), model.config, current, source_past)
                    current = source_logits[:, -1].argmax(-1, keepdim=True)
                    source_ids = torch.cat([source_ids, current], dim=-1)
                torch.testing.assert_close(generated, source_ids)
        before = model.state_dict()
        for cycle in range(2):
            dest = tmp_path / str(cycle)
            model.save_pretrained(dest)
            assert list(dest.glob("*.safetensors")) and not list(dest.glob("*.bin"))
            model = Telechat2ForCausalLM.from_pretrained(dest, dtype=dtype, attn_implementation=implementation).eval()
            assert model.config.telechat_rope_policy == policy
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name], atol=0, rtol=0)
            torch.testing.assert_close(model(ids).logits, actual, atol=0, rtol=0)


def test_generation_policy_outputs_tuple_loss_and_retry(tmp_path):
    model = _model()
    ids = torch.tensor([[1, 5, 7]])
    first = model(ids).logits[:, -1].argmax(-1).item()
    model.generation_config.eos_token_id = [first, 2]
    model.generation_config.do_sample = False
    model.save_pretrained(tmp_path)
    loaded = Telechat2ForCausalLM.from_pretrained(tmp_path).eval()
    assert loaded.generate(ids, max_new_tokens=5).shape[-1] == 4
    assert loaded.generate(ids, max_new_tokens=3, eos_token_id=None).shape[-1] == 6
    assert len(model(ids, use_cache=False, return_dict=False)) == 1
    result = model(ids, labels=ids)
    expected = F.cross_entropy(result.logits[:, :-1].float().flatten(0, 1), ids[:, 1:].flatten())
    torch.testing.assert_close(result.loss, expected)
    with pytest.raises(ValueError, match="full cached"):
        model(ids[:, :1], past_key_values=model(ids).past_key_values, attention_mask=torch.ones(1, 1))
    assert torch.isfinite(model(ids).logits).all()
    weightfile = tmp_path / "model.safetensors"
    original = weightfile.read_bytes()
    weightfile.write_bytes(b"invalid")
    with pytest.raises(Exception):
        Telechat2ForCausalLM.from_pretrained(tmp_path)
    weightfile.write_bytes(original)
    torch.testing.assert_close(Telechat2ForCausalLM.from_pretrained(tmp_path)(ids).logits, model(ids).logits)


def test_headless_checkpoint_fails_without_inventing_a_language_head(tmp_path):
    model = _model()
    model.transformer.save_pretrained(tmp_path)
    with pytest.raises(RuntimeError, match="missing.*lm_head"):
        Telechat2ForCausalLM.from_pretrained(tmp_path)


@pytest.mark.parametrize("kvheads,postnorm", [(4, False), (2, True), (1, False)])
@pytest.mark.parametrize("offload", [False, True])
def test_public_plain_awq_all_linears_and_two_exact_reloads(tmp_path, kvheads, postnorm, offload):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig, CalibrationConfig
    model = _model(vocab_size=131080, num_key_value_heads=kvheads,
                   apply_residual_connection_post_layernorm=postnorm)
    model.save_pretrained(tmp_path)
    _assets(tmp_path)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=offload)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16,
                            attn_implementation="eager")
    assert wrapper.awq_get_modules_for_scaling(wrapper.model.transformer.h[0], {}, {}) == []
    calibration = [{"input_ids": torch.tensor([[4, 25718, 32471, 5, 9, 10, 11, 12]]),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    dense = {key: value.to(torch.float16) for key, value in model.state_dict().items()}
    wrapper.quantize(CalibrationConfig(dataset=calibration, batch_size=1, min_length=1), backend=BACKEND.TORCH)
    quantized = {name for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)}
    expected = {f"transformer.h.{index}.{suffix}" for index in range(2) for suffix in (
        "self_attention.query", "self_attention.key_value", "self_attention.dense", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
    assert quantized == expected
    state = {name: value.detach().clone() for name, value in wrapper.model.state_dict().items()}
    for name, value in state.items():
        if name in dense:
            torch.testing.assert_close(value, dense[name], atol=0, rtol=0)
    with torch.inference_mode():
        output = wrapper.model(**calibration[0], use_cache=False).logits
    assert torch.isfinite(output).all()
    for cycle in range(2):
        saved = tmp_path / f"quantized{cycle}"
        wrapper.save_quantized(str(saved))
        assert list(saved.glob("*.safetensors")) and not list(saved.glob("*.bin"))
        wrapper = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16,
                                 attn_implementation="eager")
        for name, value in wrapper.model.state_dict().items():
            torch.testing.assert_close(value, state[name], atol=0, rtol=0)
        assert {name for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)} == expected
        assert wrapper.tokenizer.encode("Hello 世界!") == [25718, 32471, 126274]
        with torch.inference_mode():
            torch.testing.assert_close(wrapper.model(**calibration[0], use_cache=False).logits, output, atol=0, rtol=0)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_real_tokenizer_model_hostile_selection_and_two_reloads(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel import GPTQModel
    if bypass is not None:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    else:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    model = _model(vocab_size=131080)
    model.save_pretrained(tmp_path)
    _assets(tmp_path)
    marker = tmp_path / "EXECUTED"
    (tmp_path / "remote.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('remote')")
    for filename in ["config.json", "tokenizer_config.json"]:
        path = tmp_path / filename
        payload = json.loads(path.read_text())
        payload["auto_map"] = {"AutoConfig": "remote.Config", "AutoModelForCausalLM": "remote.Model",
                               "AutoTokenizer": ["remote.Tokenizer", None]}
        payload["tokenizer_class"] = "ForeignTokenizer"
        path.write_text(json.dumps(payload))
    for cycle in range(3):
        loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust,
                               attn_implementation="eager")
        assert type(loaded.model) is Telechat2ForCausalLM
        assert isinstance(loaded.tokenizer, Telechat2Tokenizer)
        assert not loaded.trust_remote_code
        assert loaded.tokenizer.encode("Hello 世界!") == [25718, 32471, 126274]
        ids = torch.tensor([[4, 5, 7]])
        torch.testing.assert_close(loaded.model(ids).logits, model(ids).logits, rtol=0, atol=0)
        assert loaded.model.generate(ids, do_sample=False, max_new_tokens=2).shape[-1] == 5
        if cycle < 2:
            tmp_path = tmp_path / "saved"
            loaded.model.save_pretrained(tmp_path)
            loaded.tokenizer.save_pretrained(tmp_path)
    assert not marker.exists()


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_selected_local_config_ignores_checkpoint_python(tmp_path, monkeypatch, trust, bypass):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    marker = tmp_path / "EXECUTED"
    (tmp_path / "remote.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python')\n"
    )
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "telechat", "hidden_size": 32, "n_head": 4, "n_layer": 2,
        "ffn_hidden_size": 64, "vocab_size": 32, "training_seqlen": 16, "base_seqlen": 16,
        "auto_map": {"AutoConfig": "remote.RemoteConfig", "AutoModelForCausalLM": "remote.RemoteModel"},
    }))
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config).__module__.startswith("gptqmodel.models.ports.telechat2.")
    assert config.hidden_size == 32
    assert not marker.exists()
