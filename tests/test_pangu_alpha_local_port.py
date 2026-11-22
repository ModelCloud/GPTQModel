# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real PanGu top-query decoding; no checkpoint Python is needed."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache

from gptqmodel.models.definitions.pangu_alpha import PanguAlphaQModel
from gptqmodel.models.ports.pangu_alpha import GPTPanguConfig, GPTPanguForCausalLM
from gptqmodel.models.ports.pangu_alpha.tokenization_pangu import GPTPanguTokenizer
from tests.port_assets import read_port_asset


FIXTURES = Path(__file__).parents[1] / "gptqmodel/models/ports/pangu_alpha/fixtures"


def real_assets(path):
    pytest.importorskip("jieba", reason="PanGu-α tokenizer requires the optional jieba package")
    path.mkdir(exist_ok=True, parents=True)
    data = read_port_asset(FIXTURES / "vocab.model")
    assert hashlib.sha256(data).hexdigest() == "18857e86783e50cfcaa0bc3c043fb4e9b5f240b885d2870ea593ee69b44f7a3a"
    (path / "vocab.model").write_bytes(data)
    return path


def public_fixture(path, layers=2):
    real_assets(path)
    torch.manual_seed(137)
    model = GPTPanguForCausalLM(
        tiny_config(vocab_size=40000, num_layers=layers, eos_token_id=9, pad_token_id=9)
    ).eval()
    model.save_pretrained(path, safe_serialization=True)
    tokenizer = GPTPanguTokenizer(path / "vocab.model", pad_token="<eot>")
    tokenizer.save_pretrained(path)
    sentinel = "raise RuntimeError('CHECKPOINT PYTHON EXECUTED')\n"
    (path / "hostile.py").write_text(sentinel)
    for name in ("config.json", "tokenizer_config.json"):
        data = json.loads((path / name).read_text())
        data["auto_map"] = {
            "AutoConfig": "hostile.Config",
            "AutoModelForCausalLM": "hostile.Model",
            "AutoTokenizer": ["hostile.Tokenizer", None],
        }
        (path / name).write_text(json.dumps(data))
    return model


def tiny_config(**kwargs):
    values = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_layers": 3,
        "num_heads": 4,
        "max_position_embeddings": 32,
        "resid_pdrop": 0,
        "embd_pdrop": 0,
        "attn_pdrop": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    values.update(kwargs)
    return GPTPanguConfig(**values)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_definition_owns_config_and_model():
    config = tiny_config()
    model = PanguAlphaQModel.loader.from_config(config)
    assert type(model) is GPTPanguForCausalLM
    assert not PanguAlphaQModel.require_trust_remote_code
    assert model.transformer.wqe.weight.shape == (32, 32)


def source_equations(state, config, ids, positions, mask=None, types=None, head_mask=None):
    """Independent float64 equations, not a call to a port modeling helper."""
    state = {k: v.double() for k, v in state.items()}

    def linear(x, prefix):
        return F.linear(x, state[prefix + ".weight"], state.get(prefix + ".bias"))

    def norm(x, prefix):
        return F.layer_norm(
            x, (config.hidden_size,), state[prefix + ".weight"], state[prefix + ".bias"], config.layer_norm_epsilon
        )

    def heads(x):
        return x.reshape(x.shape[0], x.shape[1], config.num_heads, -1).transpose(1, 2)

    hidden = F.embedding(ids, state["transformer.wte.weight"])
    hidden += F.embedding(positions, state["transformer.wpe.weight"])
    if types is not None:
        hidden += F.embedding(types, state["transformer.wte.weight"])
    query = F.embedding(positions, state["transformer.wqe.weight"])
    caches, states, probabilities = [], [], []
    for i in range(config.num_layers):
        p = f"transformer.h.{i}"
        if i == config.num_layers - 1:
            hidden = norm(hidden, "transformer.ln_f")
        states.append(hidden)
        normalized = norm(hidden, p + ".ln_1")
        q = heads(linear(query if i == config.num_layers - 1 else normalized, p + ".attn.q_proj"))
        k, v = (heads(linear(normalized, p + ".attn." + name)) for name in ("k_proj", "v_proj"))
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
        if config.scale_attn_weights:
            scores /= (config.hidden_size // config.num_heads) ** 0.5
        future = torch.arange(ids.shape[1])[None, :] > torch.arange(ids.shape[1])[:, None]
        scores = scores.masked_fill(future, -10000.0)
        if mask is not None:
            scores += (1 - mask[:, None, None].double()) * -10000
        weights = scores.softmax(-1)
        if head_mask is not None:
            weights = weights * head_mask[i][None, :, None, None]
        context = torch.einsum("bhqk,bhkd->bhqd", weights, v).transpose(1, 2).reshape_as(hidden)
        hidden = hidden + linear(context, p + ".attn.c_proj")
        mlp = linear(norm(hidden, p + ".ln_2"), p + ".mlp.c_fc")
        mlp = F.gelu(mlp) if config.activation_function == "gelu" else F.relu(mlp)
        hidden = hidden + linear(mlp, p + ".mlp.c_proj")
        caches.append((k, v))
        probabilities.append(weights)
    return linear(hidden, "lm_head"), caches, states + [hidden], probabilities


@pytest.mark.parametrize(
    "layers,heads,inner,scaled,activation,tied",
    [
        (1, 1, None, True, "gelu", True),
        (2, 2, 48, False, "gelu", False),
        (3, 4, 64, True, "relu", True),
        (4, 8, 40, False, "gelu", False),
    ],
)
def test_float64_source_decoder_oracle(layers, heads, inner, scaled, activation, tied):
    torch.manual_seed(127)
    config = tiny_config(
        num_layers=layers,
        num_heads=heads,
        intermediate_size=inner,
        scale_attn_weights=scaled,
        activation_function=activation,
        tie_word_embeddings=tied,
        initializer_range=0.2,
    )
    model = GPTPanguForCausalLM(config).double().eval()
    # Distinct affine norms and nonzero biases expose an omitted/moved ln_f.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "ln_" in name:
                parameter.copy_(torch.linspace(0.4, 1.4, parameter.numel()).reshape_as(parameter))
    ids = torch.tensor([[4, 8, 6, 5, 7], [9, 6, 4, 3, 8]])
    positions = torch.tensor([[1, 2, 4, 8, 15], [0, 1, 3, 9, 16]])
    mask = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    types = torch.tensor([[1, 1, 2, 2, 1], [2, 1, 2, 1, 2]])
    head_mask = torch.linspace(0.3, 1, layers * heads).reshape(layers, heads)
    expected, cache, states, probs = source_equations(
        model.state_dict(), config, ids, positions, mask, types, head_mask
    )
    actual = model(
        ids,
        position_ids=positions,
        attention_mask=mask,
        token_type_ids=types,
        head_mask=head_mask,
        output_attentions=True,
        output_hidden_states=True,
    )
    torch.testing.assert_close(actual.logits, expected, atol=2e-12, rtol=2e-12)
    for got, want in zip(actual.hidden_states, states, strict=True):
        torch.testing.assert_close(got, want, atol=2e-12, rtol=2e-12)
    for i, ((key, value), prob) in enumerate(zip(cache, probs, strict=True)):
        torch.testing.assert_close(actual.past_key_values.layers[i].keys, key, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(actual.past_key_values.layers[i].values, value, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(actual.attentions[i], prob, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("dtype,tied", [(torch.float32, True), (torch.float16, False), (torch.bfloat16, True)])
def test_cache_generation_and_two_safetensors_cycles(tmp_path, dtype, tied):
    torch.manual_seed(9)
    model = GPTPanguForCausalLM(tiny_config(tie_word_embeddings=tied)).to(dtype).eval()
    ids = torch.tensor([[4, 6, 9, 5, 7, 8], [4, 5, 7, 5, 9, 8]])
    mask = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
    atol = 1e-6 if dtype == torch.float32 else 3e-3
    with torch.no_grad():
        full = model(ids, attention_mask=mask, position_ids=positions).logits
        cache = DynamicCache()
        start = 0
        for size in (1, 4, 1):
            end = start + size
            result = model(
                ids[:, start:end],
                attention_mask=mask[:, :end],
                position_ids=positions[:, start:end],
                past_key_values=cache,
            )
            valid = mask[:, end - 1].bool()
            torch.testing.assert_close(result.logits[valid, -1], full[valid, end - 1], atol=atol, rtol=atol)
            assert cache.get_seq_length() == end
            start = end
        for beams in (1, 3):
            cached = model.generate(
                ids, attention_mask=mask, max_new_tokens=3, num_beams=beams, do_sample=False, use_cache=True
            )
            uncached = model.generate(
                ids, attention_mask=mask, max_new_tokens=3, num_beams=beams, do_sample=False, use_cache=False
            )
            torch.testing.assert_close(cached, uncached, rtol=0, atol=0)
        for cycle in range(2):
            target = tmp_path / str(cycle)
            model.save_pretrained(target, safe_serialization=True)
            assert list(target.glob("*.safetensors")) and not list(target.glob("*.bin"))
            restored = GPTPanguForCausalLM.from_pretrained(target, dtype=dtype).eval()
            assert (restored.lm_head.weight is restored.transformer.wte.weight) is tied
            assert restored.state_dict().keys() == model.state_dict().keys()
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], tensor, atol=0, rtol=0)
            torch.testing.assert_close(
                restored(ids, attention_mask=mask, position_ids=positions).logits, full, atol=0, rtol=0
            )
            model = restored


def test_failure_before_cache_mutation_then_same_cache_retry():
    model = GPTPanguForCausalLM(tiny_config(max_position_embeddings=8)).eval()
    ids = torch.tensor([[4, 5, 6, 7]])
    with torch.no_grad():
        cache = model(ids).past_key_values
        frozen = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
        with pytest.raises(ValueError, match="context"):
            model(torch.ones(1, 5, dtype=torch.long), past_key_values=cache)
        assert cache.get_seq_length() == 4
        for layer, (key, value) in zip(cache.layers, frozen, strict=True):
            torch.testing.assert_close(layer.keys, key, rtol=0, atol=0)
            torch.testing.assert_close(layer.values, value, rtol=0, atol=0)
        continuation = model(ids[:, :1], past_key_values=cache).logits
        full = model(torch.cat((ids, ids[:, :1]), -1)).logits[:, -1:]
        torch.testing.assert_close(continuation, full)


def test_invalid_mask_retry_and_read_only_cache():
    model = GPTPanguForCausalLM(tiny_config()).eval()
    ids = torch.tensor([[4, 5, 6]])
    with torch.no_grad():
        cache = model(ids).past_key_values
        with pytest.raises(ValueError, match="attention_mask"):
            model(ids[:, :1], past_key_values=cache, attention_mask=torch.ones(1, 3))
        assert cache.get_seq_length() == 3
        got = model(ids[:, :1], past_key_values=cache, use_cache=False).logits
        assert cache.get_seq_length() == 3
        expected = model(torch.cat((ids, ids[:, :1]), -1)).logits[:, -1:]
        torch.testing.assert_close(got, expected)


def test_top_query_and_final_norm_are_observable():
    torch.manual_seed(73)
    model = GPTPanguForCausalLM(tiny_config(initializer_range=0.2)).eval()
    ids = torch.tensor([[5, 3, 7, 6, 4]])
    baseline = model(ids, use_cache=False).logits
    with torch.no_grad():
        model.transformer.wqe.weight.mul_(0)
    assert (model(ids, use_cache=False).logits - baseline).abs().max() > 1e-3


@pytest.mark.parametrize("layers", [1, 3])
@pytest.mark.parametrize("shared_positions", [False, True])
def test_live_replay_reuses_real_block_positions_and_normalizes_once(layers, shared_positions):
    torch.manual_seed(221)
    model = GPTPanguForCausalLM(tiny_config(num_layers=layers, initializer_range=0.2)).eval()
    wrapper = PanguAlphaQModel.__new__(PanguAlphaQModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model
    captured = []
    handles = [
        block.register_forward_hook(
            lambda _, args, kwargs, output: captured.append((args, kwargs, output)), with_kwargs=True
        )
        for block in model.transformer.h
    ]
    positions = torch.tensor([[9, 3, 5, 1], [2, 11, 0, 4]])
    if shared_positions:
        positions = positions[:1]
    ids = torch.tensor([[4, 5, 6, 7], [4, 9, 8, 1]])
    mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    with torch.no_grad():
        model(ids, attention_mask=mask, position_ids=positions, use_cache=False)
        for handle in handles:
            handle.remove()
        first_input, first_kwargs, _ = captured[0]
        last_input, last_kwargs, (last_output,) = captured[-1]
        assert first_kwargs["position_ids"] is positions
        assert first_kwargs["attention_mask"] is mask
        replay_input = [first_input[0] if layers == 1 else captured[-2][2][0]]
        last_block = model.transformer.h[-1]
        replay_kwargs = wrapper.prepare_layer_replay_kwargs(
            last_block, replay_input, dict(first_kwargs), torch.device("cpu")
        )
        torch.testing.assert_close(replay_input[0], last_input[0], rtol=0, atol=0)
        torch.testing.assert_close(replay_kwargs["custom_query"], last_kwargs["custom_query"], rtol=0, atol=0)
        torch.testing.assert_close(last_block(*replay_input, **replay_kwargs)[0], last_output, rtol=0, atol=0)


def test_explicit_unsupported_sdpa_and_legacy_cache():
    with pytest.raises(ValueError, match="scaled_dot_product_attention"):
        GPTPanguForCausalLM._from_config(tiny_config(), attn_implementation="sdpa")
    model = GPTPanguForCausalLM(tiny_config())
    with pytest.raises(TypeError, match="DynamicCache"):
        model(torch.tensor([[4, 5]]), past_key_values=((torch.zeros(1, 4, 1, 8),) * 2,))


def test_real_tokenizer_source_ids_decode_and_two_reload_cycles(tmp_path):
    real_assets(tmp_path)
    tokenizer = GPTPanguTokenizer(tmp_path / "vocab.model", pad_token="<eot>")
    cases = json.loads(read_port_asset(FIXTURES / "tokenizer_oracle.json"))
    for cycle in range(3):
        assert tokenizer.eos_token_id == 9
        for case in cases:
            assert tokenizer.encode(case["text"], add_special_tokens=False) == case["ids"]
            assert tokenizer.decode(case["ids"]) == case["decoded"]
        target = tmp_path / str(cycle)
        tokenizer.save_pretrained(target)
        assert (target / "vocab.model").read_bytes() == read_port_asset(FIXTURES / "vocab.model")
        tokenizer = GPTPanguTokenizer.from_pretrained(target)


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_public_vocabulary_failure_and_repair(tmp_path, damage):
    from gptqmodel import BACKEND, GPTQModel

    source = tmp_path / "source"
    public_fixture(source)
    vocab = source / "vocab.model"
    data = vocab.read_bytes()
    if damage == "missing":
        vocab.unlink()
    else:
        vocab.write_bytes(b"invalid sentencepiece file")
    with pytest.raises(RuntimeError, match=r"vocab\.model"):
        GPTQModel.load(str(source), backend=BACKEND.TORCH, device="cpu", dtype=torch.float32)
    vocab.write_bytes(data)
    wrapper = GPTQModel.load(str(source), backend=BACKEND.TORCH, device="cpu", dtype=torch.float32)
    assert wrapper.tokenizer.encode("你好，世界！", add_special_tokens=False) == [5772, 10, 267, 29]


@pytest.mark.parametrize("trust,bypass", [(False, None), (True, "1")])
def test_public_real_tokenizer_model_security_and_reload(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel import BACKEND, GPTQModel

    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    source = tmp_path / "source"
    expected = public_fixture(source)
    ids = torch.tensor([[5772, 10, 267, 29]])
    with torch.no_grad():
        oracle = expected(ids).logits
    wrapper = GPTQModel.load(
        str(source), backend=BACKEND.TORCH, device="cpu", dtype=torch.float32, trust_remote_code=trust
    )
    assert type(wrapper.model) is GPTPanguForCausalLM
    assert wrapper.tokenizer.encode("你好，世界！", add_special_tokens=False) == ids[0].tolist()
    with torch.no_grad():
        torch.testing.assert_close(wrapper.model(ids).logits, oracle, rtol=0, atol=0)
    for cycle in range(2):
        target = tmp_path / str(cycle)
        wrapper.model.save_pretrained(str(target), safe_serialization=True)
        wrapper.tokenizer.save_pretrained(str(target))
        assert list(target.glob("*.safetensors")) and not list(target.glob("*.bin"))
        wrapper = GPTQModel.load(
            str(target), backend=BACKEND.TORCH, device="cpu", dtype=torch.float32, trust_remote_code=trust
        )
        assert wrapper.tokenizer.encode("你好，世界！", add_special_tokens=False) == ids[0].tolist()
        with torch.no_grad():
            torch.testing.assert_close(wrapper.model(ids).logits, oracle, rtol=0, atol=0)


@pytest.mark.parametrize("offload,dtype,layers", [(False, torch.float16, 2), (True, torch.bfloat16, 1)])
def test_public_plain_awq_pack_infer_and_two_reloads(tmp_path, offload, dtype, layers):
    from safetensors.torch import load_file

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear
    from gptqmodel.quantization.config import FORMAT, AWQConfig, CalibrationConfig

    source = tmp_path / "source"
    dense = public_fixture(source, layers=layers)
    config = AWQConfig(
        bits=4,
        group_size=16,
        format=FORMAT.GEMM,
        device="cpu",
        fallback=None,
        offload_to_disk=offload,
        offload_to_disk_path=str(tmp_path / "offload"),
    )
    wrapper = GPTQModel.load(str(source), quantize_config=config, backend=BACKEND.TORCH, dtype=dtype)
    assert (wrapper.turtle_model is not None) is offload
    assert any(parameter.is_meta for parameter in wrapper.model.parameters()) is offload
    wrapper.quantize(
        CalibrationConfig(
            dataset=["你好世界，这是模型量化测试。", "Hello 世界 GPT-QModel."], batch_size=1, min_length=1
        ),
        backend=BACKEND.TORCH,
    )
    linears = {name for name, mod in wrapper.model.named_modules() if isinstance(mod, BaseQuantLinear)}
    expected = {
        f"transformer.h.{i}.{name}"
        for i in range(layers)
        for name in ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
    }
    assert linears == expected
    for name, value in dense.state_dict().items():
        if not any(name.startswith(prefix + ".") for prefix in expected):
            loaded = wrapper.model.state_dict()[name]
            torch.testing.assert_close(loaded, value.to(loaded.dtype), rtol=0, atol=0)
    ids = torch.tensor([[5772, 10, 267, 29]])
    with torch.no_grad():
        logits = wrapper.model(ids).logits
    assert torch.isfinite(logits).all()
    previous_saved = None
    for cycle in range(2):
        snapshot = {name: value.detach().clone() for name, value in wrapper.model.state_dict().items()}
        target = tmp_path / str(cycle)
        wrapper.save(str(target))
        saved = {key: value for payload in target.glob("*.safetensors") for key, value in load_file(payload).items()}
        assert saved and not list(target.glob("*.bin"))
        for module in expected:
            assert {module + "." + name for name in ("qweight", "qzeros", "scales")} <= saved.keys()
        if previous_saved is not None:
            assert previous_saved.keys() == saved.keys()
            for name, value in saved.items():
                torch.testing.assert_close(value, previous_saved[name], rtol=0, atol=0)
        previous_saved = saved
        wrapper = GPTQModel.load(str(target), backend=BACKEND.TORCH, device="cpu", dtype=dtype)
        assert type(wrapper.model) is GPTPanguForCausalLM
        assert wrapper.tokenizer.encode("你好，世界！", add_special_tokens=False) == ids[0].tolist()
        assert wrapper.model.state_dict().keys() == snapshot.keys()
        for name, value in snapshot.items():
            torch.testing.assert_close(wrapper.model.state_dict()[name], value, rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(wrapper.model(ids).logits, logits, rtol=0, atol=0)
            assert wrapper.generate(ids, max_new_tokens=2, min_new_tokens=2).shape == (1, 6)
