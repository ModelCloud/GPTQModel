# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Offline Ripple qualification: independent equations and real public consumers."""

import copy
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache, StaticCache

from gptqmodel.models.definitions.ricursive import RicursiveQModel
from gptqmodel.models.ports.ricursive.configuration_ricursive import RicursiveConfig
from gptqmodel.models.ports.ricursive.modeling_ricursive import RicursiveForCausalLM
from tests.port_assets import iter_port_assets, read_port_asset


FIXTURES = Path(__file__).resolve().parents[1] / "gptqmodel/models/ports/ricursive/fixtures"
LINEARS = {"self_attn." + n for n in ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")} | {
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
}


@pytest.fixture(autouse=True, scope="module")
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config(**changes):
    values = {"vocab_size": 64, "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "sliding_window": 4,
              "layer_types": ["sliding_attention", "full_attention"], "max_position_embeddings": 128,
              "use_valres": True, "use_gated_attention": True, "ffn_clamp_threshold": 0.08,
              "embed_scale": 3.5, "mlp_chunk_size": 3, "bos_token_id": 1, "pad_token_id": 0, "eos_token_id": None}
    values.update(changes)
    return RicursiveConfig(**values)


def _model(**changes):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(113)
        model = RicursiveForCausalLM(_config(**changes)).eval()
        # Nonuniform affine gains and nonzero blend scalars make mistakes observable.
        with torch.no_grad():
            for name, weight in model.named_parameters():
                if "norm" in name:
                    weight.copy_(torch.linspace(0.6, 1.4, weight.numel()).reshape_as(weight))
                elif name.endswith("value_res_lambda"):
                    weight.fill_(0.73)
    return model


@torch.inference_mode()
def _oracle(model, ids, mask=None, positions=None):
    """Pinned upstream equations using tensors only, with explicit softmax and masks.

    Supports FP64 linear/attention arithmetic; RMS and RoPE deliberately retain
    the source's FP32 casts, including rounded low-precision trig tables.
    Low-precision interpolation and SDPA accumulate in FP32 before casting.
    Returns independently computed full keys/values as well as logits.
    """
    c, weights = model.config, model.state_dict()
    batch, length = ids.shape
    pos = torch.arange(length)
    positions = pos[None] if positions is None else positions

    def linear(x, name):
        return F.linear(x, weights[name + ".weight"])

    def norm(x, name):
        h = x.float()
        return (h * (h.square().mean(-1, keepdim=True) + c.rms_norm_eps).rsqrt()).to(x.dtype) * weights[name + ".weight"]

    def rotate(x):
        inv = (c.rope_theta ** (torch.arange(0, c.head_dim, 2).float() / c.head_dim)).reciprocal()
        phase = positions.float()[..., None] * inv
        cosine, sine = phase.cos().to(x.dtype).float()[:, None], phase.sin().to(x.dtype).float()[:, None]
        first, second = x.float().chunk(2, dim=-1)
        return torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1).to(x.dtype)

    hidden = F.embedding(ids, weights["model.embed_tokens.weight"])
    if c.embed_scale is not None:
        hidden = hidden * c.embed_scale
    seed = hidden
    cache = []
    for index, layer_type in enumerate(c.layer_types):
        p = f"model.layers.{index}."
        h = norm(hidden, p + "input_layernorm")
        q = norm(linear(h, p + "self_attn.q_proj").reshape(batch, length, c.num_attention_heads, c.head_dim),
                 p + "self_attn.q_norm").transpose(1, 2)
        k = norm(linear(h, p + "self_attn.k_proj").reshape(batch, length, c.num_key_value_heads, c.head_dim),
                 p + "self_attn.k_norm").transpose(1, 2)
        blend_input = h
        if c.use_valres:
            blend = weights[p + "self_attn.value_res_lambda"][0].sigmoid().to(h.dtype)
            arithmetic_dtype = torch.float64 if h.dtype == torch.float64 else torch.float32
            blend_input = (seed.to(arithmetic_dtype)
                           + (h.to(arithmetic_dtype) - seed.to(arithmetic_dtype))
                           * blend.to(arithmetic_dtype)).to(h.dtype)
        v = linear(blend_input, p + "self_attn.v_proj").reshape(batch, length, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        if layer_type == "sliding_attention":
            q, k = rotate(q), rotate(k)
        cache.append((k, v))
        k = k.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, dim=1)
        v = v.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, dim=1)
        visible = pos[:, None] >= pos[None, :]
        if layer_type == "sliding_attention" and c.sliding_window is not None:
            visible &= pos[:, None] - pos[None, :] < c.sliding_window
        visible = visible[None, None]
        attention_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        scores = q.to(attention_dtype) @ k.to(attention_dtype).transpose(-1, -2) * c.head_dim ** -0.5
        if mask is not None:
            if mask.dim() == 2:
                visible = visible & mask[:, None, None, :].bool()
            elif mask.dtype == torch.bool:
                visible = visible & mask
            else:
                scores = scores + mask
        probability = scores.masked_fill(~visible, float("-inf")).softmax(-1).nan_to_num()
        attended = (probability @ v.to(attention_dtype)).to(v.dtype).transpose(1, 2).reshape(batch, length, -1)
        if c.use_gated_attention:
            gate = linear(h, p + "self_attn.g_proj")
            if c.gate_full_precision:
                gate = gate.float()
            attended = attended * gate.sigmoid().to(attended.dtype)
        hidden = hidden + norm(linear(attended, p + "self_attn.o_proj"), p + "post_attention_layernorm")
        h = norm(hidden, p + "pre_feedforward_layernorm")
        gate, up = linear(h, p + "mlp.gate_proj"), linear(h, p + "mlp.up_proj")
        if c.ffn_clamp_threshold:
            gate = gate.clamp(max=c.ffn_clamp_threshold)
            up = up.clamp(-c.ffn_clamp_threshold, c.ffn_clamp_threshold)
        hidden = hidden + norm(linear(F.silu(gate) * up, p + "mlp.down_proj"), p + "post_feedforward_layernorm")
    return linear(norm(hidden, "model.norm"), "lm_head"), cache


@pytest.mark.parametrize("kv_heads", [1, 4])
@pytest.mark.parametrize("features", [True, False])
@pytest.mark.parametrize("layout", ["hybrid", "full", "sliding"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_independent_equations(kv_heads, features, layout, dtype):
    types = ["sliding_attention", "full_attention"] if layout == "hybrid" else [layout + "_attention"] * 2
    model = _model(num_key_value_heads=kv_heads, use_valres=features, use_gated_attention=features,
                   gate_full_precision=features, embed_scale=3.5 if features else None,
                   ffn_clamp_threshold=0.08 if features else None, layer_types=types).to(dtype)
    ids = torch.tensor([[0, 0, 3, 9, 5, 11, 17], [2, 6, 8, 19, 5, 4, 3]])
    mask = ids.ne(0)
    positions = mask.long().cumsum(-1).clamp_min(1) - 1 + 65530
    expected, _ = _oracle(model, ids, mask, positions)
    actual = model(ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
    tolerance = 0.008 if dtype == torch.bfloat16 else 0.0012 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual[mask], expected[mask], atol=tolerance, rtol=tolerance)
    perturbed = ids.clone()
    perturbed[:, -1] = 33
    valid_prefix = mask[:, :-1]
    torch.testing.assert_close(model(perturbed, attention_mask=mask, position_ids=positions, use_cache=False).logits[:, :-1][valid_prefix],
                               actual[:, :-1][valid_prefix], atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("cache_kind", ["default", "full", "hybrid", "static"])
@pytest.mark.parametrize("window", [1, 4, 32, None])
@torch.inference_mode()
def test_cache_multitoken_eviction_padding_and_independent_keys(cache_kind, window):
    model = _model(sliding_window=window)
    ids = torch.arange(3, 16)[None].repeat(2, 1)
    mask = torch.ones_like(ids)
    mask[0, :2] = 0
    positions = mask.cumsum(-1).sub(1).clamp_min(0)
    expected, expected_cache = _oracle(model, ids, mask, positions)
    cache = {"default": lambda: None, "full": DynamicCache,
             "hybrid": lambda: DynamicCache(config=model.config),
             "static": lambda: StaticCache(config=model.config, max_cache_len=24)}[cache_kind]()
    for start, end in ((0, 1), (1, 5), (5, 6), (6, 12), (12, 13)):
        actual = model(ids[:, start:end], attention_mask=mask[:, :end], position_ids=positions[:, start:end],
                       past_key_values=cache, cache_position=torch.arange(start, end), use_cache=True)
        cache = actual.past_key_values
        assert int(cache.get_seq_length()) == end
        valid = mask[:, start:end].bool()
        torch.testing.assert_close(actual.logits[valid], expected[:, start:end][valid], atol=3e-6, rtol=3e-5)
    # Full cache exposes every key. Native hybrid/sliding caches expose their
    # retained trailing window; static full layers also hold an unwritten tail.
    for index, (keys, values) in enumerate(expected_cache):
        live = cache.layers[index]
        for actual, expected_tensor in ((live.keys, keys), (live.values, values)):
            valid_keys = mask.bool()
            if cache_kind == "full":
                expected_slice = expected_tensor
            elif index == 0 and window is not None:
                retained = min(window - 1, ids.shape[1]) if cache_kind != "static" else min(window, ids.shape[1])
                expected_slice = expected_tensor[:, :, -retained:] if retained else expected_tensor[:, :, :0]
                actual = actual[:, :, :retained]
                valid_keys = valid_keys[:, -retained:] if retained else valid_keys[:, :0]
            else:
                expected_slice = expected_tensor
                actual = actual[:, :, :ids.shape[1]]
            torch.testing.assert_close(actual.transpose(1, 2)[valid_keys], expected_slice.transpose(1, 2)[valid_keys],
                                       atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("beams", [1, 3])
@pytest.mark.parametrize("cache_kind", ["dynamic", "static"])
@torch.inference_mode()
def test_generation_diagnostics_embeddings_and_two_safetensors_cycles(tmp_path, beams, cache_kind):
    model = _model()
    ids = torch.tensor([[0, 0, 3, 4, 5], [3, 4, 8, 7, 9]])
    mask = ids.ne(0).long()
    options = {"max_new_tokens": 4, "do_sample": False, "num_beams": beams, "eos_token_id": None}
    expected = model.generate(ids, attention_mask=mask, use_cache=False, **options)
    actual = model.generate(ids, attention_mask=mask, cache_implementation=cache_kind,
                            return_dict_in_generate=True, output_scores=True,
                            output_hidden_states=True, output_attentions=True, **options)
    assert torch.equal(actual.sequences, expected)
    assert len(actual.scores) == len(actual.hidden_states) == len(actual.attentions) == 4
    embedded = model.generate(inputs_embeds=model.get_input_embeddings()(ids), attention_mask=mask, **options)
    assert torch.equal(embedded, expected[:, ids.shape[1]:])
    reference = model(ids, attention_mask=mask, use_cache=False).logits
    for index in range(2):
        path = tmp_path / str(index)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.save_pretrained(path)
        assert list(path.glob("*.safetensors")) and not list(path.glob("*.bin"))
        assert not list(path.glob("*.py"))
        model = RicursiveQModel.loader.from_pretrained(path, trust_remote_code=True)
        assert "auto_map" not in model.config.to_dict()
        assert model.config.mlp_chunk_size == 3
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], atol=0, rtol=0)
        torch.testing.assert_close(model(ids, attention_mask=mask, use_cache=False).logits, reference)
        assert torch.equal(model.generate(ids, attention_mask=mask, **options), expected)


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_assignment_chunk_budget_and_failed_checkpoint_repair(tmp_path, assign, dtype):
    model = _model().to(dtype)
    other = _model(embed_scale=None).to(dtype)
    other.config.embed_scale = model.config.embed_scale
    other.load_state_dict(copy.deepcopy(model.state_dict()), assign=assign)
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    expected = model(ids, use_cache=False).logits
    torch.testing.assert_close(other(ids, use_cache=False).logits, expected)
    for layer in other.model.layers:
        layer.mlp.chunk_size = 0
    torch.testing.assert_close(other(ids, use_cache=False).logits, expected)
    model.save_pretrained(tmp_path)
    tensor_file = tmp_path / "model.safetensors"
    original = tensor_file.read_bytes()
    tensor_file.write_bytes(b"broken header")
    with pytest.raises(Exception, match="[Hh]eader|[Ss]afetensor"):
        RicursiveQModel.loader.from_pretrained(tmp_path)
    tensor_file.write_bytes(original)
    repaired = RicursiveQModel.loader.from_pretrained(tmp_path, dtype=dtype)
    torch.testing.assert_close(repaired(ids, use_cache=False).logits, expected)


def _tokenizer(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                                  **{str(i): i for i in range(4, 64)}}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]",
                            eos_token="[EOS]", unk_token="[UNK]").save_pretrained(path)


def _released_tokenizer(path, release="preview"):
    for source in iter_port_assets(FIXTURES / ("tokenizer" if release == "preview" else "base_tokenizer"), "*"):
        (path / source.name).write_bytes(read_port_asset(source))


def _assert_released_tokenizer(tokenizer, release="preview"):
    for text, ids in [("Hello world!", [9906, 1917, 0]),
                      ("你好，Ripple!", [57668, 53901, 3922, 49, 52264, 0]),
                      ("  a\tb\n\n c", [220, 264, 2282, 271, 272])]:
        assert tokenizer.encode(text, add_special_tokens=False) == ids
        assert tokenizer.decode(ids) == text
    assert tokenizer.pad_token_id == 100277
    assert tokenizer.eos_token_id == (100265 if release == "preview" else 100257)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}],
                                             tokenize=True, add_generation_prompt=True)
    ids = rendered["input_ids"] if hasattr(rendered, "keys") else rendered
    assert ids == [100264, 882, 198, 9906, 100265, 198, 100264, 78191, 198]


def _hostile(path):
    marker = path / "EXECUTED"
    payload = json.loads((path / "config.json").read_text())
    payload["auto_map"] = {"AutoConfig": "evil.Config", "AutoModel": "evil.Model", "AutoModelForCausalLM": "evil.Model"}
    (path / "config.json").write_text(json.dumps(payload))
    config = json.loads((path / "tokenizer_config.json").read_text())
    config["auto_map"] = {"AutoTokenizer": ["evil.Tokenizer", "evil.Tokenizer"]}
    (path / "tokenizer_config.json").write_text(json.dumps(config))
    code = f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('untrusted checkpoint Python')\n"
    (path / "evil.py").write_text(code)
    (path / "custom_generate").mkdir()
    (path / "custom_generate/generate.py").write_text(code)
    return marker


@pytest.mark.parametrize("trust,bypass", [(False, None), (True, "0"), (True, "1")])
@pytest.mark.parametrize("release", ["preview", "base"])
@torch.inference_mode()
def test_public_hostile_dispatch_generation_policy_retry_and_dense_reload(tmp_path, monkeypatch, trust, bypass, release):
    from gptqmodel import GPTQModel
    model = _model(vocab_size=100352, bos_token_id=100257,
                   eos_token_id=100265 if release == "preview" else 100257, pad_token_id=100277)
    ids = torch.tensor([[3, 5, 7, 9]])
    expected, _ = _oracle(model, ids, torch.ones_like(ids), torch.arange(ids.shape[1]).unsqueeze(0))
    next_token = int(expected[0, -1].argmax())
    # The second EOS entry must actually stop the native generation consumer.
    eos_policy = [60 if next_token != 60 else 61, next_token]
    model.generation_config.eos_token_id = eos_policy
    model.generation_config.bos_token_id = 1
    model.generation_config.do_sample = True
    model.generation_config.top_k = 11
    model.save_pretrained(tmp_path)
    _released_tokenizer(tmp_path, release)
    marker = _hostile(tmp_path)
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    loaded = GPTQModel.load(str(tmp_path), trust_remote_code=trust, code_revision="untrusted-code-revision",
                           device="cpu", dtype=torch.float32)
    assert isinstance(loaded, RicursiveQModel)
    assert type(loaded.model) is RicursiveForCausalLM
    assert not loaded.trust_remote_code
    assert not marker.exists()
    _assert_released_tokenizer(loaded.tokenizer, release)
    torch.testing.assert_close(loaded.model(ids, use_cache=False).logits, expected, atol=2e-5, rtol=2e-5)
    assert loaded.model.generation_config.eos_token_id == eos_policy
    assert loaded.model.generation_config.do_sample and loaded.model.generation_config.top_k == 11
    expected_tokens = model.generate(ids, max_new_tokens=3, do_sample=False)
    assert expected_tokens.shape[-1] == ids.shape[-1] + 1
    assert expected_tokens[0, -1].item() == next_token
    assert torch.equal(loaded.generate(input_ids=ids, max_new_tokens=3, do_sample=False), expected_tokens)
    logits = model(ids, use_cache=False, logits_to_keep=1).logits[:, -1].float()
    cutoff = logits.topk(11).values[:, -1:]
    expected_scores = logits.masked_fill(logits < cutoff, -float("inf"))
    torch.manual_seed(7)
    expected_sample = torch.multinomial(expected_scores.softmax(-1), num_samples=1)
    assert expected_sample.item() != next_token  # Greedy defaults cannot pass.
    torch.manual_seed(7)
    sampled = loaded.generate(input_ids=ids, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
    assert torch.equal(sampled.sequences[:, -1:], expected_sample)
    # Independent reloads may pick different CPU GEMM blockings; masked entries stay exactly -inf.
    torch.testing.assert_close(sampled.scores[0], expected_scores, atol=1e-6, rtol=1e-6)
    for cycle in range(2):
        target = tmp_path / f"dense{cycle}"
        loaded.model.save_pretrained(target)
        loaded.tokenizer.save_pretrained(target)
        loaded = GPTQModel.load(str(target), trust_remote_code=trust, code_revision="untrusted-code-revision",
                               device="cpu", dtype=torch.float32)
        torch.testing.assert_close(loaded.model(ids, use_cache=False).logits, expected, atol=2e-5, rtol=2e-5)
        assert loaded.model.generation_config.eos_token_id == eos_policy
        assert loaded.model.generation_config.do_sample and loaded.model.generation_config.top_k == 11
        assert torch.equal(loaded.generate(input_ids=ids, max_new_tokens=3, do_sample=False), expected_tokens)
        _assert_released_tokenizer(loaded.tokenizer, release)
    assert not marker.exists()


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("kv_heads,gated,valres", [(4, True, True), (2, False, True), (1, True, False)])
def test_public_awq_every_linear_dense_owner_and_two_reloads(tmp_path, offload, kv_heads, gated, valres):
    from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig
    model = _model(num_key_value_heads=kv_heads, use_gated_attention=gated, use_valres=valres)
    model.save_pretrained(tmp_path)
    _tokenizer(tmp_path)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=offload)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16)
    assert (wrapper.turtle_model is not None) == offload
    calibration = [{"input_ids": torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]]),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    # AWQ scale search folds into these norms; every other dense owner is untouched.
    fold_norms = ("pre_feedforward_layernorm",) + (() if valres else ("input_layernorm",))
    dense_reference = {k: v.to(torch.float16).clone() for k, v in model.state_dict().items()
                       if not any(k.endswith("." + name + ".weight") for name in LINEARS | set(fold_norms))}
    scaling_sources = []
    plan_scaling = wrapper.awq_get_modules_for_scaling

    def record_scaling(layer, input_feat, module_kwargs):
        nodes = plan_scaling(layer, input_feat, module_kwargs)
        names = {id(child): name for name, child in layer.named_modules()}
        scaling_sources.extend(names.get(id(node["prev_op"])) for node in nodes)
        return nodes

    wrapper.awq_get_modules_for_scaling = record_scaling
    materializations = []
    materialize = wrapper.shell_module_materialize

    def record_materialization(*args, **kwargs):
        materializations.append(kwargs.get("module_path"))
        return materialize(*args, **kwargs)

    wrapper.shell_module_materialize = record_materialization
    captured_inputs = []

    def record_decoder_input(module, args):
        if module is wrapper.model.model.layers[0] and len(captured_inputs) < 2:
            captured_inputs.append(args[0].detach().clone())

    from torch.nn.modules.module import register_module_forward_pre_hook
    hook = register_module_forward_pre_hook(record_decoder_input)
    try:
        wrapper.quantize(CalibrationConfig(dataset=calibration, batch_size=1, min_length=1), backend=BACKEND.TORCH)
    finally:
        hook.remove()
    if offload:
        assert "model.layers.0" in materializations
    # `ffn_clamp_threshold` keeps `up_proj -> down_proj` out; a GQA `v_proj -> o_proj`
    # group is emitted but dropped by the processor's width check.
    expected_sources = {"pre_feedforward_layernorm", "self_attn.v_proj"}
    if not valres:
        expected_sources.add("input_layernorm")
    assert _model().config.ffn_clamp_threshold
    assert set(scaling_sources) == expected_sources
    assert captured_inputs
    first_expected = F.embedding(calibration[0]["input_ids"], model.model.embed_tokens.weight.to(torch.float16))
    first_expected = first_expected * model.config.embed_scale
    for captured in captured_inputs:
        torch.testing.assert_close(captured, first_expected, atol=0, rtol=0)
    assert all(p.device.type != "meta" for p in wrapper.model.model.layers[0].parameters())
    expected_linears = {f"model.layers.{i}." + name for i in range(2) for name in LINEARS
                        if gated or name != "self_attn.g_proj"}
    assert {name for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)} == expected_linears
    with torch.inference_mode():
        expected = wrapper.model(**calibration[0], use_cache=False).logits
        generated = wrapper.generate(input_ids=calibration[0]["input_ids"], max_new_tokens=3, do_sample=False)
    assert torch.isfinite(expected).all()
    state = {k: v.detach().clone() for k, v in wrapper.model.state_dict().items()}
    for key, value in dense_reference.items():
        torch.testing.assert_close(state[key], value, atol=0, rtol=0)
    for cycle in range(2):
        target = tmp_path / f"awq{cycle}"
        wrapper.save_quantized(str(target))
        assert list(target.glob("*.safetensors")) and not list(target.glob("*.bin"))
        wrapper = GPTQModel.load(str(target), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
        reloaded = wrapper.model.state_dict()
        assert set(reloaded) == set(state)
        for key in state:
            assert reloaded[key].dtype == state[key].dtype
            torch.testing.assert_close(reloaded[key], state[key], atol=0, rtol=0)
        with torch.inference_mode():
            torch.testing.assert_close(wrapper.model(**calibration[0], use_cache=False).logits, expected, atol=0, rtol=0)
            assert torch.equal(wrapper.generate(input_ids=calibration[0]["input_ids"], max_new_tokens=3, do_sample=False), generated)
