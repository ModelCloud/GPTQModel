# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""CPU execution contracts for NVIDIA's actual diffusion decoder, not a causal substitute."""

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, StaticCache

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.models.definitions.nemotron_labs_diffusion import NemotronLabsDiffusionQModel as Definition
from gptqmodel.models.ports.nemotron_labs_diffusion import NemotronLabsDiffusionConfig as Config
from gptqmodel.models.ports.nemotron_labs_diffusion import NemotronLabsDiffusionModel as Model
from gptqmodel.models.ports.nemotron_labs_diffusion.generation import _crop_dynamic_cache, _get_transfer_index
from gptqmodel.models.ports.nemotron_labs_diffusion.modeling_ministral import attention_keep_mask
from gptqmodel.utils.hf import load_model_config




@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(**kwargs):
    values = {"vocab_size": 48, "hidden_size": 24, "intermediate_size": 48, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "mask_token_id": 47,
              "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": None, "max_position_embeddings": 64,
              "attn_implementation": "eager"}
    values.update(kwargs)
    return Config(**values)


def tiny_model(**kwargs):
    torch.manual_seed(750)
    model = Model(tiny_config(**kwargs)).eval()
    # Non-unit gains and non-negligible nonlinear/residual branches expose missing operations.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(torch.linspace(0.6, 1.4, p.numel()).reshape_as(p))
            else:
                p.mul_(4)
        model.diffusion_head.weight[model.mask_token_id].zero_()
    return model


def oracle(model, ids, *, causal=False, position_ids=None, padding=None, block_training=False):
    """Independent float64 equations; never calls a port layer, mask, rotary or attention helper."""
    c, s = model.config, {k: v.double() for k, v in model.state_dict().items()}
    x = s["encoder.embed_tokens.weight"][ids]
    n = ids.shape[1]
    if position_ids is None:
        positions = torch.arange(n // 2 if block_training else n).unsqueeze(0)
    else:
        positions = position_ids
    r, dim = c.rope_parameters, c.head_dim
    inv = torch.tensor([r["rope_theta"] ** (-i / dim) for i in range(0, dim, 2)], dtype=torch.float64)
    factor = 1.0
    if r["rope_type"] == "yarn":
        low = max(math.floor(dim * math.log(r["original_max_position_embeddings"] / (r["beta_fast"] * 2 * math.pi))
                             / (2 * math.log(r["rope_theta"]))), 0)
        high = min(math.ceil(dim * math.log(r["original_max_position_embeddings"] / (r["beta_slow"] * 2 * math.pi))
                              / (2 * math.log(r["rope_theta"]))), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float64) - low) / ((high - low) or 0.001)).clamp(0, 1)
        inv = inv * (1 - ramp + ramp / r["factor"])
        def mscale(scale):
            return 1 + 0.1 * scale * math.log(r["factor"])
        factor = mscale(r["mscale"]) / mscale(r["mscale_all_dim"])
    angle = torch.cat([positions[..., None] * inv] * 2, -1)
    cos, sin = angle.cos() * factor, angle.sin() * factor
    if block_training:
        cos, sin = cos.repeat(1, 2, 1), sin.repeat(1, 2, 1)
    keep = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        for j in range(n):
            if block_training:
                h = n // 2
                keep[i, j] = ((i < h and j < h and i // c.block_size == j // c.block_size)
                              or (i < h and j >= h and i // c.block_size > (j - h) // c.block_size)
                              or (i >= h and j >= h and i >= j))
            else:
                keep[i, j] = not causal or (j <= i and (c.sliding_window is None or i - j < c.sliding_window))
    keep = keep[None, None]
    if padding is not None:
        keep = keep & padding[:, None, None].bool()

    def norm(tensor, w):
        return tensor * torch.rsqrt(tensor.square().mean(-1, keepdim=True) + c.rms_norm_eps) * w

    def rotate(t):
        return t * cos[:, None] + torch.cat([-t[..., dim // 2:], t[..., :dim // 2]], -1) * sin[:, None]

    for layer in range(c.num_hidden_layers):
        p = f"encoder.layers.{layer}."
        z = norm(x, s[p + "input_layernorm.weight"])
        q, k, v = [(z @ s[p + f"self_attn.{name}_proj.weight"].T).reshape(ids.shape[0], n, heads, dim).transpose(1, 2)
                   for name, heads in [("q", c.num_attention_heads), ("k", c.num_key_value_heads), ("v", c.num_key_value_heads)]]
        q, k = rotate(q), rotate(k)
        physical = torch.arange(n, dtype=torch.float64)
        q *= (1 + r["llama_4_scaling_beta"] * (1 + torch.floor(physical / r["original_max_position_embeddings"])).log())[:, None]
        k, v = [a.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, 1) for a in [k, v]]
        score = (q @ k.transpose(-1, -2)) / math.sqrt(dim)
        weights = score.masked_fill(~keep, -torch.inf).softmax(-1).nan_to_num()
        attn = (weights @ v).transpose(1, 2).reshape(ids.shape[0], n, -1)
        x = x + attn @ s[p + "self_attn.o_proj.weight"].T
        z = norm(x, s[p + "post_attention_layernorm.weight"])
        gate = F.silu(z @ s[p + "mlp.gate_proj.weight"].T)
        x = x + (gate * (z @ s[p + "mlp.up_proj.weight"].T)) @ s[p + "mlp.down_proj.weight"].T
    return norm(x, s["encoder.norm.weight"]) @ s["diffusion_head.weight"].T


@pytest.mark.parametrize("paradigm", ["bidirectional", "autoregressive", "block_diff"])
@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_real_forward_matches_independent_decoder(paradigm, kv, impl):
    model = tiny_model(dlm_paradigm=paradigm, num_key_value_heads=kv, attn_implementation=impl)
    ids = torch.tensor([[1, 7, 8, 9], [3, 2, 4, 6]])
    expected = oracle(model, ids, causal=paradigm == "autoregressive")
    actual = model(ids, output_hidden_states=True)
    torch.testing.assert_close(actual.logits.double(), expected, rtol=2e-5, atol=2e-6)
    assert len(actual.hidden_states) == 3
    diagnostic = model(ids, output_attentions=True)
    assert len(diagnostic.attentions) == 2
    torch.testing.assert_close(diagnostic.logits, actual.logits, rtol=2e-5, atol=2e-6)
    changed = ids.clone()
    changed[:, -1] = 11
    delta = (model(changed).logits[:, 0] - actual.logits[:, 0]).abs().max().item()
    assert delta == 0 if paradigm == "autoregressive" else delta > 1e-4


def test_yarn_llama4_scale_and_padding_match_independent_equations():
    rope = {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0, "original_max_position_embeddings": 2,
            "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0, "llama_4_scaling_beta": 0.3}
    model = tiny_model(rope_parameters=rope)
    ids = torch.tensor([[0, 7, 8, 9], [1, 2, 3, 4]])
    mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    expected = oracle(model, ids, padding=mask)
    torch.testing.assert_close(model(ids, attention_mask=mask).logits.double(), expected, rtol=2e-5, atol=2e-6)
    compact = model(ids[:1, 1:], position_ids=torch.tensor([[1, 2, 3]]), cache_position=torch.tensor([1, 2, 3])).logits
    torch.testing.assert_close(model(ids[:1], attention_mask=mask[:1]).logits[:, 1:], compact, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("impl", ["eager", "sdpa"])
@pytest.mark.parametrize("kv", [1, 2, 4])
def test_low_precision_matches_independent_decoder(dtype, impl, kv):
    model = tiny_model(num_key_value_heads=kv, attn_implementation=impl).to(dtype)
    ids = torch.tensor([[1, 7, 8, 9, 10]])
    expected = oracle(model, ids)
    tolerance = 0.025 if dtype == torch.bfloat16 else 0.003
    torch.testing.assert_close(model(ids).logits.double(), expected, rtol=tolerance, atol=tolerance)
    assert model.encoder.rotary_emb.inv_freq.dtype == torch.float32
    expected_freq = 1.0 / (model.config.rope_parameters["rope_theta"] ** (torch.arange(0, 8, 2).float() / 8))
    assert torch.equal(model.encoder.rotary_emb.inv_freq, expected_freq)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_prefix_cache_read_only_then_commit_and_safetensors(tmp_path, dtype, impl):
    model = tiny_model(attn_implementation=impl).to(dtype)
    ids = torch.tensor([[1, 7, 8, 9, 10, 11]])
    full = model(ids, use_causal_mask=True).logits
    cache = model(ids[:, :1], use_cache=True, use_causal_mask=True).past_key_values
    saved = [(l.keys.clone(), l.values.clone()) for l in cache.layers]
    for _ in range(2):
        draft = model(ids[:, 1:5], past_key_values=cache, use_cache=False).logits
        assert torch.isfinite(draft).all() and cache.get_seq_length() == 1
        for layer, (k, v) in zip(cache.layers, saved):
            assert torch.equal(layer.keys, k) and torch.equal(layer.values, v)
    out = model(ids[:, 1:5], past_key_values=cache, use_cache=True, use_causal_mask=True)
    tolerance = 5e-3 if dtype != torch.float32 else 2e-6
    torch.testing.assert_close(out.logits, full[:, 1:5], rtol=tolerance, atol=tolerance)
    assert cache.get_seq_length() == 5
    tail = model(ids[:, 5:], past_key_values=cache, use_cache=True, use_causal_mask=True)
    torch.testing.assert_close(tail.logits, full[:, 5:], rtol=tolerance, atol=tolerance)
    expected = model(ids).logits
    state = {k: v.clone() for k, v in model.state_dict().items()}
    for cycle in range(2):
        destination = tmp_path / str(cycle)
        model.save_pretrained(destination, safe_serialization=True)
        assert list(destination.glob("*.safetensors")) and not list(destination.glob("*.bin"))
        model = Definition.loader.from_pretrained(destination, dtype=dtype, attn_implementation=impl).eval()
        for key, value in model.state_dict().items():
            assert value.dtype == dtype and torch.equal(value, state[key]), key
        torch.testing.assert_close(model(ids).logits, expected, rtol=0, atol=0)


def hostile_checkpoint(path, model):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    marker = path / "EXECUTED"
    (path / "evil.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint code executed')\n")
    config = json.loads((path / "config.json").read_text())
    config["auto_map"] = dict.fromkeys(["AutoConfig", "AutoModel", "AutoModelForCausalLM", "AutoTokenizer"], "evil.Evil")
    (path / "config.json").write_text(json.dumps(config))
    from tests.test_tiny_moe_quant_smoke import _build_local_tokenizer
    _build_local_tokenizer(path)
    return marker


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_public_package_owner_ignores_hostile_auto_map(tmp_path, monkeypatch, trust, bypass, dtype):
    path = tmp_path / "model"
    marker = hostile_checkpoint(path, tiny_model())
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    selected = load_model_config(path, trust_remote_code=trust)
    assert type(selected) is Config
    loaded = GPTQModel.load(str(path), backend=BACKEND.TORCH, device="cpu", dtype=dtype,
                           trust_remote_code=trust, attn_implementation="eager")
    assert isinstance(loaded, Definition) and type(loaded.model) is Model
    reference = tiny_model().to(dtype)
    for key, value in loaded.model.state_dict().items():
        assert value.dtype == dtype and torch.equal(value, reference.state_dict()[key]), key
    assert loaded.model.encoder.rotary_emb.inv_freq.dtype == torch.float32
    expected = reference(torch.tensor([[1, 3, 4]])).logits
    torch.testing.assert_close(loaded.model(torch.tensor([[1, 3, 4]])).logits, expected)
    ids = torch.tensor([[1, 3, 4]])
    options = {"max_new_tokens": 4, "block_length": 4, "threshold": 0.8}
    expected_generation = reference.generate(ids, **options)
    actual_generation = loaded.generate({"input_ids": ids, "attention_mask": torch.ones_like(ids)}, **options)
    assert torch.equal(actual_generation[0], expected_generation[0])
    assert actual_generation[1] == expected_generation[1]
    assert loaded.trust_remote_code is False and not marker.exists()


def test_failed_payload_repair_subfolder_and_retry(tmp_path):
    path = tmp_path / "nested"
    model = tiny_model()
    marker = hostile_checkpoint(path, model)
    payload = path / "model.safetensors"
    original = payload.read_bytes()
    payload.write_bytes(b"broken")
    with pytest.raises(Exception, match="[Hh]eader|[Ss]afetensor|[Dd]eserializ"):
        Definition.loader.from_pretrained(tmp_path, subfolder="nested")
    payload.write_bytes(original)
    repaired = Definition.loader.from_pretrained(tmp_path, subfolder="nested", attn_implementation="eager").eval()
    ids = torch.tensor([[1, 3, 4]])
    torch.testing.assert_close(repaired(ids).logits, model(ids).logits, rtol=0, atol=0)
    assert not marker.exists()


def independent_block_generate(model, ids, tokens, width, causal):
    """Slow no-cache scalar implementation of the pinned confidence sampler."""
    output = ids.clone()
    prefix = None
    nfe = 0
    for _ in range(tokens // width):
        block = torch.full((ids.shape[0], width), model.mask_token_id)
        if causal:
            block[:, 0] = model(output, use_causal_mask=True).logits[:, -1].argmax(-1)
        initial = (block == model.mask_token_id).sum(-1)
        for step in range(width):
            masked = block == model.mask_token_id
            if not masked.any():
                break
            # Materialize exactly the per-layer causal prefix separately, not a bidirectional whole-sequence shortcut.
            prefix = model(output, use_cache=True, use_causal_mask=causal).past_key_values
            logits = model(block, past_key_values=prefix, use_cache=False).logits
            prediction = logits.argmax(-1)
            probabilities = logits.softmax(-1).gather(-1, prediction[..., None]).squeeze(-1)
            for row in range(ids.shape[0]):
                count = int(initial[row] // width + (step < initial[row] % width))
                ranked = sorted(range(width), key=lambda col: float(probabilities[row, col]) if masked[row, col] else -math.inf, reverse=True)
                for col in ranked[:count]:
                    block[row, col] = prediction[row, col]
            nfe += 1
        output = torch.cat([output, block], -1)
        nfe += 1
    return output, nfe


@pytest.mark.parametrize("causal", [False, True])
def test_block_diffusion_generation_matches_scalar_sampler(causal):
    model = tiny_model(num_hidden_layers=1)
    # One layer allows a full recomputation oracle even for the bidirectional frozen prefix.
    ids = torch.tensor([[1, 7, 8]])
    actual, nfe = model.generate(ids, max_new_tokens=8, block_length=4, causal_context=causal)
    expected, steps = independent_block_generate(model, ids, 8, 4, causal)
    assert torch.equal(actual, expected) and nfe == steps
    assert not (actual == model.mask_token_id).any()


def test_diffusion_generation_input_aliases_and_padding_failure_retry():
    model = tiny_model()
    ids = torch.tensor([[1, 7, 8]])
    expected, _ = independent_block_generate(model, ids, tokens=8, width=4, causal=True)
    options = {"max_new_tokens": 8, "block_length": 4}
    for name in ("prompt_ids", "inputs", "input_ids"):
        output, _ = model.generate(**{name: ids}, **options)
        assert torch.equal(output, expected)
    with pytest.raises(ValueError, match="exactly one"):
        model.generate(ids, input_ids=ids, **options)
    with pytest.raises(ValueError, match="unpadded"):
        model.generate(input_ids=ids, attention_mask=torch.tensor([[0, 1, 1]]), **options)
    assert torch.equal(model.generate(inputs=ids, **options)[0], expected)


def test_foreign_cache_fails_before_mutation_then_dynamic_retry():
    model = tiny_model()
    ids = torch.tensor([[1, 7, 8]])
    cache = StaticCache(config=model.config, max_cache_len=8)
    with pytest.raises(ValueError, match="full DynamicCache"):
        model(ids, past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == 0
    output = model(ids, past_key_values=DynamicCache(), use_cache=True, use_causal_mask=True)
    assert output.past_key_values.get_seq_length() == ids.shape[1]
    torch.testing.assert_close(output.logits, model(ids, use_causal_mask=True).logits, rtol=0, atol=0)


def test_ar_and_self_speculation_equal_full_recompute_and_leave_mode_unchanged():
    model = tiny_model()
    ids = torch.tensor([[1, 7, 8]])
    before = model(ids).logits.clone()
    reference = ids.clone()
    for _ in range(8):
        next_id = model(reference, use_causal_mask=True).logits[:, -1].argmax(-1, keepdim=True)
        reference = torch.cat([reference, next_id], -1)
    ar, nfe = model.ar_generate(ids, max_new_tokens=8)
    spec, cost = model.linear_spec_generate(ids, max_new_tokens=8, block_length=4, threshold=0.99)
    assert nfe == 8 and cost > 1
    assert torch.equal(ar, reference)
    assert torch.equal(spec[:, :reference.shape[1]], reference)
    torch.testing.assert_close(model(ids).logits, before, rtol=0, atol=0)


def test_speculative_crop_uses_current_tf5_negative_removal_and_retry():
    cache = DynamicCache()
    keys = torch.arange(24.0).reshape(1, 1, 6, 4)
    cache.update(keys, keys + 1, 0)
    _crop_dynamic_cache(cache, 2)
    assert cache.get_seq_length() == 2 and torch.equal(cache.layers[0].keys, keys[:, :, :2])
    cache.update(keys[:, :, 2:4], keys[:, :, 2:4] + 1, 0)
    assert cache.get_seq_length() == 4


@pytest.mark.parametrize("paradigm", ["bidirectional", "autoregressive", "block_diff"])
def test_source_loss_and_block_mask_semantics(paradigm):
    model = tiny_model(dlm_paradigm=paradigm, block_size=2)
    ids = torch.tensor([[1, 7, 8, 9]])
    masked = torch.tensor([[False, True, False, True]])
    p = torch.full(ids.shape, 0.4)
    actual = model(ids, labels=ids, masked_indices=masked, p_mask=p)
    if paradigm == "autoregressive":
        logits = oracle(model, ids, causal=True)
        expected = F.cross_entropy(logits[:, :-1].reshape(-1, 48), ids[:, 1:].reshape(-1))
        torch.testing.assert_close(actual.loss.double(), expected, rtol=2e-5, atol=2e-6)
    else:
        noisy = torch.where(masked, model.mask_token_id, ids)
        inputs = torch.cat([noisy, ids], -1) if paradigm == "block_diff" else noisy
        logits = oracle(model, inputs, block_training=paradigm == "block_diff")
        expected = F.cross_entropy(logits[:, :4][masked], ids[masked], reduction="sum") / 0.4
        count = masked.sum()
        if paradigm == "block_diff":
            expected += F.cross_entropy(logits[:, 4:-1].reshape(-1, 48), ids[:, 1:].reshape(-1), reduction="sum")
            count += 4
        torch.testing.assert_close(actual.loss[0].double(), expected, rtol=2e-5, atol=2e-6)
        assert actual.loss[1] == count


def test_dense_training_mask_matches_independent_boolean_table():
    mask = attention_keep_mask(8, 0, torch.device("cpu"), causal=False, block_size=2)
    expected = torch.tensor([
        [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0],
        [0,0,1,1,1,1,0,0], [0,0,1,1,1,1,0,0],
        [0,0,0,0,1,0,0,0], [0,0,0,0,1,1,0,0],
        [0,0,0,0,1,1,1,0], [0,0,0,0,1,1,1,1],
    ], dtype=torch.bool)
    assert torch.equal(mask, expected)


def test_confidence_threshold_always_transfers_best_and_preserves_finalized_positions():
    logits = torch.tensor([[[0.0, 2.0, 1.0], [0.0, 0.1, 0.2], [0.0, 10.0, 0.0]]])
    masked = torch.tensor([[True, True, False]])
    tokens, selected = _get_transfer_index(logits, 0, masked, torch.tensor([[2, 2, 0]]), torch.tensor([1]), threshold=0.99)
    assert tokens.tolist() == [[1, 2, 0]] and selected.tolist() == [[True, False, False]]


def test_generation_failure_does_not_change_mode_and_can_retry(monkeypatch):
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3]])
    before = model(ids).logits
    original = model.diffusion_head.forward
    calls = []
    def fail_after_prefill(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected denoising failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(model.diffusion_head, "forward", fail_after_prefill)
    with pytest.raises(RuntimeError, match="injected denoising"):
        model.generate(ids, max_new_tokens=4, block_length=4)
    assert len(calls) == 2
    monkeypatch.setattr(model.diffusion_head, "forward", original)
    torch.testing.assert_close(model(ids).logits, before, rtol=0, atol=0)
    actual = model.generate(ids, max_new_tokens=4, block_length=4)
    expected = tiny_model().generate(ids, max_new_tokens=4, block_length=4)
    assert torch.equal(actual[0], expected[0]) and actual[1] == expected[1]


def test_self_speculation_finalizes_a_predicted_mask_and_respects_seed():
    model = tiny_model(mask_token_id=0)
    with torch.no_grad():
        model.diffusion_head.weight.zero_()
    ids = torch.tensor([[1, 7, 8]])
    calls = []

    def bound_real_decoder(module, args):
        calls.append(True)
        if len(calls) > 64:
            raise AssertionError("Speculative unmasking did not make monotonic progress")

    handle = model.encoder.register_forward_pre_hook(bound_real_decoder)
    try:
        actual, cost = model.linear_spec_generate(ids, max_new_tokens=9, block_length=4, threshold=0.99)
    finally:
        handle.remove()
    expected = torch.cat([ids, torch.zeros(1, 9, dtype=torch.long)], -1)
    assert torch.equal(actual, expected) and cost <= 16


def test_generation_eos_sampling_and_thinking_budget():
    model = tiny_model()
    ids = torch.tensor([[1, 7, 8]])
    first = model(ids, use_causal_mask=True).logits[:, -1].argmax(-1).item()
    for generate in (model.ar_generate, model.linear_spec_generate):
        output, cost = generate(ids, max_new_tokens=8, eos_token_id=first)
        assert torch.equal(output, torch.cat([ids, torch.tensor([[first]])], -1))
        assert cost == 1
    output, _ = model.ar_generate(ids, max_new_tokens=5, max_thinking_tokens=2, end_think_token_id=46)
    assert output[0, ids.shape[1] + 2] == 46
    torch.manual_seed(37)
    expected = ids.clone()
    for _ in range(5):
        probabilities = torch.softmax(oracle(model, expected, causal=True)[:, -1].float() / 0.7, -1)
        expected = torch.cat([expected, torch.multinomial(probabilities, num_samples=1)], -1)
    torch.manual_seed(37)
    assert torch.equal(model.ar_generate(ids, max_new_tokens=5, temperature=0.7)[0], expected)


@pytest.mark.parametrize("paradigm", ["bidirectional", "autoregressive", "block_diff"])
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("kv", [1, 2, 4])
def test_public_plain_awq_pack_infer_two_safetensors_cycles(tmp_path, paradigm, offload, kv):
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    path = tmp_path / "native"
    model = tiny_model(vocab_size=128, hidden_size=32, intermediate_size=64, head_dim=16,
                       dlm_paradigm=paradigm, mask_token_id=127, num_key_value_heads=kv)
    original_dense = {key: value.half().clone() for key, value in model.state_dict().items()
                      if not (".self_attn." in key or ".mlp." in key)}
    marker = hostile_checkpoint(path, model)
    config = AWQConfig(bits=4, group_size=16, device="cpu", offload_to_disk=offload, fallback=None)
    loaded = GPTQModel.load(str(path), quantize_config=config, backend=BACKEND.TORCH,
                           dtype=torch.float16, attn_implementation="eager")
    rows = [{"input_ids": torch.tensor([[1, 4, 7, 9, 10, 12, 15, 2]]), "attention_mask": torch.ones(1, 8, dtype=torch.long)},
            {"input_ids": torch.tensor([[1, 3, 6, 8, 11, 13, 16, 2]]), "attention_mask": torch.ones(1, 8, dtype=torch.long)}]
    loaded.quantize(
        CalibrationConfig(dataset=rows, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    assert loaded.quantize_config.fallback is None
    packed = {name: module for name, module in loaded.model.named_modules() if isinstance(module, AwqTorchLinear)}
    expected = {f"encoder.layers.{i}.{projection}" for i in range(2) for projection in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
    assert set(packed) == expected
    assert all(module.qweight.dtype == torch.int32 for module in packed.values())
    state = {key: value.clone() for key, value in loaded.model.state_dict().items()}
    for key, expected_value in original_dense.items():
        assert torch.equal(state[key], expected_value), key
    ids = rows[0]["input_ids"]
    expected_logits = loaded.model(ids).logits
    assert torch.isfinite(expected_logits).all()
    expected_ids, _ = loaded.model.ar_generate(ids, max_new_tokens=3)
    generation_options = {"max_new_tokens": 4, "block_length": 4, "threshold": 0.8}
    expected_diffusion, expected_nfe = loaded.model.generate(ids, **generation_options)
    actual_diffusion, actual_nfe = loaded.generate(ids, **generation_options)
    assert torch.equal(actual_diffusion, expected_diffusion) and actual_nfe == expected_nfe
    for cycle in range(2):
        destination = tmp_path / f"packed-{cycle}"
        loaded.save(str(destination))
        assert list(destination.glob("*.safetensors")) and not list(destination.glob("*.bin"))
        loaded = GPTQModel.load(str(destination), device="cpu", backend=BACKEND.AWQ_TORCH,
                                dtype=torch.float16, attn_implementation="eager")
        actual = loaded.model.state_dict()
        assert set(actual) == set(state)
        for key, expected_value in state.items():
            assert actual[key].dtype == expected_value.dtype and torch.equal(actual[key], expected_value), key
        torch.testing.assert_close(loaded.model(ids).logits, expected_logits, rtol=0, atol=0)
        assert torch.equal(loaded.model.ar_generate(ids, max_new_tokens=3)[0], expected_ids)
        actual_diffusion, actual_nfe = loaded.generate(inputs=ids, **generation_options)
        assert torch.equal(actual_diffusion, expected_diffusion) and actual_nfe == expected_nfe
    assert not marker.exists()
