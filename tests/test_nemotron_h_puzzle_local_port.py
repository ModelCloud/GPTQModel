# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Tiny real Puzzle execution, independent equations and public consumer transitions."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models.definitions.nemotron_h_puzzle import NemotronHPuzzleQModel
from gptqmodel.models.ports.nemotron_h_puzzle import NemotronHPuzzleConfig, NemotronHPuzzleForCausalLM
from gptqmodel.models.ports.nemotron_h_puzzle.modeling_nemotron_h_puzzle import NemotronHPuzzleCache
from gptqmodel.quantization import AWQConfig
from gptqmodel.utils.hf import load_model_config


PORT = Path(__file__).parents[1] / "gptqmodel/models/ports/nemotron_h_puzzle"
TOKENS = torch.tensor([[1, 5, 8, 4, 7, 6, 3], [1, 9, 3, 5, 6, 4, 7]])


@pytest.fixture(autouse=True, scope="module")
def bounded_threads():
    count = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(count)


def tiny_config(**kwargs):
    values = {"vocab_size": 32, "hidden_size": 16, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
                  "mamba_num_heads": 4, "mamba_head_dim": 4, "mamba_n_groups": 2, "ssm_state_size": 3, "mamba_chunk_size": 4,
                  "n_routed_experts": 4, "n_group": 1, "topk_group": 1, "moe_latent_size": 8,
                  "moe_shared_expert_intermediate_size": 12, "num_experts_per_tok": 1, "moe_intermediate_size": 12,
                  "routed_scaling_factor": 1.7, "eos_token_id": None,
                  "block_configs": [{"block_type": "mamba"},
                                 {"block_type": "moe", "moe_intermediate_size": 12, "num_experts_per_tok": 1},
                                 {"block_type": "attention"},
                                 {"block_type": "moe", "moe_intermediate_size": 20, "num_experts_per_tok": 2}]}
    values.update(kwargs)
    return NemotronHPuzzleConfig(**values)


def varied_model(config=None, dtype=torch.float32):
    torch.manual_seed(271)
    model = NemotronHPuzzleForCausalLM(config or tiny_config()).to(dtype).eval()
    for layer in model.model.layers:
        if layer.block_type == "moe":
            layer.mixer.gate.e_score_correction_bias = layer.mixer.gate.e_score_correction_bias.float()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("norm_f.weight"):
                p.copy_(torch.linspace(0.65, 1.35, p.numel()).reshape_as(p))
            elif name.endswith("A_log"):
                p.copy_(torch.linspace(-0.5, 0.5, p.numel()))
            elif name.endswith("dt_bias"):
                p.copy_(torch.linspace(-2.0, -0.7, p.numel()))
            elif name.endswith("D"):
                p.copy_(torch.linspace(0.5, 1.2, p.numel()))
            else:
                p.uniform_(-0.22, 0.22)
        for name, buffer in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                buffer.copy_(torch.linspace(-0.09, 0.12, buffer.numel()))
    return model


def save_tokenizer(path):
    tokenizer = Tokenizer(models.WordLevel({"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
                                            "the": 4, "quick": 5, "brown": 6, "fox": 7,
                                            "jumps": 8, "over": 9, "lazy": 10, "dog": 11}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<s>", eos_token="</s>",
                                  pad_token="<pad>", unk_token="<unk>")
    fast.save_pretrained(path)
    return fast


def hostile_checkpoint(path, model):
    model.save_pretrained(path)
    save_tokenizer(path)
    marker = path / "EXECUTED"
    (path / "hostile.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\nraise RuntimeError('checkpoint code executed')\n")
    config_path = path / "config.json"
    values = json.loads(config_path.read_text())
    values["auto_map"] = {"AutoConfig": "hostile.RemoteConfig", "AutoModelForCausalLM": "hostile.RemoteModel"}
    config_path.write_text(json.dumps(values))
    return marker


def _linear(x, module):
    result = x @ module.weight.double().T
    return result if module.bias is None else result + module.bias.double()


def _norm(x, weight, eps):
    return (x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + eps)) * weight.double()


def _mlp(x, module):
    x = _linear(x, module.up_proj)
    # Tiny oracle uses source relu2, not the port's activation registry.
    return _linear(x.relu().square(), module.down_proj)


def _moe(x, module):
    c = module.config
    flat = x.reshape(-1, x.shape[-1]).double()
    scores = (flat @ module.gate.weight.double().T).sigmoid()
    choices = scores + module.gate.e_score_correction_bias.double()
    selected = []
    for row in choices:
        group_scores = [sum(sorted(row[g:g + c.n_routed_experts // c.n_group].tolist(), reverse=True)[:2])
                        for g in range(0, c.n_routed_experts, c.n_routed_experts // c.n_group)]
        groups = sorted(range(c.n_group), key=lambda g: group_scores[g], reverse=True)[:c.topk_group]
        allowed = [row[i].item() if i // (c.n_routed_experts // c.n_group) in groups else 0.0
                   for i in range(c.n_routed_experts)]
        selected.append(sorted(range(c.n_routed_experts), key=lambda i: allowed[i], reverse=True)[:c.num_experts_per_tok])
    routed = flat if isinstance(module.fc1_latent_proj, torch.nn.Identity) else _linear(flat, module.fc1_latent_proj)
    output = torch.zeros_like(routed)
    for token, experts in enumerate(selected):
        weights = scores[token, experts]
        if c.norm_topk_prob:
            weights = weights / (weights.sum() + 1e-20)
        for expert, weight in zip(experts, weights):
            output[token] += _mlp(routed[token], module.experts[expert]) * weight * c.routed_scaling_factor
    if not isinstance(module.fc2_latent_proj, torch.nn.Identity):
        output = _linear(output, module.fc2_latent_proj)
    return output.reshape_as(x) + _mlp(x, module.shared_experts)


def _mamba(x, module, mask=None):
    """Independent scalar-in-time recurrence from the pinned Mamba-2 equations."""
    c = module.config
    if mask is not None:
        x = x * mask[..., None]
    projected = _linear(x, module.in_proj)
    inner = c.mamba_num_heads * c.mamba_head_dim
    conv_dim = inner + 2 * c.n_groups * c.ssm_state_size
    gate, conv_input, dt = projected.split((inner, conv_dim, c.mamba_num_heads), -1)
    conv_output = torch.zeros_like(conv_input)
    for t in range(x.shape[1]):
        for lag in range(c.conv_kernel):
            if t - lag >= 0:
                conv_output[:, t] += conv_input[:, t - lag] * module.conv1d.weight[:, 0, c.conv_kernel - 1 - lag].double()
    if module.conv1d.bias is not None:
        conv_output += module.conv1d.bias.double()
    conv_output = conv_output * conv_output.sigmoid()
    if mask is not None:
        conv_output *= mask[..., None]
    values, B, C = conv_output.split((inner, c.n_groups * c.ssm_state_size, c.n_groups * c.ssm_state_size), -1)
    values = values.reshape(*x.shape[:2], c.mamba_num_heads, c.mamba_head_dim)
    B = B.reshape(*x.shape[:2], c.n_groups, c.ssm_state_size).repeat_interleave(c.mamba_num_heads // c.n_groups, 2)
    C = C.reshape(*x.shape[:2], c.n_groups, c.ssm_state_size).repeat_interleave(c.mamba_num_heads // c.n_groups, 2)
    dt = torch.log1p(torch.exp(dt + module.dt_bias.double())).clamp_min(c.time_step_min)
    state = torch.zeros(x.shape[0], c.mamba_num_heads, c.mamba_head_dim, c.ssm_state_size, dtype=torch.float64)
    outputs = []
    for t in range(x.shape[1]):
        delta = dt[:, t, :, None, None]
        state = state * (-module.A_log.double().exp()[None, :, None, None] * delta).exp()
        state += delta * B[:, t, :, None, :] * values[:, t, :, :, None]
        y = (state * C[:, t, :, None, :]).sum(-1) + values[:, t] * module.D.double()[None, :, None]
        outputs.append(y.flatten(1))
    y = torch.stack(outputs, 1) * (gate * gate.sigmoid())
    groups = y.reshape(*y.shape[:-1], c.n_groups, inner // c.n_groups)
    normalized = groups * torch.rsqrt(groups.square().mean(-1, keepdim=True) + c.layer_norm_epsilon)
    normalized = normalized.reshape_as(y) * module.norm.weight.double()
    return _linear(normalized, module.out_proj)


def _attention(x, module, mask=None):
    c = module.config
    batch, length = x.shape[:2]
    q = _linear(x, module.q_proj).view(batch, length, c.num_attention_heads, module.head_dim).transpose(1, 2)
    k = _linear(x, module.k_proj).view(batch, length, c.num_key_value_heads, module.head_dim).transpose(1, 2)
    v = _linear(x, module.v_proj).view(batch, length, c.num_key_value_heads, module.head_dim).transpose(1, 2)
    k, v = [a.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, 1) for a in (k, v)]
    scores = q @ k.transpose(-1, -2) / module.head_dim ** 0.5
    blocked = torch.arange(length)[None, :] > torch.arange(length)[:, None]
    blocked = blocked[None, None].expand(batch, 1, length, length)
    if mask is not None:
        blocked = blocked | ~mask.bool()[:, None, None, :]
    scores = scores.masked_fill(blocked, torch.finfo(torch.float32).min)
    output = scores.softmax(-1) @ v
    return _linear(output.transpose(1, 2).reshape(batch, length, -1), module.o_proj)


def oracle(model, ids, mask=None):
    x = model.model.embeddings.weight[ids].double()
    for layer in model.model.layers:
        h = _norm(x, layer.norm.weight, model.config.layer_norm_epsilon)
        if layer.block_type == "mamba":
            delta = _mamba(h, layer.mixer, mask)
        elif layer.block_type == "attention":
            delta = _attention(h, layer.mixer, mask)
        else:
            delta = _moe(h, layer.mixer)
        x = x + delta
    return _linear(_norm(x, model.model.norm_f.weight, model.config.layer_norm_epsilon), model.lm_head)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("latent", [None, 8])
def test_full_decoder_matches_independent_equations(kv_heads, bias, latent):
    model = varied_model(tiny_config(num_key_value_heads=kv_heads, attention_bias=bias, mlp_bias=bias,
                                    use_bias=bias, moe_latent_size=latent, n_group=2, topk_group=1))
    with torch.no_grad():
        expected = oracle(model, TOKENS)
        actual = model(TOKENS, use_cache=False).logits
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("layout", ["hybrid", "mamba", "attention", "moe"])
def test_dtype_cache_generation_and_two_safetensors_cycles(tmp_path, dtype, implementation, layout):
    kwargs = {} if layout == "hybrid" else {"block_configs": [{"block_type": layout}] * 2}
    model = varied_model(tiny_config(attn_implementation=implementation, **kwargs), dtype)
    ids = TOKENS[:, :6]
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        prefix = model(ids[:, :1], use_cache=True)
        chunk = model(ids[:, 1:5], use_cache=True, past_key_values=prefix.past_key_values)
        final = model(ids[:, 5:], use_cache=True, past_key_values=chunk.past_key_values)
        assert final.past_key_values.get_seq_length() == 6
        tolerance = {torch.float32: 2e-6, torch.float16: 0.003, torch.bfloat16: 0.03}[dtype]
        torch.testing.assert_close(chunk.logits, expected[:, 1:5], atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(final.logits, expected[:, 5:], atol=tolerance, rtol=tolerance)
        generated = model.generate(ids[:1], max_new_tokens=3, do_sample=False)
        assert generated.shape == (1, 9)
        beam = model.generate(ids[:1], max_new_tokens=2, num_beams=2)
        assert beam.shape == (1, 8)
        torch.testing.assert_close(beam, model.generate(ids[:1], max_new_tokens=2, num_beams=2, use_cache=False))
        torch.testing.assert_close(generated, model.generate(ids[:1], max_new_tokens=3, use_cache=False))
        state = {k: v.clone() for k, v in model.state_dict().items()}
        for cycle in range(2):
            path = tmp_path / str(cycle)
            model.save_pretrained(path)
            assert not list(path.glob("*.bin"))
            model = NemotronHPuzzleForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation=implementation).eval()
            assert set(model.state_dict()) == set(state)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, state[name], atol=0, rtol=0)
            torch.testing.assert_close(model(ids, use_cache=False).logits, expected, atol=0, rtol=0)


@pytest.mark.parametrize("bypass", [None, "0", "1"])
@pytest.mark.parametrize("trust", [False, True])
def test_public_load_ignores_hostile_maps_and_retries(tmp_path, monkeypatch, bypass, trust):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    source = varied_model()
    marker = hostile_checkpoint(tmp_path, source)
    assert type(load_model_config(tmp_path, trust_remote_code=trust)) is NemotronHPuzzleConfig
    tensor_path = tmp_path / "model.safetensors"
    payload = tensor_path.read_bytes()
    tensor_path.write_bytes(b"broken safetensors")
    with pytest.raises(Exception, match="[Hh]eader|safetensors|Safetensor"):
        GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    tensor_path.write_bytes(payload)
    loaded = GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert type(loaded) is NemotronHPuzzleQModel
    assert type(loaded.model) is NemotronHPuzzleForCausalLM
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(TOKENS).logits, source(TOKENS).logits, atol=0, rtol=0)
    assert not marker.exists()


def test_left_padding_causality_replay_and_cache_reset():
    model = varied_model()
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1]])
    with torch.no_grad():
        expected = oracle(model, TOKENS, mask)
        actual = model(TOKENS, attention_mask=mask, use_cache=False).logits
        torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-5)
        changed = TOKENS.clone()
        changed[:, -1] = 11
        torch.testing.assert_close(model(changed, use_cache=False).logits[:, :-1],
                                   model(TOKENS, use_cache=False).logits[:, :-1], atol=2e-6, rtol=2e-5)
        prefix = model(TOKENS[:, :4], attention_mask=mask[:, :4], use_cache=True)
        cache = prefix.past_key_values
        suffix = model(TOKENS[:, 4:], attention_mask=mask, use_cache=True, past_key_values=cache)
        torch.testing.assert_close(suffix.logits, actual[:, 4:], atol=2e-6, rtol=2e-5)
        cache.reset()
        assert cache.get_seq_length() == 0
        torch.testing.assert_close(model(TOKENS, use_cache=True, past_key_values=cache).logits,
                                   model(TOKENS, use_cache=False).logits, atol=2e-6, rtol=2e-5)
        hidden = model.model.embeddings(TOKENS)
        for layer in model.model.layers:
            captured = hidden
            kwargs = NemotronHPuzzleQModel.prepare_layer_replay_kwargs(
                object.__new__(NemotronHPuzzleQModel), layer, [hidden],
                {"attention_mask": mask, "use_cache": False}, torch.device("cpu"))
            replay = layer(captured, **kwargs)
            h = _norm(captured, layer.norm.weight, model.config.layer_norm_epsilon)
            delta = {"mamba": _mamba, "attention": _attention}.get(layer.block_type)
            reference = _moe(h, layer.mixer) if delta is None else delta(h, layer.mixer, mask)
            torch.testing.assert_close(replay.double(), captured.double() + reference, atol=2e-6, rtol=2e-5)
            hidden = replay


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_greedy_generation_matches_independent_full_decoder_oracle(kv_heads):
    model = varied_model(tiny_config(num_key_value_heads=kv_heads, n_group=2, topk_group=1))
    expected = TOKENS[:1, :4]
    with torch.no_grad():
        for _ in range(3):
            next_token = oracle(model, expected)[:, -1].argmax(-1, keepdim=True)
            expected = torch.cat((expected, next_token), -1)
        actual = model.generate(TOKENS[:1, :4], max_new_tokens=3, do_sample=False)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("assign", [False, True])
def test_repeated_state_assignment_preserves_geometry_and_values(assign):
    source = varied_model(tiny_config(mlp_bias=True))
    target = NemotronHPuzzleForCausalLM(source.config).eval()
    for _ in range(2):
        target.load_state_dict(source.state_dict(), assign=assign)
        with torch.no_grad():
            torch.testing.assert_close(target(TOKENS).logits, source(TOKENS).logits, atol=0, rtol=0)


def test_unsupported_cache_masks_and_output_modes_fail_before_reuse():
    model = varied_model()
    with pytest.raises(ValueError, match="own dynamic"):
        from transformers.cache_utils import DynamicCache
        model(TOKENS, past_key_values=DynamicCache(config=model.config))
    with pytest.raises(ValueError, match="padding mask"):
        model(TOKENS, attention_mask=torch.ones(2, 4))
    with pytest.raises(ValueError, match="attention weights"):
        model(TOKENS, output_attentions=True)
    with pytest.raises(ValueError, match="dynamic hybrid"):
        model.generate(TOKENS, max_new_tokens=1, cache_implementation="static")
    cache = NemotronHPuzzleCache(model.config)
    with pytest.raises(ValueError, match="cropped"):
        cache.crop(-1)
    assert model(TOKENS).logits.isfinite().all()


def test_embeddings_hidden_states_loss_and_invalid_mask_retry():
    model = varied_model()
    with torch.no_grad():
        native = model(TOKENS, output_hidden_states=True)
        embedded = model(inputs_embeds=model.get_input_embeddings()(TOKENS), use_cache=False)
        torch.testing.assert_close(embedded.logits, native.logits, atol=0, rtol=0)
        assert len(native.hidden_states) == len(model.model.layers) + 1
        torch.testing.assert_close(native.hidden_states[0], model.get_input_embeddings()(TOKENS), atol=0, rtol=0)
        labels = TOKENS.clone()
        loss = model(TOKENS, labels=labels, use_cache=False).loss
        expected_loss = torch.nn.functional.cross_entropy(native.logits[:, :-1].reshape(-1, model.config.vocab_size),
                                                         labels[:, 1:].reshape(-1))
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(model(TOKENS, return_dict=False, use_cache=False)[0], native.logits, atol=0, rtol=0)
        model.config.return_dict = False
        torch.testing.assert_close(model(TOKENS, use_cache=False)[0], native.logits, atol=0, rtol=0)
        model.config.return_dict = True
        for bad in (torch.ones(2, 1, 7, 7, dtype=torch.bool), torch.ones(3, 1, 7, 7), {"linear_attention": None}):
            cache = NemotronHPuzzleCache(model.config)
            with pytest.raises(ValueError, match="mask"):
                model(TOKENS, attention_mask=bad, past_key_values=cache)
            assert cache.get_seq_length() == 0
            torch.testing.assert_close(model(TOKENS, past_key_values=cache).logits, native.logits, atol=0, rtol=0)


@pytest.mark.parametrize("kv_heads,latent,bias,offload", [(1, None, False, False), (2, 32, False, True),
                                                        (4, 32, True, False)])
def test_public_plain_awq_pack_infer_and_two_reload_cycles(tmp_path, kv_heads, latent, bias, offload):
    native = tmp_path / "native"
    native.mkdir()
    config = tiny_config(hidden_size=64, num_key_value_heads=kv_heads, head_dim=16, mamba_num_heads=4,
                         mamba_head_dim=16, moe_latent_size=latent, mlp_bias=bias, attention_bias=bias,
                         block_configs=[{"block_type": "mamba"}, {"block_type": "moe", "moe_intermediate_size": 64,
                                                                 "num_experts_per_tok": 1},
                                        {"block_type": "attention"}], moe_shared_expert_intermediate_size=64)
    source = varied_model(config, torch.float16)
    # Force unsampled experts: they must still be reached and packed through zero-valued dummy calls.
    source.model.layers[1].mixer.gate.e_score_correction_bias.copy_(torch.tensor([5., -5., -5., -5.]))
    source.save_pretrained(native)
    tokenizer = save_tokenizer(native)
    dense = {k: v.clone() for k, v in source.state_dict().items()}
    mtp_key = "mtp.layers.0.eh_proj.weight"
    dense[mtp_key] = torch.linspace(-0.4, 0.6, 32).view(2, 16)
    tensor_file = native / "model.safetensors"
    checkpoint = load_file(tensor_file)
    checkpoint[mtp_key] = dense[mtp_key]
    save_file(checkpoint, tensor_file, metadata={"format": "pt"})
    qcfg = AWQConfig(bits=4, group_size=16, fallback=None, device="cpu", offload_to_disk=offload)
    wrapped = GPTQModel.load(str(native), quantize_config=qcfg, dtype=torch.float16)
    calibration = [dict(tokenizer(text, return_tensors="pt")) for text in
                   ["the quick brown fox jumps over lazy dog", "the fox jumps over the dog"]]
    wrapped.quantize(calibration, batch_size=1, calibration_data_min_length=1, backend=BACKEND.TORCH)
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear
    packed = {name for name, module in wrapped.model.named_modules() if isinstance(module, BaseQuantLinear)}
    expected_linears = {name for name, module in source.named_modules() if isinstance(module, torch.nn.Linear)
                        and name.startswith("model.layers.") and not name.endswith("gate")}
    assert packed == expected_linears
    assert len(packed) == (18 if latent is not None else 16)
    with torch.no_grad():
        expected = wrapped.model(TOKENS, use_cache=False).logits
    assert expected.isfinite().all()
    for layer in wrapped.model.model.layers:
        assert wrapped.awq_get_modules_for_scaling(layer, {}, {}) == []
    for cycle in range(2):
        out = tmp_path / f"awq{cycle}"
        wrapped.save(str(out))
        assert not list(out.glob("*.bin"))
        files = list(out.glob("*.safetensors"))
        assert files
        disk_state = {k: v for file in files for k, v in load_file(file).items()}
        for key, value in dense.items():
            if not any(key.startswith(name + ".") for name in packed):
                torch.testing.assert_close(disk_state[key], value, atol=0, rtol=0)
        wrapped = GPTQModel.load(str(out), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
        loaded_state = wrapped.model.state_dict()
        assert set(disk_state) - set(loaded_state) == {mtp_key}
        assert all(key.endswith(".qzeros_format") for key in set(loaded_state) - set(disk_state))
        for key, value in disk_state.items():
            if key != mtp_key:
                torch.testing.assert_close(loaded_state[key], value, atol=0, rtol=0, msg=lambda text: f"{key}: {text}")
        with torch.no_grad():
            torch.testing.assert_close(wrapped.model(TOKENS, use_cache=False).logits, expected, atol=0, rtol=0)
