# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real tiny Klear execution and definition-owned loading regressions."""

import json

import pytest
import torch
import torch.nn.functional as F

from gptqmodel import CalibrationConfig
from gptqmodel.models.auto import check_and_get_model_definition
from gptqmodel.models.definitions.klear import KlearQModel
from gptqmodel.models.ports.klear.configuration_klear import KlearConfig
from gptqmodel.models.ports.klear.modeling_klear import KlearMoeForCausalLM
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code
from gptqmodel.utils.security import BYPASS_ENV


def tiny_config(**kwargs):
    values = {
        "vocab_size": 41, "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 32,
        "moe_intermediate_size": 8, "num_experts": 3, "num_experts_per_tok": 2,
        "n_shared_experts": 2, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
        "attn_implementation": "eager",
    }
    values.update(kwargs)
    return KlearConfig(**values)


def poison_checkpoint(path, model_type="Klear"):
    model = KlearMoeForCausalLM(tiny_config()).eval()
    model.save_pretrained(path, safe_serialization=True)
    marker = path / "EXECUTED"
    (path / "evil.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "raise RuntimeError('checkpoint Python must not execute')\n"
    )
    config_path = path / "config.json"
    payload = json.loads(config_path.read_text())
    payload.update(model_type=model_type, auto_map={
        "AutoConfig": "evil.Config", "AutoModel": "evil.Model", "AutoModelForCausalLM": "evil.Model",
    })
    config_path.write_text(json.dumps(payload))
    return model, marker


@pytest.mark.parametrize("model_type", ["Klear", "klear", "KLEAR"])
@pytest.mark.parametrize("trust,bypass", [(False, False), (True, True)])
def test_definition_loading_ignores_checkpoint_python(tmp_path, monkeypatch, model_type, trust, bypass):
    reference, marker = poison_checkpoint(tmp_path, model_type)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")  # Existing security boundary, not a behavior switch.
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    assert check_and_get_model_definition(str(tmp_path), trust_remote_code=trust) is KlearQModel
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config) is KlearConfig
    assert resolve_trust_remote_code(tmp_path, trust_remote_code=trust) is False
    restored = KlearQModel.loader.from_pretrained(
        tmp_path, config=config, trust_remote_code=trust, code_revision="not-a-real-revision", dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    assert type(restored) is KlearMoeForCausalLM
    assert set(reference.state_dict()) == set(restored.state_dict())
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    ids = torch.tensor([[1, 4, 7, 9]])
    with torch.inference_mode():
        # Weights are bit-identical above; mmap'd safetensors storage can take a
        # different (alignment-dependent) GEMM path, so allow 1-ulp logits drift.
        torch.testing.assert_close(reference(ids).logits, restored(ids).logits, rtol=1e-6, atol=1e-7)
    assert not marker.exists()


def _oracle(model, ids, mask=None):
    """Direct equations from upstream; no port forward/helpers or HF attention/cache."""
    c, state = model.config, model.state_dict()
    x = F.embedding(ids, state["model.embed_tokens.weight"])
    b, length = ids.shape
    positions = torch.arange(length)
    frequencies = positions[:, None].float() / (
        c.rope_parameters["rope_theta"] ** (torch.arange(0, c.head_dim, 2).float() / c.head_dim)
    )
    if c.rope_parameters["rope_type"] == "linear":
        frequencies /= c.rope_parameters["factor"]
    angles = torch.cat([frequencies, frequencies], -1)[None, None]

    def linear(x, name):
        return F.linear(x, state[name + ".weight"], state.get(name + ".bias"))

    def norm(x, name):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + c.rms_norm_eps)
        return state[name + ".weight"] * y.to(x.dtype)

    def mlp(x, name):
        return linear(F.silu(linear(x, name + ".gate_proj")) * linear(x, name + ".up_proj"), name + ".down_proj")

    routers = []
    for index in range(c.num_hidden_layers):
        prefix = f"model.layers.{index}"
        residual, y = x, norm(x, prefix + ".input_layernorm")
        attn = prefix + ".self_attn"
        q = norm(linear(y, attn + ".q_proj").reshape(b, length, c.num_attention_heads, c.head_dim), attn + ".q_norm")
        k = norm(linear(y, attn + ".k_proj").reshape(b, length, c.num_key_value_heads, c.head_dim), attn + ".k_norm")
        v = linear(y, attn + ".v_proj").reshape(b, length, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        q = q * angles.cos() + torch.cat([-q[..., c.head_dim // 2:], q[..., :c.head_dim // 2]], -1) * angles.sin()
        k = k * angles.cos() + torch.cat([-k[..., c.head_dim // 2:], k[..., :c.head_dim // 2]], -1) * angles.sin()
        k, v = [t.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, dim=1) for t in (k, v)]
        scores = q @ k.transpose(-1, -2) / c.head_dim ** 0.5
        allowed = positions[None, :] <= positions[:, None]
        if c.sliding_window is not None:
            allowed &= positions[None, :] > positions[:, None] - c.sliding_window
        allowed = allowed[None, None].expand(b, 1, length, length)
        if mask is not None:
            allowed = allowed & mask[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        y = (scores.float().softmax(-1).to(v.dtype) @ v).transpose(1, 2).reshape(b, length, -1)
        x = residual + linear(y, attn + ".o_proj")
        residual, y = x, norm(x, prefix + ".post_attention_layernorm")
        block = prefix + ".mlp"
        if block + ".gate.weight" not in state:
            y = mlp(y, block)
            routers.append(None)
        else:
            logits = F.linear(y.float(), state[block + ".gate.weight"].float())
            routers.append(logits.reshape(-1, c.num_experts))
            probabilities = logits.sigmoid()
            selected = (probabilities + state[block + ".expert_bias"]).topk(c.num_experts_per_tok, -1).indices
            weights = probabilities.gather(-1, selected)
            if c.norm_topk_prob:
                weights = weights / weights.sum(-1, keepdim=True)
            routed = torch.zeros_like(y)
            # Evaluate all experts independently, rather than the port's sparse scatter.
            for expert in range(c.num_experts):
                weight = ((selected == expert) * weights).sum(-1, keepdim=True).to(y.dtype)
                routed += mlp(y, block + f".experts.{expert}") * weight
            coefficient = linear(y, block + ".coefficient").softmax(-1)
            y = routed * coefficient[..., :1] + mlp(y, block + ".shared_experts") * coefficient[..., 1:]
        x = residual + y
    return linear(norm(x, "model.norm"), "lm_head"), routers


@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("layout", ["moe", "dense", "mixed", "periodic"])
@pytest.mark.parametrize("bias", [False, True])
def test_native_forward_matches_independent_equations(kv, layout, bias):
    torch.manual_seed(71)
    variants = {"moe": {}, "dense": {"num_experts": 0}, "mixed": {"mlp_only_layers": [0]},
                "periodic": {"decoder_sparse_step": 2}}
    model = KlearMoeForCausalLM(tiny_config(num_key_value_heads=kv, attention_bias=bias, **variants[layout])).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            if hasattr(layer.mlp, "expert_bias"):
                layer.mlp.expert_bias.copy_(torch.tensor([0.31, -0.22, 0.07]))
                layer.mlp.coefficient.bias.copy_(torch.tensor([0.23, -0.61]))
        ids = torch.tensor([[1, 4, 6, 9, 11], [1, 3, 8, 12, 0]])
        mask = ids.ne(0)
        actual = model(ids, attention_mask=mask, output_router_logits=True, use_cache=False)
        expected, routers = _oracle(model, ids, mask)
    torch.testing.assert_close(actual.logits, expected, atol=2e-6, rtol=2e-5)
    if any(router is not None for router in routers):
        routers = [router for router in routers if router is not None]
    assert len(actual.router_logits) == len(routers)
    for actual_router, expected_router in zip(actual.router_logits, routers):
        if expected_router is None:
            assert actual_router is None
        else:
            torch.testing.assert_close(actual_router, expected_router, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("window", [None, 3])
def test_cached_continuation_generation_padding_and_causality(implementation, kv, window):
    torch.manual_seed(19)
    model = KlearMoeForCausalLM(tiny_config(
        attn_implementation=implementation, num_key_value_heads=kv,
        use_sliding_window=window is not None, sliding_window=window,
    )).eval()
    ids = torch.tensor([[1, 3, 7, 8, 11, 6], [1, 4, 5, 9, 10, 0]])
    mask = ids.ne(0)
    with torch.inference_mode():
        full = model(ids, attention_mask=mask, use_cache=False).logits
        prefix = model(ids[:, :4], attention_mask=mask[:, :4], use_cache=True)
        cache = prefix.past_key_values
        assert cache.get_seq_length() == 4
        for index in range(4, 6):
            continued = model(ids[:, index:index + 1], attention_mask=mask[:, :index + 1], past_key_values=cache)
            torch.testing.assert_close(continued.logits, full[:, index:index + 1], atol=2e-6, rtol=2e-5)
        assert cache.get_seq_length() == 6
        modified = ids.clone()
        modified[:, -2:] = torch.tensor([22, 23])
        altered = model(modified, attention_mask=mask, use_cache=False).logits
        torch.testing.assert_close(full[:, :-2], altered[:, :-2], atol=2e-6, rtol=2e-5)
        prompt = ids[:1, :3]
        cached = model.generate(prompt, max_new_tokens=3, do_sample=False, use_cache=True)
        uncached = model.generate(prompt, max_new_tokens=3, do_sample=False, use_cache=False)
        assert torch.equal(cached, uncached)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tied", [False, True])
def test_safetensors_roundtrip_outputs_and_direct_state_replacement(tmp_path, dtype, tied):
    torch.manual_seed(13)
    config = tiny_config(tie_word_embeddings=tied)
    model = KlearQModel.loader.from_config(config, dtype=dtype).eval()
    ids = torch.tensor([[1, 3, 8, 9]])
    with torch.inference_mode():
        reference = model(ids, output_attentions=True, output_hidden_states=True, output_router_logits=True)
        assert len(reference.attentions) == config.num_hidden_layers
        assert len(reference.hidden_states) == config.num_hidden_layers + 1
        torch.testing.assert_close(reference.hidden_states[0], model.get_input_embeddings()(ids), rtol=0, atol=0)
        torch.testing.assert_close(model.lm_head(reference.hidden_states[-1]), reference.logits, rtol=0, atol=0)
        assert reference.router_logits[0].dtype == torch.float32
        assert reference.logits.dtype == dtype
        assert torch.isfinite(reference.logits).all()
    assert model.lm_head.weight is model.model.embed_tokens.weight if tied else model.lm_head.weight is not model.model.embed_tokens.weight
    for attempt in range(2):
        path = tmp_path / str(attempt)
        model.save_pretrained(path, safe_serialization=True)
        assert (path / "model.safetensors").is_file()
        assert not list(path.glob("*.bin")) and not list(path.glob("*.pt"))
        restored = KlearQModel.loader.from_pretrained(path, dtype=dtype, attn_implementation="eager").eval()
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(reference.logits, restored(ids).logits, rtol=0, atol=0)
        model = restored
    for assign in (False, True):
        destination = KlearQModel.loader.from_config(config, dtype=dtype).eval()
        with torch.inference_mode():
            destination(ids)  # Replace weights in an already used model, not only a shell.
        destination.load_state_dict(model.state_dict(), assign=assign)
        with torch.inference_mode():
            torch.testing.assert_close(reference.logits, destination(ids).logits, rtol=0, atol=0)


def test_invalid_inputs_and_failed_load_do_not_poison_retry(tmp_path, monkeypatch):
    model, marker = poison_checkpoint(tmp_path)
    loader = KlearQModel.loader
    monkeypatch.setenv(BYPASS_ENV, "1")
    tensor_file = tmp_path / "model.safetensors"
    held = tmp_path / "held.safetensors"
    tensor_file.rename(held)
    with pytest.raises(OSError, match="safetensors"):
        loader.from_pretrained(tmp_path, trust_remote_code=True)
    held.rename(tensor_file)
    monkeypatch.delenv(BYPASS_ENV)
    restored = loader.from_pretrained(tmp_path, trust_remote_code=False, attn_implementation="eager").eval()
    ids = torch.tensor([[1, 5, 7]])
    with torch.inference_mode():
        cache = restored(ids[:, :2], use_cache=True).past_key_values
        with pytest.raises(ValueError, match="exactly one"):
            restored(input_ids=ids, inputs_embeds=restored.get_input_embeddings()(ids), past_key_values=cache)
        assert cache.get_seq_length() == 2
        continued = restored(ids[:, 2:], past_key_values=cache).logits
        torch.testing.assert_close(continued, model(ids, use_cache=False).logits[:, 2:], atol=2e-6, rtol=2e-5)
        with pytest.raises(TypeError, match="Cache"):
            restored(ids, past_key_values=((torch.zeros(1), torch.zeros(1)),))
    assert not marker.exists()


@pytest.mark.parametrize("model_type", ["Klear", "klear", "KLEAR"])
@pytest.mark.parametrize("trust,bypass", [(False, False), (True, True)])
def test_public_load_ignores_checkpoint_python(tmp_path, monkeypatch, model_type, trust, bypass):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.quantization.config import AWQConfig
    from tests.test_tiny_moe_quant_smoke import _build_local_tokenizer

    reference, marker = poison_checkpoint(tmp_path, model_type)
    _build_local_tokenizer(tmp_path)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    loaded = GPTQModel.load(
        str(tmp_path), trust_remote_code=trust, dtype=torch.float16, backend=BACKEND.TORCH,
        quantize_config=AWQConfig(bits=4, group_size=8, device="cpu", offload_to_disk=False),
        attn_implementation="eager",
    )
    assert type(loaded) is KlearQModel
    assert type(loaded.model) is KlearMoeForCausalLM
    assert type(loaded.model.config) is KlearConfig
    assert not marker.exists()
    with torch.inference_mode():
        ids = torch.tensor([[1, 5, 7]])
        torch.testing.assert_close(loaded.model(ids).logits.float(), reference(ids).logits, atol=5e-4, rtol=5e-3)


@pytest.mark.parametrize("options", [
    {"norm_topk_prob": False}, {"n_shared_experts": 0}, {"head_dim": 6},
    {"rope_scaling": {"type": "linear", "factor": 2.0}},
    {"use_sliding_window": True, "sliding_window": 3},
])
def test_additional_checkpoint_config_variants_match_equations(options):
    torch.manual_seed(15)
    model = KlearMoeForCausalLM(tiny_config(**options)).eval()
    ids = torch.tensor([[1, 5, 6, 8, 9]])
    with torch.inference_mode():
        torch.testing.assert_close(model(ids).logits, _oracle(model, ids)[0], atol=2e-6, rtol=2e-5)


def test_left_padding_loss_tuple_and_logits_selection():
    model = KlearMoeForCausalLM(tiny_config()).eval()
    padded = torch.tensor([[0, 0, 1, 6, 7], [1, 4, 8, 9, 6]])
    mask = padded.ne(0)
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(~mask, 0)
    with torch.inference_mode():
        outputs = model(padded, attention_mask=mask, position_ids=positions, labels=padded, use_cache=False)
        single = model(padded[:1, 2:], use_cache=False).logits
        torch.testing.assert_close(outputs.logits[:1, 2:], single, rtol=2e-5, atol=2e-6)
        assert torch.isfinite(outputs.loss)
        assert outputs.aux_loss is None  # Upstream exposes router logits but defines no auxiliary loss.
        selected = model(padded, attention_mask=mask, position_ids=positions, logits_to_keep=1, use_cache=False)
        torch.testing.assert_close(selected.logits, outputs.logits[:, -1:], rtol=2e-5, atol=2e-6)
        as_tuple = model(padded, attention_mask=mask, position_ids=positions, return_dict=False, use_cache=False)
        torch.testing.assert_close(as_tuple[0], outputs.logits, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["moe", "dense", "mixed"])
def test_public_awq_scaled_quantize_pack_save_reload(tmp_path, layout):
    from safetensors.torch import load_file, save_file

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig
    from tests.test_tiny_moe_quant_smoke import _build_calibration_dataset, _build_local_tokenizer

    variants = {"moe": {}, "dense": {"num_experts": 0}, "mixed": {"mlp_only_layers": [0]}}
    config = tiny_config(
        vocab_size=128, hidden_size=32, intermediate_size=32, moe_intermediate_size=32,
        num_hidden_layers=2 if layout == "mixed" else 1, num_experts_per_tok=3,
        max_position_embeddings=256, **variants[layout],
    )
    source = KlearQModel.loader.from_config(config, dtype=torch.float16).eval()
    with torch.no_grad():
        for layer in source.model.layers:
            if hasattr(layer.mlp, "expert_bias"):
                layer.mlp.expert_bias.copy_(torch.tensor([0.00391319, -0.002332191, 0.001397211]))
    path = tmp_path / "native"
    source.save_pretrained(path, safe_serialization=True)
    tokenizer = _build_local_tokenizer(path)
    calibration = _build_calibration_dataset(tokenizer)
    qcfg = AWQConfig(bits=4, group_size=16, device="cpu", offload_to_disk=False, fallback=None)
    model = GPTQModel.load(str(path), quantize_config=qcfg, backend=BACKEND.TORCH,
                          dtype=torch.float16, attn_implementation="eager", trust_remote_code=False)
    # AWQ folds its scales into the two decoder norms and compensates the
    # router/mixer inputs; every other dense tensor must stay bit-identical.
    unchanged = {name: value.clone() for name, value in model.model.state_dict().items()
                 if not name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                                       "gate_proj.weight", "up_proj.weight", "down_proj.weight",
                                       "input_layernorm.weight", "post_attention_layernorm.weight",
                                       "mlp.gate.weight", "mlp.coefficient.weight"))}
    model.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    searched = {(int(row["layer"]), str(row["module"])) for row in model.quant_log if row["loss"] != "not applicable"}
    for index, layer in enumerate(source.model.layers):
        first = "mlp.experts.0.gate_proj" if hasattr(layer.mlp, "experts") else "mlp.gate_proj"
        assert {(index, "self_attn.q_proj"), (index, first)} <= searched, sorted(searched)
    packed = {name: module for name, module in model.model.named_modules() if isinstance(module, AwqTorchLinear)}
    expected = sum(4 + (3 * config.num_experts + 3 if hasattr(layer.mlp, "experts") else 3)
                   for layer in source.model.layers)
    assert len(packed) == expected
    assert not model.quantize_config.dynamic  # Every expert was observed; no hidden pruning.
    for name, value in unchanged.items():
        torch.testing.assert_close(model.model.state_dict()[name], value, rtol=0, atol=0)
    assert all(module.qweight.dtype == torch.int32 for module in packed.values())
    ids = calibration[0]["input_ids"]
    with torch.inference_mode():
        before = model.model(ids, use_cache=False).logits
        assert torch.isfinite(before).all()
    for attempt in range(2):
        destination = tmp_path / f"awq-{attempt}"
        model.save(destination)
        assert list(destination.glob("*.safetensors"))
        assert not list(destination.glob("*.bin")) and not list(destination.glob("*.pt"))
        if attempt == 0 and layout != "dense":
            payload = next(destination.glob("*.safetensors"))
            stored = load_file(payload)
            missing = next(name for name in stored if name.endswith("expert_bias"))
            save_file({name: value for name, value in stored.items() if name != missing}, payload,
                      metadata={"format": "pt"})
            default_dtype = torch.get_default_dtype()
            with pytest.raises(RuntimeError, match="missing from checkpoint.*expert_bias"):
                GPTQModel.load(str(destination), backend=BACKEND.TORCH, device="cpu",
                               dtype=torch.float16, attn_implementation="eager", trust_remote_code=False)
            assert torch.get_default_dtype() == default_dtype
            assert not list(destination.glob("gptqmodel-key-remap-*"))
            save_file(stored, payload, metadata={"format": "pt"})
        restored = GPTQModel.load(str(destination), backend=BACKEND.TORCH, device="cpu",
                                 dtype=torch.float16, attn_implementation="eager", trust_remote_code=False)
        for name, value in model.model.state_dict().items():
            torch.testing.assert_close(restored.model.state_dict()[name], value, rtol=0, atol=0, msg=name)
            if name.endswith("expert_bias"):
                assert value.dtype == torch.float32
        with torch.inference_mode():
            torch.testing.assert_close(restored.model(ids, use_cache=False).logits, before, rtol=0, atol=0)
        model = restored
