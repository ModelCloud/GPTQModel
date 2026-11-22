# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Exact native Nemotron-H binding, not checkpoint Python or a look-alike."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from gptqmodel import CalibrationConfig
from gptqmodel.models.definitions.nemotron_h import NemotronHQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.nemotron_h import NemotronHConfig, NemotronHForCausalLM


def config(**overrides):
    values = {
        "vocab_size": 64, "hidden_size": 32, "layers_block_type": ["mamba", "attention", "mlp", "moe"],
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "intermediate_size": 64,
        "mamba_num_heads": 4, "mamba_head_dim": 8, "ssm_state_size": 4, "n_groups": 2,
        "conv_kernel": 3, "chunk_size": 4, "n_routed_experts": 4, "num_experts_per_tok": 2,
        "moe_intermediate_size": 32, "moe_shared_expert_intermediate_size": 64,
        "attn_implementation": "eager", "num_logits_to_keep": 0,
    }
    values.update(overrides)
    return NemotronHConfig(**values)


def native(**overrides):
    torch.manual_seed(2301)
    return NemotronHForCausalLM(config(**overrides)).eval()


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def wrapper(model):
    return NemotronHQModel(model=model, quantized=False, quantize_config=None)


def checkpoint(path, bias_scale=1, mtp=False, **overrides):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    model = native(**overrides)
    # Nonzero correction values expose low-precision buffer narrowing.
    for module in model.modules():
        if hasattr(module, "e_score_correction_bias"):
            module.e_score_correction_bias.copy_(bias_scale * torch.tensor([0.100123, -0.200567, 0.300789, -0.400123]))
    _convert_model_with_defuser(NemotronHQModel, model, cleanup_original=False)
    state = {name.replace("model.", "backbone.", 1) if name.startswith("model.") else name: value.contiguous()
             for name, value in model.state_dict().items()}
    if mtp:
        state["mtp.layers.0.norm.weight"] = torch.linspace(0.01, 0.9, model.config.hidden_size)
        state["mtp.layers.0.eh_proj.weight"] = torch.arange(2 * model.config.hidden_size ** 2).reshape(model.config.hidden_size, -1).float()
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(path)
    save_file(state, path / "model.safetensors", metadata={"format": "pt"})
    tokenizer = Tokenizer(models.WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0", pad_token="t0",
                           bos_token="t1", eos_token="t2").save_pretrained(path)
    marker = path / "EXECUTED"
    (path / "hostile.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python executed')\n"
    )
    payload = json.loads((path / "config.json").read_text())
    payload["auto_map"] = dict.fromkeys(("AutoConfig", "AutoModel", "AutoModelForCausalLM"), "hostile.Forbidden")
    (path / "config.json").write_text(json.dumps(payload))
    return model, marker


def test_native_binding_is_definition_owned_and_not_inherited_by_other_architectures():
    from gptqmodel.models.definitions.nemotron_h_puzzle import NemotronHPuzzleQModel
    from gptqmodel.models.definitions.nemotron_omni import NemotronOmniQModel

    assert NemotronHForCausalLM.__module__ == "gptqmodel.models.ports.native.nemotron_h.modeling_nemotron_h"
    assert NemotronHForCausalLM.is_custom_code() is False
    assert NemotronHQModel.require_trust_remote_code is False
    assert NemotronHQModel.require_monkeypatch is False
    assert isinstance(NemotronHQModel.loader, LocalModelLoader)
    actual = NemotronHQModel.loader.from_config(config())
    assert type(actual) is NemotronHForCausalLM
    # Siblings now have their own qualified bindings; none may inherit the
    # native text-only loader descriptor or its architecture-specific policy.
    for sibling, package in ((NemotronHPuzzleQModel, "nemotron_h_puzzle"),
                             (NemotronOmniQModel, "nemotron_omni")):
        assert isinstance(sibling.__dict__["loader"], LocalModelLoader)
        assert sibling.loader is not NemotronHQModel.loader
        assert sibling.loader.model_class.startswith(f"gptqmodel.models.ports.{package}.")
        assert sibling.require_trust_remote_code is False
        assert sibling.require_monkeypatch is False


@pytest.mark.parametrize("latent", [None, 16])
def test_transformers_checkpoint_loads_bit_exactly_into_vendored_class(tmp_path, latent):
    # Transformers 5.23 is the numerical oracle; its own save layout must load unchanged.
    from transformers import NemotronHConfig as OracleConfig
    from transformers import NemotronHForCausalLM as OracleModel

    torch.manual_seed(2301)
    oracle_config = OracleConfig(**config(moe_latent_size=latent).to_dict(), attn_implementation="eager")
    oracle = OracleModel(oracle_config).eval()
    actual = native(moe_latent_size=latent)
    ids = torch.tensor([[1, 5, 8, 9, 12]])
    with torch.inference_mode():
        expected = oracle(ids).logits
        torch.testing.assert_close(actual(ids).logits, expected, atol=0, rtol=0)
        oracle.save_pretrained(tmp_path, safe_serialization=True)
        loaded = NemotronHQModel.loader.from_pretrained(tmp_path, attn_implementation="eager").eval()
        assert type(loaded) is NemotronHForCausalLM
        torch.testing.assert_close(loaded(ids).logits, expected, atol=0, rtol=0)
        assert oracle.state_dict().keys() == loaded.state_dict().keys()


def test_native_hook_never_overwrites_legacy_shaped_mixers():
    # The old hook assigned type(mixer).forward process-wide, even for a native
    # definition. Such a global mutation must not return with an optional kernel.
    def original(self, *args, **kwargs):
        return "unchanged"

    mixer_class = type("NemotronHMamba2Mixer", (), {
        "forward": original, "torch_forward": original, "cuda_kernels_forward": original,
    })
    wrapper = NemotronHQModel.__new__(NemotronHQModel)
    wrapper.__dict__["model"] = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mixer=mixer_class())]))
    wrapper.load_quantized_model = True
    wrapper.monkey_patch()
    assert mixer_class.forward is original


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_native_loading_ignores_malicious_config_and_model_maps(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel import GPTQModel
    from gptqmodel.utils.hf import load_model_config

    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    expected, marker = checkpoint(tmp_path)
    selected = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(selected) is NemotronHConfig
    actual = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32,
                           trust_remote_code=trust, attn_implementation="eager")
    assert type(actual) is NemotronHQModel
    assert type(actual.model) is NemotronHForCausalLM
    assert not marker.exists()
    _convert_model_with_defuser(NemotronHQModel, actual.model, cleanup_original=False)
    ids = torch.tensor([[1, 8, 3, 7, 11]])
    with torch.inference_mode():
        torch.testing.assert_close(actual.model(ids).logits, expected(ids).logits, atol=1e-6, rtol=1e-6)
    assert set(actual.model.state_dict()) == set(expected.state_dict())
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(actual.model.state_dict()[name], value, atol=0, rtol=0)


def source_oracle(model, ids):
    """Scalar recurrence + source MLP/router/attention equations; no native helpers.

    Reviewed original/Nano/Super modeling sources, revisions in the port NOTICE.
    No checkpoint Python is imported or executed.
    """
    c, w = model.config, {name: value.double() for name, value in model.state_dict().items()}
    hidden = F.embedding(ids, w["model.embeddings.weight"])

    def linear(x, prefix):
        return F.linear(x, w[prefix + ".weight"], w.get(prefix + ".bias"))

    def rms(x, weight):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + c.layer_norm_epsilon) * weight

    for index, kind in enumerate(c.layers_block_type):
        prefix = f"model.layers.{index}.mixer"
        x = rms(hidden, w[f"model.layers.{index}.norm.weight"])
        if kind == "full_attention":
            q, k, v = [linear(x, prefix + "." + name).reshape(ids.shape[0], ids.shape[1], -1, c.head_dim).transpose(1, 2)
                       for name in ("q_proj", "k_proj", "v_proj")]
            k = k.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, 1)
            v = v.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, 1)
            score = q @ k.transpose(-1, -2) / c.head_dim ** 0.5
            score.masked_fill_(torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool).triu(1), -torch.inf)
            y = linear((score.softmax(-1) @ v).transpose(1, 2).reshape_as(x), prefix + ".o_proj")
        elif kind == "mlp":
            y = linear(F.relu(linear(x, prefix + ".up_proj")).square(), prefix + ".down_proj")
        elif kind == "moe":
            flat = x.reshape(-1, c.hidden_size)
            score = F.linear(flat, w[prefix + ".gate.weight"]).sigmoid()
            choice = score + w[prefix + ".gate.e_score_correction_bias"]
            groups = choice.reshape(-1, c.n_group, c.n_routed_experts // c.n_group)
            selected = groups.topk(2, dim=-1).values.sum(-1).topk(c.topk_group, dim=-1).indices
            mask = torch.zeros_like(groups, dtype=torch.bool)
            mask.scatter_(1, selected[..., None].expand(-1, -1, groups.shape[-1]), True)
            experts = choice.masked_fill(~mask.reshape_as(choice), -torch.inf).topk(c.num_experts_per_tok, dim=-1).indices
            gains = score.gather(1, experts)
            if c.norm_topk_prob:
                gains = gains / gains.sum(-1, keepdim=True)
            gains *= c.routed_scaling_factor
            latent = linear(flat, prefix + ".fc1_latent_proj") if c.moe_latent_size else flat
            routed = torch.zeros_like(latent)
            for row in range(flat.shape[0]):
                for slot, expert in enumerate(experts[row].tolist()):
                    root = prefix + f".experts.{expert}"
                    value = linear(F.relu(linear(latent[row], root + ".up_proj")).square(), root + ".down_proj")
                    routed[row] += gains[row, slot] * value
            if c.moe_latent_size:
                routed = linear(routed, prefix + ".fc2_latent_proj")
            shared = linear(F.relu(linear(flat, prefix + ".shared_experts.up_proj")).square(),
                            prefix + ".shared_experts.down_proj")
            y = (routed + shared).reshape_as(x)
        else:
            heads, dim, groups, state_size = c.mamba_num_heads, c.mamba_head_dim, c.n_groups, c.ssm_state_size
            width = heads * dim
            z, conv, dt = linear(x, prefix + ".in_proj").split([width, width + 2 * groups * state_size, heads], -1)
            conv = F.conv1d(F.pad(conv.transpose(1, 2), (c.conv_kernel - 1, 0)),
                            w[prefix + ".conv1d.weight"], w.get(prefix + ".conv1d.bias"),
                            groups=conv.shape[-1]).transpose(1, 2)
            u, b, cc = F.silu(conv).split([width, groups * state_size, groups * state_size], -1)
            u = u.reshape(ids.shape[0], -1, heads, dim)
            b = b.reshape(ids.shape[0], -1, groups, state_size).repeat_interleave(heads // groups, 2)
            cc = cc.reshape(ids.shape[0], -1, groups, state_size).repeat_interleave(heads // groups, 2)
            delta = F.softplus(dt + w[prefix + ".dt_bias"]).clamp(min=c.time_step_min)
            state = torch.zeros(ids.shape[0], heads, dim, state_size, dtype=torch.float64)
            values = []
            for step in range(ids.shape[1]):
                d = delta[:, step, :, None, None]
                state = state * torch.exp(-w[prefix + ".A_log"].exp()[None, :, None, None] * d)
                state = state + d * u[:, step, :, :, None] * b[:, step, :, None, :]
                values.append((state * cc[:, step, :, None, :]).sum(-1) + w[prefix + ".D"][None, :, None] * u[:, step])
            y = torch.stack(values, 1).reshape_as(z) * F.silu(z)
            y = y.reshape(*y.shape[:-1], groups, width // groups)
            y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + c.layer_norm_epsilon)
            y = linear(y.reshape_as(z) * w[prefix + ".norm.weight"], prefix + ".out_proj")
        hidden = hidden + y
    return linear(rms(hidden, w["model.norm_f.weight"]), "lm_head")


@pytest.mark.parametrize("kv_heads", [1, 4])
@pytest.mark.parametrize("latent", [None, 16])
@pytest.mark.parametrize("bias", [False, True])
def test_full_hybrid_matches_independent_source_equations(kv_heads, latent, bias):
    model = native(num_key_value_heads=kv_heads, moe_latent_size=latent,
                   mlp_bias=bias, use_bias=bias, n_group=2, topk_group=1, routed_scaling_factor=1.7)
    _convert_model_with_defuser(NemotronHQModel, model, cleanup_original=False)
    ids = torch.tensor([[1, 8, 9, 3, 11, 5], [4, 3, 2, 7, 9, 1]])
    with torch.no_grad():
        expected = source_oracle(model, ids)
        actual = model(ids, use_cache=False).logits
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("pattern", [
    ["mamba", "attention", "mlp", "moe"],
    ["attention", "moe", "mamba", "mamba"],
    ["mlp", "moe"],
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_native_dtype_cache_one_and_multi_token_continuation_and_generation(pattern, dtype, attention):
    model = native(layers_block_type=pattern, attn_implementation=attention, moe_latent_size=16).to(dtype)
    ids = torch.tensor([[1, 4, 9, 3, 7, 6, 8], [1, 11, 5, 10, 12, 3, 4]])
    tolerance = {torch.float32: 3e-6, torch.float16: 0.001, torch.bfloat16: 0.006}[dtype]
    with torch.inference_mode():
        expected = model(ids, use_cache=False).logits
        if "attention" not in pattern:
            with pytest.raises(ValueError, match="get_seq_length.*Attention"):
                model(ids[:, :2], use_cache=True)
            generated = model.generate(ids[:, :3], use_cache=False, do_sample=False, max_new_tokens=3, eos_token_id=None)
            assert generated.shape == (2, 6)
            return
        past = None
        start = 0
        for length in [2, 1, 3, 1]:
            out = model(ids[:, start:start+length], past_key_values=past, use_cache=True)
            torch.testing.assert_close(out.logits, expected[:, start:start+length], atol=tolerance, rtol=tolerance)
            past = out.past_key_values
            start += length
        # A separate request must not reuse a previous request's SSM state.
        torch.testing.assert_close(model(ids, use_cache=False).logits, expected, atol=0, rtol=0)
        generated = model.generate(ids[:, :3], do_sample=False, max_new_tokens=3, eos_token_id=None)
        greedy = ids[:, :3]
        for _ in range(3):
            token = model(greedy, use_cache=False).logits[:, -1:].argmax(-1)
            greedy = torch.cat((greedy, token), -1)
        assert torch.equal(generated, greedy)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("latent", [None, 16])
def test_native_safetensors_two_roundtrips_assignment_and_failed_load_retry(tmp_path, dtype, latent):
    source = native(moe_latent_size=latent).to(dtype)
    for module in source.modules():
        if hasattr(module, "e_score_correction_bias"):
            module.e_score_correction_bias = module.e_score_correction_bias.float()
    ids = torch.tensor([[1, 5, 8, 9, 12]])
    expected_state = {key: value.clone() for key, value in source.state_dict().items()}
    with torch.inference_mode():
        expected = source(ids).logits
        for cycle in range(2):
            path = tmp_path / str(cycle)
            source.save_pretrained(path, safe_serialization=True)
            payload = path / "model.safetensors"
            original = payload.read_bytes()
            payload.write_bytes(b"incomplete safetensors")
            with pytest.raises(Exception, match="header|Header|safetensor"):
                NemotronHQModel.loader.from_pretrained(path, dtype=dtype)
            payload.write_bytes(original)
            source = NemotronHQModel.loader.from_pretrained(path, dtype=dtype, attn_implementation="eager").eval()
            torch.testing.assert_close(source(ids).logits, expected, atol=0, rtol=0)
            assert not list(path.glob("*.bin"))
            for key, value in source.state_dict().items():
                torch.testing.assert_close(value, expected_state[key], atol=0, rtol=0)
        for assign in [False, True]:
            fresh = native(moe_latent_size=latent).to(dtype)
            fresh.load_state_dict(source.state_dict(), assign=assign)
            fresh.load_state_dict(source.state_dict(), assign=assign)
            torch.testing.assert_close(fresh(ids).logits, expected, atol=0, rtol=0)


def test_real_mamba_padding_and_future_tokens_cannot_affect_valid_prefix():
    model = native()
    ids = torch.tensor([[0, 0, 1, 3, 9, 7], [1, 4, 8, 11, 9, 5]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    with torch.inference_mode():
        padded = model(ids, attention_mask=mask).logits
        unpadded = model(ids[:1, 2:], attention_mask=mask[:1, 2:]).logits
        torch.testing.assert_close(padded[:1, 2:], unpadded, atol=2e-6, rtol=2e-6)
        modified = ids.clone()
        modified[:, -2:] = torch.tensor([[8, 6], [6, 2]])
        actual = model(modified, attention_mask=mask).logits[:, :-2]
        torch.testing.assert_close(actual[mask[:, :-2].bool()], padded[:, :-2][mask[:, :-2].bool()], atol=5e-8, rtol=2e-6)


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("layout", ["hybrid", "dense", "moe", "latent"])
def test_public_awq_packs_every_decoder_linear_and_reload_is_exact(tmp_path, offload, layout):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization import AWQConfig

    patterns = {"hybrid": ["mamba", "attention", "mlp", "moe"], "dense": ["mamba", "attention", "mlp"],
                "moe": ["mamba", "attention", "moe"], "latent": ["mamba", "attention", "moe"]}
    source, _ = checkpoint(tmp_path / "source", layers_block_type=patterns[layout],
                           moe_latent_size=16 if layout == "latent" else None,
                           mamba_num_heads=8, mamba_head_dim=4, bias_scale=0.001, mtp=layout == "latent")
    expected_linears = {name for name, module in source.named_modules()
                        if isinstance(module, torch.nn.Linear) and name.startswith("model.layers.")
                        and not name.endswith(".gate")}
    # AWQ folds a dense mixer's block norm into its consumers, so those norms
    # are intentionally no longer bit-identical to the source checkpoint.
    # Every other non-quantized tensor must still be copied verbatim.
    dense_layers = {
        f"model.layers.{index}" for index, layer in enumerate(source.model.layers)
        if layer.block_type != "moe"
    }
    fold_sources = {
        name for name in source.state_dict()
        if name.endswith(".norm.weight") and name.rsplit(".", 2)[0] in dense_layers
    }
    dense = {name: value.clone() if "e_score_correction_bias" in name else value.half() for name, value in source.state_dict().items()
             if not any(name.startswith(root + ".") for root in expected_linears)
             and name not in fold_sources}
    qcfg = AWQConfig(bits=4, group_size=16, device="cpu", fallback=None, offload_to_disk=offload,
                     offload_to_disk_path=str(tmp_path / "offload"))
    loaded = GPTQModel.load(str(tmp_path / "source"), quantize_config=qcfg, backend=BACKEND.AWQ_TORCH,
                           dtype=torch.float16, attn_implementation="eager")
    groups = []
    original = loaded.awq_get_modules_for_scaling

    def inspect(*args, **kwargs):
        found = original(*args, **kwargs)
        groups.append(found)
        return found

    loaded.awq_get_modules_for_scaling = inspect
    samples = [torch.arange(start, start + 16) for start in range(0, 64, 16)]
    loaded.quantize(
        CalibrationConfig(dataset=samples, batch_size=1, min_length=1),
        backend=BACKEND.AWQ_TORCH,
    )
    assert groups
    # Dense mixers (mamba/attention/mlp) now search AWQ scales; only the MoE
    # mixer is fold-free because a single affine fold cannot compensate both
    # the shared and routed expert branches.
    assert any(group for group in groups)
    if "moe" in patterns[layout]:
        assert any(group == [] for group in groups)
    packed = {name for name, module in loaded.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert packed == expected_linears
    state = {name: value.clone() for name, value in loaded.model.state_dict().items()}
    for name, value in dense.items():
        torch.testing.assert_close(state[name], value, atol=0, rtol=0)
    for name in fold_sources:
        assert state[name].dtype == torch.float16, name
        assert torch.isfinite(state[name]).all(), name
    ids = torch.tensor([[1, 3, 8, 11, 9]])
    with torch.inference_mode():
        expected = loaded.model(ids).logits
        assert torch.isfinite(expected).all()
        for cycle in range(2):
            path = tmp_path / f"packed-{cycle}"
            loaded.save(str(path))
            assert list(path.glob("*.safetensors")) and not list(path.glob("*.bin"))
            if layout == "latent":
                from safetensors import safe_open

                with safe_open(tmp_path / "source/model.safetensors", framework="pt") as original:
                    extras = {key: original.get_tensor(key) for key in original.keys() if key.startswith("mtp.")}
                seen = set()
                for payload in path.glob("*.safetensors"):
                    with safe_open(payload, framework="pt") as saved:
                        for key in extras.keys() & set(saved.keys()):
                            torch.testing.assert_close(saved.get_tensor(key), extras[key], atol=0, rtol=0)
                            seen.add(key)
                assert seen == set(extras)
            loaded = GPTQModel.load(str(path), backend=BACKEND.AWQ_TORCH, device="cpu", dtype=torch.float16,
                                    attn_implementation="eager")
            assert set(loaded.model.state_dict()) == set(state)
            for name, value in loaded.model.state_dict().items():
                torch.testing.assert_close(value, state[name], atol=0, rtol=0)
            torch.testing.assert_close(loaded.model(ids).logits, expected, atol=0, rtol=0)
