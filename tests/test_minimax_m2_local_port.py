# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native M2 and source-matched original MiniMax must remain separate."""

import hashlib
import json
import re
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from awq_neutrality import assert_awq_folds_neutral
from transformers import PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.native.auto import AutoConfig, AutoModelForCausalLM
from gptqmodel.quantization.config import AWQConfig
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code
from gptqmodel.utils.security import BYPASS_ENV
from tests.port_assets import read_port_asset, read_port_json


ASSETS = Path(__file__).parents[1] / "gptqmodel/models/ports/minimax_m2"
FIXTURES = Path(__file__).parent / "fixtures/ports/minimax_m2"
# Released safetensors indexes: MiniMaxAI/MiniMax-Text-01-hf@f7ce0136, MiniMaxAI/MiniMax-M2@04312f0c.
INVENTORY_PINS = {
    "minimax": (8243, "b48fad7c4c310a42d1222f1726c698278bbfbc31a52ed3774b8ed4405c355397"),
    "minimax_m2": (96103, "27627afa3072e685a7a4ea3f23f2d61658934498d693a6fac2831ed7646cd33d"),
}
IDS = torch.tensor([[1, 5, 7, 9, 4, 3, 6], [1, 4, 8, 3, 7, 5, 9]])


@pytest.fixture(autouse=True)
def bounded_torch_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tiny_config(model_type, kv=2, layout="mixed", rotary_dim=4, shared=False, postnorm=True):
    values = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": kv,
        "head_dim": 8,
        "num_local_experts": 3,
        "num_experts_per_tok": 2,
        "max_position_embeddings": 64,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "tie_word_embeddings": False,
        "rotary_dim": rotary_dim,
        "block_size": 4,
        "attention_dropout": 0.0,
    }
    if model_type == "minimax":
        values["layer_types"] = {
            "mixed": ["linear_attention", "full_attention"],
            "full": ["full_attention"] * 2,
            "linear": ["linear_attention"] * 2,
        }[layout]
        values.update(
            full_attn_alpha_factor=1.3,
            full_attn_beta_factor=0.7,
            linear_attn_alpha_factor=1.2,
            linear_attn_beta_factor=0.8,
            mlp_alpha_factor=1.1,
            mlp_beta_factor=0.6,
            postnorm=postnorm,
            shared_intermediate_size=32 if shared else 0,
        )
    config = AutoConfig.for_model(model_type, **values)
    config._attn_implementation = "eager"
    return config


def tiny_model(model_type, **kwargs):
    torch.manual_seed(431)
    return MODEL_MAP[model_type].loader.from_config(tiny_config(model_type, **kwargs)).eval()


def bare_wrapper(model_type, model):
    definition = MODEL_MAP[model_type]
    obj = definition.__new__(definition)
    torch.nn.Module.__init__(obj)
    obj.__dict__["model"] = model
    return obj


def source_state(model):
    state = {}
    for key, value in model.state_dict().items():
        key = key.replace(".mlp.", ".block_sparse_moe.")
        if key.endswith(".experts.gate_up_proj"):
            prefix = key.removesuffix("gate_up_proj")
            for expert, packed in enumerate(value):
                gate, up = packed.chunk(2, 0)
                state[f"{prefix}{expert}.w1.weight"] = gate.contiguous()
                state[f"{prefix}{expert}.w3.weight"] = up.contiguous()
        elif key.endswith(".experts.down_proj"):
            prefix = key.removesuffix("down_proj")
            for expert, weight in enumerate(value):
                state[f"{prefix}{expert}.w2.weight"] = weight.contiguous()
        else:
            state[key] = value.contiguous()
    return state


def rms(x, weight, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.double()


def full_attention_oracle(attn, x, config, *, m2=False, rotary_dim=None):
    """Independent float64 checkpoint equations, not native attention helpers."""
    b, s, _ = x.shape
    h, kv, d = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    q = F.linear(x, attn.q_proj.weight.double())
    k = F.linear(x, attn.k_proj.weight.double())
    v = F.linear(x, attn.v_proj.weight.double()).view(b, s, kv, d).transpose(1, 2)
    if m2:
        q = rms(q, attn.q_norm.weight, config.rms_norm_eps)
        k = rms(k, attn.k_norm.weight, config.rms_norm_eps)
    q = q.view(b, s, h, d).transpose(1, 2)
    k = k.view(b, s, kv, d).transpose(1, 2)
    r = rotary_dim or getattr(config, "rotary_dim", d)
    theta = config.rope_parameters["rope_theta"]
    angles = torch.arange(s).double()[:, None] / theta ** (torch.arange(0, r, 2).double() / r)
    angles = torch.cat((angles, angles), dim=-1)[None, None]

    def rotate(t):
        a, z = t[..., :r], t[..., r:]
        rotated = torch.cat((-a[..., r // 2 :], a[..., : r // 2]), dim=-1)
        return torch.cat((a * angles.cos() + rotated * angles.sin(), z), dim=-1)

    q, k = rotate(q), rotate(k)
    k = k.repeat_interleave(h // kv, dim=1)
    v = v.repeat_interleave(h // kv, dim=1)
    scores = q @ k.transpose(-2, -1) / d**0.5
    scores.masked_fill_(torch.ones(s, s, dtype=torch.bool).triu(1), -torch.inf)
    out = scores.softmax(-1) @ v
    return F.linear(out.transpose(1, 2).reshape(b, s, h * d), attn.o_proj.weight.double())


def lightning_oracle(attn, x, config, idx):
    b, s, _ = x.shape
    h, d = config.num_attention_heads, config.head_dim
    q, k, v = F.silu(F.linear(x, attn.qkv_proj.weight.double())).reshape(b, s, h, 3 * d).chunk(3, -1)
    slope = (2 ** (-8.0 / h)) ** torch.arange(1, h + 1).double()
    slope *= 1 - idx / (config.num_hidden_layers - 1 + 1e-5) + 1e-5
    memory = torch.zeros(b, h, d, d, dtype=torch.float64)
    rows = []
    for pos in range(s):
        memory = memory * (-slope).exp()[None, :, None, None] + k[:, pos, :, :, None] * v[:, pos, :, None, :]
        rows.append(torch.einsum("bhd,bhdv->bhv", q[:, pos], memory))
    out = torch.stack(rows, dim=1).reshape(b, s, h * d)
    out = rms(out, attn.norm.weight, attn.norm.variance_epsilon)
    out *= F.linear(x, attn.output_gate.weight.double()).sigmoid()
    return F.linear(out, attn.out_proj.weight.double())


def moe_oracle(mlp, x, config, *, m2):
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    logits = F.linear(flat, mlp.gate.weight.double())
    scores = logits.sigmoid() if m2 else logits.softmax(-1)
    choices = scores + mlp.e_score_correction_bias.double() if m2 else scores
    indices = choices.topk(config.num_experts_per_tok, -1).indices
    weights = scores.gather(1, indices)
    weights = weights / weights.sum(-1, keepdim=True)
    result = torch.zeros_like(flat)
    for row in range(len(flat)):
        for slot, expert in enumerate(indices[row].tolist()):
            gate, up = mlp.experts.gate_up_proj[expert].double().chunk(2, 0)
            val = F.silu(F.linear(flat[row], gate)) * F.linear(flat[row], up)
            result[row] += F.linear(val, mlp.experts.down_proj[expert].double()) * weights[row, slot]
    return result.reshape(shape)


# shared/postnorm only exist on the original MiniMax layer.
@pytest.mark.parametrize(
    "model_type,kv,shared,postnorm",
    [("minimax", kv, shared, postnorm) for kv in (1, 2, 4) for shared in (False, True) for postnorm in (False, True)]
    + [("minimax_m2", kv, False, True) for kv in (1, 2, 4)],
)
def test_source_decoder_equations(model_type, kv, shared, postnorm):
    model = tiny_model(model_type, kv=kv, shared=shared, postnorm=postnorm)
    m2 = model_type == "minimax_m2"
    if m2:
        with torch.no_grad():
            for layer in model.model.layers:
                layer.mlp.e_score_correction_bias.copy_(torch.tensor([0.17, -0.11, 0.03]))
    config = model.config
    x = model.model.embed_tokens(IDS).double()
    with torch.inference_mode():
        for idx, layer in enumerate(model.model.layers):
            normed = rms(x, layer.input_layernorm.weight, config.rms_norm_eps)
            if m2 or config.layer_types[idx] == "full_attention":
                attn = full_attention_oracle(layer.self_attn, normed, config, m2=m2)
            else:
                attn = lightning_oracle(layer.self_attn, normed, config, idx)
            x = (
                x + attn
                if m2
                else (normed if postnorm else x) * layer.attn_alpha_factor + attn * layer.attn_beta_factor
            )
            normed = rms(x, layer.post_attention_layernorm.weight, config.rms_norm_eps)
            mlp = moe_oracle(layer.mlp, normed, config, m2=m2)
            if shared and not m2:
                shared_out = F.linear(
                    F.silu(F.linear(normed, layer.shared_mlp.gate_proj.weight.double()))
                    * F.linear(normed, layer.shared_mlp.up_proj.weight.double()),
                    layer.shared_mlp.down_proj.weight.double(),
                )
                coefficient = F.linear(normed, layer.coefficient.weight.double()).sigmoid()
                mlp = mlp * (1 - coefficient) + shared_out * coefficient
            x = x + mlp if m2 else (normed if postnorm else x) * layer.mlp_alpha_factor + mlp * layer.mlp_beta_factor
        expected = F.linear(rms(x, model.model.norm.weight, config.rms_norm_eps), model.lm_head.weight.double())
        actual = model(IDS, use_cache=False).logits
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=3e-7)


def test_original_partial_rope_changes_native_logits():
    model = tiny_model("minimax", layout="full")
    native = AutoModelForCausalLM.from_config(tiny_config("minimax", layout="full")).eval()
    native.load_state_dict(model.state_dict())
    with torch.inference_mode():
        actual = model(IDS, use_cache=False).logits
        wrong = native(IDS, use_cache=False).logits
    assert (actual - wrong).abs().max() > 1e-5, (
        "released partial RoPE must not silently use full-head native attention"
    )


@pytest.mark.parametrize("model_type", ["minimax", "minimax_m2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_native_cache_causality_generation_and_safetensors(tmp_path, model_type, dtype, attn):
    model = tiny_model(model_type).to(dtype)
    model.set_attn_implementation(attn)
    ids = IDS[:1]
    tolerance = {torch.float32: 3e-6, torch.float16: 8e-4, torch.bfloat16: 0.008}[dtype]
    with torch.inference_mode():
        whole = model(ids, use_cache=False).logits
        out = model(ids[:, :1], use_cache=True)
        pieces = [out.logits]
        for a, z in [(1, 5), (5, 7)]:
            out = model(ids[:, a:z], past_key_values=out.past_key_values, use_cache=True)
            pieces.append(out.logits)
        torch.testing.assert_close(torch.cat(pieces, 1), whole, rtol=tolerance, atol=tolerance)
        perturbed = ids.clone()
        perturbed[:, 5:] = 11
        torch.testing.assert_close(
            model(perturbed, use_cache=False).logits[:, :5], whole[:, :5], rtol=tolerance, atol=tolerance
        )
        generated = model.generate(ids, max_new_tokens=3, do_sample=False, eos_token_id=None)
        manual = ids.clone()
        for _ in range(3):
            manual = torch.cat((manual, model(manual, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)), -1)
        assert torch.equal(generated, manual)
    for cycle in range(2):
        path = tmp_path / str(cycle)
        model.save_pretrained(path, safe_serialization=True)
        assert list(path.glob("*.safetensors")) and not list(path.glob("*.bin"))
        config = load_model_config(path)
        restored = MODEL_MAP[model_type].loader.from_pretrained(path, config=config, dtype=dtype).eval()
        assert set(restored.state_dict()) == set(model.state_dict())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(restored(ids, use_cache=False).logits, whole, rtol=tolerance, atol=tolerance)
        model = restored


def local_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel({f"t{i}": i for i in range(64)}, unk_token="t3"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    result = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="t1", eos_token="t2", pad_token="t0", unk_token="t3"
    )
    result.save_pretrained(path)
    return result


def hostile_checkpoint(path, model_type, **kwargs):
    from safetensors.torch import save_file

    model = tiny_model(model_type, **kwargs)
    if model_type == "minimax_m2":
        for layer in model.model.layers:
            layer.mlp.e_score_correction_bias.copy_(torch.tensor([0.001234567, -0.01234567, 0.00345678]))
    model.save_pretrained(path)
    save_file(source_state(model), path / "model.safetensors", metadata={"format": "pt"})
    local_tokenizer(path)
    marker = path / "EXECUTED"
    (path / "hostile.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('remote executed')\n"
    )
    config_file = path / "config.json"
    payload = json.loads(config_file.read_text())
    payload["auto_map"] = {"AutoConfig": "hostile.Config", "AutoModelForCausalLM": "hostile.Model"}
    config_file.write_text(json.dumps(payload))
    return model, marker


@pytest.mark.parametrize("model_type", ["minimax", "minimax_m2"])
def test_entire_pinned_checkpoint_inventory_matches_real_meta_model(model_type):
    key_count, fixture_sha256 = INVENTORY_PINS[model_type]
    fixture = read_port_asset(ASSETS / f"{model_type}_checkpoint_keys.json")
    assert hashlib.sha256(fixture).hexdigest() == fixture_sha256
    raw = json.loads((FIXTURES / f"{model_type}_released_config.json").read_text())
    if model_type == "minimax":
        config = MODEL_MAP[model_type].loader.load_config(FIXTURES / f"{model_type}_released_config.json")
    else:
        config = AutoConfig.for_model(model_type, **{k: v for k, v in raw.items() if k != "model_type"})
    with torch.device("meta"):
        model = MODEL_MAP[model_type].loader.from_config(config)
    keys = read_port_json(ASSETS / f"{model_type}_checkpoint_keys.json")
    key_set = set(keys)
    assert len(keys) == key_count
    assert set(source_state(model)) == {key for key in keys if not key.endswith("weight_scale_inv")}
    from gptqmodel.models.module_plan import ModulePlan

    plan = ModulePlan.from_tree(MODEL_MAP[model_type].module_tree)
    for key in keys:
        if key.endswith("weight_scale_inv"):
            assert model_type == "minimax_m2" and key.removesuffix("_scale_inv") in key_set
            continue
        if not key.startswith("model.layers."):
            assert key in {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
            continue
        owner = key.split(".", 3)[-1].replace("block_sparse_moe", "mlp")
        owner = re.sub(r"\.w1\.weight$", ".gate_proj.weight", owner)
        owner = re.sub(r"\.w2\.weight$", ".down_proj.weight", owner)
        owner = re.sub(r"\.w3\.weight$", ".up_proj.weight", owner)
        owner = owner.removesuffix(".weight")
        spec = plan.find(owner)
        assert spec is not None, owner
        if owner.endswith(("norm", "layernorm", "e_score_correction_bias", ".gate")):
            assert not spec.quantize
        else:
            assert spec.quantize
    assert not any("decay" in key or "slope" in key for key in keys)


@pytest.mark.parametrize("layout", ["mixed", "full", "linear"])
def test_original_cache_layouts_beam_reset_and_foreign_cache(layout):
    from transformers import DynamicCache

    model = tiny_model("minimax", layout=layout)
    with torch.inference_mode():
        start = model(IDS[:, :3], use_cache=True)
        cache = start.past_key_values
        # Only softmax-attention KV layers carry a token count; recurrent-only caches report none.
        tracked = 0 if layout == "linear" else 1
        assert cache.get_seq_length() == 3 * tracked
        index = torch.tensor([1, 0, 1])
        cache.reorder_cache(index)
        actual = model(IDS[index, 3:6], past_key_values=cache, use_cache=True).logits
        expected = model(IDS[index, :6], use_cache=False).logits[:, 3:]
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-7)
        assert cache.get_seq_length() == 6 * tracked
        cache.reset()
        assert cache.get_seq_length() == 0 and not cache.linear_cache
        retry = model(IDS[:1], past_key_values=cache, use_cache=True).logits
        torch.testing.assert_close(retry, model(IDS[:1], use_cache=False).logits, rtol=2e-5, atol=3e-7)
        generated = model.generate(IDS[:1, :3], num_beams=2, max_new_tokens=3)
        expected = model.generate(IDS[:1, :3], num_beams=2, max_new_tokens=3, use_cache=False)
        assert torch.equal(generated, expected)
    with pytest.raises(ValueError, match="foreign or M2"):
        model(IDS, past_key_values=DynamicCache(), use_cache=True)


def test_released_rotary_width_64_of_128_has_independent_source_equations():
    config = tiny_config("minimax", layout="full")
    config.head_dim = 128
    config.rotary_dim = 64
    model = MODEL_MAP["minimax"].loader.from_config(config).eval()
    attn = model.model.layers[0].self_attn
    torch.manual_seed(37)
    x = torch.randn(2, 7, 32) * 2
    with torch.inference_mode():
        rope = model.model.rotary_emb(x, torch.arange(7)[None])
        mask = torch.zeros(7, 7).masked_fill(torch.ones(7, 7, dtype=torch.bool).triu(1), -torch.inf)
        actual = attn(x, rope, attention_mask=mask)[0]
        expected = full_attention_oracle(attn, x.double(), model.config, rotary_dim=64)
        wrong = full_attention_oracle(attn, x.double(), model.config, rotary_dim=128)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=3e-7)
    assert (actual.double() - wrong).abs().max() > 1e-4


@pytest.mark.parametrize("width", [0, 3, 10, -2])
def test_original_invalid_rotary_metadata_fails_then_valid_config_retries(width):
    from gptqmodel.models.ports.minimax_m2.configuration_minimax import MiniMaxText01Config

    with pytest.raises(ValueError, match="rotary_dim"):
        MiniMaxText01Config(head_dim=8, rotary_dim=width)
    assert MiniMaxText01Config(head_dim=8, rotary_dim=4).rotary_dim == 4


@pytest.mark.parametrize("model_type", ["minimax", "minimax_m2"])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "1"])
def test_public_hostile_selection_and_retry(tmp_path, monkeypatch, model_type, trust, bypass):
    root = tmp_path / "root"
    path = root / "nested"
    expected, marker = hostile_checkpoint(path, model_type)
    if bypass is None:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    else:
        monkeypatch.setenv(BYPASS_ENV, bypass)
    assert resolve_trust_remote_code(root, subfolder="nested", trust_remote_code=trust) is False
    config = load_model_config(root, subfolder="nested", trust_remote_code=trust)
    assert config.model_type == model_type
    wrapper = GPTQModel.load(
        str(root),
        subfolder="nested",
        device="cpu",
        dtype=torch.float32,
        backend=BACKEND.TORCH,
        trust_remote_code=trust,
    )
    assert type(wrapper.model) is type(expected)
    assert set(wrapper.model.state_dict()) == set(expected.state_dict())
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(wrapper.model.state_dict()[key], value, rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(
            wrapper.model(IDS, use_cache=False).logits, expected(IDS, use_cache=False).logits, rtol=2e-5, atol=3e-7
        )
    assert not marker.exists()
    weight_file = path / "model.safetensors"
    data = weight_file.read_bytes()
    weight_file.write_bytes(b"broken")
    with pytest.raises(Exception, match="[Hh]eader|safetensor|buffer"):
        GPTQModel.load(str(root), subfolder="nested", device="cpu", backend=BACKEND.TORCH, trust_remote_code=trust)
    weight_file.write_bytes(data)
    retry = GPTQModel.load(
        str(root),
        subfolder="nested",
        device="cpu",
        dtype=torch.float32,
        backend=BACKEND.TORCH,
        trust_remote_code=trust,
    )
    with torch.inference_mode():
        torch.testing.assert_close(
            retry.model(IDS, use_cache=False).logits, expected(IDS, use_cache=False).logits, rtol=2e-5, atol=3e-7
        )
    assert not marker.exists()


@pytest.mark.parametrize("kv", [2, 4], ids=["gqa", "mha"])
def test_minimax_m2_awq_folds_are_neutral(kv):
    model = tiny_model("minimax_m2", kv=kv)
    _convert_model_with_defuser(MODEL_MAP["minimax_m2"], model, cleanup_original=False)

    def forward():
        return model(IDS, use_cache=False).logits

    experts = range(model.config.num_local_experts)
    for layer in model.model.layers:
        checked = assert_awq_folds_neutral(MODEL_MAP["minimax_m2"], model, layer, forward)
        groups = {(source, tuple(targets)): consumers for source, targets, consumers in checked}
        assert groups[("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"))] == []
        assert (("self_attn.v_proj", ("self_attn.o_proj",)) in groups) == (kv == 4)
        targets = tuple(f"mlp.experts.{i}.{p}" for i in experts for p in ("gate_proj", "up_proj"))
        # The sigmoid router reads the normed input and absorbs the scale; the
        # score-correction bias is added after the sigmoid and is scale-free.
        assert groups[("post_attention_layernorm", targets)] == ["mlp.gate"]
        for index in experts:
            assert (f"mlp.experts.{index}.up_proj", (f"mlp.experts.{index}.down_proj",)) in groups


@pytest.mark.parametrize("layout", ["mixed", "full", "linear"])
@pytest.mark.parametrize("shared,postnorm", [(True, True), (True, False), (False, True)])
def test_minimax_text01_awq_folds_are_neutral(layout, shared, postnorm):
    model = tiny_model("minimax", kv=4, layout=layout, shared=shared, postnorm=postnorm)
    _convert_model_with_defuser(MODEL_MAP["minimax"], model, cleanup_original=False)

    def forward():
        return model(IDS, use_cache=False).logits

    experts = range(model.config.num_local_experts)
    for layer in model.model.layers:
        checked = assert_awq_folds_neutral(MODEL_MAP["minimax"], model, layer, forward)
        groups = {(source, tuple(targets)) for source, targets, _consumers in checked}
        expected = {(f"mlp.experts.{i}.up_proj", (f"mlp.experts.{i}.down_proj",)) for i in experts}
        if shared:
            expected.add(("shared_mlp.up_proj", ("shared_mlp.down_proj",)))
        # The postnorm residual is the normed value, and the lightning output
        # is sigmoid-gated after a scale-free norm, so neither norm nor any
        # attention projection may absorb a scale. Only the MLP folds remain.
        assert groups == expected


@pytest.mark.parametrize(
    "model_type,shared,postnorm", [("minimax", False, True), ("minimax", True, False), ("minimax_m2", False, True)]
)
@pytest.mark.parametrize("offload", [False, True])
def test_public_awq_pack_infer_and_two_reloads(tmp_path, model_type, shared, postnorm, offload):
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear

    path = tmp_path / "native"
    original, marker = hostile_checkpoint(path, model_type, shared=shared, postnorm=postnorm)
    config = AWQConfig(bits=4, group_size=16, device="cpu", offload_to_disk=offload, fallback=None)
    wrapper = GPTQModel.load(str(path), quantize_config=config, dtype=torch.float16, backend=BACKEND.TORCH)
    wrapper.quantize(
        CalibrationConfig(dataset=[{"input_ids": row[None], "attention_mask": torch.ones_like(row)[None]} for row in IDS], batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    packed = {name for name, m in wrapper.model.named_modules() if isinstance(m, BaseQuantLinear)}
    assert packed
    searched = {(int(row["layer"]), str(row["module"])) for row in wrapper.quant_log if row["loss"] != "not applicable"}
    expected_search = "self_attn.q_proj" if model_type == "minimax_m2" else "shared_mlp.down_proj" if shared else None
    if expected_search is not None:
        assert {(0, expected_search), (1, expected_search)} <= searched, sorted(searched)
    assert any(".self_attn." in name for name in packed)
    assert any(".mlp.experts." in name for name in packed)
    # M2 folds scales into its decoder norms and compensates the router.
    folded = (".input_layernorm.", ".post_attention_layernorm.", ".mlp.gate.") if model_type == "minimax_m2" else ()
    for key, value in original.state_dict().items():
        if ".experts." in key or any(key.startswith(name + ".") for name in packed):
            continue
        if any(part in key for part in folded):
            assert not torch.equal(wrapper.model.state_dict()[key], value.half()), key
            continue
        torch.testing.assert_close(wrapper.model.state_dict()[key], value.half(), rtol=0, atol=0)
    expected_linear_count = 26 if model_type == "minimax_m2" else 31 if shared else 25
    assert len(packed) == expected_linear_count
    assert not marker.exists()
    dense = {
        name: value.clone()
        for name, value in wrapper.model.state_dict().items()
        if not any(name.startswith(p + ".") for p in packed)
    }
    with torch.inference_mode():
        logits = wrapper.model(IDS, use_cache=False).logits
        assert torch.isfinite(logits).all()
        prefill = wrapper.model(IDS[:, :1], use_cache=True)
        continued = wrapper.model(IDS[:, 1:], past_key_values=prefill.past_key_values, use_cache=True).logits
        torch.testing.assert_close(continued, logits[:, 1:], rtol=0.01, atol=0.002)
    for cycle in range(2):
        target = tmp_path / f"awq-{cycle}"
        before = {k: v.clone() for k, v in wrapper.model.state_dict().items()}
        wrapper.save(target)
        assert list(target.glob("*.safetensors")) and not list(target.glob("*.bin"))
        wrapper = GPTQModel.load(str(target), device="cpu", dtype=torch.float16, backend=BACKEND.TORCH)
        assert set(wrapper.model.state_dict()) == set(before)
        for name, value in before.items():
            torch.testing.assert_close(wrapper.model.state_dict()[name], value, rtol=0, atol=0)
        for name, value in dense.items():
            torch.testing.assert_close(wrapper.model.state_dict()[name], value, rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(wrapper.model(IDS, use_cache=False).logits, logits, rtol=0, atol=0)
