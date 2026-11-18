# SPDX-License-Identifier: Apache-2.0
import json
import math
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models.definitions.xing4_0 import Xing4_0QModel
from gptqmodel.models.ports.xing4_0.configuration_xing4_0 import Xing4_0Config
from gptqmodel.models.ports.xing4_0.modeling_xing4_0 import Xing4_0ForCausalLM
from gptqmodel.models.ports.xing4_0.tokenization_xing4_0 import Xing4_0Tokenizer
from tests.port_assets import read_port_asset, read_port_json


ASSETS = Path(__file__).resolve().parents[1] / "gptqmodel/models/ports/xing4_0/assets"


def tiny_config(**kwargs):
    data = {"vocab_size": 48, "hidden_size": 16, "intermediate_size": 24, "moe_intermediate_size": 8,
                "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
                "q_lora_rank": 8, "kv_lora_rank": 8, "qk_rope_head_dim": 4, "qk_nope_head_dim": 4,
                "v_head_dim": 4, "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
                "first_k_dense_replace": 1, "n_group": 1, "topk_group": 1, "hc_mult": 3,
                "hc_sinkhorn_iters": 5, "max_position_embeddings": 64, "pad_token_id": 0}
    data.update(kwargs)
    return Xing4_0Config(**data)


def test_fresh_hyperconnection_initialization_and_native_forward():
    model = Xing4_0ForCausalLM(tiny_config()).eval()
    ids = torch.tensor([[4, 5, 6, 7]])
    with torch.no_grad():
        output = model(ids)
    assert output.logits.shape == (1, 4, 48)
    assert torch.isfinite(output.logits).all()


def tiny_model(**kwargs):
    torch.manual_seed(152)
    model = Xing4_0ForCausalLM(tiny_config(**kwargs)).eval()
    model.set_attn_implementation("eager")
    with torch.no_grad():
        for layer in model.model.layers:
            for hc in (layer.attn_hc, layer.ffn_hc):
                hc.hc_base.copy_(torch.linspace(-0.7, 0.8, hc.hc_base.numel()))
                hc.hc_scale.copy_(torch.tensor([0.4, 0.7, 1.3]))
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.e_score_correction_bias.copy_(torch.tensor([0.00391319, -0.002332191, 0.001397211, 0.000153939]))
    return model


def norm(x, weight, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight


def source_oracle(model, ids, positions=None, keep=None):
    """Independent float64 equations; never call port forward/rotary/router/HC."""
    c = model.config
    w = {name: value.double() for name, value in model.state_dict().items()}
    b, s = ids.shape
    if positions is None:
        positions = torch.arange(s).expand(b, -1)
    x = w["model.embed_tokens.weight"][ids].unsqueeze(2).expand(-1, -1, c.hc_mult, -1)

    def linear(x, key):
        return F.linear(x, w[key + ".weight"], w.get(key + ".bias"))

    def hc(x, prefix):
        flat = x.flatten(2)
        flat = norm(flat, 1, c.rms_norm_eps)
        mixed = F.linear(flat, w[prefix + ".hc_fn"])
        h = c.hc_mult
        pre, post, comb = mixed.split([h, h, h*h], -1)
        pre_b, post_b, comb_b = w[prefix + ".hc_base"].split([h, h, h*h])
        scales = w[prefix + ".hc_scale"]
        pre = (pre * scales[0] + pre_b).sigmoid()
        post = 2 * (post * scales[1] + post_b).sigmoid()
        logits = (comb.reshape(b, s, h, h) * scales[2] + comb_b.reshape(h, h)).clamp(
            c.mhc_h_res_clamp_min, c.mhc_h_res_clamp_max)
        comb = (logits - logits.max(-1, keepdim=True).values).exp()
        for _ in range(c.hc_sinkhorn_iters):
            comb = comb / (comb.sum(-1, keepdim=True) + c.hc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + c.hc_eps)
        return post, comb, (pre.unsqueeze(-1) * x).sum(2)

    def mlp(x, prefix):
        return linear(F.silu(linear(x, prefix + ".gate_proj")) * linear(x, prefix + ".up_proj"), prefix + ".down_proj")

    freq = c.rope_theta ** (-torch.arange(0, c.qk_rope_head_dim, 2).double() / c.qk_rope_head_dim)
    angle = positions.double().unsqueeze(-1) * freq
    co, si = angle.cos().unsqueeze(1), angle.sin().unsqueeze(1)

    def rotate(x):
        if c.rope_interleave:
            even, odd = x[..., ::2], x[..., 1::2]
        else:
            even, odd = x.chunk(2, -1)
        return torch.cat((even*co - odd*si, odd*co + even*si), -1)

    for index in range(c.num_hidden_layers):
        p = f"model.layers.{index}"
        post, comb, collapsed = hc(x, p + ".attn_hc")
        z = norm(collapsed, w[p + ".input_layernorm.weight"], c.rms_norm_eps)
        a = p + ".self_attn"
        if c.q_lora_rank is None:
            q = linear(z, a + ".q_proj")
        else:
            q = linear(norm(linear(z, a + ".q_a_proj"), w[a + ".q_a_layernorm.weight"], 1e-6), a + ".q_b_proj")
        q = q.reshape(b, s, c.num_attention_heads, c.qk_head_dim).transpose(1, 2)
        qpass, qrot = q.split([c.qk_nope_head_dim, c.qk_rope_head_dim], -1)
        compressed = linear(z, a + ".kv_a_proj_with_mqa")
        latent, krot = compressed.split([c.kv_lora_rank, c.qk_rope_head_dim], -1)
        kv = linear(norm(latent, w[a + ".kv_a_layernorm.weight"], 1e-6), a + ".kv_b_proj")
        kv = kv.reshape(b, s, c.num_attention_heads, c.qk_nope_head_dim+c.v_head_dim).transpose(1, 2)
        kpass, value = kv.split([c.qk_nope_head_dim, c.v_head_dim], -1)
        query = torch.cat((qpass, rotate(qrot)), -1)
        key = torch.cat((kpass, rotate(krot.unsqueeze(1)).expand(-1, c.num_attention_heads, -1, -1)), -1)
        scores = (query @ key.transpose(-1, -2)) / c.qk_head_dim**0.5
        allowed = torch.arange(s)[:, None] >= torch.arange(s)[None, :]
        allowed = allowed[None, None].expand(b, 1, s, s)
        if keep is not None:
            allowed = allowed & keep[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, torch.finfo(torch.float64).min)
        out = (scores.softmax(-1) @ value).transpose(1, 2).reshape(b, s, -1)
        x = post.unsqueeze(-1) * linear(out, a + ".o_proj").unsqueeze(2) + comb @ x
        post, comb, collapsed = hc(x, p + ".ffn_hc")
        z = norm(collapsed, w[p + ".post_attention_layernorm.weight"], c.rms_norm_eps)
        if index < c.first_k_dense_replace:
            out = mlp(z, p + ".mlp")
        else:
            scores = linear(z, p + ".mlp.gate").sigmoid()
            ranked = scores + w[p + ".mlp.gate.e_score_correction_bias"]
            selected = ranked.topk(c.num_experts_per_tok, sorted=False).indices
            weights = scores.gather(-1, selected)
            if c.norm_topk_prob:
                weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
            weights = weights * c.routed_scaling_factor
            out = mlp(z, p + ".mlp.shared_experts")
            for expert in range(c.n_routed_experts):
                gain = (weights * (selected == expert)).sum(-1, keepdim=True)
                out = out + mlp(z, p + f".mlp.experts.{expert}") * gain
        x = post.unsqueeze(-1) * out.unsqueeze(2) + comb @ x
    return F.linear(norm(x.mean(2), w["model.norm.weight"], c.rms_norm_eps), w["lm_head.weight"])


@pytest.mark.parametrize("qlora,bias,interleave,dense", [(8, False, True, 1), (None, True, False, 0), (8, True, False, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_independent_full_decoder_equations_and_cache(qlora, bias, interleave, dense, dtype, implementation):
    model = tiny_model(q_lora_rank=qlora, attention_bias=bias, rope_interleave=interleave,
                       first_k_dense_replace=dense).to(dtype)
    model.set_attn_implementation(implementation)
    ids = torch.tensor([[4, 7, 5, 11, 9, 6], [8, 10, 14, 5, 7, 12]])
    with torch.no_grad():
        full = model(ids, use_cache=False).logits
        expected = source_oracle(model, ids)
        atol = {torch.float32: 3e-7, torch.float16: 7e-4, torch.bfloat16: 5e-3}[dtype]
        torch.testing.assert_close(full.double(), expected, atol=atol, rtol=0.015 if dtype != torch.float32 else 2e-5)
        first = model(ids[:, :1], use_cache=True)
        other = model(ids[:, :2], use_cache=True)
        second = model(ids[:, 1:5], past_key_values=first.past_key_values, use_cache=True)
        last = model(ids[:, 5:], past_key_values=second.past_key_values, use_cache=True)
        assert other.past_key_values.get_seq_length() == 2
        assert last.past_key_values.get_seq_length() == 6
        torch.testing.assert_close(second.logits, full[:, 1:5], atol=atol, rtol=0.015)
        torch.testing.assert_close(last.logits, full[:, 5:], atol=atol, rtol=0.015)


def write_assets(path):
    path.mkdir(parents=True, exist_ok=True)
    for name in ["tokenizer_config.json", "chat_template.jinja", "generation_config.json"]:
        shutil.copyfile(ASSETS / name, path / name)
    (path / "tokenizer.model").write_bytes(read_port_asset(ASSETS / "tokenizer.model"))


def checkpoint(path, model=None):
    model = model if model is not None else tiny_model()
    model.save_pretrained(path, safe_serialization=True)
    write_assets(path)
    config = json.loads((path / "config.json").read_text())
    config["auto_map"] = dict.fromkeys(["AutoConfig", "AutoModel", "AutoModelForCausalLM"], "evil.Evil")
    (path / "config.json").write_text(json.dumps(config))
    tokenizer = json.loads((path / "tokenizer_config.json").read_text())
    tokenizer["auto_map"] = {"AutoTokenizer": ["evil.Evil", None]}
    (path / "tokenizer_config.json").write_text(json.dumps(tokenizer))
    marker = path / "executed"
    (path / "evil.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python executed')\n")
    (path / "custom_generate").mkdir(exist_ok=True)
    shutil.copyfile(path / "evil.py", path / "custom_generate/generate.py")
    return model, marker


@pytest.mark.parametrize("trust", [None, False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_hostile_maps_real_tokenizer_generation_and_two_dense_cycles(tmp_path, monkeypatch, trust, bypass):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    expected, marker = checkpoint(tmp_path / "source")
    loaded = GPTQModel.load(str(tmp_path / "source"), trust_remote_code=trust, dtype=torch.float32,
                           device="cpu", attn_implementation="eager", code_revision="untrusted")
    assert type(loaded.model) is Xing4_0ForCausalLM
    assert type(loaded.model.config) is Xing4_0Config
    assert type(loaded.tokenizer.tokenizer) is Xing4_0Tokenizer
    ids = torch.tensor([[4, 5, 6]])
    with torch.no_grad():
        logits = expected(ids, use_cache=False).logits
        generated = loaded.model.generate(ids, max_new_tokens=3, do_sample=False)
        beam = loaded.model.generate(ids, max_new_tokens=3, num_beams=3, do_sample=False)
    assert generated.shape == beam.shape == (1, 6)
    for index in range(2):
        assert loaded.model.generation_config.do_sample is True
        assert loaded.model.generation_config.top_p == 0.95
        assert loaded.model.generation_config.eos_token_id == 2
        with torch.no_grad():
            torch.testing.assert_close(loaded.model(ids, use_cache=False).logits, logits, rtol=0, atol=0)
        assert loaded.tokenizer.tokenizer.encode("Hello 世界", add_special_tokens=False) == [29864, 34912]
        destination = tmp_path / f"cycle-{index}"
        loaded.model.save_pretrained(destination, safe_serialization=True)
        loaded.tokenizer.save_pretrained(destination)
        loaded = GPTQModel.load(str(destination), dtype=torch.float32, device="cpu", trust_remote_code=trust,
                               attn_implementation="eager")
    assert not marker.exists()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("assign", [False, True])
def test_safetensors_assignment_retry_generation_and_optional_outputs(tmp_path, dtype, assign):
    model = tiny_model().to(dtype)
    for layer in model.model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.e_score_correction_bias = layer.mlp.gate.e_score_correction_bias.float()
    fresh = tiny_model()
    fresh.load_state_dict(model.state_dict(), assign=assign)
    if not assign:
        fresh.to(dtype)
    ids = torch.tensor([[4, 7, 5, 11]])
    with torch.no_grad():
        reference = model(ids, use_cache=False, output_hidden_states=True, output_attentions=True)
        assert len(reference.hidden_states) == 3
        assert len(reference.attentions) == 2
        torch.testing.assert_close(fresh(ids, use_cache=False).logits, reference.logits, atol=0, rtol=0)
        assert torch.equal(fresh.generate(ids, max_new_tokens=3, do_sample=False), model.generate(ids, max_new_tokens=3, do_sample=False))
        assert torch.equal(fresh.generate(ids, max_new_tokens=3, num_beams=3, do_sample=False), model.generate(ids, max_new_tokens=3, num_beams=3, do_sample=False))
        torch.testing.assert_close(model(ids, return_dict=False, use_cache=False)[0], reference.logits, rtol=0, atol=0)
    model.save_pretrained(tmp_path, safe_serialization=True)
    payload = tmp_path / "model.safetensors"
    original = payload.read_bytes()
    payload.write_bytes(b"bad header")
    with pytest.raises(Exception, match="[Hh]eader|[Ss]afetensor|[Dd]eserializ"):
            Xing4_0QModel.loader.from_pretrained(tmp_path, dtype=dtype, attn_implementation="eager")
    payload.write_bytes(original)
    for index in range(2):
        restored = Xing4_0QModel.loader.from_pretrained(tmp_path, dtype=dtype, attn_implementation="eager").eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(ids, use_cache=False).logits, reference.logits, rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
        assert not list(tmp_path.glob("*.bin"))
        restored.save_pretrained(tmp_path, safe_serialization=True)


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("qlora,bias,dense", [(16, False, 1), (None, True, 0), (16, True, 2)])
def test_public_plain_awq_every_linear_independent_stream_capture_two_reloads(tmp_path, offload, qlora, bias, dense):
    from safetensors.torch import load_file, save_file

    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig, CalibrationConfig

    model = tiny_model(hidden_size=32, intermediate_size=32, moe_intermediate_size=32,
                       q_lora_rank=qlora, kv_lora_rank=16, v_head_dim=16, attention_bias=bias,
                       qk_rope_head_dim=16, qk_nope_head_dim=16,
                       first_k_dense_replace=dense).half()
    for layer in model.model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.e_score_correction_bias = torch.tensor([0.00391319, -0.002332191, 0.001397211, 0.000153939], dtype=torch.float32)
    source, marker = checkpoint(tmp_path / "source", model)
    extra = {"model.layers.40.shared_head.norm.weight": torch.arange(4, dtype=torch.bfloat16),
             "model.layers.40.mlp.gate.e_score_correction_bias": torch.tensor([0.00391319, -0.002332191])}
    payload = tmp_path / "source/model.safetensors"
    save_file(load_file(payload) | extra, payload, metadata={"format": "pt"})
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=offload)
    wrapper = GPTQModel.load(str(tmp_path / "source"), quantize_config=qcfg, backend=BACKEND.TORCH,
                            dtype=torch.float16, attn_implementation="eager")
    targets = {name for name, module in source.named_modules() if isinstance(module, torch.nn.Linear)
               and name.startswith("model.layers.") and not name.endswith(".gate")}
    unchanged = {name: value.clone() for name, value in source.state_dict().items()
                 if name.rsplit(".", 1)[0] not in targets}
    rows = [{"input_ids": [4, 5, 6, 7, 8, 9], "attention_mask": [1] * 6},
            {"input_ids": [11, 10, 9, 8, 7], "attention_mask": [1] * 5}]
    expected_capture = source.model.embed_tokens.weight[torch.tensor([rows[0]["input_ids"]])].unsqueeze(2).expand(-1, -1, 3, -1)
    captures = []

    def capture(_module, args, kwargs):
        if not captures:
            captures.append((args[0] if args else kwargs["hidden_states"]).detach().cpu().clone())

    handle = wrapper.model.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for layer in wrapper.model.model.layers:
            assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
        wrapper.quantize(calibration=CalibrationConfig(dataset=rows, batch_size=1, min_length=1, sort=None), backend=BACKEND.TORCH)
    finally:
        handle.remove()
    assert captures
    torch.testing.assert_close(captures[0], expected_capture, atol=0, rtol=0)
    packed = {name for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert packed == targets
    for name, value in unchanged.items():
        torch.testing.assert_close(wrapper.model.state_dict()[name], value, atol=0, rtol=0)
    ids = torch.tensor([[4, 5, 6, 7]])
    with torch.no_grad():
        logits = wrapper.model(ids, use_cache=False).logits
        assert torch.isfinite(logits).all()
    state = {name: value.clone() for name, value in wrapper.model.state_dict().items()}
    for index in range(2):
        destination = tmp_path / f"quantized-{index}"
        wrapper.save_quantized(str(destination))
        assert list(destination.glob("*.safetensors")) and not list(destination.rglob("*.bin"))
        saved = {key: value for shard in destination.glob("*.safetensors") for key, value in load_file(shard).items()}
        for key, value in extra.items():
            torch.testing.assert_close(saved[key], value, rtol=0, atol=0)
        wrapper = GPTQModel.load(str(destination), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16,
                                attn_implementation="eager")
        assert {name for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)} == targets
        assert set(wrapper.model.state_dict()) == set(state)
        for name, value in wrapper.model.state_dict().items():
            torch.testing.assert_close(value, state[name], atol=0, rtol=0)
        with torch.no_grad():
            torch.testing.assert_close(wrapper.model(ids, use_cache=False).logits, logits, atol=0, rtol=0)
    assert not marker.exists()


TOKENIZER_ORACLE = [
    ("Hello 世界", [29864, 34912]),
    ("Hello, Xing4.0!", [29864, 124380, 1941, 424, 28, 124379, 24, 124515]),
    ("  leading\n中文 café 🧪", [396, 5136, 35, 10907, 43513, 124361, 380, 299, 307, 310]),
    ("A\x00B", [465, 140, 124428]),
    ("<_start>", [1]),
    ("<_user>Hi<_bot>", [4, 20814, 5]),
]


@pytest.mark.parametrize("bos,eos", [(False, False), (True, False), (False, True), (True, True)])
def test_pinned_source_sentencepiece_ids_special_pairs_chat_and_snapshot_reloads(tmp_path, bos, eos):
    write_assets(tmp_path / "source")
    tokenizer = Xing4_0Tokenizer.from_pretrained(tmp_path / "source", add_bos_token=bos, add_eos_token=eos)
    source_bytes = (tmp_path / "source/tokenizer.model").read_bytes()
    pair_a, pair_b = [29864, 34912], [465, 140, 124428]
    special = ([1] if bos else [], [2] if eos else [])
    expected_pair = special[0] + pair_a + special[1] + special[0] + pair_b + special[1]
    assert tokenizer.encode("Hello 世界", text_pair="A\x00B") == expected_pair
    assert tokenizer.get_special_tokens_mask(pair_a, pair_b) == (
        [1] * len(special[0]) + [0, 0] + [1] * len(special[1]) +
        [1] * len(special[0]) + [0, 0, 0] + [1] * len(special[1]))
    (tmp_path / "source/tokenizer.model").unlink()
    for index in range(2):
        for text, ids in TOKENIZER_ORACLE:
            assert tokenizer.encode(text, add_special_tokens=False) == ids
            assert tokenizer.decode(ids) == text
        assert tokenizer.get_vocab()["<_start>"] == 1
        assert tokenizer.get_vocab()["<_end>"] == 2
        for thinking, suffix in [(False, "</think>"), (True, "<think>\n")]:
            messages = [{"role": "system", "content": "Answer precisely."}, {"role": "user", "content": "Hello 世界"}]
            prompt = "<_system>Answer precisely.<_user>Hello 世界<_bot>" + suffix
            assert tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=thinking) == prompt
            assert tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=thinking, return_dict=False) == (
                [6, 42420, 19376, 124379, 4, 29864, 34912, 5] + ([9, 124361, 35] if thinking else [10]))
        destination = tmp_path / f"cycle-{index}"
        tokenizer.save_pretrained(destination)
        assert (destination / "tokenizer.model").read_bytes() == source_bytes
        tokenizer = Xing4_0Tokenizer.from_pretrained(destination)
        assert (tokenizer.add_bos_token, tokenizer.add_eos_token) == (bos, eos)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_corrupt_vocabulary_fails_then_repair_retries(tmp_path, monkeypatch, trust, bypass):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    _, marker = checkpoint(tmp_path)
    vocab = tmp_path / "tokenizer.model"
    original = vocab.read_bytes()
    vocab.write_bytes(b"invalid proto")
    with pytest.raises(RuntimeError, match="Xing4_0Tokenizer.*tokenizer.model"):
        GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert not marker.exists()
    vocab.write_bytes(original)
    restored = GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert restored.tokenizer.tokenizer.encode("A\x00B", add_special_tokens=False) == [465, 140, 124428]
    assert not marker.exists()


@pytest.mark.parametrize("heads,shared,hc", [(1, 1, 1), (2, 2, 4), (4, 3, 2)])
@pytest.mark.parametrize("qlora", [None, 8])
def test_multihead_multishared_hyperconnection_variants_and_left_padded_cache(heads, shared, hc, qlora):
    model = tiny_model(num_attention_heads=heads, num_key_value_heads=heads, n_shared_experts=shared,
                       hc_mult=hc, q_lora_rank=qlora, norm_topk_prob=False)
    ids = torch.tensor([[0, 0, 4, 7, 6], [8, 11, 5, 6, 9]])
    keep = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    positions = keep.cumsum(-1) - 1
    positions.masked_fill_(keep == 0, 1)
    with torch.no_grad():
        output = model(ids, attention_mask=keep, position_ids=positions, use_cache=False)
        expected = source_oracle(model, ids, positions=positions, keep=keep)
        torch.testing.assert_close(output.logits[keep.bool()].double(), expected[keep.bool()], atol=4e-7, rtol=3e-5)
        first = model(ids[:, :3], attention_mask=keep[:, :3], position_ids=positions[:, :3], use_cache=True)
        last = model(ids[:, 3:], attention_mask=keep, position_ids=positions[:, 3:], past_key_values=first.past_key_values)
        torch.testing.assert_close(last.logits, output.logits[:, 3:], atol=4e-7, rtol=3e-5)
        embedded = model(inputs_embeds=model.model.embed_tokens(ids), attention_mask=keep,
                         position_ids=positions, use_cache=False)
        torch.testing.assert_close(embedded.logits, output.logits, rtol=0, atol=0)


def test_release_yarn_rotary_at_large_positions_independent_scalar_equations():
    rope = json.loads((ASSETS / "config.json").read_text())["rope_scaling"]
    model = tiny_model(qk_rope_head_dim=64, rope_scaling=rope, max_position_embeddings=262144)
    positions = torch.tensor([[0, 4095, 4096, 65535, 262143]])
    cos, sin = model.model.rotary_emb(torch.zeros(1), positions)
    low = max(math.floor(64 * math.log(4096 / (32 * 2 * math.pi)) / (2 * math.log(10000))), 0)
    high = min(math.ceil(64 * math.log(4096 / (2 * math.pi)) / (2 * math.log(10000))), 63)
    frequencies = []
    for index in range(32):
        ramp = min(max((index - low) / (high - low), 0), 1)
        frequencies.append((10000 ** (-2 * index / 64)) * (1 - ramp + ramp / 64))
    # Native/source rotary explicitly evaluates the frequency-position product in FP32.
    angles = torch.tensor(frequencies, dtype=torch.float32)[None, None] * positions.float()[..., None]
    expected_cos = angles.cos().repeat(1, 1, 2)
    expected_sin = angles.sin().repeat(1, 1, 2)
    torch.testing.assert_close(cos, expected_cos, atol=0.005, rtol=0)
    torch.testing.assert_close(sin, expected_sin, atol=0.005, rtol=0)
    assert model.model.layers[0].self_attn.scaling == pytest.approx(
        (68 ** -0.5) * (1 + 0.1 * math.log(64)) ** 2)


@pytest.mark.parametrize("invalid", [{"num_key_value_heads": 1}, {"ep_size": 2}, {"scoring_func": "softmax"},
                                    {"topk_method": "greedy"}, {"hc_mult": 0}, {"hc_sinkhorn_iters": 0},
                                    {"qk_rope_head_dim": 3}])
def test_invalid_or_unqualified_layouts_fail_at_config(invalid):
    with pytest.raises(ValueError):
        tiny_config(**invalid)


def test_generation_greedy_and_beam_match_independent_full_prefix_oracle():
    from transformers import GenerationConfig

    model = tiny_model()
    prompt = torch.tensor([[4, 7, 11]])
    policy = GenerationConfig(max_new_tokens=4, do_sample=False, eos_token_id=None, pad_token_id=0)
    greedy = prompt.clone()
    for _ in range(4):
        greedy = torch.cat((greedy, source_oracle(model, greedy)[:, -1].argmax(-1, keepdim=True)), -1)
    with torch.no_grad():
        assert torch.equal(model.generate(prompt, generation_config=policy), greedy)
    prefixes = prompt.expand(3, -1).clone()
    gains = torch.tensor([0.0, -1e9, -1e9], dtype=torch.float64)
    for _ in range(4):
        log_probs = source_oracle(model, prefixes)[:, -1].log_softmax(-1)
        gains, flat = (log_probs + gains[:, None]).flatten().topk(3)
        prefixes = torch.cat((prefixes[flat // model.config.vocab_size], (flat % model.config.vocab_size)[:, None]), -1)
    with torch.no_grad():
        assert torch.equal(model.generate(prompt, generation_config=policy, num_beams=3), prefixes[:1])
    # A real generated-token EOS, not only an EOS-valued prompt token.
    policy.eos_token_id = int(greedy[0, prompt.shape[1]])
    with torch.no_grad():
        assert torch.equal(model.generate(prompt, generation_config=policy), greedy[:, :4])


def test_fp8_release_is_same_architecture_metadata_not_a_second_remote_alias():
    import hashlib

    raw = (ASSETS / "fp8_config.json").read_bytes()
    manifest = json.loads((ASSETS / "fp8_source_manifest.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == manifest["config_sha256"]
    config = Xing4_0Config.from_dict(json.loads(raw))
    assert config.model_type == "xing4_0"
    assert config.quantization_config["quant_method"] == "fp8"
    assert config.rope_parameters["rope_type"] == "yarn"
    dense = read_port_json(ASSETS / "inventory.json")
    with torch.device("meta"):
        model = Xing4_0QModel.loader.from_config(config)
    assert type(model) is Xing4_0ForCausalLM
    assert {key: list(value.shape) for key, value in model.state_dict().items()} == {
        key: value["shape"] for key, value in dense.items() if not key.startswith("model.layers.40.")}


def test_no_native_mapping_and_public_actual_release_vocabulary_text_forward(tmp_path):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    assert "xing4_0" not in CONFIG_MAPPING_NAMES
    assert "xing4_0" not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    source, marker = checkpoint(tmp_path, tiny_model(vocab_size=131072))
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, attn_implementation="eager")
    tokens = loaded.tokenizer("Hello 世界", return_tensors="pt")
    assert tokens["input_ids"].tolist() == [[29864, 34912]]
    with torch.no_grad():
        output = loaded.model(**tokens, use_cache=False)
        expected = source_oracle(source, tokens["input_ids"])
        torch.testing.assert_close(output.logits.double(), expected, atol=3e-7, rtol=3e-5)
        text = loaded.model.generate(**tokens, max_new_tokens=2, do_sample=False)
    assert text.shape == (1, 4)
    assert isinstance(loaded.tokenizer.decode(text[0]), str)
    assert not marker.exists()
