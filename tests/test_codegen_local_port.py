# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Real CodeGen checkpoint semantics; no checkpoint Python is imported."""

import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import CodeGenConfig as NativeCodeGenConfig
from transformers import CodeGenForCausalLM as NativeCodeGenForCausalLM
from transformers import PreTrainedTokenizerFast
from transformers.models.codegen.modeling_codegen import create_sinusoidal_positions

from gptqmodel import GPTQModel
from gptqmodel.models.definitions.codegen import CodeGenQModel
from gptqmodel.models.ports.codegen.configuration_codegen import CodeGenConfig
from gptqmodel.models.ports.codegen.modeling_codegen import CodeGen2Model, CodeGenForCausalLM
from gptqmodel.models.ports.native.codegen.modeling_codegen import CodeGenModel as VendoredCodeGenModel
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code
from gptqmodel.utils.security import BYPASS_ENV


CONFIGS = json.loads((Path(__file__).parent / "fixtures/ports/codegen/upstream_configs.json").read_text())


def tiny_config(partitions=8, **kwargs):
    args = {"n_embd": 64, "n_head": 8, "n_layer": 2, "rotary_dim": 4, "n_inner": 96,
            "n_ctx": 32, "n_positions": 32, "vocab_size": 41, "bos_token_id": 1, "eos_token_id": 2,
            "pad_token_id": 0, "qkv_layout": f"qvk_{partitions}", "resid_pdrop": 0., "embd_pdrop": 0., "attn_pdrop": 0.}
    args.update(kwargs)
    return CodeGenConfig(**args)


def oracle_logits(model, ids, partitions, attention_mask=None):
    """Independent Salesforce math with per-head checkpoint row selection.

    Does not call the port's partition, RoPE, attention or block helpers.
    """
    cfg = model.config
    hidden = model.transformer.wte(ids)
    length = ids.shape[1]
    head_dim = cfg.n_embd // cfg.n_head
    positions = torch.arange(length, device=ids.device)
    rotary_dim = cfg.rotary_dim or head_dim
    inv = 10000 ** (-torch.arange(0, rotary_dim, 2, device=ids.device).float() / rotary_dim)
    angles = positions[:, None] * inv[None, :]
    sin, cos = (x.repeat_interleave(2, -1)[None, :, None, :] for x in (angles.sin(), angles.cos()))

    def rotate(x):
        rotated = torch.stack((-x[..., 1:rotary_dim:2], x[..., :rotary_dim:2]), -1).flatten(-2)
        return torch.cat((x[..., :rotary_dim] * cos + rotated * sin, x[..., rotary_dim:]), -1)

    for block in model.transformer.h:
        norm = block.ln_1(hidden)
        fused = torch.nn.functional.linear(norm, block.attn.qkv_proj.weight)
        q, v, k = [], [], []
        heads_per_partition = cfg.n_head // partitions
        for head in range(cfg.n_head):
            partition, local_head = divmod(head, heads_per_partition)
            start = partition * 3 * heads_per_partition * head_dim + local_head * head_dim
            for target, slot in ((q, 0), (v, 1), (k, 2)):
                offset = start + slot * heads_per_partition * head_dim
                target.append(fused[..., offset:offset + head_dim])
        q, v, k = (torch.stack(x, -2) for x in (q, v, k))
        q, k = (rotate(x).transpose(1, 2) for x in (q, k))
        v = v.transpose(1, 2)
        scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(head_dim)
        causal = positions[None, :] <= positions[:, None]
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                scores = scores + attention_mask
            else:
                scores = scores.masked_fill(~attention_mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min)
        values = scores.softmax(-1).to(v.dtype) @ v
        attention = torch.nn.functional.linear(values.transpose(1, 2).reshape(hidden.shape), block.attn.out_proj.weight)
        mlp = torch.nn.functional.linear(norm, block.mlp.fc_in.weight, block.mlp.fc_in.bias)
        mlp = .5 * mlp * (1 + torch.tanh(math.sqrt(2 / math.pi) * (mlp + .044715 * mlp.pow(3))))
        mlp = torch.nn.functional.linear(mlp, block.mlp.fc_out.weight, block.mlp.fc_out.bias)
        hidden = attention + mlp + hidden
    return model.lm_head(model.transformer.ln_f(hidden))


@pytest.mark.parametrize("name", CONFIGS)
def test_official_legacy_metadata_and_serialization(name, tmp_path):
    payload = CONFIGS[name]["config"]
    config = CodeGenConfig(**payload)
    expected = 8 if name in {"codegen2-1B_P", "codegen2-3_7B_P"} else 4
    assert config.qkv_partitions == expected
    config.save_pretrained(tmp_path)
    restored = CodeGenQModel.loader.load_config(tmp_path)
    assert restored.qkv_layout == config.qkv_layout
    assert getattr(restored, "head_dim", None) == payload.get("head_dim")
    assert restored.n_embd == config.n_embd


def test_native_config_saves_do_not_acquire_legacy_discriminator(tmp_path):
    for name, case in CONFIGS.items():
        if "head_dim" in case["config"]:
            continue
        config = NativeCodeGenConfig(**case["config"])
        config.save_pretrained(tmp_path)
        assert "head_dim" not in json.loads((tmp_path / "config.json").read_text()), name
        assert CodeGenQModel.loader.load_config(tmp_path).qkv_partitions == 4


def test_native_model_config_remains_four_partition_after_generation_and_save(tmp_path):
    payload = tiny_config(4).to_dict()
    payload.pop("qkv_layout")
    model = NativeCodeGenForCausalLM(NativeCodeGenConfig(**payload)).eval()
    with torch.inference_mode():
        model.generate(torch.tensor([[1, 4, 9]]), max_new_tokens=2, do_sample=False)
    model.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert "head_dim" not in saved
    assert "qkv_layout" not in saved
    restored = CodeGenQModel.loader.from_pretrained(tmp_path).eval()
    assert restored.config.qkv_partitions == 4
    ids = torch.tensor([[1, 4, 9, 3]])
    with torch.inference_mode():
        torch.testing.assert_close(restored(ids).logits, model(ids).logits, rtol=0, atol=0)


def test_explicit_migration_overrides_legacy_metadata_and_validates_api_updates(tmp_path):
    payload = tiny_config(8, head_dim=8).to_dict()
    del payload["qkv_layout"]
    (tmp_path / "config.json").write_text(json.dumps(payload))
    assert CodeGenQModel.loader.load_config(tmp_path).qkv_partitions == 8
    assert CodeGenQModel.loader.load_config(tmp_path, qkv_layout="qvk_4").qkv_partitions == 4
    with pytest.raises(ValueError, match="qkv_layout"):
        CodeGenQModel.loader.load_config(tmp_path, qkv_layout="ambiguous")
    with pytest.raises(ValueError, match="head_dim"):
        CodeGenQModel.loader.load_config(tmp_path, head_dim=9)
    config = tiny_config()
    config.qkv_layout = "ambiguous"
    with pytest.raises(ValueError, match="qkv_layout"):
        CodeGenQModel.loader.from_config(config)
    config.qkv_layout = "qvk_8"
    assert CodeGenQModel.loader.from_config(config)(torch.tensor([[1, 3, 4]])).logits.isfinite().all()


@pytest.mark.parametrize("width_key,heads_key", [("n_embd", "n_head"), ("hidden_size", "num_attention_heads")])
def test_structural_api_aliases_validate_before_initialization(width_key, heads_key):
    payload = tiny_config(8, head_dim=8).to_dict()
    original = dict(payload)
    config, unused = CodeGenConfig.from_dict(payload, **{width_key: 128, heads_key: 16},
                                           return_unused_kwargs=True, unrelated_argument=17)
    assert (config.n_embd, config.n_head, config.head_dim, config.qkv_partitions) == (128, 16, 8, 8)
    assert unused == {"unrelated_argument": 17}
    assert payload == original
    with pytest.raises(ValueError, match="head_dim"):
        CodeGenConfig.from_dict(payload, **{width_key: 96})
    with pytest.raises(ValueError, match="head geometry"):
        CodeGenConfig.from_dict(payload, **{heads_key: 4})


def test_null_legacy_discriminator_requires_explicit_migration():
    payload = tiny_config().to_dict()
    payload.pop("qkv_layout")
    payload["head_dim"] = None
    with pytest.raises(ValueError, match="explicit qkv_layout"):
        CodeGenConfig.from_dict(payload)
    for partitions in (4, 8):
        config = CodeGenConfig.from_dict(payload, qkv_layout=f"qvk_{partitions}")
        assert config.qkv_partitions == partitions


@pytest.mark.parametrize("overrides,match", [
    ({"qkv_layout": "unknown"}, "qkv_layout"),
    ({"head_dim": 7}, "head_dim"),
    ({"n_head": 6}, "head geometry"),
    ({"qkv_layout": "qvk_8", "n_head": 4}, "head geometry"),
    ({"rotary_dim": 9}, "rotary_dim"),
])
def test_invalid_metadata_fails_before_weights_and_retry_recovers(tmp_path, overrides, match):
    payload = tiny_config().to_dict()
    payload.update(overrides)
    (tmp_path / "config.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        CodeGenQModel.loader.from_pretrained(tmp_path)
    tiny_config().save_pretrained(tmp_path)
    model = CodeGenQModel.loader.from_config(CodeGenQModel.loader.load_config(tmp_path))
    assert model(torch.tensor([[1, 3, 5]]), use_cache=False).logits.isfinite().all()


@pytest.mark.parametrize("partitions", [4, 8])
@pytest.mark.parametrize("rotary_dim", [4, 8, None])
@pytest.mark.parametrize("n_head", [8, 24])
def test_full_model_numerics_and_layout_regression(partitions, rotary_dim, n_head):
    torch.manual_seed(106)
    model = CodeGenForCausalLM(tiny_config(partitions, rotary_dim=rotary_dim, n_head=n_head, n_embd=8 * n_head)).eval()
    ids = torch.tensor([[1, 5, 9, 3, 2], [4, 6, 11, 8, 7]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    with torch.inference_mode():
        expected = oracle_logits(model, ids, partitions, mask)
        actual = model(ids, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    assert isinstance(model.transformer, CodeGen2Model if partitions == 8 else VendoredCodeGenModel)
    if partitions == 8:
        # The old default native load succeeds with identical key names/shapes,
        # but must fail the numerical oracle: this is the repaired regression.
        native = NativeCodeGenForCausalLM(NativeCodeGenConfig(**tiny_config(4, rotary_dim=rotary_dim, n_head=n_head, n_embd=8 * n_head).to_dict())).eval()
        native.load_state_dict(model.state_dict(), strict=True)
        with torch.inference_mode():
            wrong = native(ids, attention_mask=mask, use_cache=False).logits
        assert (wrong - expected).abs().max() > .01
    else:
        native = NativeCodeGenForCausalLM(NativeCodeGenConfig(**model.config.to_dict())).eval()
        native.load_state_dict(model.state_dict(), strict=True)
        with torch.inference_mode():
            torch.testing.assert_close(actual, native(ids, attention_mask=mask, use_cache=False).logits, rtol=0, atol=0)


@pytest.mark.parametrize("partitions", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cache_generation_and_safetensors_roundtrips(tmp_path, partitions, dtype):
    torch.manual_seed(85)
    model = CodeGenForCausalLM._from_config(tiny_config(partitions), dtype=dtype).eval()
    ids = torch.tensor([[1, 3, 5, 8, 7], [1, 6, 4, 10, 9]])
    with torch.inference_mode():
        full = model(ids, use_cache=False).logits
        prefill = model(ids[:, :2], use_cache=True)
        cache = prefill.past_key_values
        assert cache.get_seq_length() == 2
        tail = model(ids[:, 2:], past_key_values=cache, use_cache=True).logits
        assert cache.get_seq_length() == 5
        torch.testing.assert_close(tail, full[:, 2:], atol=1e-6 if dtype == torch.float32 else .003, rtol=.01)
        cached_tokens = model.generate(ids[:, :2], max_new_tokens=3, do_sample=False, use_cache=True)
        torch.testing.assert_close(cached_tokens, model.generate(ids[:, :2], max_new_tokens=3, do_sample=False, use_cache=False))
    model.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").exists()
    assert not list(tmp_path.glob("*.bin"))
    stored = load_file(tmp_path / "model.safetensors")
    assert set(stored) == set(model.state_dict())
    for key, tensor in model.state_dict().items():
        assert stored[key].dtype == dtype
        torch.testing.assert_close(stored[key], tensor, rtol=0, atol=0)
    restored = CodeGenQModel.loader.from_pretrained(tmp_path, dtype=dtype).eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(ids, use_cache=False).logits, full, rtol=0, atol=0)
    restored.save_pretrained(tmp_path / "second")
    again = CodeGenQModel.loader.from_pretrained(tmp_path / "second", dtype=dtype).eval()
    with torch.inference_mode():
        torch.testing.assert_close(again(ids, use_cache=False).logits, full, rtol=0, atol=0)


def write_checkpoint(path, partitions, model=None):
    if model is None:
        model = CodeGenForCausalLM(tiny_config(partitions)).eval()
    model.save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(model.config.vocab_size)}, unk_token="t0")),
                                        unk_token="t0", pad_token="t0", bos_token="t1", eos_token="t2")
    tokenizer.save_pretrained(path)
    payload = json.loads((path / "config.json").read_text())
    payload.update(auto_map={"AutoConfig": "evil.Config", "AutoModelForCausalLM": "evil.Model"}, trust_remote_code=True)
    (path / "config.json").write_text(json.dumps(payload))
    marker = path / "EXECUTED"
    (path / "evil.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python executed')\n")
    return model, marker


@pytest.mark.parametrize("partitions", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_postconstruction_cast_save_public_reload_and_cast_back(tmp_path, partitions, dtype):
    # Long rotary positions amplify rounding that short-position smoke tests
    # miss. The derived table is absent from state_dict(), unlike the weights.
    torch.manual_seed(2027)
    config = tiny_config(partitions, n_embd=128, n_inner=192, rotary_dim=16,
                         n_positions=2048, n_ctx=2048, vocab_size=47, initializer_range=.1)
    model = CodeGenForCausalLM(config).eval()
    original_tables = [layer.attn.embed_positions.clone() for layer in model.transformer.h]
    original_weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    model.half() if dtype == torch.float16 else model.bfloat16()
    ids = torch.tensor([[18, 14, 17]])
    positions = torch.tensor([[1900, 1901, 1902]])
    with torch.inference_mode():
        before = model(ids, position_ids=positions, use_cache=False).logits
        tokens = model.generate(ids, position_ids=positions, max_new_tokens=3, do_sample=False)
    _, marker = write_checkpoint(tmp_path, partitions, model)
    restored = CodeGenQModel.loader.from_pretrained(tmp_path, dtype=dtype).eval()
    # Equal checkpoint weights must imply equal execution, including the
    # nonpersistent FP32 positional state regenerated during reload.
    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, original_weights[name].to(dtype), rtol=0, atol=0)
        torch.testing.assert_close(tensor, model.state_dict()[name], rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(restored(ids, position_ids=positions, use_cache=False).logits,
                                   before, rtol=0, atol=0)
        torch.testing.assert_close(restored.generate(ids, position_ids=positions, max_new_tokens=3, do_sample=False), tokens)
    public = GPTQModel.load(str(tmp_path), device="cpu", dtype=dtype, trust_remote_code=False)
    with torch.inference_mode():
        torch.testing.assert_close(public.model(ids, position_ids=positions, use_cache=False).logits,
                                   before, rtol=0, atol=0)
    assert not marker.exists()
    for layer, original in zip(model.transformer.h, original_tables):
        torch.testing.assert_close(layer.attn.embed_positions, original, rtol=0, atol=0)
    if partitions == 4:
        assert type(model.transformer) is VendoredCodeGenModel
    # Restoring only the derived FP32 tables is the independent reference: no
    # trained tensor may be replaced or widened to recover numerical parity.
    for layer, original in zip(restored.transformer.h, original_tables):
        layer.attn.embed_positions = original.clone()
    model.float()
    restored.float()
    with torch.inference_mode():
        torch.testing.assert_close(model(ids, position_ids=positions, use_cache=False).logits,
                                   restored(ids, position_ids=positions, use_cache=False).logits, rtol=0, atol=0)
    model.to(device="cpu", dtype=dtype)
    for layer, original in zip(model.transformer.h, original_tables):
        torch.testing.assert_close(layer.attn.embed_positions, original, rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(model(ids, position_ids=positions, use_cache=False).logits, before, rtol=0, atol=0)
    model.save_pretrained(tmp_path / "second")
    again = CodeGenQModel.loader.from_pretrained(tmp_path / "second", dtype=dtype).eval()
    with torch.inference_mode():
        torch.testing.assert_close(again(ids, position_ids=positions, use_cache=False).logits, before, rtol=0, atol=0)
    assert not list(tmp_path.rglob("*.bin"))


@pytest.mark.parametrize("partitions", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_dtype_construction_uses_original_fp32_rotary_math(partitions, dtype):
    config = tiny_config(partitions, n_embd=128, n_inner=192, rotary_dim=16,
                         n_positions=2048, n_ctx=2048, vocab_size=47, initializer_range=.1)
    # Native formula evaluated outside the low-default-dtype HF constructor.
    original = create_sinusoidal_positions(2048, 16)
    assert original.dtype == torch.float32
    model = CodeGenForCausalLM._from_config(config, dtype=dtype).eval()
    ids = torch.tensor([[18, 14, 17]])
    positions = torch.tensor([[1900, 1901, 1902]])
    with torch.inference_mode():
        actual = model(ids, position_ids=positions, use_cache=False).logits
        for layer in model.transformer.h:
            torch.testing.assert_close(layer.attn.embed_positions, original, rtol=0, atol=0)
            layer.attn.embed_positions = original.clone()
        expected = model(ids, position_ids=positions, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("partitions", [4, 8])
@pytest.mark.parametrize("assign", [False, True])
def test_meta_move_materialization_rebuilds_fp32_rotary(partitions, assign):
    torch.manual_seed(2027)
    model = CodeGenForCausalLM(tiny_config(partitions, n_positions=2048)).eval().bfloat16()
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    original = [layer.attn.embed_positions.clone() for layer in model.transformer.h]
    ids = torch.tensor([[18, 14, 17]])
    positions = torch.tensor([[1900, 1901, 1902]])
    with torch.inference_mode():
        before = model(ids, position_ids=positions, use_cache=False).logits
    model.to("meta")
    assert all(layer.attn.embed_positions.is_meta for layer in model.transformer.h)
    model.to_empty(device="cpu")
    model.load_state_dict(state, assign=assign, strict=True)
    for layer, table in zip(model.transformer.h, original):
        assert layer.attn.embed_positions.device.type == "cpu"
        torch.testing.assert_close(layer.attn.embed_positions, table, rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(model(ids, position_ids=positions, use_cache=False).logits, before, rtol=0, atol=0)
    assert all("embed_positions" not in key for key in model.state_dict())


@pytest.mark.parametrize("partitions", [4, 8])
def test_failed_dtype_conversion_preserves_rotary_and_retry(partitions):
    model = CodeGenForCausalLM(tiny_config(partitions)).eval()
    original = [layer.attn.embed_positions.clone() for layer in model.transformer.h]
    tables_seen = []

    def interrupted_cast(tensor):
        if tensor.shape == original[0].shape:
            tables_seen.append(tensor)
            if len(tables_seen) == 2:
                raise RuntimeError("injected failure after first rotary table cast")
        return tensor.bfloat16() if tensor.is_floating_point() else tensor

    with pytest.raises(RuntimeError, match="injected failure"):
        model._apply(interrupted_cast)
    assert len(tables_seen) == 2
    for layer, table in zip(model.transformer.h, original):
        torch.testing.assert_close(layer.attn.embed_positions, table, rtol=0, atol=0)
    model.bfloat16()
    state = model.state_dict()
    reference = CodeGenForCausalLM._from_config(tiny_config(partitions), dtype=torch.bfloat16).eval()
    reference.load_state_dict(state, strict=True)
    ids = torch.tensor([[1, 7, 4, 9]])
    with torch.inference_mode():
        torch.testing.assert_close(model(ids, use_cache=False).logits, reference(ids, use_cache=False).logits, rtol=0, atol=0)


@pytest.mark.parametrize("partitions", [4, 8])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_load_ignores_checkpoint_python(tmp_path, monkeypatch, partitions, trust, bypass):
    reference, marker = write_checkpoint(tmp_path, partitions)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")  # Existing security control, not a behavior switch.
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config) is CodeGenConfig
    assert resolve_trust_remote_code(tmp_path, trust_remote_code=trust) is False
    loaded = GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert type(loaded) is CodeGenQModel
    assert type(loaded.model) is CodeGenForCausalLM
    ids = torch.tensor([[1, 4, 8, 3]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids, use_cache=False).logits,
                                   reference(ids, use_cache=False).logits, rtol=0, atol=0)
    assert not marker.exists()


def test_legacy_causal_mask_is_explicitly_discarded(tmp_path):
    model = CodeGenForCausalLM(tiny_config()).eval()
    model.config.save_pretrained(tmp_path)
    state = dict(model.state_dict())
    for i in range(model.config.n_layer):
        state[f"transformer.h.{i}.attn.causal_mask"] = torch.ones(1, 1, 32, 32, dtype=torch.bool).tril()
    save_file(state, tmp_path / "model.safetensors", metadata={"format": "pt"})
    restored, info = CodeGenForCausalLM.from_pretrained(tmp_path, output_loading_info=True)
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    ids = torch.tensor([[1, 7, 3]])
    with torch.inference_mode():
        torch.testing.assert_close(restored(ids).logits, model(ids).logits, rtol=0, atol=0)


def test_codegen2_scales_scores_before_additive_bias():
    torch.manual_seed(17)
    model = CodeGenForCausalLM(tiny_config()).eval()
    ids = torch.tensor([[1, 6, 4, 5, 8], [1, 7, 9, 10, 2]])
    bias = torch.randn(2, 1, 5, 5) * .7
    bias.masked_fill_(torch.ones(5, 5, dtype=torch.bool).triu(1), torch.finfo(torch.float32).min)
    with torch.inference_mode():
        torch.testing.assert_close(model(ids, attention_mask=bias, use_cache=False).logits,
                                   oracle_logits(model, ids, 8, bias), rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("partitions", [4, 8])
def test_padded_explicit_positions_cache_and_fresh_request(partitions):
    model = CodeGenForCausalLM(tiny_config(partitions)).eval()
    ids = torch.tensor([[0, 0, 1, 5, 8], [1, 4, 6, 3, 7]])
    mask = ids.ne(0).long()
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    with torch.inference_mode():
        expected = model(ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
        prefill = model(ids[:, :3], attention_mask=mask[:, :3], position_ids=positions[:, :3], use_cache=True)
        actual = model(ids[:, 3:], attention_mask=mask, position_ids=positions[:, 3:],
                       past_key_values=prefill.past_key_values, use_cache=True).logits
        torch.testing.assert_close(actual, expected[:, 3:], rtol=2e-5, atol=2e-7)
        # A caller-owned warm cache must never leak into a fresh model request.
        torch.testing.assert_close(model(ids, attention_mask=mask, position_ids=positions, use_cache=False).logits,
                                   expected, rtol=0, atol=0)
