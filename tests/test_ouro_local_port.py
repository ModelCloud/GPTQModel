# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Ouro's shared-depth decoder must never dispatch to checkpoint Python."""

import json

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral
from torch import nn
from torch.nn import functional as F

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.models.definitions.ouro import OuroQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.ouro import OuroConfig, OuroForCausalLM, UniversalTransformerCache
from gptqmodel.utils.hf import load_model_config


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_config(**kwargs):
    args = {
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "total_ut_steps": 4,
        "max_position_embeddings": 32,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    args.update(kwargs)
    return OuroConfig(**args)


def tiny_model(**kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        model = OuroForCausalLM(tiny_config(**kwargs)).eval()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "layernorm" in name or name == "model.norm.weight":
                    param.uniform_(0.7, 1.3)
                else:
                    param.normal_(0, 0.11)
            model.model.early_exit_gate.bias.fill_(0.4)
    return model


def oracle(model, ids, mask=None, positions=None):
    """Independent released equations: float64 math, no port layers/cache/helpers."""
    c = model.config
    s = {k: v.detach().double() for k, v in model.state_dict().items()}
    x = F.embedding(ids, s["model.embed_tokens.weight"])
    batch, length = ids.shape
    if positions is None:
        positions = torch.arange(length).expand(batch, -1)
    inv = c.rope_parameters["rope_theta"] ** (-torch.arange(0, c.head_dim, 2).double() / c.head_dim)
    angle = positions.double()[..., None] * inv
    angle = torch.cat((angle, angle), -1)[:, None]

    def norm(y, name):
        return y / (y.square().mean(-1, keepdim=True) + c.rms_norm_eps).sqrt() * s[name + ".weight"]

    def linear(y, name):
        return F.linear(y, s[name + ".weight"])

    all_hidden, all_gates, all_kv = [], [], []
    for _ in range(c.total_ut_steps):
        for i in range(c.num_hidden_layers):
            prefix = f"model.layers.{i}."
            n = norm(x, prefix + "input_layernorm")
            q = (
                linear(n, prefix + "self_attn.q_proj")
                .view(batch, length, c.num_attention_heads, c.head_dim)
                .transpose(1, 2)
            )
            k = (
                linear(n, prefix + "self_attn.k_proj")
                .view(batch, length, c.num_key_value_heads, c.head_dim)
                .transpose(1, 2)
            )
            v = (
                linear(n, prefix + "self_attn.v_proj")
                .view(batch, length, c.num_key_value_heads, c.head_dim)
                .transpose(1, 2)
            )

            def rotate(y):
                half = y.shape[-1] // 2
                return y * angle.cos() + torch.cat((-y[..., half:], y[..., :half]), -1) * angle.sin()

            q, k = rotate(q), rotate(k)
            all_kv.append((k, v))
            kv = c.num_attention_heads // c.num_key_value_heads
            k, v = k.repeat_interleave(kv, 1), v.repeat_interleave(kv, 1)
            rows = torch.arange(length)[:, None]
            cols = torch.arange(length)[None, :]
            valid = (cols <= rows)[None, None].expand(batch, 1, -1, -1)
            if mask is not None:
                valid = valid & mask[:, None, None, :].bool()
            if c.layer_types[i] == "sliding_attention":
                valid = valid & ((rows - cols) < c.sliding_window)[None, None]
            scores = (q @ k.transpose(-1, -2)) / c.head_dim**0.5
            probs = scores.masked_fill(~valid, torch.finfo(torch.float64).min).softmax(-1)
            attention = (probs @ v).transpose(1, 2).reshape(batch, length, -1)
            x = x + norm(linear(attention, prefix + "self_attn.o_proj"), prefix + "input_layernorm_2")
            n = norm(x, prefix + "post_attention_layernorm")
            mlp = linear(
                F.silu(linear(n, prefix + "mlp.gate_proj")) * linear(n, prefix + "mlp.up_proj"),
                prefix + "mlp.down_proj",
            )
            x = x + norm(mlp, prefix + "post_attention_layernorm_2")
        x = norm(x, "model.norm")
        all_hidden.append(x)
        all_gates.append(F.linear(x, s["model.early_exit_gate.weight"], s["model.early_exit_gate.bias"]).squeeze(-1))
    survival = torch.ones(batch, length, dtype=torch.float64)
    pdf = []
    for i, gate in enumerate(all_gates):
        pdf.append(survival * gate.sigmoid() if i < len(all_gates) - 1 else survival)
        survival = survival * (1 - gate.sigmoid())
    logits = [linear(h, "lm_head") for h in all_hidden]
    return all_hidden, logits, torch.stack(pdf, -1), all_kv


def test_definition_owns_loader_without_remote_hooks():
    assert not OuroQModel.require_trust_remote_code
    assert isinstance(OuroQModel.loader, LocalModelLoader)
    assert not getattr(OuroQModel, "before_model_load_requires_remote_code", False)
    assert "before_model_load" not in OuroQModel.__dict__


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("steps", [1, 3, 4])
@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_recurrent_decoder_matches_independent_equations(kv_heads, steps, sliding, backend):
    model = tiny_model(
        num_key_value_heads=kv_heads,
        total_ut_steps=steps,
        use_sliding_window=sliding,
        sliding_window=3,
        max_window_layers=1,
        attn_implementation=backend,
    )
    ids = torch.tensor([[4, 5, 6, 7, 8, 9], [9, 7, 8, 6, 5, 4]])
    _, expected, _, cache_expected = oracle(model, ids)
    with torch.inference_mode():
        # TF5 attention interface: sdpa/eager dispatch with per-layer sliding windows.
        actual = model(ids, output_hidden_states=True)
        torch.testing.assert_close(actual.logits.double(), expected[-1], atol=3e-5, rtol=2e-4)
        # Regression: sdpa boolean masks were once added to eager scores here.
        actual = model(ids, output_hidden_states=True, output_attentions=True)
    torch.testing.assert_close(actual.logits.double(), expected[-1], atol=3e-5, rtol=2e-4)
    # Captured per layer call: the embeddings plus every recurrent layer output (the last one normalized).
    assert len(actual.hidden_states) == steps * 2 + 1
    # sdpa returns no attention weights, so nothing is captured for it.
    assert len(actual.attentions) == (0 if backend == "sdpa" else steps * 2)
    assert len(actual.past_key_values.layers) == steps * 2
    for layer, (k, v) in zip(actual.past_key_values.layers, cache_expected):
        torch.testing.assert_close(layer.keys.double(), k, atol=3e-5, rtol=2e-4)
        torch.testing.assert_close(layer.values.double(), v, atol=3e-5, rtol=2e-4)


@pytest.mark.parametrize(
    "exit_mode", ["step0", "step2", "threshold", "weighted", "invalid_step_weighted", "labels", "default"]
)
def test_source_tokenwise_exit_precedence_and_loss(exit_mode):
    model = tiny_model(early_exit_threshold=None)
    ids = torch.tensor([[4, 5, 8, 6], [9, 3, 7, 4]])
    _, per_step, pdf, _ = oracle(model, ids)
    kwargs = {}
    if exit_mode.startswith("step"):
        step = int(exit_mode[-1])
        kwargs = {"exit_at_step": step, "exit_threshold": 0.2, "use_weighted_exit": True}
        expected = per_step[step]
    elif exit_mode == "threshold":
        kwargs = {"exit_threshold": 0.7, "use_weighted_exit": True}
        steps = (pdf.cumsum(-1) >= 0.7).double().argmax(-1)
        assert steps.unique().numel() > 1, "adversarial gate must select different depths per token"
        expected = torch.stack(per_step, 2).gather(2, steps[..., None, None].expand(-1, -1, 1, 32)).squeeze(2)
    elif exit_mode in ("weighted", "invalid_step_weighted", "labels"):
        kwargs = {"labels": ids, "exit_at_step": 0} if exit_mode == "labels" else {"use_weighted_exit": True}
        if exit_mode == "invalid_step_weighted":
            kwargs["exit_at_step"] = 999
        expected = sum(logits * pdf[..., i, None] for i, logits in enumerate(per_step))
    else:
        expected = per_step[-1]
    with torch.inference_mode():
        out = model(ids, use_cache=False, **kwargs)
    torch.testing.assert_close(out.logits.double(), expected, atol=3e-5, rtol=2e-4)
    if exit_mode == "labels":
        loss = F.cross_entropy(expected[:, :-1].reshape(-1, 32), ids[:, 1:].flatten())
        torch.testing.assert_close(out.loss.double(), loss, atol=3e-5, rtol=2e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_padded_cache_continuation_generation_and_safetensors(tmp_path, dtype, backend):
    model = tiny_model(attn_implementation=backend).to(dtype)
    ids = torch.tensor([[0, 0, 4, 5, 6, 8, 9], [7, 3, 9, 4, 6, 5, 8]])
    mask = (ids != 0).long()
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 1)
    with torch.inference_mode():
        reference = model(ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
        cache = None
        parts = []
        offset = 0
        for size in [1, 4, 1, 1]:
            end = offset + size
            out = model(
                ids[:, offset:end],
                attention_mask=mask[:, :end],
                position_ids=positions[:, offset:end],
                past_key_values=cache,
            )
            cache = out.past_key_values
            parts.append(out.logits)
            assert all(cache.get_seq_length(i) == end for i in range(8))
            offset = end
        tolerance = 0.04 if dtype == torch.bfloat16 else 0.006 if dtype == torch.float16 else 3e-5
        actual = torch.cat(parts, dim=1)
        torch.testing.assert_close(actual[mask.bool()], reference[mask.bool()], atol=tolerance, rtol=tolerance)
        greedy = model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False)
        beam = model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False, num_beams=2)
        assert greedy.shape == beam.shape == (2, 10)
        manual = ids.clone()
        manual_mask = mask.clone()
        for _ in range(3):
            p = manual_mask.cumsum(-1) - 1
            p.masked_fill_(manual_mask == 0, 1)
            token = model(manual, attention_mask=manual_mask, position_ids=p, use_cache=False).logits[:, -1].argmax(-1)
            manual = torch.cat((manual, token[:, None]), 1)
            manual_mask = torch.cat((manual_mask, torch.ones(2, 1, dtype=mask.dtype)), 1)
        assert torch.equal(greedy, manual)
        for cycle in range(2):
            path = tmp_path / str(cycle)
            model.save_pretrained(path, max_shard_size="20KB")
            assert list(path.glob("*.safetensors")) and not list(path.glob("*.bin"))
            loaded = OuroQModel.loader.from_pretrained(path, dtype=dtype, attn_implementation=backend).eval()
            for key, value in model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
            torch.testing.assert_close(
                loaded(ids, attention_mask=mask, position_ids=positions, use_cache=False).logits,
                reference,
                atol=tolerance,
                rtol=tolerance,
            )
            assert torch.equal(loaded.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False), greedy)
            model = loaded


def write_checkpoint(path, **config_kwargs):
    path.mkdir(parents=True, exist_ok=True)
    model = tiny_model(**config_kwargs)
    model.save_pretrained(path)
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(
        WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "a": 4, "b": 5, "c": 6, "d": 7}, unk_token="[UNK]")
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]"
    ).save_pretrained(path)
    cfg = json.loads((path / "config.json").read_text())
    cfg["auto_map"] = {"AutoConfig": "hostile.Bad", "AutoModelForCausalLM": "hostile.Bad"}
    (path / "config.json").write_text(json.dumps(cfg))
    tokenizer_cfg = json.loads((path / "tokenizer_config.json").read_text())
    tokenizer_cfg["auto_map"] = {"AutoTokenizer": [None, "hostile.Bad"]}
    (path / "tokenizer_config.json").write_text(json.dumps(tokenizer_cfg))
    marker = path / "EXECUTED"
    (path / "hostile.py").write_text(
        f'from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError("checkpoint Python")\n'
    )
    return model, marker


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_owned_loader_hostile_metadata_and_same_path_retry(tmp_path, monkeypatch, trust, bypass):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    source, marker = write_checkpoint(tmp_path)
    c = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(c) is OuroConfig
    ids = torch.tensor([[4, 5, 6, 7]])
    with torch.inference_mode():
        expected = source(ids, use_cache=False).logits
    loaded = GPTQModel.load(
        str(tmp_path),
        quantize_config=None,
        trust_remote_code=trust,
        device="cpu",
        dtype=torch.float32,
        backend=BACKEND.TORCH,
    )
    assert type(loaded.model) is OuroForCausalLM
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids, use_cache=False).logits, expected, atol=1e-6, rtol=1e-6)
        assert loaded.generate(ids, max_new_tokens=2).shape == (1, 6)
    assert not marker.exists()
    payload = tmp_path / "model.safetensors"
    original = payload.read_bytes()
    payload.write_bytes(b"broken safetensors")
    with pytest.raises(Exception, match="[Hh]eader|[Ss]afetensor|deserializ"):
        OuroQModel.loader.from_pretrained(tmp_path, trust_remote_code=trust)
    payload.write_bytes(original)
    retry = OuroQModel.loader.from_pretrained(tmp_path, trust_remote_code=trust).eval()
    with torch.inference_mode():
        torch.testing.assert_close(retry(ids, use_cache=False).logits, expected, atol=1e-6, rtol=1e-6)
    assert not marker.exists()


def test_cache_failure_clear_rebuild_and_depth_contract(monkeypatch):
    model = tiny_model()
    ids = torch.tensor([[4, 5, 6]])
    cache = UniversalTransformerCache(8)
    layer = model.model.layers[1]
    original = layer.forward
    fired = []

    def fail(*args, **kwargs):
        fired.append(True)
        raise RuntimeError("injected after slot zero")

    monkeypatch.setattr(layer, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        model(ids, past_key_values=cache)
    assert fired and cache.get_seq_length() == 3
    monkeypatch.setattr(layer, "forward", original)
    with pytest.raises(ValueError, match="Incomplete"):
        model(ids, past_key_values=cache)
    cache.clear()
    with torch.inference_mode():
        retried = model(ids, past_key_values=cache).logits
        torch.testing.assert_close(retried, model(ids, use_cache=False).logits)
    with pytest.raises(ValueError, match="depth"):
        model(ids, past_key_values=UniversalTransformerCache(6))
    with pytest.raises(ValueError, match="dynamic recurrent"):
        model(ids, past_key_values=object())
    with pytest.raises(ValueError, match="use_cache"):
        model(ids, past_key_values=cache, use_cache=False)


@pytest.mark.parametrize("configured", [False, True])
def test_source_native_dynamic_cache_input_and_continuation(configured):
    from transformers.cache_utils import DynamicCache

    model = tiny_model()
    ids = torch.tensor([[4, 5, 6, 7, 8]])
    supplied = DynamicCache(config=model.config) if configured else DynamicCache()
    with torch.inference_mode():
        output = model(ids[:, :3], past_key_values=supplied)
        if not configured:
            assert output.past_key_values is supplied
        else:
            assert type(output.past_key_values) is UniversalTransformerCache
        continuation = model(ids[:, 3:], past_key_values=output.past_key_values)
        reference = model(ids, use_cache=False)
        torch.testing.assert_close(continuation.logits, reference.logits[:, 3:], atol=3e-5, rtol=2e-4)
        assert all(continuation.past_key_values.get_seq_length(i) == 5 for i in range(8))


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_weight_assignment_logits_slices_and_embedding_inputs(assign, tied):
    model = tiny_model(tie_word_embeddings=tied)
    replacement = tiny_model(tie_word_embeddings=tied)
    replacement.load_state_dict(model.state_dict(), assign=assign)
    ids = torch.tensor([[4, 5, 6]])
    with torch.inference_mode():
        expected = model(ids, use_cache=False).logits
        torch.testing.assert_close(replacement(ids, use_cache=False).logits, expected)
        torch.testing.assert_close(
            model(inputs_embeds=model.model.embed_tokens(ids), use_cache=False).logits, expected
        )
        torch.testing.assert_close(model(ids, use_cache=False, logits_to_keep=1).logits, expected[:, -1:])
        torch.testing.assert_close(
            model(ids, use_cache=False, logits_to_keep=torch.tensor([0, 2])).logits, expected[:, [0, 2]]
        )
        assert isinstance(model(ids, use_cache=False, return_dict=False), tuple)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_source_full_head_rope_near_released_context_after_load(tmp_path, dtype):
    model = tiny_model(head_dim=128, max_position_embeddings=65536)
    model.save_pretrained(tmp_path)
    loaded = OuroQModel.loader.from_pretrained(tmp_path, dtype=dtype).eval()
    rotary = loaded.model.rotary_emb
    inv = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
    assert rotary.inv_freq.dtype == torch.float32
    torch.testing.assert_close(rotary.inv_freq, inv, rtol=0, atol=0)
    positions = torch.tensor([[65529, 65530, 65531], [65532, 65533, 65534]])
    angles = (inv[None, :, None] @ positions[:, None, :].float()).transpose(1, 2)
    angles = torch.cat((angles, angles), -1)
    with torch.inference_mode():
        actual = rotary(torch.zeros(2, 3, 32, dtype=dtype), positions)
    torch.testing.assert_close(actual[0], angles.cos().to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(actual[1], angles.sin().to(dtype), rtol=0, atol=0)


def test_config_output_defaults_and_explicit_overrides():
    model = tiny_model(output_hidden_states=True, output_attentions=True, return_dict=False)
    ids = torch.tensor([[4, 5, 6]])
    with torch.inference_mode():
        assert isinstance(model(ids, use_cache=False), tuple)
        output = model(ids, use_cache=False, return_dict=True)
        assert len(output.hidden_states) == 9
        assert len(output.attentions) == (8 if model.config._attn_implementation == "eager" else 0)
        output = model(ids, use_cache=False, return_dict=True, output_hidden_states=False, output_attentions=False)
        assert output.hidden_states is None and output.attentions is None


@pytest.mark.parametrize("kv_heads", [2, 4])
def test_awq_folds_preserve_recurrent_output_across_all_loops(kv_heads):
    model = tiny_model(num_key_value_heads=kv_heads)
    ids = torch.tensor([[4, 5, 6, 7, 6, 5]])

    def forward():
        # Every shared layer runs `total_ut_steps` times, including the early
        # exit gate between steps; a fold must be exact at every depth.
        return model(ids, use_cache=False).logits

    expected = [
        ("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
        ("self_attn.v_proj", ["self_attn.o_proj"]),
        ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
        ("mlp.up_proj", ["mlp.down_proj"]),
    ]
    if kv_heads != 4:
        # GQA: the value width differs from `o_proj`'s input.
        expected.remove(("self_attn.v_proj", ["self_attn.o_proj"]))
    for layer in model.model.layers:
        checked = assert_awq_folds_neutral(OuroQModel, model, layer, forward)
        # The sandwich post-norms `input_layernorm_2` / `post_attention_layernorm_2`
        # normalize a branch output before the residual add and never fold.
        assert [(source, targets) for source, targets, _consumers in checked] == expected


def test_loop_capture_all_depths():
    model = tiny_model()
    wrapper = OuroQModel.__new__(OuroQModel)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    calibration = [{"input_ids": torch.tensor([[4, 5, 6, 7]])}]
    bucket = wrapper._capture_looped_layer_invocations(model.model.layers, [], calibration, torch.device("cpu"))
    for layer in model.model.layers:
        captures = bucket[id(layer)]
        assert len(captures["layer_inputs"]) == 4
        assert not torch.equal(captures["layer_inputs"][0][0], captures["layer_inputs"][1][0])
        assert not layer._forward_pre_hooks
        with torch.inference_mode():
            for i, ((x,), kwargs) in enumerate(zip(captures["layer_inputs"], captures["layer_input_kwargs"])):
                assert kwargs["current_ut"] == i
                assert kwargs["position_embeddings"][0].shape == (1, 4, 8)
                output = layer(x, attention_mask=captures["attention_masks"][i], **kwargs)[0]
                assert torch.isfinite(output).all()


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("offload", [False, True])
def test_public_awq_all_reused_linears_two_exact_reloads(tmp_path, kv_heads, offload):
    from safetensors.torch import load_file

    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    source_dir = tmp_path / "source"
    source, marker = write_checkpoint(source_dir, num_key_value_heads=kv_heads)
    qcfg = AWQConfig(
        device="cpu",
        group_size=16,
        fallback=None,
        offload_to_disk=offload,
        offload_to_disk_path=str(tmp_path / "offload") if offload else None,
    )
    loaded = GPTQModel.load(str(source_dir), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16)
    assert (loaded.turtle_model is not None) is offload
    assert any(parameter.is_meta for parameter in loaded.model.parameters()) is offload
    calibration = [
        {"input_ids": torch.tensor([[4, 5, 6, 7, 6, 5, 4, 7]]), "attention_mask": torch.ones(1, 8, dtype=torch.long)}
    ]
    loaded.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    expected_names = {
        f"model.layers.{i}.{name}"
        for i in range(2)
        for name in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    }
    actual_names = {name for name, module in loaded.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert actual_names == expected_names
    previous_state = None
    previous_logits = None
    for cycle in range(2):
        path = tmp_path / f"quantized-{cycle}"
        loaded.save(str(path))
        state = {}
        for payload in path.glob("*.safetensors"):
            state.update(load_file(payload))
        assert state and not list(path.glob("*.bin"))
        for name in expected_names:
            assert {name + "." + key for key in ("qweight", "qzeros", "scales")} <= state.keys()
        for name, value in source.state_dict().items():
            if any(name.startswith(prefix + ".") for prefix in expected_names):
                continue
            if name.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")):
                # AWQ folds its searched activation scales into both pre-norms.
                assert not torch.equal(state[name], value.half()), name
            else:
                torch.testing.assert_close(state[name], value.half(), rtol=0, atol=0)
        if previous_state is not None:
            assert state.keys() == previous_state.keys()
            for key in state:
                torch.testing.assert_close(state[key], previous_state[key], rtol=0, atol=0)
        loaded = GPTQModel.load(str(path), device="cpu", backend=BACKEND.TORCH, dtype=torch.float16)
        assert type(loaded.model) is OuroForCausalLM
        with torch.inference_mode():
            logits = loaded.model(**calibration[0], use_cache=False).logits
            assert torch.isfinite(logits).all()
            assert loaded.generate(calibration[0]["input_ids"], max_new_tokens=2).shape == (1, 10)
            if previous_logits is not None:
                torch.testing.assert_close(logits, previous_logits, rtol=0, atol=0)
        previous_logits = logits.clone()
        previous_state = state
    assert not marker.exists()
