# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Spark2.5 real-model qualification; source oracle is independent of the port."""

import copy
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, GenerationConfig

from gptqmodel.models.definitions.spark2_5 import Spark2_5QModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.spark2_5.configuration_spark import Spark2_5Config
from gptqmodel.models.ports.spark2_5.modeling_spark import Spark2_5ForCausalLM
from tests.port_assets import read_port_asset


FIXTURES = Path(__file__).parents[1] / "gptqmodel/models/ports/spark2_5/fixtures"


def test_real_shared_mlp_inputs_and_excluded_head_gate():
    model = tiny_model()
    seen = {}
    handles = []
    for index, layer in enumerate(model.model.layers):
        for name in ["mlp.gate_proj", "mlp.up_proj", "self_attn.q_k_v_proj", "self_attn.g_proj"]:
            key = (index, name)

            def capture(module, args, key=key):
                seen[key] = args[0]

            handles.append(layer.get_submodule(name).register_forward_pre_hook(capture))
    try:
        with torch.inference_mode():
            model(torch.tensor([[3, 4, 5]]), use_cache=False)
        for index in range(2):
            assert seen[index, "mlp.gate_proj"] is seen[index, "mlp.up_proj"]
            assert seen[index, "self_attn.q_k_v_proj"] is seen[index, "self_attn.g_proj"]
    finally:
        for handle in handles:
            handle.remove()


@pytest.fixture(autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(**kwargs):
    values = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "headwise_attn_output_gate": True,
        "sliding_window": 3,
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000},
            "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
        },
        "pad_token_id": 2,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "attn_implementation": "eager",
    }
    values.update(kwargs)
    return Spark2_5Config(**values)


def tiny_model(**kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(451)
        model = Spark2_5ForCausalLM(tiny_config(**kwargs)).eval()
        # Nondegenerate weights expose gating, ordering, RoPE and residual errors.
        with torch.no_grad():
            for name, tensor in model.named_parameters():
                tensor.copy_(torch.randn_like(tensor) * 0.12 + (1.0 if "norm" in name else 0))
    return model


def source_oracle(weights, config, tokens, *, padding=None, offset=0):
    """Independent float64 equations of the inspected pinned Spark eager source.

    No port mask, rotary, attention, decoder, or cache helpers. Upstream rotates
    by absolute cache slots (not per-row position_ids) and keeps FP32 residuals.
    Phase arithmetic stays FP32 as in the released implementation.
    """
    hidden = F.embedding(tokens, weights["model.embedding.weight"].double())
    batch, length = tokens.shape
    heads, kv_heads, dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim

    def linear(x, name):
        bias = weights.get(name + ".bias")
        return F.linear(x, weights[name + ".weight"].double(), None if bias is None else bias.double())

    def norm(x, name):
        return x * (x.square().mean(-1, keepdim=True) + config.rms_norm_eps).rsqrt() * weights[name].double()

    for index, layer_type in enumerate(config.layer_types):
        prefix = f"model.layers.{index}."
        x = norm(hidden, prefix + "input_layernorm.weight")
        packed = linear(x, prefix + "self_attn.q_k_v_proj")
        q, k, v = packed.split([heads * dim, kv_heads * dim, kv_heads * dim], -1)
        q = q.reshape(batch, length, heads, dim).transpose(1, 2)
        k = k.reshape(batch, length, kv_heads, dim).transpose(1, 2)
        v = v.reshape(batch, length, kv_heads, dim).transpose(1, 2)
        params = config.rope_parameters.get(layer_type, {})
        rotary = int(dim * params.get("partial_rotary_factor", 1.0))
        inverse = 1.0 / (params.get("rope_theta", 10000) ** (torch.arange(0, rotary, 2).float() / rotary))
        phase = (torch.arange(offset, offset + length).float()[:, None] * inverse[None, :]).double()

        def rotate(tensor):
            a, b = tensor[..., : rotary // 2], tensor[..., rotary // 2 : rotary]
            cosine, sine = phase.cos(), phase.sin()
            return torch.cat([a * cosine - b * sine, b * cosine + a * sine, tensor[..., rotary:]], -1)

        # Match the source's FP32 trigonometric evaluation, independently.
        phase = phase.float()
        q, k = rotate(q), rotate(k)
        k = torch.repeat_interleave(k, heads // kv_heads, dim=1)
        v = torch.repeat_interleave(v, heads // kv_heads, dim=1)
        scores = q @ k.transpose(-1, -2) / dim**0.5
        row, col = torch.arange(length)[:, None], torch.arange(length)[None, :]
        allowed = col <= row
        if layer_type == "sliding_attention":
            allowed &= col > row - config.sliding_window
        allowed = allowed[None, None].expand(batch, 1, length, length)
        if padding is not None:
            allowed = allowed & padding[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, torch.finfo(torch.float32).min)
        output = scores.softmax(-1) @ v
        if config.headwise_attn_output_gate:
            gate = linear(x, prefix + "self_attn.g_proj").transpose(1, 2).unsqueeze(-1)
            gate = gate.sigmoid() if config.gate_attn_act_mode == "sigmoid" else F.silu(gate)
            output *= gate
        hidden = hidden + linear(
            output.transpose(1, 2).reshape(batch, length, heads * dim), prefix + "self_attn.out_proj"
        )
        x = norm(hidden, prefix + "post_attention_layernorm.weight")
        hidden += linear(
            F.gelu(linear(x, prefix + "mlp.gate_proj")) * linear(x, prefix + "mlp.up_proj"), prefix + "mlp.down_proj"
        )
    hidden = norm(hidden, "model.norm.weight")
    return F.linear(
        hidden, weights["model.embedding.weight" if config.tie_word_embeddings else "lm_head.weight"].double()
    )


def test_definition_owns_loading_and_never_requires_checkpoint_python():
    assert isinstance(Spark2_5QModel.loader, LocalModelLoader)
    assert Spark2_5QModel.require_trust_remote_code is False
    assert not getattr(Spark2_5QModel, "before_model_load_requires_remote_code", False)


@pytest.mark.parametrize(
    "heads,head_dim,bias,gate",
    [(1, 8, False, False), (2, 16, True, "sigmoid"), (4, 8, True, "silu"), (4, 16, False, "sigmoid")],
)
def test_real_decoder_matches_independent_source_equations(heads, head_dim, bias, gate):
    model = tiny_model(
        num_key_value_heads=heads,
        head_dim=head_dim,
        attention_bias=bias,
        mlp_bias=bias,
        headwise_attn_output_gate=bool(gate),
        gate_attn_act_mode=gate or "sigmoid",
    )
    ids = torch.tensor([[3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3]])
    padding = torch.ones_like(ids)
    padding[1, :2] = 0
    expected = source_oracle(model.state_dict(), model.config, ids, padding=padding)
    with torch.inference_mode():
        actual = model(ids, attention_mask=padding, use_cache=False).logits
    torch.testing.assert_close(actual[:, 2:].double(), expected[:, 2:], atol=3e-6, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize(
    "layout", [["sliding_attention", "full_attention"], ["full_attention"] * 2, ["sliding_attention"] * 2]
)
def test_dtype_hybrid_cache_continuation_generation_and_causality(dtype, implementation, layout):
    model = tiny_model(layer_types=layout, attn_implementation=implementation).to(dtype)
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3]])
    with torch.inference_mode():
        full = model(ids, use_cache=False).logits
        past = None
        pieces = []
        start = 0
        for count in [1, 4, 1, 1]:
            out = model(ids[:, start : start + count], past_key_values=past, use_cache=True)
            past = out.past_key_values
            start += count
            assert past.get_seq_length() == start
            pieces.append(out.logits)
        tolerance = {torch.float32: 2e-6, torch.float16: 0.003, torch.bfloat16: 0.025}[dtype]
        torch.testing.assert_close(torch.cat(pieces, 1), full, atol=tolerance, rtol=tolerance)
        changed = ids.clone()
        changed[:, 4:] = 11
        torch.testing.assert_close(model(changed, use_cache=False).logits[:, :4], full[:, :4], atol=0, rtol=0)
        for beams in [1, 3]:
            cached = model.generate(ids, max_new_tokens=3, do_sample=False, num_beams=beams)
            uncached = model.generate(ids, max_new_tokens=3, do_sample=False, num_beams=beams, use_cache=False)
            assert torch.equal(cached, uncached)


@pytest.mark.parametrize("head_dim", [8, 256])
def test_source_absolute_slot_partial_rope_near_released_context_boundary(head_dim):
    model = tiny_model(head_dim=head_dim)
    ids = torch.tensor([[3, 4, 5]])
    positions = torch.arange(1048570, 1048573)
    expected = source_oracle(model.state_dict(), model.config, ids, offset=1048570)
    with torch.inference_mode():
        output = model(ids, cache_position=positions, use_cache=False).logits
    torch.testing.assert_close(output.double(), expected, atol=4e-6, rtol=3e-5)


def write_tokenizer(path, real=False):
    if real:
        for asset in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]:
            (path / asset).write_bytes(read_port_asset(FIXTURES / asset))
    else:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast

        tokenizer = Tokenizer(
            WordLevel(
                {"[BOS]": 0, "[EOS]": 1, "[PAD]": 2, "[UNK]": 3, "a": 4, "b": 5, "c": 6, "d": 7}, unk_token="[UNK]"
            )
        )
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]", unk_token="[UNK]"
        ).save_pretrained(path)


def hostile_checkpoint(path):
    marker = path / "EXECUTED"
    config = json.loads((path / "config.json").read_text())
    config["auto_map"] = {"AutoConfig": "hostile.Config", "AutoModelForCausalLM": "hostile.Model"}
    (path / "config.json").write_text(json.dumps(config))
    tokenizer = json.loads((path / "tokenizer_config.json").read_text())
    tokenizer["auto_map"] = {"AutoTokenizer": ["hostile.Tokenizer", "hostile.Tokenizer"]}
    (path / "tokenizer_config.json").write_text(json.dumps(tokenizer))
    (path / "custom_generate").mkdir()
    payload = f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python ran')\n"
    (path / "hostile.py").write_text(payload)
    (path / "custom_generate/generate.py").write_text(payload)
    return marker


@pytest.mark.parametrize("trust,bypass", [(False, None), (True, "1")])
def test_public_hostile_load_real_tokenizer_generation_policy_and_two_reloads(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel import GPTQModel

    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    model = tiny_model(vocab_size=131072, tie_word_embeddings=True)
    model.save_pretrained(tmp_path)
    write_tokenizer(tmp_path, real=True)
    policy = GenerationConfig.from_dict(json.loads(read_port_asset(FIXTURES / "generation_config.json")))
    policy.save_pretrained(tmp_path)
    marker = hostile_checkpoint(tmp_path)
    raw = AutoTokenizer.from_pretrained(tmp_path, trust_remote_code=False)
    text = "Hello 世界!  a\n  b"
    ids = raw(text, return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        expected = model(ids, use_cache=False).logits
    for cycle in range(3):
        loaded = GPTQModel.load(
            str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust, attn_implementation="eager"
        )
        assert isinstance(loaded, Spark2_5QModel)
        assert type(loaded.model) is Spark2_5ForCausalLM
        assert loaded.trust_remote_code is False
        assert loaded.model.lm_head.weight is loaded.model.model.embedding.weight
        assert loaded.model.generation_config.do_sample is True
        assert loaded.model.generation_config.eos_token_id == 1
        assert loaded.tokenizer(text)["input_ids"] == raw(text)["input_ids"]
        with torch.inference_mode():
            torch.testing.assert_close(loaded.model(ids, use_cache=False).logits, expected, rtol=0, atol=0)
            generated = loaded.model.generate(ids, max_new_tokens=2, do_sample=False)
            assert generated.shape[1] > ids.shape[1]
        assert not marker.exists()
        if cycle < 2:
            saved = tmp_path / f"dense{cycle}"
            loaded.model.save_pretrained(saved)
            loaded.tokenizer.save_pretrained(saved)
            assert list(saved.glob("*.safetensors")) and not list(saved.glob("*.bin"))
            tmp_path = saved


@pytest.mark.parametrize("dtype,assign", [(torch.float32, True), (torch.float16, False), (torch.bfloat16, True)])
def test_state_assignment_safetensors_and_failed_load_repair_retry(tmp_path, dtype, assign):
    model = tiny_model(tie_word_embeddings=True).to(dtype)
    reference = {name: value.clone() for name, value in model.state_dict().items()}
    fresh = tiny_model(tie_word_embeddings=True).to(dtype)
    fresh.load_state_dict(reference, assign=assign)
    ids = torch.tensor([[3, 4, 5]])
    with torch.inference_mode():
        expected = model(ids, use_cache=False).logits
    for cycle in range(2):
        path = tmp_path / str(cycle)
        fresh.save_pretrained(path)
        tensor_file = next(path.glob("*.safetensors"))
        payload = tensor_file.read_bytes()
        tensor_file.write_bytes(b"truncated")
        with pytest.raises(Exception, match="header|Header"):
            Spark2_5QModel.loader.from_pretrained(path, dtype=dtype, attn_implementation="eager")
        tensor_file.write_bytes(payload)
        fresh = Spark2_5QModel.loader.from_pretrained(path, dtype=dtype, attn_implementation="eager").eval()
        for key, value in fresh.state_dict().items():
            torch.testing.assert_close(value, reference[key], atol=0, rtol=0)
        assert fresh.lm_head.weight is fresh.model.embedding.weight
        with torch.inference_mode():
            torch.testing.assert_close(fresh(ids, use_cache=False).logits, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"num_key_value_heads": 3}, "divisible"),
        ({"num_key_value_heads": 0}, "positive"),
        ({"layer_types": ["bogus"] * 2}, "layer_types"),
        ({"layer_types": ["full_attention"]}, "length"),
        ({"sliding_window": None}, "sliding_window"),
        ({"hidden_act": "silu"}, "gelu"),
        ({"gate_attn_act_mode": "softmax"}, "gate_attn"),
        ({"rope_parameters": {"full_attention": {"partial_rotary_factor": 0.125}}}, "rotary"),
        ({"rope_parameters": {"full_attention": {"rope_type": "yarn"}}}, "unscaled"),
    ],
)
def test_invalid_configuration_fails_before_model_mutation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        tiny_config(**kwargs)
    assert tiny_model()(torch.tensor([[3, 4]])).logits.isfinite().all()


def test_failed_forward_then_retry_inputs_embeds_and_independent_cache_ownership():
    model = tiny_model()
    ids = torch.tensor([[3, 4, 5]])
    with pytest.raises(ValueError, match="both"):
        model(ids, inputs_embeds=model.get_input_embeddings()(ids))
    with pytest.raises(TypeError, match="DynamicCache"):
        model(ids, past_key_values=((torch.zeros(1), torch.zeros(1)),))
    with torch.inference_mode():
        left = model(ids)
        right = model(ids[:, :1])
        assert left.past_key_values is not right.past_key_values
        assert left.past_key_values.get_seq_length() == 3
        assert right.past_key_values.get_seq_length() == 1
        torch.testing.assert_close(model(inputs_embeds=model.get_input_embeddings()(ids)).logits, left.logits)
        torch.testing.assert_close(model(ids, return_dict=False)[0], left.logits)


@pytest.mark.parametrize("gate,bias,kv_heads", [(True, False, 1), (True, True, 2), (False, False, 4)])
@pytest.mark.parametrize("offload", [False, True])
def test_real_public_awq_scaled_all_linears_and_two_exact_reloads(tmp_path, gate, bias, kv_heads, offload):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig, CalibrationConfig

    model = tiny_model(
        headwise_attn_output_gate=gate,
        attention_bias=bias,
        mlp_bias=bias,
        num_key_value_heads=kv_heads,
        tie_word_embeddings=True,
    )
    model.save_pretrained(tmp_path)
    write_tokenizer(tmp_path)
    config = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=offload)
    loaded = GPTQModel.load(str(tmp_path), quantize_config=config, backend=BACKEND.TORCH, dtype=torch.float16)
    excluded = {
        key: value.half().clone()
        for key, value in model.state_dict().items()
        if "norm" in key or ".g_proj." in key or key in {"model.embedding.weight", "lm_head.weight"}
    }
    untouched = {"model.embedding.weight", "lm_head.weight", "model.norm.weight"}
    calibration = [
        {"input_ids": torch.tensor([[4, 5, 6, 7, 4, 5, 6, 7]]), "attention_mask": torch.ones(1, 8, dtype=torch.long)}
        for _ in range(2)
    ]
    loaded.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1, sort="desc"), backend=BACKEND.TORCH
    )
    linears = {name for name, module in loaded.model.named_modules() if isinstance(module, AwqTorchLinear)}
    expected = {
        f"model.layers.{i}.{name}"
        for i in range(2)
        for name in ["self_attn.q_k_v_proj", "self_attn.out_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    }
    assert linears == expected
    # No placeholder dense .weight is installed on packed linears by a hook.
    for name in linears:
        assert not hasattr(loaded.model.get_submodule(name), "weight")
    state = {key: value.clone() for key, value in loaded.model.state_dict().items()}
    for key, value in excluded.items():
        # Excluded tensors stay dense in their source dtype and shape.
        assert state[key].dtype == value.dtype and state[key].shape == value.shape
        if key in untouched:
            torch.testing.assert_close(state[key], value, atol=0, rtol=0)
    # Scale search ran and folded into at least one decoder norm.
    assert any(not torch.equal(state[key], excluded[key]) for key in excluded if "layernorm" in key)
    for i in range(2):
        if gate:
            # AWQ folds the input-norm scale into the unquantized head gate:
            # gate columns are multiplied by exactly what the norm was divided by.
            prefix = f"model.layers.{i}."
            before = excluded[prefix + "self_attn.g_proj.weight"].float() * excluded[prefix + "input_layernorm.weight"].float()
            after = state[prefix + "self_attn.g_proj.weight"].float() * state[prefix + "input_layernorm.weight"].float()
            torch.testing.assert_close(after, before, rtol=2e-3, atol=2e-3)
    with torch.inference_mode():
        logits = loaded.model(**calibration[0], use_cache=False).logits
    assert logits.isfinite().all()
    for cycle in range(2):
        destination = tmp_path / f"packed{cycle}"
        loaded.save_quantized(str(destination))
        assert list(destination.glob("*.safetensors")) and not list(destination.glob("*.bin"))
        loaded = GPTQModel.load(str(destination), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
        assert {
            name for name, module in loaded.model.named_modules() if isinstance(module, AwqTorchLinear)
        } == expected
        actual_state = loaded.model.state_dict()
        assert set(actual_state) == set(state)
        for key, value in state.items():
            torch.testing.assert_close(actual_state[key], value, rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(loaded.model(**calibration[0], use_cache=False).logits, logits, atol=0, rtol=0)


def test_caller_rope_configuration_is_immutable():
    rope = {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000}}
    original = copy.deepcopy(rope)
    config = tiny_config(rope_parameters=rope)
    Spark2_5ForCausalLM(config)
    assert rope == original


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_released_fp32_residual_stream_and_low_precision_projections(dtype):
    model = tiny_model().to(dtype)
    observed = []
    handles = []
    for layer in model.model.layers:
        handles.append(
            layer.register_forward_hook(lambda module, args, result: observed.append((args[0].dtype, result.dtype)))
        )
    projections = []
    for layer in model.model.layers:
        handles.append(
            layer.self_attn.q_k_v_proj.register_forward_pre_hook(
                lambda module, args: projections.append(args[0].dtype)
            )
        )
    try:
        with torch.inference_mode():
            output = model(torch.tensor([[3, 4, 5]]))
        assert output.logits.dtype == dtype
        assert observed == [(torch.float32, torch.float32)] * 2
        assert projections == [dtype] * 2
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp32_norm_weights_do_not_select_projection_compute_dtype(dtype):
    model = tiny_model().to(dtype)
    ids = torch.tensor([[3, 4, 5]])
    with torch.inference_mode():
        expected = model(ids, use_cache=False).logits
    for name, module in model.named_modules():
        if "norm" in name:
            module.float()
    with torch.inference_mode():
        actual = model(ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_generation_eos_sampling_overrides_and_policy_immutability(tmp_path):
    model = tiny_model()
    ids = torch.tensor([[3, 4, 5]])
    with torch.inference_mode():
        stop = int(model(ids, use_cache=False).logits[0, -1].argmax())
    model.generation_config.eos_token_id = [stop, 1]
    model.generation_config.do_sample = False
    model.save_pretrained(tmp_path)
    for cycle in range(3):
        fresh = Spark2_5QModel.loader.from_pretrained(tmp_path, attn_implementation="eager").eval()
        original = copy.deepcopy(fresh.generation_config.to_dict())
        with torch.inference_mode():
            assert fresh.generate(ids, max_new_tokens=4).shape[1] == ids.shape[1] + 1
            assert fresh.generate(ids, max_new_tokens=4, eos_token_id=None).shape[1] == ids.shape[1] + 4
            caller = GenerationConfig(
                do_sample=True, top_k=8, top_p=0.8, eos_token_id=None, pad_token_id=2, bos_token_id=0
            )
            caller_before = copy.deepcopy(caller.to_dict())
            torch.manual_seed(245)
            actual = fresh.generate(
                ids, generation_config=caller, max_new_tokens=3, return_dict_in_generate=True, output_scores=True
            )
            # Independent autoregressive sampling using native next-token logits.
            torch.manual_seed(245)
            expected = ids.clone()
            for scores in actual.scores:
                logits = fresh(expected, use_cache=False).logits[:, -1]
                threshold = logits.topk(8).values[:, -1:]
                filtered = logits.masked_fill(logits < threshold, -torch.inf)
                sorted_scores, indices = filtered.sort(descending=False)
                remove = sorted_scores.softmax(-1).cumsum(-1) <= 0.2
                remove[:, -1] = False
                filtered = filtered.masked_fill(remove.scatter(1, indices, remove), -torch.inf)
                torch.testing.assert_close(scores, filtered)
                token = torch.multinomial(filtered.softmax(-1), num_samples=1)
                expected = torch.cat([expected, token], -1)
            assert torch.equal(actual.sequences, expected)
            assert caller.to_dict() == caller_before
        assert fresh.generation_config.to_dict() == original
        if cycle < 2:
            destination = tmp_path / f"policy{cycle}"
            fresh.save_pretrained(destination)
            tmp_path = destination


@pytest.mark.parametrize("case", range(6))
def test_pinned_real_tokenizer_source_ids_decode_chat_and_roundtrip(tmp_path, case):
    from tokenicer import Tokenicer

    write_tokenizer(tmp_path, real=True)
    expected = json.loads(read_port_asset(FIXTURES / "tokenizer_cases.json"))[case]
    tokenizer = Tokenicer.load(str(tmp_path), trust_remote_code=False, model_config=tiny_config())
    assert tokenizer(expected["text"], add_special_tokens=False)["input_ids"] == expected["ids"]
    assert tokenizer.decode(expected["ids"], skip_special_tokens=False) == expected["decoded"]
    messages = [{"role": "user", "content": "Hello 世界!"}]
    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    assert (
        rendered
        == "<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant.<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>Hello 世界!<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|></think>"
    )
    for cycle in range(2):
        saved = tmp_path / f"tok{cycle}"
        tokenizer.save_pretrained(saved)
        tokenizer = Tokenicer.load(str(saved), trust_remote_code=False, model_config=tiny_config())
        assert tokenizer(expected["text"], add_special_tokens=False)["input_ids"] == expected["ids"]
