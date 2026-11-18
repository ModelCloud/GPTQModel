# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from pathlib import Path

import defuser
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GenerationConfig, LogitsProcessor, PreTrainedTokenizerFast, SiglipImageProcessor

from gptqmodel import CalibrationConfig
from gptqmodel.models.definitions.ovis2_6_moe import Ovis2_6_MoeQModel, Ovis2_6_NextQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.ovis2_6_moe.configuration_ovis2_6 import Ovis2_6_Moe_Config, Ovis2_6_Next_Config
from tests.port_assets import read_port_asset


CASES = [(Ovis2_6_MoeQModel, Ovis2_6_Moe_Config), (Ovis2_6_NextQModel, Ovis2_6_Next_Config)]


def tiny_config(config_class=Ovis2_6_Moe_Config, kv=2, bias=False, **vision):
    return config_class(
        llm_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": kv,
            "head_dim": 8,
            "num_experts": 3,
            "num_experts_per_tok": 2,
            "vocab_size": 64,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "attention_bias": bias,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "full_attention_interval": 2,
            **({"partial_rotary_factor": 0.5} if config_class is Ovis2_6_Next_Config else {}),
        },
        vit_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 2,
            "window_size": 8,
            **vision,
        },
        visual_vocab_size=16,
    )


def build(definition, config_class, *, dtype=torch.float32, implementation="eager", convert=True, **kwargs):
    torch.manual_seed(51)
    config = tiny_config(config_class, **kwargs)
    config._attn_implementation = implementation
    model = definition.loader.from_config(config, dtype=dtype).eval()
    if convert:
        _convert_model_with_defuser(definition, model, cleanup_original=False)
    return model


def media_inputs():
    return {
        "input_ids": torch.tensor([[1, -301, -300, -302, 3, 4]]),
        "pixel_values": torch.linspace(-1, 1, 48).view(4, 12),
        "grid_thws": torch.tensor([[1, 2, 2]]),
    }


def checkpoint(path, definition, config_class, *, generation_config=None):
    model = build(definition, config_class)
    if generation_config is not None:
        model.generation_config = copy.deepcopy(generation_config)
    model.save_pretrained(path, safe_serialization=True)
    backend = Tokenizer(
        WordLevel(
            {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "a": 4, "b": 5, "user": 6, "assistant": 7},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        chat_template="{% for m in messages %}{% if m.content is string %}{{ m.content }}{% else %}{% for i in m.content %}{% if i.type == 'image' %}<image>{% elif i.type == 'video' %}<video>{% else %}{{ i.text }}{% endif %}{% endfor %}{% endif %}{% endfor %}",
    )
    tokenizer.save_pretrained(path)
    SiglipImageProcessor().save_pretrained(path)
    return model


def released_generation_policy():
    fixture = Path(__file__).parents[1] / "gptqmodel/models/ports/ovis2_6_moe/fixtures/generation_config.json"
    policy = GenerationConfig.from_dict(json.loads(fixture.read_text()))
    policy.update(
        bos_token_id=1,
        pad_token_id=0,
        eos_token_id=[2, 63],
        max_new_tokens=3,
        return_dict_in_generate=True,
        output_scores=True,
    )
    return policy


def test_released_generation_policy_is_pinned_for_both_variants():
    import hashlib

    port = Path(__file__).parents[1] / "gptqmodel/models/ports/ovis2_6_moe"
    provenance = json.loads((port / "provenance.json").read_text())
    payload = (port / "fixtures/generation_config.json").read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == provenance["fixture_sha256"]["generation_config.json"]
    assert all(provenance[size]["files"]["generation_config.json"] == digest for size in ("30B", "80B"))
    released = json.loads(payload)
    assert released["eos_token_id"] == [151645, 151643]
    assert released["do_sample"] is True
    assert [released[name] for name in ("top_k", "top_p", "temperature", "repetition_penalty")] == [20, .95, .6, 1.05]


class ForceToken(LogitsProcessor):
    def __init__(self, token):
        self.token = token

    def __call__(self, input_ids, scores):
        result = torch.full_like(scores, -torch.inf)
        result[:, self.token] = 0
        return result


@torch.inference_mode()
def assert_public_generation_policy(loaded, inputs):
    model = loaded.model
    policy = released_generation_policy()
    assert model.generation_config.to_dict() == policy.to_dict()
    saved_policy = copy.deepcopy(model.generation_config.to_dict())
    decoder_policy = copy.deepcopy(model.llm.generation_config.to_dict())
    reference = copy.deepcopy(model.llm).eval()
    embeddings = model.merge_multimodal(**inputs)
    mask = torch.ones_like(inputs["input_ids"])
    forced = [ForceToken(63)]
    stopped = loaded.generate(**inputs, logits_processor=forced)
    assert hasattr(stopped, "sequences"), "Saved root generation policy did not reach the native decoder"
    assert stopped.sequences.tolist() == [[63]], "Saved additional EOS must stop actual generation"
    assert len(stopped.scores) == 1
    for explicit_none in (False, True):
        torch.manual_seed(526)
        expected = reference.generate(
            inputs_embeds=embeddings, attention_mask=mask, generation_config=copy.deepcopy(policy)
        )
        torch.manual_seed(526)
        actual = loaded.generate(**inputs, **({"generation_config": None} if explicit_none else {}))
        assert torch.equal(actual.sequences, expected.sequences)
        assert torch.isfinite(expected.scores[0]).sum() > 1
        for got, want in zip(actual.scores, expected.scores, strict=True):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
    caller = GenerationConfig(
        do_sample=False, eos_token_id=63, pad_token_id=0, max_new_tokens=4, return_dict_in_generate=False
    )
    caller_policy = copy.deepcopy(caller.to_dict())
    assert loaded.generate(**inputs, generation_config=caller, logits_processor=forced).tolist() == [[63]]
    overridden = loaded.generate(
        **inputs,
        generation_config=caller,
        logits_processor=forced,
        eos_token_id=None,
        max_new_tokens=2,
        return_dict_in_generate=True,
        output_scores=True,
    )
    assert overridden.sequences.tolist() == [[63, 63]]
    assert len(overridden.scores) == 2
    assert caller.to_dict() == caller_policy
    assert model.generation_config.to_dict() == saved_policy
    assert model.llm.generation_config.to_dict() == decoder_policy


@pytest.mark.parametrize("definition,config_class", CASES)
@pytest.mark.parametrize("cycles", [0, 2])
@pytest.mark.parametrize("media", [False, True])
def test_public_saved_generation_policy_dense_two_reloads(definition, config_class, cycles, media, tmp_path):
    from gptqmodel import GPTQModel

    source = tmp_path / "source"
    checkpoint(source, definition, config_class, generation_config=released_generation_policy())
    loaded = GPTQModel.load(str(source), device="cpu", dtype=torch.float32, attn_implementation="eager")
    for cycle in range(cycles):
        target = tmp_path / f"dense{cycle}"
        loaded.model.save_pretrained(target, safe_serialization=True)
        loaded.processor.save_pretrained(target)
        assert list(target.glob("*.safetensors"))
        assert not list(target.glob("*.bin"))
        loaded = GPTQModel.load(str(target), device="cpu", dtype=torch.float32, attn_implementation="eager")
    inputs = media_inputs() if media else {"input_ids": torch.tensor([[1, 4, 5]])}
    assert_public_generation_policy(loaded, inputs)


@pytest.mark.parametrize("definition,config_class", CASES)
@pytest.mark.parametrize("policy_options", [
    {"return_dict_in_generate": True}, {"num_beams": 3}, {"do_sample": True, "num_return_sequences": 2},
])
def test_thinking_guard_resolves_root_policy_and_caller_repair(definition, config_class, policy_options, tmp_path):
    from gptqmodel import GPTQModel

    checkpoint(tmp_path, definition, config_class)
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, attn_implementation="eager")
    policy = GenerationConfig(max_new_tokens=40, eos_token_id=None, pad_token_id=0, **policy_options)
    loaded.model.generation_config = copy.deepcopy(policy)
    options = {"enable_thinking": True, "enable_thinking_budget": True, "thinking_budget": 1}
    inputs = {"input_ids": torch.tensor([[1, 4, 5]])}
    with pytest.raises(ValueError, match="one non-beam tensor result"):
        loaded.generate(**inputs, **options, max_new_tokens=40)
    actual = loaded.generate(
        **inputs, **options, logits_processor=[ForceToken(63)], max_new_tokens=40,
        num_beams=1, num_return_sequences=1, return_dict_in_generate=False,
    )
    assert actual.shape == (1, 40)
    assert loaded.model.generation_config.to_dict() == policy.to_dict()


@pytest.mark.parametrize("definition,config_class", CASES)
def test_thinking_uses_saved_max_new_tokens_for_both_native_stages(definition, config_class, tmp_path):
    from gptqmodel import GPTQModel

    policy = GenerationConfig(max_new_tokens=40, do_sample=False, eos_token_id=None, pad_token_id=0)
    checkpoint(tmp_path, definition, config_class, generation_config=policy)
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, attn_implementation="eager")
    options = {"enable_thinking": True, "enable_thinking_budget": True, "thinking_budget": 1}
    inputs = {"input_ids": torch.tensor([[1, 4, 5]])}
    appendent = loaded.tokenizer(
        "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n",
        return_tensors="pt", return_attention_mask=False,
    ).input_ids
    for budget in (None, 45):
        actual = loaded.generate(
            **inputs, **options, logits_processor=[ForceToken(63)],
            **({} if budget is None else {"max_new_tokens": budget}),
        )
        total = budget or policy.max_new_tokens
        expected = torch.cat([torch.tensor([[63]]), appendent, torch.full((1, total - 1 - appendent.shape[1]), 63)], dim=1)
        assert torch.equal(actual, expected)
        assert loaded.model.generation_config.to_dict() == policy.to_dict()


@pytest.mark.parametrize("definition,config_class", CASES)
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_hostile_checkpoint_is_data_only(definition, config_class, trust, bypass, tmp_path, monkeypatch):
    from gptqmodel import BACKEND, GPTQModel

    checkpoint(tmp_path, definition, config_class)
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    marker = tmp_path / "EXECUTED"
    (tmp_path / "hostile.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\nraise RuntimeError('checkpoint code')\n"
    )
    for filename in ["config.json", "tokenizer_config.json", "preprocessor_config.json"]:
        path = tmp_path / filename
        data = json.loads(path.read_text())
        data["auto_map"] = {
            "AutoConfig": "hostile.Config",
            "AutoModelForCausalLM": "hostile.Model",
            "AutoTokenizer": ["hostile.Tokenizer", "hostile.Fast"],
            "AutoProcessor": "hostile.Processor",
            "AutoImageProcessor": "hostile.Image",
        }
        path.write_text(json.dumps(data))
    loaded = GPTQModel.load(
        str(tmp_path),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.float32,
        trust_remote_code=trust,
        attn_implementation="eager",
    )
    assert isinstance(loaded, definition)
    assert type(loaded.model).__module__.startswith("gptqmodel.models.ports.ovis2_6_moe")
    assert loaded.processor.__class__.__name__ == "Ovis26Processor"
    assert loaded.tokenizer.encode("a b", add_special_tokens=False) == [4, 5]
    with torch.inference_mode():
        assert loaded.model(**media_inputs()).logits.isfinite().all()
    assert not marker.exists()


@pytest.mark.parametrize("definition,config_class", CASES)
@pytest.mark.parametrize("offload", [False, True])
def test_public_missing_post_ln_corrupt_payload_repair_retry(definition, config_class, offload, tmp_path):
    from safetensors.torch import load_file, save_file

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.quantization.config import AWQConfig

    checkpoint(tmp_path, definition, config_class)
    file = tmp_path / "model.safetensors"
    original = load_file(file)
    state = {key: tensor for key, tensor in original.items() if ".post_layernorm." not in key}
    save_file(state, file, metadata={"format": "pt"})
    payload = file.read_bytes()
    file.write_bytes(b"invalid-safetensors")
    options = {"backend": BACKEND.TORCH, "dtype": torch.float16, "device": "cpu", "attn_implementation": "eager"}
    with pytest.raises(Exception, match="header|safetensor|buffer"):
        GPTQModel.load(str(tmp_path), **options)
    file.write_bytes(payload)
    options.pop("device")
    qcfg = AWQConfig(
        device="cpu",
        group_size=16,
        fallback=None,
        offload_to_disk=offload,
        offload_to_disk_path=str(tmp_path / "offload"),
    )
    loaded = GPTQModel.load(str(tmp_path), quantize_config=qcfg, **options)
    if offload:
        loaded.pre_quantize_generate_hook_start()
        loaded.quantize(
            CalibrationConfig(dataset=[media_inputs()], batch_size=1, min_length=1),
            backend=BACKEND.TORCH,
        )
    with torch.inference_mode():
        output = loaded.model(**media_inputs()).logits
        assert output.isfinite().all()
    norm = loaded.model.visual_tokenizer.vit.vision_model.post_layernorm
    torch.testing.assert_close(norm.weight, torch.ones_like(norm.weight), atol=0, rtol=0)
    torch.testing.assert_close(norm.bias, torch.zeros_like(norm.bias), atol=0, rtol=0)


@pytest.mark.parametrize("definition,config_class", CASES)
@pytest.mark.parametrize("offload", [False, True])
def test_public_image_calibrated_plain_awq_pack_infer_two_reloads(definition, config_class, offload, tmp_path):
    import numpy as np
    from PIL import Image

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    source = tmp_path / "source"
    checkpoint(source, definition, config_class, generation_config=released_generation_policy())
    qcfg = AWQConfig(
        device="cpu",
        group_size=16,
        fallback=None,
        offload_to_disk=offload,
        offload_to_disk_path=str(tmp_path / "offload"),
    )
    loaded = GPTQModel.load(
        str(source), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16, attn_implementation="eager"
    )
    # Image assets drive actual processor patchification and multimodal capture.
    image = Image.fromarray(np.arange(48, dtype=np.uint8).reshape(4, 4, 3))
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "a b a b"}]}]
    calibration = [loaded.processor(messages, min_pixels=16, max_pixels=16) for _ in range(2)]
    reference = definition.loader.from_pretrained(source, dtype=torch.float16, attn_implementation="eager").eval()
    _convert_model_with_defuser(definition, reference, cleanup_original=True)
    dense = {key: tensor.clone() for key, tensor in reference.state_dict().items()}
    with torch.inference_mode():
        expected_capture = reference.merge_multimodal(
            calibration[0]["input_ids"], calibration[0]["pixel_values"], calibration[0]["grid_thws"]
        )
    captures = []

    def capture(module, args, kwargs):
        if not captures:
            captures.append((args[0] if args else kwargs["hidden_states"]).detach().clone())

    handle = loaded.model.llm.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    loaded.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    handle.remove()
    torch.testing.assert_close(captures[0], expected_capture, atol=0, rtol=0)
    modules = {name: module for name, module in loaded.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert len(modules) == (26 if config_class is Ovis2_6_Moe_Config else 30)
    assert all(name.startswith("llm.model.layers.") for name in modules)
    assert not any(
        "in_proj_ba" in name or name.endswith("mlp.gate") or "shared_expert_gate" in name for name in modules
    )
    for key, value in loaded.model.state_dict().items():
        if key in dense:
            torch.testing.assert_close(value, dense[key], atol=0, rtol=0, msg=key)
    with torch.inference_mode():
        expected = loaded.model(**calibration[0], use_cache=False).logits
        assert expected.isfinite().all()
        state = {key: value.clone() for key, value in loaded.model.state_dict().items()}
        buffers = {key: value.clone() for key, value in loaded.model.named_buffers()}
        attention_backends = (
            loaded.model.config._attn_implementation,
            loaded.model.config.llm_config._attn_implementation,
            loaded.model.config.vit_config._attn_implementation,
        )
        for cycle in range(2):
            target = tmp_path / f"quantized{cycle}"
            loaded.save(target)
            loaded = GPTQModel.load(
                str(target), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16, attn_implementation="eager"
            )
            assert loaded.model.state_dict().keys() == state.keys()
            for key, tensor in loaded.model.state_dict().items():
                torch.testing.assert_close(tensor, state[key], atol=0, rtol=0, msg=key)
            for key, tensor in loaded.model.named_buffers():
                if key in buffers and (key.startswith("visual_tokenizer.") or ".rotary_emb." in key):
                    torch.testing.assert_close(tensor, buffers[key], atol=0, rtol=0, msg=key)
            assert attention_backends == (
                loaded.model.config._attn_implementation,
                loaded.model.config.llm_config._attn_implementation,
                loaded.model.config.vit_config._attn_implementation,
            )
            torch.testing.assert_close(
                loaded.model(**calibration[0], use_cache=False).logits, expected, atol=0, rtol=0
            )
            assert list(target.glob("*.safetensors"))
            assert not list(target.glob("*.bin"))
        assert_public_generation_policy(loaded, media_inputs())


@pytest.mark.parametrize("definition,config_class", CASES)
def test_real_multimodal_forward_and_generation(definition, config_class):
    model = build(definition, config_class)
    inputs = media_inputs()
    with torch.no_grad():
        output = model(**inputs, use_cache=True)
        assert output.logits.shape == (1, 6, 64)
        assert output.logits.isfinite().all()
        generated = model.generate(**inputs, max_new_tokens=2, do_sample=False)
        assert generated.shape == (1, 2)


from gptqmodel.models.definitions.ovis2_6_moe import Ovis2_6_MoeQModel, Ovis2_6_NextQModel
from gptqmodel.models.local import LocalModelLoader


@pytest.mark.parametrize("definition", [Ovis2_6_MoeQModel, Ovis2_6_NextQModel])
def test_definition_owns_model_selection(definition):
    assert isinstance(definition.loader, LocalModelLoader)
    assert not definition.require_trust_remote_code


@pytest.mark.parametrize("definition,config_class", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("kv,bias", [(4, False), (2, True), (1, False)])
def test_native_decoder_multimodal_cache_and_causality(definition, config_class, dtype, implementation, kv, bias):
    model = build(definition, config_class, dtype=dtype, implementation=implementation, kv=kv, bias=bias)
    inputs = media_inputs()
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([[6, 7]])], dim=-1)
    with torch.inference_mode():
        full = model(**inputs).logits
        first = model(**{**inputs, "input_ids": inputs["input_ids"][:, :6]}, use_cache=True)
        cached = [first.logits]
        cache = first.past_key_values
        for token in inputs["input_ids"][:, 6:].split(1, dim=-1):
            step = model(token, past_key_values=cache, use_cache=True)
            cache = step.past_key_values
            cached.append(step.logits)
        tolerance = {torch.float32: 3e-6, torch.float16: 4e-3, torch.bfloat16: 3e-2}[dtype]
        torch.testing.assert_close(torch.cat(cached, 1), full, atol=tolerance, rtol=tolerance)
        changed = inputs["input_ids"].clone()
        changed[:, -1] = 11
        changed_logits = model(**{**inputs, "input_ids": changed}).logits
        torch.testing.assert_close(changed_logits[:, :-1], full[:, :-1], atol=tolerance, rtol=tolerance)
        independent = model.decoder_class._from_config(model.config.llm_config, dtype=dtype).eval()
        defuser.convert_model(independent, cleanup_original=True)
        # Separate native model, not a call through the wrapper's forward/cache path.
        independent.load_state_dict(model.llm.state_dict(), strict=True)
        embeds = model.merge_multimodal(**inputs)
        expected = independent(inputs_embeds=embeds).logits
        torch.testing.assert_close(full, expected, atol=0, rtol=0)


@pytest.mark.parametrize("definition,config_class", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_safetensors_two_cycles_and_assignment(definition, config_class, dtype, tmp_path):
    model = build(definition, config_class, dtype=dtype)
    inputs = media_inputs()
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    with torch.inference_mode():
        expected = model(**inputs).logits
        original = {key: value.clone() for key, value in model.state_dict().items()}
        buffers = {key: value.clone() for key, value in model.named_buffers()}
        for assign in [False, True]:
            fresh = build(definition, config_class, dtype=dtype)
            fresh.load_state_dict(original, assign=assign, strict=True)
            torch.testing.assert_close(fresh(**inputs).logits, expected, atol=0, rtol=0)
        for cycle in range(2):
            path = tmp_path / str(cycle)
            model.save_pretrained(path, safe_serialization=True)
            model = definition.loader.from_pretrained(path, dtype=dtype, attn_implementation="eager").eval()
            _convert_model_with_defuser(definition, model, cleanup_original=True)
            assert model.state_dict().keys() == original.keys()
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, original[key], atol=0, rtol=0)
            for key, tensor in model.named_buffers():
                torch.testing.assert_close(tensor, buffers[key], atol=0, rtol=0, msg=key)
            torch.testing.assert_close(model(**inputs).logits, expected, atol=0, rtol=0)
            assert list(path.glob("*.safetensors"))
            assert not list(path.glob("*.bin"))


@pytest.mark.parametrize("definition,config_class", CASES)
def test_beam_and_greedy_match_separate_native_decoder(definition, config_class):
    model = build(definition, config_class)
    inputs = media_inputs()
    with torch.inference_mode():
        embeds = model.merge_multimodal(**inputs)
        native = model.decoder_class(model.config.llm_config).eval()
        defuser.convert_model(native, cleanup_original=True)
        native.load_state_dict(model.llm.state_dict(), strict=True)
        for beams in [1, 3]:
            actual = model.generate(**inputs, num_beams=beams, max_new_tokens=3, do_sample=False)
            expected = native.generate(
                inputs_embeds=embeds,
                attention_mask=torch.ones(1, 6, dtype=torch.long),
                num_beams=beams,
                max_new_tokens=3,
                do_sample=False,
            )
            assert torch.equal(actual, expected)


@pytest.mark.parametrize("definition,config_class", CASES)
def test_bad_media_then_retry_does_not_mutate_model(definition, config_class):
    model = build(definition, config_class)
    inputs = media_inputs()
    with torch.inference_mode():
        expected = model(**inputs).logits
        with pytest.raises(ValueError, match="Visual atom count"):
            model(**{**inputs, "input_ids": torch.tensor([[1, -300, -300, 4]])})
        with pytest.raises(ValueError, match="require pixel_values"):
            model(inputs["input_ids"])
        with pytest.raises(ValueError, match="unknown"):
            model(torch.tensor([[1, -200, 4]]))
        torch.testing.assert_close(model(**inputs).logits, expected, atol=0, rtol=0)


def visual_oracle(model, patches, grids):
    """Independent coordinate/mask equations, without port vision/merge helpers."""
    import torch.nn.functional as F

    vision = model.visual_tokenizer.vit
    config = vision.config
    weights = vision.state_dict()
    stride, width = config.hidden_stride, config.hidden_size
    coords = [
        (media, frame, row + dr, col + dc)
        for media, (frames, height, cols) in enumerate(grids.tolist())
        for frame in range(frames)
        for row in range(0, height, stride)
        for col in range(0, cols, stride)
        for dr in range(stride)
        for dc in range(stride)
    ]
    coordinates = torch.tensor(coords)

    def linear(x, name):
        return F.linear(
            x, weights[name + ".weight"].reshape(weights[name + ".weight"].shape[0], -1), weights.get(name + ".bias")
        )

    x = linear(patches, "vision_model.embeddings.patch_embedding")
    if config.preserve_original_pe:
        side = config.image_size // config.patch_size if config.num_patches < 0 else int(config.num_patches**0.5)
        original = weights["vision_model.embeddings.position_embedding.weight"].T.reshape(1, width, side, side)
        resized = [
            F.interpolate(original, size=(int(grid[1]), int(grid[2])), mode="bicubic", align_corners=False)[0]
            for grid in grids
        ]
        x += torch.stack([resized[m][:, r, c] for m, _, r, c in coords])
    heads = config.num_attention_heads
    head_dim = width // heads
    frequencies = 10000.0 ** (-torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2))
    angles = torch.cat([coordinates[:, 2, None] * frequencies, coordinates[:, 3, None] * frequencies], dim=-1)
    angles = torch.cat([angles, angles], -1)
    cos, sin = angles.cos().float()[:, None], angles.sin().float()[:, None]
    for index in range(config.num_hidden_layers):
        root = f"vision_model.encoder.layers.{index}"

        def norm(value, name):
            return F.layer_norm(
                value,
                (width,),
                weights[root + f".{name}.weight"],
                weights[root + f".{name}.bias"],
                config.layer_norm_eps,
            )

        z = norm(x, "layer_norm1")
        query, key, value = [
            linear(z, root + ".self_attn." + name).reshape(-1, heads, head_dim)
            for name in ["q_proj", "k_proj", "v_proj"]
        ]
        if config.use_rope:

            def rotate(tensor):
                left, right = tensor.float().chunk(2, dim=-1)
                return (tensor.float() * cos + torch.cat([-right, left], -1) * sin).to(z.dtype)

            query, key = rotate(query), rotate(key)
        same = (coordinates[:, None, :2] == coordinates[None, :, :2]).all(-1)
        full = config.fullatt_block_indexes is None or index in config.fullatt_block_indexes
        if not full:
            windows = coordinates[:, 2:] // (config.window_size // config.patch_size)
            same &= (windows[:, None] == windows[None, :]).all(-1)
        scores = torch.einsum("ihd,jhd->hij", query, key) / head_dim**0.5
        scores.masked_fill_(~same, -torch.inf)
        probabilities = scores.softmax(-1, dtype=torch.float32).to(value.dtype)
        attended = torch.einsum("hij,jhd->ihd", probabilities, value).reshape(-1, width)
        x = x + linear(attended, root + ".self_attn.out_proj")
        z = norm(x, "layer_norm2")
        z = F.gelu(linear(z, root + ".mlp.fc1"), approximate="tanh")
        x = x + linear(z, root + ".mlp.fc2")
    merged = x.reshape(-1, width * stride**2)
    head = model.visual_tokenizer.head
    logits = F.layer_norm(
        F.linear(merged, head[0].weight), head[1].normalized_shape, head[1].weight, head[1].bias, head[1].eps
    )
    tokens = F.pad(logits.softmax(-1, dtype=torch.float32).to(logits.dtype), (0, 4))
    return x, tokens @ model.vte.weight


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize(
    "options",
    [{}, {"fullatt_block_indexes": [1]}, {"preserve_original_pe": False}, {"use_rope": False}, {"num_patches": 16}],
)
@torch.inference_mode()
def test_independent_float64_navit_soft_vocabulary_and_image_merge(implementation, options):
    model = build(*CASES[0], dtype=torch.float64, implementation=implementation, window_size=4, **options)
    grids = torch.tensor([[2, 4, 6], [1, 2, 4]])
    patches = torch.linspace(-2, 2, 56 * 12, dtype=torch.float64).reshape(56, 12)
    expected_features, expected_visual = visual_oracle(model, patches, grids)
    actual_features = model.visual_tokenizer.vit(patches, grids, output_hidden_states=True).hidden_states[-1]
    torch.testing.assert_close(actual_features, expected_features, atol=3e-7, rtol=3e-7)
    tokens = model.visual_tokenizer(patches, grids)
    torch.testing.assert_close(tokens[:, -4:], torch.zeros_like(tokens[:, -4:]), atol=0, rtol=0)
    torch.testing.assert_close(model.vte(tokens), expected_visual, atol=3e-8, rtol=3e-7)
    ids = torch.tensor([[1, -301, *([-300] * 14), -302, 4]])
    expected = model.get_wte()(ids.clamp(min=0))
    expected[:, 1] = model.vte.weight[-4]
    expected[:, -2] = model.vte.weight[-3]
    expected[:, 2:-2] = expected_visual
    merged = model.merge_multimodal(ids, patches, grids)
    torch.testing.assert_close(merged, expected, atol=3e-8, rtol=3e-7)
    # Checkpoint-optional post-LN is intentionally not the visual head's input.
    model.visual_tokenizer.vit.vision_model.post_layernorm.weight.fill_(9)
    model.visual_tokenizer.vit.vision_model.post_layernorm.bias.fill_(17)
    torch.testing.assert_close(model.merge_multimodal(ids, patches, grids), merged, atol=0, rtol=0)


@pytest.mark.parametrize("definition,config_class", CASES)
@pytest.mark.parametrize("kind", ["image", "video"])
def test_processor_pixels_patch_order_prompt_save_reload_and_retry(definition, config_class, kind, tmp_path):
    import numpy as np
    from PIL import Image

    from gptqmodel.models.ports.ovis2_6_moe.processing_ovis2_6 import Ovis26Processor

    model = checkpoint(tmp_path, definition, config_class)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tmp_path)
    processor = Ovis26Processor.from_pretrained(tmp_path, config=model.config, tokenizer=tokenizer)
    arrays = [
        np.arange(48, dtype=np.uint8).reshape(4, 4, 3),
        np.arange(48, dtype=np.uint8).reshape(4, 4, 3)[::-1].copy(),
    ]
    images = [Image.fromarray(array) for array in arrays]
    media = images[0] if kind == "image" else images
    message = [{"role": "user", "content": [{"type": kind, kind: media}, {"type": "text", "text": "a b"}]}]
    result = processor(message, min_pixels=16, max_pixels=16)
    expected_patches = []
    for array in arrays[:1] if kind == "image" else arrays:
        normalized = torch.tensor(array.copy()).permute(2, 0, 1).float() / 127.5 - 1
        expected_patches += [
            normalized[:, row : row + 2, col : col + 2].reshape(-1) for row in [0, 2] for col in [0, 2]
        ]
    torch.testing.assert_close(result["pixel_values"], torch.stack(expected_patches), atol=1e-7, rtol=1e-7)
    assert result["grid_thws"].tolist() == [[1 if kind == "image" else 2, 2, 2]]
    atoms = 1 if kind == "image" else 2
    begin, end = (-301, -302) if kind == "image" else (-303, -304)
    assert result["input_ids"].tolist() == [[begin, *([-300] * atoms), end, 4, 5]]
    with torch.inference_mode():
        assert model(**result).logits.isfinite().all()
    save = tmp_path / "processor"
    processor.save_pretrained(save)
    second = Ovis26Processor.from_pretrained(
        save, config=model.config, tokenizer=PreTrainedTokenizerFast.from_pretrained(save)
    )
    for key, value in result.items():
        torch.testing.assert_close(second(message, min_pixels=16, max_pixels=16)[key], value, atol=0, rtol=0)
    with pytest.raises(ValueError, match="placeholder count"):
        processor._merge_inputs([1, -200], -200, [], -301, -302)
    torch.testing.assert_close(
        processor(message, min_pixels=16, max_pixels=16)["pixel_values"], result["pixel_values"], atol=0, rtol=0
    )


@pytest.mark.parametrize("size,case", list(zip(["30B", "80B"], CASES)))
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_real_released_tokenizer_assets_ids_chat_and_reload(size, case, trust, bypass, tmp_path, monkeypatch):

    from transformers import AutoTokenizer

    from gptqmodel.models.ports.ovis2_6_moe.processing_ovis2_6 import Ovis26Processor
    from gptqmodel.utils.hf import load_hf_tokenizer

    definition, config_class = case
    fixtures = Path(__file__).parents[1] / "gptqmodel/models/ports/ovis2_6_moe/fixtures"
    (tmp_path / "tokenizer.json").write_bytes(read_port_asset(fixtures / "qwen_tokenizer.json"))
    for file in ["tokenizer_config.json", "preprocessor_config.json", "added_tokens.json", "special_tokens_map.json"]:
        (tmp_path / file).write_bytes((fixtures / f"{size}_{file}").read_bytes())
    config = config_class(**json.loads((fixtures / f"{size}_config.json").read_text()))
    config.save_pretrained(tmp_path)
    original = AutoTokenizer.from_pretrained(tmp_path, trust_remote_code=False)
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    data = json.loads((tmp_path / "tokenizer_config.json").read_text())
    data["auto_map"] = {"AutoTokenizer": ["hostile.Tokenizer", "hostile.Fast"]}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(data))
    (tmp_path / "hostile.py").write_text("raise AssertionError('remote tokenizer code must not execute')")
    tokenizer = load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=trust)
    for text in ["a b", "Hello, world!", "你好，世界！", "é Ελληνικά العربية\n\t", "<|im_start|>user\n<image>\nTest"]:
        expected = original.encode(text, add_special_tokens=False)
        assert tokenizer.encode(text, add_special_tokens=False) == expected
        assert tokenizer.decode(expected) == original.decode(expected)
    messages = [{"role": "user", "content": "Hello!"}]
    for thinking in [False, True]:
        assert tokenizer.apply_chat_template(
            messages, tokenize=True, enable_thinking=thinking
        ) == original.apply_chat_template(messages, tokenize=True, enable_thinking=thinking)
    processor = definition.loader.load_processor(tmp_path, config=config, tokenizer=tokenizer, trust_remote_code=trust)
    assert isinstance(processor, Ovis26Processor)
    saved = tmp_path / "roundtrip"
    processor.save_pretrained(saved)
    reloaded = load_hf_tokenizer(saved, model_config=config, trust_remote_code=trust)
    assert reloaded.encode("Hello, world! 你好", add_special_tokens=False) == original.encode(
        "Hello, world! 你好", add_special_tokens=False
    )
