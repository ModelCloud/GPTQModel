# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Package-owned K3 selection and real forward/cache/checkpoint qualification."""

import base64
import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import KimiLinearConfig as NativeConfig
from transformers import KimiLinearForCausalLM as NativeModel

from gptqmodel import GPTQModel
from gptqmodel.models.auto import check_and_get_model_definition
from gptqmodel.models.definitions.kimi_k3 import KimiK3QModel, KimiLinearQModel
from gptqmodel.models.definitions.kimi_linear import KimiLinearNativeQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.kimi_k3.configuration_kimi_k3 import KimiK3Config, KimiK3VisionConfig, KimiLinearConfig
from gptqmodel.models.ports.kimi_k3.kimi_k3_vision_processing import KimiK3VisionProcessor
from gptqmodel.models.ports.kimi_k3.modeling_kimi_k3 import KimiK3ForConditionalGeneration
from gptqmodel.models.ports.kimi_k3.modeling_kimi_linear import KimiLinearForCausalLM
from gptqmodel.models.ports.kimi_k3.tokenization_kimi import TikTokenTokenizer
from gptqmodel.utils.hf import load_hf_tokenizer, load_model_config
from gptqmodel.utils.security import BYPASS_ENV


PIN = "f831ab66814297da540d832a5235f8e904f29d06"
FIXTURES = Path(__file__).parent / "fixtures/ports/kimi_k3"
LINEAR_PIN = "e1df551a447157d4658b573f9a695d57658590e9"


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_native_linear_real_tokenizer_contractions_match_pinned_source(tmp_path, monkeypatch, trust, bypass):
    import ast
    from shutil import copyfile

    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    asset = (
        Path.home() / ".cache/huggingface/hub/models--moonshotai--Kimi-Linear-48B-A3B-Instruct/snapshots" / LINEAR_PIN
    )
    if not (asset / "tiktoken.model").is_file():
        pytest.skip("pinned ordinary Kimi-Linear tokenizer asset is not cached")
    if bypass is None:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    else:
        monkeypatch.setenv(BYPASS_ENV, bypass)
    for file in ("config.json", "tokenizer_config.json", "tiktoken.model", "chat_template.jinja"):
        copyfile(asset / file, tmp_path / file)
    raw = json.loads((tmp_path / "config.json").read_text())
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config) is NativeConfig and not hasattr(config, "attn_res_block_size")
    marker = hostile_checkpoint(tmp_path, config)
    assert raw["model_type"] == config.model_type
    # Read the pinned literal regex, never import or execute checkpoint Python.
    tree = ast.parse((asset / "tokenization_kimi.py").read_text())
    source_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TikTokenTokenizer")
    pattern = next(n.value for n in source_class.body if isinstance(n, ast.Assign) and n.targets[0].id == "pat_str")
    pattern = "|".join(ast.literal_eval(pattern.args[0]))
    oracle = tiktoken.Encoding(
        name="pinned-linear-oracle",
        pat_str=pattern,
        mergeable_ranks=load_tiktoken_bpe(str(asset / "tiktoken.model")),
        special_tokens={},
    )
    actual = load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=trust, local_files_only=True)
    texts = ["it's", "I'm sure we'll see it's fine", "DON'T can't shouldn't they've I'd", "你好 1234!\n"]
    for text in texts:
        expected = oracle.encode_ordinary(text)
        assert actual.encode(text) == expected
        assert actual(text, return_tensors="pt")["input_ids"].tolist() == [expected]
        assert actual.decode(expected) == text
    assert len(actual.encode(texts[1])) == 6
    actual.save_pretrained(tmp_path)
    retry = load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=trust, local_files_only=True)
    assert retry.encode(texts[1]) == oracle.encode_ordinary(texts[1])
    assert retry.encode("[BOS][EOS]") == [163584, 163585]
    assert retry.decode([163584, 163585]) == "[BOS][EOS]"
    assert retry.num_reserved_special_tokens == 258
    (tmp_path / "tiktoken.model").write_text("invalid-bpe")
    with pytest.raises(ValueError):
        load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=trust, local_files_only=True)
    copyfile(asset / "tiktoken.model", tmp_path / "tiktoken.model")
    repaired = load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=trust, local_files_only=True)
    assert repaired.encode(texts[1]) == oracle.encode_ordinary(texts[1])
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "weather",
                "parameters": {"type": "object", "properties": {"z": {"type": "string"}, "a": {"type": "integer"}}},
            },
        }
    ]
    sorted_tools = json.loads(json.dumps(tools, sort_keys=True))
    from transformers.tokenization_python import PythonBackend

    chat = [{"role": "user", "content": "What's the weather?"}]
    expected_chat = PythonBackend.apply_chat_template(
        repaired, chat, tools=sorted_tools, tokenize=False, add_generation_prompt=True
    )
    assert repaired.apply_chat_template(chat, tools=tools) == expected_chat
    assert repaired.apply_chat_template(chat, tools=tools, tokenize=True, return_dict=False) == repaired.encode(
        expected_chat
    )
    assert not marker.exists()


@pytest.fixture(autouse=True)
def bounded_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(791)
        yield
    torch.set_num_threads(old)


def tiny_config(mode="hybrid", **kwargs):
    options = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "q_lora_rank": 4,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 4,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "num_shared_experts": 1,
        "moe_intermediate_size": 12,
        "routed_expert_hidden_size": 8,
        "latent_moe_use_norm": True,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "attn_res_block_size": 2,
        "first_k_dense_replace": 1,
        "pad_token_id": 0,
    }
    if mode != "mla":
        options["linear_attn_config"] = {
            "kda_layers": [1, 3] if mode == "hybrid" else [1, 2, 3, 4],
            "full_attn_layers": [2, 4] if mode == "hybrid" else [],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        }
    options.update(kwargs)
    return KimiLinearConfig(**options)


def tiny_k3(**kwargs):
    vision = KimiK3VisionConfig(
        patch_size=2,
        init_pos_emb_height=2,
        init_pos_emb_width=2,
        vt_num_attention_heads=2,
        vt_num_hidden_layers=1,
        vt_hidden_size=8,
        vt_intermediate_size=12,
        qkv_hidden_size=8,
        text_hidden_size=16,
        merge_kernel_size=(2, 2),
        _attn_implementation="eager",
    )
    return KimiK3Config(text_config=tiny_config(**kwargs), vision_config=vision, media_placeholder_token_id=31)


@pytest.mark.parametrize("kind", ["attnres", "k3"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_published_padded_a_log_first_load_and_two_safetensors_roundtrips(tmp_path, kind, dtype):
    from safetensors.torch import load_file, save_file

    config = tiny_config() if kind == "attnres" else tiny_k3()
    text = config if kind == "attnres" else config.text_config
    text.linear_attn_config["num_heads"] = 96
    model = (KimiLinearForCausalLM(config) if kind == "attnres" else KimiK3ForConditionalGeneration(config)).eval()
    model.save_pretrained(tmp_path)
    payload_path = tmp_path / "model.safetensors"
    state = load_file(payload_path)
    expected = {name: value.clone() for name, value in state.items()}
    keys = [name for name in state if name.endswith(".self_attn.A_log")]
    assert len(keys) == 2
    reference = json.loads((FIXTURES / "a_log_reference.json").read_text())
    raw = base64.b64decode(reference["base64"])
    assert hashlib.sha256(raw).hexdigest() == reference["sha256"]
    assert reference["revision"] == PIN and reference["shape"] == [128] and reference["active_heads"] == 96
    assert (
        json.loads((FIXTURES / "checkpoint_config.json").read_text())["text_config"][
            "linear_attn_config"
        ]["num_heads"]
        == 96
    )
    published = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
    assert not bool(published[96:].view(torch.uint8).count_nonzero())
    for name in keys:
        state[name] = published.clone()
        expected[name] = published[:96].clone()
    save_file(state, payload_path, metadata={"format": "pt"})
    loader = KimiLinearQModel.loader if kind == "attnres" else KimiK3QModel.loader
    restored = loader.from_pretrained(tmp_path, local_files_only=True, dtype=dtype)
    ids = torch.tensor([[1, 4, 7]])
    for iteration in range(2):
        actual = restored.state_dict()
        assert set(actual) == set(expected)
        for name, original in expected.items():
            target_dtype = torch.float32 if name.endswith((".A_log", ".dt_bias")) else dtype
            assert actual[name].dtype == target_dtype
            torch.testing.assert_close(actual[name], original.to(target_dtype), rtol=0, atol=0)
        with torch.no_grad():
            logits = restored(ids, use_cache=False).logits
            prefix = restored(ids[:, :2], use_cache=True)
            continuation = restored(ids[:, 2:], past_key_values=prefix.past_key_values).logits
        torch.testing.assert_close(continuation, logits[:, 2:], atol=0.003, rtol=0.003)
        saved = tmp_path / f"normalized-{iteration}"
        restored.save_pretrained(saved)
        saved_state = load_file(saved / "model.safetensors")
        for name in keys:
            assert saved_state[name].shape == (96,)
            torch.testing.assert_close(saved_state[name], expected[name], rtol=0, atol=0)
        restored = loader.from_pretrained(saved, local_files_only=True, dtype=dtype)
    assert not list(tmp_path.rglob("*.bin"))


@pytest.mark.parametrize("bad", ["short", "long", "rank", "nonzero", "negative-zero", "dtype", "wrong-heads"])
def test_a_log_invalid_padding_fails_even_ignore_mismatch_then_retries(tmp_path, bad):
    from safetensors.torch import load_file, save_file

    config = tiny_config()
    config.linear_attn_config["num_heads"] = 96 if bad != "wrong-heads" else 2
    model = KimiLinearForCausalLM(config).eval()
    model.save_pretrained(tmp_path)
    file = tmp_path / "model.safetensors"
    state = load_file(file)
    key = "model.layers.0.self_attn.A_log"
    original = state[key].clone()
    state[key] = torch.zeros({"short": 95, "long": 129}.get(bad, 128), dtype=torch.float32)
    if bad == "rank":
        state[key] = state[key].reshape(1, -1)
    elif bad == "dtype":
        state[key] = state[key].to(torch.float16)
    elif bad == "nonzero":
        state[key][-1] = 0.001
    elif bad == "negative-zero":
        state[key][-1] = -0.0
    save_file(state, file, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="A_log padding"):
        KimiLinearQModel.loader.from_pretrained(tmp_path, local_files_only=True, ignore_mismatched_sizes=True)
    state[key] = original
    save_file(state, file, metadata={"format": "pt"})
    restored = KimiLinearQModel.loader.from_pretrained(tmp_path, local_files_only=True)
    torch.testing.assert_close(restored.state_dict()[key], original, rtol=0, atol=0)
    torch.testing.assert_close(restored(torch.tensor([[1, 4, 7]])).logits, model(torch.tensor([[1, 4, 7]])).logits)


@pytest.mark.parametrize("kind", ["attnres", "k3"])
def test_padded_a_log_disk_offload_restore_and_resave(tmp_path, kind):
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    config = tiny_config() if kind == "attnres" else tiny_k3()
    text = config if kind == "attnres" else config.text_config
    text.linear_attn_config["num_heads"] = 96
    model = (KimiLinearForCausalLM(config) if kind == "attnres" else KimiK3ForConditionalGeneration(config)).eval()
    model.save_pretrained(tmp_path)
    state = load_file(tmp_path / "model.safetensors")
    prefix = "model.layers.0.self_attn" if kind == "attnres" else "language_model.model.layers.0.self_attn"
    key = prefix + ".A_log"
    expected = state[key].clone()
    state[key] = torch.cat((expected, torch.zeros(32)))
    save_file(state, tmp_path / "model.safetensors", metadata={"format": "pt"})
    loader = KimiLinearQModel.loader if kind == "attnres" else KimiK3QModel.loader
    offload = tmp_path / "offload"
    device_map = {
        name: ("disk" if name == prefix else "cpu")
        for name, module in model.named_modules()
        if name and (not list(module.children()) or list(module.parameters(recurse=False)))
    }
    device_map[prefix] = "disk"
    device_map = {name: device for name, device in device_map.items() if not name.startswith(prefix + ".")}
    restored = loader.from_pretrained(tmp_path, local_files_only=True, device_map=device_map, offload_folder=offload)
    assert restored.get_parameter(key).device.type == "meta"
    converted = offload / f"{key}.safetensors"
    with safe_open(converted, framework="pt") as file:
        torch.testing.assert_close(file.get_tensor(key), expected, rtol=0, atol=0)
    ids = torch.tensor([[1, 4, 7]])
    torch.testing.assert_close(restored(ids, use_cache=False).logits, model(ids, use_cache=False).logits)
    assert restored.get_parameter(key).device.type == "meta"
    saved = tmp_path / "restored"
    restored.save_pretrained(saved)
    reloaded = loader.from_pretrained(saved, local_files_only=True)
    torch.testing.assert_close(reloaded.state_dict()[key], expected, rtol=0, atol=0)
    torch.testing.assert_close(reloaded(ids, use_cache=False).logits, model(ids, use_cache=False).logits)
    assert not list(tmp_path.rglob("*.bin"))


@pytest.mark.parametrize("beams", [1, 3])
@pytest.mark.parametrize("mode", ["mla", "hybrid", "kda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("text_only_row", [False, True])
def test_padded_mixed_images_generate_matches_independent_merge_and_text_oracle(mode, beams, dtype, text_only_row):
    model = KimiK3ForConditionalGeneration(tiny_k3(mode=mode)).to(dtype).eval()
    ids = torch.tensor([[0, 0, 1, 7] if text_only_row else [0, 1, 31, 7], [1, 31, 31, 8]])
    mask = ids.ne(0).long()
    grids = torch.tensor([[1, 4, 4], [1, 2, 2]] if text_only_row else [[1, 2, 2], [1, 4, 4], [1, 2, 2]])
    pixels = torch.randn(20 if text_only_row else 24, 3, 2, 2)
    # Independently substitute each placeholder, then left-pad complete rows.
    with torch.no_grad():
        features = model.mm_projector(model._extract_image_features(pixels, grids))
        rows, image = [], 0
        for row, valid in zip(ids, mask):
            pieces = []
            for token in row[valid.bool()]:
                if token == 31:
                    pieces.append(features[image])
                    image += 1
                else:
                    pieces.append(model.get_input_embeddings()(token.view(1)))
            rows.append(torch.cat(pieces))
        length = max(row.shape[0] for row in rows)
        embeddings = torch.stack([F.pad(row, (0, 0, length - len(row), 0)) for row in rows])
        expanded_mask = torch.stack(
            [F.pad(torch.ones(len(row), dtype=torch.long), (length - len(row), 0)) for row in rows]
        )
        assert length == 7 and int(expanded_mask[0].sum()) == (2 if text_only_row else 3)
        oracle_positions = (expanded_mask.cumsum(-1) - 1).masked_fill(expanded_mask == 0, 1)
        oracle = model.language_model(
            inputs_embeds=embeddings, attention_mask=expanded_mask, position_ids=oracle_positions, use_cache=False
        ).logits
        actual = model(ids, attention_mask=mask, pixel_values=pixels, grid_thws=grids, use_cache=False).logits
        torch.testing.assert_close(actual[expanded_mask.bool()], oracle[expanded_mask.bool()])
        expected = model.language_model.generate(
            inputs_embeds=embeddings,
            attention_mask=expanded_mask,
            max_new_tokens=3,
            num_beams=beams,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
            return_dict_in_generate=True,
            output_scores=True,
        )
        observed_masks = []

        def observe(module, args, kwargs):
            observed_masks.append((kwargs["attention_mask"].clone(), kwargs["position_ids"].clone()))

        handle = model.language_model.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            result = model.generate(
                ids,
                attention_mask=mask,
                pixel_values=pixels,
                grid_thws=grids,
                max_new_tokens=3,
                num_beams=beams,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
                return_dict_in_generate=True,
                output_scores=True,
            )
        finally:
            handle.remove()
        torch.testing.assert_close(result.sequences[:, -3:], expected.sequences[:, -3:], rtol=0, atol=0)
        for actual_score, expected_score in zip(result.scores, expected.scores):
            tolerance = 2e-6 if dtype == torch.float32 else 0.003
            torch.testing.assert_close(actual_score, expected_score, atol=tolerance, rtol=tolerance)
        assert [m.shape[1] for m, p in observed_masks] == [7, 8, 9]
        for step, (observed, positions) in enumerate(observed_masks):
            wanted = F.pad(expanded_mask, (0, step), value=1).repeat_interleave(beams, 0)
            torch.testing.assert_close(observed, wanted, rtol=0, atol=0)
            wanted_positions = (wanted.cumsum(-1) - 1).masked_fill(wanted == 0, 1)
            torch.testing.assert_close(positions, wanted_positions if step == 0 else wanted_positions[:, -1:])


@pytest.mark.parametrize("assign", [False, True])
def test_padded_a_log_direct_state_load_fresh_existing_assign_and_retry(assign):
    config = tiny_config()
    config.linear_attn_config["num_heads"] = 96
    original = KimiLinearForCausalLM(config).eval()
    state = {name: value.clone() for name, value in original.state_dict().items()}
    for name in state:
        if name.endswith(".self_attn.A_log"):
            state[name] = torch.cat([state[name], torch.zeros(32)])
    target = KimiLinearForCausalLM(config).eval()
    for _ in range(2):
        target.load_state_dict(state, assign=assign)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
    key = "model.layers.0.self_attn.A_log"
    state[key][-1] = 0.01
    with pytest.raises(ValueError, match="A_log padding"):
        target.load_state_dict(state, assign=assign)
    state[key][-1] = 0
    target.load_state_dict(state, assign=assign)
    ids = torch.tensor([[1, 4, 7]])
    torch.testing.assert_close(target(ids, use_cache=False).logits, original(ids, use_cache=False).logits)


@pytest.mark.parametrize("mask_kind", ["raw", "expanded", "none"])
@pytest.mark.parametrize("return_dict", [False, True])
def test_multimodal_bad_cached_mask_is_transactional_and_retry_matches_full(mask_kind, return_dict):
    model = KimiK3ForConditionalGeneration(tiny_k3()).eval()
    ids = torch.tensor([[0, 1, 31, 7], [1, 4, 31, 8]])
    mask = ids.ne(0).long()
    media = {"pixel_values": torch.randn(20, 3, 2, 2), "grid_thws": torch.tensor([[1, 4, 4], [1, 2, 2]])}
    with torch.no_grad():
        prefix = model(ids, attention_mask=mask, **media, use_cache=True, return_dict=return_dict)
        cache = prefix.past_key_values if return_dict else prefix[1]
        before = copy.deepcopy(cache)
        tokens = torch.tensor([[5, 9], [6, 10]])
        with pytest.raises(ValueError, match="cached prefix"):
            model(tokens, past_key_values=cache, attention_mask=torch.ones(2, 2))
        assert cache.seen_tokens == before.seen_tokens == 7
        torch.testing.assert_close(cache.multimodal_attention_mask, before.multimodal_attention_mask)
        for key in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
            for actual, expected in zip(getattr(cache, key), getattr(before, key)):
                if isinstance(actual, tuple):
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b)
                elif actual is not None:
                    torch.testing.assert_close(actual, expected)
        raw = F.pad(mask, (0, 2), value=1)
        expanded = F.pad(before.multimodal_attention_mask, (0, 2), value=1)
        full = model(torch.cat((ids, tokens), -1), attention_mask=raw, **media, use_cache=False).logits
        attention = {"raw": raw, "expanded": expanded, "none": None}[mask_kind]
        result = model(tokens, past_key_values=cache, attention_mask=attention).logits
        torch.testing.assert_close(result, full[:, -2:], atol=2e-6, rtol=2e-6)
        assert cache.seen_tokens == 9
        torch.testing.assert_close(cache.multimodal_attention_mask, expanded)
        permutation = torch.tensor([1, 0, 1])
        cache.reorder_cache(permutation)
        torch.testing.assert_close(cache.multimodal_attention_mask, expanded[permutation])
        torch.testing.assert_close(cache.multimodal_input_mask, raw[permutation])


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_native_linear_model_retains_native_implementation_and_real_source_tokenizer(
    tmp_path, monkeypatch, trust, bypass
):
    from shutil import copyfile

    from gptqmodel.models.ports.kimi_k3.tokenization_kimi import KimiLinearTokenizer

    asset = (
        Path.home() / ".cache/huggingface/hub/models--moonshotai--Kimi-Linear-48B-A3B-Instruct/snapshots" / LINEAR_PIN
    )
    if not (asset / "tiktoken.model").is_file():
        pytest.skip("pinned ordinary Kimi-Linear tokenizer asset is not cached")
    if bypass is None:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    else:
        monkeypatch.setenv(BYPASS_ENV, bypass)
    config = native_config()
    config.vocab_size = 163842
    original = NativeModel(config).eval()
    original.save_pretrained(tmp_path)
    marker = hostile_checkpoint(tmp_path, config)
    for name in ("tokenizer_config.json", "tiktoken.model", "chat_template.jinja"):
        copyfile(asset / name, tmp_path / name)
    wrapper = GPTQModel.load(
        str(tmp_path), device="cpu", backend="torch", trust_remote_code=trust, dtype=torch.float32
    )
    assert type(wrapper) is KimiLinearNativeQModel and type(wrapper.model) is NativeModel
    assert type(wrapper.model.config) is NativeConfig
    assert type(wrapper.tokenizer.tokenizer) is KimiLinearTokenizer
    expected_ids = [17172, 3287, 20944, 2050, 4643, 8201]
    actual_ids = wrapper.tokenizer.encode("I'm sure we'll see it's fine")
    assert actual_ids == expected_ids
    ids = torch.tensor([actual_ids])
    with torch.no_grad():
        torch.testing.assert_close(
            wrapper.model(ids, use_cache=False).logits, original(ids, use_cache=False).logits, rtol=0, atol=0
        )
    assert not marker.exists()


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_k3_published_padding_and_padded_images_first_load_retry_and_reload(
    tmp_path, monkeypatch, trust, bypass
):
    from safetensors.torch import load_file, save_file

    if bypass is None:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    else:
        monkeypatch.setenv(BYPASS_ENV, bypass)
    config = tiny_k3(vocab_size=512, pad_token_id=259, bos_token_id=256, eos_token_id=257)
    config.text_config.linear_attn_config["num_heads"] = 96
    config.media_placeholder_token_id = 266
    original = KimiK3ForConditionalGeneration(config).eval()
    original.save_pretrained(tmp_path)
    state = load_file(tmp_path / "model.safetensors")
    keys = [name for name in state if name.endswith(".self_attn.A_log")]
    original_state = {name: value.clone() for name, value in state.items()}
    for key in keys:
        state[key] = torch.cat((state[key], torch.zeros(32)))
    save_file(state, tmp_path / "model.safetensors", metadata={"format": "pt"})
    marker = hostile_checkpoint(tmp_path, config)
    write_tokenizer_assets(tmp_path)
    write_image_processor(tmp_path)
    state[keys[0]][-1] = 0.01
    save_file(state, tmp_path / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(ValueError, match="A_log padding"):
        GPTQModel.load(str(tmp_path), device="cpu", backend="torch", trust_remote_code=trust, dtype=torch.float32)
    state[keys[0]][-1] = 0
    save_file(state, tmp_path / "model.safetensors", metadata={"format": "pt"})
    wrapper = GPTQModel.load(
        str(tmp_path), device="cpu", backend="torch", trust_remote_code=trust, dtype=torch.float32
    )
    assert type(wrapper) is KimiK3QModel
    for name, value in wrapper.model.state_dict().items():
        torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)
    ids = torch.tensor([[259, 1, 266, 7], [1, 266, 266, 8]])
    mask = ids.ne(259).long()
    media = {"pixel_values": torch.randn(24, 3, 2, 2), "grid_thws": torch.tensor([[1, 2, 2], [1, 4, 4], [1, 2, 2]])}
    with torch.no_grad():
        expected = original.generate(
            ids, attention_mask=mask, **media, num_beams=2, max_new_tokens=3, eos_token_id=None, do_sample=False
        )
        result = wrapper.generate(
            input_ids=ids,
            attention_mask=mask,
            **media,
            num_beams=2,
            max_new_tokens=3,
            eos_token_id=None,
            do_sample=False,
        )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    saved = tmp_path / "canonical"
    wrapper.model.save_pretrained(saved)
    wrapper.tokenizer.save_pretrained(saved)
    wrapper.processor.save_pretrained(saved)
    reloaded = GPTQModel.load(str(saved), device="cpu", backend="torch", trust_remote_code=trust, dtype=torch.float32)
    for key in keys:
        assert reloaded.model.state_dict()[key].shape == (96,)
        torch.testing.assert_close(reloaded.model.state_dict()[key], original_state[key], rtol=0, atol=0)
    assert not marker.exists() and not list(tmp_path.rglob("*.bin"))


def native_config():
    return NativeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=4,
        linear_head_dim=4,
        linear_num_heads=2,
        num_experts=4,
        num_experts_per_token=2,
        num_expert_group=1,
        topk_group=1,
        num_shared_experts=1,
        mlp_layer_types=["dense", "sparse"],
        layer_types=["full_attention", "full_attention"],
        pad_token_id=0,
    )


def hostile_checkpoint(path, config):
    config.save_pretrained(path)
    payload = json.loads((path / "config.json").read_text())
    payload["auto_map"] = dict.fromkeys(
        ("AutoConfig", "AutoModelForCausalLM", "AutoTokenizer", "AutoProcessor", "AutoImageProcessor"), "evil.Remote"
    )
    payload["checkpoint_tag"] = "original-raw-metadata"
    (path / "config.json").write_text(json.dumps(payload))
    marker = path / "EXECUTED"
    (path / "evil.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('remote')\n"
    )
    return marker


@pytest.mark.parametrize("kind", ["native", "attnres", "k3"])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_dispatch_and_checkpoint_retry(tmp_path, monkeypatch, kind, trust, bypass):
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")  # Existing security exception.
    config, expected, definition = {
        "native": (native_config(), NativeModel, KimiLinearNativeQModel),
        "attnres": (tiny_config(), KimiLinearForCausalLM, KimiLinearQModel),
        "k3": (tiny_k3(), KimiK3ForConditionalGeneration, KimiK3QModel),
    }[kind]
    marker = hostile_checkpoint(tmp_path, config)
    raw = (tmp_path / "config.json").read_text()
    loaded = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(loaded) is type(config)
    assert loaded.checkpoint_tag == "original-raw-metadata"
    assert loaded.auto_map["AutoConfig"] == "evil.Remote"
    if kind == "native":
        assert not hasattr(loaded, "attn_res_block_size")
    assert check_and_get_model_definition(tmp_path, trust_remote_code=trust) is definition
    assert (tmp_path / "config.json").read_text() == raw and not marker.exists()
    loader = KimiK3QModel.loader if kind == "k3" else KimiLinearQModel.loader
    with pytest.raises(OSError):
        loader.from_pretrained(tmp_path, trust_remote_code=trust, local_files_only=True)
    original = loader.from_config(loaded, trust_remote_code=trust).eval()
    assert type(original) is expected
    original.save_pretrained(tmp_path)
    marker = hostile_checkpoint(tmp_path, loaded)
    restored = loader.from_pretrained(tmp_path, trust_remote_code=trust, code_revision="evil", local_files_only=True)
    assert type(restored) is expected
    ids = torch.tensor([[1, 5, 7, 9]])
    torch.testing.assert_close(restored(ids, use_cache=False).logits, original(ids, use_cache=False).logits)
    assert set(original.state_dict()) == set(restored.state_dict())
    assert not list(tmp_path.glob("*.bin")) and list(tmp_path.glob("*.safetensors"))
    assert not marker.exists()


@pytest.mark.parametrize("mode", ["mla", "hybrid", "kda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_real_forward_cache_chunks_generation_and_mask_retry(mode, dtype):
    model = KimiLinearForCausalLM(tiny_config(mode)).to(dtype).eval()
    ids = torch.tensor([[1, 4, 6, 8, 9], [1, 3, 5, 7, 10]])
    with torch.no_grad():
        full = model(ids, use_cache=False).logits
        assert torch.isfinite(full).all() and full.dtype == dtype
        prefix = model(ids[:, :2], use_cache=True)
        cache = prefix.past_key_values
        before = copy.deepcopy(cache)
        with pytest.raises(ValueError, match="cached prefix"):
            model(ids[:, 2:3], past_key_values=cache, attention_mask=torch.ones(2, 1))
        assert cache.seen_tokens == before.seen_tokens == 2
        suffix = model(ids[:, 2:], past_key_values=cache).logits
        tolerance = 2e-6 if dtype == torch.float32 else 0.002
        torch.testing.assert_close(suffix, full[:, 2:], rtol=tolerance, atol=tolerance)
        assert cache.get_seq_length() == ids.shape[1]
        assert not any("cache" in key for key in model.state_dict())
        # GenerationMixin must consume this model's hybrid cache, not HF DynamicCache.
        expected = ids.clone()
        for _ in range(2):
            expected = torch.cat((expected, model(expected, use_cache=False).logits[:, -1].argmax(-1)[:, None]), 1)
        actual = model.generate(ids, max_new_tokens=2, do_sample=False, eos_token_id=None)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("mode", ["mla", "hybrid", "kda"])
def test_left_padded_batch_cache_and_beam_reorder(mode):
    model = KimiLinearForCausalLM(tiny_config(mode)).eval()
    ids = torch.tensor([[0, 0, 1, 4, 6], [1, 3, 5, 7, 9]])
    mask = ids.ne(0).long()
    with torch.no_grad():
        batched = model(ids, attention_mask=mask, use_cache=True)
        single = model(ids[:1, 2:], use_cache=False).logits
        torch.testing.assert_close(batched.logits[:1, 2:], single, rtol=2e-5, atol=2e-6)
        cache = batched.past_key_values
        index = torch.tensor([1, 0, 1])
        cache.reorder_cache(index)
        next_ids = torch.tensor([[11], [12], [13]])
        continued = model(
            next_ids,
            past_key_values=cache,
            attention_mask=torch.cat((mask[index], torch.ones(3, 1, dtype=torch.long)), 1),
        ).logits
        full = model(
            torch.cat((ids[index], next_ids), 1),
            attention_mask=torch.cat((mask[index], torch.ones(3, 1, dtype=torch.long)), 1),
            use_cache=False,
        ).logits
        torch.testing.assert_close(continued[:, -1], full[:, -1], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("q_rank", [None, 4])
@pytest.mark.parametrize("gated", [False, True])
def test_mla_attention_against_float64_tensor_oracle(q_rank, gated):
    from gptqmodel.models.ports.kimi_k3.modeling_kimi_linear import KimiMLAAttention

    config = tiny_config("mla", q_lora_rank=q_rank, mla_use_output_gate=gated)
    config._attn_implementation = "eager"
    module = KimiMLAAttention(config, 0).eval()
    inputs = torch.randn(2, 5, 16)
    mask = torch.full((1, 1, 5, 5), -torch.inf).triu(1)

    def linear(x, m):
        return F.linear(x.double(), m.weight.double())

    def norm(x, m):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + m.variance_epsilon) * m.weight.double()

    q = (
        linear(inputs, module.q_proj)
        if q_rank is None
        else linear(norm(linear(inputs, module.q_a_proj), module.q_a_layernorm), module.q_b_proj)
    )
    q = q.reshape(2, 5, 2, 8).transpose(1, 2)
    latent, rotary = linear(inputs, module.kv_a_proj_with_mqa).split((4, 4), -1)
    kv = linear(norm(latent, module.kv_a_layernorm), module.kv_b_proj).reshape(2, 5, 2, 8).transpose(1, 2)
    k, v = kv.split((4, 4), -1)
    k = torch.cat((k, rotary[:, None].expand(-1, 2, -1, -1)), -1)
    output = ((q @ k.transpose(-2, -1) / 8**0.5 + mask).softmax(-1) @ v).transpose(1, 2).reshape(2, 5, 8)
    if gated:
        output *= linear(inputs, module.g_proj).sigmoid()
    expected = linear(output, module.o_proj)
    actual = module(inputs, attention_mask=mask)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=1e-6)


def write_tokenizer_assets(path):
    vocab = path / "tiktoken.model"
    vocab.write_text("".join(f"{base64.b64encode(bytes([i])).decode()} {i}\n" for i in range(256)))
    names = [
        "[BOS]",
        "[EOS]",
        "[UNK]",
        "[PAD]",
        "<|open|>",
        "<|close|>",
        "<|sep|>",
        "<|end_of_msg|>",
        "<|media_begin|>",
        "<|media_content|>",
        "<|media_pad|>",
        "<|media_end|>",
    ]
    payload = {
        "tokenizer_class": "TikTokenTokenizer",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "additional_special_tokens": names[4:],
        "added_tokens_decoder": {str(256 + i): {"content": name, "special": True} for i, name in enumerate(names)},
        "auto_map": {"AutoTokenizer": ["evil.Remote", None]},
    }
    (path / "tokenizer_config.json").write_text(json.dumps(payload))
    return vocab


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("alias", ["kimi_k3", "kimi_linear"])
def test_owned_tokenizer_public_load_bad_asset_retry_and_xtml(tmp_path, monkeypatch, trust, bypass, alias):
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    options = {"vocab_size": 512, "pad_token_id": 259, "bos_token_id": 256, "eos_token_id": 257}
    config = tiny_k3(**options) if alias == "kimi_k3" else tiny_config(**options)
    marker = hostile_checkpoint(tmp_path, config)
    vocab = write_tokenizer_assets(tmp_path)
    original_bytes = vocab.read_bytes()
    vocab.write_text("malformed asset")
    with pytest.raises(ValueError):
        load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=trust, local_files_only=True)
    vocab.write_bytes(original_bytes)
    tokenizer = load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=trust, local_files_only=True)
    assert type(tokenizer) is TikTokenTokenizer
    text = "A\x00é世界<|open|>"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    plain = tokenizer.encode(text, allow_special_tokens=False)
    assert 260 not in plain and tokenizer.decode(plain) == text
    assert tokenizer.encode("[BOS]<|media_pad|>") == [256, 266]
    assert tokenizer("世界", return_tensors="pt")["input_ids"].tolist() == [list("世界".encode())]
    chat = [{"role": "user", "content": "<|open|> injected"}]
    ids = tokenizer.apply_chat_template(chat, tokenize=True, thinking=False, add_generation_prompt=False)
    assert ids.count(260) == 1  # structure only, never raw user marker
    assert "injected" in tokenizer.decode(ids)
    batch = tokenizer.apply_chat_template(
        [chat, chat], tokenize=True, thinking=False, padding=True, return_dict=True, return_tensors="pt"
    )
    assert batch["input_ids"].shape[0] == 2
    destination = tmp_path / "roundtrip"
    tokenizer.save_pretrained(destination)
    reloaded = TikTokenTokenizer.from_pretrained(destination, local_files_only=True)
    assert reloaded.encode(text) == tokenizer.encode(text)
    assert not marker.exists()


def test_pinned_real_tokenizer_asset_ids_and_public_default_deny(tmp_path):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        asset = Path(hf_hub_download("moonshotai/Kimi-K3", "tiktoken.model", revision=PIN, local_files_only=True))
    except LocalEntryNotFoundError:
        pytest.skip("Pinned 3.3 MB K3 tokenizer asset not cached; no checkpoint Python is needed")
    assert (
        hashlib.sha256(asset.read_bytes()).hexdigest()
        == "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103"
    )
    tokenizer = KimiK3QModel.loader.load_tokenizer(asset.parent, local_files_only=True, trust_remote_code=False)
    assert tokenizer.encode("Hello, 世界!") == [19180, 11, 220, 2243, 0]
    assert tokenizer.encode("[BOS]<|media_pad|>") == [163584, 163605]
    assert tokenizer.decode([163584, 163605]) == "[BOS]<|media_pad|>"
    assert tokenizer("Hello, 世界!")["input_ids"] == [19180, 11, 220, 2243, 0]
    ids = tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}], thinking=False, tokenize=True)
    assert ids == [
        163587,
        2778,
        6244,
        878,
        2482,
        1,
        163589,
        18699,
        163588,
        2778,
        163589,
        163586,
        163587,
        2778,
        6244,
        878,
        69702,
        1,
        163589,
        163587,
        12092,
        163589,
    ]
    from shutil import copyfile

    payload = json.loads((FIXTURES / "checkpoint_config.json").read_text())
    config = KimiK3Config.from_dict(payload)
    marker = hostile_checkpoint(tmp_path, config)
    copyfile(asset, tmp_path / "tiktoken.model")
    copyfile(asset.parent / "tokenizer_config.json", tmp_path / "tokenizer_config.json")
    real = load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=False, local_files_only=True)
    assert type(real) is TikTokenTokenizer and real.pad_token_id == 163839
    assert real.encode("Hello, 世界!") == [19180, 11, 220, 2243, 0]
    (tmp_path / "tiktoken.model").write_text("corrupt pinned vocabulary")
    with pytest.raises(ValueError):
        load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=False, local_files_only=True)
    copyfile(asset, tmp_path / "tiktoken.model")
    repaired = load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=False, local_files_only=True)
    assert repaired.encode("Hello, 世界!") == real.encode("Hello, 世界!")
    assert not marker.exists()


def write_image_processor(path):
    image_processor = KimiK3VisionProcessor(
        {
            "patch_size": 2,
            "merge_kernel_size": 2,
            "in_patch_limit": 64,
            "patch_limit_on_one_side": 8,
            "fixed_output_tokens": None,
            "image_mean": [0.5] * 3,
            "image_std": [0.5] * 3,
        }
    )
    image_processor.save_pretrained(path)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_processor_real_pixels_multimodal_forward_cache_and_safetensors(tmp_path, monkeypatch, trust, bypass):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    config = tiny_k3()
    write_tokenizer_assets(tmp_path)
    write_image_processor(tmp_path)
    marker = hostile_checkpoint(tmp_path, config)
    processor = KimiK3QModel.loader.load_processor(tmp_path, local_files_only=True, trust_remote_code=trust)
    image = Image.new("RGB", (8, 8), color=(255, 0, 128))
    inputs = processor(medias=[{"type": "image", "image": image}], text="look <|kimi_image_placeholder|>")
    assert inputs["grid_thws"].tolist() == [[1, 4, 4]]
    assert inputs["pixel_values"].shape == (16, 3, 2, 2)
    with pytest.raises(ValueError, match="unsupported media"):
        processor(medias=[{"type": "video", "video": []}], text="video")
    # The model's fixture vocabulary is deliberately small; tokenizer/media are
    # independently real, so give the exact native placeholder ID to the decoder.
    ids = torch.tensor([[1, 31, 4]])
    model = KimiK3ForConditionalGeneration(config).eval()
    media = {key: inputs[key] for key in ("pixel_values", "grid_thws")}
    with torch.no_grad():
        result = model(ids, **media, use_cache=True)
        assert result.logits.shape == (1, 6, 32)
        assert torch.isfinite(result.logits).all()
        assert result.past_key_values.seen_tokens == 6
        features = model.mm_projector(model._extract_image_features(**media))
        embedding = model.get_input_embeddings()(ids)
        merged = torch.cat((embedding[:, :1], features[0][None], embedding[:, 2:]), 1)
        oracle = model.language_model(inputs_embeds=merged, use_cache=False).logits
        torch.testing.assert_close(result.logits, oracle)
        token = torch.tensor([[5]])
        suffix = model(token, past_key_values=result.past_key_values).logits
        full = model(torch.cat((ids, token), 1), **media, use_cache=False).logits
        torch.testing.assert_close(suffix[:, -1], full[:, -1], rtol=2e-5, atol=2e-6)
        generated = model.generate(
            ids, **media, attention_mask=torch.ones_like(ids), max_new_tokens=2, do_sample=False, eos_token_id=None
        )
        assert generated.shape == (1, 5)
        beams = model.generate(
            ids,
            **media,
            attention_mask=torch.ones_like(ids),
            num_beams=2,
            max_new_tokens=2,
            do_sample=False,
            eos_token_id=None,
        )
        assert beams.shape == (1, 5)
    model.save_pretrained(tmp_path)
    restored = KimiK3QModel.loader.from_pretrained(tmp_path, local_files_only=True)
    torch.testing.assert_close(restored(ids, **media, use_cache=False).logits, oracle)
    assert not marker.exists()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_public_gptqmodel_load_uses_owned_model_tokenizer_processor(tmp_path, dtype):
    config = tiny_k3(vocab_size=512, pad_token_id=259, bos_token_id=256, eos_token_id=257)
    config.media_placeholder_token_id = 266
    model = KimiK3ForConditionalGeneration(config).eval()
    model.save_pretrained(tmp_path)
    marker = hostile_checkpoint(tmp_path, config)
    write_tokenizer_assets(tmp_path)
    write_image_processor(tmp_path)
    wrapper = GPTQModel.load(str(tmp_path), device="cpu", backend="torch", trust_remote_code=False, dtype=dtype)
    assert type(wrapper) is KimiK3QModel
    assert type(wrapper.model) is KimiK3ForConditionalGeneration
    assert type(wrapper.tokenizer.tokenizer) is TikTokenTokenizer
    assert wrapper.processor.tokenizer is wrapper.tokenizer.tokenizer
    ids = torch.tensor([[1, 5, 7]])
    actual = wrapper.model(ids, use_cache=False).logits
    assert actual.dtype == dtype
    for name, parameter in wrapper.model.named_parameters():
        assert parameter.dtype == (torch.float32 if name.endswith(("A_log", "dt_bias")) else dtype)
    baseline = copy.deepcopy(model).to(dtype)
    for name, parameter in model.named_parameters():
        if name.endswith(("A_log", "dt_bias")):
            baseline.get_parameter(name).data = parameter.detach().clone()
    torch.testing.assert_close(actual, baseline(ids, use_cache=False).logits)
    assert not marker.exists()


@pytest.mark.parametrize("lower_bound", [None, -5.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_kda_against_independent_float64_transition_matrix_oracle(lower_bound, dtype):
    from gptqmodel.models.ports.kimi_k3.eager_kda import kda_forward

    q, k, v, g = (torch.randn(2, 6, 2, 4).to(dtype) for _ in range(4))
    beta = torch.randn(2, 6, 2)
    A, bias = torch.tensor([0.3, 1.2]), torch.randn(8)
    initial = torch.randn(2, 2, 4, 4)
    output, state = kda_forward(q, k, v, g, beta, A, bias, initial_state=initial, lower_bound=lower_bound)
    query = q.double() / (q.double().square().sum(-1, keepdim=True) + 1e-6).sqrt() / 2
    key = k.double() / (k.double().square().sum(-1, keepdim=True) + 1e-6).sqrt()
    gate = g.double() + bias.double().reshape(2, 4)
    gate = (
        -A.double().exp()[:, None] * F.softplus(gate)
        if lower_bound is None
        else lower_bound * (A.double().exp()[:, None] * gate).sigmoid()
    )
    state64 = initial.double()
    values = []
    identity = torch.eye(4)
    for t in range(6):
        ki = key[:, t, :, :, None]
        bi = beta[:, t, :, None, None].double().sigmoid()
        transition = (identity - bi * (ki @ ki.transpose(-2, -1))) @ torch.diag_embed(gate[:, t].exp())
        state64 = transition @ state64 + bi * (ki @ v[:, t, :, None].double())
        values.append((query[:, t, :, None] @ state64).squeeze(-2))
    expected = torch.stack(values, 1)
    torch.testing.assert_close(
        output,
        expected.to(dtype),
        rtol=1e-5 if dtype == torch.float32 else 0.002,
        atol=2e-6 if dtype == torch.float32 else 0.002,
    )
    torch.testing.assert_close(state.double(), state64, rtol=1e-5, atol=3e-6)
    first, midway = kda_forward(
        q[:, :3], k[:, :3], v[:, :3], g[:, :3], beta[:, :3], A, bias, initial_state=initial, lower_bound=lower_bound
    )
    second, final = kda_forward(
        q[:, 3:], k[:, 3:], v[:, 3:], g[:, 3:], beta[:, 3:], A, bias, initial_state=midway, lower_bound=lower_bound
    )
    torch.testing.assert_close(torch.cat((first, second), 1), output, rtol=0, atol=0)
    torch.testing.assert_close(final, state, rtol=0, atol=0)


@pytest.mark.parametrize("gate_rank", [False, True])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_delta_attention_gate_layouts_are_checkpoint_exact_and_roundtrip(tmp_path, gate_rank, lower_bound):
    config = tiny_config()
    config.linear_attn_config.update(use_full_rank_gate=gate_rank, gate_lower_bound=lower_bound)
    model = KimiLinearForCausalLM(config).eval()
    keys = set(model.model.layers[0].self_attn.state_dict())
    assert ("g_proj.weight" in keys) == gate_rank
    assert ("g_a_proj.weight" in keys and "g_b_proj.weight" in keys) == (not gate_rank)
    ids = torch.tensor([[1, 7, 8, 9]])
    full = model(ids, use_cache=False).logits
    cache = model(ids[:, :2], use_cache=True).past_key_values
    torch.testing.assert_close(model(ids[:, 2:], past_key_values=cache).logits, full[:, 2:], rtol=2e-5, atol=2e-6)
    model.save_pretrained(tmp_path)
    loaded = KimiLinearQModel.loader.from_pretrained(tmp_path, local_files_only=True)
    torch.testing.assert_close(loaded(ids, use_cache=False).logits, full)


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"model_type": "kimi_linear", "use_attn_residuals": True}, "explicit"),
        ({"model_type": "kimi_linear", "attn_res_block_size": 0}, "positive"),
        ({"model_type": "kimi_linear", "attn_res_block_size": -2}, "positive"),
        ({"model_type": "kimi_linear", "attn_res_block_size": 2, "num_nextn_predict_layers": 1}, "MTP"),
    ],
)
def test_unsupported_config_fail_closed_then_retry(tmp_path, payload, match):
    (tmp_path / "config.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        load_model_config(tmp_path, trust_remote_code=False)
    tiny_config().save_pretrained(tmp_path)
    assert type(load_model_config(tmp_path)) is KimiLinearConfig


def test_foreign_cache_and_ambiguous_inputs_fail_before_state_mutation():
    from transformers.cache_utils import DynamicCache

    model = KimiLinearForCausalLM(tiny_config()).eval()
    ids = torch.tensor([[1, 4, 8]])
    with pytest.raises(ValueError, match="KimiDynamicCache"):
        model(ids, past_key_values=DynamicCache())
    with pytest.raises(ValueError, match="exactly one"):
        model(ids, inputs_embeds=model.get_input_embeddings()(ids))
    assert torch.isfinite(model(ids).logits).all()


def test_real_vision_eager_sdpa_agree_with_unequal_packed_image_lengths():
    from gptqmodel.models.ports.kimi_k3.modeling_kimi_k3 import eager_attention, sdpa_attention

    q, k, v = (torch.randn(8, 2, 4) for _ in range(3))
    cumulative = torch.tensor([0, 3, 8], dtype=torch.int32)
    actual = sdpa_attention(q, k, v, cumulative, cumulative)
    expected = eager_attention(q, k, v, cumulative, cumulative)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("model_type,definition", [("kimi_k3", KimiK3QModel), ("kimi_linear", KimiLinearQModel)])
def test_public_config_ignores_checkpoint_code(tmp_path, model_type, definition):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "attn_res_block_size": 2,
                "auto_map": {"AutoConfig": "evil.RemoteConfig"},
            }
        )
    )
    config = load_model_config(tmp_path, trust_remote_code=False)
    assert isinstance(definition.loader, LocalModelLoader), "K3 must select package-owned code"
    assert config is not None and type(config).__module__.startswith("gptqmodel.models.ports.kimi_k3")


def test_missing_checkpoint_tensors_use_tf5_init_and_loaded_tensors_are_kept(tmp_path):
    from safetensors.torch import load_file, save_file

    model = KimiK3ForConditionalGeneration(tiny_k3()).eval()
    model.save_pretrained(tmp_path)
    state = load_file(tmp_path / "model.safetensors")
    prefix = "language_model.model.layers."
    missing = {
        prefix + "0.self_attn.A_log": None,
        prefix + "0.self_attn.dt_bias": torch.zeros,
        prefix + "0.input_layernorm.weight": torch.ones,
        prefix + "0.self_attn.o_norm.weight": torch.ones,
        prefix + "1.block_sparse_moe.gate.weight": None,
        prefix + "1.block_sparse_moe.gate.e_score_correction_bias": torch.zeros,
    }
    for name in missing:
        state.pop(name)
    save_file(state, tmp_path / "model.safetensors", metadata={"format": "pt"})
    restored = KimiK3QModel.loader.from_pretrained(tmp_path, local_files_only=True).state_dict()
    for name, value in state.items():
        torch.testing.assert_close(restored[name], value, rtol=0, atol=0)
    for name, fill in missing.items():
        assert torch.isfinite(restored[name]).all(), name
        if fill is not None:
            torch.testing.assert_close(restored[name], fill(restored[name].shape), rtol=0, atol=0)
    a_log = restored[prefix + "0.self_attn.A_log"]
    assert bool(((a_log >= 0) & (a_log <= math.log(16))).all())
    # Non-persistent vision buffers are rebuilt by _init_weights after meta-device loading.
    reloaded = KimiK3QModel.loader.from_pretrained(tmp_path, local_files_only=True)
    torch.testing.assert_close(
        reloaded.vision_tower.patch_embed.pos_emb.time_weight,
        model.vision_tower.patch_embed.pos_emb.time_weight,
        rtol=0,
        atol=0,
    )


def test_tf5_output_contract_logits_to_keep_and_mask_sizes(monkeypatch):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        text, k3 = tiny_config(), tiny_k3()
    monkeypatch.setattr(type(text), "use_return_dict", property(lambda self: pytest.fail("deprecated")), raising=False)
    monkeypatch.setattr(type(k3), "use_return_dict", property(lambda self: pytest.fail("deprecated")), raising=False)
    ids = torch.tensor([[1, 4, 6, 8]])
    for model in (KimiLinearForCausalLM(text).eval(), KimiK3ForConditionalGeneration(k3).eval()):
        with torch.no_grad():
            full = model(ids, use_cache=True)
            as_tuple = model(ids, use_cache=False, return_dict=False)
        assert isinstance(as_tuple, tuple)
        torch.testing.assert_close(as_tuple[0], full.logits, rtol=0, atol=0)
        assert full.past_key_values.get_mask_sizes(3, 0) == (ids.shape[1] + 3, 0)
    with torch.no_grad():
        last = model.language_model(ids, use_cache=False, logits_to_keep=1).logits
    assert last.shape[1] == 1
    torch.testing.assert_close(last, full.logits[:, -1:], rtol=1e-6, atol=1e-6)
