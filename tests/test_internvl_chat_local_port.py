# SPDX-License-Identifier: Apache-2.0
"""Real legacy wrapper qualification; no checkpoint Python is imported."""

import hashlib
import json
import re
import shutil
from functools import lru_cache
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import DynamicCache, StaticCache

from gptqmodel import CalibrationConfig
from gptqmodel.models.definitions.internvl_chat import InternVLChatQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.internvl_chat.configuration_internvl_chat import InternVLChatConfig
from gptqmodel.models.ports.internvl_chat.modeling_internvl_chat import InternVLChatModel, decoder_model_class


ARCHITECTURES = ["LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "Phi3ForCausalLM", "InternLM2ForCausalLM"]

FIXTURES = Path(__file__).parent / "fixtures/ports/internvl_chat"
TOKENIZER_ASSETS = json.loads((FIXTURES / "tokenizer_assets.json").read_text())
SOURCE_FAST_REFERENCE = json.loads((FIXTURES / "tokenizer_source_reference.json").read_text())
# sha256 of sorted "<key> <dim>x<dim>..." lines of the pinned 1B (Qwen3-0.6B decoder) checkpoint index, 657 tensors.
CHECKPOINT_INVENTORY_SHA256 = "e167115f15f19d5e5569d222ce1847cbac627bdf6b44cf9f7e09627343817394"


@lru_cache(maxsize=None)
def tokenizer_asset_dir(architecture):
    """Download data only, at exact released revisions, and verify every byte."""
    from huggingface_hub import hf_hub_download

    spec = TOKENIZER_ASSETS[architecture]
    for name, digest in spec["files"].items():
        resolved = Path(hf_hub_download(spec["repo"], name, revision=spec["revision"]))
        assert hashlib.sha256(resolved.read_bytes()).hexdigest() == digest, name
    return resolved.parent


@lru_cache(maxsize=None)
def expected_tokenizer_module(architecture):
    if architecture == "InternLM2ForCausalLM":
        return "gptqmodel.models.ports.internlm2"
    if architecture in {"LlamaForCausalLM", "Phi3ForCausalLM"}:
        # The vendored LlamaTokenizer, registered with Tokenicer as
        # Transformers' own class by load_hf_tokenizer.
        return "gptqmodel.models.ports.native.llama."
    return "gptqmodel.models.ports.native.qwen2."


def source_token_ids(architecture, text):
    """Independent source SentencePiece or Qwen byte-BPE equations."""
    path = tokenizer_asset_dir(architecture)
    metadata = json.loads((path / "tokenizer_config.json").read_text())
    if (path / "tokenizer.model").exists():
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=str(path / "tokenizer.model"))
        encode = sp.encode
        if architecture == "LlamaForCausalLM":
            # Source Llama fast converter prepends a dummy whitespace even
            # though this Yi SentencePiece model has add_dummy_prefix=False.
            def encode(value):
                return sp.encode(" " + value) if value else []
    else:
        import tiktoken

        codes = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        extras = [i for i in range(256) if i not in codes]
        decoder = {chr(c): c for c in codes} | {chr(256 + i): c for i, c in enumerate(extras)}
        vocab = json.loads((path / "vocab.json").read_text())
        ranks = {bytes(decoder[c] for c in word): rank for word, rank in vocab.items()}
        reference = tiktoken.Encoding(name="independent-qwen2", mergeable_ranks=ranks,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", special_tokens={})
        encode = reference.encode
    special = {value["content"]: int(key) for key, value in metadata["added_tokens_decoder"].items()}
    fragments = re.split("(" + "|".join(map(re.escape, sorted(special, key=len, reverse=True))) + ")", text)
    for index in range(1, len(fragments), 2):
        token = metadata["added_tokens_decoder"][str(special[fragments[index]])]
        if token.get("lstrip"):
            fragments[index - 1] = fragments[index - 1].rstrip()
        if token.get("rstrip"):
            fragments[index + 1] = fragments[index + 1].lstrip()
    ids = []
    for fragment in fragments:
        ids.extend([special[fragment]] if fragment in special else encode(fragment))
    default_bos = architecture == "InternLM2ForCausalLM"
    if metadata.get("add_bos_token", default_bos):
        ids.insert(0, special[metadata["bos_token"]])
    if metadata.get("add_eos_token", False):
        ids.append(special[metadata["eos_token"]])
    return ids


def tiny_config(architecture="Qwen3ForCausalLM", **kwargs):
    llm = {"architectures": [architecture], "vocab_size": 48, "hidden_size": 16, "intermediate_size": 24,
               "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
               "head_dim": 8, "max_position_embeddings": 64, "original_max_position_embeddings": 32,
               "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "tie_word_embeddings": False,
               "num_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 24}
    vision = {"hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2, "num_attention_heads": 2,
                  "image_size": 8, "patch_size": 2, "use_flash_attn": False, "norm_type": "layer_norm", "qk_normalization": False}
    vision.update(kwargs.pop("vision", {}))
    llm.update(kwargs.pop("llm", {}))
    return InternVLChatConfig(vision_config=vision, llm_config=llm, ps_version=kwargs.pop("ps_version", "v2"), **kwargs)


def save_tiny_checkpoint(path, architecture="Qwen3ForCausalLM", **kwargs):
    assets = tokenizer_asset_dir(architecture)
    metadata = json.loads((assets / "tokenizer_config.json").read_text())
    if (assets / "tokenizer.model").exists():
        import sentencepiece as spm

        vocab_size = spm.SentencePieceProcessor(model_file=str(assets / "tokenizer.model")).get_piece_size()
    else:
        vocab_size = len(json.loads((assets / "vocab.json").read_text()))
    vocab_size = max(vocab_size, max(map(int, metadata["added_tokens_decoder"])) + 1)
    llm = kwargs.pop("llm", {})
    llm.update(vocab_size=vocab_size)
    kwargs["llm"] = llm
    kwargs.setdefault("template", json.loads((assets / "config.json").read_text())["template"])
    model = InternVLChatModel(tiny_config(architecture, **kwargs)).eval()
    model.save_pretrained(path, safe_serialization=True)
    for name in TOKENIZER_ASSETS[architecture]["files"]:
        if name != "config.json":
            shutil.copyfile(assets / name, path / name)
    return model


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("text", ["Hello 世界!", "  spaces\n中文 café 🐍 123",
    "<|im_start|>user\n<img><IMG_CONTEXT></img>\nDescribe 世界.<|im_end|>\n<|im_start|>assistant\n"])
def test_pinned_real_tokenizer_source_ids_and_roundtrip(architecture, text, tmp_path):
    from gptqmodel.utils.hf import load_hf_tokenizer

    assets = tokenizer_asset_dir(architecture)
    config = InternVLChatConfig.from_pretrained(assets)
    tokenizer = load_hf_tokenizer(str(assets), model_config=config, local_files_only=True)
    expected = source_token_ids(architecture, text)
    assert tokenizer(text)["input_ids"] == expected
    assert tokenizer.__class__.__module__.startswith(expected_tokenizer_module(architecture))
    metadata = json.loads((assets / "tokenizer_config.json").read_text())
    chat_specials = ("<|user|>", "<|end|>") if architecture == "Phi3ForCausalLM" else ("<|im_start|>", "<|im_end|>")
    for name in ("<img>", "</img>", "<IMG_CONTEXT>", *chat_specials):
        expected_id = next(int(i) for i, value in metadata["added_tokens_decoder"].items() if value["content"] == name)
        assert tokenizer.convert_tokens_to_ids(name) == expected_id
    decoded = tokenizer.decode(expected, skip_special_tokens=False)
    tokenizer.save_pretrained(tmp_path)
    config.save_pretrained(tmp_path)
    reloaded = load_hf_tokenizer(str(tmp_path), model_config=config, local_files_only=True)
    assert reloaded(text)["input_ids"] == expected
    assert reloaded.decode(expected, skip_special_tokens=False) == decoded
    assert not list(tmp_path.glob("*.py"))


@pytest.mark.parametrize("architecture", SOURCE_FAST_REFERENCE)
@pytest.mark.parametrize("text", ["Hello 世界!", "  spaces\n中文 café 🐍 123", "x\n\n世界", " café 🐍", "", "\n", " spaces", "nospace"])
def test_source_fast_sentencepiece_whitespace_and_serialized_pipeline(architecture, text, tmp_path):
    from gptqmodel.utils.hf import load_hf_tokenizer

    reference = SOURCE_FAST_REFERENCE[architecture]
    assert reference["repo"] == TOKENIZER_ASSETS[architecture]["repo"]
    assert reference["revision"] == TOKENIZER_ASSETS[architecture]["revision"]
    # Captured from inspected source Transformers 4.37.2 LlamaTokenizerFast,
    # independently reproduced with SentencePiece prefix/whitespace equations.
    expected = reference["source_fast_ids"][text]
    assert source_token_ids(architecture, text) == expected
    assets = tokenizer_asset_dir(architecture)
    config = InternVLChatConfig.from_pretrained(assets)
    tokenizer = load_hf_tokenizer(str(assets), model_config=config, local_files_only=True)
    assert tokenizer(text)["input_ids"] == expected
    tokenizer.save_pretrained(tmp_path)
    config.save_pretrained(tmp_path)
    restored = load_hf_tokenizer(str(tmp_path), model_config=config, local_files_only=True)
    assert restored(text)["input_ids"] == expected


@pytest.mark.parametrize("architecture", SOURCE_FAST_REFERENCE)
def test_historical_sentencepiece_profile_explicit_caller_optout(architecture, tmp_path):
    from transformers import LlamaTokenizer

    from gptqmodel.utils.hf import load_hf_tokenizer

    assets = tokenizer_asset_dir(architecture)
    config = InternVLChatConfig.from_pretrained(assets)
    text = "  spaces\n中文 café 🐍 123"
    native = LlamaTokenizer.from_pretrained(str(assets), local_files_only=True)
    native_ids = native(text)["input_ids"]
    historical_ids = SOURCE_FAST_REFERENCE[architecture]["source_fast_ids"][text]
    assert native_ids != historical_ids
    opted_out = load_hf_tokenizer(str(assets), model_config=config, local_files_only=True, normalize_sentencepiece=False)
    assert opted_out(text)["input_ids"] == native_ids
    default = load_hf_tokenizer(str(assets), model_config=config, local_files_only=True)
    assert default(text)["input_ids"] == historical_ids
    # A pipeline deliberately authored with the opt-out remains authoritative,
    # including when reloaded under this definition's default opt-in profile.
    opted_out.save_pretrained(tmp_path)
    config.save_pretrained(tmp_path)
    restored = load_hf_tokenizer(str(tmp_path), model_config=config, local_files_only=True)
    assert restored(text)["input_ids"] == native_ids


@pytest.mark.parametrize("architecture", [name for name in ARCHITECTURES if name not in SOURCE_FAST_REFERENCE])
def test_historical_profile_leaves_internlm_and_native_qwen_unchanged(architecture):
    from gptqmodel.utils.hf import load_hf_tokenizer

    assets = tokenizer_asset_dir(architecture)
    config = InternVLChatConfig.from_pretrained(assets)
    for enabled in (False, True):
        tokenizer = load_hf_tokenizer(str(assets), model_config=config, local_files_only=True, normalize_sentencepiece=enabled)
        for text in ("Hello 世界!", "  spaces\n中文 café 🐍 123", "<img><IMG_CONTEXT></img>"):
            assert tokenizer(text)["input_ids"] == source_token_ids(architecture, text)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_real_tokenizer_public_security_and_model_image_reload(architecture, trust, bypass, tmp_path, monkeypatch):
    from PIL import Image
    from tokenicer import Tokenicer

    from gptqmodel import GPTQModel

    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    original = save_tiny_checkpoint(tmp_path, architecture, max_dynamic_patch=2, use_thumbnail=True)
    for name in ("config.json", "tokenizer_config.json"):
        file = tmp_path / name
        payload = json.loads(file.read_text())
        payload["auto_map"] = {"AutoConfig": "evil.Config", "AutoModel": "evil.Model", "AutoTokenizer": ["evil.Tokenizer", None],
            "AutoProcessor": "evil.Processor", "AutoImageProcessor": "evil.ImageProcessor"}
        if "llm_config" in payload:
            payload["llm_config"]["auto_map"] = payload["auto_map"]
            payload["vision_config"]["auto_map"] = payload["auto_map"]
        file.write_text(json.dumps(payload))
    (tmp_path / "evil.py").write_text("raise AssertionError('checkpoint Python executed')\n")
    normalized = []
    original_load = Tokenicer.load.__func__

    def record_normalization(cls, value, *args, **kwargs):
        result = original_load(cls, value, *args, **kwargs)
        normalized.append((result, kwargs))
        return result

    monkeypatch.setattr(Tokenicer, "load", classmethod(record_normalization))
    wrapper = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert len(normalized) == 1 and wrapper.tokenizer is normalized[0][0]
    assert normalized[0][1]["normalize_sentencepiece"] is True
    text = "Hello 世界!"
    expected_ids = source_token_ids(architecture, text)
    assert wrapper.tokenizer(text)["input_ids"] == expected_ids
    ids = torch.tensor([expected_ids])
    with torch.no_grad():
        torch.testing.assert_close(wrapper.model(input_ids=ids).logits, original(input_ids=ids).logits, atol=0, rtol=0)
    # The released wrapper uses this definition-owned PIL processor, not AutoProcessor.
    batch = wrapper.prepare_inputs_for_conversation([{"role": "user", "content": [Image.new("RGB", (16, 8), (35, 71, 143)),
        {"type": "text", "text": "Describe 世界."}]}])
    assert batch["pixel_values"].shape == (3, 3, 8, 8)
    assert (batch["input_ids"] == batch["img_context_token_id"]).sum().item() == 12
    wrapper.model.img_context_token_id = int(batch["img_context_token_id"])
    with torch.no_grad():
        expected = wrapper.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], pixel_values=batch["pixel_values"]).logits
    saved = tmp_path / "saved"
    wrapper.model.save_pretrained(saved, safe_serialization=True)
    wrapper.tokenizer.save_pretrained(saved)
    reloaded = GPTQModel.load(str(saved), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert len(normalized) == 2 and reloaded.tokenizer is normalized[1][0]
    assert normalized[1][1]["normalize_sentencepiece"] is True
    actual_batch = reloaded.prepare_inputs_for_conversation([{"role": "user", "content": [Image.new("RGB", (16, 8), (35, 71, 143)),
        {"type": "text", "text": "Describe 世界."}]}])
    assert torch.equal(actual_batch["input_ids"], batch["input_ids"])
    assert torch.equal(actual_batch["pixel_values"], batch["pixel_values"])
    reloaded.model.img_context_token_id = int(actual_batch["img_context_token_id"])
    with torch.no_grad():
        actual = reloaded.model(input_ids=actual_batch["input_ids"], attention_mask=actual_batch["attention_mask"], pixel_values=actual_batch["pixel_values"]).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert list(saved.glob("*.safetensors")) and not list(saved.glob("*.bin"))
    assert not list(saved.glob("*.py"))


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_real_tokenizer_missing_asset_and_foreign_metadata_repair_retry(architecture, tmp_path):
    from gptqmodel import GPTQModel

    save_tiny_checkpoint(tmp_path, architecture)
    required = "tokenizer.model" if (tmp_path / "tokenizer.model").exists() else (
        "tokenizer.json" if (tmp_path / "tokenizer.json").exists() else "vocab.json")
    # Remove the complete available representation, not merely an optional fallback.
    moved = {}
    for name in ("tokenizer.model", "tokenizer.json", "vocab.json"):
        file = tmp_path / name
        if file.exists():
            moved[name] = file.read_bytes()
            file.unlink()
    with pytest.raises((FileNotFoundError, OSError), match="tokenizer.model|vocab.json"):
        GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32)
    assert required in moved
    for name, data in moved.items():
        (tmp_path / name).write_bytes(data)
    metadata_path = tmp_path / "tokenizer_config.json"
    metadata = json.loads(metadata_path.read_text())
    original_class = metadata["tokenizer_class"]
    other_family = "Qwen2Tokenizer" if architecture == "InternLM2ForCausalLM" else "InternLM2Tokenizer"
    for invalid_class in ("ForeignTokenizer", other_family, [original_class]):
        metadata["tokenizer_class"] = invalid_class
        metadata_path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="does not support tokenizer_class"):
            GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=True)
    if architecture == "InternLM2ForCausalLM":
        metadata["tokenizer_class"] = "InternLM2TokenizerFast"
        metadata_path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="does not support tokenizer_class"):
            GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=True)
    metadata["tokenizer_class"] = original_class
    metadata_path.write_text(json.dumps(metadata))
    repaired = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32)
    assert repaired.tokenizer("Hello 世界!")["input_ids"] == source_token_ids(architecture, "Hello 世界!")


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_real_tokenizer_public_selected_subfolder_and_root_decoy(architecture, tmp_path):
    from gptqmodel import GPTQModel

    selected = tmp_path / "selected"
    save_tiny_checkpoint(selected, architecture)
    # Root discovery would choose unselected config and a forbidden tokenizer;
    # only the selected folder contains the actual tiny checkpoint tensors.
    tiny_config("Qwen3ForCausalLM").save_pretrained(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "ForeignTokenizer"}))
    wrapper = GPTQModel.load(str(tmp_path), subfolder="selected", local_files_only=True, device="cpu", dtype=torch.float32)
    assert wrapper.model.config.llm_config.architectures == [architecture]
    expected_module = expected_tokenizer_module(architecture)
    assert wrapper.tokenizer.tokenizer.__class__.__module__.startswith(expected_module)
    text = "  spaces\n中文 café 🐍 123"
    assert wrapper.tokenizer(text)["input_ids"] == source_token_ids(architecture, text)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_gptq_load_uses_local_config_and_model_with_hostile_automap(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel import GPTQModel

    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1" if bypass else "0")
    original = save_tiny_checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["auto_map"] = {"AutoConfig": "malicious.Config", "AutoModel": "malicious.Model", "AutoModelForCausalLM": "malicious.Model"}
    config["llm_config"]["auto_map"] = config["auto_map"]
    config_path.write_text(json.dumps(config))
    (tmp_path / "malicious.py").write_text("raise AssertionError('checkpoint Python executed')\n")
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert type(loaded) is InternVLChatQModel
    assert type(loaded.model) is InternVLChatModel and type(loaded.model.config) is InternVLChatConfig
    ids = torch.tensor([[7, 8, 9]])
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(input_ids=ids).logits, original(input_ids=ids).logits, atol=0, rtol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_public_awq_packs_and_reloads_actual_decoder_weights(tmp_path, architecture):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    original = save_tiny_checkpoint(tmp_path, architecture, llm={"hidden_size": 32, "intermediate_size": 64,
                                    "head_dim": 16, "moe_intermediate_size": 64})
    vision_before = {name: value.clone() for name, value in original.vision_model.state_dict().items()}
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None,
                            offload_to_disk=False), backend=BACKEND.TORCH, dtype=torch.float16)
    calibration = [{"input_ids": torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]]),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    wrapper.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    expected_count = {"InternLM2ForCausalLM": 10, "Phi3ForCausalLM": 10, "Qwen3MoeForCausalLM": 20}.get(architecture, 14)
    packed = {name: module for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert len(packed) == expected_count
    assert all(name.startswith("language_model.model.layers.") for name in packed)
    assert all(module.qweight.dtype == torch.int32 for module in packed.values())
    for name, tensor in wrapper.model.vision_model.state_dict().items():
        torch.testing.assert_close(tensor, vision_before[name].to(tensor.dtype), atol=0, rtol=0)
    with torch.inference_mode():
        expected = wrapper.model(**calibration[0], use_cache=False).logits
    assert torch.isfinite(expected).all()
    saved = tmp_path / "quantized"
    wrapper.save_quantized(str(saved))
    assert list(saved.glob("*.safetensors")) and not list(saved.glob("*.bin"))
    reloaded = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert type(reloaded.model) is InternVLChatModel
    assert sum(isinstance(module, AwqTorchLinear) for module in reloaded.model.modules()) == expected_count
    with torch.inference_mode():
        torch.testing.assert_close(reloaded.model(**calibration[0], use_cache=False).logits, expected, atol=2e-4, rtol=2e-3)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_real_pil_conversation_preparation_and_image_capture(tmp_path, architecture):
    from PIL import Image

    from gptqmodel import GPTQModel

    save_tiny_checkpoint(tmp_path, architecture, max_dynamic_patch=2, use_thumbnail=True)
    wrapper = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32)
    conversation = [{"role": "user", "content": [Image.new("RGB", (16, 8), color=(35, 71, 143)),
                                                 {"type": "text", "text": "What is in this image?"}]}]
    batch = wrapper.prepare_inputs_for_conversation(conversation)
    assert batch["pixel_values"].shape == (3, 3, 8, 8)  # two tiles and the thumbnail
    assert torch.isfinite(batch["pixel_values"]).all()
    assert int((batch["input_ids"] == batch["img_context_token_id"]).sum()) == 12
    moved = wrapper.move_input_capture_example(batch, torch.device("cpu"))
    with torch.inference_mode():
        actual = wrapper.run_input_capture(moved, use_cache=False, data_device=torch.device("cpu"))
        expected = wrapper.model.generate(pixel_values=moved["pixel_values"], input_ids=moved["input_ids"],
                    attention_mask=moved["attention_mask"], max_new_tokens=1, do_sample=False,
                    eos_token_id=batch["eos_token_id"], pad_token_id=0)
    assert actual.shape == (1, 1)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_real_decoder_multimodal_oracle_cache_generation_and_reload(architecture, tmp_path):
    torch.manual_seed(19)
    model = InternVLChatModel(tiny_config(architecture)).eval()
    model.img_context_token_id = 3
    ids = torch.tensor([[7, 3, 3, 3, 3, 8]])
    pixels = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        # Native decoder is the numerical oracle; insertion is independently
        # performed at the checkpoint-defined context positions.
        expected_embeds = model.language_model.get_input_embeddings()(ids).clone()
        expected_embeds[:, 1:5] = model.extract_feature(pixels)
        oracle = model.language_model(inputs_embeds=expected_embeds, use_cache=False).logits
        result = model(pixel_values=pixels, input_ids=ids, use_cache=True, output_hidden_states=True)
        torch.testing.assert_close(result.logits, oracle, atol=1e-6, rtol=1e-5)
        assert type(model.language_model) is decoder_model_class(architecture)
        assert result.hidden_states is not None
        continued = model(input_ids=torch.tensor([[9]]), past_key_values=result.past_key_values,
                          attention_mask=torch.ones(1, 7), use_cache=True).logits
        full = model(pixel_values=pixels, input_ids=torch.cat((ids, torch.tensor([[9]])), dim=1)).logits
        torch.testing.assert_close(continued[:, -1], full[:, -1], atol=1e-6, rtol=1e-5)
        assert result.past_key_values.get_seq_length() == 7
        generated = model.generate(pixel_values=pixels, input_ids=ids, max_new_tokens=2, do_sample=False, eos_token_id=None)
        direct = model.language_model.generate(inputs_embeds=expected_embeds, max_new_tokens=2, do_sample=False, eos_token_id=None)
        assert torch.equal(generated, direct)
        assert generated.shape == (1, 2)
        for cache_class in (DynamicCache, StaticCache):
            cache = cache_class(config=model.config.llm_config, **({"max_cache_len": 12} if cache_class is StaticCache else {}))
            model(pixel_values=pixels, input_ids=ids, past_key_values=cache, use_cache=True)
            continuation = model(input_ids=torch.tensor([[9]]), attention_mask=torch.ones(1, 7),
                                 past_key_values=cache, use_cache=True).logits
            torch.testing.assert_close(continuation[:, -1], full[:, -1], atol=1e-6, rtol=1e-5)
            cache.reset()
            short_ids = torch.tensor([[7, 8]])
            retried = model(input_ids=short_ids, past_key_values=cache, use_cache=True).logits
            torch.testing.assert_close(retried, model(input_ids=short_ids, use_cache=False).logits, atol=1e-6, rtol=1e-5)
        beam = model.generate(pixel_values=pixels, input_ids=ids, max_new_tokens=2, num_beams=2, do_sample=False, eos_token_id=None)
        direct_beam = model.language_model.generate(inputs_embeds=expected_embeds, max_new_tokens=2, num_beams=2,
                                                   do_sample=False, eos_token_id=None)
        assert torch.equal(beam, direct_beam)
        assert model(input_ids=torch.tensor([[7, 8]]), labels=torch.tensor([[7, 8]])).loss.isfinite()
        assert isinstance(model(input_ids=ids, return_dict=False), tuple)
    model.save_pretrained(tmp_path, safe_serialization=True)
    assert not list(tmp_path.glob("*.bin"))
    saved = load_file(tmp_path / "model.safetensors")
    if architecture != "Qwen3MoeForCausalLM":
        assert set(saved) == set(model.state_dict())
    else:
        # TF5 packs experts at runtime and saves the original checkpoint's
        # per-expert projections; those checkpoint owners remain the contract.
        for i in range(2):
            for expert in range(2):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    assert f"language_model.model.layers.{i}.mlp.experts.{expert}.{proj}.weight" in saved
    restored, info = InternVLChatQModel.loader.from_pretrained(tmp_path, output_loading_info=True, dtype=torch.float32)
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    restored.eval().img_context_token_id = 3
    with torch.no_grad():
        torch.testing.assert_close(restored(pixel_values=pixels, input_ids=ids).logits, oracle, atol=0, rtol=0)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_tied_embedding_checkpoint_aliases_reload_without_renaming(architecture, tmp_path):
    model = InternVLChatModel(tiny_config(architecture, llm={"tie_word_embeddings": True})).eval()
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    ids = torch.tensor([[7, 8, 9]])
    expected = model(input_ids=ids).logits
    model.save_pretrained(tmp_path, safe_serialization=True)
    restored, info = InternVLChatQModel.loader.from_pretrained(tmp_path, output_loading_info=True, dtype=torch.float32)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    torch.testing.assert_close(restored(input_ids=ids).logits, expected, atol=0, rtol=0)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_local_selection_ignores_malicious_automap_and_trust(tmp_path, monkeypatch, trust, bypass):
    # Existing security bypass is explicitly part of the public loader contract.
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1" if bypass else "0")
    cfg = tiny_config().to_dict()
    cfg["auto_map"] = {"AutoConfig": "malicious.Config", "AutoModel": "malicious.Model"}
    cfg["llm_config"]["auto_map"] = cfg["auto_map"]
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "malicious.py").write_text("raise AssertionError('checkpoint Python executed')\n")
    loader = InternVLChatQModel.loader
    assert isinstance(loader, LocalModelLoader)
    config = loader.load_config(tmp_path, trust_remote_code=trust, code_revision="bad")
    assert type(config) is InternVLChatConfig
    with torch.device("meta"):
        model = loader.from_config(config, trust_remote_code=trust, code_revision="bad")
    assert type(model) is InternVLChatModel
    assert not InternVLChatQModel.require_trust_remote_code


def test_image_failure_then_retry_preserves_state_and_matches_clean_model():
    model = InternVLChatModel(tiny_config()).eval()
    model.img_context_token_id = 3
    pixels = torch.randn(1, 3, 8, 8)
    ids = torch.tensor([[3, 3, 3, 3, 8]])
    original = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError, match="token count"):
        model(pixel_values=pixels, input_ids=ids[:, 1:])
    with torch.no_grad():
        good = model(pixel_values=pixels, input_ids=ids).logits
        flagged = model(pixel_values=torch.cat((pixels, pixels * 2)), input_ids=ids,
                        image_flags=torch.tensor([[1], [0]])).logits
    torch.testing.assert_close(good, flagged, atol=1e-6, rtol=1e-5)
    for key, value in original.items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("changes,match", [
    ({"llm": {"architectures": ["MaliciousModel"]}}, "Unsupported"),
    ({"llm": {"model_type": "llama"}}, "mismatch"),
    ({"downsample_ratio": 0.3}, "reciprocal"),
    ({"force_image_size": 7}, "divisible"),
    ({"ps_version": "bad"}, "ps_version"),
])
def test_invalid_config_rejected_before_model_allocation(changes, match):
    with pytest.raises(ValueError, match=match):
        tiny_config(**changes)


def test_mutated_config_revalidated_before_allocation():
    cfg = tiny_config()
    cfg.llm_config.architectures = ["Unknown"]
    with pytest.raises(ValueError, match="Unsupported"):
        InternVLChatModel(cfg)


def test_pinned_1b_checkpoint_parameter_names_and_shapes_on_meta():
    cfg = InternVLChatConfig(**json.loads((FIXTURES / "checkpoint_config.json").read_text()))
    with torch.device("meta"):
        model = InternVLChatModel(cfg)
    state = model.state_dict()
    assert len(state) == 657
    inventory = "\n".join(f"{key} {'x'.join(map(str, state[key].shape))}" for key in sorted(state))
    assert hashlib.sha256(inventory.encode()).hexdigest() == CHECKPOINT_INVENTORY_SHA256
    assert state["vision_model.embeddings.position_embedding"].shape == (1, 1025, 1024)
    assert state["vision_model.encoder.layers.0.attn.qkv.weight"].shape == (3072, 1024)
    assert state["mlp1.1.weight"].shape == (1024, 4096)
    assert state["language_model.model.layers.0.self_attn.q_norm.weight"].shape == (128,)


def _vision_oracle(model, pixels):
    """Independent functional eager equations from the pinned vision source."""
    state = model.state_dict()
    cfg = model.config
    patch = F.conv2d(pixels, state["embeddings.patch_embedding.weight"],
                     state["embeddings.patch_embedding.bias"], stride=cfg.patch_size)
    grid = patch.shape[-2:]
    x = patch.flatten(2).transpose(1, 2)
    x = torch.cat((state["embeddings.class_embedding"].expand(x.shape[0], -1, -1), x), dim=1)
    pos = state["embeddings.position_embedding"]
    side = cfg.image_size // cfg.patch_size
    pos_patch = pos[:, 1:].float().reshape(1, side, side, -1).permute(0, 3, 1, 2)
    pos_patch = F.interpolate(pos_patch, size=grid, mode="bicubic", align_corners=False)
    pos_patch = pos_patch.flatten(2).transpose(1, 2).to(pos.dtype)
    x = x + torch.cat((pos[:, :1], pos_patch), dim=1)
    states = [x]

    def norm(x, name):
        weight = state[name + ".weight"]
        if name + ".bias" in state:
            return F.layer_norm(x, (x.shape[-1],), weight, state[name + ".bias"], cfg.layer_norm_eps)
        y = x.float()
        return weight * (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + cfg.layer_norm_eps)).to(x.dtype)

    def linear(x, name):
        return F.linear(x, state[name + ".weight"], state.get(name + ".bias"))

    for index in range(cfg.num_hidden_layers):
        p = f"encoder.layers.{index}"
        y = norm(x, p + ".norm1")
        b, n, c = y.shape
        q, k, v = linear(y, p + ".attn.qkv").reshape(b, n, 3, cfg.num_attention_heads, -1).permute(2, 0, 3, 1, 4)
        if cfg.qk_normalization:
            q = norm(q.transpose(1, 2).flatten(-2), p + ".attn.q_norm").reshape(b, n, cfg.num_attention_heads, -1).transpose(1, 2)
            k = norm(k.transpose(1, 2).flatten(-2), p + ".attn.k_norm").reshape(b, n, cfg.num_attention_heads, -1).transpose(1, 2)
        score = ((q * (c // cfg.num_attention_heads) ** -0.5) @ k.transpose(-1, -2)).softmax(-1)
        y = (score @ v).transpose(1, 2).reshape(b, n, c)
        x = x + linear(y, p + ".attn.proj") * state[p + ".ls1"]
        y = F.gelu(linear(norm(x, p + ".norm2"), p + ".mlp.fc1"))
        x = x + linear(y, p + ".mlp.fc2") * state[p + ".ls2"]
        states.append(x)
    return states


@pytest.mark.parametrize("norm", ["layer_norm", "rms_norm"])
@pytest.mark.parametrize("qk", [False, True])
@pytest.mark.parametrize("ps_version", ["v1", "v2"])
def test_vision_computation_grid_order_and_hidden_layer_selection(norm, qk, ps_version):
    model = InternVLChatModel(tiny_config(vision={"norm_type": norm, "qk_normalization": qk},
                                         ps_version=ps_version, select_layer=1)).eval()
    # Different image size exercises learned positional-grid interpolation.
    pixels = torch.randn(2, 3, 12, 12)
    with torch.no_grad():
        oracle_states = _vision_oracle(model.vision_model, pixels)
        output = model.vision_model(pixels, output_hidden_states=True)
        for actual, expected in zip(output.hidden_states, oracle_states):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        # Source pixel shuffle is independently expressed by spatial grouping.
        grid = oracle_states[1][:, 1:].reshape(2, 6, 6, 16)
        oracle = grid.reshape(2, 3, 2, 3, 2, 16).permute(0, 1, 3, 2, 4, 5).reshape(2, 3, 3, 64)
        if ps_version == "v1":
            oracle = oracle.transpose(1, 2)
        oracle = model.mlp1(oracle.reshape(2, 9, 64))
        torch.testing.assert_close(model.extract_feature(pixels), oracle, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("kind", ["su", "yarn"])
def test_legacy_phi3_rope_short_long_and_reset_matches_equations(kind):
    import math

    cfg = tiny_config("Phi3ForCausalLM", llm={"rope_scaling": {
        "type": kind, "short_factor": [1.0] * 4, "long_factor": [2.0] * 4}})
    model = InternVLChatModel(cfg).eval()
    rope = model.language_model.model.rotary_emb
    x = torch.zeros(1, 1, 16)
    scale = 64 / 32
    amplitude = 0.1 * math.log(scale) + 1 if kind == "yarn" else math.sqrt(1 + math.log(scale) / math.log(32))
    for length in (4, 40, 4):
        pos = torch.arange(length).unsqueeze(0)
        cos, sin = rope(x, pos)
        inv = 1 / ((1 if length <= 32 else 2) * 10000 ** (torch.arange(0, 8, 2).float() / 8))
        freq = pos.unsqueeze(-1) * inv
        theta = torch.cat((freq, freq), dim=-1)
        torch.testing.assert_close(cos, theta.cos() * amplitude, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(sin, theta.sin() * amplitude, atol=1e-6, rtol=1e-5)


def test_vision_gradient_checkpointing_follows_transformers_toggle():
    torch.manual_seed(0)
    model = InternVLChatModel(tiny_config()).train()
    encoder = model.vision_model.encoder
    assert not encoder.gradient_checkpointing
    pixels = torch.randn(2, 3, 8, 8)
    model.vision_model(pixel_values=pixels).last_hidden_state.sum().backward()
    reference = [p.grad.clone() for p in model.vision_model.parameters()]
    model.zero_grad()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert all(layer.gradient_checkpointing for layer in encoder.layers)
    model.vision_model(pixel_values=pixels).last_hidden_state.sum().backward()
    for expected, param in zip(reference, model.vision_model.parameters()):
        torch.testing.assert_close(param.grad, expected)
    model.gradient_checkpointing_disable()
    assert not any(layer.gradient_checkpointing for layer in encoder.layers)
