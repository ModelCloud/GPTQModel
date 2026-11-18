# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Exact native Kimi-K2.5 loading, not a causal-LM architecture substitute."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    Kimi_K25Config,
    Kimi_K25ForConditionalGeneration,
    Kimi_K25ImageProcessor,
    Kimi_K25Processor,
    Kimi_K25VideoProcessor,
    PreTrainedTokenizerFast,
)

from gptqmodel import CalibrationConfig, GPTQModel
from gptqmodel.models.auto import MODEL_MAP, check_and_get_model_definition
from gptqmodel.models.definitions.kimi_k25 import KimiK25QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.kimi_k25.native import Kimi_K25ForConditionalGeneration as PackagedKimi
from gptqmodel.models.ports.kimi_k25.processing_kimi_k25 import Kimi_K25Processor as PackagedProcessor
from gptqmodel.models.ports.kimi_k25.tokenization_kimi import TikTokenTokenizer
from gptqmodel.utils.hf import load_hf_tokenizer, load_model_config, resolve_trust_remote_code
from gptqmodel.utils.security import BYPASS_ENV


TOKENIZER_FIXTURES = Path(__file__).resolve().parent / "fixtures/kimi_k25"


def save_real_tokenizer(path):
    for filename in ("tiktoken.model", "chat_template.jinja", "tokenizer_config.json"):
        asset = TOKENIZER_FIXTURES / filename
        shutil.copyfile(asset, path / asset.name)
    data = json.loads((path / "tokenizer_config.json").read_text())
    data["auto_map"] = {"AutoTokenizer": ["hostile.Tokenizer", None]}
    data["tokenizer_class"] = "HostileTokenizer"
    (path / "tokenizer_config.json").write_text(json.dumps(data))


def assert_shared_tokenizer_contract(tokenizer, reference):
    original_ids = reference.all_special_ids
    assert len(original_ids) == 16 and len(reference.extra_special_tokens) == 12
    assert tokenizer.extra_special_tokens == reference.extra_special_tokens + ["<|kimi_k25_video_placeholder|>"]
    assert set(tokenizer.all_special_ids) == set(original_ids) | {163840}
    assert tokenizer.convert_tokens_to_ids(reference.all_special_tokens) == original_ids
    assert tokenizer.convert_tokens_to_ids("<|kimi_k25_video_placeholder|>") == 163840
    ids = original_ids + reference.encode("Hello, 世界!") + [163840]
    assert tokenizer.decode(ids, skip_special_tokens=True) == "Hello, 世界!"
    assert tokenizer.batch_decode([ids], skip_special_tokens=True) == ["Hello, 世界!"]
    conversation = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "weather", "description": "Weather lookup",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
    for thinking in (False, True):
        for selected_tools in (None, tools):
            kwargs = {"thinking": thinking, "tools": selected_tools}
            assert tokenizer.apply_chat_template(conversation, **kwargs) == reference.apply_chat_template(
                conversation, **kwargs,
            )
            assert tokenizer.apply_chat_template(conversation, tokenize=True, **kwargs) == (
                reference.apply_chat_template(conversation, tokenize=True, **kwargs)
            )


def tiny_config(q_lora_rank=8, first_k_dense_replace=1, image_token_id=30, video_token_id=31,
                vision_patch_size=2, **overrides):
    text = {
        "model_type": "kimi_k2", "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
        "moe_intermediate_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_key_value_heads": 2, "n_routed_experts": 4, "n_shared_experts": 1,
        "num_experts_per_tok": 2, "n_group": 1, "topk_group": 1,
        "first_k_dense_replace": first_k_dense_replace, "kv_lora_rank": 8,
        "q_lora_rank": q_lora_rank, "qk_nope_head_dim": 4, "qk_rope_head_dim": 4,
        "v_head_dim": 8, "max_position_embeddings": 64, "pad_token_id": 0,
        "bos_token_id": 1, "eos_token_id": 2, "routed_scaling_factor": 2.827,
    }
    text.update(overrides)
    config = Kimi_K25Config(
        text_config=text,
        vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                       "num_attention_heads": 2, "patch_size": vision_patch_size, "pos_emb_height": 2,
                       "pos_emb_width": 2, "pos_emb_time": 4, "merge_kernel_size": [2, 2]},
        projection_hidden_size=16, image_token_id=image_token_id, video_token_id=video_token_id,
        vision_start_token_id=28, vision_end_token_id=29, tie_word_embeddings=False,
        pad_token_id=text["pad_token_id"], bos_token_id=text["bos_token_id"], eos_token_id=text["eos_token_id"],
    )
    config._attn_implementation = "eager"
    return config


def test_definition_constructs_exact_native_multimodal_task():
    config = tiny_config()
    with torch.device("meta"):
        model = KimiK25QModel.loader.from_config(config)
    assert type(model) is PackagedKimi
    assert PackagedKimi.forward is Kimi_K25ForConditionalGeneration.forward
    assert config.text_config.model_type == "deepseek_v3"
    assert next(model.parameters()).device.type == "meta"


def test_definition_owns_native_layer_path_and_latent_norms():
    assert KimiK25QModel.extract_layers_node() == ["model.language_model.layers"]
    blocks = KimiK25QModel.build_layer_modules(KimiK25QModel.module_tree, include_capture_only=True)
    excluded = {name for block in blocks for name in block if ":!" in name}
    assert {"self_attn.q_a_layernorm:!", "self_attn.kv_a_layernorm:!", "mlp.gate:!"} <= excluded


def save_processor(path):
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"w{i}": i for i in range(4, 28)})
    vocab.update({"[START]": 28, "[END]": 29, "<|media_pad|>": 30, "<|kimi_k25_video_placeholder|>": 31})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]",
        additional_special_tokens=["<|media_pad|>", "<|kimi_k25_video_placeholder|>"],
    )
    processor = Kimi_K25Processor(
        image_processor=Kimi_K25ImageProcessor(patch_size=2, size={"max_height": 2, "max_width": 2}),
        video_processor=Kimi_K25VideoProcessor(patch_size=2, size={"max_height": 2, "max_width": 2},
                                             do_sample_frames=False),
        tokenizer=tokenizer,
    )
    processor.save_pretrained(path)
    return processor


def split_expert_state(model):
    state = {}
    for name, tensor in model.state_dict().items():
        if name.endswith(".experts.gate_up_proj"):
            prefix = name.removesuffix("gate_up_proj")
            for i, expert in enumerate(tensor):
                gate, up = expert.chunk(2, dim=0)
                state[f"{prefix}{i}.gate_proj.weight"] = gate.clone()
                state[f"{prefix}{i}.up_proj.weight"] = up.clone()
        elif name.endswith(".experts.down_proj"):
            prefix = name.removesuffix("down_proj")
            for i, expert in enumerate(tensor):
                state[f"{prefix}{i}.down_proj.weight"] = expert.clone()
        else:
            state[name] = tensor.clone()
    return state


def released_state(model):
    """Independently encode released names, QKV fusion and complex-RoPE order."""
    result = {}
    qkv = {}
    for name, tensor in split_expert_state(model).items():
        if name.startswith("model.language_model."):
            name = name.replace("model.language_model.", "language_model.model.", 1)
        elif name == "lm_head.weight":
            name = "language_model.lm_head.weight"
        elif name.startswith("model.mm_projector."):
            name = name.removeprefix("model.").replace("in_proj", "proj.0").replace("out_proj", "proj.2")
        elif name.startswith("model.vision_tower."):
            name = name.removeprefix("model.")
            if ".layers." in name:
                name = name.replace(".layers.", ".encoder.blocks.")
                name = name.replace(".norm1.", ".norm0.").replace(".norm2.", ".norm1.")
                name = name.replace(".mlp.fc1.", ".mlp.fc0.").replace(".mlp.fc2.", ".mlp.fc1.")
                name = name.replace(".attn.proj.", ".wo.")
                if any(f".attn.{p}_proj." in name for p in "qkv"):
                    projection = name.split(".attn.")[1][0]
                    prefix, suffix = name.split(f".attn.{projection}_proj.")
                    if projection in "qk":
                        heads = model.config.vision_config.num_attention_heads
                        shape = tensor.shape
                        tensor = tensor.reshape(heads, 2, shape[0] // heads // 2, *shape[1:])
                        tensor = tensor.transpose(1, 2).reshape(shape)
                    qkv.setdefault(f"{prefix}.wqkv.{suffix}", {})[projection] = tensor
                    continue
            elif ".final_layernorm." in name:
                name = name.replace("vision_tower.", "vision_tower.encoder.", 1)
            name = name.replace("pos_emb.position_embeddings", "pos_emb.weight")
        result[name] = tensor.clone()
    for name, projections in qkv.items():
        result[name] = torch.cat([projections[p] for p in "qkv"]).contiguous()
    return result


def write_checkpoint(path, *, released=False, routing_bias=(.1, -.2, .3, -.4), **overrides):
    torch.manual_seed(76)
    reference = Kimi_K25ForConditionalGeneration(tiny_config(**overrides)).eval()
    with torch.no_grad():
        for layer in reference.model.language_model.layers:
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.e_score_correction_bias.copy_(torch.tensor(routing_bias))
    reference.save_pretrained(path, safe_serialization=True, save_original_format=False)
    if released:
        save_file(released_state(reference), path / "model.safetensors", metadata={"format": "pt"})
    save_processor(path)
    marker = path / "EXECUTED"
    (path / "hostile.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise AssertionError('checkpoint Python executed')\n"
    )
    payload = json.loads((path / "config.json").read_text())
    hostile_map = dict.fromkeys(
        ("AutoConfig", "AutoModel", "AutoModelForCausalLM", "AutoModelForImageTextToText",
         "AutoModelForMultimodalLM", "AutoProcessor", "AutoImageProcessor", "AutoVideoProcessor"), "hostile.Model"
    )
    payload["auto_map"] = hostile_map
    payload["text_config"]["model_type"] = "kimi_k2"
    for nested in ("text_config", "vision_config"):
        payload[nested]["auto_map"] = hostile_map
    (path / "config.json").write_text(json.dumps(payload))
    processor_payload = json.loads((path / "processor_config.json").read_text())
    processor_payload["auto_map"] = hostile_map
    (path / "processor_config.json").write_text(json.dumps(processor_payload))
    return reference, marker


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_load_ignores_checkpoint_python(tmp_path, monkeypatch, trust, bypass):
    reference, marker = write_checkpoint(
        tmp_path, vocab_size=163840, pad_token_id=163839, bos_token_id=163584, eos_token_id=163585,
    )
    save_real_tokenizer(tmp_path)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")  # Existing security bypass is explicitly under test.
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    assert {key for key, value in MODEL_MAP.items() if value is KimiK25QModel} == {"kimi_k25"}
    assert check_and_get_model_definition(str(tmp_path), trust_remote_code=trust) is KimiK25QModel
    assert resolve_trust_remote_code(tmp_path, trust_remote_code=trust) is False
    assert type(load_model_config(tmp_path, trust_remote_code=trust)) is Kimi_K25Config
    shell = KimiK25QModel.loader.from_config(load_model_config(tmp_path), trust_remote_code=trust)
    assert type(shell) is PackagedKimi
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert type(loaded) is KimiK25QModel
    assert type(loaded.model) is PackagedKimi
    assert type(loaded.processor) is PackagedProcessor
    assert type(loaded.tokenizer.tokenizer) is TikTokenTokenizer
    assert loaded.processor.tokenizer is loaded.tokenizer.tokenizer
    assert_shared_tokenizer_contract(loaded.tokenizer.tokenizer, TikTokenTokenizer.from_pretrained(
        TOKENIZER_FIXTURES,
    ))
    assert loaded.tokenizer.encode("Hello, 世界!") == [19180, 11, 220, 2243, 0]
    assert not loaded.trust_remote_code
    ids = torch.tensor([[1, 4, 5, 7]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids).logits, reference(ids).logits, rtol=1e-5, atol=1e-6)
    assert not marker.exists()


def test_shared_tokenizer_survives_failed_processor_and_two_live_consumers(tmp_path, monkeypatch):
    from transformers.models.kimi_k25.processing_kimi_k25 import Kimi_K25Processor as NativeProcessor

    _, marker = write_checkpoint(
        tmp_path, vocab_size=163840, pad_token_id=163839, bos_token_id=163584, eos_token_id=163585,
    )
    save_real_tokenizer(tmp_path)
    config = load_model_config(tmp_path)
    tokenizer = load_hf_tokenizer(tmp_path, model_config=config)
    reference = TikTokenTokenizer.from_pretrained(TOKENIZER_FIXTURES)
    assert tokenizer.all_special_ids == reference.all_special_ids
    initialize = NativeProcessor.__init__
    failed = []

    def fail_after_initialization(processor, *args, **kwargs):
        initialize(processor, *args, **kwargs)
        failed.append(processor)
        raise RuntimeError("injected failure after native processor initialization")

    with monkeypatch.context() as patch:
        patch.setattr(NativeProcessor, "__init__", fail_after_initialization)
        with pytest.raises(RuntimeError, match="injected failure after native processor"):
            KimiK25QModel.loader.load_processor(tmp_path, config=config, tokenizer=tokenizer)
    assert len(failed) == 1 and failed[0].tokenizer is tokenizer
    assert_shared_tokenizer_contract(tokenizer, reference)
    processors = [KimiK25QModel.loader.load_processor(tmp_path, config=config, tokenizer=tokenizer) for _ in range(2)]
    assert processors[0] is not processors[1]
    for processor in processors:
        assert processor.tokenizer is tokenizer
        assert processor.video_token_id == 163840 and processor.image_token_id == 163605
        assert_shared_tokenizer_contract(processor.tokenizer, reference)
        assert processor.batch_decode([[163587, 19180, 163605, 163586, 163840]],
                                      skip_special_tokens=True) == ["Hello"]
    assert not marker.exists()


@pytest.mark.parametrize("q_lora_rank", [None, 16], ids=["dense_q", "split_q"])
def test_public_awq_preserves_shared_tokenizer_across_two_save_reloads(tmp_path, q_lora_rank):
    from gptqmodel import BACKEND
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    _, marker = write_checkpoint(
        tmp_path, routing_bias=(.01, -.02, .03, -.04), q_lora_rank=q_lora_rank,
        vocab_size=163840, hidden_size=32, intermediate_size=32,
        moe_intermediate_size=16, kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16,
        image_token_id=163605, video_token_id=163840, pad_token_id=163839, bos_token_id=163584, eos_token_id=163585,
    )
    save_real_tokenizer(tmp_path)
    reference = TikTokenTokenizer.from_pretrained(TOKENIZER_FIXTURES)
    config = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=False)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=config, backend=BACKEND.TORCH,
                            dtype=torch.float16, attn_implementation="eager", trust_remote_code=False)
    assert wrapper.processor.tokenizer is wrapper.tokenizer.tokenizer
    assert_shared_tokenizer_contract(wrapper.tokenizer.tokenizer, reference)
    calibration = [wrapper.tokenizer(text, return_tensors="pt") for text in (
        "Hello, 世界! GPT-QModel 123456", "Quantization preserves special-token semantics.",
        "量化😀 学习语言模型的每一个专家。", "def weather(city: str): return city.lower()",
        "Paris, Tokyo, Москва, Αθήνα, القاهرة!", "Numbers: 9876543210 3.141592653589793238462643",
        "The quick brown fox jumps over the lazy dog.", "\t    \n{'role': 'user', 'content': 'Hello'}",
    )]
    calibration = [{key: value for key, value in sample.items() if key in {"input_ids", "attention_mask"}}
                   for sample in calibration]
    with torch.inference_mode():
        dense_logits = wrapper.model(**calibration[0], use_cache=False).logits
        assert torch.isfinite(dense_logits).all() and dense_logits.std() > .01
    norms = {name: value.clone() for name, value in wrapper.model.state_dict().items() if "layernorm" in name}
    wrapper.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    # AWQ scale search is active: decoder norms absorbed non-unit scales.
    assert any(not torch.equal(wrapper.model.state_dict()[name], value) for name, value in norms.items())
    packed_count = 26 if q_lora_rank is None else 28
    assert sum(isinstance(module, AwqTorchLinear) for module in wrapper.model.modules()) == packed_count
    with torch.inference_mode():
        expected = wrapper.model(**calibration[0], use_cache=False).logits
        assert torch.isfinite(expected).all()
    for attempt in range(2):
        saved = tmp_path / f"quantized-{attempt}"
        wrapper.save_quantized(str(saved))
        assert list(saved.glob("*.safetensors")) and not list(saved.glob("*.bin")) and not list(saved.glob("*.pt"))
        saved_tokenizer = TikTokenTokenizer.from_pretrained(saved)
        assert_shared_tokenizer_contract(saved_tokenizer, reference)
        serialized = json.loads((saved / "tokenizer_config.json").read_text())
        assert len(serialized["extra_special_tokens"]) == 13
        wrapper = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16,
                                attn_implementation="eager", trust_remote_code=False)
        assert wrapper.quantized and wrapper.processor.tokenizer is wrapper.tokenizer.tokenizer
        assert_shared_tokenizer_contract(wrapper.tokenizer.tokenizer, reference)
        assert sum(isinstance(module, AwqTorchLinear) for module in wrapper.model.modules()) == packed_count
        with torch.inference_mode():
            torch.testing.assert_close(wrapper.model(**calibration[0], use_cache=False).logits, expected,
                                       atol=2e-4, rtol=2e-3)
    assert not marker.exists()


def test_pinned_tokenizer_ids_chat_tools_and_roundtrip(tmp_path):
    assets = TOKENIZER_FIXTURES
    assert hashlib.sha256((assets / "tiktoken.model").read_bytes()).hexdigest() == (
        "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103"
    )
    tokenizer = TikTokenTokenizer.from_pretrained(assets)
    assert (tokenizer.vocab_size, tokenizer.bos_token_id, tokenizer.eos_token_id,
            tokenizer.pad_token_id, tokenizer.unk_token_id) == (163840, 163584, 163585, 163839, 163838)
    examples = {
        "Hello, 世界!": [19180, 11, 220, 2243, 0],
        "GPT-QModel\n量化 123456 😀": [111021, 46319, 3836, 198, 34302, 220, 6694, 12972, 70005, 222],
        "[BOS]hello[EOS]": [163584, 22931, 163585],
        "<|media_pad|>": [163605],
    }
    for text, ids in examples.items():
        assert tokenizer.encode(text) == ids
        assert tokenizer.decode(ids) == text
        assert tokenizer(text, return_tensors="pt")["input_ids"].tolist() == [ids]
    conversation = [{"role": "user", "content": "Hello"}]
    expected = "<|im_user|>user<|im_middle|>Hello<|im_end|><|im_assistant|>assistant<|im_middle|><think>"
    assert tokenizer.apply_chat_template(conversation) == expected
    assert tokenizer.apply_chat_template(conversation, thinking=False) == expected + "</think>"
    assert tokenizer.apply_chat_template(conversation, tokenize=True)["input_ids"] == [
        163587, 2482, 163601, 19180, 163586, 163588, 69702, 163601, 163606,
    ]
    tools = [{"type": "function", "function": {"name": "weather", "description": "Weather lookup",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                             "required": ["city"]}}}]
    rendered = tokenizer.apply_chat_template(conversation, tools=tools)
    assert rendered.startswith("<|im_system|>tool_declare<|im_middle|>")
    assert "namespace functions" in rendered and "weather" in rendered and "city: string" in rendered
    reordered = [{"function": dict(reversed(list(tools[0]["function"].items()))), "type": "function"}]
    assert tokenizer.apply_chat_template(conversation, tools=reordered) == rendered
    long_text = "量化😀" * 10000 + " " * 30000 + "a" * 30000
    assert tokenizer.decode(tokenizer.encode(long_text)) == long_text
    tokenizer.save_pretrained(tmp_path)
    restored = TikTokenTokenizer.from_pretrained(tmp_path)
    assert restored.apply_chat_template(conversation, tools=tools, thinking=False) == rendered + "</think>"
    assert restored.encode("Hello, 世界!") == examples["Hello, 世界!"]


def test_owned_tokenizer_malformed_asset_fails_without_remote_fallback_and_retries(tmp_path):
    _, marker = write_checkpoint(tmp_path)
    save_real_tokenizer(tmp_path)
    (tmp_path / "tiktoken.model").write_text("invalid vocabulary\n")
    with pytest.raises(ValueError):
        load_hf_tokenizer(tmp_path, model_config=load_model_config(tmp_path))
    assert not marker.exists()
    save_real_tokenizer(tmp_path)
    tokenizer = load_hf_tokenizer(tmp_path, model_config=load_model_config(tmp_path))
    assert type(tokenizer) is TikTokenTokenizer
    assert tokenizer.encode("Hello, 世界!") == [19180, 11, 220, 2243, 0]
    assert not marker.exists()


@pytest.mark.parametrize("patch_size", [2, 14])
def test_legacy_media_metadata_owns_native_components_and_overrides(tmp_path, patch_size):
    save_real_tokenizer(tmp_path)
    config = tiny_config(vocab_size=163840, pad_token_id=163839, bos_token_id=163584, eos_token_id=163585)
    config.save_pretrained(tmp_path)
    legacy = json.loads((TOKENIZER_FIXTURES / "upstream_preprocessor.json").read_text())
    legacy["media_proc_cfg"]["patch_size"] = patch_size
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(legacy))
    processor = KimiK25QModel.loader.load_processor(tmp_path, config=config)
    assert type(processor) is PackagedProcessor
    assert isinstance(processor.tokenizer, TikTokenTokenizer)
    assert type(processor.image_processor) is Kimi_K25ImageProcessor
    assert type(processor.video_processor) is Kimi_K25VideoProcessor
    assert processor.image_processor.patch_size == processor.video_processor.patch_size == patch_size
    assert processor.image_processor.max_patches == 16384
    assert processor.video_processor.max_patches == 4096
    assert processor.video_processor.temporal_patch_size == 4
    assert processor.video_processor.fps == 2.0
    assert tuple(processor.image_processor.image_mean) == tuple(processor.image_processor.image_std) == (.5, .5, .5)
    assert processor.image_processor.size == {"max_height": 512, "max_width": 512}
    assert processor.video_token_id == 163840
    assert processor.tokenizer("<|kimi_k25_video_placeholder|>", return_tensors="pt")["input_ids"].tolist() == [[163840]]
    with pytest.raises(ValueError, match="no videos were passed"):
        processor(text="<|kimi_k25_video_placeholder|>", return_tensors="pt")
    before = processor(images=torch.zeros(3, patch_size * 2, patch_size * 2, dtype=torch.uint8),
                       text="<|media_pad|>", return_tensors="pt")
    assert before["input_ids"].tolist() == [[163605]]
    assert before["image_grid_thw"].tolist() == [[1, 2, 2]]
    saved = tmp_path / "saved"
    processor.save_pretrained(saved)
    reloaded = KimiK25QModel.loader.load_processor(saved, config=config)
    after = reloaded(images=torch.zeros(3, patch_size * 2, patch_size * 2, dtype=torch.uint8),
                     text="<|media_pad|>", return_tensors="pt")
    torch.testing.assert_close(before["pixel_values"], after["pixel_values"], rtol=0, atol=0)
    assert reloaded.video_token_id == processor.video_token_id


@pytest.mark.parametrize("media", ["image", "video", "mixed"])
def test_real_released_media_tokenizer_to_native_generation(tmp_path, media):
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    reference, marker = write_checkpoint(
        tmp_path, released=True, vocab_size=163840, image_token_id=163605,
        video_token_id=163840, vision_patch_size=14, pad_token_id=163839, bos_token_id=163584, eos_token_id=163585,
    )
    save_real_tokenizer(tmp_path)
    (tmp_path / "processor_config.json").unlink()
    (tmp_path / "preprocessor_config.json").write_bytes((TOKENIZER_FIXTURES / "upstream_preprocessor.json").read_bytes())
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=False)
    pixels = torch.arange(3 * 28 * 28, dtype=torch.uint8).reshape(3, 28, 28)
    kwargs = {}
    text = "Hello"
    if media in {"image", "mixed"}:
        kwargs["images"] = pixels
        text += " <|media_pad|>"
    if media in {"video", "mixed"}:
        kwargs.update(videos=[torch.stack([pixels, pixels.flip(-1), pixels, pixels.flip(-2), pixels])],
                      do_sample_frames=False,
                      video_metadata=[{"fps": 2.0, "total_num_frames": 5, "frames_indices": [0, 1, 2, 3, 4]}])
        text += " <|kimi_k25_video_placeholder|>"
    encoded = loaded.processor(text=text, return_tensors="pt", **kwargs)
    inputs = {key: value for key, value in encoded.items() if key in {
        "input_ids", "attention_mask", "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw",
    }}
    oracle_mm = {}
    with torch.inference_mode():
        if "images" in kwargs:
            assert encoded["input_ids"].eq(163605).sum() == 1
            oracle_mm["image"] = reference.model.get_image_features(
                inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True,
            )
        if "videos" in kwargs:
            assert encoded["input_ids"].eq(163840).sum() == 2
            assert encoded["video_grid_thw"].tolist() == [[4, 2, 2], [1, 2, 2]]
            first = reference.model.vision_tower(inputs["pixel_values_videos"][:16], grid_thw=inputs["video_grid_thw"][:1])
            last = reference.model.vision_tower(inputs["pixel_values_videos"][16:], grid_thw=inputs["video_grid_thw"][1:])
            oracle_mm["video"] = BaseModelOutputWithPooling(pooler_output=(
                reference.model.mm_projector(first.pooler_output).squeeze(1),
                reference.model.mm_projector(last.pooler_output).squeeze(1),
            ))
        expected = reference(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                             mm_encoder_outputs=oracle_mm).logits
        actual = loaded.model(**inputs).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert torch.isfinite(actual).all() and actual.std() > .01
        cached = loaded.model.generate(**inputs, max_new_tokens=2, do_sample=False, use_cache=True)
        uncached = loaded.model.generate(**inputs, max_new_tokens=2, do_sample=False, use_cache=False)
        assert torch.equal(cached, uncached)
        decoded = loaded.processor.batch_decode(cached, skip_special_tokens=False)[0]
        clean = loaded.processor.batch_decode(cached, skip_special_tokens=True)[0]
        if "videos" in kwargs:
            assert "<|kimi_k25_video_placeholder|>" in decoded
        assert "<|kimi_k25_video_placeholder|>" not in clean
    assert not marker.exists()


@pytest.mark.parametrize("tail_frames", [1, 2, 3, 4])
def test_native_ragged_video_pooling_matches_independent_clip_forwards(tail_frames):
    from transformers.models.kimi_k25.modeling_kimi_k25 import Kimi_K25VisionModel as NativeVision

    from gptqmodel.models.ports.kimi_k25.native import Kimi_K25VisionModel as PackagedVision

    torch.manual_seed(109)
    config = tiny_config().vision_config
    reference = NativeVision(config).eval()
    model = PackagedVision(config).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    pixels = torch.randn((4 + tail_frames) * 4, 3, 2, 2)
    grid = torch.tensor([[4, 2, 2], [tail_frames, 2, 2]])
    with torch.inference_mode():
        first = reference(pixels[:16], grid[:1], output_hidden_states=True)
        last = reference(pixels[16:], grid[1:], output_hidden_states=True)
        actual = model(pixels, grid, output_hidden_states=True)
    torch.testing.assert_close(actual.pooler_output, torch.cat([first.pooler_output, last.pooler_output]),
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.last_hidden_state, torch.cat([first.last_hidden_state, last.last_hidden_state]),
                               rtol=1e-5, atol=1e-6)
    assert len(actual.hidden_states) == len(first.hidden_states) == len(last.hidden_states)
    for got, left, right in zip(actual.hidden_states, first.hidden_states, last.hidden_states):
        torch.testing.assert_close(got, torch.cat([left, right]), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("field,value", [
    ("fixed_output_tokens", 16), ("in_patch_limit_video", 32),
    ("max_num_frames_each_video", 4), ("timestamp_mode", "unsupported"),
])
def test_unsupported_legacy_media_policy_fails_closed_then_retries(tmp_path, field, value):
    save_real_tokenizer(tmp_path)
    config = tiny_config(vocab_size=163840, pad_token_id=163839, bos_token_id=163584, eos_token_id=163585)
    config.save_pretrained(tmp_path)
    original = (TOKENIZER_FIXTURES / "upstream_preprocessor.json").read_bytes()
    payload = json.loads(original)
    payload["media_proc_cfg"][field] = value
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(payload))
    tokenizer = TikTokenTokenizer.from_pretrained(tmp_path)
    with pytest.raises(ValueError, match=field):
        KimiK25QModel.loader.load_processor(tmp_path, config=config, tokenizer=tokenizer)
    (tmp_path / "preprocessor_config.json").write_bytes(original)
    processor = KimiK25QModel.loader.load_processor(tmp_path, config=config, tokenizer=tokenizer)
    assert processor.image_processor.patch_size == processor.video_processor.patch_size == 14


@pytest.mark.parametrize("media", ["image", "video"])
def test_real_processor_media_forward_cache_and_failed_placeholder_retry(tmp_path, media):
    reference, marker = write_checkpoint(tmp_path, released=True)
    model = KimiK25QModel.loader.from_pretrained(tmp_path, attn_implementation="eager").eval()
    processor = KimiK25QModel.loader.load_processor(
        tmp_path, config=model.config, tokenizer=save_processor(tmp_path).tokenizer, trust_remote_code=True,
    )
    assert type(processor) is PackagedProcessor
    pixels = torch.arange(48, dtype=torch.uint8).reshape(3, 4, 4)
    if media == "image":
        encoded = processor(images=pixels, text="w4 <|media_pad|> w5", return_tensors="pt")
        assert encoded["image_grid_thw"].tolist() == [[1, 2, 2]]
        inputs = {key: value for key, value in encoded.items() if key != "token_type_ids"}
        assert inputs["input_ids"].eq(30).sum() == 1
    else:
        encoded = processor.video_processor(
            videos=[torch.stack([pixels, pixels.flip(-1)])], do_sample_frames=False, return_tensors="pt",
        )
        assert encoded["video_grid_thw"].tolist() == [[2, 2, 2]]
        inputs = {"input_ids": torch.tensor([[1, 31, 5]]),
                  "pixel_values_videos": encoded["pixel_values_videos"], "video_grid_thw": encoded["video_grid_thw"]}
    pixel_key = "pixel_values" if media == "image" else "pixel_values_videos"
    with torch.inference_mode():
        with pytest.raises(ValueError, match="requires image_grid_thw"):
            model.model.get_image_features(inputs[pixel_key])
        if media == "video":
            from transformers.modeling_outputs import BaseModelOutputWithPooling

            vision = reference.model.vision_tower(inputs["pixel_values_videos"], grid_thw=inputs["video_grid_thw"])
            projected = reference.model.mm_projector(vision.pooler_output).squeeze(1)
            assert projected.shape == (1, reference.config.text_config.hidden_size)
            expected = reference(
                inputs["input_ids"], mm_encoder_outputs={"video": BaseModelOutputWithPooling(pooler_output=(projected,))},
                use_cache=False,
            ).logits
        else:
            expected = reference(**inputs, use_cache=False).logits
        torch.testing.assert_close(model(**inputs, use_cache=False).logits, expected, rtol=1e-5, atol=1e-6)
        changed = dict(inputs)
        changed[pixel_key] = inputs[pixel_key] * .3 + .1
        assert not torch.allclose(model(**changed).logits, expected, rtol=1e-5, atol=1e-6)
        broken = dict(inputs, input_ids=torch.tensor([[1, 4, 5]]))
        with pytest.raises((ValueError, RuntimeError), match="features and .*tokens do not match"):
            model(**broken)
        prefill = model(**inputs, use_cache=True)
        assert prefill.past_key_values.get_seq_length() == 3
        tail = torch.tensor([[7, 8]])
        full = dict(inputs, input_ids=torch.cat([inputs["input_ids"], tail], dim=1))
        full.pop("attention_mask", None)
        continued = model(tail, past_key_values=prefill.past_key_values, use_cache=True)
        torch.testing.assert_close(continued.logits, model(**full).logits[:, -2:], rtol=1e-5, atol=1e-6)
    assert not marker.exists()


def test_bad_expert_payload_fails_then_same_path_retry_is_clean(tmp_path):
    reference, marker = write_checkpoint(tmp_path, released=True)
    state = released_state(reference)
    broken = dict(state)
    broken["language_model.model.layers.1.mlp.experts.2.gate_proj.weight"] = torch.ones(3, 3)
    save_file(broken, tmp_path / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises((RuntimeError, ValueError), match="automatic conversion of the weights"):
        KimiK25QModel.loader.from_pretrained(tmp_path, trust_remote_code=True)
    assert not marker.exists()
    save_file(state, tmp_path / "model.safetensors", metadata={"format": "pt"})
    loaded = KimiK25QModel.loader.from_pretrained(tmp_path, attn_implementation="eager").eval()
    with torch.inference_mode():
        ids = torch.tensor([[1, 4, 9]])
        torch.testing.assert_close(loaded(ids).logits, reference(ids).logits, rtol=1e-5, atol=1e-6)
    assert not marker.exists()


@pytest.mark.parametrize("released", [False, True], ids=["native", "checkpoint"])
@pytest.mark.parametrize("q_lora_rank", [None, 8], ids=["dense_q", "split_q"])
def test_native_conversion_cache_generation_and_safetensors(tmp_path, released, q_lora_rank):
    reference, marker = write_checkpoint(tmp_path, released=released, q_lora_rank=q_lora_rank)
    model, info = KimiK25QModel.loader.from_pretrained(
        tmp_path, output_loading_info=True, attn_implementation="eager", trust_remote_code=True,
    )
    model.eval()
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert model.state_dict().keys() == reference.state_dict().keys()
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0, msg=name)
    ids = torch.tensor([[1, 4, 9, 3, 8], [1, 8, 5, 6, 4]])
    with torch.inference_mode():
        expected = reference(ids, use_cache=False).logits
        assert expected.std() > .01
        torch.testing.assert_close(model(ids).logits, expected, rtol=1e-5, atol=1e-6)
        prefill = model(ids[:, :3], use_cache=True)
        assert prefill.past_key_values.get_seq_length() == 3
        continued = model(ids[:, 3:], past_key_values=prefill.past_key_values, use_cache=True)
        assert continued.past_key_values.get_seq_length() == 5
        torch.testing.assert_close(continued.logits, expected[:, 3:], rtol=1e-5, atol=1e-6)
        cached = model.generate(ids[:, :3], max_new_tokens=3, do_sample=False, eos_token_id=None)
        uncached = model.generate(ids[:, :3], max_new_tokens=3, do_sample=False, eos_token_id=None, use_cache=False)
        assert torch.equal(cached, uncached)
        assert _convert_model_with_defuser(KimiK25QModel, model, cleanup_original=True)
        torch.testing.assert_close(model(ids).logits, expected, rtol=1e-5, atol=1e-6)
    assert model.state_dict().keys() == split_expert_state(reference).keys()
    for step in range(2):
        output = tmp_path / f"saved{step}"
        model.save_pretrained(output, safe_serialization=True)
        assert (output / "model.safetensors").exists()
        assert not list(output.glob("*.bin")) and not list(output.glob("*.pt"))
        model = KimiK25QModel.loader.from_pretrained(output, attn_implementation="eager").eval()
        with torch.inference_mode():
            torch.testing.assert_close(model(ids).logits, expected, rtol=1e-5, atol=1e-6)
    assert not marker.exists()
