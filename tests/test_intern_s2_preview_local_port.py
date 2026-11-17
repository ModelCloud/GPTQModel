# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real local Intern-S2 loading, multimodal, cache and serialization regressions."""

import json
from shutil import copyfile

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLVideoProcessor

from gptqmodel import CalibrationConfig
from gptqmodel.models.definitions.intern_s2_preview import InternS2PreviewQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.intern_s2_preview.configuration_interns2_preview import InternS2PreviewConfig
from gptqmodel.models.ports.intern_s2_preview.modeling_interns2_preview import InternS2PreviewForConditionalGeneration
from gptqmodel.models.ports.intern_s2_preview.processing_interns2_preview import InternS2PreviewProcessor
from gptqmodel.models.ports.intern_s2_preview.tokenization_interns1 import InternS1Tokenizer, bytes_to_unicode


def tiny_config(layer_types=None, **text_overrides):
    text = {
        "vocab_size": 96, "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8, "linear_conv_kernel_dim": 3, "linear_key_head_dim": 8,
        "linear_value_head_dim": 8, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16, "num_experts": 3,
        "num_experts_per_tok": 2, "max_position_embeddings": 128,
        "layer_types": layer_types or ["linear_attention", "full_attention"],
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": 1., "mrope_section": [1, 1, 2]},
        "pad_token_id": 0, "eos_token_id": None,
    }
    text.update(text_overrides)
    config = InternS2PreviewConfig(
        text_config=text,
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 24, "num_heads": 2,
                           "out_hidden_size": 32, "patch_size": 2, "temporal_patch_size": 2,
                           "spatial_merge_size": 2, "num_position_embeddings": 16},
        ts_config={"d_model": 16, "encoder_attention_heads": 2, "encoder_ffn_dim": 32,
                       "encoder_layers": 1, "max_source_positions": 32, "out_hidden_size": 32,
                       "ts_adapt_in_dim": 256, "ts_adapt_out_dim": 16, "ts_hidden_dim": 16},
        image_token_id=80, video_token_id=81, vision_start_token_id=82, vision_end_token_id=83,
        ts_token_id=84, ts_start_id=85, ts_end_id=86,
    )
    config._attn_implementation = "eager"
    return config


@pytest.mark.parametrize("layer_types", [
    ["linear_attention", "full_attention"], ["full_attention", "linear_attention"],
    ["linear_attention"] * 2, ["full_attention"] * 2,
])
@pytest.mark.parametrize("continuation", [1, 4])
@torch.no_grad()
def test_real_hybrid_forward_and_cache(layer_types, continuation):
    torch.manual_seed(17)
    model = InternS2PreviewForConditionalGeneration(tiny_config(layer_types)).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    expected = model(ids, use_cache=False).logits
    prefill = model(ids[:, :3], use_cache=True)
    end = 3 + continuation
    output = model(ids[:, 3:end], past_key_values=prefill.past_key_values, use_cache=True)
    torch.testing.assert_close(output.logits, expected[:, 3:end], rtol=2e-5, atol=2e-6)
    assert output.past_key_values.get_seq_length() == end


def tiny_processor():
    vocab = {f"token_{i}": i for i in range(80)}
    special = ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>",
               "<TS_CONTEXT>", "<|ts|>", "<|/ts|>"]
    vocab.update({token: i for i, token in enumerate(special, 80)})
    backend = Tokenizer(WordLevel(vocab, unk_token="token_0"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="token_0", pad_token="token_0",
                                       additional_special_tokens=special)
    common = {"patch_size": 2, "temporal_patch_size": 2, "merge_size": 2,
                  "size": {"shortest_edge": 16, "longest_edge": 64}, "image_mean": [.5]*3, "image_std": [.5]*3}
    return InternS2PreviewProcessor(
        Qwen2VLImageProcessor(**common), tokenizer,
        Qwen3VLVideoProcessor(**common, min_frames=2, max_frames=4, do_sample_frames=False),
    )


def tiny_scientific_tokenizer(path):
    """Real byte BPE + three distinct-offset SentencePiece domains, offline."""
    import sentencepiece as spm

    path.mkdir(parents=True, exist_ok=True)
    corpus = path / "corpus.txt"
    corpus.write_text("CCO CCC NNC MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE ATCGATCGATCGATCGATCGATCGATCGATCG\n")
    prefix = str(path / "tokenizer_SMILES")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=prefix, vocab_size=48, model_type="bpe",
        bos_id=-1, eos_id=-1, pad_id=-1, add_dummy_prefix=False, minloglevel=2,
        normalization_rule_name="identity", num_threads=1,
    )
    for domain in ("PROT", "XNA"):
        copyfile(path / "tokenizer_SMILES.model", path / f"tokenizer_{domain}.model")
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|image_pad|>", "<|video_pad|>",
               "<|vision_start|>", "<|vision_end|>", "<TS_CONTEXT>", "<|ts|>", "<|/ts|>"]
    domains = ["<SMILES>", "</SMILES>", "<protein>", "</protein>", "<dna>", "</dna>", "<rna>", "</rna>"]
    vocab = {char: i for i, char in enumerate(bytes_to_unicode().values())}
    vocab.update({token: len(vocab) + i for i, token in enumerate(special + domains)})
    (path / "vocab.json").write_text(json.dumps(vocab))
    (path / "merges.txt").write_text("#version: 0.2\n")
    offsets = {"offset_" + domain: len(vocab) + i * 48 for i, domain in enumerate(("SMILES", "PROT", "XNA"))}
    tokenizer = InternS1Tokenizer(
        str(path / "vocab.json"), str(path / "merges.txt"), **offsets,
        additional_special_tokens=special,
        extra_special_tokens={"image_token": "<|image_pad|>", "video_token": "<|video_pad|>",
                              "vision_bos_token": "<|vision_start|>", "vision_eos_token": "<|vision_end|>"},
    )
    tokenizer.add_tokens(special, special_tokens=True)
    tokenizer.add_tokens(domains, special_tokens=False)
    tokenizer.chat_template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}"
    tokenizer.save_pretrained(path)
    return tokenizer


def save_tiny_owned_checkpoint(path):
    tokenizer = tiny_scientific_tokenizer(path)
    config = tiny_config(vocab_size=512)
    config.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    config.video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    config.vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    config.vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    model = InternS2PreviewForConditionalGeneration(config).eval()
    # Upstream initializes routers to zero: a synthetic checkpoint needs
    # nonuniform routers to exercise every expert during calibration.
    for layer in model.model.language_model.layers:
        torch.nn.init.normal_(layer.mlp.gate.weight, std=0.1)
    model.save_pretrained(path)
    media = tiny_processor()
    InternS2PreviewProcessor(media.image_processor, tokenizer, media.video_processor).save_pretrained(path)
    return model, tokenizer


@pytest.mark.parametrize("asset", InternS1Tokenizer.vocab_files_names.values())
def test_missing_owned_asset_identifies_filename_and_retries(tmp_path, asset):
    tokenizer = tiny_scientific_tokenizer(tmp_path)
    path = tmp_path / asset
    content = path.read_bytes()
    path.unlink()
    with pytest.raises(RuntimeError, match=asset):
        InternS1Tokenizer.from_pretrained(tmp_path, local_files_only=True)
    path.write_bytes(content)
    restored = InternS1Tokenizer.from_pretrained(tmp_path, local_files_only=True)
    text = "<SMILES>CCO</SMILES>"
    assert restored(text)["input_ids"] == tokenizer(text)["input_ids"]


def test_scientific_tokenizer_real_domain_ids_rendering_save_reload_and_retry(tmp_path):
    tokenizer = tiny_scientific_tokenizer(tmp_path / "source")
    protein = "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE"
    dna = "ATCG" * 7 + "GATTACA"
    for text, domain, marker in ((protein, "PROT", "protein"), (dna, "XNA", "dna"), ("CCO", "SMILES", "SMILES")):
        sp = getattr(tokenizer, "sp_model_" + domain)
        expected = [sp.offset + i for i in sp.encode(text, out_type=int)]
        assert len(set(expected)) > 1
        explicit = tokenizer(f"<{marker}>{text}</{marker}>")["input_ids"]
        assert explicit == [tokenizer.convert_tokens_to_ids(f"<{marker}>")] + expected + [
            tokenizer.convert_tokens_to_ids(f"</{marker}>")]
        assert tokenizer.decode(explicit) == f"<{marker}>{text}</{marker}>"
        if domain != "SMILES":
            assert tokenizer(text)["input_ids"] == expected
            assert tokenizer.tokenize(text)[0] == f"<{domain}_AUTO_DETECT>"
            assert tokenizer.decode(expected) == text
    messages = [{"role": "user", "content": "<dna>ATCG</dna>"}]
    rendered = "<|im_start|>user\n<dna>ATCG</dna><|im_end|>\n"
    assert tokenizer.apply_chat_template(messages, tokenize=False) == rendered
    assert tokenizer.apply_chat_template(messages, return_dict=False) == tokenizer(rendered)["input_ids"]
    saved = tmp_path / "saved"
    tokenizer.save_pretrained(saved)
    for asset in tokenizer.vocab_files_names.values():
        assert (saved / asset).is_file()
    xna = saved / "tokenizer_XNA.model"
    xna.rename(saved / "hold.model")
    with pytest.raises((OSError, RuntimeError), match="tokenizer_XNA.model"):
        InternS1Tokenizer.from_pretrained(saved, local_files_only=True)
    (saved / "hold.model").rename(xna)
    restored = InternS1Tokenizer.from_pretrained(saved, local_files_only=True)
    for text in (protein, dna, "<SMILES>CCO</SMILES>", rendered, "café e\u0301"):
        assert restored(text)["input_ids"] == tokenizer(text)["input_ids"]
        assert restored.decode(restored(text)["input_ids"]) == tokenizer.decode(tokenizer(text)["input_ids"])


@pytest.mark.parametrize("kind", ["image", "video"])
@torch.no_grad()
def test_real_media_processor_forward_generation_and_reload(tmp_path, kind):
    processor = tiny_processor()
    processor.save_pretrained(tmp_path)
    tiny_config().save_pretrained(tmp_path)
    restored = InternS2PreviewQModel.loader.load_processor(
        tmp_path, tokenizer=PreTrainedTokenizerFast.from_pretrained(tmp_path), local_files_only=True,
    )
    assert type(restored.image_processor) is Qwen2VLImageProcessor
    assert type(restored.video_processor) is Qwen3VLVideoProcessor
    if kind == "image":
        data = {"text": ["token_1 <|vision_start|><|image_pad|><|vision_end|>"],
                    "images": [torch.arange(3*4*8, dtype=torch.uint8).reshape(3,4,8)]}
    else:
        data = {"text": ["token_1 <|video_pad|>"],
                    "videos": [torch.arange(3*3*4*8, dtype=torch.uint8).reshape(3,3,4,8)],
                    "video_metadata": [{"fps": 2., "total_num_frames": 3, "frames_indices": [0,1,2]}]}
    batch = restored(**data, return_tensors="pt")
    expected = processor(**data, return_tensors="pt")
    assert batch.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(batch[key], expected[key], rtol=0, atol=0)
    model = InternS2PreviewForConditionalGeneration(tiny_config()).eval()
    output = model(**batch, use_cache=False)
    assert output.logits.shape[:2] == batch.input_ids.shape
    assert torch.isfinite(output.logits).all()
    generation = {"max_new_tokens": 2, "do_sample": False, "suppress_tokens": list(range(80,87))}
    generated = model.generate(**batch, **generation)
    uncached = model.generate(**batch, **generation, use_cache=False)
    assert torch.equal(generated, uncached)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
@torch.no_grad()
def test_public_loader_ignores_checkpoint_python_and_reloads(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel import GPTQModel
    from gptqmodel.utils.hf import load_model_config

    torch.manual_seed(11)
    model, tokenizer = save_tiny_owned_checkpoint(tmp_path)
    # A valid generic serialized backend must not replace scientific semantics.
    tiny_processor().tokenizer.backend_tokenizer.save(str(tmp_path / "tokenizer.json"))
    poison = dict.fromkeys(("AutoConfig", "AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoProcessor",
                           "AutoImageProcessor", "AutoVideoProcessor"), "poison.Evil")
    for name in ("config.json", "processor_config.json", "tokenizer_config.json"):
        path = tmp_path / name
        data = json.loads(path.read_text())
        data["auto_map"] = poison
        data["processor_class"] = "EvilProcessor"
        if name == "config.json":
            data["text_config"]["auto_map"] = poison
            data["text_config"]["tokenizer_class"] = "EvilTokenizer"
        if name == "tokenizer_config.json":
            data["tokenizer_class"] = "EvilTokenizer"
            data["auto_map"]["AutoTokenizer"] = ["tokenization_poison.EvilTokenizer", None]
        path.write_text(json.dumps(data))
    (tmp_path / "poison.py").write_text("raise AssertionError('checkpoint Python executed')\n")
    (tmp_path / "tokenization_poison.py").write_text("raise AssertionError('checkpoint tokenizer Python executed')\n")
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    config = load_model_config(str(tmp_path), trust_remote_code=trust)
    assert type(config) is InternS2PreviewConfig
    loaded = GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert type(loaded) is InternS2PreviewQModel
    assert type(loaded.model) is InternS2PreviewForConditionalGeneration
    assert type(loaded.processor) is InternS2PreviewProcessor
    assert type(loaded.processor.tokenizer) is InternS1Tokenizer
    assert loaded.processor.tokenizer("<dna>ATCG</dna>")["input_ids"] == tokenizer("<dna>ATCG</dna>")["input_ids"]
    assert loaded.trust_remote_code is False
    ids = torch.tensor([[1,2,3]])
    torch.testing.assert_close(loaded.model(ids, use_cache=False).logits,
                               model(ids, use_cache=False).logits, rtol=2e-5, atol=2e-6)
    assert list(tmp_path.glob("*.safetensors"))
    assert not list(tmp_path.glob("*.bin"))


@torch.no_grad()
def test_public_loader_owns_tokenizer_despite_unknown_checkpoint_class(tmp_path, monkeypatch):
    from gptqmodel import GPTQModel

    _, tokenizer = save_tiny_owned_checkpoint(tmp_path)
    path = tmp_path / "tokenizer_config.json"
    data = json.loads(path.read_text())
    # Selected model metadata owns the exact scientific implementation even
    # when untrusted checkpoint data names a different tokenizer.
    data.update(tokenizer_class="UnavailableCheckpointTokenizer", auto_map={
        "AutoTokenizer": ["tokenization_poison.UnavailableCheckpointTokenizer", None]})
    path.write_text(json.dumps(data))
    (tmp_path / "tokenization_poison.py").write_text("raise AssertionError('checkpoint tokenizer Python executed')\n")
    monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    loaded = GPTQModel.load(str(tmp_path), trust_remote_code=False, device="cpu", dtype=torch.float32)
    assert type(loaded.model) is InternS2PreviewForConditionalGeneration
    assert type(loaded.processor.tokenizer) is InternS1Tokenizer
    assert loaded.processor.tokenizer("Hello")["input_ids"] == tokenizer("Hello")["input_ids"]


@torch.no_grad()
def test_public_missing_scientific_asset_fails_without_serialized_fallback_then_retries(tmp_path, monkeypatch):
    from gptqmodel import GPTQModel

    _, tokenizer = save_tiny_owned_checkpoint(tmp_path)
    tiny_processor().tokenizer.backend_tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenization_poison.py").write_text("raise AssertionError('checkpoint tokenizer Python executed')\n")
    path = tmp_path / "tokenizer_config.json"
    data = json.loads(path.read_text())
    data["auto_map"] = {"AutoTokenizer": ["tokenization_poison.InternS1Tokenizer", None]}
    path.write_text(json.dumps(data))
    monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    asset = tmp_path / "tokenizer_XNA.model"
    asset.rename(tmp_path / "hold.model")
    options = {"trust_remote_code": False, "device": "cpu", "dtype": torch.float32}
    with pytest.raises((OSError, RuntimeError), match="tokenizer_XNA.model"):
        GPTQModel.load(str(tmp_path), **options)
    (tmp_path / "hold.model").rename(asset)
    loaded = GPTQModel.load(str(tmp_path), **options)
    assert type(loaded.processor.tokenizer) is InternS1Tokenizer
    assert loaded.processor.tokenizer("<dna>ATCG</dna>")["input_ids"] == tokenizer("<dna>ATCG</dna>")["input_ids"]
    assert torch.isfinite(loaded.model(torch.tensor([[1,2,3]]), use_cache=False).logits).all()


@torch.no_grad()
def test_public_plain_awq_quantized_safetensors_roundtrip(tmp_path):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    torch.manual_seed(53)
    save_tiny_owned_checkpoint(tmp_path)
    wrapped = GPTQModel.load(str(tmp_path), quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None),
                            backend=BACKEND.TORCH, dtype=torch.float16, trust_remote_code=False)
    assert type(wrapped.model) is InternS2PreviewForConditionalGeneration
    calibration = [{"input_ids": torch.arange(1,17).reshape(1,-1),
                    "attention_mask": torch.ones(1,16,dtype=torch.long)},
                   {"input_ids": torch.arange(16,0,-1).reshape(1,-1),
                    "attention_mask": torch.ones(1,16,dtype=torch.long)}]
    wrapped.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    counts = [sum(isinstance(mod, AwqTorchLinear) for mod in layer.modules())
              for layer in wrapped.model.model.language_model.layers]
    assert counts == [15,16]
    ids = torch.tensor([[1,2,3,4]])
    expected = wrapped.model(ids, use_cache=False).logits
    time_series = (torch.randn(1,64,2).half(), torch.tensor([64]), torch.tensor([10.]), torch.tensor([2]))
    ts_features, ts_mask = wrapped.model.model.get_ts_feature(*time_series)
    assert ts_features.dtype == torch.float16 and torch.isfinite(ts_features).all()
    state = {name: value.clone() for name, value in wrapped.model.state_dict().items()}
    assert torch.isfinite(expected).all()
    output = tmp_path / "quantized"
    wrapped.save(str(output))
    assert list(output.glob("*.safetensors")) and not list(output.glob("*.bin"))
    restored = GPTQModel.load(str(output), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16, trust_remote_code=False)
    assert type(restored.model) is InternS2PreviewForConditionalGeneration
    assert restored.model.state_dict().keys() == state.keys()
    for name, value in restored.model.state_dict().items():
        # Dense buffers follow the explicitly requested reload dtype; packed
        # integer payloads and their scales/zeros remain exactly identical.
        torch.testing.assert_close(value, state[name].to(value.dtype), rtol=0, atol=0,
                                   msg=lambda message: f"{name}: {message}")
    torch.testing.assert_close(restored.model(ids, use_cache=False).logits, expected, rtol=2e-5, atol=2e-6)
    restored_features, restored_mask = restored.model.model.get_ts_feature(*time_series)
    torch.testing.assert_close(restored_features, ts_features, rtol=0.003, atol=0.002)
    assert torch.equal(restored_mask, ts_mask)


def test_definition_owns_model_and_processor_selection():
    assert isinstance(InternS2PreviewQModel.loader, LocalModelLoader)
    assert InternS2PreviewQModel.loader.processor_class.endswith("InternS2PreviewProcessor")
    assert InternS2PreviewQModel.loader.tokenizer_class == (
        "gptqmodel.models.ports.intern_s2_preview.tokenization_interns1.InternS1Tokenizer"
    )
    assert not InternS2PreviewQModel.require_trust_remote_code
    assert not InternS2PreviewQModel.require_monkeypatch


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_time_series_real_encoder_forward_and_placeholder_retry(dtype):
    torch.manual_seed(19)
    model = InternS2PreviewForConditionalGeneration(tiny_config()).eval().to(dtype)
    values = torch.randn(1,64,2).to(dtype)
    options = {"ts_values": values, "ts_lens": torch.tensor([64]), "ts_sr": torch.tensor([10.]),
                   "ts_channels": torch.tensor([2]), "use_cache": False}
    features, mask = model.model.get_ts_feature(values, options['ts_lens'], options['ts_sr'], options['ts_channels'])
    assert features.dtype == dtype and features.shape[-1] == 32
    count = int((~mask).sum())
    with pytest.raises(AssertionError, match="Mismatch"):
        model(torch.tensor([[1,84,2]]), **options)
    ids = torch.tensor([[1]+[84]*count+[2]])
    actual = model(ids, **options).logits
    embeddings = model.get_input_embeddings()(ids)
    embeddings[ids == 84] = features[~mask]
    expected = model(inputs_embeds=embeddings, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("seq", [1, 7, 67])
@torch.no_grad()
def test_delta_math_matches_independent_float64_recurrence(seq):
    from gptqmodel.models.ports.intern_s2_preview.modeling_interns2_preview import (
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )
    torch.manual_seed(29)
    q, k = [torch.randn(2,seq,2,4) for _ in range(2)]
    v = torch.randn(2,seq,2,3)
    g, beta = -torch.rand(2,seq,2), torch.rand(2,seq,2)
    initial = torch.randn(2,2,4,3)
    state, outputs = initial.double().clone(), []
    for i in range(seq):
        key, query = k[:,i].double(), q[:,i].double()
        key = key / (key.square().sum(-1,keepdim=True)+1e-6).sqrt()
        query = query / (query.square().sum(-1,keepdim=True)+1e-6).sqrt() / 2
        state = state * g[:,i].double().exp()[...,None,None]
        residual = beta[:,i].double()[...,None] * (v[:,i].double()-torch.einsum('bhk,bhkv->bhv',key,state))
        state = state + torch.einsum('bhk,bhv->bhkv',key,residual)
        outputs.append(torch.einsum('bhk,bhkv->bhv',query,state))
    for kernel in (torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule):
        actual, final = kernel(q,k,v,g,beta,initial_state=initial,output_final_state=True,
                               use_qk_l2norm_in_kernel=True)
        torch.testing.assert_close(actual,torch.stack(outputs,1).float(),rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(final,state.float(),rtol=2e-5,atol=2e-6)
