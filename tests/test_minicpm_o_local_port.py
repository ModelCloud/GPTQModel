# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models.definitions.minicpm_o import MiniCPMOQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.minicpm_o.configuration_minicpmo import MiniCPMOConfig
from gptqmodel.models.ports.minicpm_o.modeling_minicpmo import MiniCPMO
from gptqmodel.models.ports.minicpm_o.processing_minicpmo import (
    MiniCPMAAudioProcessor,
    MiniCPMOProcessor,
    MiniCPMVImageProcessor,
)
from gptqmodel.models.ports.minicpm_o.tokenization_minicpmo_fast import MiniCPMOTokenizerFast


PORT = Path(__file__).resolve().parents[1] / "gptqmodel/models/ports/minicpm_o"


@pytest.fixture(autouse=True, scope="module")
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tiny_config(version="4.5", kv_heads=2, bias=False, attn="eager", tts=False):
    return MiniCPMOConfig(
        version=version,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        head_dim=8,
        attention_bias=bias,
        vocab_size=512,
        max_position_embeddings=256,
        query_num=4,
        init_vision=True,
        init_audio=True,
        init_tts=tts,
        drop_vision_last_layer=False,
        audio_pool_step=5 if version == "4.5" else 2,
        audio_chunk_length=-1,
        attn_implementation=attn,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 2,
        },
        audio_config={
            "d_model": 16,
            "encoder_ffn_dim": 64,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "num_mel_bins": 8,
            "max_source_positions": 256,
            "dropout": 0,
            "attention_dropout": 0,
            "activation_dropout": 0,
        },
        tts_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "llm_dim": 32,
            "num_audio_tokens": 16,
            "num_text_tokens": 128,
            "num_vq": 2,
        },
    )


def tiny_model(dtype=torch.float32, **kwargs):
    torch.manual_seed(239)
    model = MiniCPMO._from_config(tiny_config(**kwargs), dtype=dtype).eval()
    torch.nn.init.normal_(model.resampler.query, std=0.1)
    return model


def tiny_tokenizer():
    from tokenizers.pre_tokenizers import ByteLevel

    special = [
        "<pad>",
        "<s>",
        "<|im_end|>",
        "<unk>",
        "<|endoftext|>",
        "<|im_start|>",
        "<image>",
        "</image>",
        "<slice>",
        "</slice>",
        "<image_id>",
        "</image_id>",
        "<|audio_start|>",
        "<|audio_end|>",
        "<|spk_bos|>",
        "<|spk_eos|>",
        "<|tts_bos|>",
        "<|tts_eos|>",
        "<|spk|>",
        "<|listen|>",
    ]
    vocab = {token: index for index, token in enumerate(special)}
    for token in sorted(ByteLevel.alphabet()):
        vocab.setdefault(token, len(vocab))
    tokenizer = MiniCPMOTokenizerFast(
        vocab=vocab,
        merges=[],
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="<|im_end|>",
        additional_special_tokens=special[4:],
    )
    tokenizer.chat_template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    return tokenizer


def tiny_processor(version="4.5"):
    cls = MiniCPMOProcessor
    if version == "2.6":
        from gptqmodel.models.ports.minicpm_o.processing_minicpmo_26 import MiniCPMOProcessor26

        cls = MiniCPMOProcessor26
    return cls(
        MiniCPMVImageProcessor(
            scale_resolution=8,
            patch_size=2,
            image_feature_size=4,
            max_slice_nums=1,
            use_image_id=False,
            version=float(version),
        ),
        MiniCPMAAudioProcessor(feature_size=8, n_fft=400, hop_length=160),
        tiny_tokenizer(),
        audio_pool_step=5 if version == "4.5" else 2,
    )


def write_checkpoint(path, model=None, hostile=False):
    model = model or tiny_model()
    model.save_pretrained(path)
    tiny_processor(model.config.version).save_pretrained(path)
    marker = path / "CHECKPOINT_PYTHON_EXECUTED"
    if hostile:
        for filename, entries in (
            ("config.json", {"AutoConfig": "poison.Config", "AutoModel": "poison.Model"}),
            ("tokenizer_config.json", {"AutoTokenizer": ["poison.Tokenizer", "poison.Tokenizer"]}),
            (
                "preprocessor_config.json",
                {
                    "AutoProcessor": "poison.Processor",
                    "AutoImageProcessor": "poison.Image",
                    "AutoFeatureExtractor": "poison.Audio",
                },
            ),
        ):
            values = json.loads((path / filename).read_text())
            values["auto_map"] = entries
            if filename == "preprocessor_config.json":
                values["processor_class"] = "PoisonProcessor"
                for key in ("image_processor", "audio_processor"):
                    values[key]["auto_map"] = {"AutoProcessor": "poison.Component"}
                    values[key]["processor_class"] = "PoisonComponent"
            (path / filename).write_text(json.dumps(values))
        (path / "poison.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise AssertionError('checkpoint Python executed')\n"
        )
    return marker


@pytest.mark.parametrize("version", ["4.5", "2.6"])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_local_security_policy_and_native_owners(tmp_path, monkeypatch, version, trust, bypass):
    from transformers import Qwen2ForCausalLM, Qwen3ForCausalLM, cache_utils

    originals = cache_utils.Cache, cache_utils.DynamicCache
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    source = tiny_model(version=version)
    marker = write_checkpoint(tmp_path, source, True)
    model = GPTQModel.load(
        str(tmp_path), backend=BACKEND.TORCH, device="cpu", dtype=torch.float32, trust_remote_code=trust
    )
    assert type(model) is MiniCPMOQModel
    assert type(model.model.config) is MiniCPMOConfig
    assert isinstance(model.model.llm, Qwen3ForCausalLM if version == "4.5" else Qwen2ForCausalLM)
    assert type(model.processor).__module__.startswith("gptqmodel.models.ports.minicpm_o.")
    assert type(model.tokenizer.tokenizer) is MiniCPMOTokenizerFast
    assert type(model.processor.image_processor) is MiniCPMVImageProcessor
    assert not marker.exists()
    assert (cache_utils.Cache, cache_utils.DynamicCache) == originals
    ids = torch.tensor([[1, 23, 24, 25]])
    for key, value in source.state_dict().items():
        torch.testing.assert_close(model.model.state_dict()[key], value, atol=0, rtol=0)
    with torch.no_grad():
        torch.testing.assert_close(
            model.model(input_ids=ids).logits, source(input_ids=ids).logits, atol=2e-7, rtol=1e-5
        )


@pytest.mark.parametrize("version", ["4.5", "2.6"])
@pytest.mark.parametrize("kv_heads", [4, 2, 1])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_native_decoder_equivalence_cache_and_generation(version, kv_heads, bias, attn):
    from transformers import Qwen2ForCausalLM, Qwen3ForCausalLM

    model = tiny_model(version=version, kv_heads=kv_heads, bias=bias, attn=attn)
    native = (Qwen3ForCausalLM if version == "4.5" else Qwen2ForCausalLM)(model.llm.config).eval()
    native.load_state_dict(model.llm.state_dict(), strict=True)
    ids = torch.tensor([[1, 21, 22, 23, 24, 25]])
    with torch.no_grad():
        expected = native(ids, use_cache=False).logits
        full = model(input_ids=ids, use_cache=False).logits
        torch.testing.assert_close(full, expected, atol=0, rtol=0)
        prefix = model(input_ids=ids[:, :2], use_cache=True)
        tail = model(input_ids=ids[:, 2:], past_key_values=prefix.past_key_values, use_cache=True)
        torch.testing.assert_close(tail.logits, expected[:, 2:], atol=2e-7, rtol=1e-5)
        assert tail.past_key_values.get_seq_length() == 6
        for beams in [1, 2]:
            actual = model.llm.generate(ids, max_new_tokens=3, num_beams=beams, do_sample=False)
            reference = native.generate(ids, max_new_tokens=3, num_beams=beams, do_sample=False)
            assert torch.equal(actual, reference)


def linear(x, module):
    return F.linear(x, module.weight.double(), None if module.bias is None else module.bias.double())


def norm(x, module):
    return F.layer_norm(x, module.normalized_shape, module.weight.double(), module.bias.double(), module.eps)


def attention(x, module, heads):
    query = linear(x, module.q_proj).view(1, -1, heads, x.shape[-1] // heads).transpose(1, 2)
    key = linear(x, module.k_proj).view_as(query.transpose(1, 2)).transpose(1, 2)
    value = linear(x, module.v_proj).view_as(query.transpose(1, 2)).transpose(1, 2)
    weights = (query @ key.transpose(-1, -2) / (x.shape[-1] // heads) ** 0.5).softmax(-1)
    return linear((weights @ value).transpose(1, 2).reshape_as(x), module.out_proj)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_vision_audio_and_resampler_against_independent_equations(dtype):
    model = tiny_model().to(dtype)
    pixels = torch.linspace(-0.8, 0.9, 48).reshape(1, 3, 4, 4).to(dtype)
    layer = model.vpm.encoder.layers[0]
    embedding = model.vpm.embeddings
    x = (
        F.conv2d(
            pixels.double(),
            embedding.patch_embedding.weight.double(),
            embedding.patch_embedding.bias.double(),
            stride=2,
        )
        .flatten(2)
        .transpose(1, 2)
    )
    x = x + embedding.position_embedding.weight.double()[[0, 2, 8, 10]].unsqueeze(0)
    x = x + attention(norm(x, layer.layer_norm1), layer.self_attn, 2)
    x = x + linear(F.gelu(linear(norm(x, layer.layer_norm2), layer.mlp.fc1), approximate="tanh"), layer.mlp.fc2)
    expected = norm(x, model.vpm.post_layernorm)
    with torch.no_grad():
        actual = model.vpm(pixels, tgt_sizes=torch.tensor([[2, 2]])).last_hidden_state
    tolerance = {torch.float32: 2e-5, torch.float16: 0.007, torch.bfloat16: 0.07}[dtype]
    torch.testing.assert_close(actual.double(), expected, atol=tolerance, rtol=tolerance)
    audio = torch.linspace(-0.9, 0.8, 160).reshape(1, 8, 20).to(dtype)
    apm = model.apm
    x = F.gelu(F.conv1d(audio.double(), apm.conv1.weight.double(), apm.conv1.bias.double(), padding=1))
    x = F.gelu(F.conv1d(x, apm.conv2.weight.double(), apm.conv2.bias.double(), padding=1, stride=2)).transpose(1, 2)
    x = x + apm.embed_positions.weight[:10].double()
    al = apm.layers[0]
    x = x + attention(norm(x, al.self_attn_layer_norm), al.self_attn, 2)
    x = x + linear(F.gelu(linear(norm(x, al.final_layer_norm), al.fc1)), al.fc2)
    expected_audio = norm(x, apm.layer_norm)
    with torch.no_grad():
        actual_audio = apm(audio, output_hidden_states=True).last_hidden_state
    torch.testing.assert_close(actual_audio.double(), expected_audio, atol=tolerance, rtol=tolerance)
    resampler = model.resampler
    with torch.no_grad():
        resampler.query.copy_(torch.linspace(-0.6, 0.8, 128).reshape(4, 32))
        out = resampler(actual, tgt_sizes=torch.tensor([[2, 2]]))
    kv = norm(linear(actual.double(), resampler.kv_proj), resampler.ln_kv)
    q = norm(resampler.query.double(), resampler.ln_q).unsqueeze(0)
    # Four deterministic 2D positions, independently constructed from the sinusoid equations.
    pos = []
    omega = 10000 ** (-torch.arange(8, dtype=torch.float64) / 8)
    for row, col in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        pos.append(torch.cat([(col * omega).sin(), (col * omega).cos(), (row * omega).sin(), (row * omega).cos()]))
    weights, biases = resampler.attn.in_proj_weight.double().chunk(3), resampler.attn.in_proj_bias.double().chunk(3)
    q = F.linear(q, weights[0], biases[0])
    k = F.linear(kv + torch.stack(pos).unsqueeze(0), weights[1], biases[1])
    v = F.linear(kv, weights[2], biases[2])
    x = linear(((q @ k.transpose(-1, -2)) / 32**0.5).softmax(-1) @ v, resampler.attn.out_proj)
    expected_resampler = norm(x, resampler.ln_post) @ resampler.proj.double()
    torch.testing.assert_close(out.double(), expected_resampler, atol=tolerance, rtol=tolerance)


def media_inputs(processor, media="both"):
    from PIL import Image

    kwargs = {}
    text = "word23 "
    if media in ("image", "both"):
        kwargs["images"] = [[Image.fromarray(np.arange(192, dtype=np.uint8).reshape(8, 8, 3))]]
        text += "(<image>./</image>) " if str(processor.version) == "2.6" else "<image>./</image> "
    if media in ("audio", "both"):
        kwargs["audios"] = [[np.sin(np.arange(3200, dtype=np.float32) * 0.04)]]
        text += "(<audio>./</audio>) " if str(processor.version) == "2.6" else "<audio>./</audio> "
    if str(processor.version) == "2.6":
        kwargs["chunk_input"] = False
    return processor(text=[text + "word24"], return_tensors="pt", sampling_rate=16000, **kwargs)


@pytest.mark.parametrize("version", ["4.5", "2.6"])
@pytest.mark.parametrize("media", ["image", "audio", "both"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_real_media_conditioning_generation_and_safetensors(tmp_path, version, media, dtype):
    model = tiny_model(version=version, dtype=dtype)
    processor = tiny_processor(version)
    data = media_inputs(processor, media)
    ids = data["input_ids"].long()
    with torch.no_grad():
        actual = model(data=data, use_cache=True)
        empty = model(input_ids=ids, use_cache=False)
        assert torch.isfinite(actual.logits).all()
        assert (actual.logits - empty.logits).abs().max() > 0.001
        assert actual.past_key_values.get_seq_length() == ids.shape[1]
        _, generated = model.generate(
            **{k: v for k, v in data.items() if k != "image_sizes"},
            tokenizer=processor.tokenizer,
            max_new_tokens=2,
            do_sample=False,
        )
        assert generated.sequences.shape[-1] in (1, 2)
    model.save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)
    assert list(tmp_path.glob("*.safetensors")) and not list(tmp_path.glob("*.bin"))
    loaded = MiniCPMO.from_pretrained(tmp_path, dtype=dtype, attn_implementation="eager").eval()
    reprocessor = MiniCPMOProcessor.from_pretrained(tmp_path, config=loaded.config)
    again = media_inputs(reprocessor, media)
    assert torch.equal(data["input_ids"], again["input_ids"])
    assert type(reprocessor) is type(processor)
    with torch.no_grad():
        restored = loaded(data=again, use_cache=False).logits
    torch.testing.assert_close(restored, actual.logits, atol=0, rtol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("version", ["4.5", "2.6"])
def test_tts_native_llama_state_cache_and_safetensors(tmp_path, version):
    from transformers import LlamaModel

    model = tiny_model(version=version, tts=True)
    reference = LlamaModel(model.tts.model.config).eval()
    reference.load_state_dict(model.tts.model.state_dict(), strict=True)
    hidden = torch.linspace(-1, 1, 64).reshape(1, 4, 16)
    with torch.no_grad():
        expected = reference(inputs_embeds=hidden, use_cache=False).last_hidden_state
        prefix = model.tts.model(inputs_embeds=hidden[:, :1], use_cache=True)
        tail = model.tts.model(inputs_embeds=hidden[:, 1:], past_key_values=prefix.past_key_values, use_cache=True)
    torch.testing.assert_close(tail.last_hidden_state, expected[:, 1:], atol=3e-7, rtol=1e-5)
    assert tail.past_key_values.get_seq_length() == 4
    assert model.tts.model.embed_tokens.weight.shape == (32000, 16)
    assert model.tts.emb_text.weight.shape == (128, 16)
    model.save_pretrained(tmp_path)
    loaded = MiniCPMO.from_pretrained(tmp_path, attn_implementation="eager").eval()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
    with torch.no_grad():
        torch.testing.assert_close(loaded.tts.model(inputs_embeds=hidden).last_hidden_state, expected, atol=0, rtol=0)


@pytest.mark.parametrize("version", ["4.5", "2.6"])
def test_real_image_audio_chat_with_checkpoint_chunk_policy(version):
    from PIL import Image

    model = tiny_model(version=version)
    processor = tiny_processor(version)
    model.config.use_image_id = False
    model.config.audio_chunk_length = 0.1
    image = Image.fromarray(np.arange(192, dtype=np.uint8).reshape(8, 8, 3))
    audio = np.sin(np.arange(3200, dtype=np.float32) * 0.04)
    options = {"do_sample": False} if version == "4.5" else {"sampling": False, "chunk_input": False}
    answer = model.chat(
        msgs=[{"role": "user", "content": [image, audio, "word21"]}],
        processor=processor,
        tokenizer=processor.tokenizer,
        max_new_tokens=2,
        num_beams=1,
        **options,
    )
    assert isinstance(answer, str)


@pytest.mark.parametrize("version", ["4.5", "2.6"])
def test_failed_config_asset_and_payload_repaired_retry(tmp_path, version):
    source = tiny_model(version=version)
    write_checkpoint(tmp_path, source, hostile=True)
    original = (tmp_path / "config.json").read_text()
    values = json.loads(original)
    values["version"] = "9.9"
    (tmp_path / "config.json").write_text(json.dumps(values))
    with pytest.raises(ValueError, match="version"):
        GPTQModel.load(str(tmp_path), backend=BACKEND.TORCH, device="cpu", dtype=torch.float32)
    (tmp_path / "config.json").write_text(original)
    saved = (tmp_path / "preprocessor_config.json").read_text()
    (tmp_path / "preprocessor_config.json").write_text("{bad")
    with pytest.raises(json.JSONDecodeError):
        GPTQModel.load(str(tmp_path), backend=BACKEND.TORCH, device="cpu", dtype=torch.float32)
    (tmp_path / "preprocessor_config.json").write_text(saved)
    from safetensors.torch import load_file, save_file

    payload = load_file(tmp_path / "model.safetensors")
    payload["llm.model.layers.0.self_attn.q_proj.weight"] = torch.zeros(1, 1)
    save_file(payload, tmp_path / "model.safetensors")
    with pytest.raises((RuntimeError, ValueError)):
        GPTQModel.load(str(tmp_path), backend=BACKEND.TORCH, device="cpu", dtype=torch.float32)
    source.save_pretrained(tmp_path)
    restored = GPTQModel.load(str(tmp_path), backend=BACKEND.TORCH, device="cpu", dtype=torch.float32)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[key], value, atol=0, rtol=0)
    assert not (tmp_path / "CHECKPOINT_PYTHON_EXECUTED").exists()


@pytest.mark.parametrize("version", ["4.5", "2.6"])
@pytest.mark.parametrize("offload", [False, True])
def test_actual_public_plain_awq_image_capture_pack_infer_and_two_reloads(tmp_path, version, offload):
    from PIL import Image

    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    source = tiny_model(version=version, dtype=torch.float16, tts=True)
    write_checkpoint(tmp_path, source)
    config = AWQConfig(
        device="cpu",
        group_size=16,
        fallback=None,
        offload_to_disk=offload,
        offload_to_disk_path=str(tmp_path / "offload"),
    )
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=config, backend=BACKEND.TORCH, dtype=torch.float16)
    dense = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if not key.startswith("llm.model.layers.") or "norm" in key
    }
    image = Image.fromarray(np.arange(192, dtype=np.uint8).reshape(8, 8, 3))
    calibration = [
        [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": "word21 word22 word23"}],
            }
        ]
    ] * 2
    prompt = wrapper.processor.tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": ("(<image>./</image>)" if version == "2.6" else "<image>./</image>")
                + "\nword21 word22 word23",
            }
        ],
        tokenize=False,
    )
    independent = wrapper.processor(text=[prompt], images=[[image]], return_tensors="pt")
    assert independent["image_bound"][0].shape == (1, 2)
    with torch.no_grad():
        expected_input, _ = source.get_vllm_embedding(independent)
    captured = []

    def record_input(module, args, kwargs):
        captured.append((args[0] if args else kwargs["hidden_states"]).detach().clone())

    handle = wrapper.model.llm.model.layers[0].register_forward_pre_hook(record_input, with_kwargs=True)
    wrapper.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    handle.remove()
    assert captured
    torch.testing.assert_close(captured[0], expected_input.expand(captured[0].shape[0], -1, -1), atol=0, rtol=0)
    packed = {name: module for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert len(packed) == 14
    assert all(name.startswith("llm.model.layers.") for name in packed)
    for key, value in dense.items():
        torch.testing.assert_close(wrapper.model.state_dict()[key], value, atol=0, rtol=0)
    data = media_inputs(wrapper.processor, "image")
    with torch.no_grad():
        expected = wrapper.model(data=data, use_cache=False).logits
    assert torch.isfinite(expected).all()
    for cycle in [1, 2]:
        saved = tmp_path / f"quantized-{cycle}"
        wrapper.save_quantized(str(saved))
        assert list(saved.glob("*.safetensors")) and not list(saved.rglob("*.bin"))
        restored = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
        assert sum(isinstance(m, AwqTorchLinear) for m in restored.model.modules()) == 14
        for key, value in wrapper.model.state_dict().items():
            torch.testing.assert_close(restored.model.state_dict()[key], value, atol=0, rtol=0)
        with torch.no_grad():
            torch.testing.assert_close(restored.model(data=data, use_cache=False).logits, expected, atol=0, rtol=0)
        wrapper = restored


def test_definition_selects_local_omni_without_global_cache_mutation():
    from transformers import cache_utils

    cache, dynamic = cache_utils.Cache, cache_utils.DynamicCache
    assert isinstance(MiniCPMOQModel.loader, LocalModelLoader)
    assert not MiniCPMOQModel.require_trust_remote_code
    assert cache_utils.Cache is cache
    assert cache_utils.DynamicCache is dynamic
