# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Tiny real MiniCPM-V checkpoint, media and trust-boundary regressions."""

import json

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from gptqmodel.models.definitions.minicpmv import MiniCPMVQModel
from gptqmodel.models.ports.minicpmv.configuration_minicpm import MiniCPMVConfig
from gptqmodel.models.ports.minicpmv.image_processing_minicpmv import MiniCPMVImageProcessor
from gptqmodel.models.ports.minicpmv.modeling_minicpmv import MiniCPMV
from gptqmodel.models.ports.minicpmv.processing_minicpmv import MiniCPMVProcessor
from gptqmodel.utils.hf import load_model_config


VERSIONS = [1.0, 2.0, 2.5, 2.6, 4.0, 4.5]


def tiny_config(version=4.5, **kwargs):
    options = {"version": version, "hidden_size": 128, "intermediate_size": 192, "num_hidden_layers": 1,
                   "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "vocab_size": 128,
                   "query_num": 4, "image_size": 28, "drop_vision_last_layer": False, "max_position_embeddings": 256,
                   "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "tie_word_embeddings": False,
                   "scale_emb": 2.0, "scale_depth": 1.4, "dim_model_base": 32,
                   "vision_config": {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                                      "num_attention_heads": 4, "image_size": 28, "patch_size": 14},
                   "slice_config": {"max_slice_nums": 1, "scale_resolution": 28, "patch_size": 14},
                   "legacy_vision_config": {"embed_dim": 32, "depth": 1, "num_heads": 4, "mlp_ratio": 2, "img_size": 28}}
    options.update(kwargs)
    return MiniCPMVConfig(**options)


def tokenizer_fixture(path=None):
    tokens = ["[PAD]", "[BOS]", "[EOS]", "<unk>", "<image>", "</image>", "<slice>", "</slice>",
              "<image_id>", "</image_id>", "hello", "image", "test", "0", "1", "2"]
    backend = Tokenizer(WordLevel({t: i for i, t in enumerate(tokens)}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]",
                                       eos_token="[EOS]", unk_token="<unk>")
    tokenizer.add_special_tokens({"additional_special_tokens": tokens[4:10]})
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    if path is not None:
        tokenizer.save_pretrained(path)
    return tokenizer


def processor_fixture(config, path=None):
    tokenizer = tokenizer_fixture(path)
    image_processor = MiniCPMVImageProcessor(version=config.version, max_slice_nums=1, scale_resolution=28,
                                             patch_size=14, image_feature_size=config.query_num,
                                             slice_mode=config.slice_mode, use_image_id=config.use_image_id)
    processor = MiniCPMVProcessor(image_processor, tokenizer)
    if path is not None:
        processor.save_pretrained(path)
    return processor


def media_fixture(processor, frames=1, temporal=False):
    import numpy as np

    images = [Image.fromarray((np.arange(28 * 28 * 3).reshape(28, 28, 3) * 13 + i * 47).astype("uint8"))
              for i in range(frames)]
    prompt = "hello " + "(<image>./</image>) " * frames + "test image"
    inputs = processor(prompt, [images], temporal_ids=[[[i * 3 for i in range(frames)]]] if temporal else None)
    inputs.pop("image_sizes")
    return inputs


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_real_images_cache_generate_and_safetensors(tmp_path, version, implementation, dtype):
    torch.manual_seed(19)
    config = tiny_config(version, attn_implementation=implementation)
    model = MiniCPMVQModel.loader.from_config(config, dtype=dtype).eval()
    processor = processor_fixture(config)
    inputs = media_fixture(processor)
    with torch.inference_mode():
        full = model(**inputs, use_cache=False).logits
        prefix = model(**inputs, use_cache=True)
        token = torch.tensor([[11]])
        continued = model(input_ids=token, past_key_values=prefix.past_key_values).logits
        extended = dict(inputs)
        extended["input_ids"] = torch.cat((inputs["input_ids"], token), -1)
        extended["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(token)), -1)
        reference = model(**extended, use_cache=False).logits[:, -1:]
        tolerance = 2e-6 if dtype == torch.float32 else 0.006 if dtype == torch.float16 else 0.04
        torch.testing.assert_close(continued, reference, atol=tolerance, rtol=tolerance)
        cached = model.generate(**inputs, max_new_tokens=3, do_sample=False)
        uncached = model.generate(**inputs, max_new_tokens=3, do_sample=False, use_cache=False)
        assert torch.equal(cached, uncached)
        changed = dict(inputs)
        changed["pixel_values"] = [[-x for x in inputs["pixel_values"][0]]]
        assert (model(**changed).logits - full).abs().max() > 1e-5
    model.save_pretrained(tmp_path, safe_serialization=True)
    processor.save_pretrained(tmp_path)
    restored = MiniCPMVQModel.loader.from_pretrained(tmp_path, attn_implementation=implementation, dtype=dtype).eval()
    reprocessor = MiniCPMVQModel.loader.load_processor(tmp_path, config=restored.config, tokenizer=processor.tokenizer)
    reinputs = media_fixture(reprocessor)
    with torch.inference_mode():
        torch.testing.assert_close(restored(**reinputs).logits, full, rtol=0, atol=0)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    assert list(tmp_path.glob("*.safetensors"))
    assert not list(tmp_path.glob("*.bin"))


def test_minicpmv_config_ignores_checkpoint_dispatch(tmp_path):
    payload = {
        "model_type": "minicpmv", "version": 4.5,
        "hidden_size": 128, "intermediate_size": 192, "num_hidden_layers": 1,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
        "vocab_size": 64, "query_num": 4,
        "auto_map": {"AutoConfig": "evil.Config", "AutoModel": "evil.Model"},
    }
    (tmp_path / "config.json").write_text(json.dumps(payload))
    (tmp_path / "evil.py").write_text("raise AssertionError('checkpoint Python executed')")
    config = load_model_config(tmp_path, trust_remote_code=False)
    assert type(config).__module__.startswith("gptqmodel.models.ports.minicpmv.")
    assert config.model_type == "minicpmv"


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_media_dtype_and_text_native_oracle(version, dtype):
    torch.manual_seed(11)
    model = MiniCPMVQModel.loader.from_config(tiny_config(version), dtype=dtype).eval()
    inputs = media_fixture(processor_fixture(model.config))
    with torch.inference_mode():
        actual = model(**inputs).logits
        embeds, features = model.get_vllm_embedding(inputs)
        assert embeds.dtype == dtype
        assert torch.isfinite(actual).all()
        # Independently assemble language input from the actual resampler
        # output and original IDs, then call the exact native text decoder.
        ref = model.llm.get_input_embeddings()(inputs["input_ids"])
        if version < 2.5:
            ref *= model.config.scale_emb
        for start, stop in inputs["image_bound"][0]:
            ref[0, start:stop] = features[0].flatten(0, 1).to(ref)
        expected = model.llm(inputs_embeds=ref, attention_mask=inputs["attention_mask"]).logits
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
@pytest.mark.parametrize("version", VERSIONS)
def test_public_hostile_checkpoint_processor_and_retry(tmp_path, monkeypatch, trust, bypass, version):
    from gptqmodel import GPTQModel

    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    config = tiny_config(version)
    MiniCPMV(config).save_pretrained(tmp_path, safe_serialization=True)
    processor = processor_fixture(config, tmp_path)
    for filename in ["config.json", "tokenizer_config.json", "preprocessor_config.json", "processor_config.json"]:
        path = tmp_path / filename
        data = json.loads(path.read_text())
        data["auto_map"] = dict.fromkeys(["AutoConfig", "AutoModel", "AutoTokenizer", "AutoProcessor", "AutoImageProcessor"], "evil.Evil")
        if filename == "tokenizer_config.json":
            data["auto_map"]["AutoTokenizer"] = ["evil.Evil", "evil.Evil"]
            data["tokenizer_class"] = "MiniCPMVTokenizerFast"
        path.write_text(json.dumps(data))
    (tmp_path / "evil.py").write_text("raise AssertionError('checkpoint Python executed')")
    model = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32,
                          trust_remote_code=trust, attn_implementation="eager")
    assert type(model) is MiniCPMVQModel
    assert type(model.model) is MiniCPMV
    assert type(model.processor) is MiniCPMVProcessor
    inputs = media_fixture(model.processor)
    with torch.inference_mode():
        torch.testing.assert_close(model.model(**inputs).logits,
                                   model.model(**media_fixture(processor)).logits, rtol=0, atol=0)
    path = tmp_path / "preprocessor_config.json"
    contents = path.read_bytes()
    path.write_text("{")
    with pytest.raises(json.JSONDecodeError):
        MiniCPMVQModel.loader.load_processor(tmp_path, config=config, tokenizer=processor.tokenizer)
    path.write_bytes(contents)
    repaired = MiniCPMVQModel.loader.load_processor(tmp_path, config=config, tokenizer=processor.tokenizer)
    assert torch.equal(media_fixture(repaired)["input_ids"], inputs["input_ids"])


@pytest.mark.parametrize("batch_mode", [True, False])
def test_temporal_pooling_images_and_chat(tmp_path, batch_mode):
    config = tiny_config(batch_3d_resampler=batch_mode)
    model = MiniCPMV(config).eval()
    processor = processor_fixture(config, tmp_path)
    inputs = media_fixture(processor, frames=3, temporal=True)
    with torch.inference_mode():
        embeds, features = model.get_vllm_embedding(inputs)
        assert features[0].shape == (1, config.query_num, config.hidden_size)
        assert embeds.shape[:2] == inputs["input_ids"].shape
        assert torch.isfinite(model(**inputs).logits).all()
        assert model.generate(**inputs, max_new_tokens=2, do_sample=False).shape == (1, 2)
    result = model.chat(msgs=[{"role": "user", "content": [Image.new("RGB", (28, 28)), "hello image"]}],
                        processor=processor, sampling=False, max_new_tokens=2)
    assert isinstance(result, str)


def test_tokenizer_asset_failure_repair_and_subfolder(tmp_path):
    from gptqmodel.utils.hf import load_hf_tokenizer

    folder = tmp_path / "weights"
    folder.mkdir()
    config = tiny_config()
    config.save_pretrained(folder)
    tokenizer_fixture(folder)
    metadata = json.loads((folder / "tokenizer_config.json").read_text())
    metadata.update(tokenizer_class="MiniCPMVTokenizerFast", auto_map={"AutoTokenizer": ["evil.Evil", "evil.Evil"]})
    (folder / "tokenizer_config.json").write_text(json.dumps(metadata))
    (folder / "evil.py").write_text("raise AssertionError('checkpoint tokenizer Python executed')")
    contents = (folder / "tokenizer.json").read_bytes()
    (folder / "tokenizer.json").write_text("{")
    with pytest.raises(json.JSONDecodeError, match="Expecting property"):
        load_hf_tokenizer(str(tmp_path), subfolder="weights", model_config=config)
    (folder / "tokenizer.json").write_bytes(contents)
    tokenizer = load_hf_tokenizer(str(tmp_path), subfolder="weights", model_config=config)
    assert tokenizer.encode("hello image") == [10, 11]
    metadata["tokenizer_class"] = "UnqualifiedForeignTokenizer"
    (folder / "tokenizer_config.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Unqualified MiniCPM-V tokenizer"):
        load_hf_tokenizer(str(folder), model_config=config)


def test_invalid_architecture_and_media_fail_then_retry():
    for version in [3, 4.6, 4.7, 5]:
        with pytest.raises(ValueError, match="Unqualified MiniCPM-V"):
            tiny_config(version)
    with pytest.raises(ValueError, match="requires a supported version"):
        MiniCPMVConfig(hidden_size=128)
    config = tiny_config()
    model = MiniCPMV(config).eval()
    processor = processor_fixture(config)
    inputs = media_fixture(processor)
    bad = dict(inputs, image_bound=[torch.tensor([[1, 2]])])
    with pytest.raises(ValueError, match="image_bound must match"):
        model(**bad)
    with pytest.raises(ValueError, match="truncated or malformed"):
        processor("hello (<image>./</image>)", [[Image.new("RGB", (28, 28))]], max_length=4)
    with torch.inference_mode():
        assert torch.isfinite(model(**inputs).logits).all()


def _linear(x, module):
    return torch.nn.functional.linear(x, module.weight.double(),
                                      module.bias.double() if module.bias is not None else None)


def _norm(x, module):
    return torch.nn.functional.layer_norm(x, module.normalized_shape, module.weight.double(), module.bias.double(), module.eps)


def _spatial_reference(dim, h, w, dtype=torch.float32):
    # Pinned Alibaba/OpenBMB formula, expressed independently with Torch.
    frequencies = 10000 ** (-torch.arange(dim // 4, dtype=dtype) / (dim / 4))
    positions = []
    for y in range(h):
        for x in range(w):
            positions.append(torch.cat(((x * frequencies).sin(), (x * frequencies).cos(),
                                        (y * frequencies).sin(), (y * frequencies).cos())))
    return torch.stack(positions).double()


def _resampler_reference(module, features, sizes, groups=None, temporal=False):
    x = _linear(features.double(), module.kv_proj) if isinstance(module.kv_proj, torch.nn.Linear) else features.double()
    x = _norm(x, module.ln_kv)
    q = _norm(module.query.double(), module.ln_q)
    dim, heads = module.embed_dim, module.attn.num_heads
    if module.config.version < 2.5:
        if module.config.version == 2:
            pos = _spatial_reference(dim, *map(int, sizes[0]), dtype=torch.float64)
        else:
            n = int(module.num_queries ** 0.5)
            side = int(features.shape[1] ** 0.5)
            pos = torch.nn.functional.interpolate(module.pos_embed.double().reshape(n, n, dim).permute(2, 0, 1)[None],
                                                 size=(side, side), mode="bicubic", align_corners=False)
            pos = pos[0].permute(1, 2, 0).reshape(-1, dim)
        k, v = x + pos, x
        q = q + module.pos_embed.double()
        batches = [(k[0], v[0], torch.zeros(k.shape[1], dtype=torch.bool))]
    else:
        positions = torch.nn.utils.rnn.pad_sequence([_spatial_reference(dim, int(h), int(w)) for h, w in sizes],
                                                   batch_first=True)
        k, v = x + positions, x
        mask = torch.arange(x.shape[1])[None] >= sizes.prod(-1)[:, None]
        batches = []
        if temporal:
            offset = 0
            for group in groups:
                ks, vs = [], []
                for j, t in enumerate(group):
                    freq = 10000 ** (-torch.arange(dim // 2, dtype=torch.float32) / (dim / 2))
                    tp = torch.cat(((t * freq).sin(), (t * freq).cos())).double() if t >= 0 else torch.zeros(dim)
                    ks.append(k[offset + j] + tp)
                    vs.append(v[offset + j])
                if module.config.batch_3d_resampler:
                    kg, vg = torch.cat(ks), torch.cat(vs)
                else:
                    kg = torch.stack(ks, 1).flatten(0, 1)
                    vg = torch.stack(vs, 1).flatten(0, 1)
                batches.append((kg, vg, mask[offset:offset + len(group)].flatten()))
                offset += len(group)
        else:
            batches = list(zip(k, v, mask))
    matrices = module.attn.in_proj_weight.double().chunk(3)
    biases = module.attn.in_proj_bias.double().chunk(3)
    outputs = []
    for key, value, masked in batches:
        qp, kp, vp = [torch.nn.functional.linear(item, matrix, bias).reshape(-1, heads, dim // heads).transpose(0, 1)
                      for item, matrix, bias in zip((q, key, value), matrices, biases)]
        scores = qp @ kp.transpose(-1, -2) / (dim // heads) ** 0.5
        scores[..., masked] = -torch.inf
        out = (scores.softmax(-1) @ vp).transpose(0, 1).reshape(len(q), dim)
        outputs.append(_linear(out, module.attn.out_proj))
    return _norm(torch.stack(outputs), module.ln_post) @ module.proj.double()


@pytest.mark.parametrize("version", VERSIONS)
def test_independent_resampler_equations(version):
    torch.manual_seed(37)
    model = MiniCPMV(tiny_config(version)).eval()
    sizes = torch.tensor([[2, 3]]) if version != 1 else torch.tensor([[2, 2]])
    features = torch.randn(1, int(sizes.prod()), 32)
    with torch.inference_mode():
        expected = _resampler_reference(model.resampler, features, sizes)
        actual = model.resampler(features, sizes)
    torch.testing.assert_close(actual.double(), expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("batch_mode", [True, False])
def test_independent_ragged_temporal_resampler(batch_mode):
    torch.manual_seed(29)
    model = MiniCPMV(tiny_config(batch_3d_resampler=batch_mode)).eval()
    features = torch.randn(4, 6, 32)
    sizes = torch.tensor([[2, 3], [2, 2], [1, 4], [2, 3]])
    groups = [[-1], [2, 6, 9]]
    with torch.inference_mode():
        actual = model.resampler(features, sizes, groups)
        expected = _resampler_reference(model.resampler, features, sizes, groups, temporal=True)
    torch.testing.assert_close(actual.double(), expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("version", [2.5, 2.6, 4.0, 4.5])
def test_independent_vision_equations_and_patch_positions(implementation, version):
    import torch.nn.functional as F

    torch.manual_seed(7)
    model = MiniCPMV(tiny_config(version, attn_implementation=implementation)).eval()
    pixels = torch.randn(2, 3, 14, 84)
    sizes = torch.tensor([[2, 3], [2, 2]])
    mask = torch.tensor([[[True] * 6], [[True] * 4 + [False] * 2]])
    tower = model.vpm
    conv = tower.embeddings.patch_embedding
    x = F.conv2d(pixels.double(), conv.weight.double(), conv.bias.double(), stride=14).flatten(2).transpose(1, 2)
    ids = torch.tensor([[0, 0, 1, 2, 2, 3], [0, 1, 2, 3, 0, 0]])
    x += tower.embeddings.position_embedding.weight.double()[ids]
    for layer in tower.encoder.layers:
        normalized = _norm(x, layer.layer_norm1)
        q, k, v = [_linear(normalized, getattr(layer.self_attn, name)).reshape(2, 6, 4, 8).transpose(1, 2)
                   for name in ["q_proj", "k_proj", "v_proj"]]
        scores = (q @ k.transpose(-1, -2)) / 8 ** 0.5
        scores.masked_fill_(~mask[:, None], -torch.inf)
        attn = (scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 6, 32)
        x = x + _linear(attn, layer.self_attn.out_proj)
        x = x + _linear(F.gelu(_linear(_norm(x, layer.layer_norm2), layer.mlp.fc1), approximate="tanh"), layer.mlp.fc2)
    expected = _norm(x, tower.post_layernorm)
    with torch.inference_mode():
        actual = tower(pixels, mask, sizes).last_hidden_state
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=3e-5)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("kv", [1, 2, 4])
def test_independent_original_minicpm_mup_decoder(version, kv):
    import torch.nn.functional as F

    torch.manual_seed(71)
    model = MiniCPMV(tiny_config(version, num_key_value_heads=kv, num_hidden_layers=2)).eval()
    ids = torch.tensor([[1, 10, 11, 12, 13]])
    llm = model.llm
    config = llm.config
    x = llm.model.embed_tokens.weight[ids].double() * config.scale_emb
    def rms(item, module):
        return item * (item.square().mean(-1, keepdim=True) + module.variance_epsilon).rsqrt() * module.weight.double()
    freq = 10000 ** (-torch.arange(0, 32, 2, dtype=torch.float64) / 32)
    angles = torch.arange(5)[:, None] * freq
    cos, sin = torch.cat((angles, angles), -1).cos(), torch.cat((angles, angles), -1).sin()
    def rotate(item):
        left, right = item.chunk(2, -1)
        return item * cos + torch.cat((-right, left), -1) * sin
    for layer in llm.model.layers:
        normed = rms(x, layer.input_layernorm)
        attn = layer.self_attn
        q = _linear(normed, attn.q_proj).reshape(1, 5, 4, 32).transpose(1, 2)
        k = _linear(normed, attn.k_proj).reshape(1, 5, kv, 32).transpose(1, 2)
        v = _linear(normed, attn.v_proj).reshape(1, 5, kv, 32).transpose(1, 2)
        q, k = rotate(q), rotate(k).repeat_interleave(4 // kv, 1)
        v = v.repeat_interleave(4 // kv, 1)
        scores = (q @ k.transpose(-1, -2)) / 32 ** 0.5
        scores.masked_fill_(torch.ones(5, 5, dtype=torch.bool).triu(1), -torch.inf)
        out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(1, 5, 128)
        x = x + _linear(out, attn.o_proj) * (config.scale_depth / config.num_hidden_layers ** 0.5)
        normed = rms(x, layer.post_attention_layernorm)
        mlp = layer.mlp
        out = _linear(F.silu(_linear(normed, mlp.gate_proj)) * _linear(normed, mlp.up_proj), mlp.down_proj)
        x = x + out * (config.scale_depth / config.num_hidden_layers ** 0.5)
    expected = _linear(rms(x, llm.model.norm) / (config.hidden_size / config.dim_model_base), llm.lm_head)
    with torch.inference_mode():
        actual = model(input_ids=ids).logits
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=3e-5)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("offload", [False, True])
def test_public_image_awq_plain_quantize_pack_infer_and_reload(tmp_path, version, offload):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    config = tiny_config(version, intermediate_size=128)
    source = MiniCPMVQModel.loader.from_config(config, dtype=torch.float16).eval()
    path = tmp_path / "source"
    source.save_pretrained(path, safe_serialization=True)
    processor_fixture(config, path)
    qcfg = AWQConfig(bits=4, group_size=32, device="cpu", offload_to_disk=offload,
                     offload_to_disk_path=str(tmp_path / "offload"), fallback=None)
    model = GPTQModel.load(str(path), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16,
                          trust_remote_code=False, attn_implementation="eager")
    inputs = media_fixture(model.processor)
    calibration = [[{"role": "user", "content": [
        {"type": "image", "image": Image.new("RGB", (28, 28), (80, 140, 200))},
        {"type": "text", "text": "hello image test hello test image hello image test"},
    ]}] for _ in range(2)]
    dense = {name: tensor.clone() for name, tensor in source.state_dict().items()
             if not name.startswith("llm.model.layers.0.") or not name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight",
                                                                               "o_proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight"))}
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    packed = {name: module for name, module in model.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert len(packed) == 7
    assert all(name.startswith("llm.model.layers.0.") for name in packed)
    assert all(module.qweight.dtype == torch.int32 for module in packed.values())
    # LazyTurtle's offload path starts with meta shells. Materialize the float
    # exclusions before comparing/inference; the reference is the saved source,
    # never a pre-quantization meta shell.
    model.pre_quantize_generate_hook_start()
    model.shell_module_materialize(model.model.llm.model.norm, "cpu")
    model.shell_module_materialize(model.model.llm.lm_head, "cpu")
    for name, tensor in dense.items():
        torch.testing.assert_close(model.model.state_dict()[name], tensor, rtol=0, atol=0)
    with torch.inference_mode():
        expected = model.model(**inputs).logits
        assert torch.isfinite(expected).all()
    for i in range(2):
        destination = tmp_path / f"quantized-{i}"
        model.save(destination)
        assert list(destination.glob("*.safetensors"))
        assert not list(destination.glob("*.bin"))
        restored = GPTQModel.load(str(destination), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16,
                                 trust_remote_code=False, attn_implementation="eager")
        for name, tensor in model.model.state_dict().items():
            torch.testing.assert_close(restored.model.state_dict()[name], tensor, rtol=0, atol=0, msg=name)
        with torch.inference_mode():
            torch.testing.assert_close(restored.model(**inputs).logits, expected, rtol=0, atol=0)
        model = restored
