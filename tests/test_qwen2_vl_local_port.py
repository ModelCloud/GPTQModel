# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Exact native tasks: conditional generation versus headless extraction."""

import copy
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

# Transformers' own Qwen2-VL is used only as the numerical oracle.
from transformers.models.qwen2_vl import modeling_qwen2_vl as oracle_modeling
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig as OracleQwen2VLConfig
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLTextConfig as OracleQwen2VLTextConfig
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor as OracleQwen2VLProcessor

from gptqmodel import CalibrationConfig, GPTQModel
from gptqmodel.models.definitions.qwen2_vl import Qwen2VLQModel
from gptqmodel.models.ports import qwen2_vl
from gptqmodel.models.ports.native.auto import AutoModel, AutoModelForCausalLM, NativeForImageTextToText
from gptqmodel.models.ports.native.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLTextConfig
from gptqmodel.models.ports.native.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from gptqmodel.models.ports.native.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLTextModel
from gptqmodel.models.ports.native.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from gptqmodel.models.ports.native.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
from gptqmodel.utils.hf import load_model_config
from tests.port_assets import read_port_asset, read_port_json


@pytest.fixture(autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(text=False, kv_heads=2, tied=False):
    cfg = Qwen2VLTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        max_position_embeddings=128,
        rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 2], "rope_theta": 10000.0},
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=tied,
    )
    if text:
        return cfg
    return Qwen2VLConfig(
        text_config=cfg.to_dict(),
        vision_config={
            "depth": 1,
            "embed_dim": 16,
            "hidden_size": 32,
            "num_heads": 2,
            "mlp_ratio": 2,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
        tie_word_embeddings=tied,
    )


def native_model(config):
    model_class = Qwen2VLTextModel if config.model_type == "qwen2_vl_text" else Qwen2VLForConditionalGeneration
    return model_class._from_config(config, attn_implementation="eager").eval()


def save_fixture(path, text=False, hostile=False, legacy=False):
    torch.manual_seed(317)
    model = native_model(tiny_config(text, tied=legacy))
    model.save_pretrained(path, safe_serialization=True)
    if legacy:
        from safetensors.torch import save_file

        assert not text
        payload = {}
        for key, tensor in model.state_dict().items():
            if key == "lm_head.weight":
                continue  # The released 2B checkpoint omits this tied alias.
            key = key.replace("model.language_model.", "model.").replace("model.visual.", "visual.")
            payload[key] = tensor.detach().clone().contiguous()
        save_file(payload, path / "model.safetensors", metadata={"format": "pt"})
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "<pad>": 0,
                "<s>": 1,
                "</s>": 2,
                "<unk>": 3,
                **{f"t{i}": i for i in range(4, 60)},
                "<|image_pad|>": 60,
                "<|video_pad|>": 61,
                "<|vision_start|>": 62,
                "<|vision_end|>": 63,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
        vision_start_token="<|vision_start|>",
        vision_end_token="<|vision_end|>",
    )
    template = "{% for message in messages %}{% for content in message['content'] %}{% if content['type'] == 'image' %} <|vision_start|><|image_pad|><|vision_end|> {% else %}{{ content['text'] }}{% endif %}{% endfor %}{% endfor %} t9 t12 t6 t8"
    tokenizer.chat_template = template
    tokenizer.save_pretrained(path)
    if not text:
        Qwen2VLProcessor(
            image_processor=Qwen2VLImageProcessor(
                patch_size=2, temporal_patch_size=2, merge_size=2, min_pixels=16, max_pixels=64
            ),
            tokenizer=tokenizer,
            chat_template=template,
            video_processor=Qwen2VLVideoProcessor(
                patch_size=2, temporal_patch_size=2, merge_size=2, min_pixels=16, max_pixels=64
            ),
        ).save_pretrained(path)
    if hostile:
        marker = path / "EXECUTED"
        (path / "evil.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python')\n"
        )
        (path / "custom_generate").mkdir()
        (path / "custom_generate" / "generate.py").write_text((path / "evil.py").read_text())
        for filename in (
            "config.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "processor_config.json",
        ):
            f = path / filename
            if f.exists():
                data = json.loads(f.read_text())
                data["auto_map"] = dict.fromkeys(
                    (
                        "AutoConfig",
                        "AutoModel",
                        "AutoModelForCausalLM",
                        "AutoModelForImageTextToText",
                        "AutoProcessor",
                        "AutoImageProcessor",
                        "AutoVideoProcessor",
                    ),
                    "evil.Hijack",
                )
                if filename == "tokenizer_config.json":
                    data["auto_map"]["AutoTokenizer"] = ["evil.Hijack", "evil.Hijack"]
                if filename == "processor_config.json":
                    data["processor_class"] = "ForeignCheckpointProcessor"
                f.write_text(json.dumps(data))
    return model


def output_tensor(model, ids, **kwargs):
    output = model(ids, **kwargs)
    native = model.model if isinstance(model, Qwen2VLQModel) else model
    return output.last_hidden_state if isinstance(native, Qwen2VLTextModel) else output.logits


def test_native_task_mapping_is_not_a_causal_alias():
    cfg = tiny_config(True)
    assert type(AutoModel.from_config(cfg)) is Qwen2VLTextModel
    for build in (AutoModelForCausalLM.from_config, NativeForImageTextToText._from_config):
        with pytest.raises(ValueError, match="no vendored"):
            build(cfg)
    assert type(Qwen2VLQModel.loader.from_config(cfg)) is Qwen2VLTextModel


@pytest.mark.parametrize("text", [False, True])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_vendored_classes_match_transformers_oracle(text, implementation):
    torch.manual_seed(5)
    cfg = tiny_config(text, kv_heads=2)
    model = Qwen2VLQModel.loader.from_config(copy.deepcopy(cfg), attn_implementation=implementation).eval()
    oracle_config = (OracleQwen2VLTextConfig if text else OracleQwen2VLConfig)(**cfg.to_dict())
    oracle_class = oracle_modeling.Qwen2VLTextModel if text else oracle_modeling.Qwen2VLForConditionalGeneration
    oracle = oracle_class._from_config(oracle_config, attn_implementation=implementation).eval()
    assert type(model).__module__.startswith("gptqmodel.models.ports.native.qwen2_vl.")
    assert {k: v.shape for k, v in model.state_dict().items()} == {k: v.shape for k, v in oracle.state_dict().items()}
    oracle.load_state_dict(model.state_dict())
    ids = torch.tensor([[1, 7, 4, 17, 8, 12]])
    with torch.no_grad():
        if text:
            assert torch.equal(model(ids).last_hidden_state, oracle(ids).last_hidden_state)
        else:
            assert torch.equal(model(ids).logits, oracle(ids).logits)
            assert torch.equal(
                model.generate(ids, max_new_tokens=4, do_sample=False),
                oracle.generate(ids, max_new_tokens=4, do_sample=False),
            )


@pytest.mark.parametrize("text", [False, True])
@pytest.mark.parametrize("trust,bypass", [(None, None), (False, "0"), (True, "1")])
def test_selected_native_classes_and_public_security(tmp_path, monkeypatch, text, trust, bypass):
    path = tmp_path / "selected"
    reference = save_fixture(path, text, hostile=True)
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    config = load_model_config(path, trust_remote_code=trust)
    assert type(config) is type(reference.config)
    model = GPTQModel.load(
        str(path), device="cpu", dtype=torch.float32, trust_remote_code=trust, attn_implementation="eager"
    )
    assert type(model.model) is type(reference)
    assert set(model.model.state_dict()) == set(reference.state_dict())
    ids = torch.tensor([[1, 4, 8, 11]])
    torch.testing.assert_close(output_tensor(model, ids), output_tensor(reference, ids))
    if text:
        assert model.processor is None
        assert not hasattr(model.model, "lm_head")
        with pytest.raises(NotImplementedError, match="headless"):
            model.generate(input_ids=ids, max_new_tokens=2)
    else:
        assert isinstance(model.processor, Qwen2VLProcessor)
        assert model.generate(input_ids=ids, max_new_tokens=2, do_sample=False).shape == (1, 6)
    assert not (path / "EXECUTED").exists()


@pytest.mark.parametrize("text", [False, True])
@pytest.mark.parametrize("kv_heads,dtype,implementation", [
    (1, torch.float32, "eager"), (2, torch.float16, "sdpa"), (4, torch.bfloat16, "eager"),
])
def test_native_cache_dtype_and_two_safetensors_cycles(tmp_path, text, kv_heads, dtype, implementation):
    torch.manual_seed(123)
    cfg = tiny_config(text, kv_heads, tied=kv_heads == 1)
    model = Qwen2VLQModel.loader.from_config(cfg, dtype=dtype, attn_implementation=implementation).eval()
    reference = type(model)._from_config(copy.deepcopy(cfg), dtype=dtype, attn_implementation=implementation).eval()
    reference.load_state_dict(model.state_dict())
    ids = torch.tensor([[1, 7, 4, 17, 8, 12]])
    with torch.no_grad():
        expected = output_tensor(reference, ids)
        torch.testing.assert_close(output_tensor(model, ids), expected)
        first = model(ids[:, :2], use_cache=True)
        continued = model(ids[:, 2:5], past_key_values=first.past_key_values, use_cache=True)
        tail = model(ids[:, 5:], past_key_values=continued.past_key_values, use_cache=True)
        actual = tail.last_hidden_state if text else tail.logits
        torch.testing.assert_close(
            actual,
            expected[:, -1:],
            atol=0.004 if dtype != torch.float32 else 1e-6,
            rtol=0.02 if dtype != torch.float32 else 1e-5,
        )
        assert tail.past_key_values.get_seq_length() == 6
        if not text:
            for beams in (1, 3):
                assert torch.equal(
                    model.generate(ids, max_new_tokens=3, num_beams=beams),
                    reference.generate(ids, max_new_tokens=3, num_beams=beams),
                )
        state = {k: v.clone() for k, v in model.state_dict().items()}
        for cycle in range(2):
            path = tmp_path / str(cycle)
            model.save_pretrained(path, safe_serialization=True)
            assert not list(path.glob("*.bin"))
            config = Qwen2VLQModel.loader.load_config(path)
            model = Qwen2VLQModel.loader.from_pretrained(
                path, config=config, dtype=dtype, attn_implementation=implementation
            ).eval()
            for key, value in state.items():
                assert torch.equal(model.state_dict()[key], value), key
            torch.testing.assert_close(output_tensor(model, ids), expected)


def test_failed_config_load_then_same_path_repair(tmp_path):
    path = tmp_path / "repair"
    reference = save_fixture(path, True)
    config_path = path / "config.json"
    valid = config_path.read_text()
    data = json.loads(valid)
    data["model_type"] = "qwen2"
    config_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsupported Qwen2-VL"):
        Qwen2VLQModel.loader.load_config(path)
    config_path.write_text(valid)
    reloaded = Qwen2VLQModel.loader.from_pretrained(path)
    ids = torch.tensor([[1, 4, 6]])
    torch.testing.assert_close(output_tensor(reloaded, ids), output_tensor(reference, ids))


@pytest.mark.parametrize("text", [False, True])
@pytest.mark.parametrize("assign", [False, True])
def test_direct_assignment_and_subfolder_public_reload(tmp_path, text, assign):
    path = tmp_path / "checkpoint" / "nested"
    expected = save_fixture(path, text)
    model = Qwen2VLQModel.loader.from_config(copy.deepcopy(expected.config))
    model.load_state_dict(expected.state_dict(), strict=True, assign=assign)
    ids = torch.tensor([[1, 4, 19, 8]])
    torch.testing.assert_close(output_tensor(model, ids), output_tensor(expected, ids))
    actual = GPTQModel.load(
        str(path.parent),
        subfolder="nested",
        dtype=torch.float32,
        device="cpu",
        trust_remote_code=False,
        attn_implementation="eager",
    )
    torch.testing.assert_close(output_tensor(actual.model, ids), output_tensor(expected, ids))


@pytest.mark.parametrize("text", [False, True])
def test_public_corrupt_safetensors_then_repaired_retry(tmp_path, text):
    from safetensors import SafetensorError

    expected = save_fixture(tmp_path, text)
    checkpoint = tmp_path / "model.safetensors"
    good = checkpoint.read_bytes()
    checkpoint.write_bytes(b"invalid checkpoint header")
    with pytest.raises((ValueError, OSError, SafetensorError), match="(?i)(header|safetensor)"):
        GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32)
    checkpoint.write_bytes(good)
    actual = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, attn_implementation="eager")
    ids = torch.tensor([[1, 9, 4, 16]])
    torch.testing.assert_close(output_tensor(actual.model, ids), output_tensor(expected, ids))


def test_foreign_config_rejected_without_initializing_wrong_model():
    from gptqmodel.models.ports.native.qwen2.configuration_qwen2 import Qwen2Config

    with pytest.raises(ValueError, match="exact native"):
        Qwen2VLQModel.loader.from_config(Qwen2Config(vocab_size=64, hidden_size=32))


@pytest.mark.parametrize("dtype,implementation", [(torch.float32, "eager"), (torch.bfloat16, "sdpa")])
def test_native_image_video_cache_and_generation(tmp_path, dtype, implementation):
    import numpy as np
    from PIL import Image

    save_fixture(tmp_path)
    model = GPTQModel.load(str(tmp_path), device="cpu", dtype=dtype, attn_implementation=implementation)
    image = Image.fromarray(((np.arange(8 * 8 * 3).reshape(8, 8, 3) * 7 + 13) % 256).astype(np.uint8))
    video = [image, image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)]
    inputs = model.processor(
        text=["t4 <|vision_start|><|image_pad|><|vision_end|> t7 <|vision_start|><|video_pad|><|vision_end|> t9"],
        images=[image],
        videos=[video],
        return_tensors="pt",
    ).to(dtype=dtype)
    assert inputs.image_grid_thw.shape == inputs.video_grid_thw.shape == (1, 3)
    with torch.no_grad():
        first = model.model(**inputs, use_cache=True)
        last_token = torch.tensor([[11]])
        continued = model.model(input_ids=last_token, past_key_values=first.past_key_values, use_cache=True)
        extended = dict(inputs)
        extended["input_ids"] = torch.cat([inputs.input_ids, last_token], -1)
        extended["attention_mask"] = torch.cat([inputs.attention_mask, torch.ones(1, 1, dtype=torch.long)], -1)
        extended["mm_token_type_ids"] = torch.cat([inputs.mm_token_type_ids, torch.zeros(1, 1, dtype=torch.long)], -1)
        full = model.model(**extended).logits[:, -1:]
        torch.testing.assert_close(
            continued.logits,
            full,
            atol=0.004 if dtype != torch.float32 else 1e-6,
            rtol=0.02 if dtype != torch.float32 else 1e-5,
        )
        for beams in (1, 3):
            generated = model.generate(**inputs, max_new_tokens=2, num_beams=beams, do_sample=False)
            assert generated.shape == (1, inputs.input_ids.shape[1] + 2)
            assert torch.equal(generated[:, : inputs.input_ids.shape[1]], inputs.input_ids)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_independent_decoder_equations_and_3axis_rope(kv_heads, implementation):
    import torch.nn.functional as F

    torch.manual_seed(219)
    cfg = tiny_config(True, kv_heads)
    model = Qwen2VLQModel.loader.from_config(cfg, attn_implementation=implementation).double().eval()
    # Nonzero Q/K/V biases prevent a projection-order or bias omission from hiding.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.copy_(torch.linspace(-0.13, 0.21, parameter.numel()))
    state = model.state_dict()
    ids = torch.tensor([[1, 7, 9, 4, 11], [1, 3, 12, 5, 13]])
    positions = torch.tensor([[7, 7, 8, 8, 9], [19, 20, 19, 20, 19], [37, 37, 38, 39, 38]])[:, None].expand(-1, 2, -1)
    head_dim, heads, seq = 8, 4, ids.shape[1]
    inv = 10000.0 ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    angles = positions[..., None].float() * inv
    # The first, second, final two frequency channels use T, H, W respectively.
    freq = torch.cat([angles[0, ..., :1], angles[1, ..., 1:2], angles[2, ..., 2:]], -1)
    cos, sin = torch.cat([freq.cos()] * 2, -1)[:, None], torch.cat([freq.sin()] * 2, -1)[:, None]

    def norm(x, weight):
        variance = x.float().square().mean(-1, keepdim=True)
        return weight * (x.float() * (variance + cfg.rms_norm_eps).rsqrt()).to(x.dtype)

    def rotate(x):
        return x * cos + torch.cat([-x[..., head_dim // 2 :], x[..., : head_dim // 2]], -1) * sin

    hidden = F.embedding(ids, state["embed_tokens.weight"])
    for index in range(2):
        root = f"layers.{index}."
        x = norm(hidden, state[root + "input_layernorm.weight"])
        q, k, v = [
            F.linear(x, state[root + f"self_attn.{name}_proj.weight"], state[root + f"self_attn.{name}_proj.bias"])
            .reshape(2, seq, -1, head_dim)
            .transpose(1, 2)
            for name in "qkv"
        ]
        # Native Qwen2-VL deliberately computes rotary in FP32 even for FP64 input.
        q = rotate(q.float()).double()
        k = rotate(k.float()).double().repeat_interleave(heads // kv_heads, dim=1)
        v = v.repeat_interleave(heads // kv_heads, dim=1)
        scores = q @ k.transpose(-1, -2) / head_dim**0.5
        scores.masked_fill_(torch.ones(seq, seq, dtype=torch.bool).triu(1), -torch.inf)
        attended = (scores.float().softmax(-1).double() @ v).transpose(1, 2).reshape(2, seq, 32)
        hidden = hidden + F.linear(attended, state[root + "self_attn.o_proj.weight"])
        x = norm(hidden, state[root + "post_attention_layernorm.weight"])
        hidden = hidden + F.linear(
            F.silu(F.linear(x, state[root + "mlp.gate_proj.weight"]))
            * F.linear(x, state[root + "mlp.up_proj.weight"]),
            state[root + "mlp.down_proj.weight"],
        )
    expected = norm(hidden, state["norm.weight"])
    with torch.no_grad():
        actual = model(ids, position_ids=positions).last_hidden_state
        torch.testing.assert_close(actual, expected, atol=4e-7, rtol=3e-6)
        changed = ids.clone()
        changed[:, -1] = 21
        torch.testing.assert_close(model(changed, position_ids=positions).last_hidden_state[:, :-1], actual[:, :-1])


@pytest.mark.parametrize("text", [False, True])
def test_real_pinned_tokenizer_and_media_processor(tmp_path, text):
    import hashlib

    import numpy as np
    from PIL import Image

    port = Path(qwen2_vl.__file__).parent
    assets = read_port_json(port / "released_processor_assets.json")
    source = tmp_path / "real-assets"
    source.mkdir()
    cfg = tiny_config(text)
    text_config = cfg if text else cfg.text_config
    text_config.vocab_size = 152064
    text_config.bos_token_id, text_config.eos_token_id, text_config.pad_token_id = 151643, 151645, 151643
    if not text:
        cfg.vision_config.patch_size = 14
        cfg.image_token_id, cfg.video_token_id = 151655, 151656
        cfg.vision_start_token_id, cfg.vision_end_token_id = 151652, 151653
    native_model(cfg).save_pretrained(source)
    for name, digest in assets["sha256"].items():
        content = read_port_asset(port / "released_assets" / name)
        assert hashlib.sha256(content).hexdigest() == digest
        (source / name).write_bytes(content)
    if not text:
        Qwen2VLVideoProcessor(
            patch_size=14, temporal_patch_size=2, merge_size=2, min_pixels=3136, max_pixels=3136
        ).save_pretrained(source)
    model = GPTQModel.load(str(source), dtype=torch.float32, device="cpu", attn_implementation="eager")
    for oracle in assets["oracles"]:
        assert model.tokenizer.encode(oracle["text"], add_special_tokens=False) == oracle["ids"]
        assert model.tokenizer.decode(oracle["ids"], skip_special_tokens=False) == oracle["text"]
    assert model.tokenizer.eos_token_id == 151645
    if text:
        with torch.no_grad():
            assert model.model(torch.tensor([assets["oracles"][0]["ids"]])).last_hidden_state.shape == (1, 9, 32)
        assert model.processor is None
        return
    processor = model.processor
    assert processor.tokenizer is model.tokenizer.tokenizer
    image_array = ((np.arange(56 * 56 * 3).reshape(56, 56, 3) * 7 + 17) % 256).astype(np.uint8)
    image = Image.fromarray(image_array)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert (
        prompt
        == "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe.<|im_end|>\n<|im_start|>assistant\n"
    )
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    assert inputs.image_grid_thw.tolist() == [[1, 4, 4]]
    assert (inputs.input_ids == cfg.image_token_id).sum().item() == 4
    normalized = torch.tensor(image_array.copy()).permute(2, 0, 1).float() / 255
    normalized = (normalized - torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]) / torch.tensor(
        [0.26862954, 0.26130258, 0.27577711]
    )[:, None, None]
    patches = []
    for group_y in range(0, 4, 2):
        for group_x in range(0, 4, 2):
            for y in range(group_y, group_y + 2):
                for x in range(group_x, group_x + 2):
                    patches.append(
                        normalized[:, y * 14 : (y + 1) * 14, x * 14 : (x + 1) * 14][:, None]
                        .expand(-1, 2, -1, -1)
                        .reshape(-1)
                    )
    torch.testing.assert_close(inputs.pixel_values, torch.stack(patches), atol=5e-7, rtol=1e-6)
    with torch.no_grad():
        logits = model.model(**inputs).logits
        changed = dict(inputs)
        changed["pixel_values"] = inputs.pixel_values.flip(1)
        assert not torch.allclose(model.model(**changed).logits[:, -1], logits[:, -1])
        assert model.generate(**inputs, max_new_tokens=2, do_sample=False).shape[1] == inputs.input_ids.shape[1] + 2
    destination = tmp_path / "processor-reload"
    processor.save_pretrained(destination)
    restored = OracleQwen2VLProcessor.from_pretrained(destination)
    restored_inputs = restored(text=[prompt], images=[image], return_tensors="pt")
    for key in inputs:
        torch.testing.assert_close(restored_inputs[key], inputs[key])


@pytest.mark.parametrize("text,legacy", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("offload", [False, True])
def test_public_awq_pack_and_two_reloads(tmp_path, text, legacy, offload):
    from gptqmodel import BACKEND
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    source = tmp_path / "source"
    reference = save_fixture(source, text, legacy=legacy).half()
    qcfg = AWQConfig(
        device="cpu",
        group_size=16,
        fallback=None,
        offload_to_disk=offload,
        offload_to_disk_path=str(tmp_path / "offload"),
    )
    wrapper = GPTQModel.load(
        str(source), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16, attn_implementation="eager"
    )
    calibration = [
        {"input_ids": torch.tensor([[1, 4, 7, 13, 9, 12, 6, 8]]), "attention_mask": torch.ones(1, 8, dtype=torch.long)}
        for _ in range(2)
    ]
    quantization_dataset = calibration
    if not text:
        import numpy as np
        from PIL import Image

        image = Image.fromarray(((np.arange(8 * 8 * 3).reshape(8, 8, 3) * 7) % 256).astype(np.uint8))
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": "t4 t7 "}, {"type": "image", "image": image}]}
        ]
        quantization_dataset = [conversation, conversation]
        prompt = wrapper.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        calibration = [wrapper.processor(text=[prompt], images=[image], return_tensors="pt")]
        expected_capture, actual_capture = [], []
        handle = reference.model.language_model.layers[0].register_forward_pre_hook(
            lambda _module, args: expected_capture.append(args[0].detach().clone())
        )
        with torch.no_grad():
            reference(**calibration[0])
        handle.remove()
        original_capture = wrapper.run_input_capture

        def capture(example, use_cache, data_device):
            handle = wrapper.model.model.language_model.layers[0].register_forward_pre_hook(
                lambda _module, args: actual_capture.append(args[0].detach().clone()), prepend=True
            )
            try:
                return original_capture(example, use_cache, data_device)
            finally:
                handle.remove()

        wrapper.run_input_capture = capture
    wrapper.quantize(
        CalibrationConfig(dataset=quantization_dataset, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    if not text:
        assert actual_capture
        for actual in actual_capture:
            torch.testing.assert_close(actual, expected_capture[0])
    assert sum(isinstance(m, AwqTorchLinear) for m in wrapper.model.modules()) == 14
    for name, value in reference.state_dict().items():
        if name in wrapper.model.state_dict():
            actual = wrapper.model.state_dict()[name]
            expect_fold = name.endswith((".input_layernorm.weight", ".post_attention_layernorm.weight"))
            assert torch.equal(actual, value) is not expect_fold, name
    ids = calibration[0]["input_ids"]
    with torch.no_grad():
        media = {} if text else {k: v for k, v in calibration[0].items() if k not in ("input_ids", "attention_mask")}
        expected = output_tensor(wrapper.model, ids, **media)
        assert torch.isfinite(expected).all()
        state = {k: v.detach().clone() for k, v in wrapper.model.state_dict().items()}
        for cycle in range(2):
            path = tmp_path / f"packed-{cycle}"
            wrapper.save(str(path))
            assert list(path.glob("*.safetensors"))
            wrapper = GPTQModel.load(
                str(path), device="cpu", backend=BACKEND.TORCH, dtype=torch.float16, attn_implementation="eager"
            )
            assert set(wrapper.model.state_dict()) == set(state)
            for key, value in state.items():
                torch.testing.assert_close(wrapper.model.state_dict()[key], value, atol=0, rtol=0)
            torch.testing.assert_close(output_tensor(wrapper.model, ids, **media), expected)
