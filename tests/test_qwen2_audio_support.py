# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for the Qwen2-Audio (audio-to-text) quantization path.

The real `Qwen/Qwen2-Audio-7B-Instruct` checkpoint is GPU-evaluated by
`tests/models/test_qwen2_audio.py`. These tests keep the cheap contract checks on
CPU: the registry wiring, the module tree shape, audio-tower handling, the
processor-backed calibration preparation, and a tiny end-to-end GPTQ
quantize/save/reload smoke run.
"""

from types import SimpleNamespace

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import nn
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2AudioConfig,
    Qwen2AudioForConditionalGeneration,
    Qwen2AudioProcessor,
    WhisperFeatureExtractor,
)

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.models import auto
from gptqmodel.models.definitions import qwen2_audio as qwen2_audio_definition
from gptqmodel.models.definitions.qwen2_audio import Qwen2AudioGPTQ
from gptqmodel.nn_modules.qlinear.torch import TorchLinear


AUDIO_TOKEN = "<|AUDIO|>"
AUDIO_TOKEN_ID = 4


def _tiny_qwen2_audio_config() -> Qwen2AudioConfig:
    return Qwen2AudioConfig(
        audio_config={
            "d_model": 16,
            "encoder_layers": 2,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "num_mel_bins": 8,
            "max_source_positions": 100,
            "output_dim": 16,
            "scale_embedding": True,
        },
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
            "max_position_embeddings": 128,
        },
        audio_token_id=AUDIO_TOKEN_ID,
    )


def _build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, AUDIO_TOKEN: AUDIO_TOKEN_ID}
    vocab.update({f"tok{i}": i + 5 for i in range(123)})
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        additional_special_tokens=[AUDIO_TOKEN],
    )
    # The upstream Qwen2-Audio tokenizer ships a text-only template while the processor
    # keeps the multimodal (audio-aware) one; mirror that split so the save path is covered.
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    return tokenizer


def _build_tiny_fixture(path) -> Qwen2AudioForConditionalGeneration:
    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = Qwen2AudioForConditionalGeneration(_tiny_qwen2_audio_config())
    model.save_pretrained(path)

    feature_extractor = WhisperFeatureExtractor(
        feature_size=8,
        sampling_rate=16000,
        hop_length=160,
        n_fft=400,
        chunk_length=2,
    )
    processor = Qwen2AudioProcessor(
        feature_extractor=feature_extractor,
        tokenizer=_build_tiny_tokenizer(),
    )
    processor.save_pretrained(path)
    # Mirror the upstream Qwen2-Audio checkpoint layout: no top-level `chat_template.jinja`
    # yet, the processor falls back to its audio-aware default template, and the tokenizer
    # carries a text-only template inside `tokenizer_config.json`.
    (path / "chat_template.jinja").unlink()
    processor.tokenizer.save_pretrained(path, save_jinja_files=False)
    return model


def _synthetic_audio(sample_rate: int = 16000) -> np.ndarray:
    timeline = np.arange(sample_rate // 2, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * timeline / sample_rate).astype(np.float32)


def test_qwen2_audio_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="qwen2_audio")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/qwen2-audio") is Qwen2AudioGPTQ
    assert auto.MODEL_MAP["qwen2_audio"] is Qwen2AudioGPTQ


def test_qwen2_audio_definition_metadata():
    from transformers import AutoModelForPreTraining

    assert Qwen2AudioGPTQ.loader is AutoModelForPreTraining
    assert Qwen2AudioGPTQ.require_load_processor is True
    assert Qwen2AudioGPTQ.pre_lm_head_norm_module == "model.language_model.norm"
    assert Qwen2AudioGPTQ.rotary_embedding == "model.language_model.rotary_emb"
    assert Qwen2AudioGPTQ.extract_layers_node() == ["model.language_model.layers"]


def test_qwen2_audio_definition_is_exported_from_definitions_package():
    from gptqmodel.models import definitions

    assert definitions.Qwen2AudioGPTQ is Qwen2AudioGPTQ


def test_qwen2_audio_module_tree_covers_decoder_paths():
    config = _tiny_qwen2_audio_config()
    layer_modules = Qwen2AudioGPTQ.simple_layer_modules(
        model_config=config,
        quantize_config=SimpleNamespace(dynamic=None),
    )

    assert layer_modules == [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]

    full_modules = {name for block in Qwen2AudioGPTQ.full_layer_modules(config) for name in block}
    assert "input_layernorm:!" in full_modules
    assert "post_attention_layernorm:!" in full_modules

    model = Qwen2AudioForConditionalGeneration(config)
    decoder_layer = model.model.language_model.layers[0]
    for module_name in {name.split(":")[0] for name in full_modules}:
        module = decoder_layer
        for attribute in module_name.split("."):
            module = getattr(module, attribute)
        assert module is not None


def test_qwen2_audio_base_modules_cover_audio_stack_and_text_endpoints():
    model = Qwen2AudioForConditionalGeneration(_tiny_qwen2_audio_config())

    base_modules = set(Qwen2AudioGPTQ.get_base_modules(model))

    assert base_modules == {
        "model.audio_tower",
        "model.multi_modal_projector",
        "model.language_model.embed_tokens",
        "model.language_model.norm",
        "model.language_model.rotary_emb",
    }


def _fake_qwen2_audio_qmodel():
    model = nn.Module()
    model.model = nn.Module()
    core_model = model.model
    core_model.audio_tower = nn.Linear(4, 4)
    core_model.multi_modal_projector = nn.Linear(4, 4)
    core_model.language_model = nn.Module()
    language_model = core_model.language_model
    language_model.embed_tokens = nn.Embedding(4, 4)
    language_model.rotary_emb = nn.Linear(4, 4)

    qmodel = object.__new__(Qwen2AudioGPTQ)
    nn.Module.__init__(qmodel)
    qmodel.model = model
    qmodel.turtle_model = None
    qmodel._direct_parameter_names = ()
    return qmodel, model


def test_qwen2_audio_pre_quantize_hook_materializes_audio_stack():
    qmodel, model = _fake_qwen2_audio_qmodel()
    qmodel.quantize_config = SimpleNamespace(device=torch.device("cpu"))
    materialized = []

    def shell_module_materialize(module, device):
        materialized.append((module, device))
        return module

    qmodel.shell_module_materialize = shell_module_materialize

    qmodel.pre_quantize_generate_hook_start()

    core_model = model.model
    assert materialized == [
        (core_model.audio_tower, torch.device("cpu")),
        (core_model.multi_modal_projector, torch.device("cpu")),
        (core_model.language_model.embed_tokens, torch.device("cpu")),
        (core_model.language_model.rotary_emb, torch.device("cpu")),
    ]


def test_qwen2_audio_pre_quantize_hook_end_offloads_audio_stack(monkeypatch):
    qmodel, model = _fake_qwen2_audio_qmodel()
    qmodel.quantize_config = SimpleNamespace(
        device=torch.device("cpu"),
        offload_to_disk=True,
        offload_to_disk_path="/tmp/qwen2-audio-offload",
    )
    offloaded = []

    def offload_to_disk(model, module, disk_path):
        offloaded.append((model, module, disk_path))

    monkeypatch.setattr(qwen2_audio_definition, "offload_to_disk", offload_to_disk)

    qmodel.pre_quantize_generate_hook_end()

    core_model = model.model
    language_model = core_model.language_model
    assert offloaded == [
        (core_model, core_model.audio_tower, "/tmp/qwen2-audio-offload"),
        (core_model, core_model.multi_modal_projector, "/tmp/qwen2-audio-offload"),
        (language_model, language_model.embed_tokens, "/tmp/qwen2-audio-offload"),
        (language_model, language_model.rotary_emb, "/tmp/qwen2-audio-offload"),
    ]


def test_qwen2_audio_prepare_dataset_keeps_text_and_audio_inputs(tmp_path):
    model_dir = tmp_path / "native"
    _build_tiny_fixture(model_dir)

    qmodel = object.__new__(Qwen2AudioGPTQ)
    nn.Module.__init__(qmodel)
    qmodel.model_local_path = str(model_dir)

    audio = _synthetic_audio()
    calibration = [
        "plain text calibration sample",
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": "What is this sound?"},
                ],
            }
        ],
    ]

    prepared = qmodel.prepare_dataset(calibration, batch_size=1)

    assert len(prepared) == 2

    text_batch = prepared[0]
    assert "input_features" not in text_batch
    assert AUDIO_TOKEN_ID not in text_batch["input_ids"].tolist()[0]

    audio_batch = prepared[1]
    assert "input_features" in audio_batch
    assert "feature_attention_mask" in audio_batch
    assert AUDIO_TOKEN_ID in audio_batch["input_ids"].tolist()[0]


def test_tiny_qwen2_audio_gptq_quantization_smoke(tmp_path):
    model_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"
    _build_tiny_fixture(model_dir)

    audio = _synthetic_audio()
    calibration = [
        "tiny qwen2 audio calibration sample one with enough tokens to survive filtering",
        "tiny qwen2 audio calibration sample two with repeated words for the decoder path",
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": "What is this sound?"},
                ],
            }
        ],
    ]

    model = GPTQModel.load(
        str(model_dir),
        quantize_config=QuantizeConfig(bits=4, group_size=32, desc_act=False, device="cpu"),
        backend=BACKEND.TORCH,
    )
    model.quantize(
        calibration,
        batch_size=1,
        backend=BACKEND.TORCH,
        calibration_data_min_length=1,
    )
    model.save(quantized_dir)

    quantized = GPTQModel.load(str(quantized_dir), backend=BACKEND.TORCH, device="cpu")
    modules = dict(quantized.model.named_modules())

    # `tokenizer.save_pretrained` must not clobber the processor-level chat template.
    saved_template = (quantized_dir / "chat_template.jinja").read_text(encoding="utf-8")
    assert "AUDIO" in saved_template

    assert isinstance(modules["model.language_model.layers.0.self_attn.q_proj"], TorchLinear)
    assert isinstance(modules["model.language_model.layers.0.mlp.down_proj"], TorchLinear)
    # The audio tower and projector stay dense so audio conditioning is preserved.
    assert isinstance(modules["model.multi_modal_projector.linear"], torch.nn.Linear)
    assert not isinstance(modules["model.audio_tower.layers.0.self_attn.q_proj"], TorchLinear)
    assert not isinstance(modules["model.audio_tower.layers.0.fc1"], TorchLinear)

    processor = Qwen2AudioProcessor.from_pretrained(quantized_dir)

    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "Say hello"}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=text, return_tensors="pt")
    generated = quantized.generate(**inputs, max_new_tokens=4, do_sample=False)
    assert generated.shape[1] > inputs["input_ids"].shape[1]

    audio_text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio", "audio": audio}, {"type": "text", "text": "What is this?"}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    audio_inputs = processor(text=audio_text, audio=[audio], return_tensors="pt")
    assert "input_features" in audio_inputs
    generated = quantized.generate(**audio_inputs, max_new_tokens=4, do_sample=False)
    assert generated.shape[1] > audio_inputs["input_ids"].shape[1]
