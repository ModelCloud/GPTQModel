# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import types

from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import nn
from transformers import PreTrainedTokenizerFast

from gptqmodel.models.definitions import base_qwen2_5_omni, base_qwen2_vl
from gptqmodel.utils.hf import load_tokenizer


def test_qwen2_vl_image_only_process_vision_info_returns_image_list():
    image = Image.new("RGB", (2, 2), color="white")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    image_inputs = base_qwen2_vl.BaseQwen2VLGPTQ.process_vision_info(messages)

    assert isinstance(image_inputs, list)
    assert image_inputs == [image]


def test_qwen2_vl_pre_quantize_hooks_use_inner_model_layout():
    instance = object.__new__(base_qwen2_vl.BaseQwen2VLGPTQ)
    instance.model = types.SimpleNamespace(
        language_model=types.SimpleNamespace(
            embed_tokens=nn.Embedding(4, 4),
            rotary_emb=nn.Identity(),
        ),
        visual=nn.Identity(),
    )
    instance.quantize_config = types.SimpleNamespace(
        device="cpu",
        offload_to_disk=False,
        offload_to_disk_path="/tmp/unused",
    )

    instance.pre_quantize_generate_hook_start()
    instance.pre_quantize_generate_hook_end()

    assert instance.model.language_model.embed_tokens.weight.device.type == "cpu"


def test_qwen2_5_omni_image_only_process_vision_info_returns_image_list():
    image = Image.new("RGB", (2, 2), color="white")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    image_inputs = base_qwen2_5_omni.BaseQwen2_5_OmniGPTQ.process_vision_info(messages)

    assert isinstance(image_inputs, list)
    assert image_inputs == [image]


def test_qwen2_5_omni_forward_delegates_to_thinker():
    sentinel = object()
    instance = object.__new__(base_qwen2_5_omni.BaseQwen2_5_OmniGPTQ)
    instance.model = types.SimpleNamespace(thinker=lambda *args, **kwargs: (args, kwargs, sentinel))

    result = instance.forward("hello", temperature=0.1)

    assert result == (("hello",), {"temperature": 0.1}, sentinel)


def test_qwen2_5_omni_pre_quantize_hooks_use_thinker_layout():
    loaded_speakers = []
    materialized = []

    class _Visual(nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_pos_emb = nn.Identity()

    class _Thinker(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = types.SimpleNamespace(
                embed_tokens=nn.Embedding(4, 4),
                rotary_emb=nn.Identity(),
                layers=[
                    types.SimpleNamespace(
                        self_attn=types.SimpleNamespace(rotary_emb=nn.Identity()),
                        )
                    ],
                )
            self.visual = _Visual()
            self.audio_tower = nn.Identity()

    instance = object.__new__(base_qwen2_5_omni.BaseQwen2_5_OmniGPTQ)
    instance.model_local_path = "/tmp/qwen2_5_omni"
    instance.model = types.SimpleNamespace(
        load_speakers=lambda path: loaded_speakers.append(path),
        thinker=_Thinker(),
    )
    instance.quantize_config = types.SimpleNamespace(
        device="cpu",
        offload_to_disk=False,
        offload_to_disk_path="/tmp/unused",
    )
    instance.shell_module_materialize = lambda module, device: materialized.append((module, device)) or module

    instance.pre_quantize_generate_hook_start()
    instance.pre_quantize_generate_hook_end()

    assert loaded_speakers == ["/tmp/qwen2_5_omni/spk_dict.pt"]
    assert len(materialized) == 6
    assert instance.model.thinker.model.embed_tokens.weight.device.type == "cpu"


def test_load_tokenizer_uses_text_config_for_qwen2_5_omni_style_composite_configs():
    backend = Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "hello": 2}, unk_token="<pad>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>")

    text_config = types.SimpleNamespace(
        model_type="qwen2_5_omni_text",
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
    )

    class _CompositeConfig:
        def get_text_config(self):
            return text_config

    wrapped = load_tokenizer(tokenizer, model_config=_CompositeConfig())

    assert wrapped.model_config is text_config
    assert wrapped.eos_token_id == tokenizer.eos_token_id
    assert text_config.eos_token_id == tokenizer.eos_token_id
