# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os.path
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from model_test import ModelTest
from ovis import image_to_test_dataset
from PIL import Image
from safetensors.torch import save_file
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText

from gptqmodel.models import auto
from gptqmodel.models.definitions.deepseek_vl import DeepSeekVLQModel
from gptqmodel.utils.structure import LazyTurtle


MODEL_PATH = Path("/monster/data/model/deepseek-vl-1.3b-chat")


def test_deepseek_vl_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="deepseek_vl")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/deepseek-vl") is DeepSeekVLQModel


def test_deepseek_vl_module_tree_covers_llama_decoder_paths():
    layer_modules = DeepSeekVLQModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert DeepSeekVLQModel.layer_modules_strict is False
    assert DeepSeekVLQModel.require_load_processor is True
    assert DeepSeekVLQModel.support_batch_quantize is False
    assert DeepSeekVLQModel.pre_lm_head_norm_module == "model.language_model.norm"
    assert DeepSeekVLQModel.rotary_embedding == "model.language_model.rotary_emb"
    assert DeepSeekVLQModel.extract_layers_node() == ["model.language_model.layers"]

    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.k_proj" in flat_modules
    assert "self_attn.v_proj" in flat_modules
    assert "self_attn.o_proj" in flat_modules
    assert "mlp.gate_proj" in flat_modules
    assert "mlp.up_proj" in flat_modules
    assert "mlp.down_proj" in flat_modules


def test_deepseek_vl_base_modules_include_vision_aligner_and_language_roots():
    class _LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(4, 4)
            self.layers = nn.ModuleList([nn.Identity()])
            self.norm = nn.LayerNorm(4)
            self.rotary_emb = nn.Identity()

    class _CoreModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = nn.Identity()
            self.aligner = nn.Identity()
            self.language_model = _LanguageModel()

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _CoreModel()

    base_modules = set(DeepSeekVLQModel.get_base_modules(_Wrapper()))

    assert "model.vision_model" in base_modules
    assert "model.aligner" in base_modules
    assert "model.language_model.embed_tokens" in base_modules
    assert "model.language_model.norm" in base_modules
    assert "model.language_model.rotary_emb" in base_modules


def test_deepseek_vl_pre_quantize_hook_materializes_base_modules():
    model = nn.Module()
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    model.model.language_model.embed_tokens = nn.Embedding(4, 4)
    model.model.language_model.norm = nn.LayerNorm(4)
    model.model.language_model.rotary_emb = nn.Identity()
    model.model.vision_model = nn.Linear(4, 4)
    model.model.aligner = nn.Linear(4, 4)

    qmodel = object.__new__(DeepSeekVLQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = model
    qmodel.quantize_config = SimpleNamespace(device=torch.device("cpu"))
    materialized = []

    def shell_module_materialize(module, device):
        materialized.append((module, device))
        return module

    qmodel.shell_module_materialize = shell_module_materialize

    qmodel.pre_quantize_generate_hook_start()

    assert materialized == [
        (model.model.language_model.embed_tokens, torch.device("cpu")),
        (model.model.language_model.norm, torch.device("cpu")),
        (model.model.language_model.rotary_emb, torch.device("cpu")),
        (model.model.vision_model, torch.device("cpu")),
        (model.model.aligner, torch.device("cpu")),
    ]


def test_prepare_deepseek_vl_dataset_reuses_shared_dataset(monkeypatch):
    calls = {}

    def fake_prepare_dataset(format_func, n_sample):
        calls["format_func"] = format_func
        calls["n_sample"] = n_sample
        return [format_func("https://example.com/cat.jpg", "caption")]

    monkeypatch.setattr(image_to_test_dataset, "prepare_dataset", fake_prepare_dataset)

    dataset = image_to_test_dataset.prepare_deepseek_vl_dataset(n_sample=3)

    assert calls == {
        "format_func": image_to_test_dataset.format_deepseek_vl_dataset,
        "n_sample": 3,
    }
    assert dataset == [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://example.com/cat.jpg"},
                    {"type": "text", "text": "generate a caption for this image"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "caption"},
                ],
            },
        ]
    ]


def test_deepseek_vl_lazy_turtle_resolves_siglip_vision_model_prefix(tmp_path):
    checkpoint_tensors = {
        "model.vision_model.vision_model.embeddings.patch_embedding.weight": torch.zeros(2, 2),
    }
    model_dir = tmp_path / "source_model"
    model_dir.mkdir()
    shard_name = "model.safetensors"
    save_file(checkpoint_tensors, str(model_dir / shard_name))
    (model_dir / "model.safetensors.index.json").write_text(
        '{"weight_map":{"model.vision_model.vision_model.embeddings.patch_embedding.weight":"model.safetensors"}}',
        encoding="utf-8",
    )

    class _PatchEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(2, 2, device="meta"))

    class _Embeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embedding = _PatchEmbedding()

    class _VisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = _Embeddings()

    class _CoreModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = _VisionModel()

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _CoreModel()

    shell = _Wrapper()
    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
        module_tree=DeepSeekVLQModel.module_tree,
        hf_conversion_map_reversed=DeepSeekVLQModel.resolve_hf_conversion_map_reversed(),
        target_model=shell,
    )

    assert turtle is not None
    assert (
        turtle._resolve_checkpoint_tensor_name(
            "model.vision_model.embeddings.patch_embedding",
            "weight",
        )
        == "model.vision_model.vision_model.embeddings.patch_embedding.weight"
    )


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="DeepSeek-VL model not found")
def test_deepseek_vl_native_shell_matches_definition_tree():
    from accelerate import init_empty_weights

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=False)
    with init_empty_weights(include_buffers=True):
        shell = AutoModelForImageTextToText.from_config(config, trust_remote_code=False)

    layer = shell.model.language_model.layers[0]

    assert config.model_type == "deepseek_vl"
    assert auto.check_and_get_model_definition(MODEL_PATH) is DeepSeekVLQModel
    assert hasattr(shell.model, "vision_model")
    assert hasattr(shell.model, "aligner")
    assert hasattr(shell.model, "language_model")
    assert hasattr(layer.self_attn, "q_proj")
    assert hasattr(layer.self_attn, "k_proj")
    assert hasattr(layer.self_attn, "v_proj")
    assert hasattr(layer.self_attn, "o_proj")
    assert hasattr(layer.mlp, "gate_proj")
    assert hasattr(layer.mlp, "up_proj")
    assert hasattr(layer.mlp, "down_proj")


class TestDeepSeekVL(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/deepseek-vl-1.3b-chat"
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False
    OFFLOAD_TO_DISK = False
    EVAL_BATCH_SIZE = 1

    def test_deepseek_vl(self):
        with self.model_compat_test_context():
            model, _tokenizer, processor = self.quantModel(
                self.NATIVE_MODEL_ID,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                batch_size=1,
                call_perform_post_quant_validation=False,
            )

        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ovis/10016.jpg")
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
                    {"type": "text", "text": "What does this picture show?"},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        print("ggg", generated_ids)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        print("output_text:", output_text)

        self.assertIn("snow", output_text.lower())
        self.check_kernel(model, self.KERNEL_INFERENCE)
