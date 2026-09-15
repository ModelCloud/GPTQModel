# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json

import torch
import torch.nn as nn
from safetensors.torch import save_file

from gptqmodel.nn_modules.qlinear.torch import TorchQuantEmbeddings
from gptqmodel.utils.model import find_modules, is_embeddings_module_quantized


def test_find_modules_includes_embeddings():
    model = nn.Module()
    model.embed_tokens = nn.Embedding(16, 8)
    model.proj = nn.Linear(8, 8)

    modules = find_modules(model)

    assert modules["embed_tokens"] is model.embed_tokens
    assert modules["proj"] is model.proj


def test_quantized_embedding_detection_reads_safetensors_keys(tmp_path):
    save_file(
        {
            "model.embed_tokens.qweight": torch.zeros((2, 2), dtype=torch.int32),
            "model.embed_tokens.scales": torch.ones((2, 2), dtype=torch.float16),
            "lm_head.weight": torch.zeros((2, 2), dtype=torch.float16),
        },
        tmp_path / "model.safetensors",
    )

    assert is_embeddings_module_quantized(
        model_dir=str(tmp_path),
        input_embed_name="model.embed_tokens",
        output_embed_name="lm_head",
    ) == (True, False)


def test_quantized_embedding_detection_reads_sharded_index(tmp_path):
    index = {
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "lm_head.qweight": "model-00002-of-00002.safetensors",
            "lm_head.qzeros": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    assert is_embeddings_module_quantized(
        model_dir=str(tmp_path),
        input_embed_name="model.embed_tokens",
        output_embed_name="lm_head",
    ) == (False, True)


def test_embedding_kernel_is_role_only():
    assert TorchQuantEmbeddings.SUPPORTS_BACKEND_SELECTION is False
