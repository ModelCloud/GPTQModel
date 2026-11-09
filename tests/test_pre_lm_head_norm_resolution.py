# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for pre-lm-head norm path resolution.

Definitions may declare one path or an ordered list of layout alternatives.
The field must be normalized exactly once: wrapping a list in another list (or
calling string methods on it) makes multimodal quantization crash before the
final norm runs.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate import init_empty_weights
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForImageTextToText, AutoModelForPreTraining

from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions.base_qwen3_vl import BaseQwen3VLGPTQ
from gptqmodel.models.definitions.ernie4_5_vl_moe import Ernie4_5_VLMoeQModel
from gptqmodel.models.definitions.glm4v import Glm4vGPTQ
from gptqmodel.models.definitions.glm4v_moe import Glm4vMoeQModel
from gptqmodel.models.definitions.glmasr import GlmASRGPTQ
from gptqmodel.models.definitions.llama4 import Llama4QModel
from gptqmodel.models.definitions.mistral3 import Mistral3GPTQ
from gptqmodel.models.definitions.ovis2 import Ovis2QModel
from gptqmodel.models.definitions.qwen2_5_vl import Qwen2_5_VLQModel
from gptqmodel.models.definitions.qwen2_vl import Qwen2VLQModel
from gptqmodel.models.definitions.qwen3_vl import Qwen3_VL_MoeQModel, Qwen3_VLQModel
from gptqmodel.models.definitions.voxtral import VoxtralGPTQ


def _bare(cls, model=None):
    instance = cls.__new__(cls)
    nn.Module.__init__(instance)
    if model is not None:
        instance.model = model
    instance.quantized = True
    instance.load_quantized_model = False
    instance.quantize_config = SimpleNamespace(
        device=torch.device("cpu"),
        activation_mode=None,
        activation_recipe=None,
    )
    return instance


def _module_with_path(path: str, dim: int = 8) -> nn.Module:
    root = nn.Module()
    node = root
    parts = path.split(".")
    for part in parts[:-1]:
        child = nn.Module()
        setattr(node, part, child)
        node = child
    setattr(node, parts[-1], nn.RMSNorm(dim))
    return root


class _MissingNormQModel(BaseQModel):
    pre_lm_head_norm_module = "model.missing_norm"
    module_tree = ["model", "layers", "#", {}]


def _missing_norm_model() -> nn.Module:
    model = nn.Module()
    model.layers = nn.ModuleList()
    return model


def test_pre_lm_head_norm_candidates_normalize_declaration():
    assert _bare(BaseQModel).pre_lm_head_norm_module is None
    assert _bare(BaseQModel)._pre_lm_head_norm_candidates() == []

    string_instance = _bare(BaseQModel)
    string_instance.pre_lm_head_norm_module = "model.norm"
    assert string_instance._pre_lm_head_norm_candidates() == ["model.norm"]

    list_instance = _bare(BaseQModel)
    list_instance.pre_lm_head_norm_module = ["model.language_model.norm", "model.norm", "model.norm"]
    assert list_instance._pre_lm_head_norm_candidates() == ["model.language_model.norm", "model.norm"]

    tuple_instance = _bare(BaseQModel)
    tuple_instance.pre_lm_head_norm_module = ("a.norm", "b.norm")
    assert tuple_instance._pre_lm_head_norm_candidates() == ["a.norm", "b.norm"]


@pytest.mark.parametrize("declared", [123, ["model.norm", None], [""], [object()]])
def test_pre_lm_head_norm_candidates_reject_invalid_declarations(declared):
    instance = _bare(BaseQModel)
    instance.pre_lm_head_norm_module = declared
    with pytest.raises(TypeError, match="pre_lm_head_norm_module"):
        instance._pre_lm_head_norm_candidates()


def test_list_valued_norm_path_resolves_without_nesting():
    """C1: a list declaration must not be wrapped again or used as a string."""

    instance = _bare(Qwen2VLQModel, _module_with_path("model.language_model.norm"))
    assert instance._pre_lm_head_norm_candidates() == [
        "model.language_model.norm",
        "language_model.norm",
    ]
    assert instance.norm_module_names(skip=("no-match",)) == ["model.language_model.norm"]

    # The hook must resolve the same flat candidate list.
    value = torch.randn(2, 8)
    output = instance.lm_head_pre_quantize_generate_hook([[value.clone()]])
    assert torch.allclose(output[0][0], nn.RMSNorm(8)(value), atol=1e-6)


def test_missing_declared_norm_raises_instead_of_passing_none():
    """C2: a declared-but-missing path must raise, never call pre_quantize(None)."""

    instance = _bare(_MissingNormQModel, _missing_norm_model())
    with pytest.raises(ValueError, match="cannot resolve `pre_lm_head_norm_module`"):
        instance.norm_module_names()
    with pytest.raises(ValueError, match="cannot resolve `pre_lm_head_norm_module`"):
        instance.lm_head_pre_quantize_generate_hook([[torch.randn(2, 8)]])


def test_declared_norm_path_falls_back_to_language_model_layout():
    """C2: Mistral3 keeps its final norm under ``model.language_model``."""

    instance = _bare(Mistral3GPTQ, _module_with_path("model.language_model.norm"))
    assert instance.norm_module_names() == ["model.language_model.norm"]
    value = torch.randn(2, 8)
    output = instance.lm_head_pre_quantize_generate_hook([[value.clone()]])
    assert torch.allclose(output[0][0], nn.RMSNorm(8)(value), atol=1e-6)


def test_qwen3_vl_base_candidates_cover_wrapped_and_direct_layouts():
    """Qwen3-VL's base definition covers both supported wrapper layouts."""

    wrapped = _bare(BaseQwen3VLGPTQ, _module_with_path("model.language_model.norm"))
    assert wrapped.norm_module_names() == ["model.language_model.norm"]

    direct = _bare(BaseQwen3VLGPTQ, _module_with_path("language_model.norm"))
    assert direct.norm_module_names() == ["language_model.norm"]


def _meta_model(loader, model_id):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    with init_empty_weights(include_buffers=True):
        model = loader.from_config(config, trust_remote_code=False)
    return config, model


_META_CASES = [
    # definition, loader, checkpoint, expected resolved path
    (Qwen3_VLQModel, AutoModelForImageTextToText, "Qwen3-VL-2B-Instruct", "model.language_model.norm"),
    (Mistral3GPTQ, AutoModelForPreTraining, "Ministral-3-3B-Instruct-2512", "model.language_model.norm"),
    (Glm4vGPTQ, AutoModelForImageTextToText, "GLM-4.1V-9B-Thinking", "model.language_model.norm"),
    (VoxtralGPTQ, AutoModelForPreTraining, "Voxtral-Mini-3B-2507", "model.language_model.norm"),
    (Ovis2QModel, AutoModelForImageTextToText, "Ovis2-2B-hf", "model.language_model.norm"),
    (GlmASRGPTQ, AutoModel, "GLM-ASR-Nano-2512", "language_model.norm"),
    (Qwen2VLQModel, AutoModelForImageTextToText, "Qwen2-VL-2B-Instruct", "model.language_model.norm"),
    (Ernie4_5_VLMoeQModel, AutoModelForImageTextToText, "ERNIE-4.5-VL-28B-A3B-PT", "model.language_model.norm"),
    (Qwen3_VL_MoeQModel, AutoModelForImageTextToText, "Qwen3-VL-30B-A3B-Instruct", "model.language_model.norm"),
    (Qwen2_5_VLQModel, AutoModelForImageTextToText, "Qwen2.5-VL-3B-Instruct", "model.language_model.norm"),
    (Glm4vMoeQModel, AutoModelForImageTextToText, "GLM-4.5V", "model.language_model.norm"),
    (Llama4QModel, AutoModelForImageTextToText, "Llama-4-Scout-17B-16E-Instruct", "language_model.model.norm"),
]


@pytest.mark.parametrize("definition,loader,checkpoint,expected", _META_CASES)
def test_declared_norm_path_resolves_on_meta_model(definition, loader, checkpoint, expected):
    model_id = Path("/monster/data/model") / checkpoint
    if not model_id.exists():
        pytest.skip(f"local checkpoint missing: {model_id}")

    config, model = _meta_model(loader, str(model_id))
    assert definition.pre_lm_head_norm_module is not None, config.model_type

    instance = _bare(definition, model)
    resolved = instance._resolved_pre_lm_head_norm_candidates()
    assert resolved, f"{definition.__name__} declared paths did not resolve"
    assert resolved[0][1] == expected
    assert isinstance(resolved[0][0], nn.Module)
