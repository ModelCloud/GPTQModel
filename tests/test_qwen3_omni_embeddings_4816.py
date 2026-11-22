# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni must expose the thinker embedding endpoints (issue #4816).

transformers 5.22 leaves ``get_input_embeddings`` unimplemented on
``Qwen3OmniMoeForConditionalGeneration`` (vendored unchanged in
``gptqmodel.models.ports.native.qwen3_omni_moe``, which the definition loads) and returns ``None`` from
``get_output_embeddings`` because the getter delegates to it. Both endpoints
live on the nested ``thinker``, so ``quantize()`` crashed during input capture
before layer 0. These tests use a meta-device tiny config: no checkpoint and no
30B download.
"""

import pytest
import torch
from torch import nn
from transformers import PreTrainedModel

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions.qwen3_omni_moe import Qwen3OmniMoeGPTQ
from gptqmodel.models.ports.native.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
from gptqmodel.models.ports.native.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
)
from gptqmodel.utils.model import thinker_text_model


def _tiny_qwen3_omni():
    config = Qwen3OmniMoeConfig(
        enable_audio_output=False,
        thinker_config={
            "text_config": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 64,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 16,
                "tie_word_embeddings": False,
            },
            "vision_config": {
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 32,
                "num_position_embeddings": 16,
            },
            "audio_config": {
                "d_model": 16,
                "encoder_layers": 1,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 32,
                "num_mel_bins": 8,
                "output_dim": 32,
            },
        },
    )
    torch.manual_seed(4816)
    with torch.device("meta"):
        return Qwen3OmniMoeForConditionalGeneration(config).eval()


def _wrapper(definition, model):
    instance = definition.__new__(definition)
    nn.Module.__init__(instance)
    instance.__dict__["model"] = model
    return instance


def test_hf_wrapper_really_raises_without_the_definition_override():
    """Guard the premise: the plain transformers getters do not work here."""

    model = _tiny_qwen3_omni()
    if Qwen3OmniMoeForConditionalGeneration.get_input_embeddings is not PreTrainedModel.get_input_embeddings:
        pytest.skip("transformers now implements the Qwen3-Omni embedding getters")
    with pytest.raises(NotImplementedError):
        model.get_input_embeddings()
    # transformers returns None instead of raising for the output head, which
    # silently disabled lm_head quantization for this family.
    assert model.get_output_embeddings() is None


def test_qwen3_omni_resolves_the_thinker_embedding_endpoints():
    model = _tiny_qwen3_omni()
    instance = _wrapper(Qwen3OmniMoeGPTQ, model)
    text_model = thinker_text_model(model)

    assert instance.get_input_embeddings() is text_model.embed_tokens
    assert instance.get_input_embeddings_name() == "thinker.model.language_model.embed_tokens"
    assert instance.get_output_embeddings() is model.thinker.lm_head
    assert instance.get_output_embeddings_name() == "thinker.lm_head"
    assert Qwen3OmniMoeGPTQ.lm_head == "thinker.lm_head"


def test_qwen3_omni_loaded_instance_resolves_hf_getters_without_patching_the_class():
    model = _tiny_qwen3_omni()
    wrapper_cls = Qwen3OmniMoeForConditionalGeneration
    class_state = dict(vars(wrapper_cls))
    Qwen3OmniMoeGPTQ.after_defuser_conversion(model)
    # The quantized loader discovers the endpoints on the loaded model itself.
    assert model.get_input_embeddings() is thinker_text_model(model).embed_tokens
    assert model.get_output_embeddings() is model.thinker.lm_head
    # Only this instance changed: the vendored class is untouched.
    assert dict(vars(wrapper_cls)) == class_state
    assert wrapper_cls.get_input_embeddings is PreTrainedModel.get_input_embeddings
    with pytest.raises(NotImplementedError):
        _tiny_qwen3_omni().get_input_embeddings()
    # Idempotent.
    Qwen3OmniMoeGPTQ.after_defuser_conversion(model)
    assert model.get_output_embeddings() is model.thinker.lm_head


def test_composite_wrapper_without_overrides_resolves_the_input_embedding():
    """The generic base getter must not raise for a declared decoder tree."""

    class BareComposite(BaseQModel):
        module_tree = Qwen3OmniMoeGPTQ.module_tree

    model = _tiny_qwen3_omni()
    instance = _wrapper(BareComposite, model)
    assert BareComposite.get_input_embeddings is BaseQModel.get_input_embeddings

    assert instance.get_input_embeddings() is thinker_text_model(model).embed_tokens
    assert instance.get_input_embeddings_name() == "thinker.model.language_model.embed_tokens"


def test_unresolvable_composite_wrapper_still_raises():
    """An endpoint the tree cannot resolve must stay a loud failure."""

    class NoEmbedding(BaseQModel):
        module_tree = ["model", "layers", "#", {"input_layernorm": ("input_layernorm:!",)}]

    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])

    def _raise():
        raise NotImplementedError("`get_input_embeddings` not auto-handled for Anonymous")

    model.get_input_embeddings = _raise
    instance = _wrapper(NoEmbedding, model)
    with pytest.raises(NotImplementedError):
        instance.get_input_embeddings()


def _synthetic_wrapper(layer_paths):
    """A wrapper that raises from `get_input_embeddings`, with the declared tree."""

    wrapper = nn.Module()

    def _raise():
        raise NotImplementedError("`get_input_embeddings` not auto-handled for SyntheticWrapper")

    wrapper.get_input_embeddings = _raise
    embeddings = {}
    for path in layer_paths:
        parts = path.split(".")
        node = wrapper
        for part in parts[:-1]:
            child = getattr(node, part, None)
            if child is None:
                child = nn.Module()
                setattr(node, part, child)
            node = child
        if not hasattr(node, parts[-1]):
            setattr(node, parts[-1], nn.ModuleList())
        embedding = embeddings.setdefault(path, nn.Embedding(8, 4))
        if not hasattr(node, "embed_tokens"):
            node.add_module("embed_tokens", embedding)
    return wrapper, embeddings


@pytest.mark.parametrize("model_type", sorted(MODEL_MAP))
def test_every_definition_resolves_a_declared_input_embedding(model_type):
    """The base getter must resolve for every definition that relies on it."""

    definition = MODEL_MAP[model_type]
    if definition.get_input_embeddings is not BaseQModel.get_input_embeddings:
        pytest.skip("definition overrides get_input_embeddings")

    layer_paths = definition.extract_layers_node()
    if isinstance(layer_paths, str):
        layer_paths = [layer_paths]
    if not layer_paths:
        pytest.skip("definition declares no `layers` node")

    model, embeddings = _synthetic_wrapper(layer_paths)
    instance = _wrapper(definition, model)
    resolved = instance.get_input_embeddings()
    assert resolved is not None
    assert any(resolved is embedding for embedding in embeddings.values())
