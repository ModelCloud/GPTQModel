# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from gptqmodel.models.base import (
    ROTATION_SUPPORTED_MODEL_TYPES,
    rotation_supported,
    validate_rotation_model_support,
)
from gptqmodel.models.definitions.gemma2 import Gemma2QModel
from gptqmodel.models.definitions.gemma3 import Gemma3QModel
from gptqmodel.models.definitions.gemma4 import Gemma4TextQModel
from gptqmodel.models.definitions.hunyuan_v1_dense import HunYuanDenseV1QModel
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.definitions.olmo2 import Olmo2QModel
from gptqmodel.models.definitions.qwen2 import Qwen2QModel
from gptqmodel.models.definitions.qwen3 import Qwen3QModel
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU_cuda
from gptqmodel.quantization.rotation.rotation import fuse_layer_norms, rotate_model


def _fake_definition(cls, model_type):
    """Build a definition instance without loading a checkpoint."""

    instance = object.__new__(cls)
    instance.model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    return instance


def _hook_online_hadamard(linear):
    """Mirror the loader's online Hadamard on a rotated `down_proj`."""

    def forward(x):
        x = matmul_hadU_cuda(x, linear.had_K, linear.K)
        return F.linear(x, linear.weight, linear.bias)

    linear.forward = forward


def _neutral_rotation_error(hf_cls, config, qmodel_cls):
    """Fuse norms, rotate, and return (relative logit error, argmax agreement)."""

    torch.manual_seed(42)
    model = hf_cls(config).double().eval()
    with torch.no_grad():
        # Gemma-style norms initialise their gain to zero; give every norm a
        # non-trivial gain so the fold is actually exercised.
        for module in model.modules():
            name = module.__class__.__name__
            if name.endswith("RMSNorm") or name.endswith("LayerNorm"):
                module.weight.normal_(0.0, 1.0)

        tokens = torch.tensor([[1, 4, 9, 12]])
        expected = model(tokens).logits.clone()

        plan = ModulePlan.from_tree(qmodel_cls.module_tree)
        fuse_layer_norms(
            model,
            pre_lm_head_norm_module_name=qmodel_cls.pre_lm_head_norm_module,
            layers_node="model.layers",
            lm_head_name="lm_head",
            plan=plan,
        )
        rotate_model(
            model,
            "hadamard",
            torch.device("cpu"),
            lm_head_name="lm_head",
            layers_node="model.layers",
            plan=plan,
            embedding_name="model.embed_tokens",
        )
        for layer in model.model.layers:
            _hook_online_hadamard(layer.mlp.down_proj)
        actual = model(tokens).logits

    relative = (actual - expected).norm().item() / expected.norm().item()
    agreement = (actual.argmax(-1) == expected.argmax(-1)).double().mean().item()
    return relative, agreement


def test_rotation_is_neutral_for_every_opted_in_definition():
    """The test that would have caught Gemma2: neutral rotation must be ~exact."""

    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    cases = (
        (
            "llama",
            LlamaForCausalLM,
            LlamaConfig(
                hidden_size=32,
                intermediate_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_hidden_layers=1,
                vocab_size=64,
                tie_word_embeddings=False,
            ),
            LlamaQModel,
        ),
        (
            "qwen2",
            Qwen2ForCausalLM,
            Qwen2Config(
                hidden_size=32,
                intermediate_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_hidden_layers=1,
                vocab_size=64,
                tie_word_embeddings=False,
            ),
            Qwen2QModel,
        ),
    )

    for name, hf_cls, config, qmodel_cls in cases:
        assert rotation_supported(_fake_definition(qmodel_cls, name)), name
        relative, agreement = _neutral_rotation_error(hf_cls, config, qmodel_cls)
        print(f"{name}: relative logits error = {relative:.3e}, argmax agreement = {agreement:.3f}")
        assert relative < 1e-3, f"{name}: relative logits error = {relative:.3e}"
        assert agreement == 1.0, f"{name}: argmax agreement = {agreement:.3f}"


@pytest.mark.parametrize(
    ("cls", "model_type"),
    (
        (Gemma2QModel, "gemma2"),
        (Gemma3QModel, "gemma3_text"),
        (Gemma4TextQModel, "gemma4_text"),
        (HunYuanDenseV1QModel, "hunyuan_v1_dense"),
        (Olmo2QModel, "olmo2"),
        (Qwen3QModel, "qwen3"),
    ),
)
def test_rotation_raises_for_definition_that_did_not_opt_in(cls, model_type):
    """The gate `quantize()` calls must reject every non-opted-in layout."""

    model_def = _fake_definition(cls, model_type)
    assert rotation_supported(model_def) is False

    with pytest.raises(ValueError) as excinfo:
        validate_rotation_model_support(model_def)

    message = str(excinfo.value)
    assert cls.__name__ in message
    assert f"model_type={model_type!r}" in message
    assert "supported model types: llama, qwen2" in message
    assert "No weights were modified" in message


def test_rotation_opt_in_is_not_inherited_by_a_subclass():
    """A subclass without its own declaration must fail closed."""

    class _UnmarkedLlama(LlamaQModel):
        pass

    model_def = _fake_definition(_UnmarkedLlama, "llama")
    assert rotation_supported(model_def) is False
    with pytest.raises(ValueError, match=r"_UnmarkedLlama"):
        validate_rotation_model_support(model_def)


def test_rotation_opt_in_can_be_declared_by_a_subclass():
    """The positive control for the non-inheriting lookup."""

    class _MarkedLlama(LlamaQModel):
        supports_rotation = frozenset({"llama"})

    assert rotation_supported(_fake_definition(_MarkedLlama, "llama")) is True
    # A class-level opt-in still names its model types; a different model
    # type mapped onto the same class is rejected.
    assert rotation_supported(_fake_definition(_MarkedLlama, "mistral")) is False


def test_supported_model_types_constant_matches_the_declarations():
    """The error message and docs cannot drift from the opt-in declarations."""

    import gptqmodel.models.definitions  # noqa: F401 - load every definition
    from gptqmodel.models.base import BaseQModel

    declared = set()
    stack = [BaseQModel]
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        value = cls.__dict__.get("supports_rotation", frozenset())
        assert isinstance(value, frozenset), f"{cls.__name__} declares {value!r}"
        assert all(isinstance(model_type, str) for model_type in value), cls.__name__
        declared.update(value)

    assert frozenset(declared) == ROTATION_SUPPORTED_MODEL_TYPES
