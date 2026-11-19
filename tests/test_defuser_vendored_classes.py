# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Vendored Transformers 5.23 classes get exactly Defuser's native treatment.

Defuser keys its runtime patches, expert specs and construction-time block
replacements by ``transformers.models.*`` class path. GPT-QModel builds every
supported model from ``gptqmodel.models.ports.native.*`` copies, so the loader
presents those classes to Defuser under the Transformers path. After
conversion the vendored model must have the same module tree, parameter names
and logits as the Transformers class converted by Defuser, and every module
must be back on its vendored class. The oracle may import
``transformers.models`` (tests only).
"""

import importlib
import io
import json
from pathlib import Path

import defuser
import pytest
import torch

from gptqmodel.models.loader import _convert_model_with_defuser


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ports" / "native_parity"

# Families whose fused blocks Defuser splits by class path at runtime (gate_up
# MLP splits, DBRX/Longcat experts) or by its generic experts pass (MoE blocks
# Defuser otherwise replaces at construction time).
FAMILIES = (
    "phi3", "glm", "glm4", "glm_ocr", "zamba2", "dbrx", "longcat_flash",
    "qwen2_moe", "qwen3_moe", "mixtral", "glm4_moe", "qwen3_next",
    # Defuser `EXPERTS_DEFUSE` specs (instance `linear_loop` forward).
    "afmoe", "glm5_next", "flint", "bailing_hybrid",
)


# Defuser 0.0.31's DBRX runtime forward reshapes tokens by ``ffn_hidden_size``
# instead of ``hidden_size`` and fails on the native class too whenever they
# differ (every released DBRX); only the structure is compared for it.
FORWARD_UNAVAILABLE = {"dbrx"}


class _Definition:
    pass


def _build(family, root):
    spec = json.loads((FIXTURES / f"{family}.json").read_text())
    config_class = getattr(importlib.import_module(f"{root}.{family}.{spec['config_module']}"), spec["config_class"])
    module, name = spec["model_classes"][-1]
    model_class = getattr(importlib.import_module(f"{root}.{family}.{module}"), name)
    config = config_class.from_dict(dict(spec["config"]))
    torch.manual_seed(0)
    return model_class(config).eval()


def _pickle_round_trip(model):
    archive = io.BytesIO()
    torch.save(model, archive)
    archive.seek(0)
    return torch.load(archive, weights_only=False)


@pytest.mark.parametrize("family", [*FAMILIES, "glm4v"])
def test_defused_vendored_model_pickle_round_trip(family):
    model = _build(family, "gptqmodel.models.ports.native")
    _convert_model_with_defuser(_Definition, model, cleanup_original=True)

    restored = _pickle_round_trip(model)
    torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)
    for name, module in model.named_modules():
        other = restored.get_submodule(name)
        if getattr(type(module), "_compact_expert_repr_installed", False):
            assert len(other) == len(module)
            assert other[0] is other._modules["0"]
        if "forward" in vars(module):
            assert other.forward.__func__ is module.forward.__func__

    if family == "glm4v":
        mlp, restored_mlp = model.model.language_model.layers[0].mlp, restored.model.language_model.layers[0].mlp
        hidden = torch.ones(2, 3, model.config.text_config.hidden_size)
        with torch.no_grad():
            torch.testing.assert_close(restored_mlp(hidden), mlp(hidden), rtol=0, atol=0)
        return
    if family in FORWARD_UNAVAILABLE:
        return
    vocab = model.config.get_text_config().vocab_size
    ids = torch.randint(1, min(vocab, 64), (2, 7), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        torch.testing.assert_close(restored(input_ids=ids).logits, model(input_ids=ids).logits, rtol=0, atol=0)


def test_defused_qwen3_omni_pickle_round_trip_preserves_text_runtime():
    from gptqmodel.models.ports.native.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
    from gptqmodel.models.ports.native.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeForConditionalGeneration,
    )

    thinker = _build("qwen3_omni_moe", "gptqmodel.models.ports.native")
    config = Qwen3OmniMoeConfig(thinker_config=thinker.config, enable_audio_output=False)
    model = Qwen3OmniMoeForConditionalGeneration(config).eval()
    assert _convert_model_with_defuser(_Definition, model, cleanup_original=True)
    restored = _pickle_round_trip(model)
    ids = torch.tensor([[2, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(restored(input_ids=ids).logits, model(input_ids=ids).logits, rtol=0, atol=0)
        assert torch.equal(restored.generate(input_ids=ids, max_new_tokens=1), model.generate(input_ids=ids, max_new_tokens=1))


@pytest.mark.parametrize("family", FAMILIES)
def test_vendored_classes_are_defused_like_transformers(family):
    native = _build(family, "transformers.models")
    vendored = _build(family, "gptqmodel.models.ports.native")
    vendored.load_state_dict(native.state_dict(), strict=True)
    vendored_classes = {name: type(module) for name, module in vendored.named_modules()}

    assert defuser.convert_model(native, cleanup_original=True)
    assert _convert_model_with_defuser(_Definition, vendored, cleanup_original=True)

    assert [name for name, _ in vendored.named_modules()] == [name for name, _ in native.named_modules()]
    assert [name for name, _ in vendored.named_parameters()] == [name for name, _ in native.named_parameters()]
    assert not any(name.endswith("gate_up_proj") for name, _ in vendored.named_parameters())
    # Runtime forward overrides (split MLPs, ``linear_loop`` experts) land on the same modules.
    native_forwards = {name for name, module in native.named_modules() if "forward" in vars(module)}
    # Experts Defuser unfused behind ``@use_experts_implementation`` dispatch
    # ``linear_loop`` through Transformers' experts registry on the native
    # model; the vendored model carries that forward on the instance instead.
    registry_experts = {
        name for name, module in native.named_modules()
        if "forward" not in vars(module) and "0" in module._modules
        and hasattr(type(module).forward, "__wrapped__")
        and getattr(getattr(module, "config", None), "_experts_implementation", None) == "linear_loop"
    }
    assert {name for name, module in vendored.named_modules() if "forward" in vars(module)} == (
        native_forwards | registry_experts
    )
    for name, module in vendored.named_modules():
        assert not type(module).__module__.startswith("transformers.models."), name
        if name in vendored_classes:
            # Back on the vendored class, or on a subclass in the vendored module
            # carrying what Defuser added to the class (e.g. expert indexing).
            original = vendored_classes[name]
            assert issubclass(type(module), original), name
            assert (type(module).__module__, type(module).__name__) == (original.__module__, original.__name__), name
    for name, module in native.named_modules():
        if hasattr(type(module), "__getitem__"):
            assert hasattr(type(vendored.get_submodule(name)), "__getitem__"), name

    if family in FORWARD_UNAVAILABLE:
        return
    vocab = native.config.get_text_config().vocab_size
    ids = torch.randint(1, min(vocab, 64), (2, 7), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(vendored(input_ids=ids).logits, native(input_ids=ids).logits)


def test_glm4v_construction_time_replacement_matches_defuser_block():
    """``Glm4vTextMLP`` is only replaced at construction by Defuser; the vendored
    class is split to ``LinearGlm4vTextMLP``'s parameters and forward."""

    from defuser.modeling.glm4v import LinearGlm4vTextMLP

    vendored = _build("glm4v", "gptqmodel.models.ports.native")
    mlp = vendored.model.language_model.layers[0].mlp
    hidden = torch.randn(2, 3, vendored.config.text_config.hidden_size)
    replacement = LinearGlm4vTextMLP(vendored.config.text_config).eval()
    gate, up = mlp.gate_up_proj.weight.detach().chunk(2, dim=0)
    replacement.load_state_dict({"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": mlp.down_proj.weight})

    _convert_model_with_defuser(_Definition, vendored, cleanup_original=True)

    assert sorted(name for name, _ in mlp.named_parameters()) == sorted(name for name, _ in replacement.named_parameters())
    with torch.no_grad():
        assert torch.equal(mlp(hidden), replacement(hidden))


@pytest.mark.parametrize("family", ["mixtral", "qwen3_moe", "qwen2_moe"])
def test_gptqmodel_load_of_a_transformers_saved_moe_checkpoint(family, tmp_path):
    """The public load path (with Defuser) reads a Transformers-saved MoE
    checkpoint into the vendored class: every weight lands, the experts are
    split per expert, and the logits equal the native Transformers model."""

    from gptqmodel import GPTQModel
    from gptqmodel.quantization import QuantizeConfig
    from tests.test_internlm_local_port import _write_tokenizer

    native = _build(family, "transformers.models")
    native.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    reloaded_native = type(native).from_pretrained(tmp_path).eval()

    loaded = GPTQModel.load(
        str(tmp_path), device="cpu", dtype=torch.float32,
        quantize_config=QuantizeConfig(bits=4, group_size=16, offload_to_disk=False),
    )
    model = loaded.model.eval()
    assert type(model).__module__.startswith("gptqmodel.models.ports.native.")
    names = [name for name, _ in model.named_parameters()]
    assert any(".experts.0.gate_proj.weight" in name for name in names)
    assert not any(name.endswith(("experts.gate_up_proj", "experts.down_proj")) for name in names)

    # Every native (fused) expert tensor equals the per-expert split.
    native_state = reloaded_native.state_dict()
    state = model.state_dict()
    for key, tensor in native_state.items():
        if key.endswith("experts.gate_up_proj"):
            prefix = key[: -len("gate_up_proj")]
            for expert, fused in enumerate(tensor):
                gate, up = fused.chunk(2, dim=0)
                assert torch.equal(state[f"{prefix}{expert}.gate_proj.weight"], gate), key
                assert torch.equal(state[f"{prefix}{expert}.up_proj.weight"], up), key
        elif key.endswith("experts.down_proj"):
            prefix = key[: -len("down_proj")]
            for expert, weight in enumerate(tensor):
                assert torch.equal(state[f"{prefix}{expert}.down_proj.weight"], weight), key
        else:
            assert torch.equal(state[key], tensor), key

    # Oracle: the native class given Defuser's own treatment, on the same attention backend.
    defuser.convert_model(reloaded_native, cleanup_original=True)
    reloaded_native.set_attn_implementation(model.config._attn_implementation)
    ids = torch.randint(3, 30, (2, 7), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids).logits, reloaded_native(input_ids=ids).logits)
