# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real-model module-plan contract for the ERNIE 4.5 VL MoE text/vision experts.

ERNIE 4.5 VL ships a *fused* checkpoint: layer ``mlp.experts`` holds both the
text and the vision routed experts, and ``mlp.gate`` / ``mlp.gate.weight_1``
carry the two routers.  Defuser's ``ErnieFuseAndSplitTextVisionExperts``
conversion turns that into separate ``mlp.text_moe`` / ``mlp.vision_moe``
sub-trees, each with its own ``experts`` module list.

The definition used to omit ``dynamic_expert_index``, so ``simple_layer_modules``
kept the literal ``{expert_index}`` placeholder.  Because the definition also
sets ``layer_modules_strict = False``, ``create_named_modules`` only recorded a
miss; quantization "succeeded" while both expert families stayed in the source
dtype.  These tests build the real Transformers model, run the real Defuser
conversion, and check the plan consumers end to end.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.models._const import EXPERT_INDEX_PLACEHOLDER
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.ernie4_5_vl_moe import Ernie4_5_VLMoeQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.utils.model import find_modules, get_layers_with_prefixes


QC = SimpleNamespace(dynamic=None)
MODEL_TYPE = "ernie4_5_vl_moe"

# Layer 0 of the stock config is dense; every other layer routes through both
# expert families.
DENSE_LAYER = 0
SPARSE_LAYER = 1


pytestmark = pytest.mark.skipif(
    MODEL_TYPE not in CONFIG_MAPPING,
    reason=f"Installed Transformers has no {MODEL_TYPE!r} implementation",
)


def _quantizable_specs(definition, config) -> list[str]:
    blocks = definition.simple_layer_modules(model_config=config, quantize_config=QC)
    return sorted({name.split(":", 1)[0] for block in blocks for name in block})


def _resolved_modules(layers) -> set[str]:
    resolved = set()
    for layer in layers:
        resolved.update(find_modules(layer))
    return resolved


def _missing_specs(specs, resolved) -> list[str]:
    missing = []
    for spec in specs:
        pattern = re.compile(
            "^" + re.escape(spec).replace(re.escape(EXPERT_INDEX_PLACEHOLDER), r"\d+") + "$"
        )
        if not any(pattern.match(name) for name in resolved):
            missing.append(spec)
    return missing


def _weight_owners(layer) -> set[str]:
    """Modules owning a checkpoint ``state_dict()`` entry, per layer."""

    modules = {name for name, _ in layer.named_modules() if name}
    owners = set()
    for weight_path in layer.state_dict():
        owner = max(
            (
                name
                for name in modules
                if weight_path.startswith(f"{name}.")
                and "." not in weight_path[len(name) + 1 :]
            ),
            key=len,
            default="",
        )
        assert owner, f"state entry {weight_path!r} is not owned by a module"
        owners.add(owner)
    return owners


def _normalize_owner(owner: str) -> str:
    return re.sub(r"(?<=\.)\d+(?=\.|$)", "#", owner)


@pytest.fixture(scope="module")
def ernie_case():
    config = transformers.AutoConfig.for_model(MODEL_TYPE)
    definition = MODEL_MAP[MODEL_TYPE]
    with torch.device("meta"):
        model = definition.loader.from_config(config)

    _convert_model_with_defuser(definition, model, cleanup_original=False)
    layers, layer_names = get_layers_with_prefixes(model, definition.extract_layers_node())
    assert len(layers) == config.text_config.num_hidden_layers

    return SimpleNamespace(
        config=config,
        definition=definition,
        model=model,
        layers=layers,
        layer_names=layer_names,
        specs=_quantizable_specs(definition, config),
        resolved=_resolved_modules(layers),
        plan=ModulePlan.from_tree(definition.module_tree),
    )


def test_definition_declares_the_text_config_expert_count():
    config = transformers.AutoConfig.for_model(MODEL_TYPE)
    definition = Ernie4_5_VLMoeQModel

    # The field lives on the nested text config; the multimodal wrapper does not
    # expose it directly. Both routed-expert families share that cardinality.
    assert not hasattr(config, "moe_num_experts")
    assert config.text_config.moe_num_experts > 0

    assert definition.dynamic_expert_index == "moe_num_experts"
    assert definition.get_num_experts(config) == config.text_config.moe_num_experts
    assert (
        definition.get_num_experts_for_module(
            config, "mlp.text_moe.experts.{expert_index}.gate_proj"
        )
        == config.text_config.moe_num_experts
    )
    assert (
        definition.get_num_experts_for_module(
            config, "mlp.vision_moe.experts.{expert_index}.down_proj"
        )
        == config.text_config.moe_num_experts
    )


def test_defuser_layout_splits_the_fused_checkpoint_experts(ernie_case):
    """The converted runtime layout must expose both expert families."""

    sparse = ernie_case.layers[SPARSE_LAYER]
    num_experts = ernie_case.config.text_config.moe_num_experts

    for family in ("text_moe", "vision_moe"):
        experts = sparse.get_submodule(f"mlp.{family}.experts")
        container = getattr(experts, "num_experts", None)
        assert container in (None, num_experts)
        for index in range(num_experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                module = sparse.get_submodule(
                    f"mlp.{family}.experts.{index}.{projection}"
                )
                assert isinstance(module, torch.nn.Linear), (
                    f"mlp.{family}.experts.{index}.{projection}"
                )

        # Routers and their score-correction statistics are checkpoint-owned
        # but never quantized.
        assert sparse.get_submodule(f"mlp.{family}.gate") is not None
        assert (
            sparse.get_submodule(f"mlp.{family}.gate.moe_statics")
            is not None
        )


def test_layer_modules_expand_every_expert(ernie_case):
    num_experts = ernie_case.config.text_config.moe_num_experts
    specs = ernie_case.specs

    assert not [spec for spec in specs if EXPERT_INDEX_PLACEHOLDER in spec], (
        "simple_layer_modules still emits literal placeholders; quantizer name "
        "matching would silently miss every routed expert"
    )

    for family in ("text_moe", "vision_moe"):
        concrete = {
            spec
            for spec in specs
            if f".{family}.experts." in spec
        }
        assert len(concrete) == 3 * num_experts, family
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for index in range(num_experts):
                assert f"mlp.{family}.experts.{index}.{projection}" in concrete


def test_every_quantizable_spec_resolves_on_the_real_layers(ernie_case):
    missing = _missing_specs(ernie_case.specs, ernie_case.resolved)
    assert not missing, (
        f"{len(missing)} layer-module spec(s) match no decoder layer and would be "
        f"skipped by create_named_modules: {missing[:10]}"
    )


def test_every_checkpoint_weight_owner_is_declared_with_the_right_tag(ernie_case):
    quantized_patterns = [
        re.compile(
            "^"
            + re.escape(spec).replace(re.escape(EXPERT_INDEX_PLACEHOLDER), r"\d+")
            + "$"
        )
        for spec in ernie_case.specs
    ]

    for layer, layer_name in zip(ernie_case.layers, ernie_case.layer_names):
        owners = _weight_owners(layer)
        normalized = {_normalize_owner(owner) for owner in owners}

        for owner in owners:
            spec = ernie_case.plan.find(owner)
            assert spec is not None, (
                f"{layer_name}: checkpoint weights owned by undeclared module {owner!r}"
            )
            is_quantized = any(pattern.match(owner) for pattern in quantized_patterns)
            if not is_quantized:
                assert not spec.quantize, (
                    f"{layer_name}: non-quantized weight owner {owner!r} must carry "
                    "an explicit `:!`/`:?` tag"
                )

        # Dense and sparse layers own different weights; both must be fully
        # declared, including non-quantized norms.
        assert normalized


def test_routers_and_score_correction_bias_are_not_quantized(ernie_case):
    plan = ernie_case.plan
    for family in ("text_moe", "vision_moe"):
        gate = plan.find(f"mlp.{family}.gate")
        statics = plan.find(f"mlp.{family}.gate.moe_statics")
        assert gate is not None and not gate.quantize
        assert statics is not None and not statics.quantize
        for flag in ("!",):
            assert flag in gate.flags

    for index in range(ernie_case.config.text_config.moe_num_experts):
        for family in ("text_moe", "vision_moe"):
            assert plan.find(f"mlp.{family}.experts.{index}.gate_proj").quantize
            assert plan.find(f"mlp.{family}.experts.{index}.up_proj").quantize
            assert plan.find(f"mlp.{family}.experts.{index}.down_proj").quantize


def test_module_plan_exposes_both_expert_containers(ernie_case):
    plan = ernie_case.plan
    assert tuple(spec.template for spec in plan.moe_roots()) == ("mlp",)
    assert set(plan.expert_containers()) == {
        "mlp.text_moe.experts",
        "mlp.vision_moe.experts",
    }
    assert set(plan.shared_expert_containers()) == {"mlp.shared_experts"}

    for family in ("text_moe", "vision_moe"):
        assert plan.is_expert(f"mlp.{family}.experts.3.gate_proj")
        assert plan.expert_root(f"mlp.{family}.experts.3.gate_proj") == (
            f"mlp.{family}.experts.3"
        )
        assert plan.same_moe_family(
            f"mlp.{family}.experts.3.gate_proj",
            f"mlp.{family}.experts.3.down_proj",
        )
        assert not plan.same_moe_family(
            f"mlp.{family}.experts.3.gate_proj",
            f"mlp.{family}.experts.4.down_proj",
        )
        # Families are distinct: a text expert must not look like a vision one.
        assert not plan.same_moe_family(
            f"mlp.{family}.experts.3.gate_proj",
            "mlp.shared_experts.gate_proj",
        )


def _build_tiny_fixture(model_dir: Path, num_experts: int = 2) -> None:
    """Write a tiny but structurally faithful ERNIE 4.5 VL MoE checkpoint.

    The fixture is built locally so the post-quantization test needs no network
    access.  It keeps the fused ``mlp.experts`` checkpoint layout, one dense and
    one sparse layer, and both routers so Defuser performs the real split.
    """

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import PreTrainedTokenizerFast
    from transformers.models.ernie4_5_vl_moe.configuration_ernie4_5_vl_moe import (
        Ernie4_5_VLMoeConfig,
    )
    from transformers.models.ernie4_5_vl_moe.image_processing_ernie4_5_vl_moe import (
        Ernie4_5_VLMoeImageProcessor,
    )
    from transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe import (
        Ernie4_5_VLMoeForConditionalGeneration,
    )
    from transformers.models.ernie4_5_vl_moe.processing_ernie4_5_vl_moe import (
        Ernie4_5_VLMoeProcessor,
    )
    from transformers.models.ernie4_5_vl_moe.video_processing_ernie4_5_vl_moe import (
        Ernie4_5_VLMoeVideoProcessor,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    corpus = ["hello world this is a tiny calibration sample for expert coverage"] * 4
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        corpus,
        trainer=WordLevelTrainer(
            special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        ),
    )
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    fast_tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<mask:5>",
                "<mask:6>",
                "<mask:7>",
                "<|<IMAGE_PLACEHOLDER|>",
                "<|<VIDEO_PLACEHOLDER|>",
            ]
        }
    )
    fast_tokenizer.save_pretrained(model_dir)
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text())
    tokenizer_config.update(
        {
            "sep_token": "<mask:5>",
            "cls_token": "<mask:6>",
            "image_token": "<|<IMAGE_PLACEHOLDER|>",
            "image_start_token": "<mask:6>",
            "image_end_token": "<mask:7>",
            "video_token": "<|<VIDEO_PLACEHOLDER|>",
            "video_start_token": "<mask:6>",
            "video_end_token": "<mask:7>",
        }
    )
    (model_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2))
    fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)

    config = Ernie4_5_VLMoeConfig(
        text_config={
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": len(fast_tokenizer),
            "max_position_embeddings": 128,
            "moe_intermediate_size": [32, 16],
            "moe_k": 1,
            "moe_num_experts": num_experts,
            "moe_num_shared_experts": 1,
            "mlp_layer_types": ["dense", "sparse"],
            "rope_parameters": {
                "rope_type": "default", "rope_theta": 10000.0, "mrope_section": [6, 6, 4]
            },
        },
        vision_config={
            "depth": 1, "hidden_size": 128, "intermediate_size": 256, "num_heads": 4, "patch_size": 8
        },
    )
    config._attn_implementation = "eager"
    Ernie4_5_VLMoeForConditionalGeneration(config).save_pretrained(model_dir)

    processor = Ernie4_5_VLMoeProcessor(
        image_processor=Ernie4_5_VLMoeImageProcessor(
            patch_size=8, merge_size=2, temporal_patch_size=1, resample=3
        ),
        tokenizer=fast_tokenizer,
        video_processor=Ernie4_5_VLMoeVideoProcessor(
            patch_size=8,
            merge_size=2,
            temporal_patch_size=2,
            resample=3,
            draw_on_frames=False,
        ),
    )
    processor.save_pretrained(model_dir)
    (model_dir / "chat_template.jinja").write_text(
        "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    )


def test_quantized_model_covers_both_expert_families(tmp_path):
    """End-to-end guard: every text/vision expert is a quantized linear after ``quantize``."""

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.quantization import QuantizeConfig

    model_dir = tmp_path / "ernie_vl_moe"
    num_experts = 2
    _build_tiny_fixture(model_dir, num_experts=num_experts)

    module = GPTQModel.load(
        str(model_dir),
        quantize_config=QuantizeConfig(
            bits=4, group_size=32, desc_act=False, device="cpu"
        ),
        backend=BACKEND.TORCH,
    )
    module.quantize(
        [[{"role": "user", "content": "tiny expert coverage calibration sample"}]] * 4,
        batch_size=1,
        backend=BACKEND.TORCH,
        calibration_data_min_length=1,
    )

    modules = dict(module.model.named_modules())
    for family in ("text_moe", "vision_moe"):
        for index in range(num_experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = (
                    f"model.language_model.layers.1.mlp.{family}"
                    f".experts.{index}.{projection}"
                )
                assert isinstance(modules[name], TorchLinear), name


def test_looper_records_no_miss_for_expanded_expert_specs(ernie_case):
    """The concrete names handed to ``create_named_modules`` all hit."""

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = SimpleNamespace(module_plan=lambda: ernie_case.plan)

    sparse = ernie_case.layers[SPARSE_LAYER]
    full = set(find_modules(sparse))
    blocks = ernie_case.definition.simple_layer_modules(
        model_config=ernie_case.config, quantize_config=QC
    )
    names = [name.split(":", 1)[0] for block in blocks for name in block]

    # Dense-layer-only specs legitimately miss a sparse layer (the coverage
    # tracker reports those as "partial"); routed experts must never miss.
    expert_names = [
        name
        for name in names
        if ".text_moe.experts." in name or ".vision_moe.experts." in name
    ]
    assert expert_names
    assert all(EXPERT_INDEX_PLACEHOLDER not in name for name in expert_names)

    missed = [name for name in expert_names if name not in full]
    assert not missed, f"expert specs recorded as misses: {missed[:10]}"

    for name in expert_names:
        assert looper._extract_moe_group_key(name) is not None
