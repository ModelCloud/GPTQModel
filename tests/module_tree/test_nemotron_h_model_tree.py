# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Real-model module-tree coverage for the Nemotron-H family.

Transformers 5.22 renames the native tree root ``backbone`` -> ``model`` via
the ``nemotron_h`` checkpoint conversion map, so the definition must resolve
its decoder layers from ``model.layers`` and cover the native mixer
projections.

The multimodal Nemotron Omni checkpoint nests the same decoder under
``language_model.backbone``. Its definition previously kept a local, mixer-only
copy of the layer inventory, so every routed expert, shared expert, router and
SSM convolution stayed in the source dtype with no error (issue #4355). The
tests below pin the shared declaration and check it against the checkpoint's
real decoder layout.
"""

from pathlib import Path

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.nemotron_h import NemotronHQModel
from gptqmodel.models.module_plan import ModulePlan


def _tiny_nemotron_h_config():
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

    return NemotronHConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_n_groups=1,
        mamba_d_head=16,
        mamba_n_heads=4,
        layers_block_type=["mamba", "attention", "mlp", "moe"],
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def test_nemotron_h_tree_root_matches_native_5_22_model_root():
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

    assert NemotronHQModel.extract_layers_node() == ["model.layers"]

    model = NemotronHForCausalLM(_tiny_nemotron_h_config())
    assert not hasattr(model, "backbone")
    assert len(model.model.layers) == 4


def test_nemotron_h_definition_covers_native_mixer_projections():
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

    model = NemotronHForCausalLM(_tiny_nemotron_h_config())
    plan = ModulePlan.from_tree(NemotronHQModel.module_tree)
    declared = {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "gate_proj", "up_proj", "down_proj"}

    covered = 0
    for layer in model.model.layers:
        for name, module in layer.mixer.named_modules():
            if not name or not isinstance(module, nn.Linear) or name.split(".")[0] not in declared:
                continue
            spec = plan.find(f"mixer.{name}")
            assert spec is not None, (type(layer.mixer).__name__, name)
            assert spec.quantize
            covered += 1

    assert covered >= 4




# --- Nemotron Omni: the embedded language backbone must publish the same
# --- decoder inventory as Nemotron-H (issue #4355).


OMNI_CHECKPOINT_CANDIDATES = (
    "/monster/data/model/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
)


def _omni_checkpoint():
    """Return a local Omni checkpoint directory, or ``None`` when absent."""

    for candidate in OMNI_CHECKPOINT_CANDIDATES:
        path = Path(candidate)
        if (path / "modeling_nemotron_h.py").is_file() and (path / "config.json").is_file():
            return path
    return None


def _omni_language_backbone():
    """Mount the checkpoint's real Nemotron-H decoder at ``language_model``.

    The multimodal wrapper instantiates ``NemotronHForCausalLM(config.llm_config)``
    as ``language_model``; its decoder root is the remote implementation's
    ``backbone``. Building it from the checkpoint's own modeling file keeps the
    tested names and owner layout identical to inference.
    """
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    checkpoint = _omni_checkpoint()
    if checkpoint is None:
        pytest.skip(
            "local Nemotron Omni checkpoint is unavailable; the remote "
            "`language_model.backbone` layout cannot be built offline"
        )
    config = AutoConfig.from_pretrained(
        str(checkpoint), trust_remote_code=True, local_files_only=True
    )
    language_config = config.llm_config
    language_config.num_hidden_layers = 4
    language_config.layers_block_type = ["mamba", "attention", "moe", "moe"]
    language_config.hidden_size = 64
    language_config.intermediate_size = 32
    language_config.moe_intermediate_size = 32
    language_config.moe_shared_expert_intermediate_size = 64
    language_config.n_routed_experts = 4
    language_config.num_experts_per_tok = 2
    language_config.num_attention_heads = 4
    language_config.num_key_value_heads = 2
    language_config.head_dim = 16
    language_config.mamba_num_heads = 2
    language_config.mamba_head_dim = 16
    language_config.ssm_state_size = 16
    language_config.n_groups = 2
    language_config.vocab_size = 128
    language_config.moe_latent_size = None
    language_config._attn_implementation = "eager"

    model_cls = get_class_from_dynamic_module(
        "modeling_nemotron_h.NemotronHForCausalLM", str(checkpoint)
    )
    language_model = model_cls(language_config).eval()

    class _OmniShell(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.language_model = wrapped

    return _OmniShell(language_model)


def test_nemotron_omni_reuses_the_nemotron_h_layer_declaration():
    from gptqmodel.models.definitions.nemotron_h import NEMOTRON_H_LAYER_MODULES
    from gptqmodel.models.definitions.nemotron_omni import NemotronOmniQModel

    tree = NemotronOmniQModel.module_tree
    assert tree[:4] == ["language_model", "model", "layers", "#"]
    # One shared declaration: a local copy is how the routed experts, shared
    # experts, router and conv silently disappeared from the Omni definition.
    assert tree[4] is NEMOTRON_H_LAYER_MODULES
    assert NemotronHQModel.module_tree[3] is NEMOTRON_H_LAYER_MODULES
    assert NemotronOmniQModel.extract_layers_node() == ["language_model.model.layers"]


def test_nemotron_omni_plan_covers_moe_router_conv_and_experts():
    from gptqmodel.models.definitions.nemotron_omni import NemotronOmniQModel

    plan = ModulePlan.from_tree(NemotronOmniQModel.module_tree)
    for path in (
        "mixer.experts.0.up_proj",
        "mixer.experts.0.down_proj",
        "mixer.experts.127.up_proj",
        "mixer.shared_experts.up_proj",
        "mixer.shared_experts.down_proj",
    ):
        spec = plan.find(path)
        assert spec is not None, f"{path} is undeclared"
        assert spec.quantize, (path, spec.flags)
        assert plan.is_expert(path) == (".experts." in path)
    for path in ("mixer.gate", "mixer.conv1d"):
        spec = plan.find(path)
        assert spec is not None, f"{path} is undeclared"
        assert not spec.quantize, (path, spec.flags)
    # `up_proj` output crosses the non-homogeneous `relu2` activation.
    assert "nosrc" in plan.find("mixer.up_proj").flags


def test_nemotron_omni_real_decoder_leaves_are_declared():
    from gptqmodel.models.definitions.nemotron_omni import NemotronOmniQModel

    shell = _omni_language_backbone()
    layers = shell.language_model.backbone.layers
    assert [layer.block_type for layer in layers] == ["mamba", "attention", "moe", "moe"]
    plan = ModulePlan.from_tree(NemotronOmniQModel.module_tree)

    for layer_index, layer in enumerate(layers):
        for weight_name in layer.state_dict():
            owner = weight_name.rsplit(".", 1)[0]
            module = layer.get_submodule(owner)
            if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)):
                # Parameters/buffers such as `mixer.conv1d.weight` are covered
                # through their owning module; scalars have no projection role.
                continue
            assert plan.find(owner) is not None, (layer_index, owner, type(module).__name__)

    moe_layer = layers[2]
    assert len(moe_layer.mixer.experts) == 4
    for expert_index, expert in enumerate(moe_layer.mixer.experts):
        for name, module in expert.named_modules():
            if not name or not isinstance(module, nn.Linear):
                continue
            path = f"mixer.experts.{expert_index}.{name}"
            spec = plan.find(path)
            assert spec is not None and spec.quantize, path
    assert isinstance(moe_layer.mixer.shared_experts.up_proj, nn.Linear)
    spec = plan.find("mixer.shared_experts.up_proj")
    assert spec is not None and spec.quantize
    for path in ("mixer.gate", "mixer.conv1d"):
        assert plan.find(path) is not None


def test_nemotron_omni_simple_layer_modules_expand_the_routed_experts():
    from types import SimpleNamespace

    from gptqmodel.models.definitions.nemotron_omni import NemotronOmniQModel

    language_config = SimpleNamespace(n_routed_experts=3)
    config = SimpleNamespace(
        llm_config=language_config,
        text_config=language_config,
        n_routed_experts=3,
    )

    assert NemotronOmniQModel.get_num_experts(config) == 3
    entries = {
        name
        for block in NemotronOmniQModel.simple_layer_modules(
            model_config=config, quantize_config=SimpleNamespace(dynamic=None)
        )
        for name in block
    }
    assert "mixer.experts.0.up_proj" in entries
    assert "mixer.experts.2.down_proj" in entries
    assert not any("expert_index" in entry for entry in entries)
    # The router and conv stay in the checkpoint dtype, not in the quantized set.
    assert "mixer.gate" not in entries
    assert "mixer.conv1d" not in entries


def test_nemotron_omni_awq_groups_preserve_the_real_decoder_layer_output():
    """The shared declaration must emit output-preserving AWQ groups here too.

    ``NemotronHQModel`` has its own fold-preservation case in
    ``tests/test_awq_scaling_groups.py``. Reusing its declaration makes the Omni
    definition claim the same scaling groups, so the claim is re-checked against
    the checkpoint's real decoder layer (``relu2`` MLP, top-k router, routed and
    shared experts) rather than being inherited on trust.
    """
    from types import SimpleNamespace

    from gptqmodel.models.definitions.nemotron_omni import NemotronOmniQModel
    from gptqmodel.quantization.awq.quantize.scale import apply_scale, is_foldable_source
    from gptqmodel.quantization.awq.utils.module import get_op_name

    shell = _omni_language_backbone()
    shell.config = shell.language_model.config
    qmodel = NemotronOmniQModel.__new__(NemotronOmniQModel)
    nn.Module.__init__(qmodel)
    qmodel.__dict__["model"] = shell
    tree = NemotronOmniQModel._resolve_effective_module_tree(
        shell, SimpleNamespace(method="gptq")
    )
    qmodel.__dict__["effective_module_tree"] = tree
    qmodel.__dict__["module_tree"] = tree
    plan = ModulePlan.from_tree(tree)

    layers = shell.language_model.backbone.layers
    torch.manual_seed(0)
    exercised = set()
    worst = 0.0
    for layer in layers:
        features = {}
        for spec in plan.specs:
            if not spec.quantize:
                continue
            templates = [spec.template]
            if spec.is_expert_placeholder:
                head, _, tail = spec.template.partition("{expert_index}")
                container_path = head.rstrip(".")
                try:
                    container = layer.get_submodule(container_path)
                except AttributeError:
                    continue
                templates = [
                    f"{container_path}.{name}.{tail.lstrip('.')}"
                    for name, _ in container.named_children()
                    if name.isdigit()
                ]
            for template in templates:
                try:
                    module = layer.get_submodule(template)
                except AttributeError:
                    continue
                weight = getattr(module, "weight", None)
                if weight is not None and weight.ndim == 2:
                    features[template] = torch.randn(4, weight.shape[1])

        nodes = qmodel.awq_get_modules_for_scaling(layer, features, {})
        groups = [
            node
            for node in nodes
            if node.get("prev_op") is not None
            and node.get("layers")
            and is_foldable_source(node["prev_op"])
            and (
                not isinstance(node["prev_op"], nn.Linear)
                or node["prev_op"].weight.shape[0] == node["layers"][0].weight.shape[1]
            )
        ]
        assert groups, f"{type(layer.mixer).__name__}: no AWQ scaling group was built"

        hidden = layer.norm.weight.numel()
        x = torch.randn(1, 5, hidden)
        with torch.no_grad():
            baseline = layer(x).clone()
        for node in groups:
            prev_op = node["prev_op"]
            prev_name = get_op_name(layer, prev_op)
            target_names = [get_op_name(layer, target) for target in node["layers"]]
            consumers = node.get("_scale_consumers") or []
            consumer_map = {prev_name: consumers} if consumers else None
            fold_width = (
                prev_op.weight.shape[0]
                if isinstance(prev_op, nn.Linear)
                else prev_op.weight.numel()
            )
            scales = torch.rand(fold_width) + 0.5
            apply_scale(
                layer, [(prev_name, target_names, scales, None)], consumer_map=consumer_map
            )
            with torch.no_grad():
                folded = layer(x)
            restored_scales = 1.0 / scales
            apply_scale(
                layer,
                [(prev_name, target_names, restored_scales, None)],
                consumer_map=consumer_map,
            )
            worst = max(worst, float((folded - baseline).norm() / baseline.norm()))
            exercised.add(prev_name)

    # The decoder-block input norms are the only remaining fold sources; every
    # `relu2` MLP producer is declared `:nosrc` and folds into its norm instead.
    assert "norm" in exercised
    assert not {name for name in exercised if "experts" in name}
    assert worst < 1e-5, f"Nemotron Omni AWQ scaling changed the layer output by {worst:.3e}"
