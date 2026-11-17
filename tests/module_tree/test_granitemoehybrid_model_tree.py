# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Real-model module-plan coverage for Granite MoE Hybrid.

The decoder interleaves a Mamba branch with full attention and a MoE block. The
Mamba branch owns checkpoint state the generic Llama tree cannot name: the
depthwise ``conv1d`` and the gated output ``norm`` are modules, while
``dt_bias``/``A_log``/``D`` are parameters of the ``mamba`` container itself.
The MoE router is a checkpoint-owned 2-D parameter of ``block_sparse_moe``.
Without explicit ``:!`` entries those owners either fail ``ModulePlan.find()``
or resolve with no policy flag, so nothing records that they must stay dense.

The file builds the installed ``GraniteMoeHybridForCausalLM`` from a tiny
config on ``meta`` and runs the same Defuser conversion as the loader. It also
pins the AWQ policy: the state-space path between ``in_proj`` and ``out_proj``
is not a homogenous linear boundary, and the post-attention norm feeds a
``shared_mlp`` consumer outside the MoE replay boundary.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models._const import SUPPORTS_MODULE_TYPES
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.quantization.awq.utils.module import get_op_name
from gptqmodel.utils.model import find_modules, get_layers_with_prefixes


QC = SimpleNamespace(dynamic=None)
MODEL_TYPE = "granitemoehybrid"
DENSE_MAMBA_OWNERS = ("mamba", "mamba.conv1d", "mamba.norm")
QUANTIZED_MAMBA_OWNERS = ("mamba.in_proj", "mamba.out_proj")
# The state-space path between these projections is not a foldable linear
# boundary. A scale folded from `in_proj` into `out_proj` would leave the
# convolution and scan consumers uncompensated.
NON_SOURCES = ("mamba", "mamba.in_proj", "mamba.conv1d")


def _tiny_config():
    from transformers.models.granitemoehybrid import GraniteMoeHybridConfig

    # `mamba_d_head * mamba_n_heads` must equal `hidden_size * mamba_expand`.
    return GraniteMoeHybridConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=1,
        shared_intermediate_size=16,
        mamba_n_heads=4,
        mamba_n_groups=1,
        mamba_d_state=8,
        mamba_d_head=16,
        mamba_d_conv=4,
        mamba_expand=2,
        # One Mamba layer and one full-attention layer: both tree branches must
        # resolve on the real classes.
        layer_types=["linear_attention", "full_attention"],
    )


def _light_wrapper(definition, model):
    """Build a definition instance without the class-level Mamba patch.

    ``BaseQModel.__init__`` installs ``monkey_patch`` on the Transformers
    class. These structural tests only need the plan and AWQ entry points, and
    leaving the class patched would couple them to the monkeypatch test's
    ordering.
    """

    wrapper = definition.__new__(definition)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    wrapper.__dict__["effective_module_tree"] = definition.module_tree
    wrapper.__dict__["module_tree"] = definition.module_tree
    return wrapper


@pytest.fixture(scope="module")
def case():
    config = _tiny_config()
    definition = MODEL_MAP[MODEL_TYPE]
    with torch.device("meta"):
        model = definition.loader.from_config(config)
    _convert_model_with_defuser(definition, model, cleanup_original=False)

    wrapper = _light_wrapper(definition, model)
    layers, layer_names = get_layers_with_prefixes(model, definition.extract_layers_node())
    assert len(layers) == config.num_hidden_layers
    assert layer_names[0] == f"{definition.extract_layers_node()[0]}.0"

    quantized = {
        name.split(":", 1)[0] for block in definition.simple_layer_modules(model.config, QC) for name in block
    }
    return SimpleNamespace(
        config=config,
        definition=definition,
        model=model,
        wrapper=wrapper,
        layers=layers,
        quantized=quantized,
        plan=ModulePlan.from_tree(definition.module_tree),
    )


def _features(layer):
    """Synthetic per-module inputs; only widths matter for routing decisions."""

    features = {}
    for name, module in find_modules(layer).items():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.dim() == 2:
            features[name] = torch.randn(4, weight.shape[1])
    return features


def test_every_checkpoint_tensor_owner_is_declared(case):
    """Every `state_dict()` owner must resolve with an explicit policy flag."""

    missing = {}
    unflagged = {}
    for layer_index, layer in enumerate(case.layers):
        for weight_name in layer.state_dict():
            owner = weight_name.rpartition(".")[0]
            spec = case.plan.find(owner)
            if spec is None:
                missing.setdefault(owner, set()).add(layer_index)
                continue
            if spec.quantize:
                assert owner in case.quantized, (owner, case.quantized)
                continue
            if not set(spec.flags) & {"!", "?"}:
                unflagged.setdefault(owner, set()).add(layer_index)

    assert not missing, f"checkpoint-owned decoder owners are absent from the module plan: {sorted(missing)}"
    assert not unflagged, f"checkpoint-owned dense owners carry no explicit :!/:? policy: {sorted(unflagged)}"


def test_mamba_branch_owners_are_explicit_and_dense(case):
    plan = case.plan
    for owner in DENSE_MAMBA_OWNERS:
        spec = plan.find(owner)
        assert spec is not None, owner
        assert not spec.quantize, (owner, spec.flags)
        assert set(spec.flags) & {"!", "?"}, (owner, spec.flags)
        assert owner not in case.quantized, owner
        assert plan.find(f"{owner}.undeclared") is None, owner

    for owner in QUANTIZED_MAMBA_OWNERS:
        spec = plan.find(owner)
        assert spec is not None and spec.quantize, (owner, spec.flags)
        assert owner in case.quantized, owner

    mamba_layer = case.layers[0].mamba
    assert isinstance(mamba_layer.conv1d, nn.Conv1d)
    assert isinstance(mamba_layer.norm.weight, nn.Parameter)
    # The container owns `dt_bias`, `A_log` and `D` directly.
    for parameter in ("dt_bias", "A_log", "D"):
        assert isinstance(getattr(mamba_layer, parameter), nn.Parameter), parameter


def test_moe_router_is_checkpoint_owned_and_unquantized(case):
    plan = case.plan
    spec = plan.find("block_sparse_moe.router")
    assert spec is not None and not spec.quantize, spec
    assert set(spec.flags) & {"!", "?"}
    assert "block_sparse_moe.router" not in case.quantized
    assert plan.is_moe("block_sparse_moe.router")

    for layer in case.layers:
        router = layer.block_sparse_moe.router
        assert isinstance(router.weight, nn.Parameter) and router.weight.dim() == 2
        assert not any(
            isinstance(child, tuple(SUPPORTS_MODULE_TYPES)) for child in router.modules() if child is not router
        )


def test_mamba_projections_keep_their_declared_blocks(case):
    blocks = case.definition.simple_layer_modules(case.model.config, QC)
    assert ["mamba.in_proj"] in blocks
    assert ["mamba.out_proj"] in blocks


def test_awq_does_not_fold_across_the_state_space_path(case):
    """`in_proj` is a target only; `out_proj` folds the gated output norm."""

    assert "nosrc" in case.plan.find("mamba.in_proj").flags
    for layer in case.layers:
        nodes = case.wrapper.awq_get_modules_for_scaling(layer, _features(layer), {})
        prev_names = {get_op_name(layer, node["prev_op"]) for node in nodes if node.get("prev_op") is not None}
        assert not prev_names & set(NON_SOURCES), (type(layer).__name__, prev_names)

        mamba_nodes = [
            (get_op_name(layer, node["prev_op"]), [get_op_name(layer, t) for t in node["layers"]])
            for node in nodes
            if get_op_name(layer, node["prev_op"]).startswith("mamba")
            or any(get_op_name(layer, t).startswith("mamba.") for t in node["layers"])
        ]
        if layer.mamba is None:
            assert not mamba_nodes
            continue
        assert mamba_nodes == [
            ("input_layernorm", ["mamba.in_proj"]),
            ("mamba.norm", ["mamba.out_proj"]),
        ], mamba_nodes


def test_awq_post_attention_norm_is_not_a_scale_source(case):
    """The MoE source feeds three consumers but the replay boundary is one block.

    ``post_attention_layernorm`` drives the router, the routed experts and
    ``shared_mlp``. The group's ``module2inspect`` is the MoE block, which does
    not contain ``shared_mlp``, so compensating only the router and the experts
    still changes the shared branch (measured 1.5e-2 relative logit drift on a
    tiny model). The declaration must refuse the fold; every projection still
    quantizes plainly.
    """

    assert "nosrc" in case.plan.find("post_attention_layernorm").flags
    for layer in case.layers:
        nodes = case.wrapper.awq_get_modules_for_scaling(layer, _features(layer), {})
        assert all(node.get("prev_op") is not layer.post_attention_layernorm for node in nodes)
        for name in (
            "block_sparse_moe.experts.0.gate_proj",
            "block_sparse_moe.experts.0.up_proj",
            "block_sparse_moe.experts.0.down_proj",
            "shared_mlp.input_linear",
            "shared_mlp.output_linear",
        ):
            assert name in case.quantized, name


EAGER_MAMBA_WRAPPERS = (
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "mamba2_chunk_scan",
    "mamba2_split_conv1d_scan_combined",
    "mamba2_selective_state_update",
)


@contextmanager
def _eager_torch_mamba_kernels():
    """Pin the Mamba-2 reference implementations for one CPU forward.

    ``transformers`` resolves ``mamba_ssm``/``causal_conv1d`` at import time and
    those kernels are CUDA-only. ``functools.wraps`` keeps the undecorated Torch
    reference on ``__wrapped__``, so the invariant can still execute the real
    model forward without claiming the installed kernels were exercised.
    """

    from transformers.models.granitemoehybrid import modeling_granitemoehybrid as modeling

    saved = {}
    try:
        for name in EAGER_MAMBA_WRAPPERS:
            wrapper = getattr(modeling, name, None)
            reference = getattr(wrapper, "__wrapped__", None)
            if wrapper is None or reference is None:
                continue
            saved[name] = wrapper
            setattr(modeling, name, reference)
        yield
    finally:
        for name, wrapper in saved.items():
            setattr(modeling, name, wrapper)


def _capture_logits_and_features(model, layer, ids):
    features = {}
    handles = []

    def record(name):
        def hook(_module, args, kwargs):
            value = args[0] if args else kwargs.get("hidden_states")
            if isinstance(value, torch.Tensor):
                features[name] = value.detach()

        return hook

    for name, module in layer.named_modules():
        handles.append(module.register_forward_pre_hook(record(name), with_kwargs=True))
    try:
        with torch.inference_mode():
            reference = model(ids).logits.clone()
    finally:
        for handle in handles:
            handle.remove()
    return reference, features


def test_awq_folds_preserve_the_full_model_output():
    """A non-unit fold into every usable source must leave the logits unchanged.

    The installed ``mamba_ssm`` kernels are CUDA-only, so the invariant pins
    transformers' own Torch reference before running the CPU forward.
    """

    from transformers.models.granitemoehybrid import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM

    from gptqmodel.quantization.awq.quantize.scale import apply_scale, is_foldable_source

    config = GraniteMoeHybridConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=1,
        shared_intermediate_size=32,
        mamba_n_heads=8,
        mamba_n_groups=1,
        mamba_d_state=8,
        mamba_d_head=16,
        mamba_d_conv=4,
        mamba_expand=2,
        layer_types=["linear_attention", "full_attention"],
    )
    config._attn_implementation = "eager"
    device = torch.device("cpu")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = GraniteMoeHybridForCausalLM(config).eval().to(device).float()

    definition = MODEL_MAP[MODEL_TYPE]
    _convert_model_with_defuser(definition, model, cleanup_original=False)
    wrapper = _light_wrapper(definition, model)
    ids = torch.randint(2, config.vocab_size, (1, 16))

    checked = 0
    sources = []
    with _eager_torch_mamba_kernels():
        for layer in model.model.layers:
            reference, features = _capture_logits_and_features(model, layer, ids)
            nodes = wrapper.awq_get_modules_for_scaling(layer, features, {})
            snapshot = {name: value.clone() for name, value in layer.state_dict().items()}
            try:
                for node in nodes:
                    prev_op = node.get("prev_op")
                    targets = node.get("layers") or []
                    if prev_op is None or not targets or not is_foldable_source(prev_op):
                        continue
                    first_weight = getattr(targets[0], "weight", None)
                    source_weight = getattr(prev_op, "weight", None)
                    if not isinstance(first_weight, torch.Tensor) or first_weight.ndim != 2:
                        continue
                    # GQA: `v_proj` is narrower than `o_proj`'s input, so the
                    # producer cannot supply the consumer's columns. The
                    # processor drops that group before folding; match it.
                    if (
                        isinstance(source_weight, torch.Tensor)
                        and source_weight.ndim == 2
                        and source_weight.shape[0] != first_weight.shape[1]
                    ):
                        continue
                    source = get_op_name(layer, prev_op)
                    target_names = [get_op_name(layer, target) for target in targets]
                    consumers = node.get("_scale_consumers") or []
                    scales = torch.linspace(0.55, 1.45, node["inp"].shape[-1])
                    apply_scale(
                        layer,
                        [(source, target_names, scales, None)],
                        consumer_map={source: consumers},
                    )
                    try:
                        with torch.inference_mode():
                            actual = model(ids).logits
                        relative_error = (actual - reference).norm() / reference.norm()
                    finally:
                        layer.load_state_dict(snapshot)
                    assert relative_error < 1e-5, (
                        source,
                        target_names,
                        float(relative_error),
                    )
                    sources.append(source)
                    checked += 1
            finally:
                layer.load_state_dict(snapshot)

    assert checked, "no AWQ scaling group survived validation"
    # The state-space path is never a fold source. `out_proj` keeps the exact
    # fold from the gated output norm, which is the only linear producer there.
    assert "mamba.in_proj" not in sources
    assert "mamba.conv1d" not in sources
    assert "mamba" not in sources
    assert "mamba.norm" in sources
