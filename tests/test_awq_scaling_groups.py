# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ scaling groups must be output-preserving for every registered family.

Each case builds a tiny Hugging Face model, asks the definition for its scaling
groups, folds a random scale with the library's own ``apply_scale``, and checks
that the full-model logits are unchanged. A group that crosses a residual add,
a parallel branch, a non-homogeneous activation, or a second consumer of the
same norm shows up here as a large relative change even though the folded group
still reproduces its own consumer.
"""

import dataclasses
import importlib
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from gptqmodel.looper.awq_processor import AWQProcessor, _AWQScaleGroupError
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.quantization.awq.quantize.scale import apply_scale, is_foldable_source
from gptqmodel.quantization.awq.utils.module import get_op_name


# Model types exercised by the output-preservation invariant. The definition
# module and class are resolved from MODEL_MAP, so adding a model type here
# automatically tracks its registered definition.
FAMILY_MODEL_TYPES = [
    "agnes",
    # Dense decoder families.
    "llama",
    "mistral",
    "gemma",
    "granite",
    "helium",
    "stablelm",
    "seed_oss",
    "smollm3",
    "olmo",
    "olmo2",
    "olmo3",
    "exaone4",
    "opt",
    "phi",
    "gptj",
    "starcoder2",
    "phi3",
    "apertus",
    "codegen",
    "mpt",
    "cohere",
    "cohere2",
    "bloom",
    "gpt_neox",
    "gpt_neo",
    "gpt_bigcode",
    "gpt2",
    "falcon",
    "falcon_h1",
    "falcon_mamba",
    "gemma4_text",
    "hrm_text",
    # Issue #4121 items 7-10: trees that folded across a nonlinearity or a
    # residual. `olmo_hybrid` alternates pre-norm linear-attention layers with
    # post-norm full-attention layers, so one static tree cannot describe both.
    "olmo_hybrid",
    "gemma4",
    "gemma4_unified",
    "nemotron_h",
    # Remote-code families from issue #4093 items 1-2. Their definitions live in
    # the checkpoint's own code, so the tiny config is derived from the local
    # checkpoint config.json instead of `AutoConfig.for_model`.
    "instella",
    "k2_horizon",
    # Issue #4121 item 7: PanGu's top-query layer reads `custom_query` instead
    # of `ln_1`'s output, and its MLP has a GELU between `c_fc` and `c_proj`.
    "gpt_pangu",
    # MoE families.
    "cohere2_moe",
    "gpt_oss",
    "afmoe",
    "olmoe",
    "flint",
    "laguna",
    "solar_open",
    "minimax",
    "minimax_m2",
    "deepseek_v2",
    "lfm2_moe",
    "qwen2_moe",
    "qwen3_moe",
    "mixtral",
]

FAMILIES = {
    model_type: (MODEL_MAP[model_type].__module__, MODEL_MAP[model_type].__name__, model_type)
    for model_type in FAMILY_MODEL_TYPES
}

_BASE_CONFIG = {
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "max_position_embeddings": 64,
    "tie_word_embeddings": False,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "bos_token_id": 0,
}

# Architectures that only exist in a checkpoint's remote code. The values are
# the fixtures used to build the tiny model; the invariant is skipped with a
# precise reason when a fixture is absent.
REMOTE_CODE_MODEL_PATHS = {
    "instella": "/monster/data/model/Instella-3B-Instruct",
    "k2_horizon": "/monster/data/model/K2-Horizon-0.9B",
    "gpt_pangu": "/monster/data/model/pangu_alpha_2_6B",
}

# Remote-code configs whose decoder fields are not named like `_BASE_CONFIG`'s.
# PanGu spells the layer/head counts `num_layers` / `num_heads`.
REMOTE_CODE_MODEL_OVERRIDES = {
    "gpt_pangu": {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 256,
        "num_layers": 2,
        "num_heads": 4,
        "max_position_embeddings": 64,
        "layer_norm_epsilon": 1e-5,
        "activation_function": "gelu",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    },
}

# Composite (text + vision) configs keep the decoder under `text_config`; the
# generic builder would only shrink the unused top-level fields.
COMPOSITE_TEXT_CONFIG_FAMILIES = {"gemma4", "gemma4_unified"}


def _kernel_fallback_implementation(func):
    """Return the implementation a hub-kernel fallback wrapper resolved.

    ``use_kernel_func_from_hub_with_fallback`` closes over the resolved
    ``implementation`` cell, so the decorated module attribute always reports
    ``transformers`` as its own module. The closure is the only place that says
    whether the eager Torch reference or an external fused package is in use.
    """

    cells = dict(zip(getattr(func.__code__, "co_freevars", ()), func.__closure__ or ()))
    implementation = cells.get("implementation")
    if implementation is None:
        return None
    try:
        return implementation.cell_contents
    except ValueError:
        return None


# Modules whose Mamba-2 forward kernels ``transformers`` resolves through the
# hub-kernel fallback decorator, paired with the attribute holding the wrapper.
MAMBA_SSM_KERNEL_BINDINGS = (
    ("transformers.models.mamba.modeling_mamba", ("mamba_selective_scan", "mamba_inner_fn")),
    ("transformers.models.falcon_h1.modeling_falcon_h1", ("mamba2_chunk_scan", "mamba2_split_conv1d_scan_combined")),
    ("transformers.models.nemotron_h.modeling_nemotron_h", ("mamba2_chunk_scan", "mamba2_split_conv1d_scan_combined")),
)


def _mamba_ssm_cpu_unavailable_reason():
    """Report whether ``mamba_ssm`` would displace the eager Torch Mamba-2 path.

    ``transformers`` resolves the Mamba-2 kernels through
    ``use_kernel_func_from_hub_with_fallback``, which prefers an installed
    ``mamba_ssm`` over its own eager reference. Those implementations are
    CUDA-only: they raise ``Expected u.is_cuda()`` or hand a CPU pointer to
    Triton. A host that merely happens to have the package installed therefore
    cannot run this CPU-only fold invariant for the hybrid SSM families.

    The check reads the resolved implementation out of each decorator closure
    rather than testing whether the distribution imports, so a host without
    ``mamba_ssm`` still runs the invariant against the Torch fallback.
    """

    displaced = []
    for module_path, names in MAMBA_SSM_KERNEL_BINDINGS:
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue
        for name in names:
            wrapper = getattr(module, name, None)
            if wrapper is None or not hasattr(wrapper, "__closure__"):
                continue
            implementation = _kernel_fallback_implementation(wrapper)
            resolved = getattr(implementation, "__module__", "") or ""
            if not resolved.startswith("mamba_ssm"):
                continue
            displaced.append(f"{module_path.rsplit('.', 1)[-1]}.{name}")
    if not displaced:
        return None
    return (
        "mamba_ssm displaced the eager Torch Mamba-2 path with CUDA-only kernels "
        f"({', '.join(sorted(displaced))}); this CPU fold invariant needs the Torch fallback"
    )


def _field_names(config):
    if dataclasses.is_dataclass(config):
        return {field.name for field in dataclasses.fields(config)}
    return set(dir(config))


def _tiny_config(model_type):
    if model_type == "agnes":
        from tests.test_agnes_local_port import tiny_config

        return tiny_config()
    if model_type in REMOTE_CODE_MODEL_PATHS:
        return _tiny_remote_config(model_type)
    if model_type in COMPOSITE_TEXT_CONFIG_FAMILIES:
        return _tiny_composite_config(model_type)
    base = AutoConfig.for_model(model_type)
    fields = _field_names(base)
    kwargs = dict(_BASE_CONFIG)
    if model_type in {"flint", "laguna"}:
        # These families keep the first two decoder layers dense, so a third
        # layer is required to exercise the routed + shared expert group.
        kwargs["num_hidden_layers"] = 3
    if model_type == "lfm2_moe" and "layer_types" in fields:
        # LFM2-MoE has no default `layer_types`; without one the decoder layer
        # indexes `None`. Exercise the short-convolution and attention mixers.
        kwargs["layer_types"] = ["conv", "full_attention"]
        kwargs["num_dense_layers"] = 1
    for key in ("head_dim", "rotary_dim"):
        if key in fields:
            kwargs[key] = 8
    for key in (
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "q_lora_rank",
        "kv_lora_rank",
        "index_head_dim",
        "index_topk",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
    ):
        if key in fields:
            kwargs[key] = 8
    for key in ("num_experts", "n_routed_experts", "num_local_experts"):
        if key in fields:
            kwargs[key] = 4
    if "num_experts_per_tok" in fields:
        kwargs["num_experts_per_tok"] = 2
    for key in ("n_shared_experts", "num_shared_experts"):
        if key in fields:
            kwargs[key] = 1
    for key in ("num_dense_layers", "first_k_dense_replace"):
        if key in fields:
            kwargs[key] = 1
    if "attention_types" in fields:
        kwargs["attention_types"] = [[["global"], 2]]

    config_cls = type(base)
    while True:
        try:
            return config_cls(**kwargs)
        except TypeError as exc:
            removed = False
            for key in list(kwargs):
                if key in str(exc) and key not in (
                    "vocab_size",
                    "hidden_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                ):
                    kwargs.pop(key)
                    removed = True
                    break
            if not removed:
                raise


def _tiny_remote_config(model_type):
    """Build a tiny config from a checkpoint whose architecture is remote code."""

    model_path = Path(REMOTE_CODE_MODEL_PATHS[model_type])
    if not model_path.is_dir():
        pytest.skip(f"local {model_type} fixture is unavailable: {model_path}")

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    for key, value in _BASE_CONFIG.items():
        if hasattr(config, key):
            setattr(config, key, value)
    for key, value in REMOTE_CODE_MODEL_OVERRIDES.get(model_type, {}).items():
        setattr(config, key, value)
    # K2's attention width is `num_attention_heads * head_dim`, which is wider
    # than `hidden_size` (2048 vs 1536 on the 0.9B checkpoint). Mirror that so
    # `o_proj` does not accidentally match the norm width and merge with it.
    tiny_head_dim = 16 if model_type == "k2_horizon" else 8
    for key in ("head_dim", "rotary_dim", "rope_head_dim"):
        if hasattr(config, key):
            setattr(config, key, tiny_head_dim)
    # Per-layer lists must describe the tiny layer count, not the checkpoint's.
    if hasattr(config, "num_key_value_heads_per_layer"):
        config.num_key_value_heads_per_layer = [4] * config.num_hidden_layers
    if hasattr(config, "mlp_only_layers"):
        config.mlp_only_layers = list(range(config.num_hidden_layers))
    config._attn_implementation = "eager"
    # The invariant compares logits before/after a fold; a bf16 tiny model
    # would report the dtype's own rounding (~8e-3) instead of fold error.
    for dtype_attr in ("dtype", "torch_dtype"):
        if hasattr(config, dtype_attr):
            setattr(config, dtype_attr, torch.float32)
    return config


def _tiny_composite_config(model_type):
    """Tiny decoder for a composite (text + vision) checkpoint config.

    The decoder lives under ``text_config``. The per-layer token embeddings
    (``vocab_size_per_layer_input`` x ``hidden_size_per_layer_input``) dominate
    the parameter count, so they are shrunk with the decoder; otherwise a
    2-layer decoder would still allocate >100M parameters.
    """

    config = AutoConfig.for_model(model_type)
    text = config.text_config
    for key, value in _BASE_CONFIG.items():
        if hasattr(text, key):
            setattr(text, key, value)
    text.vocab_size = _BASE_CONFIG["vocab_size"]
    for key, value in (
        ("vocab_size_per_layer_input", _BASE_CONFIG["vocab_size"]),
        ("hidden_size_per_layer_input", _BASE_CONFIG["hidden_size"]),
        ("head_dim", 8),
    ):
        setattr(text, key, value)
    text._attn_implementation = "eager"
    # The invariant compares logits before/after a fold; a bf16 tiny model
    # would report the dtype's own rounding instead of fold error.
    for dtype_attr in ("dtype", "torch_dtype"):
        if hasattr(text, dtype_attr):
            setattr(text, dtype_attr, torch.float32)
    return config


def _in_features(module):
    weight = getattr(module, "weight", None)
    if weight is None or weight.ndim != 2:
        return None
    # transformers Conv1D stores [in, out].
    return weight.shape[0] if type(module).__name__ == "Conv1D" else weight.shape[1]


def _vocab_size(config) -> int:
    """Vocabulary size of the decoder, which composite configs keep in `text_config`."""

    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, "vocab_size"):
        return int(text_config.vocab_size)
    return int(config.vocab_size)


def _out_features(module):
    weight = getattr(module, "weight", None)
    if weight is None or weight.ndim != 2:
        return None
    return weight.shape[1] if type(module).__name__ == "Conv1D" else weight.shape[0]


def _features(layer, plan):
    features = {}
    for spec in plan.specs:
        if not spec.quantize:
            continue
        templates = [spec.template]
        if "{expert_index}" in spec.template:
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
            width = _in_features(module)
            if width is not None:
                features[template] = torch.randn(4, width)
    return features


def _tiny_model(model_type):
    qmodel_cls = MODEL_MAP[model_type]
    torch.manual_seed(0)
    if model_type == "agnes":
        hf_model = qmodel_cls.loader.from_config(_tiny_config(model_type)).eval()
    else:
        hf_model = AutoModelForCausalLM.from_config(
            _tiny_config(model_type), trust_remote_code=True
        ).eval()
    try:
        from gptqmodel.models.loader import _convert_model_with_defuser

        _convert_model_with_defuser(qmodel_cls, hf_model, cleanup_original=False)
    except Exception:  # noqa: BLE001 - defuser is optional for dense families
        pass
    qmodel = qmodel_cls.__new__(qmodel_cls)
    nn.Module.__init__(qmodel)
    qmodel.__dict__["model"] = hf_model
    tree = qmodel_cls._resolve_effective_module_tree(hf_model, None)
    qmodel.__dict__["effective_module_tree"] = tree
    qmodel.__dict__["module_tree"] = tree
    layer_paths = qmodel_cls.extract_layers_node()
    if isinstance(layer_paths, str):
        layer_paths = [layer_paths]
    layers = hf_model.get_submodule(layer_paths[0])
    return hf_model, qmodel, list(layers), tree


def _foldable_groups(layer, nodes):
    """Keep the groups ``AWQProcessor._quantize_layer`` keeps and ``apply_scale`` can fold."""

    groups = []
    for node in nodes:
        layers = node.get("layers") or []
        prev_op = node.get("prev_op")
        if not layers or prev_op is None:
            continue
        if not is_foldable_source(prev_op):
            continue
        first_in = _in_features(layers[0])
        if first_in is None:
            continue
        if isinstance(prev_op, nn.Linear) and _out_features(prev_op) != first_in:
            continue
        groups.append(node)
    return groups


def _fold_scale_width(prev_op):
    if isinstance(prev_op, nn.Linear):
        return _out_features(prev_op)
    return prev_op.weight.numel()


def _worst_relative_change(hf_model, ids, layer, groups):
    worst = 0.0
    for node in groups:
        prev_op = node["prev_op"]
        prev_name = get_op_name(layer, prev_op)
        layer_names = [get_op_name(layer, module) for module in node["layers"]]
        scales = torch.rand(_fold_scale_width(prev_op)) + 0.5
        consumers = node.get("_scale_consumers") or []
        consumer_map = {prev_name: consumers} if consumers else None
        with torch.no_grad():
            before = hf_model(ids).logits.clone()
        apply_scale(layer, [(prev_name, layer_names, scales, None)], consumer_map=consumer_map)
        with torch.no_grad():
            after = hf_model(ids).logits
        worst = max(worst, float((after - before).norm() / before.norm()))
        apply_scale(layer, [(prev_name, layer_names, 1.0 / scales, None)], consumer_map=consumer_map)
    return worst


# Families whose decoder layer has no foldable AWQ group and is quantized plainly.
PLAIN_QUANTIZATION_FAMILIES = {
    "afmoe", "agnes", "apertus", "codegen", "deepseek_v2", "deepseek_v3", "deepseek_v4", "gpt2", "minimax"
}

# Hybrid SSM families whose decoder forward resolves Mamba-2 scan kernels through
# `transformers`' hub-kernel fallback. When that decorator selects CUDA-only
# `mamba_ssm` code the CPU-only fold invariant cannot execute at all.
MAMBA_SSM_KERNEL_FAMILIES = {"falcon_h1", "falcon_mamba", "nemotron_h"}

# Issue #4039 families that can be instantiated as a tiny ``AutoModelForCausalLM``.
# ``exaone4_5`` and ``muse_glimmer`` are vision-language wrappers with no
# ``AutoModelForCausalLM`` entry point; their flags are pinned statically by
# ``test_declared_non_source_flags_are_present`` (plus ``exaone4``, which shares
# the Exaone 4 decoder tree, for the numeric invariant),
# and ``deepseek_v3``/``deepseek_v32`` fail to build a tiny config in this
# transformers release (``RuntimeError: shape '[-1, 8, 0]'``).
ISSUE_4039_FAMILIES = {
    "afmoe",
    "bloom",
    "cohere",
    "cohere2",
    "cohere2_moe",
    "exaone4",
    "falcon",
    "falcon_h1",
    "flint",
    "gemma4_text",
    "gpt_bigcode",
    "gpt_neo",
    "gpt_neox",
    "gpt_oss",
    "hrm_text",
    "laguna",
    "minimax",
    "olmoe",
}


def test_every_family_is_a_registered_model_type():
    unknown = set(FAMILY_MODEL_TYPES) - set(MODEL_MAP)
    assert not unknown, f"unregistered model types in FAMILIES: {sorted(unknown)}"
    assert set(FAMILIES) == set(FAMILY_MODEL_TYPES)


def test_issue_4039_families_are_covered():
    """Every family the issue names must be exercised, or documented as excluded."""

    missing = ISSUE_4039_FAMILIES - set(FAMILY_MODEL_TYPES)
    assert not missing, f"issue #4039 families missing from FAMILIES: {sorted(missing)}"


@pytest.mark.parametrize("family", ["cohere", "cohere2"])
def test_parallel_norm_group_cannot_score_only_attention(family):
    _hf_model, qmodel, layers, tree = _tiny_model(family)
    layer = layers[0]
    features = _features(layer, ModulePlan.from_tree(tree))
    nodes = qmodel.awq_get_modules_for_scaling(layer, features, {})
    node = next(node for node in nodes if node["prev_op"] is layer.input_layernorm)
    names = [get_op_name(layer, target) for target in node["layers"]]
    assert "self_attn.q_proj" in names
    assert "mlp.gate_proj" in names
    assert node["module2inspect"] is layer.self_attn
    with pytest.raises(_AWQScaleGroupError, match="mlp.gate_proj"):
        AWQProcessor._validate_scale_group(
            layer, node["prev_op"], node["layers"], node["inp"],
            node["module2inspect"], node.get("_scale_consumers"),
        )


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_scaling_groups_preserve_decoder_layer_output(family):
    if family in MAMBA_SSM_KERNEL_FAMILIES:
        # The invariant runs a real CPU forward; `mamba_ssm` would route it to
        # CUDA-only kernels and fail on the device, not on the fold math.
        reason = _mamba_ssm_cpu_unavailable_reason()
        if reason is not None:
            pytest.skip(reason)
    hf_model, qmodel, layers, tree = _tiny_model(family)
    plan = ModulePlan.from_tree(tree)
    ids = torch.randint(0, _vocab_size(hf_model.config), (1, 8))
    worst = 0.0
    saw_group = False
    for layer in layers:
        features = _features(layer, plan)
        nodes = qmodel.awq_get_modules_for_scaling(layer, features, {})
        groups = _foldable_groups(layer, nodes)
        if not groups:
            continue
        saw_group = True
        worst = max(worst, _worst_relative_change(hf_model, ids, layer, groups))

    if family in PLAIN_QUANTIZATION_FAMILIES:
        assert worst < 1e-5, f"{family}: AWQ scaling changed the layer output by {worst:.3e}"
        return
    assert saw_group, f"{family}: no AWQ scaling groups were built"
    assert worst < 1e-5, f"{family}: AWQ scaling changed the layer output by {worst:.3e}"


MOE_SHARED_FAMILIES = [
    "cohere2_moe",
    "flint",
    "laguna",
    "glm_moe_dsa",
    "hy_v3",
    "solar_open",
    "qwen2_moe",
]


@pytest.mark.parametrize("family", MOE_SHARED_FAMILIES)
def test_routed_and_shared_experts_share_one_group(family):
    """A shared norm may only be divided once, so routed + shared experts fold together."""

    _, qmodel, layers, tree = _tiny_model(family)
    plan = ModulePlan.from_tree(tree)
    checked = False
    for layer in layers:
        nodes = qmodel.awq_get_modules_for_scaling(layer, _features(layer, plan), {})
        for node in nodes:
            names = [get_op_name(layer, module) for module in node.get("layers") or []]
            routed = [name for name in names if ".experts." in name]
            shared = [name for name in names if "shared_expert" in name]
            if routed and shared:
                checked = True
    assert checked, f"{family}: routed and shared experts were not folded into one group"


@pytest.mark.parametrize(
    "family,forbidden",
    [
        ("olmo2", ["post_attention_layernorm"]),
        ("olmo3", ["post_attention_layernorm"]),
        ("exaone4", ["post_attention_layernorm"]),
        ("phi", ["input_layernorm", "self_attn.dense", "mlp.fc1"]),
        ("gptj", ["ln_1", "attn.out_proj", "mlp.fc_in"]),
        ("starcoder2", ["mlp.c_fc"]),
        ("apertus", ["mlp.up_proj"]),
        ("mpt", ["ffn.up_proj"]),
        ("opt350", ["self_attn_layer_norm"]),
        ("gpt_neox", ["mlp.dense_h_to_4h"]),
        ("gpt_neo", ["mlp.c_fc"]),
        ("gpt_bigcode", ["mlp.c_fc"]),
        ("bloom", ["mlp.dense_h_to_4h"]),
        ("gpt_oss", ["mlp.experts.0.up_proj"]),
        ("afmoe", ["self_attn.gate_proj"]),
        ("muse_glimmer", ["self_attn.gate_proj"]),
        ("hrm_text", ["self_attn.gate_proj"]),
        ("gemma4_text", ["self_attn.v_proj"]),
        # Issue #4121 items 7-10: the composite/unified Gemma 4 trees did not
        # copy the text tree's declarations, and PanGu/NemotronH folded across a
        # nonlinearity. `olmo_hybrid` is absent on purpose: its
        # `post_attention_layernorm` is a valid source on linear-attention
        # layers, so only the per-layer fold invariant can pin it.
        # Issue #4355: the routed and shared expert MLPs are the same `relu2`
        # network as `mixer.up_proj`, so neither may be a fold source.
        ("gemma4", ["self_attn.v_proj", "post_feedforward_layernorm", "post_per_layer_input_norm"]),
        ("gemma4_unified", ["self_attn.v_proj", "post_feedforward_layernorm"]),
        ("gpt_pangu", ["mlp.c_fc"]),
        ("nemotron_h", ["mixer.up_proj", "mixer.shared_experts.up_proj", "mixer.experts.0.up_proj"]),
        ("falcon", ["mlp.dense_h_to_4h"]),
        ("falcon_h1", ["mamba.in_proj"]),
    ],
)
def test_declared_non_sources_never_become_prev_ops(family, forbidden):
    if family not in FAMILIES:
        pytest.skip(f"{family} is not exercised as a buildable CausalLM family")
    _, qmodel, layers, tree = _tiny_model(family)
    plan = ModulePlan.from_tree(tree)
    used = set()
    for layer in layers:
        nodes = qmodel.awq_get_modules_for_scaling(layer, _features(layer, plan), {})
        for node in nodes:
            prev_op = node.get("prev_op")
            if prev_op is not None:
                used.add(get_op_name(layer, prev_op))
    assert not (used & set(forbidden)), f"{family}: folded a scale into {used & set(forbidden)}"


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_excluded_projection_is_never_a_fold_source(family):
    """A `:!`/`:?` projection cannot absorb a scale: nothing compensates it (#4651).

    Folding into an excluded module divides its output rows and multiplies the
    consumer's input columns, but the plan keeps the source in the checkpoint
    dtype and no later step multiplies it back. Jamba's
    `['mamba.in_proj', 'mamba.conv1d:!', 'mamba.dt_proj:!']` made the excluded
    `mamba.dt_proj` the source for `mamba.x_proj` (width 64 -> 34), silently
    rescaling the discretization projection. A gain norm is still a valid
    source; a projection is only valid when the plan quantizes it.
    """

    _, qmodel, layers, tree = _tiny_model(family)
    plan = ModulePlan.from_tree(tree)
    checked = 0
    for layer in layers:
        nodes = qmodel.awq_get_modules_for_scaling(layer, _features(layer, plan), {})
        for node in nodes:
            prev_op = node.get("prev_op")
            weight = getattr(prev_op, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                continue
            checked += 1
            name = get_op_name(layer, prev_op)
            spec = plan.find(name)
            assert spec is not None, (family, name)
            assert spec.quantize, (
                f"{family}: scaling group folded a scale into excluded `{name}`; "
                f"layers={[get_op_name(layer, m) for m in node['layers']]}"
            )
    # `falcon_mamba` declares every projection `:nosrc`, so it has no Linear
    # source at all and is quantized plainly.
    assert checked or family in PLAIN_QUANTIZATION_FAMILIES or family == "falcon_mamba", family


# Jamba's Mamba-2 mixer resolves its scan/conv kernels through
# `use_kernel_func_from_hub_with_fallback`, which prefers an installed
# `mamba_ssm` (CUDA-only) over the eager Torch reference. The CPU fold invariant
# needs the reference math, and the decoder is attention-interleaved, so the
# default cache cannot be used either (`get_seq_length` rejects a cache that
# holds only linear-attention layers).
JAMBA_TORCH_KERNELS = (
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "mamba_inner_fn",
    "mamba_selective_scan",
    "mamba_selective_state_update",
)


@pytest.fixture()
def jamba_torch_mixer(monkeypatch):
    """Build a tiny Jamba whose mixer runs the eager Torch Mamba-2 reference."""

    module = importlib.import_module("transformers.models.jamba.modeling_jamba")
    restored = []
    for name in JAMBA_TORCH_KERNELS:
        wrapper = getattr(module, name)
        reference = getattr(wrapper, "__wrapped__", None)
        if reference is None:
            continue
        monkeypatch.setattr(module, name, reference)
        restored.append(name)
    hf_model, qmodel, layers, tree = _tiny_model("jamba")
    hf_model.config.use_cache = False
    return hf_model, qmodel, layers, tree, restored


def test_jamba_never_folds_x_proj_scale_into_excluded_dt_proj(jamba_torch_mixer):
    """Issue #4651: the fold must land on `in_proj`, never on excluded `dt_proj`."""

    hf_model, qmodel, layers, tree, restored = jamba_torch_mixer
    assert restored, "jamba: no hub-kernel wrapper resolved; the Torch reference was not installed"
    plan = ModulePlan.from_tree(tree)
    layer = layers[0]
    nodes = qmodel.awq_get_modules_for_scaling(layer, _features(layer, plan), {})
    groups = _foldable_groups(layer, nodes)
    by_source = {get_op_name(layer, node["prev_op"]): node for node in groups}

    assert "mamba.dt_proj" not in by_source, "jamba: excluded `mamba.dt_proj` is still a fold source"
    # `x_proj` reads `silu(conv1d(in_proj(x)))`: neither `in_proj`, the excluded
    # `dt_proj`, nor the excluded `conv1d` produces its input. It must be
    # quantized without a fold instead of borrowing a width-compatible source.
    for node in nodes:
        assert get_op_name(layer, node["prev_op"]) != "mamba.dt_proj"
        assert "mamba.x_proj" not in [get_op_name(layer, module) for module in node["layers"]]

    ids = torch.randint(0, _vocab_size(hf_model.config), (1, 8))
    with torch.no_grad():
        before = hf_model(ids).logits.clone()
    dt_proj_before = layer.mamba.dt_proj.weight.detach().clone()
    conv1d_before = layer.mamba.conv1d.weight.detach().clone()
    x_proj_before = layer.mamba.x_proj.weight.detach().clone()
    for node in groups:
        source = get_op_name(layer, node["prev_op"])
        targets = [get_op_name(layer, module) for module in node["layers"]]
        width = (
            node["prev_op"].out_features
            if isinstance(node["prev_op"], nn.Linear)
            else node["prev_op"].weight.numel()
        )
        scales = torch.linspace(0.55, 1.45, width)
        apply_scale(layer, [(source, targets, scales, None)])
        try:
            with torch.no_grad():
                after = hf_model(ids).logits
            relative = float((after - before).norm() / before.norm())
            assert relative < 1e-5, f"jamba: folding {source} -> {targets} changed the logits by {relative:.3e}"
            assert torch.equal(layer.mamba.dt_proj.weight, dt_proj_before), (
                f"jamba: the excluded `mamba.dt_proj` weight changed while folding {source}"
            )
            assert torch.equal(layer.mamba.conv1d.weight, conv1d_before)
            assert torch.equal(layer.mamba.x_proj.weight, x_proj_before)
        finally:
            apply_scale(layer, [(source, targets, 1.0 / scales, None)])


def test_jamba_awq_scaling_groups_preserve_decoder_layer_output(jamba_torch_mixer):
    """Full tiny-model invariant for the hybrid SSM decoder (#4651)."""

    hf_model, qmodel, layers, tree, _restored = jamba_torch_mixer
    plan = ModulePlan.from_tree(tree)
    ids = torch.randint(0, _vocab_size(hf_model.config), (1, 8))
    worst = 0.0
    saw_group = False
    for layer in layers:
        nodes = qmodel.awq_get_modules_for_scaling(layer, _features(layer, plan), {})
        groups = _foldable_groups(layer, nodes)
        if not groups:
            continue
        saw_group = True
        worst = max(worst, _worst_relative_change(hf_model, ids, layer, groups))
    assert saw_group, "jamba: no AWQ scaling groups were built"
    assert worst < 1e-5, f"jamba: AWQ scaling changed the model output by {worst:.3e}"


def test_exaone4_5_marks_post_attention_norm_as_postnorm():
    """Exaone 4.5 re-declares the norm without :postnorm; the MLP must not fold into it."""

    from gptqmodel.models.definitions.exaone4 import Exaone4_5QModel

    assert "post_attention_layernorm" in Exaone4_5QModel.get_post_norm_modules()


@pytest.mark.parametrize(
    "module_path,class_name,leaf",
    [
        # Vision-language wrappers with no AutoModelForCausalLM entry point.
        ("gptqmodel.models.definitions.muse_glimmer", "MuseGlimmerQModel", "self_attn.gate_proj"),
        ("gptqmodel.models.definitions.exaone4", "Exaone4_5QModel", "post_attention_layernorm"),
        # Gated attention and non-homogeneous MLP activations.
        ("gptqmodel.models.definitions.hrm_text", "HrmTextQModel", "self_attn.gate_proj"),
        ("gptqmodel.models.definitions.afmoe", "AfMoeQModel", "self_attn.gate_proj"),
        ("gptqmodel.models.definitions.gemma4", "Gemma4TextQModel", "self_attn.v_proj"),
        ("gptqmodel.models.definitions.gpt_oss", "GPTOSSGPTQ", "mlp.experts.{expert_index}.up_proj"),
        ("gptqmodel.models.definitions.gpt_neox", "GPTNeoXQModel", "mlp.dense_h_to_4h"),
        ("gptqmodel.models.definitions.gpt_neo", "GptNeoQModel", "mlp.c_fc"),
        ("gptqmodel.models.definitions.gpt_bigcode", "GptBigCodeQModel", "mlp.c_fc"),
        ("gptqmodel.models.definitions.bloom", "BloomQModel", "mlp.dense_h_to_4h"),
        ("gptqmodel.models.definitions.rw", "RwgQModel", "mlp.dense_h_to_4h"),
        ("gptqmodel.models.definitions.rw", "FalconQModel", "mlp.dense_h_to_4h"),
        ("gptqmodel.models.definitions.falcon_h1", "FalconH1QModel", "mamba.in_proj"),
        ("gptqmodel.models.definitions.minimax_m2", "MiniMaxQModel", "self_attn.output_gate"),
        ("gptqmodel.models.definitions.deepseek_v4", "DeepSeekV4QModel", "self_attn.o_b_proj"),
    ],
)
def test_declared_non_source_flags_are_present(module_path, class_name, leaf):
    """Definitions that cannot build a tiny CausalLM still declare their non-sources."""

    from gptqmodel.models.base import AWQ_NO_SCALE_SOURCE_FLAG, POST_NORM_FLAG

    definition = importlib.import_module(module_path)
    cls = getattr(definition, class_name)
    plan = ModulePlan.from_tree(cls.module_tree)
    spec = plan.find(leaf)
    assert spec is not None, f"{class_name}: `{leaf}` is not declared in the module tree"
    declared = {AWQ_NO_SCALE_SOURCE_FLAG, POST_NORM_FLAG.lstrip(":")} & set(spec.flags)
    assert declared, f"{class_name}: `{leaf}` must declare `:nosrc` or `:postnorm` (flags={spec.flags})"
