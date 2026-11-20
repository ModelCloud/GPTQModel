# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.models import auto
from gptqmodel.models.definitions.nemotron_h_puzzle import NemotronHPuzzleQModel


def test_nemotron_h_puzzle_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="nemotron_h_puzzle")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/nemotron_h_puzzle") is NemotronHPuzzleQModel


def test_nemotron_h_puzzle_module_tree_covers_hybrid_mixers_and_moe():
    layer_modules = NemotronHPuzzleQModel.simple_layer_modules(
        model_config=SimpleNamespace(n_routed_experts=3),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert NemotronHPuzzleQModel.require_trust_remote_code is True
    assert NemotronHPuzzleQModel.layer_modules_strict is False
    assert NemotronHPuzzleQModel.pre_lm_head_norm_module == "model.norm_f"
    assert NemotronHPuzzleQModel.extract_layers_node() == ["model.layers"]

    assert "mixer.q_proj" in flat_modules
    assert "mixer.k_proj" in flat_modules
    assert "mixer.v_proj" in flat_modules
    assert "mixer.o_proj" in flat_modules
    assert "mixer.in_proj" in flat_modules
    assert "mixer.out_proj" in flat_modules
    assert "mixer.fc1_latent_proj" in flat_modules
    assert "mixer.fc2_latent_proj" in flat_modules
    assert "mixer.experts.0.up_proj" in flat_modules
    assert "mixer.experts.1.down_proj" in flat_modules
    assert "mixer.experts.2.up_proj" in flat_modules
    assert "mixer.shared_experts.up_proj" in flat_modules
    assert "mixer.shared_experts.down_proj" in flat_modules
    assert "mixer.gate" not in flat_modules


def test_nemotron_h_puzzle_module_tree_matches_checkpoint_layout():
    assert NemotronHPuzzleQModel.module_tree[:3] == ["model", "layers", "#"]
    mixer_tree = NemotronHPuzzleQModel.module_tree[-1]["mixer:moe"]

    assert "experts" in mixer_tree
    assert "#" in mixer_tree["experts"]
    assert "shared_experts" in mixer_tree


def test_nemotron_h_puzzle_moe_projection_order_matches_forward():
    layer_modules = NemotronHPuzzleQModel.simple_layer_modules(
        model_config=SimpleNamespace(n_routed_experts=2),
        quantize_config=SimpleNamespace(dynamic=None),
    )

    fc1_index = next(i for i, block in enumerate(layer_modules) if "mixer.fc1_latent_proj" in block)
    expert_index = next(i for i, block in enumerate(layer_modules) if "mixer.experts.0.up_proj" in block)
    fc2_index = next(i for i, block in enumerate(layer_modules) if "mixer.fc2_latent_proj" in block)
    shared_index = next(i for i, block in enumerate(layer_modules) if "mixer.shared_experts.up_proj" in block)

    assert fc1_index < expert_index < fc2_index < shared_index


def test_nemotron_h_puzzle_replay_drops_unsupported_use_cache():
    model_def = object.__new__(NemotronHPuzzleQModel)
    attention_mask = torch.ones((1, 4), dtype=torch.long)
    cache_position = torch.arange(4)
    replay_kwargs = model_def.prepare_layer_replay_kwargs(
        layer=SimpleNamespace(),
        layer_input=[torch.zeros((1, 4, 8))],
        additional_inputs={
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "output_attentions": False,
            "use_cache": False,
        },
        target_device=torch.device("cpu"),
    )

    assert "use_cache" not in replay_kwargs
    assert replay_kwargs["attention_mask"] is attention_mask
    assert replay_kwargs["cache_position"] is cache_position
    assert replay_kwargs["output_attentions"] is False


def test_nemotron_h_puzzle_replay_builds_attention_causal_mask():
    model_def = object.__new__(NemotronHPuzzleQModel)
    mixer_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    layer = SimpleNamespace(block_type="attention", mixer=SimpleNamespace(config=mixer_config))
    hidden_states = torch.zeros((2, 4, 8), dtype=torch.bfloat16)
    padding_mask = torch.tensor(
        [
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    replay_kwargs = model_def.prepare_layer_replay_kwargs(
        layer=layer,
        layer_input=[hidden_states],
        additional_inputs={
            "attention_mask": padding_mask,
            "cache_position": torch.arange(4),
            "use_cache": False,
        },
        target_device=torch.device("cpu"),
    )

    causal_mask = replay_kwargs["attention_mask"]
    min_dtype = torch.finfo(torch.bfloat16).min

    assert mixer_config._attn_implementation == "eager"
    assert causal_mask.shape == (2, 1, 4, 4)
    assert causal_mask.dtype == torch.bfloat16
    assert torch.all(causal_mask[0, 0, :, 0] == min_dtype)
    assert causal_mask[1, 0, 3, 3] == 0
    assert causal_mask[1, 0, 0, 1] == min_dtype


def test_nemotron_h_puzzle_replay_keeps_mamba_padding_mask():
    model_def = object.__new__(NemotronHPuzzleQModel)
    padding_mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)
    layer = SimpleNamespace(block_type="mamba", mixer=SimpleNamespace())

    replay_kwargs = model_def.prepare_layer_replay_kwargs(
        layer=layer,
        layer_input=[torch.zeros((1, 4, 8), dtype=torch.bfloat16)],
        additional_inputs={"attention_mask": padding_mask, "use_cache": False},
        target_device=torch.device("cpu"),
    )

    assert replay_kwargs["attention_mask"] is padding_mask


class _WeightlessLinear(nn.Module):
    """QuantLinear-shaped test double which intentionally has no dense weight."""

    def forward(self, hidden_states):
        return hidden_states


class _WeightlessExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = _WeightlessLinear()
        self.down_proj = _WeightlessLinear()

    def forward(self, hidden_states):
        return self.down_proj(self.up_proj(hidden_states))


class NemotronHMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([_WeightlessExpert(), _WeightlessExpert()])


def test_nemotron_h_puzzle_empty_expert_does_not_require_dense_weight():
    moe = NemotronHMoE()
    model_def = SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[SimpleNamespace(mixer=moe)],
            )
        )
    )
    NemotronHPuzzleQModel.monkey_patch(model_def)

    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    # Only expert 0 is routed; expert 1 exercises the empty-expert branch.
    topk_indices = torch.zeros((2, 1), dtype=torch.long)
    topk_weights = torch.ones((2, 1), dtype=torch.bfloat16)

    output = moe.moe(hidden_states, topk_indices, topk_weights)

    assert torch.equal(output, hidden_states)
    assert not hasattr(moe.experts[1].down_proj, "weight")


# --- Issue #5739: the expert MLP is `down_proj(relu2(up_proj(x)))` ------------
#
# `relu2` is not positively homogeneous of degree 1, so an expert `up_proj` is
# not a valid AWQ scale source for its own `down_proj`. The Puzzle definition
# overrides `module_tree` and used to drop the parent's `:nosrc` flag, which let
# `_filter_awq_scaling_sources` keep the `up_proj -> down_proj` group. Folding
# `s` there computes `down(relu2(u) / s)` instead of `down(relu2(u))`.


class _ReluSquared(nn.Module):
    def forward(self, hidden_states):
        return torch.relu(hidden_states) ** 2


class _PuzzleRelu2Expert(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = _ReluSquared()

    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(self.up_proj(hidden_states)))


class _PuzzleMoE(nn.Module):
    """Routed and shared experts reading one post-attention norm."""

    def __init__(self, hidden_size, intermediate_size, shared_intermediate_size, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([_PuzzleRelu2Expert(hidden_size, intermediate_size) for _ in range(num_experts)])
        self.shared_experts = _PuzzleRelu2Expert(hidden_size, shared_intermediate_size)

    def forward(self, hidden_states):
        output = self.shared_experts(hidden_states)
        for expert in self.experts:
            output = output + expert(hidden_states)
        return output


class _PuzzleLayer(nn.Module):
    def __init__(self, hidden_size=16, intermediate_size=8, shared_intermediate_size=12, num_experts=2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.mixer = _PuzzleMoE(hidden_size, intermediate_size, shared_intermediate_size, num_experts)

    def forward(self, hidden_states):
        return self.mixer(self.norm(hidden_states))


class _PuzzleShell(nn.Module):
    """The `model.layers` shell the Puzzle definition resolves its tree against."""

    def __init__(self, hidden_size=16, intermediate_size=8, shared_intermediate_size=12, num_experts=2):
        super().__init__()
        self.config = SimpleNamespace(n_routed_experts=num_experts)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_PuzzleLayer(hidden_size, intermediate_size, shared_intermediate_size, num_experts)]
        )


def _puzzle_awq_layer(seed=0):
    """Build a synthetic Puzzle MoE layer, its plan and its AWQ scaling nodes."""

    from gptqmodel.models.module_plan import ModulePlan

    torch.manual_seed(seed)
    shell = _PuzzleShell()
    model_def = NemotronHPuzzleQModel.__new__(NemotronHPuzzleQModel)
    nn.Module.__init__(model_def)
    model_def.__dict__["model"] = shell
    tree = NemotronHPuzzleQModel._resolve_effective_module_tree(shell, None)
    model_def.__dict__["effective_module_tree"] = tree
    model_def.__dict__["module_tree"] = tree

    layer = shell.model.layers[0]
    plan = ModulePlan.from_tree(tree)
    features = {}
    for spec in plan.specs:
        if not spec.quantize:
            continue
        templates = [spec.template]
        if "{expert_index}" in spec.template:
            head, _, tail = spec.template.partition("{expert_index}")
            container_path = head.rstrip(".")
            container = layer.get_submodule(container_path)
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
            if weight is None or weight.ndim != 2:
                continue
            features[template] = torch.randn(4, weight.shape[1])

    return layer, plan, model_def.awq_get_modules_for_scaling(layer, features, {})


def test_nemotron_h_puzzle_expert_up_proj_is_not_an_awq_scale_source():
    """#5739: routed and shared `up_proj` must keep the parent's `:nosrc`."""

    from gptqmodel.models.definitions.nemotron_h import NemotronHQModel
    from gptqmodel.models.module_plan import ModulePlan
    from gptqmodel.quantization.awq.utils.module import get_op_name

    layer, _plan, nodes = _puzzle_awq_layer()
    puzzle_plan = ModulePlan.from_tree(NemotronHPuzzleQModel.module_tree)
    parent_plan = ModulePlan.from_tree(NemotronHQModel.module_tree)

    expert_sources = (
        "mixer.experts.0.up_proj",
        "mixer.experts.1.up_proj",
        "mixer.shared_experts.up_proj",
    )
    for path in expert_sources:
        spec = puzzle_plan.find(path)
        assert spec is not None, path
        # `nosrc` only removes the fold source; the projection stays quantized.
        assert "nosrc" in spec.flags, (path, spec.flags)
        assert spec.quantize, (path, spec.flags)
        parent = parent_plan.find(path)
        assert parent is not None and parent.flags == spec.flags, (path, spec.flags, parent.flags)

    groups = [
        (get_op_name(layer, node["prev_op"]), [get_op_name(layer, module) for module in node["layers"]])
        for node in nodes
        if node.get("prev_op") is not None
    ]
    used_sources = {prev_op for prev_op, _consumers in groups}
    assert not (used_sources & set(expert_sources)), (
        f"expert up_proj is still an AWQ fold source: {sorted(used_sources & set(expert_sources))}"
    )

    # The up_proj projections are still folded from the shared post-attention
    # norm, together with the shared expert: one norm, one scale group.
    assert len(groups) == 1, groups
    prev_op, consumers = groups[0]
    assert prev_op == "norm", groups
    assert sorted(consumers) == sorted(expert_sources), groups


def test_nemotron_h_puzzle_expert_neutral_fold_preserves_relu2_output():
    """A neutral (unquantized) fold of every emitted group must not change the layer.

    Before the fix the extra `up_proj -> down_proj` groups fail this check: the
    fold divides `up_proj` rows by `s` and multiplies `down_proj` columns by
    `s`, and `relu2(u / s) != relu2(u) / s`.
    """

    from gptqmodel.quantization.awq.quantize.scale import apply_scale, is_foldable_source
    from gptqmodel.quantization.awq.utils.module import get_op_name

    layer, _plan, nodes = _puzzle_awq_layer()
    hidden = torch.randn(6, layer.norm.weight.numel())
    with torch.no_grad():
        before = layer(hidden).clone()

    folded = 0
    worst = 0.0
    for node in nodes:
        prev_op = node.get("prev_op")
        consumers = node.get("layers") or []
        if prev_op is None or not consumers or not is_foldable_source(prev_op):
            continue
        prev_name = get_op_name(layer, prev_op)
        layer_names = [get_op_name(layer, module) for module in consumers]
        # A Linear source divides its output rows; a norm divides its gain.
        source_width = prev_op.weight.shape[0] if isinstance(prev_op, nn.Linear) else prev_op.weight.numel()
        scales = torch.rand(source_width) + 0.5
        consumers_map = node.get("_scale_consumers") or []
        consumer_map = {prev_name: consumers_map} if consumers_map else None
        with torch.no_grad():
            apply_scale(layer, [(prev_name, layer_names, scales, None)], consumer_map=consumer_map)
            after = layer(hidden)
        worst = max(worst, float((after - before).norm() / before.norm()))
        with torch.no_grad():
            apply_scale(layer, [(prev_name, layer_names, 1.0 / scales, None)], consumer_map=consumer_map)
        folded += 1

    assert folded >= 1, "the Puzzle layer emitted no foldable AWQ group"
    assert worst < 1e-5, f"neutral AWQ fold changed the relu2 expert layer output by {worst:.3e}"


def test_nemotron_h_puzzle_gptq_grouping_is_unchanged_by_nosrc():
    """`nosrc` is an AWQ source flag: it must not change the GPTQ quantized blocks."""

    config = SimpleNamespace(n_routed_experts=2)
    expected = [
        ["norm:!"],
        ["mixer.q_proj", "mixer.k_proj", "mixer.v_proj", "mixer.in_proj"],
        ["mixer.o_proj", "mixer.out_proj"],
        ["mixer.fc1_latent_proj"],
        ["mixer.experts.0.up_proj", "mixer.experts.1.up_proj"],
        ["mixer.experts.0.down_proj", "mixer.experts.1.down_proj"],
        ["mixer.fc2_latent_proj"],
        ["mixer.shared_experts.up_proj"],
        ["mixer.shared_experts.down_proj"],
    ]
    gptq_blocks = NemotronHPuzzleQModel.full_layer_modules(config, is_awq_quantize=False, include_capture_only=True)
    awq_blocks = NemotronHPuzzleQModel.full_layer_modules(config, is_awq_quantize=True, include_capture_only=True)

    assert gptq_blocks == expected
    # The flag only removes the AWQ fold source, so both methods quantize the
    # same blocks in the same order.
    assert awq_blocks == gptq_blocks
