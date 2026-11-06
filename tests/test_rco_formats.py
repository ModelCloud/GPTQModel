# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Per-format RCO adapters: option sets, costs, discovery and allocation."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.quantization.config import (
    FORMAT_FIELD_CODE,
    AWQConfig,
    EXL3Config,
    GGUFConfig,
)
from gptqmodel.quantization.rco_formats import (
    AWQ_STAGED_WIDTHS,
    EXL3_STAGED_WIDTHS,
    GGUF_STAGED_ALIASES,
    RCOFormatError,
    RCOPlan,
    adapter_for,
    allocate,
    awq_runtime_requirement,
    default_resolution,
    discover_projections,
    module_weights,
    relative_reconstruction_error,
    repair_overspend,
)


def script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "scripts" / "gsq_paper" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExtendedLlama(LlamaQModel):
    module_tree = copy.deepcopy(LlamaQModel.module_tree)
    module_tree[-1]["self_attn"] += ("g_proj:0:in=x",)


def tiny_model(layers=2, with_g_proj=True, hidden=64):
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config).eval()
    if with_g_proj:
        for layer in model.model.layers:
            layer.self_attn.g_proj = torch.nn.Linear(64, 64, bias=False)
    return model


def test_discovery_includes_extra_projections_and_excludes_the_head():
    names = discover_projections(tiny_model(), model_definition=ExtendedLlama)
    assert any(name.endswith("self_attn.g_proj") for name in names)
    assert not any("lm_head" in name for name in names)
    assert not any("embed_tokens" in name for name in names)
    # Every block contributes the same eight projections, ordered by block.
    assert len(names) == 16
    assert names[0].startswith("model.layers.0.")
    assert names[-1].startswith("model.layers.1.")


def test_discovery_honours_definition_tags_and_exclusions():
    class MLPOnly(ExtendedLlama):
        module_tree = ["model", "layers", "#", {"mlp": ExtendedLlama.module_tree[-1]["mlp"]}]

    model = tiny_model()
    only_mlp = discover_projections(model, model_definition=MLPOnly)
    assert len(only_mlp) == 6
    excluded = discover_projections(model, model_definition=ExtendedLlama, exclude=[only_mlp[0]])
    assert only_mlp[0] not in excluded
    assert len(excluded) == 15


def test_module_weights_counts_the_quantised_matrix_only():
    model = tiny_model()
    names = discover_projections(model, model_definition=ExtendedLlama)
    weights, kept = module_weights(model, names)
    assert kept == names
    # q_proj is 64x64 and gate_proj is 128x64 in this fixture.
    assert float(weights[names.index("model.layers.0.self_attn.q_proj")]) == 64 * 64
    assert float(weights[names.index("model.layers.0.mlp.gate_proj")]) == 128 * 64


def test_awq_options_carry_the_scale_overhead():
    adapter = adapter_for(AWQConfig(bits=4, group_size=128))
    assert adapter.search == "reconstruction"
    assert adapter.labels() == tuple(f"w{bits}" for bits in AWQ_STAGED_WIDTHS)
    costs = adapter.costs().tolist()
    for bits, cost in zip(AWQ_STAGED_WIDTHS, costs):
        # One fp16 scale plus one `bits`-wide zero point per group.
        assert cost == pytest.approx(bits + (16 + bits) / 128)


def test_awq_mixed_width_requires_the_portable_kernel():
    adapter = adapter_for(AWQConfig(bits=4, group_size=128))
    note = adapter.runtime_requirement({"a": 0, "b": 2})
    assert "AWQ_TORCH" in note and "4-bit" in note
    assert adapter.runtime_requirement({"a": 2, "b": 2}) is None
    assert awq_runtime_requirement([4]) == ""


def test_exl3_options_are_integer_trellis_widths():
    adapter = adapter_for(EXL3Config(bits=3))
    assert adapter.search == "reconstruction"
    assert adapter.labels() == tuple(f"k{bits}" for bits in EXL3_STAGED_WIDTHS)
    assert adapter.costs().tolist() == [float(bits) for bits in EXL3_STAGED_WIDTHS]
    assert all(float(option.payload["bits"]).is_integer() for option in adapter.options())


def test_gguf_options_use_the_ggml_block_costs():
    adapter = adapter_for(GGUFConfig(bits=4))
    assert adapter.search == "reconstruction"
    assert adapter.labels() == GGUF_STAGED_ALIASES
    from gptqmodel.nn_modules.qlinear.gguf import _GGUF_TYPE_INFO, _normalize_gguf_bits

    assert adapter.costs().tolist() == [
        _GGUF_TYPE_INFO[_normalize_gguf_bits(alias)[1]]["type_size"] * 8 /
        _GGUF_TYPE_INFO[_normalize_gguf_bits(alias)[1]]["block_size"] for alias in GGUF_STAGED_ALIASES
    ]
    assert adapter.costs()[-1] == 6.5625
    entry = adapter.dynamic_entry(adapter.options()[0])
    assert entry["bits"] == GGUF_STAGED_ALIASES[0]
    assert entry[FORMAT_FIELD_CODE] == GGUF_STAGED_ALIASES[0]


def test_unknown_gguf_alias_is_rejected():
    adapter = adapter_for(GGUFConfig(bits=4))
    adapter.aliases = ("q4_k", "not_a_ggml_type")
    with pytest.raises(RCOFormatError):
        adapter.options()


def test_gguf_module_resolves_the_per_module_type_from_dynamic():
    """A width alone is ambiguous: `4` is `q4_0` by default, not `q4_k`.

    RCO's GGUF plan writes the ggml type into `dynamic`, so the module factory
    must fold that back into the structured spec instead of defaulting by width.
    """
    from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
    from gptqmodel.quantization import FORMAT
    from gptqmodel.utils.model import create_quant_module

    names = ("model.layers.0.mlp.down_proj", "model.layers.0.mlp.up_proj")
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    root.model.layers[0].mlp = torch.nn.Module()
    for name in names:
        setattr(root.model.layers[0].mlp, name.rsplit(".", 1)[-1], torch.nn.Linear(256, 256, bias=False))

    config = GGUFConfig(
        bits="q4_k",
        dynamic={
            names[0]: {"bits": 4, "format": "q_k"},
            names[1]: {"bits": 3, "format": "q_k"},
        },
    )
    for name in names:
        create_quant_module(
            name=name,
            linear_cls=GGUFTorchLinear,
            bits=config.runtime_bits,
            desc_act=False,
            dynamic=config.dynamic,
            group_size=-1,
            module=root,
            submodule=root.get_submodule(name),
            sym=True,
            device="cpu",
            lm_head_name="lm_head",
            pack_dtype=torch.int32,
            format=FORMAT.GGUF,
            register_buffers=False,
        )
    assert root.get_submodule(names[0]).gguf_tensor_qtype == "Q4_K"
    assert root.get_submodule(names[1]).gguf_tensor_qtype == "Q3_K"


@pytest.mark.parametrize("method", ["awq", "exl3", "gguf"])
def test_allocation_respects_the_budget_and_the_option_set(method):
    model = tiny_model()
    names = discover_projections(model, model_definition=ExtendedLlama)
    weights, names = module_weights(model, names)
    adapter = adapter_for(
        {"awq": AWQConfig(bits=4), "exl3": EXL3Config(bits=3), "gguf": GGUFConfig(bits=4)}[method]
    )
    torch.manual_seed(7)
    scores = -torch.rand(len(names), len(adapter.options()))
    plan = allocate(adapter, scores, weights, 3.0, names=names)
    assert plan.achieved_bits <= 3.0 + 1e-3
    assert set(plan.dynamic) == set(names)
    assert sum(plan.counts.values()) == len(names)
    assert {plan.labels[index] for index in plan.assignment} == set(plan.counts)
    for name, entry in plan.dynamic.items():
        assert entry == adapter.dynamic_entry(adapter.options()[plan.assignment[plan.names.index(name)]])


def test_plan_round_trips_through_json():
    model = tiny_model()
    weights, names = module_weights(model, discover_projections(model, model_definition=ExtendedLlama))
    adapter = adapter_for(EXL3Config(bits=3))
    torch.manual_seed(3)
    scores = -torch.rand(len(names), len(adapter.options()))
    plan = allocate(adapter, scores, weights, 2.5, names=names)
    restored = RCOPlan.from_dict(plan.to_dict())
    assert restored.dynamic == plan.dynamic
    assert restored.assignment == plan.assignment
    assert restored.achieved_bits == plan.achieved_bits
    assert restored.runtime_note == plan.runtime_note


def test_allocation_rejects_a_bad_score_shape():
    adapter = adapter_for(EXL3Config(bits=3))
    weights = torch.ones(4)
    with pytest.raises(RCOFormatError):
        allocate(adapter, torch.zeros(4, 2), weights, 2.5, names=[f"m{i}" for i in range(4)])
    with pytest.raises(RCOFormatError):
        allocate(adapter, torch.zeros(4, 3), torch.ones(3), 2.5, names=[f"m{i}" for i in range(4)])


def test_allocation_never_overspends():
    adapter = adapter_for(EXL3Config(bits=3))
    weights = torch.ones(4)
    names = [f"m{i}" for i in range(4)]
    # Every option is maximally attractive at 4 bits, so a 2.5 target must still
    # come back feasible.
    scores = torch.tensor([[0.0, 0.0, 1.0]] * 4)
    plan = allocate(adapter, scores, weights, 2.5, names=names)
    assert plan.achieved_bits == pytest.approx(2.5)
    # 3 x k2 + 1 x k4 spends exactly 10 of the 10 available units and is the
    # only feasible mix that includes the top-scoring k4 option.
    assert plan.counts == {"k2": 3, "k4": 1}


def test_default_resolution_is_large_enough_for_the_optimum():
    assert default_resolution(8) >= 256
    assert default_resolution(640) == 64 * 640
    assert default_resolution(0) == 256


def test_repair_overspend_reaches_the_budget_without_mutating_the_input():
    costs = torch.tensor([2.0, 3.0, 4.0])
    weights = torch.ones(4)
    scores = torch.tensor([[0.0, 0.0, 1.0]] * 4)
    assignment = torch.tensor([2, 2, 2, 0])  # 14 / 4 = 3.5 bits, over a 3.0 target
    repaired = repair_overspend(assignment, scores, weights, costs, 3.0)
    assert float((costs[repaired] * weights).sum() / weights.sum()) <= 3.0
    assert assignment.tolist() == [2, 2, 2, 0]
    # One k4 -> k2 swap saves two bits for the same score loss as two k4 -> k3
    # swaps, so the greedy repair takes the single larger step.
    assert repaired.tolist() == [0, 2, 2, 0]


def test_large_allocation_never_overspends_the_budget():
    """The 640-module case is where DP rounding used to break the budget guard."""
    torch.manual_seed(23)
    modules = 640
    weights = torch.linspace(1e6, 5e7, modules)
    adapter = adapter_for(EXL3Config(bits=3))
    scores = -torch.rand(modules, len(adapter.options())) * 0.05
    names = [f"model.layers.{index // 8}.p{index % 8}" for index in range(modules)]
    plan = allocate(adapter, scores, weights, 2.5, names=names)
    assert plan.achieved_bits <= 2.5 + 1e-9
    assert sum(plan.counts.values()) == modules


@pytest.mark.parametrize("with_hessian", [False, True])
def test_reconstruction_error_matches_an_fp64_oracle(with_hessian):
    torch.manual_seed(11)
    reference = torch.randn(16, 32, dtype=torch.float32)
    dequantized = reference + 0.05 * torch.randn(16, 32, dtype=torch.float32)
    hessian = torch.randn(32, 32, dtype=torch.float32)
    hessian = hessian @ hessian.t()

    got = relative_reconstruction_error(
        reference, dequantized, hessian if with_hessian else None
    )

    ref = reference.double().numpy()
    err = dequantized.double().numpy() - ref
    if with_hessian:
        h = hessian.double().numpy()
        expected = ((err @ h) * err).sum() / ((ref @ h) * ref).sum()
    else:
        expected = (err * err).sum() / (ref * ref).sum()
    assert got == pytest.approx(float(expected), rel=1e-6, abs=1e-12)
    assert isinstance(got, float)


def test_reconstruction_error_rejects_a_misshaped_hessian():
    reference = torch.randn(4, 8)
    with pytest.raises(RCOFormatError):
        relative_reconstruction_error(reference, reference, torch.eye(4))


def test_awq_processor_resolves_per_module_bits():
    """The staging path must use the width the module is packed at."""

    class Stub:
        qcfg = AWQConfig(
            bits=4,
            dynamic={"model.layers.0.mlp.down_proj": {"bits": 3}},
        )
        _module_bits = AWQProcessor._module_bits

    stub = Stub()
    assert stub._module_bits("model.layers.0.mlp.down_proj") == 3
    assert stub._module_bits("model.layers.1.mlp.down_proj") == 4
    assert stub._module_bits(None) == 4


def test_awq_processor_ignores_non_numeric_dynamic_overrides():
    """An excluded module or a rule that only sets another key is not width 0."""

    class Stub:
        qcfg = AWQConfig(
            bits=4,
            dynamic={
                "-model.layers.0.mlp.up_proj": {"bits": 2},
                "model.layers.1.mlp.up_proj": {"group_size": 64},
            },
        )
        _module_bits = AWQProcessor._module_bits

    stub = Stub()
    assert stub._module_bits("model.layers.0.mlp.up_proj") == 4
    assert stub._module_bits("model.layers.1.mlp.up_proj") == 4


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_runner_awq_quantizer_matches_the_processor(bits, sym):
    """The RCO score must come from the same affine quantizer AWQ stages."""
    runner = script("rco_format_allocate")

    class Stub:
        qcfg = AWQConfig(bits=4, group_size=32, sym=sym)
        pseudo_quantize_tensor = AWQProcessor.pseudo_quantize_tensor

    torch.manual_seed(17)
    weight = torch.randn(8, 64, dtype=torch.float32)
    expected, _, _ = Stub().pseudo_quantize_tensor(weight, bits=bits)
    got = runner.awq_dequantize(weight, bits, 32, sym)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_awq_pseudo_quantize_honours_the_requested_width():
    """A 3-bit stage must produce a 3-bit code range and pack into 3 bits."""

    class Stub:
        qcfg = AWQConfig(bits=4, group_size=32, sym=True)
        _module_bits = AWQProcessor._module_bits
        pseudo_quantize_tensor = AWQProcessor.pseudo_quantize_tensor

    stub = Stub()
    torch.manual_seed(5)
    weight = torch.randn(16, 64, dtype=torch.float32)
    wq4, scales4, zeros4 = stub.pseudo_quantize_tensor(weight)
    wq3, scales3, zeros3 = stub.pseudo_quantize_tensor(weight, bits=3)

    # Codes are recovered as (wq / scale) + zero point, so the widest code must
    # stay inside the signed 3-bit range.
    per_column3 = scales3.repeat_interleave(32, dim=1)
    codes3 = (wq3.view(16, -1) / per_column3).round()
    # Symmetric 3-bit codes span [-4, 3].
    assert codes3.abs().max() <= 4
    assert codes3.abs().max() >= 3  # the range is actually used
    assert zeros3.max() == 4  # 2 ** (3 - 1)
    assert zeros4.max() == 8  # 2 ** (4 - 1)
    assert not torch.equal(wq3, wq4)
    assert torch.isfinite(wq3).all()


def test_runner_weights_the_budget_by_parameter_count(tmp_path):
    """A 5.2M k_proj must not outweigh a 209.7M mlp projection in the budget.

    The budget is a ratio over quantised parameters, so the solver has to see
    real `in*out` counts. Uniform weights let it "buy" accuracy by upgrading
    every small matrix and downgrading the MLPs, which the parameter budget
    would never allow.
    """
    runner = script("rco_format_allocate")

    names = ["model.layers.0.self_attn.k_proj", "model.layers.0.mlp.up_proj"]
    # Identical error curves: only the parameter counts can break the tie.
    scores = {
        names[0]: [0.5, 0.1, 0.01],
        names[1]: [0.5, 0.1, 0.01],
    }
    params = {names[0]: 5_242_880, names[1]: 209_715_200}
    path_scores = tmp_path / "scores.json"
    path_scores.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "search": "reconstruction",
                "score_metric": "relative_frobenius",
                "settings": {},
                "shapes": {names[0]: [1024, 5120], names[1]: [40960, 5120]},
                "method": "gguf",
                "options": ["q2_k", "q3_k", "q4_k"],
                "errors": scores,
                "params": params,
            }
        )
    )

    loaded_names, loaded_errors = runner._load_score_matrix(str(path_scores))
    assert loaded_names == names
    assert loaded_errors.shape == (2, 3)

    weighted = torch.tensor([params[name] for name in loaded_names], dtype=torch.float32)
    assert weighted[1] / weighted[0] == pytest.approx(40.0)

    plan = allocate(
        _gguf_adapter(("q2_k", "q3_k", "q4_k")),
        -loaded_errors,
        weighted,
        3.0,
        names=loaded_names,
    )
    # The cheap-to-upgrade small matrix takes the wide option, the 40x larger
    # MLP stays narrow: uniform weights would upgrade both.
    assert plan.dynamic[names[1]]["bits"] in {"q2_k", "q3_k"}
    assert plan.dynamic[names[0]]["bits"] == "q4_k"


def _gguf_adapter(aliases):
    adapter = adapter_for(GGUFConfig(bits="q4_k"))
    adapter.aliases = tuple(aliases)
    return adapter


def test_runner_loads_staged_arm_scores_as_columns(tmp_path):
    """The EXL3 sweep writes one file per width; they must fuse into a matrix."""
    runner = script("rco_format_allocate")

    names = ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.up_proj"]
    paths = []
    for index, width in enumerate((2.0, 3.0, 4.0)):
        path = tmp_path / f"arm_{index}.json"
        path.write_text(
            json.dumps({
                "schema_version": 1, "method": "exl3", "search": "reconstruction",
                "score_metric": "relative_hessian", "settings": {"codebook": "mcg", "out_scales": "auto"},
                "options": [f"k{int(width)}"], "params": dict.fromkeys(names, 16384),
                "shapes": dict.fromkeys(names, [128, 128]),
                "losses": {name: 0.5**width + offset for offset, name in enumerate(names)},
            })
        )
        paths.append(str(path))

    fused_names, fused = runner._load_score_matrix(",".join(paths))
    assert fused_names == names
    assert fused.shape == (2, 3)
    # Column j must hold arm j's value for every module, not arm j's rows.
    for index, path in enumerate(paths):
        arm = json.loads(Path(path).read_text())["losses"]
        for row, name in enumerate(names):
            assert fused[row][index].item() == pytest.approx(arm[name])


def test_repair_can_downgrade_the_same_module_more_than_once():
    assignment = torch.tensor([2])
    repaired = repair_overspend(assignment, torch.tensor([[0., 100., 101.]]),
                                torch.ones(1), torch.tensor([1., 2., 3.]), 1.)
    assert repaired.tolist() == [0]
    assert assignment.tolist() == [2]


def test_budget_checks_do_not_round_parameter_counts_to_fp32():
    adapter = adapter_for(EXL3Config(bits=3), options=(2, 4))
    weights = torch.tensor([16777217, 16777216], dtype=torch.float64)
    plan = allocate(adapter, torch.tensor([[0., 10.], [0., 1.]]), weights, 3., names=['a', 'b'])
    total_bits = sum(weights[i].item() * plan.costs[option] for i, option in enumerate(plan.assignment))
    assert total_bits <= 3 * weights.sum().item()
    assert plan.achieved_bits == total_bits / weights.sum().item()


@pytest.mark.parametrize('weights,target', [([0., 1.], 3.), ([-1., 2.], 3.), ([float('nan'), 1.], 3.),
                                           ([1., 1.], 1.999), ([1., 1.], float('nan'))])
def test_invalid_or_infeasible_budgets_are_rejected(weights, target):
    with pytest.raises(RCOFormatError):
        allocate(adapter_for(EXL3Config(bits=3)), torch.zeros(2, 3), torch.tensor(weights), target, names=['a', 'b'])


def test_duplicate_names_are_rejected():
    with pytest.raises(RCOFormatError, match='unique'):
        allocate(adapter_for(EXL3Config(bits=3)), torch.zeros(2, 3), torch.ones(2), 3., names=['a', 'a'])


def test_row_specific_costs_match_a_brute_force_oracle():
    import itertools

    from gptqmodel.quantization.rco import budget_constrained_argmax

    scores = torch.tensor([[1., 8., 10.], [1., 5., 7.], [2., 9., 11.]], dtype=torch.float64)
    weights = torch.tensor([1., 2., 3.], dtype=torch.float64)
    costs = torch.tensor([[2., 3., 4.], [2.5, 3.5, 4.5], [2., 3., 4.]], dtype=torch.float64)
    target = 3.
    feasible = [a for a in itertools.product(range(3), repeat=3)
                if sum(float(costs[i, j] * weights[i]) for i, j in enumerate(a)) <= target * float(weights.sum())]
    expected = max(feasible, key=lambda a: sum(float(scores[i, j]) for i, j in enumerate(a)))
    got = budget_constrained_argmax(scores, weights, costs, target, resolution=1200)
    assert tuple(got.tolist()) == expected


@pytest.mark.parametrize('group_size', [-1, 16, 32])
@pytest.mark.parametrize('width', [2, 3, 4])
def test_awq_cost_includes_partial_words_and_metadata(group_size, width):
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    adapter = adapter_for(AWQConfig(bits=4, group_size=group_size), options=(width,))
    module = AwqTorchLinear(bits=width, group_size=group_size, sym=True, desc_act=False,
                            in_features=32, out_features=8, register_buffers=True)
    actual = sum(t.numel() * t.element_size() * 8 for t in module.buffers())
    assert adapter.cost_matrix([(8, 32)]).item() * 256 == actual


@pytest.mark.parametrize('alias', ['q4_0', 'q2_k', 'q3_k', 'q6_k'])
def test_gguf_cost_and_measurement_include_row_padding(alias):
    from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear

    adapter = adapter_for(GGUFConfig(bits=alias), options=(alias,))
    module = GGUFTorchLinear(bits=alias, group_size=-1, sym=True, desc_act=False,
                             in_features=260, out_features=8, register_buffers=True)
    actual = module.qweight.numel() * module.qweight.element_size() * 8
    assert adapter.cost_matrix([(8, 260)]).item() * (8 * 260) == pytest.approx(actual)
    if alias in {'q4_0', 'q6_k'}:  # CPU quantizers available without optional native ggml.
        weight = torch.randn(8, 260)
        assert script('rco_format_allocate').gguf_dequantize(weight, alias).shape == weight.shape


def test_exl3_cost_includes_sign_scales_and_codebook_marker():
    adapter = adapter_for(EXL3Config(bits=3), options=(3,))
    # trellis[int16, 8, 8, 48], suh[fp16,128], svh[fp16,128], mcg[int32].
    expected = 8 * 8 * 48 * 16 + 2 * 128 * 16 + 32
    assert adapter.cost_matrix([(128, 128)]).item() * (128 * 128) == expected


def score_payload(method='awq', options=('w4', 'w2'), names=('a', 'b')):
    settings = {'group_size': 32, 'sym': False, 'format': 'gemm'} if method == 'awq' else {}
    return {'schema_version': 1, 'method': method, 'search': 'reconstruction', 'settings': settings,
            'score_metric': 'relative_frobenius', 'options': list(options),
            'errors': {name: [0.01, 0.2] for name in names},
            'params': dict.fromkeys(names, 4096), 'shapes': dict.fromkeys(names, [64, 64])}


def test_runner_preserves_custom_option_order_and_measured_settings(tmp_path):
    runner = script('rco_format_allocate')
    scores, plan_path = tmp_path / 'scores.json', tmp_path / 'plan.json'
    scores.write_text(json.dumps(score_payload()))
    runner.main(['allocate', '--method', 'awq', '--scores', str(scores), '--target-bits', '3.6', '--out', str(plan_path)])
    plan = RCOPlan.from_dict(json.loads(plan_path.read_text()))
    assert plan.labels == ('w4', 'w2')
    assert plan.search == 'reconstruction'
    args = runner.parse_args(['quantize', '--method', 'awq', '--out', 'unused', '--device', 'cpu'])
    config = runner.quant_config('awq', plan, args)
    assert config.group_size == 32 and config.sym is False
    with pytest.raises(ValueError, match='method'):
        runner.quant_config('gguf', plan, args)
    args.group_size = 128
    with pytest.raises(ValueError, match='measured'):
        runner.quant_config('awq', plan, args)


@pytest.mark.parametrize('mutation', ['method', 'missing_params', 'bad_count', 'negative_error', 'duplicate_options'])
def test_runner_rejects_mislabelled_or_unbudgeted_scores(tmp_path, mutation):
    runner = script('rco_format_allocate')
    payload = score_payload()
    if mutation == 'method':
        payload['method'] = 'exl3'
    elif mutation == 'missing_params':
        del payload['params']
    elif mutation == 'bad_count':
        payload['params']['a'] = 1
    elif mutation == 'negative_error':
        payload['errors']['a'][0] = -1
    elif mutation == 'duplicate_options':
        payload['options'] = ['w2', 'w2']
    path = tmp_path / 'scores.json'
    path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, TypeError)):
        runner._load_score_matrix(str(path))


@pytest.mark.parametrize('mutation', [
    'dynamic', 'achieved_bits', 'params', 'module_costs', 'width_counts', 'search', 'budget_unit', 'options',
])
def test_plan_revalidates_assignment_and_budget_on_read(mutation):
    adapter = adapter_for(AWQConfig(bits=4, group_size=32, sym=False), options=(4, 2))
    plan = allocate(adapter, torch.tensor([[-.01, -.2], [-.01, -.2]]), torch.full((2,), 4096.), 3.6,
                    names=['a', 'b'], shapes={'a': (64, 64), 'b': (64, 64)})
    payload = plan.to_dict()
    if mutation == 'dynamic':
        payload['dynamic']['a']['bits'] = 3
    elif mutation == 'achieved_bits':
        payload['achieved_bits'] = 0
    elif mutation == 'params':
        payload['params']['a'] = 1
    elif mutation == 'module_costs':
        payload['module_costs']['a'] = [2., 2.]
    elif mutation == 'width_counts':
        payload['width_counts'] = {}
    elif mutation == 'search':
        payload['search'] = 'interpolation'
    elif mutation == 'budget_unit':
        payload['budget_unit'] = 'nominal_weight_bits'
    elif mutation == 'options':
        payload['options'] = []
    with pytest.raises(RCOFormatError):
        RCOPlan.from_dict(payload)


@pytest.mark.parametrize('method', ['awq', 'gguf'])
def test_measure_allocate_quantize_and_reload(tmp_path, method):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from gptqmodel import GPTQModel
    from gptqmodel.utils.backend import BACKEND

    model_dir = tmp_path / 'dense'
    tiny_model(layers=1, with_g_proj=False, hidden=128).save_pretrained(model_dir)
    tokenizer = Tokenizer(WordLevel({str(i): i for i in range(64)}, unk_token='0'))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='0', pad_token='0').save_pretrained(model_dir)
    runner = script('rco_format_allocate')
    scores, plan_path, output = tmp_path / 'scores.json', tmp_path / 'plan.json', tmp_path / 'quantized'
    common = ['--method', method, '--model', str(model_dir), '--device', 'cpu', '--dtype', 'float16']
    measure_options = (['--options', '3,2', '--group-size', '32', '--no-sym'] if method == 'awq' else
                       ['--options', 'q6_k,q4_0'])
    runner.main(['measure', *common, *measure_options, '--out', str(scores)])
    payload = json.loads(scores.read_text())
    assert payload['options'] == (['w3', 'w2'] if method == 'awq' else ['q6_k', 'q4_0'])
    assert len(payload['errors']) == 7
    target = '3.1' if method == 'awq' else '6.0'
    runner.main(['allocate', '--method', method, '--scores', str(scores), '--target-bits', target, '--out', str(plan_path)])
    calibration = tmp_path / 'calibration.json'
    calibration.write_text(json.dumps([' '.join(str(i) for i in range(1, 33))] * 4))
    runner.main(['quantize', *common, '--plan', str(plan_path), '--calibration', str(calibration), '--out', str(output)])
    backend = BACKEND.AWQ_TORCH if method == 'awq' else BACKEND.GGUF_TORCH
    loaded = GPTQModel.load(str(output), backend=backend, device='cpu', dtype=torch.float16)
    plan = RCOPlan.from_dict(json.loads(plan_path.read_text()))
    actual_bits = 0
    for name in plan.names:
        module = loaded.model.get_submodule(name)
        assert module.bits == plan.dynamic[name]['bits']
        if method == 'awq':
            assert module.group_size == 32
        actual_bits += sum(t.numel() * t.element_size() * 8 for key, t in module.named_buffers()
                           if key in {'qweight', 'qzeros', 'scales'})
    assert actual_bits / sum(plan.weights) <= plan.target_bits
    assert actual_bits / sum(plan.weights) == pytest.approx(plan.achieved_bits)
    with torch.inference_mode():
        output_logits = loaded.model(torch.tensor([[1, 2, 3, 4]])).logits
    assert torch.isfinite(output_logits).all()


def test_exl3_measurement_recomputes_error_instead_of_using_native_sentinels(monkeypatch):
    import sys
    from types import SimpleNamespace

    runner = script('rco_format_allocate')
    seen = {}

    def quantize(weight, h_data, args, return_weight_q):
        seen.update(args)
        assert return_weight_q is True
        h_data['H'].zero_()  # native finalization mutates its private copy
        return weight * .5, -1., {}

    monkeypatch.setitem(sys.modules, 'gptqmodel.exllamav3.modules.quant.exl3_lib.quantize',
                        SimpleNamespace(quantize_exl3=quantize))
    weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    hessian = torch.eye(4)
    config = EXL3Config(bits=3, codebook='mul1', out_scales='never')
    assert runner.exl3_error(weight, 3, hessian, torch.device('cpu'), config) == pytest.approx(.25)
    assert seen['K'] == 3 and seen['mul1'] is True and seen['apply_out_scales'] is False
    assert torch.equal(hessian, torch.eye(4))


def test_quantize_rejects_a_partial_measured_inventory(tmp_path, monkeypatch):
    from gptqmodel import GPTQModel
    from tests.test_rco_discovery import wrapper_for

    model = tiny_model(layers=1, with_g_proj=False)
    names = discover_projections(model, model_definition=LlamaQModel)[:1]
    shapes = {name: tuple(model.get_submodule(name).weight.shape) for name in names}
    weights, _ = module_weights(model, names)
    adapter = adapter_for(AWQConfig(bits=4, group_size=32), options=(2, 3))
    plan = allocate(adapter, torch.zeros(1, 2), weights, 4., names=names, shapes=shapes)
    path = tmp_path / 'partial.json'
    path.write_text(json.dumps(plan.to_dict()))
    wrapper = wrapper_for(model, LlamaQModel)
    monkeypatch.setattr(GPTQModel, 'load', lambda *args, **kwargs: wrapper)
    with pytest.raises(ValueError, match='complete model definition'):
        script('rco_format_allocate').main(['quantize', '--method', 'awq', '--model', 'unused', '--plan', str(path),
                                          '--out', str(tmp_path / 'out'), '--device', 'cpu'])
    assert not (tmp_path / 'out').exists()
