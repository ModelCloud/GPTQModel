# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Per-format RCO adapters: option sets, costs, discovery and allocation."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.config import (
    _GGUF_APPROX_BITS_PER_WEIGHT_BY_ALIAS,
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


def tiny_model(layers=2, with_g_proj=True):
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
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
    names = discover_projections(tiny_model())
    assert any(name.endswith("self_attn.g_proj") for name in names)
    assert not any("lm_head" in name for name in names)
    assert not any("embed_tokens" in name for name in names)
    # Every block contributes the same eight projections, ordered by block.
    assert len(names) == 16
    assert names[0].startswith("model.layers.0.")
    assert names[-1].startswith("model.layers.1.")


def test_discovery_honours_suffix_and_exclusion_filters():
    model = tiny_model()
    only_mlp = discover_projections(model, suffixes=("gate_proj", "up_proj", "down_proj"))
    assert {name.rsplit(".", 1)[-1] for name in only_mlp} == {
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    excluded = discover_projections(model, exclude=[only_mlp[0]])
    assert only_mlp[0] not in excluded
    assert len(excluded) == len(discover_projections(model)) - 1


def test_module_weights_counts_the_quantised_matrix_only():
    model = tiny_model()
    names = discover_projections(model)
    weights, kept = module_weights(model, names)
    assert kept == names
    # q_proj is 64x64 and gate_proj is 128x64 in this fixture.
    assert float(weights[names.index("model.layers.0.self_attn.q_proj")]) == 64 * 64
    assert float(weights[names.index("model.layers.0.mlp.gate_proj")]) == 128 * 64


def test_awq_options_carry_the_scale_overhead():
    adapter = adapter_for(AWQConfig(bits=4, group_size=128))
    assert adapter.search == "interpolation"
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
    assert adapter.costs().tolist() == [
        _GGUF_APPROX_BITS_PER_WEIGHT_BY_ALIAS[alias] for alias in GGUF_STAGED_ALIASES
    ]
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
    try:
        from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCppKernel
    except Exception:  # pragma: no cover - depends on the ggml runtime
        pytest.skip("ggml runtime is unavailable")
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
            linear_cls=GGUFCppKernel,
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
    names = discover_projections(model)
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
    weights, names = module_weights(model, discover_projections(model))
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


def test_runner_weights_the_budget_by_parameter_count():
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
    path_scores = Path("/tmp/rco_runner_params_scores.json")
    path_scores.write_text(
        json.dumps(
            {
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


def test_runner_loads_staged_arm_scores_as_columns():
    """The EXL3 sweep writes one file per width; they must fuse into a matrix."""
    runner = script("rco_format_allocate")

    names = ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.up_proj"]
    paths = []
    for index, width in enumerate((2.0, 3.0, 4.0)):
        path = Path(f"/tmp/rco_runner_arm_{index}.json")
        path.write_text(
            json.dumps({"losses": {name: 0.5**width + offset for offset, name in enumerate(names)}})
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
