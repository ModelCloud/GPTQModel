# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Non-uniform per-layer bit allocation for staged scalar GSQ.

A budget-constrained search such as RCO emits one width per decoder block.
GSQ can only act on that assignment if a block trains and exports at mixed
widths, so these tests pin the resolution, budget arithmetic and export
behaviour that the paper's 2.37/2.62 bits-per-parameter rows depend on.
"""

import pytest
import torch


def test_projection_bits_accept_scalar_and_map():
    from gptqmodel.quantization.gsq_training import resolve_projection_bits

    names = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
             'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
    assert resolve_projection_bits(3) == {name: 3 for name in names}

    partial = {name: (3 if name.startswith('self_attn.q') else 2) for name in names}
    assert resolve_projection_bits(partial) == partial


@pytest.mark.parametrize(
    'value',
    [5, 1, True, 2.0, {'self_attn.q_proj': 3}, {'self_attn.q_proj': 5, 'self_attn.k_proj': 3}],
)
def test_projection_bits_reject_invalid_widths(value):
    from gptqmodel.quantization.gsq_training import resolve_projection_bits

    with pytest.raises(ValueError):
        resolve_projection_bits(value)


def test_projection_bits_reject_unknown_projection():
    from gptqmodel.quantization.gsq_training import resolve_projection_bits

    names = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
             'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
    mapping = {name: 2 for name in names}
    mapping['mlp.nonexistent'] = 3
    with pytest.raises(ValueError):
        resolve_projection_bits(mapping)


def test_budget_plan_reproduces_paper_promoted_layers():
    """Sixteen equal-width blocks promote 6 / 10 layers at 2.375 / 2.625 bits."""
    from gptqmodel.quantization.gsq_training import projection_bits_for_budget

    counts = {index: 1 for index in range(16)}
    plan = projection_bits_for_budget(counts, [2.375, 2.625])

    for target, expected in ((2.375, 6), (2.625, 10)):
        widths = plan[target]
        assert sum(1 for value in widths.values() if value == 3) == expected
        assert sum(1 for value in widths.values() if value == 2) == 16 - expected
        average = sum(widths.values()) / len(widths)
        assert average == pytest.approx(target)


def test_budget_plan_weights_by_parameter_count():
    from gptqmodel.quantization.gsq_training import projection_bits_for_budget

    # One block holds 3x the parameters. Promoting the *small* block is what
    # reaches 2.25: (3*2 + 1*3) / 4. Promoting the large one would overshoot to
    # 2.75, so a parameter-blind schedule would miss the budget.
    counts = {0: 3, 1: 1}
    plan = projection_bits_for_budget(counts, [2.25])
    assert plan[2.25] == {0: 2, 1: 3}

    total = sum(counts.values())
    achieved = sum(plan[2.25][index] * count for index, count in counts.items()) / total
    assert achieved == pytest.approx(2.25)


@pytest.mark.parametrize('target', [1.5, 4.0])
def test_budget_plan_rejects_out_of_range_targets(target):
    from gptqmodel.quantization.gsq_training import projection_bits_for_budget

    with pytest.raises(ValueError):
        projection_bits_for_budget({0: 1}, [target])


def test_dynamic_layer_bits_resolves_only_whole_blocks():
    from types import SimpleNamespace

    from gptqmodel.looper.gsq_training_model import dynamic_layer_bits

    config = SimpleNamespace(dynamic={'model.layers.0': {'bits': 3}, 'model.layers.1': {'bits': 2}})
    assert dynamic_layer_bits(config) == {0: 3, 1: 2}

    assert dynamic_layer_bits(SimpleNamespace(dynamic=None)) is None
    assert dynamic_layer_bits(SimpleNamespace(dynamic={})) is None


@pytest.mark.parametrize(
    'dynamic',
    [
        {'model.layers.0.q_proj': {'bits': 3}},
        {'model.layers.0': {'bits': 3, 'group_size': 64}},
        {'model.layers.0': {'sym': False}},
        {'model.layers.0': {'bits': 5}},
        {'model.layers.0': {'bits': True}},
        {'model.layers.x': {'bits': 3}},
    ],
)
def test_dynamic_layer_bits_rejects_unsupported_overrides(dynamic):
    from types import SimpleNamespace

    from gptqmodel.looper.gsq_training_model import dynamic_layer_bits

    with pytest.raises(ValueError):
        dynamic_layer_bits(SimpleNamespace(dynamic=dynamic))


def _one_block_fixture():
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=128, hidden_size=256, intermediate_size=768,
                         num_attention_heads=4, num_key_value_heads=4, head_dim=64,
                         max_position_embeddings=256, num_hidden_layers=1)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).eval()
    documents = [{'input_ids': torch.randint(0, 128, (48,)).tolist()} for _ in range(4)]
    return model, documents


def test_block_exports_mixed_projection_widths():
    """One block must pack different widths per projection."""
    from gptqmodel.looper.gsq_training_capture import capture_llama_gsq_inputs
    from gptqmodel.quantization.gsq_training import quantize_llama_gsq_block
    from gptqmodel.quantization.gsq_training_config import GSQTrainingConfig

    model, documents = _one_block_fixture()
    cache = capture_llama_gsq_inputs(model, documents, layer_index=0)
    batches = [(hidden[0], kwargs)
               for hidden, kwargs in zip(cache.layer_inputs, cache.layer_input_kwargs)]

    widths = {'self_attn.q_proj': 3, 'self_attn.k_proj': 3, 'self_attn.v_proj': 2,
              'self_attn.o_proj': 2, 'mlp.gate_proj': 2, 'mlp.up_proj': 2, 'mlp.down_proj': 3}
    gsq = GSQTrainingConfig(enabled=False, epochs=1, batch_size=2, microbatch_size=2, qk_steps=4)
    packed, result = quantize_llama_gsq_block(model.model.layers[0], batches, bits=widths,
                                              group_size=128, gsq=gsq, pack=True)

    assert result['projection_bits'] == widths
    for name, want in widths.items():
        module = packed.get_submodule(name)
        assert int(module.bits) == want
        assert torch.isfinite(module.dequantize_weight()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs an accelerator')
def test_search_state_keeps_one_device():
    """Costs must follow the control device, not stay on the host.

    The database is assembled from CPU checkpoints while the search runs on an
    accelerator, so an unpinned ``costs`` tensor fails only at retraction time.
    """
    from gptqmodel.quantization.rco import InterpolatedLlamaLayers, retraction

    device = torch.device('cuda:0')
    costs = torch.tensor([2.0, 3.0])
    interpolated = InterpolatedLlamaLayers(
        object(), {'layers.0.self_attn.q_proj': []},
        torch.tensor([10.0]), costs, control_device=device)

    assert interpolated.costs.device == device
    assert interpolated.weights.device == device
    assert interpolated.fraction.device == device
    assert interpolated.alpha.device == device

    # The retraction is where a device mismatch surfaced: it rebuilds a softmax
    # from alpha and costs on every bisection probe.
    achieved = retraction(interpolated.alpha, interpolated.costs, 2.375,
                          interpolated.fraction)
    assert abs(achieved - 2.375) < 1e-5

    # Rounding reads its dynamic-programming table on the host but must hand the
    # assignment back on the caller's device, where the search indexes it.
    from gptqmodel.quantization.rco import round_with_budget

    probs = torch.softmax(torch.randn(2, 2, device=device), dim=-1)
    assignment = round_with_budget(probs, torch.tensor([10.0, 12.0], device=device),
                                   torch.tensor([2.0, 3.0], device=device), 2.375)
    assert assignment.device == device


def test_mixed_export_publishes_per_module_bits():
    """A mixed export must survive a reload at its own widths.

    Load-time module construction resolves bits from ``dynamic``, so a mixed
    checkpoint that does not publish the assignment would rebuild every layer
    at the single top-level width.
    """
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

    from gptqmodel.looper.gsq_training_model import prepare_llama_gsq_export
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=128, hidden_size=256, intermediate_size=768,
                         num_attention_heads=4, num_key_value_heads=4, head_dim=64,
                         max_position_embeddings=256, num_hidden_layers=2)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).eval()

    names = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
             'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
    for layer in model.model.layers:
        for name in names:
            source = layer.get_submodule(name)
            linear = TorchLinear(bits=2, group_size=128, sym=True, desc_act=False,
                                 in_features=source.in_features,
                                 out_features=source.out_features, bias=False)
            parent, leaf = name.rsplit('.', 1)
            setattr(layer.get_submodule(parent), leaf, linear)

    # Layer 0 at W3, layer 1 at W2: genuinely mixed, so the base (2) needs an
    # override for exactly the layers that differ from it.
    run = {
        'state': 'complete',
        'layer_indices': [0, 1],
        'bits': {0: 3, 1: 2},
        'layer_bits': {0: 3, 1: 2},
        'group_size': 128,
        'gsq_training': {'enabled': True},
    }
    model.gsq_training_runs = [run]
    exported = prepare_llama_gsq_export(model, run)

    assert exported.bits == 2
    assert exported.dynamic is not None
    assert set(exported.dynamic) == {f'model.layers.0.{name}' for name in names}
    assert all(entry == {'bits': 3} for entry in exported.dynamic.values())
    assert exported.meta['gsq_layer_bits'] == {'0': 3, '1': 2}


def test_block_grouping_hits_paper_budgets_exactly():
    """Sixteen block groups reproduce the paper's 6/10 promotions.

    The knapsack selects the best-scoring feasible assignment, so it only
    spends the full budget when the objective prefers it; a converged soft
    solution that favours W3 in the first k blocks must land exactly on the
    paper's 2.375 / 2.625 bits per parameter.
    """
    from gptqmodel.quantization.rco import achieved_bits, round_with_budget

    weights = torch.tensor([7 * 86_900_000.0] * 16)
    costs = torch.tensor([2.0, 3.0])
    for target, promoted in ((2.375, 6), (2.625, 10)):
        alpha = torch.full((16, 2), -6.0)
        alpha[:, 0] = 6.0
        alpha[:promoted, 0] = -6.0
        alpha[:promoted, 1] = 6.0
        assignment = round_with_budget(torch.softmax(alpha, dim=-1), weights, costs, target)
        assert int((assignment == 1).sum()) == promoted
        assert abs(achieved_bits(assignment, weights, costs) - target) < 1e-6


def test_groups_partition_modules():
    """A grouped search shares one decision across a block's projections."""
    from gptqmodel.quantization.rco import InterpolatedLlamaLayers

    names = [f'model.layers.{block}.p{index}' for block in range(2) for index in range(3)]
    groups = [[n for n in names if int(n.split('.')[2]) == block] for block in range(2)]
    interpolated = InterpolatedLlamaLayers(
        object(), {name: [] for name in names}, torch.tensor([10.0, 10.0]),
        torch.tensor([2.0, 3.0]), control_device='cpu', groups=groups)

    assert interpolated.alpha.shape == (2, 2)
    assert [interpolated.index_of(name) for name in names] == [0, 0, 0, 1, 1, 1]


def test_groups_must_partition_exactly():
    from gptqmodel.quantization.rco import InterpolatedLlamaLayers

    with pytest.raises(ValueError):
        InterpolatedLlamaLayers(object(), {'a': [], 'b': []}, torch.tensor([1.0]),
                                torch.tensor([2.0, 3.0]), control_device='cpu',
                                groups=[['a']])
