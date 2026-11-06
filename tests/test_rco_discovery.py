# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Projection identity and completeness must come from the model definition."""

import copy
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.definitions.ricursive import RicursiveQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import GPTQConfig
from gptqmodel.quantization.rco import block_of, discover_blocks, discover_projections
from tests.test_rco_paper import allocate, materialize


class TaggedModel(BaseQModel):
    # Deliberately neither Llama names nor a fixed path depth. Numeric ancestors
    # and experts must not be mistaken for decoder block indices.
    layer_modules_strict = False
    module_tree = [
        'trunk', 'branches', '1', 'stages', '#',
        {'': ('renamed:0:q', 'embed:0:v', 'optional:1:o', 'skip:!:norm', 'capture:?')},
    ]

    @classmethod
    def should_quantize_module(cls, model, module_name, module, quantize_config):
        return not getattr(module, 'keep_dense', False)


def tagged_model(layers=2):
    model = torch.nn.Module()
    model.config = SimpleNamespace(num_experts=2)
    model.trunk = torch.nn.Module()
    model.trunk.branches = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    model.trunk.branches[0].decoy = torch.nn.Linear(64, 64, bias=False)
    stages = torch.nn.ModuleList()
    for index in range(layers):
        block = torch.nn.Module()
        for name in ('renamed', 'embed', 'skip', 'capture', 'q_proj'):
            setattr(block, name, torch.nn.Linear(64, 64, bias=False))
        if index == 0:
            block.optional = torch.nn.Linear(64, 64, bias=False)
        stages.append(block)
    model.trunk.branches[1].stages = stages
    return model


def inventory(model):
    return discover_blocks(model, model_definition=TaggedModel)


def wrapper_for(model, definition, tree=None):
    # Only the fields discovery consumes; no loader/tokenizer needed for a tree test.
    wrapper = definition.__new__(definition)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.effective_module_tree = tree if tree is not None else definition.module_tree
    wrapper.quantize_config = GPTQConfig(offload_to_disk=False)
    return wrapper


def test_tags_select_modules_independently_of_names_and_numeric_ancestors():
    model = tagged_model(12)
    blocks = inventory(model)
    assert len(blocks) == 12
    assert blocks[0].projections == ('embed', 'optional', 'renamed')
    assert blocks[1].projections == ('embed', 'renamed')
    names = discover_projections(model, model_definition=TaggedModel)
    assert len(names) == 25
    assert {block_of(name, blocks) for name in names} == set(range(12))
    assert block_of('trunk.branches.0.decoy', blocks) is None
    assert all('skip' not in name and 'capture' not in name and 'q_proj' not in name for name in names)
    counts = {name: index + 1 for index, name in enumerate(names)}
    groups, weights = allocate.allocation_groups(counts, 'block', blocks=blocks)
    assert groups == [list(block.names) for block in blocks]
    assert weights.tolist() == [sum(counts[name] for name in block.names) for block in blocks]


def test_effective_tree_and_definition_hook_override_the_class_declaration():
    model = tagged_model(1)
    block = model.trunk.branches[1].stages[0]
    block.renamed.keep_dense = True
    tree = [*TaggedModel.module_tree[:-1], {'': ('renamed:q', 'q_proj:k', 'embed:!')}]
    wrapper = wrapper_for(model, TaggedModel, tree)
    assert discover_blocks(wrapper)[0].projections == ('q_proj',)
    assert inventory(model)[0].projections == ('embed', 'optional')


def test_dynamic_and_exact_exclusions():
    model = tagged_model(1)
    wrapper = wrapper_for(model, TaggedModel)
    path = inventory(model)[0].path
    wrapper.quantize_config.dynamic = {f'-:^{path}\\.embed$': {}}
    assert discover_blocks(wrapper)[0].projections == ('optional', 'renamed')
    assert discover_projections(wrapper, exclude=[f'{path}.optional']) == (f'{path}.renamed',)


def test_moe_placeholders_expand_from_definition_without_allocating_router():
    class Experts(TaggedModel):
        dynamic_expert_index = 'num_experts'
        module_tree = [*TaggedModel.module_tree[:-1], {
            'bank:moe:?': {'router': ('router:!',), 'units': {'#': ('a:0:gate', 'b:1:down')}}}]

    model = tagged_model(1)
    bank = torch.nn.Module()
    bank.router = torch.nn.Linear(64, 2)
    bank.units = torch.nn.ModuleList()
    for _ in range(2):
        expert = torch.nn.Module()
        expert.a = torch.nn.Linear(64, 64, bias=False)
        expert.b = torch.nn.Linear(64, 64, bias=False)
        bank.units.append(expert)
    model.trunk.branches[1].stages[0].bank = bank
    blocks = discover_blocks(model, model_definition=Experts)
    assert blocks[0].projections == ('bank.units.0.a', 'bank.units.0.b', 'bank.units.1.a', 'bank.units.1.b')
    assert {block_of(name, blocks) for name in blocks[0].names} == {0}


def test_tree_variants_use_their_own_leaf_declarations():
    class Variants(TaggedModel):
        module_tree = [
            ['trunk', 'branches', '0', 'stages', '#', {'': ('q_proj:q',)}],
            TaggedModel.module_tree,
        ]

    model = tagged_model(1)
    model.trunk.branches[0].stages = torch.nn.ModuleList([copy.deepcopy(model.trunk.branches[1].stages[0])])
    blocks = discover_blocks(model, model_definition=Variants)
    assert [block.projections for block in blocks] == [('q_proj',), ('embed', 'optional', 'renamed')]
    assert [block.index for block in blocks] == [0, 1]


def test_missing_definition_and_unsupported_declared_modules_fail_without_fallback():
    model = tagged_model(1)
    with pytest.raises(ValueError, match='explicit model definition'):
        discover_projections(model)
    model.trunk.branches[1].stages[0].renamed = torch.nn.Conv1d(64, 64, 1)
    with pytest.raises(ValueError, match='cannot restore declared projection'):
        inventory(model)
    wrapper = wrapper_for(tagged_model(1), TaggedModel, [*TaggedModel.module_tree[:-1], {'': ('skip:!',)}])
    with pytest.raises(ValueError, match='no allocatable projections'):
        discover_blocks(wrapper)

    class Strict(TaggedModel):
        layer_modules_strict = True

    with pytest.raises(ValueError, match='declared projection is missing'):
        discover_blocks(tagged_model(2), model_definition=Strict)


def test_llama_and_ripple_use_their_own_declared_projection_sets():
    model = torch.nn.Module()
    model.config = SimpleNamespace()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    for _ in range(2):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'g_proj'):
            setattr(layer.self_attn, name, torch.nn.Linear(64, 64, bias=False))
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            setattr(layer.mlp, name, torch.nn.Linear(64, 64, bias=False))
        model.model.layers.append(layer)
    llama = discover_projections(model, model_definition=LlamaQModel)
    ripple = discover_projections(model, model_definition=RicursiveQModel)
    assert len(llama) == 14 and len(ripple) == 16
    assert set(ripple) - set(llama) == {f'model.layers.{i}.self_attn.g_proj' for i in range(2)}


def test_assignment_completeness_is_independent_of_the_payload():
    blocks = inventory(tagged_model())
    expected = {name: 2 for block in blocks for name in block.names}
    assert materialize.assignment_projection_bits({'layer_bits': {'0': 2, '1': 2}}, blocks) == expected
    assert materialize.assignment_projection_bits({'projection_bits': expected}, blocks) == expected
    # An entire projection omitted in every block used to validate itself.
    missing = {name: width for name, width in expected.items() if not name.endswith('.embed')}
    for broken in (missing, {}, {**expected, 'trunk.branches.0.decoy': 2}):
        with pytest.raises(ValueError, match='every projection'):
            materialize.assignment_projection_bits({'projection_bits': broken}, blocks)
    with pytest.raises(ValueError, match='every projection'):
        materialize.assignment_projection_bits({'layer_bits': {'0': 2, '1': 2, '2': 2}}, blocks)


def write_arm(model, block, directory, width):
    packed = copy.deepcopy(model.get_submodule(block.path))
    for name in block.projections:
        source = packed.get_submodule(name)
        linear = TorchLinear(bits=width, group_size=32, sym=True, desc_act=False,
                             in_features=source.in_features, out_features=source.out_features, bias=False)
        linear.qweight.fill_(width)
        linear.scales.fill_(width)
        packed.set_submodule(name, linear)
    directory.mkdir()
    checkpoint = directory / f'layer-{block.index:02d}.pt'
    torch.save({'layer_index': block.index, 'signature': {'bits': width, 'group_size': 32},
                'state_dict': packed.state_dict()}, checkpoint)
    return checkpoint


def test_loading_and_materializing_use_all_declared_packed_projections(tmp_path):
    model = tagged_model(1)
    block = inventory(model)[0]
    for width in (2, 3):
        path = write_arm(model, block, tmp_path / f'w{width}', width)
        restored = allocate.load_block(model, path, width, block=block, group_size=32)
        for name in block.projections:
            assert isinstance(restored.get_submodule(name), TorchLinear)
            assert (restored.get_submodule(name).qweight == width).all()
    widths = {name: 2 + i % 2 for i, name in enumerate(block.projections)}
    restored = materialize.restore_assignment_block(
        model.get_submodule(block.path), block.index, widths, str(tmp_path / 'w{bits}'),
        group_size=32, device='cpu', projections=block.projections)
    for name, width in widths.items():
        saved = torch.load(tmp_path / f'w{width}' / 'layer-00.pt', weights_only=True)['state_dict']
        for key, value in restored.get_submodule(name).state_dict().items():
            assert torch.equal(value, saved[f'{name}.{key}'])
    assert isinstance(restored.q_proj, torch.nn.Linear)  # undeclared decoy stays dense
    with pytest.raises(ValueError, match='every declared projection'):
        materialize.restore_assignment_block(
            model.get_submodule(block.path), block.index, {'renamed': 2}, str(tmp_path / 'w{bits}'),
            group_size=32, device='cpu', projections=block.projections)
    path = tmp_path / 'w2' / 'layer-00.pt'
    saved = torch.load(path, weights_only=True)
    del saved['state_dict']['embed.qweight']
    torch.save(saved, path)
    with pytest.raises(RuntimeError, match='embed.qweight'):
        allocate.load_block(model, path, 2, block=block, group_size=32)


def test_export_uses_effective_definition_including_an_extra_projection():
    from gptqmodel.looper.gsq_training_model import finalize_llama_gsq_wrapper
    from tests.test_rco_paper import tiny_model

    model = tiny_model(torch.float16, hidden=64)
    model.model.layers[0].extra_projection = torch.nn.Linear(64, 64, bias=False)
    tree = copy.deepcopy(LlamaQModel.module_tree)
    tree[-1][''] = ('extra_projection:0:q',)
    wrapper = wrapper_for(model, LlamaQModel, tree)
    block = discover_blocks(wrapper)[0]
    assert len(block.projections) == 8
    projection_bits = {}
    for name, full_name in zip(block.projections, block.names):
        source = model.get_submodule(full_name)
        width = 3 if name == 'extra_projection' else 2
        model.set_submodule(full_name, TorchLinear(
            bits=width, group_size=32, sym=True, desc_act=False,
            in_features=source.in_features, out_features=source.out_features, bias=False))
        projection_bits[full_name] = width
    run = {'state': 'complete', 'layer_indices': [0], 'bits': 2, 'projection_bits': projection_bits,
           'group_size': 32, 'gsq_training': {'enabled': False}}
    model.gsq_training_runs = [run]
    omitted = projection_bits.pop('model.layers.0.extra_projection')
    with pytest.raises(ValueError, match='exact all-layer projection assignment'):
        finalize_llama_gsq_wrapper(wrapper, run)
    projection_bits['model.layers.0.extra_projection'] = omitted
    finalize_llama_gsq_wrapper(wrapper, run)
    assert wrapper.quantize_config.dynamic == {'model.layers.0.extra_projection': {'bits': 3}}
    assert wrapper.quantize_config.meta['gsq_projection_bits'] == projection_bits
