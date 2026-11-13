# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Teacher-KL oracles and projection allocation/materialization regressions."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.quantization.rco import (
    InterpolatedLlamaLayers,
    LlamaTeacherKL,
    discover_blocks,
    projected_gumbel_search,
)


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'scripts/gsq_paper' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


allocate = script('rco_allocate')
materialize = script('rco_materialize')


@pytest.mark.parametrize(
    'requested,backend,sequence_length,batch_tokens,expected',
    [
        (None, 'torch', 128, 2048, 4),
        (None, 'torch', 1, 2048, 4),
        (None, 'mlx', 128, 2048, 16),
        (None, 'mlx', 512, 2048, 4),
        (None, 'mlx', 2048, 2048, 1),
    ],
)
def test_batch_size_defaults_follow_the_backend(
    requested, backend, sequence_length, batch_tokens, expected
):
    assert (
        allocate.resolve_batch_size(requested, backend, sequence_length, batch_tokens)
        == expected
    )


def test_explicit_batch_size_overrides_the_mlx_token_budget():
    assert allocate.resolve_batch_size(3, 'mlx', 128, 2048) == 3
    assert allocate.resolve_batch_size(64, 'mlx', 128, 2048) == 64


def test_mlx_batch_size_never_drops_below_one_sequence():
    assert allocate.resolve_batch_size(None, 'mlx', 100_000, 2048) == 1


def tiny_model(dtype=torch.float64, hidden=16, layers=1):
    torch.manual_seed(13)
    config = LlamaConfig(vocab_size=32, hidden_size=hidden, intermediate_size=2*hidden,
                         num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=layers)
    config._attn_implementation = 'eager'
    return LlamaForCausalLM(config).to(dtype=dtype).eval().requires_grad_(False)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('chunk_tokens', [1, 3, 128])
def test_teacher_kl_value_and_gradient_match_full_vocabulary_oracle(tmp_path, dtype, chunk_tokens):
    model = tiny_model(dtype)
    ids = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    with torch.no_grad():
        teacher_logits = model(ids, use_cache=False).logits.detach().clone()
    objective = LlamaTeacherKL(model, [ids], tmp_path, chunk_tokens=chunk_tokens)
    # Mutate only the student AFTER caching the dense teacher. A shared model
    # recomputed as its own teacher would incorrectly report zero KL here.
    parameter = model.model.layers[0].mlp.down_proj.weight
    with torch.no_grad():
        parameter.add_(torch.randn_like(parameter) * .2)
    parameter.requires_grad_(True)
    actual = objective(0)
    actual_gradient, = torch.autograd.grad(actual, parameter)
    student_logits = model(ids, use_cache=False).logits
    # Independent dense calculation in FP64 from the same source logits.
    reference_logp = teacher_logits.double().log_softmax(-1)
    student_logp = student_logits.double().log_softmax(-1)
    expected = (reference_logp.exp() * (reference_logp - student_logp)).sum(-1).mean()
    expected_gradient, = torch.autograd.grad(expected, parameter)
    assert expected > 0
    tolerance = 1e-12 if dtype == torch.float64 else 1e-6
    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=tolerance, atol=tolerance)
    assert model.lm_head.weight.grad is None
    cached = torch.load(tmp_path / '00000.pt', weights_only=True)
    assert cached.shape == (2, 4, 16) and not cached.requires_grad


def test_teacher_kl_identical_model_is_zero(tmp_path):
    model = tiny_model()
    objective = LlamaTeacherKL(model, [torch.tensor([[1, 2, 3]])], tmp_path)
    assert objective(0).item() == 0


def test_weighted_multisample_search_is_invariant_to_batch_partition(tmp_path):
    ids = torch.tensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
    results = []
    for batch_size in (2, 3):
        model = tiny_model()
        batches = list(ids.split(batch_size))
        objective = LlamaTeacherKL(model, batches, tmp_path / str(batch_size), chunk_tokens=2)
        names = ['model.layers.0.mlp.down_proj', 'model.layers.0.mlp.up_proj']
        sources = {name: model.get_submodule(name).weight.detach().clone() for name in names}
        interp = InterpolatedLlamaLayers(model, {name: [source*.2, source*.8] for name, source in sources.items()},
                                         torch.tensor([source.numel() for source in sources.values()]),
                                         torch.tensor([2., 3.]),
                                         control_device='cpu').setup()
        interp.retract(2.6)
        initial_alpha = interp.alpha.detach().clone()
        torch.manual_seed(51)
        try:
            _, history = projected_gumbel_search(
                interp, list(range(len(batches))), 2.6, steps=2, gumbel_samples=2,
                objective=objective, batch_weights=[batch.numel() for batch in batches], log_every=0)
            assert not torch.equal(interp.alpha.detach(), initial_alpha)
            results.append((interp.alpha.detach().clone(), history))
        finally:
            interp.cleanup()
    torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-6, atol=1e-6)
    for first, second in zip(results[0][1], results[1][1]):
        assert first['loss'] == pytest.approx(second['loss'], abs=1e-12)
        assert first['drift'] < 1e-6


@pytest.mark.parametrize('resident_cache', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_teacher_kl_resident_cache_matches_disk(tmp_path, dtype, resident_cache):
    model = tiny_model(dtype)
    ids = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    with torch.no_grad():
        teacher_logits = model(ids, use_cache=False).logits.detach().clone()
    objective = LlamaTeacherKL(
        model, [ids], tmp_path / str(resident_cache), chunk_tokens=3, resident_cache=resident_cache)
    parameter = model.model.layers[0].mlp.down_proj.weight
    with torch.no_grad():
        parameter.add_(torch.randn_like(parameter) * .2)
    parameter.requires_grad_(True)
    actual = objective(0)
    actual_gradient, = torch.autograd.grad(actual, parameter)
    student_logits = model(ids, use_cache=False).logits
    reference_logp = teacher_logits.double().log_softmax(-1)
    student_logp = student_logits.double().log_softmax(-1)
    expected = (reference_logp.exp() * (reference_logp - student_logp)).sum(-1).mean()
    expected_gradient, = torch.autograd.grad(expected, parameter)
    tolerance = 1e-12 if dtype == torch.float64 else 1e-6
    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=tolerance, atol=tolerance)
    assert bool(list((tmp_path / str(resident_cache)).glob('*.pt'))) is not resident_cache


def test_control_tensors_follow_the_model_device_by_default():
    model = tiny_model()
    names = ['model.layers.0.mlp.down_proj']
    source = model.get_submodule(names[0]).weight.detach().clone()
    objective = InterpolatedLlamaLayers(
        model, {names[0]: [source*.2, source*.8]}, torch.tensor([source.numel()]),
        torch.tensor([2., 3.]))
    assert objective.control_device == model.model.layers[0].mlp.down_proj.weight.device
    assert objective.costs.device == objective.control_device
    assert objective.weights.device == objective.control_device
    assert objective.alpha.device == objective.control_device


def test_preset_defaults_and_explicit_overrides():
    required = ['--model', 'model', '--corpus', 'data', '--out', 'out', '--target-bits', '2.5', '--arms', 'w{bits}']
    default = allocate.parse_args(required)
    assert default.preset == 'paper'
    assert default.device == 'auto' and default.backend == 'auto'
    assert default.teacher_cache == 'device'
    assert (default.objective, default.grouping, default.gumbel_samples, default.options) == ('kl', 'projection', 4, '2,3,4')
    old = allocate.parse_args(required + ['--preset', 'legacy'])
    assert (old.objective, old.grouping, old.gumbel_samples, old.options) == ('ce', 'block', 1, '2,3')
    paper = allocate.parse_args(required + ['--preset', 'paper'])
    assert (paper.objective, paper.grouping, paper.gumbel_samples, paper.options) == ('kl', 'projection', 4, '2,3,4')
    assert vars(paper) == vars(default)
    custom = allocate.parse_args(required + ['--grouping', 'block', '--gumbel-samples', '8'])
    assert (custom.objective, custom.grouping, custom.gumbel_samples) == ('kl', 'block', 8)
    with pytest.raises(SystemExit):
        allocate.parse_args(required + ['--gumbel-samples', '0'])
    with pytest.raises(SystemExit):
        allocate.parse_args(required + ['--backend', 'ane'])


def test_projection_groups_preserve_unequal_parameter_counts():
    counts = {'model.layers.0.self_attn.q_proj': 16, 'model.layers.0.self_attn.k_proj': 8,
              'model.layers.1.mlp.up_proj': 32}
    groups, weights = allocate.allocation_groups(counts, 'projection')
    assert len(groups) == 3
    assert dict(zip((group[0] for group in groups), weights.tolist())) == counts
    blocks = discover_blocks(tiny_model(layers=2), model_definition=LlamaQModel)
    groups, weights = allocate.allocation_groups(counts, 'block', blocks=blocks)
    assert weights.tolist() == [24., 32.]
    assert len(groups) == 2


def test_assignment_schema_accepts_legacy_and_rejects_incomplete_projection_map():
    blocks = discover_blocks(tiny_model(), model_definition=LlamaQModel)
    expected = {name: 2 for block in blocks for name in block.names}
    assert materialize.assignment_projection_bits({'layer_bits': {'0': 2}}, blocks) == expected
    assert materialize.assignment_projection_bits({'projection_bits': expected}, blocks) == expected
    for broken in ({}, {**expected, 'model.layers.9.mlp.up_proj': 2}):
        with pytest.raises(ValueError, match='every projection'):
            materialize.assignment_projection_bits({'projection_bits': broken}, blocks)


def test_projection_materialization_packed_bytes_and_save_reload(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from gptqmodel import GPTQModel
    from gptqmodel.looper.gsq_training_model import finalize_llama_gsq_wrapper, quantize_llama_gsq_model
    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.quantization import FORMAT, GPTQConfig
    from gptqmodel.utils.backend import BACKEND

    model = tiny_model(torch.float16, hidden=64)
    model.save_pretrained(tmp_path / 'dense')
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token='0')),
        unk_token='0', pad_token='0')
    tokenizer.save_pretrained(tmp_path / 'dense')
    config = GPTQConfig(bits=2, group_size=32, format=FORMAT.GPTQ_V2, act_group_aware=False,
                        device='cpu', offload_to_disk=False)
    wrapper = LlamaQModel(model=model, quantized=False, quantize_config=config,
                          tokenizer=tokenizer, model_local_path=str(tmp_path / 'dense'))
    for width in (2, 3):
        quantize_llama_gsq_model(copy.deepcopy(model), [{'input_ids': list(range(1, 9))}],
                                 bits=width, group_size=32, checkpoint_directory=tmp_path / f'w{width}')
    block = discover_blocks(wrapper)[0]
    widths = {name: 2 + i % 2 for i, name in enumerate(block.projections)}
    restored = materialize.restore_assignment_block(
        model.model.layers[0], 0, widths, str(tmp_path / 'w{bits}'), group_size=32, device='cpu',
        projections=block.projections)
    model.model.layers[0] = restored
    # Codes, scales and group indices must come unchanged from the selected arm.
    for name, width in widths.items():
        saved = torch.load(tmp_path / f'w{width}' / 'layer-00.pt', weights_only=True)['state_dict']
        for key, tensor in restored.get_submodule(name).state_dict().items():
            assert torch.equal(tensor, saved[f'{name}.{key}'])
    projection_bits = {f'model.layers.0.{name}': width for name, width in widths.items()}
    run = {'state': 'complete', 'layer_indices': [0], 'bits': 2, 'projection_bits': projection_bits,
           'group_size': 32, 'gsq_training': {'enabled': False}}
    model.gsq_training_runs = [run]
    finalize_llama_gsq_wrapper(wrapper, run)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected_logits = model(ids, use_cache=False).logits
    wrapper.save(str(tmp_path / 'mixed'))
    loaded = GPTQModel.load(str(tmp_path / 'mixed'), backend=BACKEND.TORCH, device='cpu',
                            dtype=torch.float16, attn_implementation='eager')
    for name, width in projection_bits.items():
        assert loaded.model.get_submodule(name).bits == width
        for key, tensor in model.get_submodule(name).state_dict().items():
            assert torch.equal(tensor, loaded.model.get_submodule(name).state_dict()[key])
    with torch.no_grad():
        actual_logits = loaded.model(ids, use_cache=False).logits
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    assert loaded.quantize_config.meta['gsq_projection_bits'] == projection_bits
    with pytest.raises(ValueError, match='group size mismatch'):
        materialize.restore_assignment_block(restored, 0, widths, str(tmp_path / 'w{bits}'),
                                             group_size=128, device='cpu', projections=block.projections)

    # Exercise the real CLI wiring: cache the dense teacher, search, serialize,
    # materialize and reload, on a locally generated model and corpus.
    from datasets import Dataset

    Dataset.from_dict({'input_ids': [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8]]}).save_to_disk(
        str(tmp_path / 'corpus' / 'validation'))
    root = Path(__file__).parents[1]
    env = dict(os.environ, PYTHONPATH=str(root), TOKENIZERS_PARALLELISM='false')
    command = [sys.executable, str(root / 'scripts/gsq_paper/rco_allocate.py'),
               '--model', str(tmp_path / 'dense'), '--corpus', str(tmp_path / 'corpus'),
               '--out', str(tmp_path / 'allocation'), '--arms', str(tmp_path / 'w{bits}'),
               '--options', '2,3', '--target-bits', '2.5',
               '--group-size', '32', '--calib-seqs', '3', '--batch-size', '2',
               '--steps', '2', '--device', 'cpu', '--backend', 'torch']
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads((tmp_path / 'allocation' / 'assignment.json').read_text())
    assert payload['preset'] == 'paper'
    assert payload['backend'] == 'torch'
    assert payload['objective'] == 'kl' and payload['grouping'] == 'projection'
    assert payload['gumbel_samples'] == 4 and payload['group_count'] == 7
    assert len(payload['projection_bits']) == 7 and payload['group_size'] == 32
    assert not list((tmp_path / 'allocation').glob('rco-teacher-*'))
    command = [sys.executable, str(root / 'scripts/gsq_paper/rco_materialize.py'),
               '--model', str(tmp_path / 'dense'), '--assignment', str(tmp_path / 'allocation' / 'assignment.json'),
               '--arms', str(tmp_path / 'w{bits}'), '--out', str(tmp_path / 'cli-model'), '--device', 'cpu']
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr

    from gptqmodel.quantization.rco_mlx import rco_mlx_available
    if rco_mlx_available():
        command = [sys.executable, str(root / 'scripts/gsq_paper/rco_allocate.py'),
                   '--model', str(tmp_path / 'dense'), '--corpus', str(tmp_path / 'corpus'),
                   '--out', str(tmp_path / 'mlx-allocation'), '--arms', str(tmp_path / 'w{bits}'),
                   '--options', '2,3', '--target-bits', '2.5',
                   '--group-size', '32', '--calib-seqs', '3', '--batch-size', '2',
                   '--steps', '1', '--device', 'cpu', '--backend', 'mlx']
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        mlx_payload = json.loads((tmp_path / 'mlx-allocation' / 'assignment.json').read_text())
        assert mlx_payload['backend'] == 'mlx'
        assert mlx_payload['grouping'] == 'projection'
        assert len(mlx_payload['projection_bits']) == 7
        saved = torch.load(tmp_path / 'mlx-allocation' / 'probs.pt', weights_only=True)
        assert set(saved) == {'probs', 'logits', 'history', 'weights', 'costs', 'names', 'groups',
                              'target_bits'}
        assert not list((tmp_path / 'mlx-allocation').glob('rco-teacher-*'))
        assert mlx_payload['seed'] == 7
        repeated = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
        assert repeated.returncode == 0, repeated.stdout + repeated.stderr
        repeated_saved = torch.load(tmp_path / 'mlx-allocation' / 'probs.pt', weights_only=True)
        torch.testing.assert_close(saved['logits'], repeated_saved['logits'], rtol=0, atol=0)
        command = [sys.executable, str(root / 'scripts/gsq_paper/rco_materialize.py'),
                   '--model', str(tmp_path / 'dense'),
                   '--assignment', str(tmp_path / 'mlx-allocation' / 'assignment.json'),
                   '--arms', str(tmp_path / 'w{bits}'), '--out', str(tmp_path / 'mlx-cli-model'),
                   '--device', 'cpu']
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
