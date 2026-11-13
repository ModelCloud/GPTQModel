# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Independent staged-update oracle and real torchrun integration tests.

Ordinary pytest runs CPU arithmetic and one two-process Gloo integration job.
To check RCCL on allocated GPUs, including eight ranks AFTER coordination:

  PYTHONPATH=/monster/home/gptqmodel-ddp python -m torch.distributed.run \
    --standalone --nproc-per-node=2 tests/test_gsq_distributed_stage.py --worker --device cuda

Change nproc-per-node to 4 or 8 without changing the global batch or microbatch
size. Every worker compares actual GSQ gradients, parameters, and Lion momentum
against independent float64 Torch softmax/autograd/Lion arithmetic. Rank zero
also runs the actual SINGLE-device update over the concatenation of ALL rank
microbatches, with the exact same per-microbatch draws. It is a secondary check,
not the mathematical oracle. No checkpoint or model artifact is written.
"""

import argparse
import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gptqmodel.quantization import gsq_training as stage
from scripts.gsq_paper.bench_distributed_stage import (
    draw_seed,
    initialize_distributed,
    microbatch_plan,
)


class RecordedLion(stage.GSQLion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = 0
        self.gradients = []

    def step(self, closure=None):
        self.gradients = [parameter.grad.detach().clone() for group in self.param_groups
                          for parameter in group['params']]
        self.steps += 1
        return super().step(closure)


def fixture(bits, device):
    """Source tensors define the oracle; candidate intermediates are never reused."""
    rows, columns, group = 3, 8, 4
    choices = 4 if bits == 2 else 5
    codes = (torch.arange(rows*columns).reshape(rows, columns) % (2**bits) - 2**(bits-1)).double()
    scales = torch.tensor([[.25, .5], [.5, .25], [.25, .5]], dtype=torch.float64)
    weight = codes*scales.repeat_interleave(group, dim=1)
    generator = torch.Generator().manual_seed(311)
    logits = torch.randn(choices, rows, columns, generator=generator, dtype=torch.float64)*.07
    quantizer = stage.GSQScalarTrainingModule(
        weight.to(device), scales.to(device), group, bits=bits,
        noise=torch.zeros_like(logits, device=device)).to(dtype=torch.float64)
    with torch.no_grad():
        quantizer.logits.copy_(logits)
        quantizer.scales.copy_(scales)
    # Everything needed by the oracle comes from the specified fixture inputs.
    reference = {'logits': logits.to(device).requires_grad_(),
                     'scales': scales.to(device).requires_grad_(), 'codes': codes.to(device),
                     'group': group, 'bits': bits}
    return quantizer, reference


def source_batches(chunks, device):
    result = []
    for index, start, end in chunks:
        generator = torch.Generator().manual_seed(899 + index)
        # The first chunk has fewer valid output elements. Equal rank averaging
        # is wrong even when every rank has the same number of documents.
        length = 1 if index == 0 else 3
        inputs = torch.randn(end-start, length, 8, generator=generator, dtype=torch.float64).to(device)
        target = torch.randn(end-start, length, 3, generator=generator, dtype=torch.float64).to(device)*.4
        result.append(((inputs, target), target.numel()))
    return result


def explicit_draws(chunks, shape, device, update):
    result = []
    for index, _, _ in chunks:
        generator = torch.Generator().manual_seed(draw_seed(7, update, index, 0))
        uniform = torch.rand(shape, generator=generator, dtype=torch.float64)
        # Include both endpoints as well as ordinary draws in the independent oracle.
        uniform.flatten()[0], uniform.flatten()[1] = 0., 1.
        result.append({'weight': uniform.to(device)})
    return result


def objective(batch, weights):
    inputs, target = batch
    return ((inputs @ weights['weight'].T) - target).square().mean()


def oracle_weight(reference, uniform, temperature, multiplier):
    logits, scales = reference['logits'], reference['scales']
    if reference['bits'] == 2:
        candidates = torch.arange(-2, 2, dtype=torch.float64, device=logits.device)[:, None, None]
        scores = logits
        initial = 0.
    else:
        candidates = torch.arange(-2, 3, dtype=torch.float64, device=logits.device)[:, None, None]
        initial = reference['codes']
        valid = (initial+candidates >= -4) & (initial+candidates <= 3)
        scores = logits.masked_fill(~valid, -torch.inf)
    noise = -torch.log(-torch.log(uniform+1e-8)+1e-8)
    probability = torch.softmax((scores*multiplier+noise)/temperature, dim=0)
    expected_code = initial + (probability*candidates).sum(dim=0)
    return expected_code*scales.repeat_interleave(reference['group'], dim=1)


def oracle_update(reference, batches, draws, momentum, *, temperature, multiplier, lrs):
    total = sum(count for _, count in batches)
    # Sum the actual output squared errors, not a mean of per-rank means.
    loss = sum(((batch[0] @ oracle_weight(reference, draw['weight'], temperature, multiplier).T)
                - batch[1]).square().sum()
               for (batch, _), draw in zip(batches, draws))/total
    parameters = [reference['logits'], reference['scales']]
    gradients = torch.autograd.grad(loss, parameters)
    with torch.no_grad():
        for parameter, gradient, average, lr, decay in zip(parameters, gradients, momentum, lrs, (.3, 0.)):
            direction = torch.sign(.9*average + .1*gradient)
            parameter.copy_(parameter*(1-lr*decay)-lr*direction)
            average.copy_(.95*average + .05*gradient)
    return loss.detach(), gradients


def check_updates(device, *, context=None, rank=0, world_size=1, bits=2, updates=4, global_batch=64):
    plan = microbatch_plan(global_batch, 2, world_size)
    all_chunks = [chunk for shard in plan for chunk in shard]
    global_batches = source_batches(all_chunks, device)
    local_batches = source_batches(plan[rank], device)
    quantizer, reference = fixture(bits, device)
    if context is not None:
        # W2 candidates are an expanded, overlapping buffer. Real initialization
        # must synchronize that buffer as well as parameters without copy_ into
        # overlapping storage; constructing directly on-device preserves it.
        context.sync_quantizers({'weight': quantizer})
    single, _ = fixture(bits, device)
    groups = {'assignment_lr': .003, 'scale_lr': .002, 'weight_decay': .3}
    optimizer = RecordedLion(quantizer.optimizer_groups(**groups), betas=(.9, .95))
    single_optimizer = RecordedLion(single.optimizer_groups(**groups), betas=(.9, .95)) if rank == 0 else None
    momentum = [torch.zeros_like(reference[key]) for key in ('logits', 'scales')]
    generator = torch.Generator(device=device).manual_seed(123+rank)
    max_error = 0.

    def compare(actual, expected):
        nonlocal max_error
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        max_error = max(max_error, float((actual.detach()-expected.detach()).abs().max()))
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)

    for step in range(updates):
        temperature = 1.3-.1*step
        multiplier = 2.+.2*step
        lrs = [.003*(1-.1*step), .002*(1-.1*step)]
        for opt in (optimizer, single_optimizer):
            if opt is not None:
                for group, lr in zip(opt.param_groups, lrs):
                    group['lr'] = lr
        all_draws = explicit_draws(all_chunks, quantizer.logits.shape, device, step)
        local_draws = explicit_draws(plan[rank], quantizer.logits.shape, device, step)
        expected_loss, expected_gradients = oracle_update(
            reference, global_batches, all_draws, momentum,
            temperature=temperature, multiplier=multiplier, lrs=lrs)
        arguments = {'generator': generator, 'temperature': temperature, 'multiplier': multiplier}
        if rank == 0:
            single_loss = stage.train_stage_update(
                {'weight': single}, single_optimizer, global_batches, objective,
                uniforms=all_draws, **arguments)
            compare(single_loss, expected_loss)
        loss = stage.train_stage_update(
            {'weight': quantizer}, optimizer, local_batches, objective,
            uniforms=local_draws, **arguments, **({'distributed': context} if context is not None else {}))
        compare(loss, expected_loss)
        for opt, module in ((optimizer, quantizer), (single_optimizer, single)):
            if opt is None:
                continue
            assert opt.steps == step+1
            for i, (name, parameter) in enumerate(module.named_parameters()):
                compare(opt.gradients[i], expected_gradients[i])
                compare(parameter, reference[name])
                compare(opt.state[parameter]['exp_avg'], momentum[i])
        # Hard assignments are compared independently, including endpoint codes.
        logits = reference['logits'].detach()
        candidates = torch.arange(-2, 2 if bits == 2 else 3, device=device)[:, None, None]
        if bits == 3:
            candidates = candidates+reference['codes']
            logits = logits.masked_fill((candidates < -4) | (candidates > 3), -torch.inf)
        selected = candidates.expand_as(logits).gather(0, logits.argmax(0, keepdim=True)).squeeze(0)
        hard = selected*reference['scales'].detach().repeat_interleave(4, 1)
        compare(quantizer.hard_weight(), hard)
    return {'bits': bits, 'world_size': world_size, 'updates': updates, 'global_documents': global_batch,
                'global_output_elements': sum(count for _, count in global_batches),
                'rank_output_elements': [sum(count for _, count in source_batches(shard, device)) for shard in plan],
                'maximum_absolute_error': max_error, 'oracle_dtype': 'float64',
                'single_device_checked': rank == 0, 'gradients_parameters_and_lion_checked': True}


class ScalarFitQuantizer(torch.nn.Module):
    """Analytic fit fixture: scalar moves by the Lion LR, no stochastic ambiguity."""

    def __init__(self, value, device):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([value], dtype=torch.float64, device=device))
        self.scales = torch.nn.Parameter(torch.ones(1, dtype=torch.float64, device=device))

    def forward(self, *, uniform, temperature, multiplier, trusted_uniform=False):
        return self.logits + 0*self.scales

    def hard_weight(self):
        return self.logits.detach().clone()

    def optimizer_groups(self, *, assignment_lr, scale_lr, weight_decay):
        return [{'params': [self.logits], 'lr': assignment_lr, 'weight_decay': weight_decay},
                {'params': [self.scales], 'lr': scale_lr, 'weight_decay': 0.}]


def check_fit_selection(device, context=None, rank=0, world_size=1):
    # Rank zero alone prefers the INITIAL checkpoint. The global weighted target
    # is .75, and with .25 steps every rank must choose epoch 2 and restore .75.
    quantizer = ScalarFitQuantizer(0. if rank == 0 else -3., device)
    if world_size == 1:
        validation = [[((0., 'validation'), 1), ((1., 'validation'), 3)]]
    else:
        validation = [[((0. if rank == 0 else 1., 'validation'),
                        world_size-1 if rank == 0 else 3)]]
    draws = []

    def loss(batch, weights):
        target, _ = batch
        return (weights['weight']-target).square().mean()

    handle = quantizer.register_forward_pre_hook(lambda module, args, kwargs: draws.append(
        float(kwargs['uniform'].flatten()[0])), with_kwargs=True)
    try:
        result = stage.fit_reconstruction_stage(
            {'weight': quantizer}, [[((10., 'train'), 1)]], loss, epochs=5, seed=19,
            assignment_lr=.25, scale_lr=.01, weight_decay=0., decay='constant',
            validation_batches=validation, restore_best=True,
            **({'distributed': context} if context is not None else {}))
    finally:
        handle.remove()
    assert result['hard_loss_before'] == pytest.approx(100.)  # sync before evaluation
    assert result['validation_hard_loss_before'] == pytest.approx(.75)
    assert result['best_validation_epoch'] == 2
    assert result['validation_hard_loss_after'] == pytest.approx(.1875)
    assert result['restored_best_validation_checkpoint']
    assert float(quantizer.logits.detach()[0]) == pytest.approx(.75)
    for epoch, entry in enumerate(result['validation_history']):
        value = .25*(epoch+1)
        assert entry['hard_loss'] == pytest.approx(.25*value**2+.75*(value-1)**2)
    assert len(set(draws)) == 5
    if world_size > 1:
        rank_draws = [None]*world_size
        dist.all_gather_object(rank_draws, draws)
        assert len({tuple(values) for values in rank_draws}) == world_size
    return {'best_epoch': result['best_validation_epoch'], 'heldout_loss': .1875,
                'initial_sync_checked': True, 'distinct_rank_rng_checked': world_size > 1}


def check_fit_budget(device, context=None, rank=0, world_size=1):
    # Run all 1280 updates, using a tiny analytic objective. The batch dimensions
    # represent output elements; schedule expectations below use independent math.
    quantizer = ScalarFitQuantizer(0., device)
    local = microbatch_plan(64, 2, world_size)[rank]
    batch = [((start, end), end-start) for _, start, end in local]
    total_steps = 20*(4096//64)
    result = stage.fit_reconstruction_stage(
        {'weight': quantizer}, [batch]*64,
        lambda payload, weights: (weights['weight']-1.).square().mean(), epochs=20,
        assignment_lr=.001, scale_lr=.0005, weight_decay=0., min_lr=.1, decay='cosine',
        temperature=(2., .05), multiplier=(100., 500.), export_weights=False,
        **({'distributed': context} if context is not None else {}))
    history = result['history']
    assert len(history) == total_steps == 1280
    assert sum(end-start for shard in microbatch_plan(64, 2, world_size) for _, start, end in shard) == 64
    for step, entry in enumerate(history):
        fraction = step/(1280-1)
        factor = .1+.9*.5*(1+math.cos(math.pi*fraction))
        assert entry['step'] == step
        assert entry['learning_rates'] == pytest.approx([.001*factor, .0005*factor], rel=1e-13)
        assert entry['temperature'] == pytest.approx(2.+(.05-2.)*fraction)
        assert entry['multiplier'] == pytest.approx(100.+400.*fraction)
    if world_size > 1:
        summaries = [None]*world_size
        dist.all_gather_object(summaries, [(item['step'], item['batch'], item['learning_rates']) for item in history])
        assert all(summary == summaries[0] for summary in summaries)
    return {'optimizer_updates': total_steps, 'global_batch': 64, 'train_documents': 4096, 'epochs': 20,
                'initial_lrs': history[0]['learning_rates'], 'final_lrs': history[-1]['learning_rates']}


@pytest.mark.parametrize('bits', [2, 3])
def test_single_stage_matches_independent_float64_oracle(bits):
    torch.set_num_threads(1)
    check_updates(torch.device('cpu'), bits=bits, global_batch=17)


def test_global_microbatch_partition_and_noise_cadence():
    expected = microbatch_plan(64, 2)[0]
    for world_size in (1, 2, 4, 8):
        plan = microbatch_plan(64, 2, world_size)
        assert [chunk for shard in plan for chunk in shard] == expected
        assert len({draw_seed(7, 0, index, 0) for shard in plan for index, _, _ in shard}) == 32
        assert sum(end-start for shard in plan for _, start, end in shard) == 64
    with pytest.raises(ValueError, match='fewer global microbatches'):
        microbatch_plan(64, 16, 8)


def test_single_fit_selection_and_1280_step_schedule():
    torch.set_num_threads(1)
    check_fit_selection(torch.device('cpu'))
    check_fit_budget(torch.device('cpu'))


def test_two_rank_actual_stage_and_fit_with_gloo():
    if dist.is_initialized():
        pytest.skip('use the --worker entry point under torchrun')
    if (importlib.util.find_spec('gptqmodel.quantization.gsq_distributed') is None
            or 'distributed' not in inspect.signature(stage.train_stage_update).parameters):
        pytest.skip('GSQDistributed integration is not present in this worktree yet')
    environment = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1', HIP_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='')
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        environment.pop(key, None)
    completed = subprocess.run(
        [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
         str(Path(__file__).resolve()), '--worker', '--device', 'cpu'],
        cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=240, check=False)
    assert completed.returncode == 0, completed.stdout
    reports = [json.loads(line.partition('=')[2]) for line in completed.stdout.splitlines()
               if line.startswith('GSQ_CORRECTNESS_JSON=')]
    assert len(reports) == 1, completed.stdout
    assert reports[0]['world_size'] == 2
    assert reports[0]['budget']['optimizer_updates'] == 1280
    assert len(reports[0]['updates']) == 2


def worker_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--backend', choices=('auto', 'gloo', 'nccl'), default='auto')
    args = parser.parse_args()
    torch.set_num_threads(1)
    device, rank, world_size, context = initialize_distributed(args.device, args.backend)
    try:
        checks = [check_updates(device, context=context, rank=rank, world_size=world_size, bits=bits)
                  for bits in (2, 3)]
        fit = check_fit_selection(device, context, rank, world_size)
        budget = check_fit_budget(device, context, rank, world_size)
        if rank == 0:
            print('GSQ_CORRECTNESS_JSON=' + json.dumps({
                'world_size': world_size, 'import_path': str(Path(stage.__file__).resolve()),
                'device': str(device), 'backend': dist.get_backend() if world_size > 1 else 'single',
                'hip_version': torch.version.hip, 'updates': checks, 'validation': fit, 'budget': budget}), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    worker_main()
