# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Independent distributed GSQ arithmetic and collective-failure regressions."""

import random
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gptqmodel.quantization import gsq_distributed
from gptqmodel.quantization.gsq_distributed import GSQDistributed


def _same_error(context, fn, match):
    with pytest.raises(RuntimeError, match=match) as caught:
        fn()
    messages = [None] * context.world_size
    dist.all_gather_object(messages, str(caught.value), group=context.group)
    assert len(set(messages)) == 1, messages
    context.barrier()


def _weighted_update(context):
    """Compare microbatch accumulation with a separate concatenated objective."""
    device = context.device
    for dtype in (torch.float32, torch.float64):
        tolerance = 1e-12 if dtype == torch.float64 else 1e-6
        inputs = torch.arange(33, dtype=dtype, device=device).reshape(11, 3) / 17 - .7
        targets = torch.arange(22, dtype=dtype, device=device).reshape(11, 2) / 13 - .2
        initial = torch.arange(6, dtype=dtype, device=device).reshape(2, 3).t() / 19 + .123456789123
        weight = torch.nn.Parameter(initial.clone())
        bias = torch.nn.Parameter(torch.tensor([.125, -.375], dtype=dtype, device=device))
        unused = torch.nn.Parameter(torch.ones(2, dtype=dtype, device=device))
        optimizer = torch.optim.SGD([{'params': [weight]}, {'params': [bias, unused]}], lr=.07)
        reference_weight = initial.clone().requires_grad_()
        reference_bias = bias.detach().clone().requires_grad_()
        reference_optimizer = torch.optim.SGD([reference_weight, reference_bias], lr=.07)
        start, stop = (0, 3) if context.rank == 0 else (3, 11)
        count = (stop - start) * targets.shape[1]
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            reported = torch.zeros((), dtype=dtype, device=device)
            # Unequal local totals (6 versus 16 elements) and uneven microbatches.
            for left, right in ((start, start + 1), (start + 1, stop)):
                loss = (inputs[left:right] @ weight + bias - targets[left:right]).square().mean()
                fraction = (right - left) * targets.shape[1] / count
                (loss * fraction).backward()
                reported += loss.detach() * fraction
            assert not weight.grad.is_contiguous()
            expected = (inputs @ reference_weight + reference_bias - targets).square().mean()
            expected.backward()
            actual = context.reduce_update(optimizer, reported, count)
            assert actual.dtype == dtype and actual.device == device
            assert not actual.requires_grad
            torch.testing.assert_close(actual, expected.detach(), rtol=tolerance, atol=tolerance)
            for parameter, reference in ((weight, reference_weight), (bias, reference_bias)):
                torch.testing.assert_close(parameter.grad, reference.grad, rtol=tolerance, atol=tolerance)
            assert unused.grad is None
            optimizer.step()
            reference_optimizer.step()
            torch.testing.assert_close(weight, reference_weight, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(bias, reference_bias, rtol=tolerance, atol=tolerance)


def _low_precision_chunks(context, *, large=False):
    device = context.device
    for dtype in (torch.float16, torch.bfloat16):
        size = gsq_distributed._REDUCE_CHUNK_ELEMENTS + 19 if large else 259
        chunk_size = gsq_distributed._REDUCE_CHUNK_ELEMENTS if large else 33
        parameter = torch.nn.Parameter(torch.zeros(size, dtype=dtype, device=device))
        # These input gradients and their weighted sum are exactly representable.
        parameter.grad = torch.full_like(parameter, 1.25 if context.rank == 0 else 3.5)
        optimizer = torch.optim.SGD([parameter], lr=.125)
        calls = []
        original = dist.all_reduce

        def observe(tensor, *args, calls=calls, original=original, **kwargs):
            calls.append((tensor.numel(), tensor.dtype))
            return original(tensor, *args, **kwargs)

        with patch.object(gsq_distributed, '_REDUCE_CHUNK_ELEMENTS', chunk_size), \
                patch.object(dist, 'all_reduce', side_effect=observe):
            actual = context.reduce_update(optimizer, torch.tensor(1.5 + context.rank, device=device),
                                           1 if context.rank == 0 else 3)
        # Oracle builds the four per-element contributions independently.
        oracle = torch.tensor([1.25, 3.5, 3.5, 3.5], device=device).mean().to(dtype)
        torch.testing.assert_close(parameter.grad, oracle.expand_as(parameter), rtol=0, atol=0)
        torch.testing.assert_close(actual, torch.tensor(2.25, device=device), rtol=0, atol=0)
        payloads = [(count, kind) for count, kind in calls if count > 1]
        assert payloads == [(min(chunk_size, size - offset), torch.float32)
                            for offset in range(0, size, chunk_size)]


def _sync_and_primary(context):
    from gptqmodel.quantization.gsq_training import GSQScalarTrainingModule

    device = context.device
    quantizers = {}
    for bits in (2, 3):
        weight = torch.arange(32, dtype=torch.float32, device=device).reshape(4, 8) / 32 + context.rank
        scales = torch.full((4, 1), .5 + context.rank, device=device)
        noise = torch.full((4 if bits == 2 else 5, 4, 8), .1 + context.rank, device=device)
        quantizers[str(bits)] = GSQScalarTrainingModule(weight, scales, 8, bits=bits, noise=noise,
                                                       logits_dtype=torch.float32)
        quantizers[str(bits)].register_buffer('transposed', torch.full((2, 3), 2. + context.rank, device=device).t(),
                                             persistent=False)
    # Mapping order is not a tensor identity and may differ between ranks.
    if context.rank:
        quantizers = dict(reversed(list(quantizers.items())))
    candidates = quantizers['2'].candidates
    assert 0 in candidates.stride()
    pointer = candidates.untyped_storage().data_ptr()
    storage_bytes = candidates.untyped_storage().nbytes()
    root_state = context.primary_call(lambda: {
        name: {**dict(module.named_parameters()), **dict(module.named_buffers())}
        for name, module in quantizers.items()
    })
    context.sync_quantizers(quantizers)
    assert candidates.untyped_storage().data_ptr() == pointer
    assert candidates.untyped_storage().nbytes() == storage_bytes
    assert storage_bytes < candidates.numel() * candidates.element_size()
    for name, module in quantizers.items():
        for key, tensor in (*module.named_parameters(), *module.named_buffers()):
            torch.testing.assert_close(tensor, root_state[name][key], rtol=0, atol=0)

    calls = []
    object_broadcast = dist.broadcast_object_list

    def assert_descriptor(value):
        assert not isinstance(value, torch.Tensor), 'tensor payload was pickled'
        if isinstance(value, dict):
            for key, item in value.items():
                assert_descriptor(key)
                assert_descriptor(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                assert_descriptor(item)

    def observe(objects, *args, **kwargs):
        assert_descriptor(objects)
        return object_broadcast(objects, *args, **kwargs)

    def initialize():
        calls.append(context.rank)
        assert context.is_primary
        # QK fitting needs autograd inside the primary callback.
        weight = torch.tensor([2., 3.], device=device, requires_grad=True)
        trained = weight.square()
        trained.sum().backward()
        assert torch.equal(weight.grad, 2 * weight.detach())
        return {'weights': trained, 'records': [torch.arange(6).reshape(2, 3).t(),
                                               (None, True, 17, 1.25, 'ok', b'bytes')],
                'expanded': torch.arange(3.).reshape(3, 1).expand(3, 4),
                'empty': torch.empty((0, 2), dtype=torch.bfloat16),
                'scalar': torch.tensor(9, dtype=torch.int64),
                'complex': torch.tensor([1 + 2j]).conj(), 'containers': ({}, [], ())}

    with patch.object(dist, 'broadcast_object_list', side_effect=observe):
        result = context.primary_call(initialize)
    assert calls == ([0] if context.is_primary else [])
    assert result['records'][1] == (None, True, 17, 1.25, 'ok', b'bytes')
    assert result['containers'] == ({}, [], ())
    assert isinstance(result['records'], list) and isinstance(result['records'][1], tuple)
    expected = {
        'weights': torch.tensor([4., 9.]), 'records': torch.arange(6).reshape(2, 3).t(),
        'expanded': torch.arange(3.).reshape(3, 1).expand(3, 4),
        'empty': torch.empty((0, 2), dtype=torch.bfloat16), 'scalar': torch.tensor(9),
        'complex': torch.tensor([1 - 2j]),
    }
    for key, value in expected.items():
        actual = result['records'][0] if key == 'records' else result[key]
        assert actual.device == device and not actual.requires_grad and actual.grad_fn is None
        torch.testing.assert_close(actual, value.to(device), rtol=0, atol=0)

    def fail():
        assert context.is_primary
        raise ValueError('primary initializer exploded')

    _same_error(context, lambda: context.primary_call(fail), 'ValueError: primary initializer exploded')
    _same_error(context, lambda: context.primary_call(lambda: {'bad': object()}), 'Unsupported.*object')
    assert context.primary_call(lambda: None) is None
    assert context.primary_call(lambda: {'recovered': True}) == {'recovered': True}


def _validation_and_documents(context):
    # Rank zero alone prefers epoch 0; global element weighting prefers epoch 1.
    per_element = [torch.tensor([0.] + [4.] * 9, dtype=torch.float64),
                   torch.tensor([3.] + [1.] * 9, dtype=torch.float64),
                   torch.tensor([5.] + [2.] * 9, dtype=torch.float64)]
    actual = []
    for values in per_element:
        shard = values[:1] if context.rank == 0 else values[1:]
        actual.append(context.mean_loss(float(shard.mean()), shard.numel()))
    expected = [float(values.mean()) for values in per_element]
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=1e-6, atol=1e-6)
    assert min(range(len(actual)), key=actual.__getitem__) == 1
    assert context.document_indices(12, 4) == ([0, 1, 4, 5, 8, 9] if context.rank == 0 else [2, 3, 6, 7, 10, 11])
    assert context.document_indices(0, 4) == []
    for length, batch in ((10, 4), (12, 3), (12, 0), (-1, 4), (True, 4), (12, False)):
        _same_error(context, lambda length=length, batch=batch: context.document_indices(length, batch),
                    'document_indices failed')
    _same_error(context, lambda: context.document_indices(8 + context.rank * 4, 4), 'metadata mismatch')
    before = torch.random.get_rng_state().clone()
    python_before = random.getstate()
    seed = context.seed(37)
    assert seed == context.seed(37)
    assert context.seed((1 << 63) - 1) >= 0
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    assert random.getstate() == python_before
    samples = torch.rand(8, generator=torch.Generator().manual_seed(seed))
    by_rank = [None] * context.world_size
    dist.all_gather_object(by_rank, samples, group=context.group)
    assert not torch.equal(by_rank[0], by_rank[1])
    with pytest.raises(TypeError, match='integer'):
        context.seed(True)


def _failures(context):
    device = context.device
    for field in ('loss', 'gradient'):
        for nonfinite in (float('nan'), float('inf'), -float('inf')):
            parameter = torch.nn.Parameter(torch.ones(5, device=device))
            parameter.grad = torch.ones_like(parameter)
            loss = torch.tensor(1., device=device)
            if context.rank == 1:
                if field == 'loss':
                    loss.fill_(nonfinite)
                else:
                    parameter.grad[2] = nonfinite
            optimizer = torch.optim.SGD([parameter], lr=.5)

            def update(optimizer=optimizer, loss=loss):
                context.reduce_update(optimizer, loss, 3 + context.rank)
                optimizer.step()

            _same_error(context, update, 'nonfinite')
            torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0, atol=0)
    for invalid in (0, -1, 1.5, True, 2**54):
        parameter = torch.nn.Parameter(torch.ones(5, device=device))
        parameter.grad = torch.ones_like(parameter)
        optimizer = torch.optim.SGD([parameter], lr=.5)
        count = invalid if context.rank else 3
        _same_error(context, lambda optimizer=optimizer, count=count:
                    context.reduce_update(optimizer, torch.tensor(1., device=device), count),
                    'positive integer element count')
        _same_error(context, lambda count=count: context.mean_loss(1., count), 'positive integer element count')
    _same_error(context, lambda: context.mean_loss(float('nan') if context.rank else 1., 3), 'nonfinite')
    _same_error(context, lambda: context.mean_loss(torch.ones(2) if context.rank else 1., 3), 'floating scalar')

    for mismatch in ('missing', 'shape', 'dtype', 'groups', 'sparse', 'loss_shape'):
        dtype = torch.float64 if context.rank and mismatch == 'dtype' else torch.float32
        parameter = torch.nn.Parameter(torch.ones(6 if context.rank and mismatch == 'shape' else 5,
                                                 device=device, dtype=dtype))
        parameter.grad = torch.ones_like(parameter)
        loss = torch.tensor(1., device=device)
        if context.rank:
            if mismatch == 'missing':
                parameter.grad = None
            elif mismatch == 'sparse':
                parameter.grad = parameter.grad.to_sparse()
            elif mismatch == 'loss_shape':
                loss = torch.ones(2, device=device)
        groups = [{'params': [parameter]}]
        if mismatch == 'groups' and context.rank:
            groups.append({'params': []})
        optimizer = torch.optim.SGD(groups, lr=.1)
        _same_error(context, lambda optimizer=optimizer, loss=loss: context.reduce_update(optimizer, loss, 3),
                    'dense, strided' if mismatch == 'sparse' else
                    'floating scalar' if mismatch == 'loss_shape' else 'metadata mismatch')
    for mismatch in ('shape', 'dtype', 'name', 'expanded', 'module'):
        module = torch.nn.Module()
        tensor = torch.ones(1, 3, device=device)
        if mismatch == 'expanded':
            tensor = tensor.expand(2, 3)
            if context.rank:
                tensor = tensor.clone()
        elif mismatch == 'shape' and context.rank:
            tensor = torch.ones(1, 4, device=device)
        elif mismatch == 'dtype' and context.rank:
            tensor = tensor.double()
        module.register_buffer('other' if mismatch == 'name' and context.rank else 'buffer', tensor)
        quantizers = {'weight': object() if mismatch == 'module' and context.rank else module}
        _same_error(context, lambda quantizers=quantizers: context.sync_quantizers(quantizers),
                    'torch modules' if mismatch == 'module' else 'metadata mismatch')
    assert context.mean_loss(2., 3) == 2.


def _worker(rank, world_size, rendezvous, backend, case):
    torch.set_num_threads(1)
    device = torch.device('cuda', rank) if backend == 'nccl' else torch.device('cpu')
    if backend == 'nccl':
        torch.cuda.set_device(device)
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=90))
    try:
        if case == 'subgroup':
            group = dist.new_group([1, 2], backend='gloo', timeout=timedelta(seconds=60))
            if rank == 0:
                with pytest.raises(ValueError, match='membership'):
                    GSQDistributed(group=group)
            else:
                context = GSQDistributed(group=group)
                assert context.rank == rank - 1 and context.world_size == 2
                assert context.primary_call(lambda: rank) == 1
                _weighted_update(context)
                context.barrier()
                dist.destroy_process_group(group)
            dist.barrier()
            return
        context = GSQDistributed(device=device)
        assert context.rank == rank and context.world_size == world_size
        assert context.is_primary == (rank == 0)
        if case in ('update', 'gpu'):
            _weighted_update(context)
            _low_precision_chunks(context, large=case == 'gpu')
        if case in ('sync', 'gpu'):
            _sync_and_primary(context)
        if case in ('validation', 'gpu'):
            _validation_and_documents(context)
        if case in ('failures', 'gpu'):
            _failures(context)
        context.barrier()
    finally:
        dist.destroy_process_group()


def _spawn(tmp_path, case, *, backend='gloo', world_size=2):
    rendezvous = (tmp_path / 'process-group').as_uri()
    mp.spawn(_worker, args=(world_size, rendezvous, backend, case), nprocs=world_size, join=True)


def test_requires_explicit_initialized_group(monkeypatch):
    assert not dist.is_initialized()
    for key, value in (('RANK', '0'), ('LOCAL_RANK', '0'), ('WORLD_SIZE', '2')):
        monkeypatch.setenv(key, value)
    before = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='requires torch.distributed to be initialized'):
        GSQDistributed()
    assert not dist.is_initialized()
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)


@pytest.mark.timeout(180)
@pytest.mark.parametrize('case', ['update', 'sync', 'validation', 'failures', 'subgroup'])
@pytest.mark.skipif(not dist.is_gloo_available(), reason='Gloo is unavailable')
def test_gloo_primitives(tmp_path, case):
    _spawn(tmp_path, case, world_size=3 if case == 'subgroup' else 2)


@pytest.mark.cuda
@pytest.mark.timeout(240)
@pytest.mark.skipif(not dist.is_nccl_available() or not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason='Two GPUs and NCCL/RCCL are required')
def test_two_gpu_nccl_primitives(tmp_path):
    _spawn(tmp_path, 'gpu', backend='nccl')
