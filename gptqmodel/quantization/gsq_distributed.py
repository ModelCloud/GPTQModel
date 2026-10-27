# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Explicit distributed collectives for staged GSQ reconstruction training."""

import hashlib
from collections.abc import Mapping

import torch
import torch.distributed as dist

_REDUCE_CHUNK_ELEMENTS = 16 * 1024 * 1024


class GSQDistributed:
    """Coordinate an already initialized Gloo or NCCL process group.

    Ranks are relative to ``group``; its rank zero owns initialization. The
    caller owns process-group lifetime, CUDA device selection and optimizer
    steps. Communication methods must be called by every member in the same
    order. Constructing this context never initializes distributed execution
    or changes a global random generator.
    """

    def __init__(self, group=None, device=None):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError('GSQDistributed requires torch.distributed to be initialized')
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError('GSQDistributed requires membership in the process group')
        backend = dist.get_backend(group)
        if backend not in ('gloo', 'nccl'):
            raise ValueError('GSQDistributed requires a Gloo or NCCL process group')
        if device is None:
            device = torch.device('cuda', torch.cuda.current_device()) if backend == 'nccl' else torch.device('cpu')
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if self.device.type not in ('cpu', 'cuda') or (backend == 'nccl' and self.device.type != 'cuda'):
            raise ValueError('GSQDistributed requires a CPU/CUDA device, and NCCL requires CUDA')
        # Gloo can also stage CUDA tensors through bounded CPU transfers.
        self._collective_device = self.device if backend == 'nccl' else torch.device('cpu')
        self.is_primary = self.rank == 0

    def _global_rank(self, rank):
        return rank if self.group is None else dist.get_global_rank(self.group, rank)

    def _broadcast_object(self, value, rank=0):
        objects = [value]
        dist.broadcast_object_list(objects, src=self._global_rank(rank), group=self.group,
                                   device=self._collective_device)
        return objects[0]

    def _agree(self, operation, metadata=None, error=None):
        """Check structure before shape-dependent collectives, without pickling tensors.

        Only a fixed-size fingerprint crosses the wire on the successful path.
        On failure, broadcast the first failing rank's diagnostic so all ranks
        raise the same exception before entering a tensor-payload collective.
        """
        digest = hashlib.sha256(repr((operation, metadata)).encode()).digest()
        signature = [int(error is not None)] + [
            int.from_bytes(digest[start:start + 8], 'little', signed=True) for start in (0, 8)
        ]
        local = torch.tensor(signature, dtype=torch.int64, device=self._collective_device)
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local, group=self.group)
        signatures = torch.stack(gathered).cpu().tolist()
        failed = next((rank for rank, value in enumerate(signatures) if value[0]), None)
        if failed is not None:
            source_operation, message = self._broadcast_object((operation, error), rank=failed)
            raise RuntimeError(f'GSQ distributed {source_operation} failed on rank {failed}: {message}')
        if any(value[1:] != signatures[0][1:] for value in signatures[1:]):
            source_operation = self._broadcast_object(operation)
            raise RuntimeError(f'GSQ distributed {source_operation} metadata mismatch across ranks')

    @staticmethod
    def _tensor_metadata(tensor):
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.is_quantized or tensor.device.type not in ('cpu', 'cuda')):
            raise ValueError('GSQ distributed tensors must be dense, strided CPU/CUDA tensors')
        return tuple(tensor.shape), tensor.dtype

    def _broadcast_tensor(self, tensor):
        # Byte transport also supports packed integer and FP8 buffers. Neither
        # object serialization nor the backend needs to understand their dtype.
        # W2 candidate banks expand a tiny level vector with zero strides.
        # Broadcast its compact storage: copying into the expanded view would
        # fail, and materializing it would needlessly replicate the whole bank.
        target = tensor.detach()
        for dimension, (size, stride) in enumerate(zip(tensor.shape, tensor.stride())):
            if size > 1 and stride == 0:
                target = target.narrow(dimension, 0, 1)
        contiguous = target.contiguous()
        flat = contiguous.reshape(-1).view(torch.uint8)
        for start in range(0, flat.numel(), _REDUCE_CHUNK_ELEMENTS):
            destination = flat[start:start + _REDUCE_CHUNK_ELEMENTS]
            payload = destination.to(self._collective_device)
            dist.broadcast(payload, src=self._global_rank(0), group=self.group)
            if payload.device != destination.device:
                destination.copy_(payload)
        if not target.is_contiguous():
            target.copy_(contiguous)

    @torch.no_grad()
    def sync_quantizers(self, quantizers):
        """Broadcast all parameters and buffers, including nonpersistent buffers.

        Names, shapes, dtypes and trainability must agree on every rank. Mapping
        insertion order need not agree. Call before the initial hard evaluation.
        """
        tensors, metadata, error = [], [], None
        try:
            if not isinstance(quantizers, Mapping) or not all(isinstance(name, str) for name in quantizers):
                raise TypeError('GSQ quantizers must be a mapping with string names')
            for name in sorted(quantizers):
                quantizer = quantizers[name]
                if not isinstance(quantizer, torch.nn.Module):
                    raise TypeError('GSQ quantizers must be torch modules')
                metadata.append(('module', name))
                for kind, values in (('parameter', quantizer.named_parameters()),
                                     ('buffer', quantizer.named_buffers())):
                    for tensor_name, tensor in sorted(values):
                        tensor_meta = self._tensor_metadata(tensor)
                        expanded = tuple(dimension for dimension, (size, stride)
                                         in enumerate(zip(tensor.shape, tensor.stride())) if size > 1 and stride == 0)
                        metadata.append((name, kind, tensor_name, tensor_meta, tensor.requires_grad, expanded))
                        tensors.append(tensor)
        except Exception as exc:  # noqa: BLE001 - every rank must receive validation failures
            error = f'{type(exc).__name__}: {exc}'
        self._agree('sync_quantizers', metadata, error)
        for tensor in tensors:
            self._broadcast_tensor(tensor)

    @staticmethod
    def _validate_count(count):
        if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= 2**53:
            raise ValueError('GSQ distributed loss requires a positive integer element count <= 2**53')

    def _loss_tensor(self, value, *, double=False):
        if isinstance(value, torch.Tensor):
            if value.ndim != 0 or not value.is_floating_point():
                raise ValueError('GSQ distributed loss must be a floating scalar')
            dtype = torch.float64 if double or value.dtype == torch.float64 else torch.float32
            return value.detach().to(device=self._collective_device, dtype=dtype)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError('GSQ distributed loss must be a floating scalar')
        return torch.tensor(value, dtype=torch.float64 if double else torch.float32,
                            device=self._collective_device)

    def _weighted_loss(self, loss, count):
        total = torch.tensor(count, dtype=torch.float64, device=self._collective_device)
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=self.group)
        fraction = count / total
        result = loss * fraction.to(loss.dtype)
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=self.group)
        return result, fraction

    def _check_finite(self, finite):
        invalid = (~finite).to(dtype=torch.int32, device=self._collective_device)
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=self.group)
        if invalid.item():
            raise RuntimeError('GSQ distributed objective or gradient is nonfinite')

    @torch.no_grad()
    def reduce_update(self, optimizer, reported, count):
        """Reduce accumulated local-element-mean gradients before ``step``.

        Each rank contributes ``count / global_count`` of its mean gradient
        and reported mean loss. FP16/BF16 gradients reduce in FP32, in chunks
        bounded by 16M elements; FP32 and FP64 retain their arithmetic dtype.
        The result is written back to each parameter's existing gradient dtype.
        An identically missing gradient is skipped; inconsistent missing
        gradients, parameter metadata or invalid inputs fail on every rank.
        Nonfinite inputs/results likewise raise on all ranks before returning.

        Optimizer parameter order must identify the same parameters on every
        rank; shapes alone cannot identify reordered equal-shaped parameters.
        No optimizer step or state update is performed here.
        """
        gradients, metadata, error = [], [], None
        loss = None
        try:
            self._validate_count(count)
            loss = self._loss_tensor(reported)
            for group in optimizer.param_groups:
                parameters = []
                for parameter in group['params']:
                    parameter_meta = self._tensor_metadata(parameter)
                    if parameter.device != self.device or parameter.dtype not in (
                            torch.float16, torch.bfloat16, torch.float32, torch.float64):
                        raise ValueError('GSQ optimizer parameters must be FP16/BF16/FP32/FP64 on the local device')
                    gradient = parameter.grad
                    gradient_meta = None if gradient is None else self._tensor_metadata(gradient)
                    if gradient is not None:
                        if gradient_meta != parameter_meta or gradient.device != parameter.device:
                            raise ValueError('GSQ gradients must match their parameter shape, dtype and device')
                        gradients.append(gradient)
                    parameters.append((parameter_meta, parameter.requires_grad, gradient_meta))
                metadata.append(parameters)
        except Exception as exc:  # noqa: BLE001 - synchronize errors before tensor collectives
            error = f'{type(exc).__name__}: {exc}'
        self._agree('reduce_update', (metadata, None if loss is None else loss.dtype), error)
        result, fraction = self._weighted_loss(loss, count)
        finite = torch.isfinite(loss) & torch.isfinite(result)
        for gradient in gradients:
            # Noncontiguous gradients need a contiguous copy in their native
            # dtype, but never a full-size promoted FP32 copy of BF16 logits.
            flat = gradient.detach().reshape(-1)
            dtype = torch.float32 if gradient.dtype in (torch.float16, torch.bfloat16) else gradient.dtype
            weight = fraction.to(dtype)
            for start in range(0, flat.numel(), _REDUCE_CHUNK_ELEMENTS):
                destination = flat[start:start + _REDUCE_CHUNK_ELEMENTS]
                reduced = destination.to(device=self._collective_device, dtype=dtype, copy=True)
                finite.logical_and_(torch.isfinite(reduced).all())
                reduced.mul_(weight)
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=self.group)
                rounded = reduced.to(gradient.dtype)
                finite.logical_and_(torch.isfinite(rounded).all())
                destination.copy_(rounded)
            if not gradient.is_contiguous():
                gradient.copy_(flat.reshape(gradient.shape))
        self._check_finite(finite)
        return result.to(self.device)

    @torch.no_grad()
    def mean_loss(self, value, count):
        """Return a globally element-weighted validation mean as a Python float."""
        loss, error = None, None
        try:
            self._validate_count(count)
            loss = self._loss_tensor(value, double=True)
        except Exception as exc:  # noqa: BLE001 - synchronize errors before tensor collectives
            error = f'{type(exc).__name__}: {exc}'
        self._agree('mean_loss', error=error)
        result, _ = self._weighted_loss(loss, count)
        self._check_finite(torch.isfinite(loss) & torch.isfinite(result))
        return result.item()

    def seed(self, seed):
        """Return a deterministic per-rank sampling seed without changing RNGs."""
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError('GSQ distributed sampling seed must be an integer')
        return (seed + self.rank) % (1 << 63)

    def barrier(self):
        dist.barrier(group=self.group)

    def local_call(self, fn):
        """Run rank-local work, synchronizing failures without sharing its result."""
        result, error = None, None
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - propagate any rank's local capture/replay failure
            error = f'{type(exc).__name__}: {exc}'
        self._agree('local_call', error=error)
        return result

    def primary_call(self, fn):
        """Run ``fn`` only on group rank zero and broadcast its detached result.

        Supported trees contain dict/list/tuple containers, primitive leaves
        (None, bool, int, float, str, bytes), and dense CPU/CUDA tensors. Dict
        keys must be primitives. All tensor leaves, including CPU records, are
        returned on ``self.device``. Only descriptors use object broadcast;
        tensor payloads use bounded byte broadcasts. Callback/serialization
        errors are reported identically on every rank.
        """
        self._agree('primary_call')
        tensors = []

        def describe(value):
            if isinstance(value, torch.Tensor):
                shape, dtype = self._tensor_metadata(value)
                index = len(tensors)
                tensors.append(value.detach().resolve_conj().resolve_neg())
                return 'tensor', index, shape, dtype
            if value is None or isinstance(value, (bool, int, float, str, bytes)):
                return 'value', value, None, None
            if isinstance(value, dict):
                if any(key is not None and not isinstance(key, (bool, int, float, str, bytes)) for key in value):
                    raise TypeError('GSQ primary_call dict keys must be primitives')
                return 'dict', [(key, describe(item)) for key, item in value.items()], None, None
            if isinstance(value, (list, tuple)):
                return 'tuple' if isinstance(value, tuple) else 'list', [describe(item) for item in value], None, None
            raise TypeError(f'Unsupported GSQ primary_call result type: {type(value).__name__}')

        envelope = None
        if self.is_primary:
            try:
                envelope = ('ok', describe(fn()))
            except Exception as exc:  # noqa: BLE001 - propagate the primary callback's failure
                envelope = ('error', f'{type(exc).__name__}: {exc}')
                tensors.clear()
        status, description = self._broadcast_object(envelope)
        if status == 'error':
            raise RuntimeError(f'GSQ distributed primary_call failed on rank 0: {description}')

        def restore(node):
            kind, payload, shape, dtype = node
            if kind == 'value':
                return payload
            if kind == 'tensor':
                index = payload
                if self.is_primary:
                    tensors[index] = tensors[index].to(self.device).contiguous()
                else:
                    tensors.append(torch.empty(shape, dtype=dtype, device=self.device))
                return tensors[index]
            if kind == 'dict':
                return {key: restore(item) for key, item in payload}
            values = [restore(item) for item in payload]
            return tuple(values) if kind == 'tuple' else values

        result, error = None, None
        try:
            result = restore(description)
        except Exception as exc:  # noqa: BLE001 - other ranks must not enter a failed allocation's broadcast
            error = f'{type(exc).__name__}: {exc}'
        self._agree('primary_call tensor allocation', error=error)
        for tensor in tensors:
            self._broadcast_tensor(tensor)
        return result

    def document_indices(self, length, global_batch_size):
        """Assign each global batch's contiguous rank chunk, without dropping data.

        Dataset length must be divisible by the positive global batch size,
        which must be divisible by world size. All ranks must supply the same
        arguments. Empty datasets return an empty list on every rank.
        """
        error = None
        try:
            if isinstance(length, bool) or not isinstance(length, int) or length < 0:
                raise ValueError('GSQ dataset length must be a nonnegative integer')
            if (isinstance(global_batch_size, bool) or not isinstance(global_batch_size, int)
                    or global_batch_size <= 0 or global_batch_size % self.world_size):
                raise ValueError('GSQ global batch size must be positive and divisible by world size')
            if length % global_batch_size:
                raise ValueError('GSQ dataset length must be divisible by global batch size; dropping data is unsupported')
        except Exception as exc:  # noqa: BLE001 - validate sharding collectively
            error = f'{type(exc).__name__}: {exc}'
        self._agree('document_indices', (length, global_batch_size), error)
        local_batch_size = global_batch_size // self.world_size
        return [index for start in range(0, length, global_batch_size)
                for index in range(start + self.rank * local_batch_size, start + (self.rank + 1) * local_batch_size)]
