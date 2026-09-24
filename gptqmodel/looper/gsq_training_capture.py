# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
import shutil
from pathlib import Path
import torch
from ..nn_modules.hooked_linear import HookedLinear


class DiskBackedLlamaCapture:
    """A bounded-memory decoder-input spool with shared Llama metadata."""

    def __init__(self, directory, paths, kwargs, position_ids, attention_mask, shape, dtype):
        self.directory = Path(directory)
        self.paths = tuple(paths)
        self.kwargs = kwargs
        self.position_ids = position_ids
        self.attention_mask = attention_mask
        self.shape = tuple(shape)
        self.dtype = dtype

    def cleanup(self):
        shutil.rmtree(self.directory, ignore_errors=True)


class DiskBackedLlamaDocuments:
    """Load one captured decoder input at a time from a disk spool."""

    offloaded = True

    def __init__(self, capture, kwargs):
        self.capture = capture
        self.kwargs = kwargs
        self.fixed_sequence_length = capture.shape[1]
        self.has_attention_mask = capture.attention_mask is not None

    def __len__(self):
        return len(self.capture.paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(index)
        hidden = torch.load(self.capture.paths[index], map_location='cpu', weights_only=True)
        if hidden.shape != self.capture.shape or hidden.dtype != self.capture.dtype:
            raise ValueError('Disk-backed GSQ capture geometry or dtype changed')
        return hidden, self.kwargs


class DeviceBackedLlamaCapture:
    """A GPU-resident decoder-input spool with per-chunk Llama metadata."""

    def __init__(self, chunks):
        self.chunks = tuple(chunks)
        self.shape = tuple(chunks[0][0].shape[1:])
        self.dtype = chunks[0][0].dtype
        self.length = sum(hidden.shape[0] for hidden, _ in chunks)

    def cleanup(self):
        self.chunks = ()


class DeviceBackedLlamaDocuments:
    """Load one captured decoder input view from retained GPU chunks."""

    offloaded = False

    def __init__(self, capture):
        self.capture = capture
        self.fixed_sequence_length = capture.shape[0]
        self.has_attention_mask = any(kwargs.get('attention_mask') is not None for _, kwargs in capture.chunks)

    def __len__(self):
        return self.capture.length

    @staticmethod
    def _slice(value, start, stop, batch_size):
        if isinstance(value, torch.Tensor):
            return value[start:stop] if value.ndim and value.shape[0] == batch_size else value
        if isinstance(value, tuple):
            return tuple(DeviceBackedLlamaDocuments._slice(item, start, stop, batch_size) for item in value)
        if isinstance(value, list):
            return [DeviceBackedLlamaDocuments._slice(item, start, stop, batch_size) for item in value]
        if isinstance(value, dict):
            return {key: DeviceBackedLlamaDocuments._slice(item, start, stop, batch_size)
                    for key, item in value.items()}
        return value

    def _locate(self, index):
        offset = index
        for hidden, kwargs in self.capture.chunks:
            if offset < hidden.shape[0]:
                return hidden, kwargs, offset
            offset -= hidden.shape[0]
        raise IndexError(index)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(index)
        hidden, kwargs, offset = self._locate(index)
        return hidden[offset:offset+1], self._slice(kwargs, offset, offset+1, hidden.shape[0])

    def iter_hidden_batches(self):
        for hidden, _ in self.capture.chunks:
            yield hidden

    def iter_batches(self):
        for hidden, kwargs in self.capture.chunks:
            yield hidden, kwargs


def prepare_llama_gsq_capture(layer, cache, *, device=None):
    """Copy a pristine block and its exact captured masks/rotary tensors.

    Shared capture runs in inference mode. Neither its tensors nor HookedLinear
    forwards are suitable for attention/block training. Conversion leaves the
    live block and cache untouched. Padding policy is still caller-owned.
    """
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    if not isinstance(layer, LlamaDecoderLayer):
        raise TypeError('Staged capture currently requires a LlamaDecoderLayer')
    disk_backed = isinstance(cache, DiskBackedLlamaCapture)
    device_backed = isinstance(cache, DeviceBackedLlamaCapture)
    if disk_backed:
        if not cache.paths:
            raise ValueError('Staged capture requires nonempty disk-backed inputs')
    elif device_backed:
        if not cache.chunks:
            raise ValueError('Staged capture requires nonempty device-backed inputs')
    else:
        fields = (cache.layer_inputs, cache.layer_input_kwargs, cache.position_ids, cache.attention_masks)
        if not fields[0] or any(len(values) != len(fields[0]) for values in fields):
            raise ValueError('Staged capture requires aligned nonempty input and metadata lists')
    for module in layer.modules():
        if module._forward_hooks or module._forward_pre_hooks:
            raise ValueError('Remove capture hooks before preparing the pristine GSQ training copy')
        if isinstance(module, HookedLinear) and (
                getattr(module, 'online_full_had', False) or getattr(module, 'online_partial_had', False)):
            raise ValueError('Staged capture does not yet preserve online Hadamard transforms')
    device = next(layer.parameters()).device if device is None else torch.device(device)
    with torch.inference_mode(False), torch.no_grad():
        prepared = copy.deepcopy(layer).to(device).eval().requires_grad_(False)
        for name, module in list(prepared.named_modules()):
            if not isinstance(module, HookedLinear):
                continue
            linear = torch.nn.Linear.__new__(torch.nn.Linear)
            torch.nn.Module.__init__(linear)
            linear.in_features, linear.out_features = module.in_features, module.out_features
            linear.weight, linear.bias = module.weight, module.bias
            parent, leaf = name.rsplit('.', 1)
            setattr(prepared.get_submodule(parent), leaf, linear)

        batch_device = torch.device('cpu') if disk_backed else (
            device if device_backed else fields[0][0][0].device)
        if not disk_backed and (batch_device.type != 'cpu' or device.type == 'cpu'):
            batch_device = device
        clones = {}

        def clone(value):
            identity = id(value)
            if identity in clones:
                return clones[identity]
            if isinstance(value, torch.Tensor):
                result = value.detach().to(batch_device).clone()
                clones[identity] = result
                return result
            if isinstance(value, tuple):
                result = tuple(clone(v) for v in value)
                clones[identity] = result
                return result
            if isinstance(value, list):
                result = [clone(v) for v in value]
                clones[identity] = result
                return result
            if isinstance(value, dict):
                result = {k: clone(v) for k, v in value.items()}
                clones[identity] = result
                return result
            return copy.deepcopy(value)

        if disk_backed:
            kwargs = clone(cache.kwargs)
            if kwargs.get('use_cache') or kwargs.get('past_key_values') is not None:
                raise ValueError('Staged capture requires cache-free calibration')
            if kwargs.get('position_embeddings') is None:
                raise ValueError('Staged capture requires the actual captured rotary embeddings')
            kwargs['use_cache'] = False
            if cache.position_ids is not None:
                kwargs['position_ids'] = clone(cache.position_ids)
            kwargs['attention_mask'] = clone(cache.attention_mask)
            return prepared, DiskBackedLlamaDocuments(cache, kwargs)

        if device_backed:
            return prepared, DeviceBackedLlamaDocuments(cache)

        batches = []
        for inputs, kwargs, positions, mask in zip(*fields):
            if len(inputs) != 1 or inputs[0].ndim != 3:
                raise ValueError('Staged capture requires one [batch,tokens,hidden] input per call')
            kwargs = clone(kwargs)
            if kwargs.get('use_cache') or kwargs.get('past_key_values') is not None:
                raise ValueError('Staged capture requires cache-free calibration')
            if kwargs.get('position_embeddings') is None:
                raise ValueError('Staged capture requires the actual captured rotary embeddings')
            kwargs['use_cache'] = False
            if positions is not None:
                kwargs['position_ids'] = clone(positions)
            kwargs['attention_mask'] = clone(mask)
            batches.append((clone(inputs[0]), kwargs))
    return prepared, batches


def quantize_llama_gsq_capture(layer, cache, *, bits, group_size, gsq=None, pack=True, device=None,
                                initialization_cache=None, validation_cache=None):
    """Train/export a shared captured block even when its caller uses inference mode.

    The shared looper still owns installing the result and replaying it before
    capturing the next layer. This function does not mutate that live state.
    """
    from ..quantization.gsq_training import quantize_llama_gsq_block

    with torch.inference_mode(False), torch.enable_grad():
        prepared, batches = prepare_llama_gsq_capture(layer, cache, device=device)
        initialization_batches = None
        validation_batches = None
        if initialization_cache is not None:
            _, initialization_batches = prepare_llama_gsq_capture(layer, initialization_cache, device=device)
        if validation_cache is not None:
            _, validation_batches = prepare_llama_gsq_capture(layer, validation_cache, device=device)
        return quantize_llama_gsq_block(prepared, batches, bits=bits, group_size=group_size,
                                        gsq=gsq, pack=pack,
                                        initialization_batches=initialization_batches,
                                        validation_batches=validation_batches)


def capture_llama_gsq_inputs(
        model, documents, *, layer_index=0, offload_to_cpu=False, offload_directory=None, capture_batch_size=1):
    """Capture actual Llama decoder calls, including masks and rotary state.

    Replays the current model prefix, so previously installed quantized blocks
    participate in downstream capture. Documents must be unpadded. The caller
    controls placement; capture stops before the selected decoder executes.
    """
    import logging
    import time

    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    from .input_cache import InputCache

    if not isinstance(model, LlamaForCausalLM):
        raise TypeError('Staged capture currently requires LlamaForCausalLM')
    if (isinstance(layer_index, bool) or not isinstance(layer_index, int)
            or not 0 <= layer_index < len(model.model.layers)):
        raise ValueError('Invalid Llama layer index')
    if not documents:
        raise ValueError('Staged capture requires nonempty calibration documents')
    if not isinstance(offload_to_cpu, bool):
        raise TypeError('offload_to_cpu must be boolean')
    if isinstance(capture_batch_size, bool) or not isinstance(capture_batch_size, int) or capture_batch_size < 1:
        raise ValueError('capture_batch_size must be a positive integer')
    if offload_directory is not None and not offload_to_cpu:
        raise ValueError('Disk-backed capture requires CPU offloading')
    offload_directory = None if offload_directory is None else Path(offload_directory)
    batched_capture = not offload_to_cpu and capture_batch_size > 1
    if offload_to_cpu or batched_capture:
        lengths = {len(document.get('input_ids', ())) for document in documents}
        if len(lengths) != 1 or any(set(document) != {'input_ids'} for document in documents):
            raise ValueError('Batched GSQ capture requires equal-length input_ids without explicit metadata')
    if offload_directory is not None:
        offload_directory.mkdir(parents=True, exist_ok=False)
    captured = []
    shared_kwargs = None
    progress_at = time.monotonic()+60

    class CaptureComplete(Exception):
        pass

    def capture(_module, args, kwargs):
        nonlocal shared_kwargs
        kwargs = dict(kwargs)
        hidden = args[0] if args else kwargs.pop('hidden_states')
        def detach(value):
            if isinstance(value, torch.Tensor):
                return value.detach()
            if isinstance(value, tuple):
                return tuple(detach(item) for item in value)
            if isinstance(value, list):
                return [detach(item) for item in value]
            if isinstance(value, dict):
                return {key: detach(item) for key, item in value.items()}
            return value

        if offload_to_cpu:
            hidden = hidden.detach().cpu()
            if shared_kwargs is None:
                def cpu(value):
                    if isinstance(value, torch.Tensor):
                        return value.detach().cpu()
                    if isinstance(value, tuple):
                        return tuple(cpu(item) for item in value)
                    if isinstance(value, list):
                        return [cpu(item) for item in value]
                    if isinstance(value, dict):
                        return {key: cpu(item) for key, item in value.items()}
                    return copy.deepcopy(value)

                shared_kwargs = cpu(kwargs)
            kwargs = shared_kwargs
        elif batched_capture:
            hidden = hidden.detach()
            kwargs = detach(kwargs)
        if offload_directory is None:
            captured.append(([hidden], kwargs))
        else:
            path = offload_directory/f'{len(captured):05d}.pt'
            torch.save(hidden, path)
            captured.append(path)
            del hidden
        raise CaptureComplete

    layer = model.model.layers[layer_index]
    handle = layer.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.inference_mode():
            for document_index in range(0, len(documents), capture_batch_size if batched_capture else 1):
                group = documents[document_index:document_index+(capture_batch_size if batched_capture else 1)]
                ids = torch.as_tensor([document['input_ids'] for document in group],
                                      device=model.model.embed_tokens.weight.device)
                if ids.ndim != 2 or not ids.shape[0] or not ids.shape[1]:
                    raise ValueError('Staged capture requires nonempty unpadded documents')
                kwargs = {}
                if not batched_capture:
                    document = group[0]
                    for name in ('attention_mask', 'position_ids'):
                        if name in document:
                            value = torch.as_tensor(document[name], device=ids.device)
                            if value.ndim == 1:
                                value = value.unsqueeze(0)
                            if value.shape != ids.shape or (name == 'attention_mask' and not (value == 1).all()):
                                raise ValueError('Staged capture requires aligned unpadded token metadata')
                            kwargs[name] = value
                count = len(captured)
                try:
                    model(input_ids=ids, use_cache=False, **kwargs)
                except CaptureComplete:
                    pass
                if len(captured) != count+1:
                    raise RuntimeError('Selected Llama decoder was not captured exactly once')
                if time.monotonic() >= progress_at:
                    logging.getLogger(__name__).info(
                        'GSQ capture layer=%d documents=%d/%d offload=%s disk=%s',
                        layer_index,
                        min(document_index+len(group), len(documents)),
                        len(documents),
                        offload_to_cpu,
                        offload_directory is not None,
                    )
                    progress_at = time.monotonic()+60
    except BaseException:
        if offload_directory is not None:
            shutil.rmtree(offload_directory, ignore_errors=True)
        raise
    finally:
        handle.remove()
    if offload_directory is not None:
        values = dict(shared_kwargs)
        positions = values.pop('position_ids', None)
        mask = values.pop('attention_mask', None)
        sample = torch.load(captured[0], map_location='cpu', weights_only=True)
        shape, dtype = sample.shape, sample.dtype
        del sample
        return DiskBackedLlamaCapture(
            offload_directory,
            captured,
            values,
            positions,
            mask,
            shape,
            dtype,
        )
    if batched_capture:
        chunks = []
        for hidden, values in captured:
            values = dict(values)
            if values.get('use_cache') or values.get('past_key_values') is not None:
                raise ValueError('Staged capture requires cache-free calibration')
            if values.get('position_embeddings') is None:
                raise ValueError('Staged capture requires the actual captured rotary embeddings')
            values['use_cache'] = False
            chunks.append((hidden[0], values))
        return DeviceBackedLlamaCapture(chunks)
    inputs, kwargs, positions, masks = [], [], [], []
    for hidden, values in captured:
        values = dict(values)
        inputs.append(hidden)
        positions.append(values.pop('position_ids', None))
        masks.append(values.pop('attention_mask', None))
        kwargs.append(values)
    return InputCache(inputs, kwargs, positions, masks)
