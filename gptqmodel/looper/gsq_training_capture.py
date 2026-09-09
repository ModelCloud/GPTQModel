"""Convert shared Llama capture state into autograd-capable staged GSQ inputs."""

import copy

import torch

from ..nn_modules.hooked_linear import HookedLinear


def prepare_llama_gsq_capture(layer, cache, *, device=None):
    """Copy a pristine block and its exact captured masks/rotary tensors.

    Shared capture runs in inference mode. Neither its tensors nor HookedLinear
    forwards are suitable for attention/block training. Conversion leaves the
    live block and cache untouched. Padding policy is still caller-owned.
    """
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    if not isinstance(layer, LlamaDecoderLayer):
        raise TypeError('Staged capture currently requires a LlamaDecoderLayer')
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

        def clone(value):
            if isinstance(value, torch.Tensor):
                return value.detach().to(device).clone()
            if isinstance(value, tuple):
                return tuple(clone(v) for v in value)
            if isinstance(value, list):
                return [clone(v) for v in value]
            if isinstance(value, dict):
                return {k: clone(v) for k, v in value.items()}
            return copy.deepcopy(value)

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


def quantize_llama_gsq_capture(layer, cache, *, bits, group_size, gsq=None, pack=True, device=None):
    """Train/export a shared captured block even when its caller uses inference mode.

    The shared looper still owns installing the result and replaying it before
    capturing the next layer. This function does not mutate that live state.
    """
    from ..quantization.gsq_training import quantize_llama_gsq_block

    with torch.inference_mode(False), torch.enable_grad():
        prepared, batches = prepare_llama_gsq_capture(layer, cache, device=device)
        return quantize_llama_gsq_block(prepared, batches, bits=bits, group_size=group_size,
                                        gsq=gsq, pack=pack)


def fit_llama_awq_gsq_capture(layer, cache, initializers, *, group_size, epochs, seed=7,
                              device=None, scale_dtype=torch.float16, **training):
    """Fit/pack supplied AWQ initializers using the transformed pre-clip teacher.

    The caller must copy the layer after AWQ scaling and before clipping, retain
    the corresponding full block cache, and supply (codes, scales, zeros) for all
    seven projections. Installing/replaying the packed block remains caller-owned.
    """
    from ..quantization.gsq_training import fit_llama_stages
    from ..quantization.gsq_training_affine import pack_llama_affine_stages

    with torch.inference_mode(False), torch.enable_grad():
        prepared, batches = prepare_llama_gsq_capture(layer, cache, device=device)
        destination = next(prepared.parameters()).device
        values = {name: tuple(value.detach().to(destination).clone() for value in tensors)
                  for name, tensors in initializers.items()}
        fitted, records = fit_llama_stages(
            prepared, values, batches, bits=4, group_size=group_size, epochs=epochs, seed=seed,
            affine_initializers=True, reinitialize_mlp=False, **training)
        packed = pack_llama_affine_stages(fitted, records, group_size=group_size, scale_dtype=scale_dtype)
    return packed, dict(initializer='provided_awq', stages=records)


def capture_llama_gsq_inputs(model, documents, *, layer_index=0):
    """Capture actual Llama decoder calls, including masks and rotary state.

    Replays the current model prefix, so previously installed quantized blocks
    participate in downstream capture. Documents must be unpadded. The caller
    controls placement; capture stops before the selected decoder executes.
    """
    from .input_cache import InputCache
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    if not isinstance(model, LlamaForCausalLM):
        raise TypeError('Staged capture currently requires LlamaForCausalLM')
    if isinstance(layer_index, bool) or not isinstance(layer_index, int) or not 0 <= layer_index < len(model.model.layers):
        raise ValueError('Invalid Llama layer index')
    if not documents:
        raise ValueError('Staged capture requires nonempty calibration documents')
    captured = []

    class CaptureComplete(Exception):
        pass

    def capture(_module, args, kwargs):
        kwargs = dict(kwargs)
        hidden = args[0] if args else kwargs.pop('hidden_states')
        captured.append(([hidden], kwargs))
        raise CaptureComplete

    layer = model.model.layers[layer_index]
    handle = layer.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.inference_mode():
            for document in documents:
                ids = torch.as_tensor(document['input_ids'], device=model.model.embed_tokens.weight.device)
                if ids.ndim == 1:
                    ids = ids.unsqueeze(0)
                if ids.ndim != 2 or ids.shape[0] != 1 or not ids.shape[1]:
                    raise ValueError('Staged capture requires one nonempty unpadded document per call')
                kwargs = {}
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
    finally:
        handle.remove()
    inputs, kwargs, positions, masks = [], [], [], []
    for hidden, values in captured:
        inputs.append(hidden)
        positions.append(values.pop('position_ids', None))
        masks.append(values.pop('attention_mask', None))
        kwargs.append(values)
    return InputCache(inputs, kwargs, positions, masks)
