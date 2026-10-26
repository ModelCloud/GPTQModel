# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import hashlib
import os
from pathlib import Path
import torch
from ..quantization.gsq_training_config import GSQTrainingConfig
from .gsq_training_capture import capture_llama_gsq_inputs, quantize_llama_gsq_capture


def _checkpoint_documents_digest(documents):
    """Identify the exact token documents consumed by staged capture."""
    digest = hashlib.sha256()
    digest.update(len(documents).to_bytes(8, 'little'))
    for document in documents:
        if not isinstance(document, dict) or 'input_ids' not in document:
            raise ValueError('GSQ checkpointing requires documents with input_ids')
        ids = torch.as_tensor(document['input_ids'], device='cpu')
        if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('GSQ checkpointing requires rank-1 integer input_ids')
        ids = ids.to(torch.int64).contiguous()
        digest.update(ids.numel().to_bytes(8, 'little'))
        digest.update(ids.numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_module_digest(module):
    """Identify a source block without retaining a second full model copy."""
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'Unsupported GSQ checkpoint state value: {name}')
        value = value.detach().to(device='cpu').contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(repr(tuple(value.shape)).encode('ascii'))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_prefix_digests(model, layer_indices):
    """Identify the model weights that produced each selected layer's inputs."""
    requested = set(layer_indices)
    maximum = max(requested)
    digest = hashlib.sha256()
    digest.update(b'model.embed_tokens\0')
    digest.update(bytes.fromhex(_checkpoint_module_digest(model.model.embed_tokens)))
    result = {}
    for index, layer in enumerate(model.model.layers[:maximum+1]):
        digest.update(index.to_bytes(8, 'little'))
        digest.update(bytes.fromhex(_checkpoint_module_digest(layer)))
        if index in requested:
            result[index] = digest.hexdigest()
    return result


def _restore_packed_layer(layer, state_dict, *, names, bits, group_size, device):
    """Rebuild packed modules from the caller's block and tensor-only state."""
    import copy

    from ..nn_modules.qlinear.torch import TorchLinear
    from ..utils.backend import BACKEND

    packed = copy.deepcopy(layer).cpu()
    for name in names:
        source = packed.get_submodule(name)
        linear = TorchLinear(bits=bits, group_size=group_size, sym=True, desc_act=False,
                             in_features=source.in_features, out_features=source.out_features,
                             bias=source.bias is not None, backend=BACKEND.TORCH)
        linear = linear.to(dtype=source.weight.dtype)
        parent, leaf = name.rsplit('.', 1)
        setattr(packed.get_submodule(parent), leaf, linear)
    packed.load_state_dict(state_dict, strict=True)
    for name in names:
        packed.get_submodule(name).post_init()
    return packed.to(device).eval()


def staged_documents_from_prepared(prepared):
    """Convert standard prepared token batches into unpadded staged documents.

    Only contiguous valid spans are accepted: dropping holes would change token
    positions and attention semantics. Token weighting and other metadata need
    explicit staged support rather than being silently discarded here.
    """
    documents = []
    for batch in prepared:
        if set(batch)-{'input_ids', 'attention_mask'}:
            raise ValueError('Unsupported prepared calibration metadata for staged GSQ')
        ids = batch.get('input_ids')
        if (not isinstance(ids, torch.Tensor) or ids.ndim != 2
                or ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError('Staged GSQ requires prepared rank-2 integer token IDs')
        mask = batch.get('attention_mask', torch.ones_like(ids))
        if (not isinstance(mask, torch.Tensor) or mask.shape != ids.shape
                or not ((mask == 0) | (mask == 1)).all()):
            raise ValueError('Staged GSQ requires an aligned binary padding mask')
        for row, keep in zip(ids.cpu(), mask.cpu().bool()):
            indices = keep.nonzero().flatten()
            if not len(indices) or int(indices[-1]-indices[0]+1) != len(indices):
                raise ValueError('Staged GSQ requires a nonempty contiguous valid token span')
            documents.append({'input_ids': row[keep].tolist()})
    if not documents:
        raise ValueError('Staged GSQ requires nonempty prepared calibration')
    return documents


def quantize_llama_gsq_model(model, documents, *, bits, group_size, gsq=None, layer_indices=None,
                             offload_capture=False, capture_directory=None, capture_batch_size=1,
                             initialization_documents=None, validation_documents=None,
                             checkpoint_directory=None):
    """Quantize selected Llama blocks in place and replay the packed prefix.

    Defaults select every decoder block. This runtime entry point does not write
    a complete checkpoint or replace GPTQModel.quantize dispatch. The caller
    owns placement and supplies unpadded tokenized documents. Optional GPTQ
    and validation documents are captured separately at every block. Completed
    blocks and diagnostics remain available if a later block fails. When a
    checkpoint directory is supplied, completed packed blocks are saved there
    atomically and replayed on a later call with the same run configuration.
    Only load checkpoints created by a trusted local run: they contain pickled
    PyTorch modules.
    """
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    if not isinstance(model, LlamaForCausalLM) or model.training:
        raise ValueError('Staged model quantization requires an eval-mode LlamaForCausalLM')
    if gsq is None:
        gsq = GSQTrainingConfig()
    elif isinstance(gsq, dict):
        gsq = GSQTrainingConfig(**gsq)
    if not isinstance(gsq, GSQTrainingConfig):
        raise TypeError('Staged model quantization requires GSQTrainingConfig, a dictionary or None')
    if not isinstance(offload_capture, bool):
        raise TypeError('offload_capture must be boolean')
    if isinstance(capture_batch_size, bool) or not isinstance(capture_batch_size, int) or capture_batch_size < 1:
        raise ValueError('capture_batch_size must be a positive integer')
    if capture_directory is not None and not offload_capture:
        raise ValueError('Disk-backed capture requires capture offloading')
    capture_directory = None if capture_directory is None else Path(capture_directory)
    checkpoint_directory = None if checkpoint_directory is None else Path(checkpoint_directory)
    effective = gsq.to_dict()
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (2, 3, 4):
        raise ValueError('Staged model quantization supports W2/W3/W4')
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError('Staged model quantization requires a positive contiguous group size')
    if not documents:
        raise ValueError('Staged model quantization requires calibration documents')
    if initialization_documents is not None and not initialization_documents:
        raise ValueError('Staged model quantization requires nonempty GPTQ initialization documents')
    if validation_documents is not None and not validation_documents:
        raise ValueError('Staged model quantization requires nonempty validation documents')
    if validation_documents is not None and not gsq.enabled:
        raise ValueError('Validation documents require enabled staged GSQ')
    layers = model.model.layers
    indices = list(range(len(layers))) if layer_indices is None else list(layer_indices)
    if (not indices or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(layers) for i in indices)
            or indices != sorted(set(indices))):
        raise ValueError('Selected layers must be unique, valid and ordered')
    devices = {parameter.device for parameter in model.parameters()}
    if len(devices) != 1 or next(iter(devices)).type == 'meta':
        raise ValueError('Staged model quantization currently requires one materialized model device')
    device = next(iter(devices))
    names = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
             'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
    for index in indices:
        for name in names:
            module = layers[index].get_submodule(name)
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f'Layer {index} {name} is already quantized or unsupported')
            if module.in_features % group_size:
                raise ValueError('Staged model quantization requires complete contiguous groups')
    run = dict(state='running', gsq_training=effective, bits=bits, group_size=group_size,
               layer_indices=indices, device=str(device), blocks=[],
               training_documents=len(documents),
               initialization_documents=len(initialization_documents) if initialization_documents is not None else len(documents),
               validation_documents=len(validation_documents) if validation_documents is not None else 0,
               offload_capture=offload_capture,
               capture_storage='disk' if capture_directory is not None else ('cpu' if offload_capture else 'device'),
               deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    if not hasattr(model, 'gsq_training_runs'):
        model.gsq_training_runs = []
    model.gsq_training_runs.append(run)
    signature = None
    source_prefixes = None
    if checkpoint_directory is not None:
        signature = dict(bits=bits, group_size=group_size, gsq_training=effective,
                         layer_indices=indices, model_config=model.config.to_dict(),
                         training_documents=_checkpoint_documents_digest(documents),
                         initialization_documents=_checkpoint_documents_digest(
                             documents if initialization_documents is None else initialization_documents),
                         validation_documents=(None if validation_documents is None else
                                               _checkpoint_documents_digest(validation_documents)),
                         capture_batch_size=capture_batch_size,
                         attention_implementation=getattr(model.config, '_attn_implementation', None),
                         device=str(device),
                         deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        source_prefixes = _checkpoint_prefix_digests(model, indices)
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
    try:
        for index in indices:
            import logging

            logging.getLogger(__name__).info('Staged model block %d/%d', index+1, len(layers))
            run['current_layer'] = index
            checkpoint = (checkpoint_directory/f'layer-{index:02d}.pt'
                          if checkpoint_directory is not None else None)
            if checkpoint is not None and checkpoint.exists():
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                layer_signature = dict(signature, source_prefix=source_prefixes[index])
                if (not isinstance(saved, dict) or saved.get('signature') != layer_signature
                        or saved.get('layer_index') != index
                        or not isinstance(saved.get('state_dict'), dict)
                        or not isinstance(saved.get('record'), dict)):
                    raise ValueError(f'GSQ block checkpoint does not match this run: {checkpoint}')
                packed = _restore_packed_layer(
                    layers[index], saved['state_dict'], names=names, bits=bits,
                    group_size=group_size, device=device)
                from ..nn_modules.qlinear.torch import TorchLinear

                if any(not isinstance(packed.get_submodule(name), TorchLinear) for name in names):
                    raise ValueError(f'GSQ block checkpoint is not fully packed: {checkpoint}')
                layers[index] = packed
                run['blocks'].append(saved['record'])
                del saved
                logging.getLogger(__name__).info('Resumed packed GSQ block %d/%d', index+1, len(layers))
                continue
            layer_signature = (dict(signature, source_prefix=source_prefixes[index])
                               if checkpoint is not None else None)
            capture_options = {}
            if offload_capture:
                capture_options['offload_to_cpu'] = True
            if capture_directory is not None:
                capture_options['offload_directory'] = capture_directory/f'layer-{index:02d}'/'train'
            if not offload_capture and capture_batch_size != 1:
                capture_options['capture_batch_size'] = capture_batch_size
            cache = capture_llama_gsq_inputs(model, documents, layer_index=index, **capture_options)
            initialization_cache = None
            validation_cache = None
            try:
                if initialization_documents is not None:
                    init_options = dict(capture_options)
                    if capture_directory is not None:
                        init_options['offload_directory'] = capture_directory/f'layer-{index:02d}'/'gptq'
                    initialization_cache = capture_llama_gsq_inputs(
                        model, initialization_documents, layer_index=index, **init_options)
                if validation_documents is not None and gsq.enabled:
                    val_options = dict(capture_options)
                    if capture_directory is not None:
                        val_options['offload_directory'] = capture_directory/f'layer-{index:02d}'/'validation'
                    validation_cache = capture_llama_gsq_inputs(
                        model, validation_documents, layer_index=index, **val_options)
                packed, result = quantize_llama_gsq_capture(layers[index], cache, bits=bits, group_size=group_size,
                                                           gsq=gsq, pack=True, device=device,
                                                           initialization_cache=initialization_cache,
                                                           validation_cache=validation_cache)
            finally:
                for captured in (cache, initialization_cache, validation_cache):
                    if captured is not None and hasattr(captured, 'cleanup'):
                        captured.cleanup()
            record = dict(layer_index=index, stages={
                name: {key: value for key, value in stage.items() if key not in ('weights', 'scales')}
                for name, stage in result['stages'].items()},
                initializer_metadata=result['initializer_metadata'])
            if checkpoint is not None:
                temporary = checkpoint.with_suffix('.pt.tmp')
                try:
                    torch.save(dict(signature=layer_signature, layer_index=index,
                                    state_dict={name: value.detach().cpu().clone()
                                                for name, value in packed.state_dict().items()},
                                    record=record), temporary)
                    os.replace(temporary, checkpoint)
                finally:
                    temporary.unlink(missing_ok=True)
                logging.getLogger(__name__).info('Saved packed GSQ block %d/%d to %s',
                                                  index+1, len(layers), checkpoint)
            packed = packed.to(device).eval()
            # Installation precedes the next capture: its inputs therefore
            # include both the selected assignments and stored-scale rounding.
            layers[index] = packed
            run['blocks'].append(record)
            del result, cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        run['state'] = 'complete'
    except Exception as error:
        run.update(state='failed', error_type=type(error).__name__, error=str(error))
        raise
    return run


def prepare_llama_gsq_export(model, run):
    """Finalize a complete staged model without creating a wrapper or writing files.

    Returns its matching quantization configuration. A public lifecycle caller
    can retain its existing wrapper and install this configuration only after
    successful preparation. This converts floating runtime tensors to FP16.
    """
    from ..nn_modules.qlinear.torch import TorchLinear
    from ..quantization.config import FORMAT, GPTQConfig

    if (run['state'] != 'complete' or run['layer_indices'] != list(range(len(model.model.layers)))
            or not any(record is run for record in getattr(model, 'gsq_training_runs', []))):
        raise ValueError('Uniform staged export requires this model’s completed all-layer run')
    names = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
             'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
    if any(not isinstance(layer.get_submodule(name), TorchLinear)
           for layer in model.model.layers for name in names):
        raise ValueError('Uniform staged export requires all seven packed projections in every layer')
    qcfg = GPTQConfig(bits=run['bits'], group_size=run['group_size'], sym=True, desc_act=False,
                       act_group_aware=False, format=FORMAT.GPTQ_V2, offload_to_disk=False,
                       device=str(next(model.parameters()).device))
    qcfg.meta['gsq_training'] = run['gsq_training']
    qcfg.meta['gsq_training_recipe'] = 'experimental_staged_llama_v1'
    model.half()
    # RoPE inverse frequencies are nonpersistent. Casting their FP32 values to
    # FP16 would change the live runtime while reload reconstructs them from
    # config in FP32. Rebuild the same canonical rotary state as a fresh load.
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    # The public loader constructs these frequencies on CPU before transfer;
    # constructing directly on CUDA can differ by FP32 ulps in pow/division.
    model.model.rotary_emb = LlamaRotaryEmbedding(model.config).to(
        model.model.embed_tokens.weight.device).eval()
    return qcfg


def finalize_llama_gsq_wrapper(wrapper, run):
    """Install completed staged export state on the caller's existing wrapper.

    The prepared configuration describes the actual uniform packed format;
    caller-owned model path and tokenizer identities remain on the wrapper.
    """
    import copy

    from ..nn_modules.qlinear.torch import TorchLinear

    requested = wrapper.quantize_config.to_dict()
    caller_meta = copy.deepcopy(wrapper.quantize_config.meta)
    config = prepare_llama_gsq_export(wrapper.model, run)
    caller_meta.update(config.meta)
    caller_meta['gsq_requested_quantization'] = requested
    config.meta = caller_meta
    wrapper.quantize_config = config
    wrapper.qlinear_kernel = TorchLinear
    wrapper.quantized = True
    wrapper.gsq_training_run = run
    return wrapper


def quantize_llama_gsq_prepared(wrapper, prepared, *, gsq):
    """Execute staged fitting and finalization from validated prepared inputs.

    Public dispatch owns configuration compatibility and device preparation.
    This boundary consumes already prepared token batches and preserves the
    existing wrapper. On failure it retains the model's partial run diagnostics.
    """
    documents = staged_documents_from_prepared(prepared)
    run = quantize_llama_gsq_model(wrapper.model, documents,
                                   bits=wrapper.quantize_config.bits,
                                   group_size=wrapper.quantize_config.group_size, gsq=gsq)
    finalize_llama_gsq_wrapper(wrapper, run)
    return run


def quantize_llama_gsq_public(wrapper, *, calibration, tokenizer, backend,
                              calibration_concat_size, calibration_sort, calibration_data_min_length,
                              calibration_concat_separator, unsupported):
    """Experimental public GPTQ dispatch for a uniform materialized Llama model.

    GSQTrainingConfig owns the staged prior and optimization recipe. Additional
    ordinary GPTQ transforms and recovery processors are not silently applied.
    """
    from ..quantization.config import FORMAT, METHOD
    from ..utils.backend import BACKEND, normalize_backend

    qcfg = wrapper.quantize_config
    if wrapper.quantized or calibration is None:
        raise ValueError('Staged GSQ requires an unquantized model and calibration')
    if qcfg.method != METHOD.GPTQ or qcfg.format not in (FORMAT.GPTQ, FORMAT.GPTQ_V2):
        raise ValueError('Public staged GSQ requires GPTQ scalar export')
    if qcfg.gsq_training.initializer not in ('gptq', 'rtn'):
        raise ValueError('Public staged GSQ requires a GPTQ or RTN initializer')
    if qcfg.bits not in (2, 3, 4) or not qcfg.sym or qcfg.desc_act or qcfg.act_group_aware:
        raise ValueError('Public staged GSQ requires symmetric W2/W3/W4 without activation ordering')
    if any(value is not None for value in unsupported.values()):
        raise ValueError('Unsupported additional public quantization options for staged GSQ')
    for name in ('dynamic', 'rotation', 'adapter', 'gptaq', 'foem', 'mock_quantization',
                 'static_groups', 'adjacent_model', 'adaptive_clipping', 'lm_head', 'preprocessors',
                 'smoother', 'offload_to_disk'):
        if getattr(qcfg, name, None):
            raise ValueError(f'Public staged GSQ does not yet support {name}')
    if getattr(getattr(qcfg, 'adaptive_damping', None), 'enabled', False):
        raise ValueError('Public staged GSQ does not yet support adaptive damping')
    if qcfg.pack_dtype != torch.int32:
        raise ValueError('Public staged GSQ currently requires int32 packed storage')
    if normalize_backend(backend, quant_method=METHOD.GPTQ) not in (None, BACKEND.AUTO, BACKEND.GPTQ_TORCH):
        raise ValueError('Public staged GSQ currently requires the Torch packing backend')
    devices = {p.device for p in wrapper.model.parameters()}
    if len(devices) != 1 or next(iter(devices)).type not in ('cpu', 'cuda'):
        raise ValueError('Public staged GSQ requires a materialized model on one CPU/CUDA device')
    if wrapper.model.config._attn_implementation != 'eager':
        raise ValueError('Public staged GSQ requires eager attention')
    if tokenizer is not None:
        wrapper.tokenizer = tokenizer
    prepared = wrapper.prepare_dataset(
        calibration, calibration_dataset_concat_size=calibration_concat_size,
        calibration_dataset_sort=calibration_sort, batch_size=1,
        calibration_data_min_length=calibration_data_min_length,
        calibration_concat_separator=calibration_concat_separator)
    staged = qcfg.gsq_training
    run = quantize_llama_gsq_prepared(wrapper, prepared, gsq=staged)
    wrapper.quantize_config.gsq_training = staged
    return {'gsq_training': run['blocks']}
