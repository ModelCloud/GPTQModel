# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import hashlib
import os
import time
from pathlib import Path
import torch
from ..quantization.gsq_training_config import GSQTrainingConfig
from .gsq_training_capture import advance_llama_gsq_inputs, capture_llama_gsq_inputs, quantize_llama_gsq_capture


def _checkpoint_documents_digest(documents):
    """Identify the token documents and explicit metadata consumed by staged capture."""
    digest = hashlib.sha256()
    digest.update(len(documents).to_bytes(8, "little"))
    for document in documents:
        if not isinstance(document, dict) or "input_ids" not in document:
            raise ValueError("GSQ checkpointing requires documents with input_ids")
        ids = torch.as_tensor(document["input_ids"], device="cpu")
        if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("GSQ checkpointing requires rank-1 integer input_ids")
        ids = ids.to(torch.int64).contiguous()
        digest.update(ids.numel().to_bytes(8, "little"))
        digest.update(ids.numpy().tobytes())
        for name in ("attention_mask", "position_ids"):
            digest.update(name.encode("ascii"))
            if name not in document:
                digest.update(b"\x00")
                continue
            digest.update(b"\x01")
            value = torch.as_tensor(document[name], device="cpu").detach().contiguous()
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(repr(tuple(value.shape)).encode("ascii"))
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
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
                             checkpoint_directory=None, distributed=None,
                             target_cache='none', activation_checkpointing=False,
                             resume_prefix_directory=None, reuse_capture=False):
    """Quantize selected Llama blocks in place and replay the packed prefix.

    Defaults select every decoder block. This runtime entry point does not write
    a complete checkpoint or replace GPTQModel.quantize dispatch. The caller
    owns placement and supplies unpadded tokenized documents. Optional GPTQ
    and validation documents are captured separately at every block. Completed
    blocks and diagnostics remain available if a later block fails. When a
    checkpoint directory is supplied, completed packed blocks are saved there
    atomically and replayed on a later call with the same model, documents and
    run configuration. Checkpoints contain tensor state and use weights-only
    loading.
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
    if target_cache not in ('none', 'cpu', 'device'):
        raise ValueError('GSQ target cache must be none, cpu or device')
    if not isinstance(activation_checkpointing, bool):
        raise TypeError('GSQ activation_checkpointing must be boolean')
    if not isinstance(reuse_capture, bool):
        raise TypeError('GSQ reuse_capture must be boolean')
    if reuse_capture and capture_directory is not None:
        raise ValueError('Incremental GSQ capture requires CPU or device memory, not a disk spool')
    if isinstance(capture_batch_size, bool) or not isinstance(capture_batch_size, int) or capture_batch_size < 1:
        raise ValueError('capture_batch_size must be a positive integer')
    if capture_directory is not None and not offload_capture:
        raise ValueError('Disk-backed capture requires capture offloading')
    capture_directory = None if capture_directory is None else Path(capture_directory)
    checkpoint_directory = None if checkpoint_directory is None else Path(checkpoint_directory)
    effective = gsq.to_dict()
    primary = distributed is None or distributed.is_primary
    if distributed is not None:
        if not gsq.enabled:
            raise ValueError('Distributed GSQ requires enabled staged training')
        if gsq.batch_size % distributed.world_size or gsq.microbatch_size > gsq.batch_size // distributed.world_size:
            raise ValueError('GSQ global batch must divide world size and local batch must fit a microbatch')
        train_indices = distributed.document_indices(len(documents), gsq.batch_size)
        validation_indices = (None if validation_documents is None else
                              distributed.document_indices(len(validation_documents), gsq.batch_size))
    resume_prefix_directory = None if resume_prefix_directory is None else Path(resume_prefix_directory)
    if resume_prefix_directory is not None and checkpoint_directory is None:
        raise ValueError('Importing a completed prefix requires a destination checkpoint directory')
    if isinstance(bits, dict):
        if not bits:
            raise ValueError('Per-layer staged quantization requires a nonempty width map')
        if any(isinstance(value, bool) or not isinstance(value, int) or value not in (2, 3, 4)
               for value in bits.values()):
            raise ValueError('Staged model quantization supports W2/W3/W4')
        if any(isinstance(key, bool) or not isinstance(key, int) or not 0 <= key < len(model.model.layers)
               for key in bits):
            raise ValueError('Per-layer staged quantization requires decoder layer indices')
    elif isinstance(bits, bool) or not isinstance(bits, int) or bits not in (2, 3, 4):
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
    run = dict(state='running', gsq_training=effective,
               bits=bits if not isinstance(bits, dict) else dict(bits),
               layer_bits=({index: bits if not isinstance(bits, dict) else bits[index] for index in indices}
                           if isinstance(bits, dict) else {index: bits for index in indices}),
               group_size=group_size,
               layer_indices=indices, device=str(device), blocks=[],
               training_documents=len(documents),
               initialization_documents=len(initialization_documents) if initialization_documents is not None else len(documents),
               validation_documents=len(validation_documents) if validation_documents is not None else 0,
               offload_capture=offload_capture,
               capture_storage='disk' if capture_directory is not None else ('cpu' if offload_capture else 'device'),
               deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    run.update(world_size=1 if distributed is None else distributed.world_size,
               target_cache=target_cache, activation_checkpointing=activation_checkpointing,
               reuse_capture=reuse_capture,
               resume_prefix_directory=None if resume_prefix_directory is None else str(resume_prefix_directory))
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
        if distributed is not None:
            signature.update(device=device.type, distributed_world_size=distributed.world_size)
        if target_cache != 'none' or activation_checkpointing:
            signature.update(target_cache=target_cache, activation_checkpointing=activation_checkpointing)
        source_prefixes = (_checkpoint_prefix_digests(model, indices) if distributed is None else
                           distributed.primary_call(lambda: _checkpoint_prefix_digests(model, indices)))
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
    # Keep global identities in the checkpoint; capture only this rank's shard.
    if distributed is not None:
        if initialization_documents is None:
            initialization_documents = documents
        documents = [documents[index] for index in train_indices]
        if validation_documents is not None:
            validation_documents = [validation_documents[index] for index in validation_indices]
        if capture_directory is not None:
            capture_directory = capture_directory/f'rank-{distributed.rank:02d}'
    carried = None
    active_captures = ()
    try:
        for index in indices:
            import logging

            block_started = time.monotonic()
            logging.getLogger(__name__).info('Staged model block %d/%d', index+1, len(layers))
            run['current_layer'] = index
            checkpoint = (checkpoint_directory/f'layer-{index:02d}.pt'
                          if checkpoint_directory is not None else None)
            def select_checkpoint(checkpoint=checkpoint):
                if checkpoint is not None and checkpoint.exists():
                    return str(checkpoint), False
                if (checkpoint is not None and resume_prefix_directory is not None
                        and (resume_prefix_directory/checkpoint.name).is_file()):
                    return str(resume_prefix_directory/checkpoint.name), True
                return None, False

            source_checkpoint, imported = (select_checkpoint() if distributed is None else
                                           distributed.primary_call(select_checkpoint))
            if source_checkpoint is not None:
                source_checkpoint = Path(source_checkpoint)
                if carried is not None:
                    for captured in carried:
                        if captured is not None:
                            captured.cleanup()
                    carried = None
                layer_signature = dict(signature, source_prefix=source_prefixes[index])
                def read_checkpoint(source_checkpoint=source_checkpoint, layer_signature=layer_signature,
                                    imported=imported, index=index, checkpoint=checkpoint):
                    saved = torch.load(source_checkpoint, map_location='cpu', weights_only=True)
                    expected = dict(layer_signature)
                    actual = dict(saved.get('signature', {})) if isinstance(saved, dict) else {}
                    if imported:
                        # Explicit migration may change execution placement but must
                        # preserve the model, corpora, bits and optimizer recipe.
                        for key in ('device', 'distributed_world_size', 'target_cache', 'activation_checkpointing'):
                            expected.pop(key, None)
                            actual.pop(key, None)
                    if (not isinstance(saved, dict) or actual != expected
                            or saved.get('layer_index') != index
                            or not isinstance(saved.get('state_dict'), dict)
                            or not isinstance(saved.get('record'), dict)):
                        raise ValueError(f'GSQ block checkpoint does not match this run: {source_checkpoint}')
                    if imported:
                        saved['signature'] = layer_signature
                        saved['record'] = dict(saved['record'], imported_from=str(source_checkpoint))
                        temporary = checkpoint.with_suffix('.pt.tmp')
                        try:
                            torch.save(saved, temporary)
                            os.replace(temporary, checkpoint)
                        finally:
                            temporary.unlink(missing_ok=True)
                    return saved

                saved = read_checkpoint() if distributed is None else distributed.primary_call(read_checkpoint)
                packed = _restore_packed_layer(
                    layers[index], saved['state_dict'], names=names, bits=run['layer_bits'][index],
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
            if carried is None:
                cache = capture_llama_gsq_inputs(model, documents, layer_index=index, **capture_options)
                initialization_cache = None
                validation_cache = None
            else:
                cache, initialization_cache, validation_cache = carried
            active_captures = (cache, initialization_cache, validation_cache)
            try:
                if initialization_documents is not None and carried is None:
                    def capture_initialization(index=index, capture_options=capture_options):
                        nonlocal initialization_cache
                        init_options = dict(capture_options)
                        if capture_directory is not None:
                            init_options['offload_directory'] = capture_directory/f'layer-{index:02d}'/'gptq'
                        initialization_cache = capture_llama_gsq_inputs(
                            model, initialization_documents, layer_index=index, **init_options)

                    if distributed is None:
                        capture_initialization()
                    else:
                        # Propagate capture failures without broadcasting the
                        # full calibration activation cache to other ranks.
                        distributed.primary_call(capture_initialization)
                if validation_documents is not None and gsq.enabled and carried is None:
                    val_options = dict(capture_options)
                    if capture_directory is not None:
                        val_options['offload_directory'] = capture_directory/f'layer-{index:02d}'/'validation'
                    validation_cache = capture_llama_gsq_inputs(
                        model, validation_documents, layer_index=index, **val_options)
                active_captures = (cache, initialization_cache, validation_cache)
                packed, result = quantize_llama_gsq_capture(layers[index], cache,
                                                           bits=run['layer_bits'][index],
                                                           group_size=group_size,
                                                           gsq=gsq, pack=True, device=device,
                                                           initialization_cache=initialization_cache,
                                                           validation_cache=validation_cache,
                                                           distributed=distributed, target_cache=target_cache,
                                                           activation_checkpointing=activation_checkpointing)
            finally:
                if not reuse_capture:
                    for captured in (cache, initialization_cache, validation_cache):
                        if captured is not None and hasattr(captured, 'cleanup'):
                            captured.cleanup()
            record = dict(layer_index=index, stages={
                name: {key: value for key, value in stage.items() if key not in ('weights', 'scales')}
                for name, stage in result['stages'].items()},
                initializer_metadata=result['initializer_metadata'])
            record['block_seconds'] = time.monotonic()-block_started
            if checkpoint is not None:
                def write_checkpoint(checkpoint=checkpoint, layer_signature=layer_signature,
                                     index=index, packed=packed, record=record):
                    temporary = checkpoint.with_suffix('.pt.tmp')
                    try:
                        torch.save(dict(signature=layer_signature, layer_index=index,
                                        state_dict={name: value.detach().cpu().clone()
                                                    for name, value in packed.state_dict().items()},
                                        record=record), temporary)
                        os.replace(temporary, checkpoint)
                    finally:
                        temporary.unlink(missing_ok=True)

                if distributed is None:
                    write_checkpoint()
                else:
                    distributed.primary_call(write_checkpoint)
                if primary:
                    logging.getLogger(__name__).info('Saved packed GSQ block %d/%d to %s (%.1fs)',
                                                      index+1, len(layers), checkpoint, record['block_seconds'])
            packed = packed.to(device).eval()
            # Installation precedes the next capture: its inputs therefore
            # include both the selected assignments and stored-scale rounding.
            layers[index] = packed
            run['blocks'].append(record)
            if reuse_capture and index+1 in indices:
                carried = tuple(None if captured is None else advance_llama_gsq_inputs(packed, captured, device=device)
                                for captured in active_captures)
            else:
                carried = None
                for captured in active_captures:
                    if captured is not None:
                        captured.cleanup()
                active_captures = ()
            del result, cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        run['state'] = 'complete'
    except Exception as error:
        run.update(state='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for captured in active_captures:
            if captured is not None:
                captured.cleanup()
    return run


def dynamic_layer_bits(quantize_config):
    """Resolve a `dynamic` map into ``{decoder_layer_index: bits}`` or ``None``.

    A budget-constrained bit-allocation search produces one width per decoder
    block. Accept exactly that shape: entries must name whole blocks under
    ``model.layers.<n>`` and may only set ``bits``. Anything else (a different
    module, or a non-width override such as ``group_size``/``sym``) is refused
    rather than silently ignored, because staged GSQ trains one scalar grid per
    block.
    """
    import re

    dynamic = getattr(quantize_config, 'dynamic', None)
    if not dynamic:
        return None
    pattern = re.compile(r'^model\.layers\.(\d+)$')
    resolved = {}
    for module_name, overrides in dynamic.items():
        match = pattern.match(module_name)
        if match is None:
            raise ValueError(
                'Staged GSQ accepts per-layer bit allocation only as `model.layers.<n>` entries; '
                f'got {module_name!r}')
        if not isinstance(overrides, dict) or set(overrides)-{'bits'}:
            raise ValueError(
                'Staged GSQ per-layer allocation may only override `bits`; '
                f'got {module_name!r} -> {overrides!r}')
        bits = overrides.get('bits')
        if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (2, 3, 4):
            raise ValueError(f'Staged GSQ per-layer allocation requires W2/W3/W4; got {module_name!r} -> {bits!r}')
        index = int(match.group(1))
        if resolved.get(index, bits) != bits:
            raise ValueError(f'Staged GSQ per-layer allocation repeats layer {index} with different widths')
        resolved[index] = bits
    return resolved or None


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
    layer_bits = dict(run.get('layer_bits') or {})
    widths = set(layer_bits.values())
    uniform = None if not layer_bits else (next(iter(widths)) if len(widths) == 1 else None)
    if len(widths) > 1 and not layer_bits:
        raise ValueError('Mixed staged export requires per-layer bit widths')
    qcfg = GPTQConfig(bits=uniform if uniform is not None else
                      (run['bits'] if not isinstance(run['bits'], dict) else min(widths)),
                      group_size=run['group_size'], sym=True, desc_act=False,
                      act_group_aware=False, format=FORMAT.GPTQ_V2, offload_to_disk=False,
                      device=str(next(model.parameters()).device))
    if len(widths) > 1:
        # Load-time module construction resolves per-module bits from `dynamic`
        # (utils/model.py), so a non-uniform export must publish the assignment
        # there or every layer would be rebuilt at the top-level width.
        base = run['bits'] if not isinstance(run['bits'], dict) else min(widths)
        qcfg.bits = base
        qcfg.dynamic = {
            f'model.layers.{index}.{name}': {'bits': int(width)}
            for index, width in sorted(layer_bits.items())
            for name in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                         'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
            if int(width) != base
        } or None
        qcfg.meta['gsq_layer_bits'] = {str(k): int(v) for k, v in sorted(layer_bits.items())}
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
    # Non-uniform per-layer bit allocation (e.g. an RCO search result) is the
    # only dynamic override this path can honor: it changes widths alone, and
    # the staged exporter publishes the same assignment back as `dynamic`.
    layer_bits = dynamic_layer_bits(qcfg)
    for name in ('rotation', 'adapter', 'gptaq', 'foem', 'mock_quantization',
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
    if layer_bits is None:
        run = quantize_llama_gsq_prepared(wrapper, prepared, gsq=staged)
    else:
        documents = staged_documents_from_prepared(prepared)
        run = quantize_llama_gsq_model(wrapper.model, documents,
                                       bits=layer_bits,
                                       group_size=qcfg.group_size, gsq=staged)
        finalize_llama_gsq_wrapper(wrapper, run)
    wrapper.quantize_config.gsq_training = staged
    return {'gsq_training': run['blocks']}
