"""Sequential model installation for the experimental staged Llama GSQ path."""

import torch

from .gsq_training_capture import capture_llama_gsq_inputs, quantize_llama_gsq_capture
from ..quantization.gsq_training_config import GSQTrainingConfig


def quantize_llama_gsq_model(model, documents, *, bits, group_size, gsq=None, layer_indices=None):
    """Quantize selected Llama blocks in place and replay the packed prefix.

    Defaults select every decoder block. This runtime entry point does not write
    a complete checkpoint or replace GPTQModel.quantize dispatch. The caller
    owns placement and supplies unpadded tokenized documents. Completed blocks
    and diagnostics remain available if a later block fails.
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
    effective = gsq.to_dict()
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (2, 3, 4):
        raise ValueError('Staged model quantization supports W2/W3/W4')
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError('Staged model quantization requires a positive contiguous group size')
    if not documents:
        raise ValueError('Staged model quantization requires calibration documents')
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
               deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    if not hasattr(model, 'gsq_training_runs'):
        model.gsq_training_runs = []
    model.gsq_training_runs.append(run)
    try:
        for index in indices:
            import logging

            logging.getLogger(__name__).info('Staged model block %d/%d', index+1, len(layers))
            run['current_layer'] = index
            cache = capture_llama_gsq_inputs(model, documents, layer_index=index)
            packed, result = quantize_llama_gsq_capture(layers[index], cache, bits=bits, group_size=group_size,
                                                       gsq=gsq, pack=True, device=device)
            packed = packed.to(device).eval()
            # Installation precedes the next capture: its inputs therefore
            # include both the selected assignments and stored-scale rounding.
            layers[index] = packed
            stages = {name: {key: value for key, value in stage.items() if key not in ('weights', 'scales')}
                      for name, stage in result['stages'].items()}
            run['blocks'].append(dict(layer_index=index, stages=stages,
                                       initializer_metadata=result['initializer_metadata']))
            del result, cache
        run['state'] = 'complete'
    except Exception as error:
        run.update(state='failed', error_type=type(error).__name__, error=str(error))
        raise
    return run


def save_llama_gsq_model(model, run, output, *, tokenizer, source_model):
    """Convert a complete staged model to FP16 and use the public GPTQ writer.

    Saving is explicit because it changes the floating runtime tensors to FP16.
    Partial-layer runs require a dynamic checkpoint configuration and are not
    handled by this uniform-format exporter.
    """
    import json
    from pathlib import Path

    from ..models.definitions.llama import LlamaQModel
    from ..nn_modules.qlinear.torch import TorchLinear
    from ..quantization.config import FORMAT, GPTQConfig

    output, source_model = Path(output), Path(source_model)
    if output.exists() or not source_model.is_dir():
        raise ValueError('Staged export requires a fresh destination and existing dense source directory')
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
    model.model.rotary_emb = LlamaRotaryEmbedding(
        model.config, device='cpu').to(model.model.embed_tokens.weight.device).eval()
    wrapper = LlamaQModel(model=model, quantized=True, quantize_config=qcfg, tokenizer=tokenizer,
                          qlinear_kernel=TorchLinear, model_local_path=str(source_model))
    wrapper.save(str(output))
    export = dict(training=run, runtime_dtype='float16', format='gptq_v2',
                  source_model=str(source_model.resolve()))
    (output/'gsq_training_run.json').write_text(json.dumps(export, indent=2)+'\n')
    return wrapper
