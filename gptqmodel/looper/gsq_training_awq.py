"""Staged affine fitting at AWQ's scaled-teacher and packing boundaries."""

import torch

from .gsq_training_capture import prepare_llama_gsq_capture


PROJECTIONS = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
               'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')


def capture_awq_staged_teacher(processor, layer, modules, fallback_names):
    """Capture the scaled, pre-clipping teacher before AWQ overwrites weights."""
    if set(modules) != set(PROJECTIONS) or fallback_names:
        raise ValueError('Staged AWQ GSQ requires all seven Llama projections without fallback')
    if any(module.state.get('tp_pad_info') for module in modules.values()):
        raise ValueError('Staged AWQ GSQ does not support tensor-parallel padding')
    if layer.self_attn.config._attn_implementation != 'eager':
        raise ValueError('Staged AWQ GSQ requires eager attention')
    return prepare_llama_gsq_capture(layer, processor.inputs_cache)


def refine_awq_staged_layer(processor, teacher_capture, modules, *, layer_index):
    """Fit all stages and prepare every payload before replacing AWQ metadata."""
    from ..quantization.gsq_scalar import affine_codes
    from ..quantization.gsq_training import fit_llama_stages
    from ..quantization.gsq_training_affine import prepare_affine_export

    teacher, batches = teacher_capture
    device = next(teacher.parameters()).device
    config = processor.qcfg.gsq_training
    options = config.training_kwargs()
    options.pop('initializer')
    initializers = {}
    with torch.inference_mode(False), torch.enable_grad():
        for name, named in modules.items():
            named.stream_sync()
            scales = named.state['q_scales'].to(device).clone()
            zeros = named.state['q_zeros'].to(device).clone()
            weight = named.state['wq'].to(device).clone()
            groups = torch.arange(weight.shape[1], device=device)//processor.qcfg.group_size
            codes = affine_codes(weight, scales, zeros, groups, 4, packing='awq_gemm', scale_dtype=scales.dtype)
            initializers[name] = (codes, scales, zeros)
        _, records = fit_llama_stages(teacher, initializers, batches, bits=4,
                                     group_size=processor.qcfg.group_size, affine_initializers=True,
                                     reinitialize_mlp=False, **options)
        payloads = {}
        for name, named in modules.items():
            stage = name if name in records else ('attention' if name.startswith('self_attn.') else 'mlp')
            key = 'weight' if stage == name else name+'.weight'
            record = records[stage]
            linear = processor.resolve_quant_source_module(named)
            groups = torch.arange(linear.in_features, device=device)//processor.qcfg.group_size
            exported = prepare_affine_export(record['codes'][key], record['scales'][key], record['zeros'][key],
                                             groups, bits=4, packing='awq_gemm',
                                             scale_dtype=(initializers[name][1].dtype if initializers[name][1].dtype
                                                          in (torch.float16, torch.bfloat16) else torch.float16))
            weight = exported['weight'].to(device=linear.weight.device, dtype=linear.weight.dtype)
            recovered = affine_codes(weight, exported['scales'].to(weight.device),
                                     exported['zeros'].to(weight.device), groups.to(weight.device),
                                     4, packing='awq_gemm', scale_dtype=exported['scales'].dtype)
            if not torch.isfinite(weight).all() or not torch.equal(recovered, exported['codes'].to(weight.device)):
                raise ValueError('Staged AWQ weights do not retain hard codes in replay dtype')
            payloads[name] = (linear, weight, exported)
        for name, (linear, weight, exported) in payloads.items():
            named = modules[name]
            named.stream_state_payload_to_cpu({'q_scales': exported['scales'], 'q_zeros': exported['zeros']})
            named.state['wq'] = weight
            linear.weight.data = weight
    diagnostics = {stage: {key: value for key, value in record.items()
                           if key in ('history', 'elapsed_seconds', 'objective', 'initializer_timing')}
                   for stage, record in records.items()}
    processor.qcfg.meta.setdefault('gsq_training_runs', []).append(
        dict(layer_index=layer_index, initializer='awq', config=config.to_dict(), stages=diagnostics))
