"""Real Llama layer-0 grouped optimization and native QKV export validation."""

import argparse
import copy
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from scripts.validate_qvq_gsq_layers import digest, write_json


def execute(prepared, output, scope):
    if output.exists():
        raise ValueError('Use a fresh execution directory')
    provenance = json.loads((prepared / 'provenance.json').read_text())
    for path, sha in provenance['files'].items():
        if digest(path) != sha:
            raise ValueError(f'Prepared source changed: {path}')
    if digest(prepared / 'layer-inputs.pt') != provenance['artifact_sha256']:
        raise ValueError('Prepared decoder inputs changed')
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not uuid.startswith('GPU-') or ',' in uuid or not os.environ.get('GPU_ALLOCATOR_LEASE_ID'):
        raise ValueError('Requires one exclusive UUID GPU allocator lease')
    for _ in range(3):
        inventory = subprocess.check_output(['nvidia-smi', '--id=' + uuid,
            '--query-gpu=index,pci.bus_id,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            text=True).strip()
        fields = [v.strip() for v in inventory.split(',')]
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                            '--format=csv,noheader'], text=True)
        if fields[2] != uuid or int(fields[4]) > 8 or int(fields[5]) or uuid in processes:
            raise RuntimeError('Idle GPU preflight failed: ' + inventory)
        print('IDLE', inventory, flush=True)
        time.sleep(1)

    import torch
    from transformers import AutoConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
    from gptqmodel.looper.input_cache import InputCache
    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
    from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
    from gptqmodel.quantization.config import GSQConfig, ParoConfig
    from gptqmodel.quantization.gsq_scalar import affine_codes
    from gptqmodel.quantization.paroquant.optimization import _apply_inverse_rotation

    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    data = torch.load(prepared / 'layer-inputs.pt', weights_only=True)
    config = AutoConfig.from_pretrained(provenance['dense'], local_files_only=True)
    config._attn_implementation = 'eager'

    class Layer(LlamaDecoderLayer):
        def forward(self, hidden_states, attention_mask=None, position_ids=None, position_embeddings=None, **kwargs):
            if position_ids is None:
                position_ids = torch.arange(hidden_states.shape[-2], device=hidden_states.device).unsqueeze(0)
            if position_embeddings is None:
                position_embeddings = self.rotary(hidden_states, position_ids)
            if attention_mask is None:
                length = hidden_states.shape[-2]
                attention_mask = torch.full((length, length), torch.finfo(hidden_states.dtype).min,
                                            device=hidden_states.device, dtype=hidden_states.dtype).triu(1)[None, None]
            return super().forward(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                                   position_embeddings=position_embeddings, **kwargs)

    dense = Layer(config, layer_idx=0).float()
    dense.load_state_dict(data['layer_state'], strict=True)
    dense.rotary = LlamaRotaryEmbedding(config)
    dense = dense.cuda().eval().requires_grad_(False)
    calibration = data['inputs']['train'] + data['inputs']['validation']
    with torch.no_grad():
        targets = [dense(x.cuda().half().float()).cpu() for x in calibration]
        heldout_targets = [dense(x.cuda().half().float()).cpu() for x in data['inputs']['heldout']]
    output.mkdir(parents=True)
    report = {'state': 'running', 'scope': scope, 'inventory': inventory, 'provenance': provenance,
              'runner_sha256': digest(__file__), 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
              'epochs': {'rotation': 2, 'finetune': 2}, 'arms': {}}
    write_json(output / 'report.json', report)
    baseline = {}
    for arm in ('baseline', 'gsq_fixed', 'gsq_scales'):
        print('GROUP_START', scope, arm, flush=True)
        report['state'] = 'optimizing ' + arm
        write_json(output / 'report.json', report)
        started = time.monotonic()
        layer = copy.deepcopy(dense).half().eval()
        processor = object.__new__(ParoQuantProcessor)
        processor.qcfg = ParoConfig(bits=4, group_size=128, krot=8, opt_scope=scope, opt_seed=7,
            opt_rotation_epochs=2, opt_finetune_epochs=2, opt_fused_rotation=True, opt_stage_cudagraph=False,
            offload_to_disk=False, gsq=None if arm == 'baseline' else GSQConfig(
                enabled=True, seed=7, steps=100, learn_scales=arm == 'gsq_scales'))
        write_json(output / (arm + '.config.json'), processor.qcfg.to_dict())
        processor.lock = threading.Lock()
        processor._layer_states_lock = threading.Lock()
        processor._layer_states = {}
        processor._batch_tls = threading.local()
        processor.tasks = {}
        processor.fallback = False
        processor.calculate_w_wq_diff = False
        processor.gptq_model = SimpleNamespace(support_batch_quantize=True)
        processor._has_explicit_validation_calibration = True
        processor._train_calibration_batch_count = 12
        processor._validation_calibration_batch_count = 4
        processor.inputs_cache = InputCache(layer_inputs=[[x] for x in calibration], layer_input_kwargs=[{}]*16,
                                            position_ids=[None]*16, attention_masks=[None]*16)
        state = processor._get_layer_state(0)
        state.layer_module = layer
        state.layer_inputs = [[x.half()] for x in calibration]
        state.layer_outputs = [[x] for x in targets]
        modules, hooks = {}, []
        for role in ('q', 'k', 'v'):
            name = 'self_attn.' + role + '_proj'
            module = NamedModule(getattr(layer.self_attn, role + '_proj'), name, 'model.layers.0.' + name, 0)
            module.state['module_tree_flags'] = frozenset({role})
            modules[name] = module
            processor.tasks[name] = {'inputs': [], 'batch_indices': [], 'layer_index': 0}
            hooks.append(module.module.register_forward_hook(processor.pre_process_fwd_hook(name)))
        state.modules = modules.copy()
        try:
            with torch.no_grad():
                for index, x in enumerate(calibration):
                    processor._set_current_batch_index(index)
                    layer(x.cuda().half())
        finally:
            for hook in hooks:
                hook.remove()
            processor._set_current_batch_index(None)
        processor._log_quant_result = lambda *args: None
        processor._quantize_layer(0, state)
        layer.cuda()
        checks = []
        native_hooks = []
        arm_report = {'modules': {}, 'native_checks': checks}
        for name, module in modules.items():
            stored = module.state
            rows, width = stored['pack_weight'].shape
            kwargs = dict(bits=4, group_size=128, sym=True, desc_act=False, in_features=width, out_features=rows,
                          bias=False, register_buffers=True, krot=8)
            packed = ParoLinear(**kwargs)
            transport = torch.nn.Linear(width, rows, bias=False, dtype=torch.float16)
            transport.weight.data.copy_(stored['pack_weight'])
            packed.pack(transport, stored['q_scales'], stored['q_zeros'])
            for key in ('pairs', 'theta', 'channel_scales'):
                getattr(packed, key).copy_(stored[key].reshape_as(getattr(packed, key)))
            path = output / f'{name}.{arm}.pt'
            torch.save(packed.state_dict(), path)
            restored = ParoLinear(**kwargs)
            restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
            payload = restored.state_dict()
            if arm == 'baseline':
                baseline[name] = {key: value.clone() for key, value in payload.items()}
            equal = all(torch.equal(value, baseline[name][key]) for key, value in payload.items())
            groups = torch.arange(width) // 128
            codes = affine_codes(stored['pack_weight'], stored['q_scales'], stored['q_zeros'], groups, 4,
                                 packing='awq_gemm')
            grid = (codes.float()-stored['q_zeros'][:, groups]) * stored['q_scales'].float()[:, groups]
            reference = _apply_inverse_rotation(grid, stored['pairs'], stored['theta'].float(),
                        group_size=128, fused_rotation=False) * stored['channel_scales'].float().reshape(-1)
            reference = reference.cuda()
            restored = restored.cuda().eval()
            restored.post_init()

            def check(_module, args, result, reference=reference, name=name):
                delta = (result.float() - args[0].float() @ reference.T).abs()
                row = {'module': name, 'mean': delta.mean().item(), 'max': delta.max().item()}
                checks.append(row)
                if not torch.isfinite(result).all() or row['mean'] > .002 or row['max'] > .046875:
                    write_json(output / 'failed-native.json', row)
                    raise AssertionError(str(row))

            native_hooks.append(restored.register_forward_hook(check))
            setattr(layer.self_attn, name.rsplit('.', 1)[-1], restored)
            arm_report['modules'][name] = {'gsq': stored.get('gsq_diagnostics'),
                                          'payload_equal_baseline': equal, 'sha256': digest(path)}
        error, count = 0., 0
        with torch.no_grad():
            for x, target in zip(data['inputs']['heldout'], heldout_targets, strict=True):
                actual = layer(x.cuda().half()).float()
                error += (actual.double() - target.cuda().double()).square().sum().item()
                count += actual.numel()
        for hook in native_hooks:
            hook.remove()
        arm_report.update(heldout_layer_mse=error/count, seconds=time.monotonic()-started)
        report['arms'][arm] = arm_report
        write_json(output / 'report.json', report)
        print('GROUP_DONE', scope, arm, error/count, flush=True)
        del layer, processor, state, modules, packed, restored
        torch.cuda.empty_cache()
    report['state'] = 'complete'
    write_json(output / 'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', choices=['layer', 'compute_block'], required=True)
    args = parser.parse_args()
    execute(args.prepared.resolve(), args.output.resolve(), args.scope)
