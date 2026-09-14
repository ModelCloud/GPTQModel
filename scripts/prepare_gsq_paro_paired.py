"""Derive paired layer-1 inputs from dense and canonical F6 layer-0 replay."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from scripts.validate_qvq_gsq_layers import digest, write_json


def prepare(source, layer0, output):
    if output.exists():
        raise ValueError('Use a fresh output directory')
    source_provenance = json.loads((source / 'provenance.json').read_text())
    prepared = json.loads((layer0 / 'provenance.json').read_text())
    if digest(layer0 / 'layer-inputs.pt') != prepared['artifact_sha256']:
        raise ValueError('Layer-0 input artifact changed')
    if digest(layer0 / 'documents.json') != prepared['documents_sha256']:
        raise ValueError('Document split changed')
    for path, sha in source_provenance['file_hashes'].items():
        if digest(path) != sha:
            raise ValueError(f'Model source changed: {path}')
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not uuid.startswith('GPU-') or ',' in uuid or not os.environ.get('GPU_ALLOCATOR_LEASE_ID'):
        raise ValueError('Requires one exclusive GPU allocator UUID lease')
    for _ in range(3):
        inventory = subprocess.check_output(['nvidia-smi', '--id=' + uuid,
            '--query-gpu=index,pci.bus_id,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            text=True).strip()
        fields = [v.strip() for v in inventory.split(',')]
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                            '--format=csv,noheader'], text=True)
        if fields[2] != uuid or int(fields[4]) > 8 or int(fields[5]) or uuid in processes:
            raise RuntimeError('Idle preflight failed: ' + inventory)
        print('IDLE', inventory, flush=True)
        time.sleep(1)

    import torch
    from transformers import AutoConfig
    from safetensors import safe_open
    from scripts.gsq_f6_reference import install_f6
    from scripts.validate_gsq_paro_group import make_decoder_layer

    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    data = torch.load(layer0 / 'layer-inputs.pt', weights_only=True)
    config = AutoConfig.from_pretrained(prepared['dense'], local_files_only=True)
    config._attn_implementation = 'eager'
    dense = make_decoder_layer(config, data['layer_state']).cuda().eval().requires_grad_(False)
    wrapper = torch.nn.Module()
    wrapper.model = torch.nn.Module()
    wrapper.model.layers = torch.nn.ModuleList([make_decoder_layer(config, data['layer_state']).cuda().eval()])
    snapshot = Path(source_provenance['snapshot']) / 'qvq-p32'
    installed = install_f6(wrapper, snapshot, module_prefix='model.layers.0')
    if installed['snapshot_quantized_modules'] != 7:
        raise ValueError(f'Expected all seven F6 layer-0 projections: {installed}')
    paired, diagnostics = {}, {}
    with torch.no_grad():
        for split, rows in data['inputs'].items():
            clean, noisy, differences = [], [], []
            for x in rows:
                x = x.cuda().float()
                teacher = dense(x)
                runtime = wrapper.model.layers[0](x)
                if not torch.isfinite(teacher).all() or not torch.isfinite(runtime).all():
                    raise ValueError('Nonfinite paired layer output')
                clean.append(teacher.cpu())
                noisy.append(runtime.cpu())
                differences.append((teacher-runtime).square().mean().item())
            paired[split] = {'clean': clean, 'noisy': noisy}
            diagnostics[split] = differences
            print('PAIRED', split, len(rows), 'mean document MSE', sum(differences)/len(differences), flush=True)
    with safe_open(Path(prepared['dense']) / 'model.safetensors', framework='pt', device='cpu') as model:
        layer_state = {key.removeprefix('model.layers.1.'): model.get_tensor(key)
                       for key in model.keys() if key.startswith('model.layers.1.')}
    output.mkdir(parents=True)
    torch.save({'layer_state': layer_state, 'paired_inputs': paired}, output / 'layer-inputs.pt')
    write_json(output / 'documents.json', json.loads((layer0 / 'documents.json').read_text()))
    write_json(output / 'provenance.json', {
        'state': 'prepared_not_quantized', 'dense': prepared['dense'], 'snapshot': str(snapshot),
        'seed': 7, 'layer_index': 1, 'inventory': inventory, 'installed': installed,
        'documents': prepared['documents'], 'tokens': prepared['tokens'],
        'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'upstream_document_mse': diagnostics, 'artifact_sha256': digest(output / 'layer-inputs.pt'),
        'documents_sha256': digest(output / 'documents.json'),
        'files': {str(p.resolve()): digest(p) for p in [source / 'provenance.json', layer0 / 'provenance.json',
                  layer0 / 'layer-inputs.pt', Path(__file__), Path('scripts/gsq_f6_reference.py'),
                  Path('scripts/validate_gsq_paro_group.py')]},
        'semantics': 'dense FP32 and canonical F6 FP32 layer-0 outputs on identical dense embedding inputs',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--layer0', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.layer0.resolve(), args.output.resolve())
