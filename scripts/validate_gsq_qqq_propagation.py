"""Propagate saved QQQ QKV payloads through F6 with native activation quantization.

Only block-0 QKV use native QQQ. Other F6 operators use the canonical FP32
reference. This is not a complete native QQQ model export.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from scripts.gsq_f6_reference import install_f6
from scripts.validate_qvq_gsq_layers import TARGETS, digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--layers', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    local = json.loads((args.layers / 'report.json').read_text())
    inputs = Path(local['provenance']['inputs'])
    source = json.loads((inputs / 'provenance.json').read_text())
    if local['state'] != 'complete':
        raise ValueError('Requires completed local lifecycle validation')
    if args.prepare:
        if args.output.exists():
            raise ValueError('Use a fresh output directory')
        for path, sha in source['file_hashes'].items():
            if digest(path) != sha:
                raise ValueError(f'Changed source model: {path}')
        files = {str(args.layers / 'report.json'): digest(args.layers / 'report.json'),
                 str(inputs / 'inputs.json'): source['inputs_sha256'],
                 str(inputs / 'provenance.json'): digest(inputs / 'provenance.json')}
        files.update(source['file_hashes'])
        # Bind saved data, not historical experiment source paths that may
        # legitimately have moved on. Bind this execution's implementations below.
        files.update({path: sha for path, sha in local['provenance']['files'].items()
                      if Path(path).suffix not in ('.py', '.cu')})
        for name in TARGETS:
            for arm in ('baseline', 'gsq_fixed'):
                path = args.layers / f'{name}.{arm}.pt'
                files[str(path)] = local['layers'][name][arm]['payload_sha256']
        for path in (Path(__file__), Path('scripts/gsq_f6_reference.py'),
                     Path('scripts/p32_twenty/scorecard.py'), Path('gptqmodel/nn_modules/qlinear/qqq.py'),
                     Path('gptqmodel_ext/qqq/qqq_gemm.cu')):
            files[str(path)] = digest(path)
        for path, sha in files.items():
            if digest(path) != sha:
                raise ValueError(f'Changed input: {path}')
        args.output.mkdir(parents=True)
        write_json(args.output / 'provenance.json', {
            'layers': str(args.layers), 'files': files, 'seed': 7,
            'repository_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'scope': __doc__, 'group_size': local['provenance']['group_size'],
        })
        return
    provenance = json.loads((args.output / 'provenance.json').read_text())
    if provenance['layers'] != str(args.layers) or (args.output / 'report.json').exists():
        raise ValueError('Changed layers or existing execution')
    for path, sha in provenance['files'].items():
        if digest(path) != sha:
            raise ValueError(f'Prepared file changed: {path}')
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not uuid.startswith('GPU-') or ',' in uuid or not os.environ.get('GPU_ALLOCATOR_LEASE_ID'):
        raise ValueError('Requires an exclusive UUID GPU lease')
    for _ in range(3):
        inventory = subprocess.check_output([
            'nvidia-smi', '--id=' + uuid, '--query-gpu=index,pci.bus_id,uuid,name,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
        fields = [v.strip() for v in inventory.split(',')]
        processes = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        if fields[2] != uuid or int(fields[4]) > 8 or int(fields[5]) != 0 or uuid in processes:
            raise ValueError('Idle preflight failed: ' + inventory)
        print('IDLE', inventory, flush=True)
        time.sleep(1)

    import torch
    from transformers import AutoModelForCausalLM
    from gptqmodel.nn_modules.qlinear.qqq import QQQLinear
    from scripts.p32_twenty.scorecard import logits_metrics

    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    rows = json.loads((inputs / 'inputs.json').read_text())['heldout']
    report = {'state': 'dense teacher', 'provenance': provenance, 'inventory': inventory,
              'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'arms': {}}
    write_json(args.output / 'report.json', report)
    model = AutoModelForCausalLM.from_pretrained(
        source['dense'], dtype=torch.float32, device_map={'': 'cuda:0'},
        attn_implementation='eager', local_files_only=True).eval().requires_grad_(False)
    (args.output / 'teacher').mkdir()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            logits = model(torch.tensor([row['input_ids']], device='cuda'), use_cache=False).logits[0]
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite dense logits')
            torch.save(logits.cpu(), args.output / 'teacher' / f'{i}.pt')
    report.update(install_f6(model, Path(source['snapshot']) / 'qvq-p32'))
    baseline_hashes = []
    for arm in ('f6', 'baseline', 'gsq_fixed'):
        if arm != 'f6':
            for name in TARGETS:
                state = torch.load(args.layers / f'{name}.{arm}.pt', weights_only=True)
                fixture = torch.load(inputs / f'{name}.inputs.pt', weights_only=True)
                out_features, in_features = fixture['weight'].shape
                native = QQQLinear(bits=4, group_size=provenance['group_size'], sym=True, desc_act=False,
                                   in_features=in_features, out_features=out_features, bias=False)
                native.load_state_dict(state, strict=True)
                native = native.cuda().eval()
                native.post_init()
                parent, leaf = name.rsplit('.', 1)
                setattr(model.get_submodule(parent), leaf, native)
        results = []
        for i, row in enumerate(rows):
            with torch.inference_mode():
                teacher_path = args.output / 'teacher' / f'{i}.pt'
                teacher = torch.load(teacher_path, weights_only=True).cuda()
                logits = model(torch.tensor([row['input_ids']], device='cuda'), use_cache=False).logits[0]
                chunks = [logits_metrics(logits[j:j+32], teacher[j:j+32]) for j in range(0, len(logits), 32)]
                metrics = {k: sum(c[k] * c['tokens'] for c in chunks) / len(logits)
                           for k in chunks[0] if k != 'tokens'}
                checksum = hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest()
                if arm == 'baseline':
                    baseline_hashes.append(checksum)
                metrics.update(tokens=len(logits), mse=(logits-teacher).square().mean().item(),
                               logits_sha256=checksum, teacher_sha256=digest(teacher_path))
                if arm == 'gsq_fixed':
                    metrics['baseline_logits_equal'] = checksum == baseline_hashes[i]
                results.append(metrics)
            numeric = ('kl_teacher_candidate', 'mse', 'top1_agreement', 'top5_agreement', 'top10_agreement')
            total = sum(v['tokens'] for v in results)
            report['arms'][arm] = {'rows': results, 'complete': len(results) == len(rows),
                                   'mean': {k: sum(v[k]*v['tokens'] for v in results)/total for k in numeric}}
            report['state'] = f'{arm} {i+1}/{len(rows)}'
            write_json(args.output / 'report.json', report)
            print('MODEL', report['state'], json.dumps(report['arms'][arm]['mean']), flush=True)
    report['state'] = 'complete'
    write_json(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
