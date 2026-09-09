"""Load a canonical .partial snapshot and verify its actual runtime logits."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.validate_qvq_gsq_layers import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.snapshot/'snapshot_manifest.json').read_text())
    for name, entry in manifest['files'].items():
        if digest(args.snapshot/name) != entry['sha256']:
            raise ValueError('Snapshot file changed: '+name)
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not uuid.startswith('GPU-') or ',' in uuid or not os.environ.get('GPU_ALLOCATOR_LEASE_ID'):
        raise ValueError('Requires exclusive UUID lease')
    idle = 0
    for _ in range(60):
        inventory = subprocess.check_output(['nvidia-smi', '--id='+uuid,
            '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
        fields = [v.strip() for v in inventory.split(',')]
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                             '--format=csv,noheader'], text=True)
        idle = idle+1 if fields[0] == uuid and int(fields[1]) <= 8 and int(fields[2]) == 0 and uuid not in processes else 0
        print('PREFLIGHT', inventory, 'consecutive_idle', idle, flush=True)
        if idle == 3:
            break
        time.sleep(1)
    else:
        raise ValueError('Idle preflight timeout')
    import torch
    from gptqmodel import GPTQModel
    from gptqmodel.utils.backend import BACKEND

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = GPTQModel.load(str(args.snapshot/'gptq-v2'), backend=BACKEND.TORCH, device='cuda:0',
                           dtype=torch.float16, attn_implementation='eager').model
    documents = json.loads((args.snapshot/'calibration/derived/inputs.json').read_text())['heldout'][:3]
    result = dict(state='running', snapshot=str(args.snapshot), reference=str(args.reference_run),
                  inventory=inventory, torch=str(torch.__version__), cuda=torch.version.cuda, rows=[])
    with torch.inference_mode():
        for i, document in enumerate(documents):
            actual = model(torch.tensor([document['input_ids']], device='cuda'), use_cache=False).logits[0]
            reference = torch.load(args.reference_run/'after_reload'/f'{i}.pt', weights_only=True).cuda()
            row = dict(tokens=len(actual), exact=torch.equal(actual, reference),
                       max_abs=(actual-reference).abs().max().item(), finite=torch.isfinite(actual).all().item(),
                       reference_sha256=digest(args.reference_run/'after_reload'/f'{i}.pt'))
            result['rows'].append(row)
            print('SNAPSHOT', i, json.dumps(row), flush=True)
    result['state'] = 'passed' if all(row['exact'] and row['finite'] for row in result['rows']) else 'failed'
    write_json(args.snapshot/'snapshot-reload.json', result)
    if result['state'] != 'passed':
        raise ValueError('Snapshot runtime does not match source checkpoint')


if __name__ == '__main__':
    main()
