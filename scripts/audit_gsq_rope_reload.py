"""Isolate nonpersistent rotary-buffer rounding in a saved full-model reload."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.validate_qvq_gsq_layers import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh result path')
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not uuid.startswith('GPU-') or ',' in uuid or not os.environ.get('GPU_ALLOCATOR_LEASE_ID'):
        raise ValueError('Requires exclusive UUID lease')
    idle = 0
    for _ in range(60):
        inventory = subprocess.check_output(['nvidia-smi', '--id='+uuid,
            '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
        fields = [value.strip() for value in inventory.split(',')]
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
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = json.loads((args.run/'report.json').read_text())
    if source['state'] != 'complete':
        raise ValueError('Requires completed source model')
    for name, expected in source['model_hashes'].items():
        if digest(args.run/'model'/name) != expected:
            raise ValueError('Checkpoint changed')
    model = GPTQModel.load(str(args.run/'model'), backend=BACKEND.TORCH, device='cuda:0',
                           dtype=torch.float16, attn_implementation='eager').model
    loaded_frequency = model.model.rotary_emb.inv_freq.detach().cpu().clone()
    cpu_rotary = LlamaRotaryEmbedding(model.config, device='cpu').eval()
    gpu_rotary = LlamaRotaryEmbedding(model.config, device='cuda').eval()
    frequency = dict(loaded_dtype=str(loaded_frequency.dtype),
                     cpu_max_abs=(loaded_frequency.float()-cpu_rotary.inv_freq.float()).abs().max().item(),
                     gpu_max_abs=(loaded_frequency.float()-gpu_rotary.inv_freq.float().cpu()).abs().max().item(),
                     loaded=loaded_frequency.float().tolist(), cpu=cpu_rotary.inv_freq.float().tolist())
    print('FREQUENCY', json.dumps(frequency), flush=True)
    rows = json.loads((args.run/'inputs.json').read_text())['heldout']
    result = dict(run=str(args.run), source_report_sha256=digest(args.run/'report.json'),
                  script_sha256=digest(__file__), torch=str(torch.__version__), cuda=torch.version.cuda,
                  inventory=inventory, frequency=frequency, rows=[])
    for i, row in enumerate(rows):
        model.model.rotary_emb = LlamaRotaryEmbedding(model.config, device='cpu').to('cuda').eval()
        ids = torch.tensor([row['input_ids']], device='cuda')
        with torch.inference_mode():
            canonical = model(ids, use_cache=False).logits[0]
            model.model.rotary_emb.half()
            rounded = model(ids, use_cache=False).logits[0]
            before = torch.load(args.run/'before_reload'/f'{i}.pt', weights_only=True).cuda()
            after = torch.load(args.run/'after_reload'/f'{i}.pt', weights_only=True).cuda()
            metrics = dict(canonical_matches_reload=torch.equal(canonical, after),
                           rounded_matches_pre_reload=torch.equal(rounded, before),
                           canonical_reload_max_abs=(canonical-after).abs().max().item(),
                           rounded_pre_reload_max_abs=(rounded-before).abs().max().item())
            result['rows'].append(metrics)
        print('ROPE', i+1, json.dumps(metrics), flush=True)
    write_json(args.output, result)


if __name__ == '__main__':
    main()
