"""Isolate scale-gradient repeatability using real saved W2 projection weights.

This is a reduction correctness diagnostic, not model-quality evidence.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.validate_qvq_gsq_layers import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh result path')
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not uuid.startswith('GPU-') or ',' in uuid or not os.environ.get('GPU_ALLOCATOR_LEASE_ID'):
        raise ValueError('Requires an exclusive UUID GPU lease')
    for _ in range(3):
        inventory = subprocess.check_output(['nvidia-smi', '--id='+uuid,
            '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
        fields = [v.strip() for v in inventory.split(',')]
        if fields[0] != uuid or int(fields[1]) > 8 or int(fields[2]):
            raise ValueError('Idle preflight failed: '+inventory)
        print('IDLE', inventory, flush=True)
        time.sleep(1)
    import torch
    from gptqmodel.quantization.gsq_training import GSQScalarTrainingModule

    torch.set_num_threads(4)
    torch.manual_seed(7)
    saved = torch.load(args.source, map_location='cpu', weights_only=True)
    weight, scale = saved['initializers']['self_attn.q_proj']
    weight, scale = weight.cuda(), scale.cuda()
    quantizer = GSQScalarTrainingModule(weight, scale, 128, bits=2,
                                        noise=torch.randn(4, *weight.shape, device='cuda'))
    uniform = torch.rand_like(quantizer.logits)
    result = dict(source=str(args.source), payload_sha256=digest(args.source), inventory=inventory,
                  gpu=str(torch.cuda.get_device_properties(0)), torch=str(torch.__version__), cuda=torch.version.cuda,
                  source_hashes={p: digest(p) for p in (str(Path(__file__)),
                                 'gptqmodel/quantization/gsq_training.py')}, modes={})
    for deterministic in (False, True):
        torch.use_deterministic_algorithms(deterministic)
        first = None
        errors = []
        for _ in range(20):
            quantizer.zero_grad(set_to_none=True)
            output = quantizer(uniform=uniform, temperature=2., multiplier=100.)
            (output-weight).square().sum().backward()
            gradient = quantizer.scales.grad.detach().clone()
            if first is None:
                first = gradient
            errors.append(dict(max_abs=(gradient-first).abs().max().item(),
                               changed_elements=(gradient != first).sum().item()))
        result['modes'][str(deterministic)] = errors
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({mode: {'max_abs': max(r['max_abs'] for r in rows),
                            'changed_elements': max(r['changed_elements'] for r in rows)}
                      for mode, rows in result['modes'].items()}, indent=2))


if __name__ == '__main__':
    main()
