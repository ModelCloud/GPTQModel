"""Matched-noise CUDA quantizer comparison against a pinned author checkout."""

import argparse
import importlib.util
import json
import os
import subprocess
import time
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--author', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lion-wheel', type=Path)
    args = parser.parse_args()
    if args.lion_wheel:
        sys.path.insert(0, str(args.lion_wheel))
    if not os.environ.get('GPU_ALLOCATOR_LEASE_ID'):
        raise ValueError('Requires a GPU allocator lease')
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    for _ in range(3):
        sample = subprocess.check_output(['nvidia-smi', '--id='+uuid,
            '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
        fields = [v.strip() for v in sample.split(',')]
        if fields[0] != uuid or int(fields[1]) > 8 or int(fields[2]):
            raise ValueError('Idle preflight failed: '+sample)
        print('IDLE', sample, flush=True)
        time.sleep(1)
    import torch
    from gptqmodel.quantization.gsq_training import GSQScalarTrainingModule, GSQLion

    results = []
    for bits in (2, 3, 4):
        file = 'gumbel_quantizer_2bit.py' if bits == 2 else 'gumbel_quantizer_int.py'
        spec = importlib.util.spec_from_file_location('author', args.author/'src/quantization'/file)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        for dtype, logits_dtype in ((torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16),
                                    (torch.bfloat16, torch.float32)):
            weight = (torch.arange(256, device='cuda').reshape(8, 32) % (2**bits)-2**(bits-1)).to(dtype)*.125
            scales = torch.full((8, 2), .125, device='cuda', dtype=dtype)
            torch.cuda.manual_seed(7)
            noise = torch.randn((4 if bits == 2 else 5, 8, 32), device='cuda', dtype=dtype)
            local = GSQScalarTrainingModule(weight, scales, 16, bits=bits, noise=noise, logits_dtype=logits_dtype)
            cls = reference.GumbelQuantizer2Bit if bits == 2 else reference.GumbelQuantizerInt
            torch.cuda.manual_seed(7)
            author = cls(weight, scales, 16, .01, 6., 'cuda:0', dtype, logits_dtype=logits_dtype, **({} if bits == 2 else {'bits': bits}))
            torch.cuda.manual_seed(11)
            uniform = torch.rand_like(local.logits)
            actual = local(uniform=uniform, temperature=.7, multiplier=20.)
            torch.cuda.manual_seed(11)
            expected = author(.7, 20.)
            actual.float().square().sum().backward()
            expected.float().square().sum().backward()
            row = dict(bits=bits, dtype=str(dtype), logits_dtype=str(logits_dtype))
            for name, a, b in [('forward', actual, expected), ('logit_gradient', local.logits.grad, author.quant_logits.grad),
                               ('scale_gradient', local.scales.grad, author.scales.grad)]:
                row[name] = float((a.float()-b.float()).abs().max())
            if args.lion_wheel:
                from lion_pytorch import Lion

                local_optimizer = GSQLion(local.optimizer_groups(assignment_lr=1e-4, scale_lr=5e-5,
                                                                 weight_decay=1.), betas=(.9, .95))
                author_optimizer = Lion([{'params': [author.quant_logits], 'lr': 1e-4, 'weight_decay': 1.},
                                         {'params': [author.scales], 'lr': 5e-5, 'weight_decay': 0.}], betas=(.9, .95))
                for step in range(20):
                    local_optimizer.zero_grad(set_to_none=True)
                    author_optimizer.zero_grad(set_to_none=True)
                    torch.cuda.manual_seed(100+step)
                    uniform = torch.rand_like(local.logits)
                    local(uniform=uniform, temperature=.7, multiplier=20.).float().square().sum().backward()
                    torch.cuda.manual_seed(100+step)
                    author(.7, 20.).float().square().sum().backward()
                    local_optimizer.step()
                    author_optimizer.step()
                row['trajectory_logit_max_error'] = float((local.logits-author.quant_logits).abs().max())
                row['trajectory_scale_max_error'] = float((local.scales-author.scales).abs().max())
            row['hard_weight_max_error'] = float((local.hard_weight().float()-author.get_hard_weights()[0].float()).abs().max())
            results.append(row)
    args.output.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
