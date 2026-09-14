"""Compare completed staged training runs without claiming bitwise determinism."""

import argparse
import json
from pathlib import Path

import torch

from scripts.validate_qvq_gsq_layers import digest


def compare(reference, candidate):
    reports = [json.loads((root/'report.json').read_text()) for root in (reference, candidate)]
    if any(report['state'] != 'complete' for report in reports):
        raise ValueError('Requires completed runs')
    for root, report in zip((reference, candidate), reports):
        if digest(root/'stages.pt') != report['payload_sha256']:
            raise ValueError('Payload changed')
    if digest(reference/'inputs.json') != digest(candidate/'inputs.json'):
        raise ValueError('Requires identical captured documents')
    left, right = [torch.load(root/'stages.pt', map_location='cpu', weights_only=True)
                   for root in (reference, candidate)]
    def delta(a, b):
        diff = a.float()-b.float()
        return dict(exact=torch.equal(a, b), max_abs=diff.abs().max().item(), mse=diff.square().mean().item())
    result = dict(reference=str(reference), candidate=str(candidate),
                  payload_hashes=[r['payload_sha256'] for r in reports],
                  initializers={}, weights={}, stages={}, heldout=[])
    for name, (weight, scale) in left['initializers'].items():
        other_weight, other_scale = right['initializers'][name]
        result['initializers'][name] = dict(weight=delta(weight, other_weight), scale=delta(scale, other_scale))
    for name, weight in left['state_dict'].items():
        result['weights'][name] = delta(weight, right['state_dict'][name])
    for stage, values in left['stages'].items():
        old, new = values['history'], right['stages'][stage]['history']
        if len(old) != len(new):
            raise ValueError('Different stage budgets')
        first = next((i for i, (a, b) in enumerate(zip(old, new)) if a['loss'] != b['loss']), None)
        result['stages'][stage] = dict(
            steps=len(old), first_loss_difference=first,
            scales={name: delta(scale, right['stages'][stage]['scales'][name])
                    for name, scale in values['scales'].items()},
            first_losses=None if first is None else [old[first]['loss'], new[first]['loss']],
            schedule_equal=all({k: v for k, v in a.items() if k != 'loss'} ==
                               {k: v for k, v in b.items() if k != 'loss'} for a, b in zip(old, new)))
    for report in reports:
        rows = report['heldout']
        count = sum(row['elements'] for row in rows)
        result['heldout'].append({arm: sum(row[arm+'_sse'] for row in rows)/count
                                 for arm in ('baseline', 'staged')})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'stages': result['stages'], 'heldout': result['heldout']}, indent=2))
