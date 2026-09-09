"""Paired document bootstrap for a completed FP8 propagation diagnostic."""

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(path, arms=('gsq_weight', 'gsq_calibrated')):
    report = json.loads(path.read_text())
    if report['state'] != 'complete':
        raise ValueError('Requires all arms completed')
    baseline = report['arms']['baseline']['rows']
    rng = np.random.default_rng(7)
    indices = rng.integers(0, len(baseline), size=(10000, len(baseline)))
    weights = np.array([row['tokens'] for row in baseline])
    result = {'method': 'paired document bootstrap; token-weighted means', 'seed': 7,
              'draws': 10000, 'arms': {}}
    for arm in arms:
        rows = report['arms'][arm]['rows']
        if len(rows) != len(baseline) or any(a['tokens'] != b['tokens'] for a, b in zip(rows, baseline)):
            raise ValueError('Mismatched paired documents')
        metrics = {}
        for key in ('kl_teacher_candidate', 'mse', 'top1_agreement', 'top5_agreement', 'top10_agreement'):
            delta = np.array([row[key]-base[key] for row, base in zip(rows, baseline)])
            bootstrap = (delta[indices]*weights[indices]).sum(1)/weights[indices].sum(1)
            low, high = np.quantile(bootstrap, [.025, .975])
            metrics[key] = {'candidate_minus_baseline': float(np.average(delta, weights=weights)),
                            'ci95': [float(low), float(high)],
                            'classification': 'noise-consistent' if low <= 0 <= high else (
                                'clear positive' if (high < 0 if key in ('kl_teacher_candidate', 'mse')
                                                     else low > 0) else 'clear negative')}
        result['arms'][arm] = metrics
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', default=['gsq_weight', 'gsq_calibrated'])
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.report, arms=args.arms), indent=2)+'\n')
