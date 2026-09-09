"""Matched final-logit comparison for completed full-model staged GSQ arms."""

import argparse
import json
import math
from pathlib import Path

from scripts.analyze_gsq_fp8_propagation import analyze
from scripts.validate_qvq_gsq_layers import digest, write_json


def validate_matched_recipe(baseline, staged):
    """Match recipes, allowing unused baseline epoch/optimizer settings to differ."""
    for key in ('source_model', 'bits', 'group_size', 'train_precision', 'export_precision',
                'calibration_samples', 'calibration_tokens', 'weighting'):
        if key not in baseline or baseline[key] != staged.get(key):
            raise ValueError(f'Mismatched or missing recipe field: {key}')
    configs = []
    for report, expected in ((baseline, False), (staged, True)):
        config = dict(report['gsq_training'])
        if config.pop('enabled', None) is not expected:
            raise ValueError('Incorrect GSQ enable flag for comparison arm')
        # Historical reports predate the explicit initializer selector.
        config.setdefault('initializer', 'gptq')
        config.setdefault('batch_size', 1)
        config.setdefault('microbatch_size', 1)
        # The disabled arm never enters stage training. Reusing its checkpoint
        # for the user's epoch change does not change its quantized weights.
        config.pop('epochs', None)
        optimizer = config.pop('optimizer', 'lion')
        if optimizer not in ('lion', 'adamw'):
            raise ValueError('Unknown staged optimizer in comparison')
        configs.append(config)
    if configs[0] != configs[1]:
        raise ValueError('GSQ recipes differ beyond enable flag and unused baseline epochs/optimizer')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--staged', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    reports = [json.loads((root/'report.json').read_text()) for root in (args.baseline, args.staged)]
    if any(report['state'] != 'complete' for report in reports):
        raise ValueError('Requires completed full-model arms')
    if digest(args.baseline/'inputs.json') != digest(args.staged/'inputs.json'):
        raise ValueError('Different captured documents')
    validate_matched_recipe(*reports)
    result = dict(state='complete', source_reports={str(root/'report.json'): digest(root/'report.json')
                  for root in (args.baseline, args.staged)}, arms={})
    metrics = ('kl_teacher_candidate', 'mse', 'top1_agreement', 'top5_agreement', 'top10_agreement')
    for name, report in zip(('baseline', 'staged'), reports):
        if report['arm'] != name:
            raise ValueError('Arm identity mismatch')
        rows = report['rows']
        if not rows or len(rows) != 32:
            raise ValueError('Expected all 32 locked held-out documents')
        for index, row in enumerate(rows):
            if not isinstance(row['tokens'], int) or row['tokens'] <= 0:
                raise ValueError('Invalid held-out token count')
            if any(not math.isfinite(row[key]) for key in metrics):
                raise ValueError('Nonfinite held-out metric')
            reference = reports[0]['rows'][index]
            if row['tokens'] != reference['tokens'] or row.get('teacher_sha256') != reference.get('teacher_sha256'):
                raise ValueError('Different held-out teacher or token count')
            if not row.get('teacher_sha256'):
                raise ValueError('Missing held-out teacher hash')
        total = sum(row['tokens'] for row in rows)
        result['arms'][name] = dict(rows=rows, complete=True,
                                    mean={key: sum(row[key]*row['tokens'] for row in rows)/total for key in metrics},
                                    reload_exact=all(row['reload_exact'] for row in rows),
                                    reload_max_abs=max(row['reload_max_abs'] for row in rows))
    args.output.mkdir(parents=True)
    write_json(args.output/'report.json', result)
    bootstrap = analyze(args.output/'report.json', arms=['staged'])
    write_json(args.output/'bootstrap.json', bootstrap)
    print(json.dumps({name: {key: value for key, value in arm.items() if key != 'rows'}
                      for name, arm in result['arms'].items()}, indent=2))
    print(json.dumps(bootstrap, indent=2))


if __name__ == '__main__':
    main()
