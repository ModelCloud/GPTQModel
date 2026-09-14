"""Copy a validated full staged/GPTQ experiment into the canonical snapshot store."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from scripts.validate_qvq_gsq_layers import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run/'report.json').read_text())
    if report['state'] != 'complete':
        raise ValueError('Requires completed full-model execution')
    if not all(row['reload_exact'] for row in report['rows']):
        audit = json.loads((args.run/'rope-reload-audit-cpu.json').read_text())
        if (audit['source_report_sha256'] != digest(args.run/'report.json')
                or len(audit['rows']) != len(report['rows'])
                or not all(row['canonical_matches_reload'] and row['rounded_matches_pre_reload']
                           for row in audit['rows'])):
            raise ValueError('Unresolved reload drift')
    training = json.loads((args.run/'training.json').read_text())
    model_config = json.loads((args.run/'model/config.json').read_text())
    expected_layers = list(range(model_config['num_hidden_layers']))
    if (training['state'] != 'complete' or training['layer_indices'] != expected_layers
            or len(training['blocks']) != len(expected_layers)):
        raise ValueError('Requires complete uniform-model training')
    for name, sha in report['model_hashes'].items():
        if digest(args.run/'model'/name) != sha:
            raise ValueError('Model output changed')
    source = json.loads((args.inputs/'provenance.json').read_text())
    calibration_samples = source['train_rows']
    for path, sha in source['source_hashes'].items():
        if digest(path) != sha:
            raise ValueError('Calibration source changed: '+path)
    comparison = json.loads((args.comparison/'report.json').read_text())
    if comparison['source_reports'].get(str(args.run/'report.json')) != digest(args.run/'report.json'):
        raise ValueError('Comparison is not bound to this run')
    base = Path('/monster/data/model/qvq')
    content = hashlib.sha256(json.dumps(report['model_hashes'], sort_keys=True).encode()).hexdigest()[:12]
    day = datetime.now(timezone.utc).strftime('%Y%m%d')
    experiment = f'gsq-w{report["bits"]}-{report["arm"]}'
    if report['gsq_training']['enabled']:
        experiment += f'-{report["gsq_training"].get("optimizer", "lion")}-e{report["gsq_training"]["epochs"]}'
    name = (f'modelcloud-qvq__llama-3.2-1b-instruct__{experiment}__gptq-v2__'
            f'calib{calibration_samples}-unweighted__seed7__{day}__commit{report["commit"][:12]}__{content}')
    final = base/name
    staging = base/(name+'.partial')
    if final.exists() or staging.exists():
        raise ValueError('Snapshot already exists; do not overwrite')
    staging.mkdir()
    origins = {}

    def copy_file(source_path, relative):
        source_path = Path(source_path)
        destination = staging/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        origins[str(relative)] = str(source_path.resolve())

    for path in (args.run/'model').rglob('*'):
        if path.is_file():
            copy_file(path, Path('gptq-v2')/path.relative_to(args.run/'model'))
    for path in source['source_hashes']:
        copy_file(path, Path('calibration/source')/Path(path).name)
    copy_file(args.inputs/'inputs.json', Path('calibration/derived/inputs.json'))
    copy_file(args.inputs/'provenance.json', Path('metadata/input-provenance.json'))
    for path in args.run.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.md', '.py') and 'model' not in path.relative_to(args.run).parts:
            copy_file(path, Path('metadata/run')/path.relative_to(args.run))
    for path in args.comparison.iterdir():
        if path.is_file() and path.suffix in ('.json', '.md', '.py'):
            copy_file(path, Path('metadata/comparison')/path.name)
    copy_file(args.log, Path('logs/quantization.log'))
    rationale = (f'Complete all-layer W{report["bits"]} {report["arm"]} control for the user-requested staged GSQ reproduction work. '
                 f'Real Llama 3.2 1B Instruct, group{report["group_size"]}, seed7, {calibration_samples} unweighted '
                 'calibration documents and 32 locked '
                 'held-out documents. FP32 training and FP16 GPTQ-v2 deployment. This small reused dataset is not '
                 'the paper calibration recipe; the snapshot is experimental, not a promoted default. '
                 'The no-GSQ arm is the matched staged-path GPTQ initializer, not the package-default '
                 'true-sequential GPTQModel.quantize recipe.')
    (staging/'why.md').write_text('# Why this snapshot exists\n\n'+rationale+'\n')
    (staging/'model_run.md').write_text('# Full-model GSQ experiment snapshot\n\n'
        f'Stored path after validation: `{final}`. Run ID: `{report["run_id"]}`. Arm: `{report["arm"]}`.\n\n'
        +rationale+'\n\nFull command, effective configuration, exact model/source hashes, hardware/runtime and '
        'evaluation records are copied under `metadata/run`; matched metrics and bootstrap are under '
        '`metadata/comparison`. The exact calibration corpus and selected tokens are copied under `calibration`. '
        'Full QVQ commit: `'+report['commit']+'`; ZML is not used. Quantization log: `logs/quantization.log`, SHA256 `'
        +digest(args.log)+'`. This directory remains .partial until a load from this path is verified.\n\n'
        '```json\n'+json.dumps(report, indent=2)+'\n```\n')
    manifest = dict(state='copied_pending_snapshot_reload', final_path=str(final), arm=report['arm'],
                    repository='https://github.com/ModelCloud/QvQ.git', branch='codex/gsq-p32-window',
                    commit=report['commit'], model=report['source_model'], formats=['gptq-v2'], seed=7,
                    bits=report['bits'], group_size=report['group_size'],
                    created_utc=datetime.now(timezone.utc).isoformat(), files={})
    for path in staging.rglob('*'):
        if path.is_file():
            relative = str(path.relative_to(staging))
            manifest['files'][relative] = dict(source=origins.get(relative, 'generated snapshot metadata'),
                                              bytes=path.stat().st_size, sha256=digest(path))
    write_json(staging/'snapshot_manifest.json', manifest)
    print(json.dumps(dict(staging=str(staging), final=str(final)), indent=2))


if __name__ == '__main__':
    main()
