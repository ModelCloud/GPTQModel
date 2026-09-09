"""Prepare genuine decoder-layer inputs for real grouped ParoQuant validation."""

import argparse
import json
import subprocess
from pathlib import Path

from scripts.validate_qvq_gsq_layers import digest, write_json


def prepare(inputs, output):
    import torch
    from safetensors import safe_open

    if output.exists():
        raise ValueError('Use a fresh output directory')
    provenance = json.loads((inputs / 'provenance.json').read_text())
    if digest(inputs / 'inputs.json') != provenance['inputs_sha256']:
        raise ValueError('Locked document selection changed')
    docs = json.loads((inputs / 'inputs.json').read_text())
    if len(docs['train']) != 16 or len(docs['heldout']) != 32:
        raise ValueError('Expected 16 calibration and 32 held-out documents')
    split = {'train': docs['train'][:12], 'validation': docs['train'][12:], 'heldout': docs['heldout']}
    keys = {name: {tuple(row['input_ids']) for row in rows} for name, rows in split.items()}
    if any(keys[a] & keys[b] for a, b in [('train', 'validation'), ('train', 'heldout'), ('validation', 'heldout')]):
        raise ValueError('Token-identical documents cross the split boundary')
    dense = Path(provenance['dense'])
    source_files = [dense / 'model.safetensors', dense / 'config.json']
    for path in source_files:
        if digest(path) != provenance['file_hashes'][str(path)]:
            raise ValueError(f'Dense model binding changed: {path}')
    for path, sha in provenance['source_hashes'].items():
        if digest(path) != sha:
            raise ValueError(f'Calibration source changed: {path}')
    with safe_open(source_files[0], framework='pt', device='cpu') as model:
        embeddings = model.get_tensor('model.embed_tokens.weight')
        layer = {key.removeprefix('model.layers.0.'): model.get_tensor(key)
                 for key in model.keys() if key.startswith('model.layers.0.')}
        values = {}
        for name, rows in split.items():
            values[name] = [torch.nn.functional.embedding(torch.tensor(row['input_ids']), embeddings).unsqueeze(0)
                            for row in rows]
            if any(not torch.isfinite(x).all() for x in values[name]):
                raise ValueError('Nonfinite dense embedding output')
    if not layer:
        raise ValueError('Dense layer 0 is missing')
    output.mkdir(parents=True)
    torch.save({'layer_state': layer, 'inputs': values}, output / 'layer-inputs.pt')
    write_json(output / 'documents.json', split)
    files = [*source_files, inputs / 'inputs.json', inputs / 'provenance.json', Path(__file__).resolve()]
    write_json(output / 'provenance.json', {
        'state': 'prepared_not_executed', 'dense': str(dense), 'seed': 7, 'layer_index': 0,
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'files': {str(path.resolve()): digest(path) for path in files},
        'artifact_sha256': digest(output / 'layer-inputs.pt'), 'documents_sha256': digest(output / 'documents.json'),
        'documents': {name: len(rows) for name, rows in split.items()},
        'tokens': {name: sum(len(row['input_ids']) for row in rows) for name, rows in split.items()},
        'input_semantics': 'exact dense embedding lookup before layer-0 input normalization',
        'weighting': 'unweighted activations; no source weighting by input scaling through nonlinear layer',
        'scope': 'layer-0 grouped optimization, selected QKV exports; no full-model recovery claim',
        'paired_scope': 'clean and noisy layer-0 inputs coincide; later-layer noisy propagation remains separate',
    })
    print('Prepared', output, {name: len(rows) for name, rows in split.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.inputs.resolve(), args.output.resolve())
