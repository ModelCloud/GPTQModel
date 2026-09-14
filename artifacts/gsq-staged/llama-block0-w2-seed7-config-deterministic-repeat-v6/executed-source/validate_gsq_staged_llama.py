"""Real block-0 Llama GSQ staged experiment; selected-layer artifacts only."""

import argparse
from datetime import datetime, timezone
import shutil
import sys
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.validate_qvq_gsq_layers import digest, write_json


def execute(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = args.inputs.resolve()
    provenance = json.loads((source/'provenance.json').read_text())
    documents = json.loads((source/'inputs.json').read_text())
    started = datetime.now(timezone.utc).isoformat()
    if set(tuple(row['input_ids']) for row in documents['train']) & set(
            tuple(row['input_ids']) for row in documents['heldout']):
        raise ValueError('Train and held-out token sequences overlap')
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not uuid.startswith('GPU-') or ',' in uuid or not os.environ.get('GPU_ALLOCATOR_LEASE_ID'):
        raise ValueError('Requires an exclusive single GPU allocator lease')
    for _ in range(3):
        inventory = subprocess.check_output(['nvidia-smi', '--id='+uuid,
            '--query-gpu=index,pci.bus_id,uuid,name,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
        fields = [field.strip() for field in inventory.split(',')]
        if fields[2] != uuid or int(fields[4]) > 8 or int(fields[5]):
            raise RuntimeError('GPU is not idle: '+inventory)
        print('IDLE', inventory, flush=True)
        time.sleep(1)
    import torch
    from transformers import AutoModelForCausalLM
    from gptqmodel.quantization import GSQTrainingConfig
    from gptqmodel.quantization.gsq_training import quantize_llama_gsq_block

    torch.use_deterministic_algorithms(args.deterministic)
    torch.manual_seed(7)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    files = [Path(__file__).resolve(), Path('gptqmodel/quantization/gsq_training.py').resolve(),
             Path('gptqmodel/quantization/gsq_training_config.py').resolve(),
             source/'inputs.json', source/'provenance.json',
             Path(provenance['dense'])/'model.safetensors', Path(provenance['dense'])/'config.json']
    report = dict(state='loading', commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_hashes={str(path): digest(path) for path in files}, inventory=inventory,
                  torch=str(torch.__version__), cuda=torch.version.cuda, seed=7, bits=args.bits, epochs=args.epochs,
                  qk_steps=args.qk_steps, group_size=128, attention='eager', cache=False, graphs=False, precision='float32',
                  source_model=provenance['dense'], scope='block0, local reconstruction; no paper-reproduction claim')
    training_config = GSQTrainingConfig(enabled=True, epochs=args.epochs, qk_steps=args.qk_steps,
                                         damp_percent=args.damp_percent)
    report['gsq_training'] = training_config.to_dict()
    report['deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
    report['cublas_workspace_config'] = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    report['qk_learning_rate_decay'] = 'constant'
    report['mlp_initializer_timing'] = 'after_attention'
    report['qk_damp_percent'] = args.damp_percent
    report['initializer_damp_percent'] = args.damp_percent
    report.update(started_utc=started, argv=sys.argv, run_id=output.name,
                  weighting='unweighted documents; not YAQA 1.25/N-mode reproduction',
                  gpu_properties=str(torch.cuda.get_device_properties(0)))
    snapshot = output/'executed-source'
    snapshot.mkdir()
    for path in files:
        if path.suffix == '.py':
            shutil.copy2(path, snapshot/path.name)
    shutil.copy2(source/'inputs.json', output/'inputs.json')
    write_json(output/'report.json', report)
    (output/'model_run.md').write_text('# Staged GSQ block experiment\n\n'
        'Status: running. Selected-layer experimental artifact; not a portable full model.\n\n'
        'Effective configuration and provenance:\n```json\n'+json.dumps(report, indent=2)+'\n```\n')
    model = AutoModelForCausalLM.from_pretrained(provenance['dense'], dtype=torch.float32,
                attn_implementation='eager', local_files_only=True).eval()
    layer = model.model.layers[0].to('cuda')
    embedding = model.model.embed_tokens.to('cuda')
    rotary = model.model.rotary_emb.to('cuda')
    batches = {}
    with torch.no_grad():
        for split in ('train', 'heldout'):
            batches[split] = []
            for row in documents[split]:
                ids = torch.tensor(row['input_ids'], device='cuda')[None]
                hidden = embedding(ids)
                length = ids.shape[1]
                positions = torch.arange(length, device='cuda')[None]
                kwargs = dict(position_embeddings=rotary(hidden, positions), use_cache=False,
                              attention_mask=torch.full((length, length), -torch.inf, device='cuda').triu(1)[None, None])
                batches[split].append((hidden, kwargs))
    report['state'] = 'initializing'
    write_json(output/'report.json', report)
    fitted, result = quantize_llama_gsq_block(layer, batches['train'], bits=args.bits, group_size=128,
                                             gsq=training_config, pack=False)
    seeds, stages = result['initializers'], result['stages']
    report['initializer'] = result['initializer_metadata']
    torch.save({'initializers': seeds, 'stages': stages, 'state_dict': fitted.cpu().state_dict()}, output/'stages.pt')
    fitted.to('cuda')
    import copy
    baseline = copy.deepcopy(layer)
    with torch.no_grad():
        for name, (weight, _) in seeds.items():
            baseline.get_submodule(name).weight.copy_(weight)
        rows = []
        for hidden, kwargs in batches['heldout']:
            teacher = layer(hidden, **kwargs)
            row = {'elements': teacher.numel()}
            for name, candidate in (('baseline', baseline), ('staged', fitted)):
                error = candidate(hidden, **kwargs).double()-teacher.double()
                row[name+'_sse'] = error.square().sum().item()
            rows.append(row)
    report.update(state='complete', heldout=rows, payload_sha256=digest(output/'stages.pt'),
                  finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--bits', type=int, choices=(2, 3, 4), default=4)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--qk-steps', type=int, default=2000)
    parser.add_argument('--damp-percent', type=float, default=.01)
    execute(parser.parse_args())
