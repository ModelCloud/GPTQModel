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
    if len(documents['train']) < 128:
        raise ValueError('Real-model staged validation requires at least 128 calibration documents')
    started = datetime.now(timezone.utc).isoformat()
    if set(tuple(row['input_ids']) for row in documents['train']) & set(
            tuple(row['input_ids']) for row in documents['heldout']):
        raise ValueError('Train and held-out token sequences overlap')
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not uuid.startswith('GPU-') or ',' in uuid or not os.environ.get('GPU_ALLOCATOR_LEASE_ID'):
        raise ValueError('Requires an exclusive single GPU allocator lease')
    idle = 0
    for _ in range(60):
        inventory = subprocess.check_output(['nvidia-smi', '--id='+uuid,
            '--query-gpu=index,pci.bus_id,uuid,name,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
        fields = [field.strip() for field in inventory.split(',')]
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                             '--format=csv,noheader'], text=True)
        idle = idle+1 if fields[2] == uuid and int(fields[4]) <= 8 and int(fields[5]) == 0 and uuid not in processes else 0
        print('IDLE', inventory, idle, flush=True)
        if idle == 3:
            break
        time.sleep(1)
    else:
        raise RuntimeError('GPU idle preflight timeout')
    import torch
    from transformers import AutoModelForCausalLM
    from gptqmodel.quantization import GSQTrainingConfig
    from gptqmodel.quantization.gsq_training import quantize_llama_gsq_block

    torch.use_deterministic_algorithms(args.deterministic)
    torch.manual_seed(7)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    model_files = sorted(Path(provenance['dense']).glob('model*.safetensors'))
    if not model_files:
        raise FileNotFoundError(f'No model safetensors found under {provenance["dense"]}')
    files = [Path(__file__).resolve(), Path('gptqmodel/quantization/gsq_training.py').resolve(),
             Path('gptqmodel/quantization/gsq_training_config.py').resolve(),
             Path('gptqmodel/quantization/gsq_initialization.py').resolve(),
             Path('gptqmodel/quantization/gsq_batching.py').resolve(),
             Path('gptqmodel/quantization/gptq.py').resolve(),
             Path('gptqmodel/quantization/quantizer.py').resolve(),
             Path('gptqmodel/looper/gsq_training_capture.py').resolve(),
             source/'inputs.json', source/'provenance.json',
             *model_files, Path(provenance['dense'])/'config.json']
    report = dict(state='loading', commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_hashes={str(path): digest(path) for path in files}, inventory=inventory,
                  torch=str(torch.__version__), cuda=torch.version.cuda, seed=7, bits=args.bits, epochs=args.epochs,
                  qk_steps=args.qk_steps, group_size=128, attention='eager', cache=False, graphs=False,
                  precision=args.train_precision,
                  source_model=provenance['dense'], scope='block0, local reconstruction; no paper-reproduction claim')
    training_config = GSQTrainingConfig(enabled=True, epochs=args.epochs, qk_steps=args.qk_steps,
                                         damp_percent=args.damp_percent, initializer=args.initializer,
                                         batch_size=args.batch_size, microbatch_size=args.microbatch_size)
    report['calibration_samples'] = len(documents['train'])
    report['calibration_tokens'] = sum(len(row['input_ids']) for row in documents['train'])
    report['gsq_training'] = training_config.to_dict()
    report['capture'] = 'shared_inference' if args.shared_capture else 'manual_embedding'
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
    model = AutoModelForCausalLM.from_pretrained(provenance['dense'], dtype=getattr(torch, args.train_precision),
                attn_implementation='eager', local_files_only=True).eval()
    layer = model.model.layers[0].to('cuda')
    embedding = model.model.embed_tokens.to('cuda')
    rotary = model.model.rotary_emb.to('cuda')
    batches = {}
    if args.shared_capture:
        from gptqmodel.looper.gsq_training_capture import (
            capture_llama_gsq_inputs, prepare_llama_gsq_capture, quantize_llama_gsq_capture,
        )
        from gptqmodel.nn_modules.hooked_linear import HookedLinear

        for name, module in list(layer.named_modules()):
            if isinstance(module, torch.nn.Linear):
                parent, leaf = name.rsplit('.', 1)
                setattr(layer.get_submodule(parent), leaf, HookedLinear.from_linear(module))
        caches = {split: capture_llama_gsq_inputs(model, documents[split]) for split in ('train', 'heldout')}
        _, batches['heldout'] = prepare_llama_gsq_capture(layer, caches['heldout'])
        with torch.inference_mode():
            fitted, result = quantize_llama_gsq_capture(layer, caches['train'], bits=args.bits, group_size=128,
                                                       gsq=training_config, pack=False)
    else:
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
            if not torch.isfinite(teacher).all():
                raise ValueError('Nonfinite held-out teacher output')
            row = {'elements': teacher.numel()}
            for name, candidate in (('baseline', baseline), ('staged', fitted)):
                prediction = candidate(hidden, **kwargs)
                if not torch.isfinite(prediction).all():
                    raise ValueError(f'Nonfinite held-out {name} output')
                error = prediction.double()-teacher.double()
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
    parser.add_argument('--shared-capture', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--bits', type=int, choices=(2, 3, 4), default=4)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--qk-steps', type=int, default=2000)
    parser.add_argument('--damp-percent', type=float, default=.01)
    parser.add_argument('--initializer', choices=('gptq', 'gptq_signed'), default='gptq')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--microbatch-size', type=int, default=1)
    parser.add_argument('--train-precision', choices=('float32', 'bfloat16'), default='float32')
    args = parser.parse_args()
    output_existed = args.output.exists()
    try:
        execute(args)
    except Exception as error:
        report_path = args.output/'report.json'
        if not output_existed and report_path.exists():
            report = json.loads(report_path.read_text())
            report.update(state='failed', error_type=type(error).__name__, error=str(error),
                          finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(report_path, report)
        raise
