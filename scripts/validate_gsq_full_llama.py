"""Full Llama staged/GPTQ model quantization, public export/reload and logits."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.validate_qvq_gsq_layers import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=('baseline', 'staged'), required=True)
    parser.add_argument('--train-samples', type=int, default=128)
    parser.add_argument('--bits', type=int, choices=(2, 3, 4), default=2)
    parser.add_argument('--initializer', choices=('gptq', 'gptq_signed'), default='gptq',
                        help='Matched GPTQ prior for both arms; signed uses the author-style scalar range search')
    parser.add_argument('--batch-size', type=int, default=1, help='Documents per staged optimizer update')
    parser.add_argument('--microbatch-size', type=int, default=1, help='Documents per staged forward pass')
    parser.add_argument('--train-precision', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--epochs', type=int, default=5, help='Attention/MLP training epochs; Q/K budget is separate')
    parser.add_argument('--qk-steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--optimizer', choices=('lion', 'adamw'), default='lion')
    parser.add_argument('--lifecycle', choices=('dedicated', 'public'), default='dedicated')
    parser.add_argument('--attn-implementation', choices=('eager', 'sdpa'), default='eager')
    parser.add_argument('--offload-capture', action='store_true')
    args = parser.parse_args()
    if args.lifecycle == 'public' and args.arm != 'staged':
        parser.error('Public lifecycle validation currently selects the staged arm; disabled uses ordinary GPTQ')
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.inputs/'provenance.json').read_text())
    documents = json.loads((args.inputs/'inputs.json').read_text())
    if args.train_samples < 128 or len(documents['train']) != args.train_samples:
        raise ValueError('Full-model validation requires the requested calibration count, at least 128')
    if {tuple(v['input_ids']) for v in documents['train']} & {tuple(v['input_ids']) for v in documents['heldout']}:
        raise ValueError('Calibration/evaluation overlap')
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not uuid.startswith('GPU-') or ',' in uuid or not os.environ.get('GPU_ALLOCATOR_LEASE_ID'):
        raise ValueError('Requires exclusive UUID lease')
    idle = 0
    for _ in range(60):
        inventory = subprocess.check_output(['nvidia-smi', '--id='+uuid,
            '--query-gpu=index,pci.bus_id,uuid,name,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
        fields = [v.strip() for v in inventory.split(',')]
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                             '--format=csv,noheader'], text=True)
        idle = (
            idle+1
            if fields[2] == uuid and int(fields[4]) <= 8 and int(fields[5]) == 0 and uuid not in processes
            else 0
        )
        print('IDLE', inventory, idle, flush=True)
        if idle == 3:
            break
        time.sleep(1)
    else:
        raise ValueError('Idle preflight timeout')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gptqmodel import GPTQModel
    from gptqmodel.looper.gsq_training_model import quantize_llama_gsq_model, save_llama_gsq_model
    from gptqmodel.quantization import GSQTrainingConfig
    from gptqmodel.utils.backend import BACKEND
    from scripts.p32_twenty.scorecard import logits_metrics

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    config = GSQTrainingConfig(enabled=args.arm == 'staged', initializer=args.initializer,
                               batch_size=args.batch_size, microbatch_size=args.microbatch_size, epochs=args.epochs,
                               optimizer=args.optimizer, qk_steps=args.qk_steps, seed=args.seed)
    files = [Path(__file__), Path('gptqmodel/looper/gsq_training_model.py'),
             Path('gptqmodel/looper/gsq_training_capture.py'), Path('gptqmodel/quantization/gsq_training.py'),
             Path('gptqmodel/quantization/gsq_training_config.py'), Path('scripts/p32_twenty/scorecard.py'),
             Path('gptqmodel/quantization/gsq_initialization.py'), Path('gptqmodel/quantization/gptq.py'),
             Path('gptqmodel/quantization/gsq_batching.py'), Path('gptqmodel/models/base.py'),
             Path('gptqmodel/utils/calibration.py'),
             Path('gptqmodel/quantization/quantizer.py'), Path('gptqmodel/quantization/config.py'),
             args.inputs/'inputs.json', args.inputs/'provenance.json',
             Path(source['dense'])/'model.safetensors', Path(source['dense'])/'config.json']
    report = dict(state='loading', arm=args.arm, run_id=args.output.name, lifecycle=args.lifecycle,
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  argv=sys.argv, source_model=source['dense'], gsq_training=config.to_dict(),
                  bits=args.bits, group_size=128, train_precision=args.train_precision, export_precision='float16',
                  attention_implementation=args.attn_implementation, offload_capture=args.offload_capture,
                  calibration_samples=len(documents['train']),
                  calibration_tokens=sum(len(row['input_ids']) for row in documents['train']),
                  calibration_token_cap=source['token_cap'],
                  weighting='unweighted fixed-length documents; W4 Llama 3.2 1B is not the paper rate/model',
                  started_utc=datetime.now(timezone.utc).isoformat(), inventory=inventory,
                  torch=str(torch.__version__), cuda=torch.version.cuda,
                  gpu=str(torch.cuda.get_device_properties(0)),
                  deterministic_algorithms=True, cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  gpu_allocator_lease_id=os.environ.get('GPU_ALLOCATOR_LEASE_ID'),
                  gptqmodel_cuda_block=os.environ.get('GPTQMODEL_CUDA_BLOCK'),
                  source_hashes={str(p.resolve()): digest(p) for p in files})
    snapshot = args.output/'executed-source'
    snapshot.mkdir()
    for file in files:
        if file.suffix == '.py':
            shutil.copyfile(file, snapshot/file.name)
    shutil.copyfile(args.inputs/'inputs.json', args.output/'inputs.json')
    write_json(args.output/'report.json', report)
    (args.output/'model_run.md').write_text(f'# Full W{args.bits} model experiment\n\nStatus: running.\n\n```json\n'
                                           +json.dumps(report, indent=2)+'\n```\n')
    try:
        model = AutoModelForCausalLM.from_pretrained(source['dense'], dtype=getattr(torch, args.train_precision),
                    device_map={'': 'cuda:0'}, attn_implementation=args.attn_implementation,
                    local_files_only=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(source['dense'], local_files_only=True)
        for name in ('teacher', 'before_reload', 'after_reload'):
            (args.output/name).mkdir()
        with torch.inference_mode():
            for i, row in enumerate(documents['heldout']):
                logits = model(torch.tensor([row['input_ids']], device='cuda'), use_cache=False).logits[0]
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite dense teacher')
                torch.save(logits.cpu(), args.output/'teacher'/f'{i}.pt')
        report['state'] = 'quantizing'
        write_json(args.output/'report.json', report)
        if args.lifecycle == 'public':
            from gptqmodel.models.definitions.llama import LlamaQModel
            from gptqmodel.quantization import FORMAT, GPTQConfig

            qcfg = GPTQConfig(bits=args.bits, group_size=128, sym=True, desc_act=False,
                              act_group_aware=False, format=FORMAT.GPTQ_V2, offload_to_disk=False,
                              device='cuda:0', gsq_training=config)
            wrapper = LlamaQModel(model=model, quantized=False, quantize_config=qcfg,
                                  tokenizer=tokenizer, model_local_path=source['dense'])
            wrapper.quantize(documents['train'], backend=BACKEND.TORCH, calibration_sort=None,
                              calibration_data_min_length=1)
            run = wrapper.gsq_training_run
            wrapper.save(str(args.output/'model'))
            del wrapper
        else:
            run = quantize_llama_gsq_model(
                model,
                documents['train'],
                bits=args.bits,
                group_size=128,
                gsq=config,
                offload_capture=args.offload_capture,
            )
            save_llama_gsq_model(model, run, args.output/'model', tokenizer=tokenizer, source_model=source['dense'])
        write_json(args.output/'training.json', run)
        with torch.inference_mode():
            for i, row in enumerate(documents['heldout']):
                logits = model(torch.tensor([row['input_ids']], device='cuda'), use_cache=False).logits[0]
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite pre-reload model')
                torch.save(logits.cpu(), args.output/'before_reload'/f'{i}.pt')
        del model
        torch.cuda.empty_cache()
        restored = GPTQModel.load(str(args.output/'model'), backend=BACKEND.TORCH, device='cuda:0',
                                  dtype=torch.float16, attn_implementation=args.attn_implementation)
        rows = []
        for i, row in enumerate(documents['heldout']):
            with torch.inference_mode():
                ids = torch.tensor([row['input_ids']], device='cuda')
                logits = restored.model(ids, use_cache=False).logits[0]
                before = torch.load(args.output/'before_reload'/f'{i}.pt', weights_only=True).cuda()
                teacher = torch.load(args.output/'teacher'/f'{i}.pt', weights_only=True).cuda()
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite reloaded model')
                torch.save(logits.cpu(), args.output/'after_reload'/f'{i}.pt')
                chunks = [logits_metrics(logits[j:j+32].float(), teacher[j:j+32])
                          for j in range(0, len(logits), 32)]
                metrics = {k: sum(c[k]*c['tokens'] for c in chunks)/len(logits)
                           for k in chunks[0] if k != 'tokens'}
                metrics.update(tokens=len(logits), mse=(logits.float()-teacher).square().mean().item(),
                               reload_exact=torch.equal(logits, before),
                               reload_max_abs=(logits-before).abs().max().item(),
                               teacher_sha256=digest(args.output/'teacher'/f'{i}.pt'))
                rows.append(metrics)
            report.update(state=f'evaluation {i+1}/{len(documents["heldout"])}', rows=rows)
            write_json(args.output/'report.json', report)
            print('MODEL', report['state'], json.dumps(metrics), flush=True)
        report.update(state='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
                      model_hashes={str(p.relative_to(args.output/'model')): digest(p)
                                    for p in (args.output/'model').rglob('*') if p.is_file()})
    except Exception as error:
        report.update(state='failed', error_type=type(error).__name__, error=str(error))
        write_json(args.output/'report.json', report)
        raise
    write_json(args.output/'report.json', report)


if __name__ == '__main__':
    main()
