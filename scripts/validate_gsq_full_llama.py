"""Full Llama staged/GPTQ model quantization, public export/reload and logits."""

import argparse
import atexit
import json
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.validate_qvq_gsq_layers import digest, write_json


class ResourceMonitor:
    def __init__(self, interval=60):
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def run(self):
        while not self.stop_event.wait(self.interval):
            status = {}
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith(('VmRSS:', 'VmHWM:')):
                    key, value = line.split(':', 1)
                    status[key] = int(value.split()[0])
            memory = {}
            for line in Path('/proc/meminfo').read_text().splitlines():
                if line.startswith(('MemAvailable:', 'Cached:')):
                    key, value = line.split(':', 1)
                    memory[key] = int(value.split()[0])
            print(
                'RESOURCE'
                f' rss_gib={status.get("VmRSS", 0)/2**20:.3f}'
                f' peak_rss_gib={status.get("VmHWM", 0)/2**20:.3f}'
                f' mem_available_gib={memory.get("MemAvailable", 0)/2**20:.3f}'
                f' cached_gib={memory.get("Cached", 0)/2**20:.3f}',
                flush=True,
            )


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
    model_config = json.loads((Path(source['dense'])/'config.json').read_text())
    hidden_bytes = args.train_samples*source['token_cap']*model_config['hidden_size']*2
    optimizer_batch_bytes = args.batch_size*source['token_cap']*model_config['hidden_size']*2
    total_memory = os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    available_memory = int(next(
        line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:')
    ))*1024
    free_disk = shutil.disk_usage(args.output.parent).free
    estimated_cpu_peak = 2*optimizer_batch_bytes+2*1024**3
    if estimated_cpu_peak > total_memory//4 or estimated_cpu_peak > available_memory//2:
        raise ValueError('Estimated CPU peak exceeds the bounded-memory test budget')
    if args.offload_capture and hidden_bytes*1.1 > free_disk:
        raise ValueError('Disk-backed capture spool exceeds available disk')
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
                  capture_storage='disk' if args.offload_capture else 'device',
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
                  estimated_capture_spool_bytes=hidden_bytes if args.offload_capture else 0,
                  estimated_cpu_peak_bytes=estimated_cpu_peak,
                  host_total_memory_bytes=total_memory, host_available_memory_bytes=available_memory,
                  free_output_disk_bytes=free_disk,
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
    print(
        'RESOURCE_BUDGET'
        f' estimated_cpu_peak_gib={estimated_cpu_peak/2**30:.3f}'
        f' capture_spool_gib={(hidden_bytes if args.offload_capture else 0)/2**30:.3f}'
        f' host_available_gib={available_memory/2**30:.3f}',
        flush=True,
    )
    monitor = ResourceMonitor()
    monitor.start()
    atexit.register(monitor.stop)
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
                capture_directory=args.output/'capture-staging' if args.offload_capture else None,
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
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        write_json(args.output/'report.json', report)
        monitor.stop()
        atexit.unregister(monitor.stop)
        raise
    report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    write_json(args.output/'report.json', report)
    monitor.stop()
    atexit.unregister(monitor.stop)


if __name__ == '__main__':
    main()
