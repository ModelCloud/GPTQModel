"""Matched GSM8K Platinum evaluation of saved GSQ experimental checkpoints."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.validate_qvq_gsq_layers import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--max-rows', type=int)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not uuid.startswith('GPU-') or ',' in uuid or not os.environ.get('GPU_ALLOCATOR_LEASE_ID'):
        raise ValueError('Requires exclusive UUID lease')
    idle = 0
    for _ in range(60):
        inventory = subprocess.check_output(['nvidia-smi', '--id='+uuid,
            '--query-gpu=uuid,name,pci.bus_id,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
        fields = [v.strip() for v in inventory.split(',')]
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                             '--format=csv,noheader'], text=True)
        idle = idle+1 if int(fields[-2]) <= 8 and int(fields[-1]) == 0 and uuid not in processes else 0
        print('PREFLIGHT', inventory, idle, flush=True)
        if idle == 3:
            break
        time.sleep(1)
    else:
        raise ValueError('Idle preflight timeout')
    import torch
    import evalution
    from tests.eval import evaluate, get_eval_task_results
    from gptqmodel.utils.backend import BACKEND

    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings = dict(dtype='float16', device='cuda:0', attn_implementation='eager', seed=7)
    suite = dict(dataset_path=str(args.dataset.resolve()), dataset_name=None, max_rows=args.max_rows,
                 fewshot_seed=7, max_new_tokens=256, stream=False)
    report = dict(state='running', run_id=args.output.name, arm=args.arm, argv=sys.argv,
                  model=str(args.model.resolve()), model_args=settings, suite_kwargs=suite,
                  batch_size=args.batch_size, task='gsm8k_platinum_cot', apply_chat_template=True,
                  qvq_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  evalution_source=evalution.__file__, torch=str(torch.__version__), cuda=torch.version.cuda,
                  gpu=inventory, source_sha256=digest(__file__),
                  model_hashes={p.name: digest(p) for p in args.model.iterdir() if p.is_file()},
                  dataset_hashes={p.name: digest(p) for p in args.dataset.iterdir() if p.is_file()})
    write_json(args.output/'run.json', report)
    (args.output/'evaluation.md').write_text('# GSM8K Platinum evaluation\n\nRunning.\n\n```json\n'
                                           +json.dumps(report, indent=2)+'\n```\n')
    started = time.monotonic()
    try:
        result = evaluate(model_or_id_or_path=str(args.model.resolve()), tasks=['gsm8k_platinum_cot'],
                          backend=BACKEND.TORCH, model_args=settings, batch_size=args.batch_size,
                          apply_chat_template=True, gen_kwargs='do_sample=false,temperature=0.0',
                          suite_kwargs=suite, output_path=str(args.output/'raw.json'))
        report.update(state='complete', seconds=time.monotonic()-started,
                      metrics=get_eval_task_results(result), raw_sha256=digest(args.output/'raw.json'))
        print('RESULT', json.dumps(report['metrics']), flush=True)
    except Exception as error:
        report.update(state='failed', error=repr(error), seconds=time.monotonic()-started)
        raise
    finally:
        write_json(args.output/'run.json', report)
        (args.output/'evaluation.md').write_text('# GSM8K Platinum evaluation\n\n```json\n'
                                               +json.dumps(report, indent=2)+'\n```\n')


if __name__ == '__main__':
    main()
