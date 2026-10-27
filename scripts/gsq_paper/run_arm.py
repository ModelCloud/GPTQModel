# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Quantize one GSQ-paper arm and record quality, cost and per-block losses.

Arms share everything except the staged GSQ switch, so a GPTQ-only arm and a
GSQ arm differ only in whether the reconstruction stages run:

    gptq : GPTQ calibration on the 512-sequence split, no GSQ
    gsq  : GPTQ initialization on the same 512 sequences, then staged GSQ
           trained on the 4096-sequence split with held-out validation

Recipe follows the paper's Llama experiment (arXiv 2604.18556, App. G.1):
20 epochs, batch 64, group size 128, temperature 2 -> 0.05, multiplier
100 -> 500, Lion at 1e-4 (logits) / 5e-5 (scales), weight decay 1.0,
betas (.9, .95), cosine to 0.1, prior strength 6, noise std 0.01, and 2000
Q/K steps.

Each arm writes ``results.json`` (quality, timings, per-block stage losses)
and the packed model under ``<out>/model``.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_from_disk

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.looper.gsq_training_model import (
    finalize_llama_gsq_wrapper,
    quantize_llama_gsq_model,
)
from gptqmodel.models._const import DEVICE
from gptqmodel.quantization import FORMAT, GPTQConfig, GSQTrainingConfig

ZERO_SHOT_TASKS = ['arc_easy', 'arc_challenge', 'hellaswag', 'winogrande', 'piqa']


def documents(corpus, split, count):
    dataset = load_from_disk(str(Path(corpus) / split))
    if count > len(dataset):
        raise ValueError(f'{split} holds {len(dataset)} sequences, {count} requested')
    return [{'input_ids': dataset[index]['input_ids']} for index in range(count)]


def load_assignment(path):
    """Read a per-block width map produced by ``rco_allocate.py``."""
    payload = json.loads(Path(path).read_text())
    layer_bits = payload.get('layer_bits')
    if not layer_bits:
        raise ValueError(f'{path} has no layer_bits map')
    bits = {int(index): int(width) for index, width in layer_bits.items()}
    if sorted(bits) != sorted(torch.arange(len(bits)).tolist()):
        raise ValueError(f'{path} does not assign every block from 0: {sorted(bits)}')
    return bits


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', choices=['gptq', 'gsq'], required=True)
    parser.add_argument('--model', required=True, help='tokenizer/model path or hub id')
    parser.add_argument('--corpus', required=True, help='root produced by build_corpus.py')
    parser.add_argument('--out', required=True)
    parser.add_argument('--bits', type=int, default=3,
                        help='uniform width, ignored when --assignment is given')
    parser.add_argument('--assignment', default=None,
                        help='RCO assignment.json: one width per decoder block')
    parser.add_argument('--device', default='cuda:0',
                        help='accelerator for quantization and evaluation')
    parser.add_argument('--group-size', type=int, default=128)
    parser.add_argument('--train-seqs', type=int, default=4096)
    parser.add_argument('--gptq-seqs', type=int, default=512)
    parser.add_argument('--val-seqs', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--qk-steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--microbatch', type=int, default=16)
    parser.add_argument('--layers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=7,
                        help='vary for independent replicates')
    parser.add_argument('--restore-best', choices=['auto', 'on', 'off'], default='auto')
    parser.add_argument('--capture', choices=['cpu', 'disk', 'gpu'], default='cpu',
                        help='where each block input cache lives between stages')
    parser.add_argument('--deterministic', action='store_true',
                        help='enable torch deterministic algorithms; makes Q/K bit-identical')
    parser.add_argument('--resume', action='store_true',
                        help='reuse completed packed blocks from <out>/ckpt')
    parser.add_argument('--eval-limit', type=int, default=0,
                        help='cap rows per zero-shot task; 0 runs the full task')
    parser.add_argument('--skip-eval', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    (out / 'logs').mkdir(parents=True, exist_ok=True)

    bits = load_assignment(args.assignment) if args.assignment else args.bits
    config = GPTQConfig(bits=args.bits, group_size=args.group_size, sym=True,
                        desc_act=False, act_group_aware=False, format=FORMAT.GPTQ_V2,
                        device=DEVICE.CUDA, offload_to_disk=False)
    wrapper = GPTQModel.load(args.model, quantize_config=config, dtype=torch.bfloat16,
                             attn_implementation='sdpa')
    wrapper.model.to(args.device)

    train = documents(args.corpus, 'train', args.train_seqs)
    gptq = documents(args.corpus, 'gptq', args.gptq_seqs)
    validation = documents(args.corpus, 'validation', args.val_seqs)

    staged = args.variant == 'gsq'
    restore_best = {'auto': None, 'on': True, 'off': False}[args.restore_best]
    recipe = GSQTrainingConfig(
        enabled=staged, epochs=args.epochs, batch_size=args.batch_size,
        microbatch_size=args.microbatch, qk_steps=args.qk_steps,
        damp_percent=0.01, assignment_lr=1e-4, scale_lr=5e-5, weight_decay=1.0,
        betas=(.9, .95), temperature=(2., .05), multiplier=(100., 500.),
        warmup_steps=0, min_lr=.1, decay='cosine', seed=args.seed,
        initializer='gptq', optimizer='lion', restore_best=restore_best,
    )

    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    started = time.monotonic()
    run = quantize_llama_gsq_model(
        wrapper.model, train if staged else gptq,
        initialization_documents=gptq if staged else None,
        validation_documents=validation if staged else None,
        bits=bits, group_size=args.group_size, gsq=recipe,
        offload_capture=(args.capture != 'gpu'),
        capture_directory=(str(out / 'capture') if args.capture == 'disk' else None),
        checkpoint_directory=(str(out / 'ckpt') if args.resume else None),
        layer_indices=list(range(args.layers)))
    quantize_seconds = time.monotonic() - started

    per_layer = []
    for block in run['blocks']:
        record = {'layer': block['layer_index']}
        for name, stage in block['stages'].items():
            record[name] = {
                'before': stage.get('hard_loss_before'),
                'after': stage.get('hard_loss_after'),
                'val_before': stage.get('validation_hard_loss_before'),
                'val_hist': [epoch.get('hard_loss')
                             for epoch in (stage.get('validation_history') or [])],
                'best_epoch': stage.get('best_epoch'),
                'epochs': len(stage.get('history') or []),
            }
        per_layer.append(record)

    model_dir = out / 'model'
    finalize_llama_gsq_wrapper(wrapper, run)
    wrapper.save(str(model_dir))
    del wrapper, run
    torch.cuda.empty_cache()

    evaluation, eval_seconds = None, 0.0
    if not args.skip_eval:
        from tests.eval import evaluate, get_eval_task_results
        suite = {'batch_size': 16}
        if args.eval_limit:
            suite['max_rows'] = args.eval_limit
        started = time.monotonic()
        raw = evaluate(
            model_or_id_or_path=str(model_dir), tasks=ZERO_SHOT_TASKS, backend=BACKEND.TORCH,
            batch_size=16, apply_chat_template=False,
            model_args={'dtype': 'bfloat16', 'attn_implementation': 'sdpa',
                        'device': args.device},
            gen_kwargs='do_sample=false,temperature=0.0,top_p=1.0,top_k=50',
            suite_kwargs=suite)
        evaluation = get_eval_task_results(raw)
        eval_seconds = time.monotonic() - started

    result = {
        'variant': args.variant,
        'bits': bits if isinstance(bits, int) else dict(sorted(bits.items())),
        'seed': args.seed,
        'restore_best': args.restore_best,
        'model': args.model,
        'device': torch.cuda.get_device_name(0),
        'deterministic': torch.are_deterministic_algorithms_enabled(),
        'torch': torch.__version__,
        'rocm': torch.version.rocm,
        'recipe': {
            'epochs': args.epochs, 'qk_steps': args.qk_steps,
            'batch_size': args.batch_size, 'microbatch': args.microbatch,
            'train_seqs': args.train_seqs, 'gptq_seqs': args.gptq_seqs,
            'val_seqs': args.val_seqs, 'group_size': args.group_size,
            'layers': args.layers, 'capture': args.capture,
        },
        'quantize_seconds': round(quantize_seconds, 1),
        'eval_seconds': round(eval_seconds, 1),
        'per_layer': per_layer,
        'eval': evaluation,
    }
    (out / 'results.json').write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({'variant': args.variant, 'bits': result['bits'],
                      'quantize_seconds': result['quantize_seconds'],
                      'eval': evaluation}, indent=2, default=str))


if __name__ == '__main__':
    main()
