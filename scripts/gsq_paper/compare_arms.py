# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Score saved arms with held-out NLL and print the paper comparison table.

Zero-shot accuracy moves in whole percentage points and saturates on easy
tasks, so it needs a large evaluation to resolve a small quantization gain.
Held-out negative log-likelihood on the validation split is far more
sensitive and is the metric used here to compare arms.

Per-document NLL is reported alongside the pooled number because a single
outlier document can dominate the mean: an arm that repairs one document GPTQ
collapses on will look better on the mean while being worse on the bulk of the
corpus. ``--per-doc-out`` writes the full breakdown.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk

from gptqmodel import BACKEND, GPTQModel


def _score_range(model_path, ids_by_document, device, backend, dtype):
    """Score one shard of the documents on one device."""
    # Pin placement explicitly. With every card visible, auto-placement may
    # split the model across devices, and a sharded forward then fails on the
    # first embedding lookup.
    model = GPTQModel.load(model_path, backend=getattr(BACKEND, backend),
                           dtype=dtype, attn_implementation='sdpa',
                           device_map={'': device})
    model.model.to(device).eval()
    per_document, total, tokens = [], 0.0, 0
    with torch.inference_mode():
        for index in ids_by_document:
            ids = torch.tensor([ids_by_document[index]], device=device)
            output = model.model(input_ids=ids, labels=ids)
            counted = ids.numel() - 1
            per_document.append((index, float(output.loss)))
            total += float(output.loss) * counted
            tokens += counted
    del model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return per_document, total, tokens


def score(model_path, ids_by_document, backend, dtype, devices=None):
    """Score every document, sharded across ``devices`` when more than one.

    Sharding is document-level, so each device sees whole sequences and the
    pooled NLL is identical to a single-device pass.
    """
    if not devices:
        devices = ['cuda:0']
    if len(devices) == 1:
        per_document, total, tokens = _score_range(
            model_path, ids_by_document, devices[0], backend, dtype)
    else:
        import multiprocessing as mp
        shards = {device: {} for device in devices}
        for position, index in enumerate(sorted(ids_by_document)):
            shards[devices[position % len(devices)]][index] = ids_by_document[index]
        context = mp.get_context('spawn')
        with context.Pool(len(devices)) as pool:
            outputs = pool.starmap(
                _score_range,
                [(model_path, shards[device], device, backend, dtype)
                 for device in devices])
        per_document, total, tokens = [], 0.0, 0
        for documents, shard_total, shard_tokens in outputs:
            per_document.extend(documents)
            total += shard_total
            tokens += shard_tokens
        per_document.sort()
    losses = [loss for _, loss in per_document]
    nll = total / tokens
    return {
        'nll': nll,
        'ppl': float(torch.tensor(nll).exp()),
        'tokens': tokens,
        'per_document': losses,
        'mean_per_document': sum(losses) / len(losses),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--models', nargs='+', required=True,
                        help='one or more <arm>/model directories')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='display name per model; defaults to the directory name')
    parser.add_argument('--dense', default=None,
                        help='optional unquantized reference model')
    parser.add_argument('--n', type=int, default=128, help='validation documents to score')
    parser.add_argument('--devices', default='cuda:0',
                        help='comma-separated devices for document-sharded scoring')
    parser.add_argument('--backend', default='TORCH')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--out', default='')
    parser.add_argument('--per-doc-out', default='')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_from_disk(str(Path(args.corpus) / 'validation'))
    if args.n > len(dataset):
        raise ValueError(f'validation holds {len(dataset)} sequences, {args.n} requested')
    ids_by_document = {index: dataset[index]['input_ids'] for index in range(args.n)}
    devices = [device for device in args.devices.split(',') if device]
    dtype = getattr(torch, args.dtype)

    labels = args.labels or [Path(path).parent.name for path in args.models]
    if len(labels) != len(args.models):
        raise SystemExit('--labels must match --models in length')

    results = {}
    if args.dense:
        dense = score(args.dense, ids_by_document, args.backend, dtype, devices)
        results[args.dense] = dense
        print(f'{"dense":<24} nll={dense["nll"]:.5f} ppl={dense["ppl"]:.3f}', flush=True)
    for label, model_path in zip(labels, args.models):
        result = score(model_path, ids_by_document, args.backend, dtype, devices)
        results[model_path] = result
        print(f'{label:<24} nll={result["nll"]:.5f} ppl={result["ppl"]:.3f}', flush=True)

    print()
    print(f'{"arm":<24} {"NLL":>9} {"PPL":>9} {"vs first":>10}')
    print('-' * 56)
    baseline = results[args.models[0]]['nll']
    rows = ([('dense', results[args.dense])] if args.dense else []) + \
           [(label, results[path]) for label, path in zip(labels, args.models)]
    for label, result in rows:
        delta = result['nll'] - baseline
        print(f'{label:<24} {result["nll"]:>9.5f} {result["ppl"]:>9.3f} {delta:>+10.5f}')

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    if args.per_doc_out:
        Path(args.per_doc_out).write_text(json.dumps(
            {label: results[path]['per_document']
             for label, path in zip(labels, args.models)}, indent=2))
    scored_tokens = sum(len(ids) - 1 for ids in ids_by_document.values())
    print(f'\nscored {len(ids_by_document)} validation documents ({scored_tokens} tokens)')


if __name__ == '__main__':
    main()
