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


def score(model_path, documents, backend, dtype):
    model = GPTQModel.load(model_path, backend=getattr(BACKEND, backend),
                           dtype=dtype, attn_implementation='sdpa')
    model.model.to('cuda:0').eval()
    per_document, total, tokens = [], 0.0, 0
    with torch.inference_mode():
        for ids in documents:
            output = model.model(input_ids=ids, labels=ids)
            counted = ids.numel() - 1
            per_document.append(float(output.loss))
            total += float(output.loss) * counted
            tokens += counted
    del model
    torch.cuda.empty_cache()
    nll = total / tokens
    return {
        'nll': nll,
        'ppl': float(torch.tensor(nll).exp()),
        'tokens': tokens,
        'per_document': per_document,
        'mean_per_document': sum(per_document) / len(per_document),
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
    documents = [torch.tensor([dataset[index]['input_ids']], device='cuda:0')
                 for index in range(args.n)]
    dtype = getattr(torch, args.dtype)

    labels = args.labels or [Path(path).parent.name for path in args.models]
    if len(labels) != len(args.models):
        raise SystemExit('--labels must match --models in length')

    results = {}
    if args.dense:
        dense = score(args.dense, documents, args.backend, dtype)
        results[args.dense] = dense
        print(f'{"dense":<24} nll={dense["nll"]:.5f} ppl={dense["ppl"]:.3f}', flush=True)
    for label, model_path in zip(labels, args.models):
        result = score(model_path, documents, args.backend, dtype)
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
    print(f'\nscored {len(documents)} validation documents '
          f'({sum(document.numel() - 1 for document in documents)} tokens)')


if __name__ == '__main__':
    main()
