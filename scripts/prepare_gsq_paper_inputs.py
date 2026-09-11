"""Prepare the GSQ author's FineWeb-Edu calibration construction with full provenance."""

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dense', type=Path, required=True)
    parser.add_argument('--heldout-inputs', type=Path, required=True)
    parser.add_argument('--train-samples', type=int, default=4096)
    parser.add_argument('--sequence-length', type=int, default=4096)
    parser.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu')
    parser.add_argument('--subset', default='sample-10BT')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shuffle-buffer-size', type=int, default=100_000)
    args = parser.parse_args()
    if args.train_samples < 1 or args.sequence_length < 1 or args.shuffle_buffer_size < 1:
        parser.error('Sample count, sequence length and shuffle buffer must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.dense, local_files_only=True)
    revision = HfApi().dataset_info(args.dataset).sha
    dataset = load_dataset(args.dataset, args.subset, split='train', streaming=True, revision=revision)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
    token_buffer = []
    chunks = []
    rows = []
    tokens_consumed = 0
    for stream_index, row in enumerate(dataset):
        ids = tokenizer(row['text'], return_tensors=None)['input_ids']
        token_buffer.extend(ids)
        tokens_consumed += len(ids)
        rows.append({
            key: row.get(key)
            for key in ('id', 'dump', 'url', 'file_path', 'language', 'language_score', 'token_count', 'score',
                        'int_score')
        } | {'shuffled_stream_index': stream_index, 'tokenized_length': len(ids)})
        while len(token_buffer) >= args.sequence_length:
            chunks.append({'input_ids': token_buffer[:args.sequence_length]})
            del token_buffer[:args.sequence_length]
            if len(chunks) % 64 == 0 or len(chunks) == args.train_samples:
                elapsed = time.perf_counter()-clock
                print(
                    f'CALIBRATION chunks={len(chunks)}/{args.train_samples} rows={len(rows)} '
                    f'tokens={tokens_consumed} elapsed={elapsed:.1f}s',
                    flush=True,
                )
            if len(chunks) >= args.train_samples:
                break
        if len(chunks) >= args.train_samples:
            break
    if len(chunks) != args.train_samples or any(len(row['input_ids']) != args.sequence_length for row in chunks):
        raise RuntimeError('FineWeb-Edu stream ended before the requested fixed-length calibration was prepared')
    heldout_source = json.loads(args.heldout_inputs.read_text())
    heldout = heldout_source['heldout']
    if not heldout:
        raise ValueError('Held-out source contains no documents')
    if {tuple(row['input_ids']) for row in chunks} & {tuple(row['input_ids']) for row in heldout}:
        raise ValueError('Prepared calibration overlaps the held-out input IDs')
    inputs_path = args.output/'inputs.json'
    with open(inputs_path, 'w') as handle:
        json.dump({'train': chunks, 'heldout': heldout}, handle, separators=(',', ':'))
        handle.write('\n')
    rows_path = args.output/'fineweb_rows.json.gz'
    with gzip.open(rows_path, 'wt') as handle:
        json.dump(rows, handle, separators=(',', ':'))
        handle.write('\n')
    tokenizer_files = sorted(
        path for path in args.dense.iterdir()
        if path.is_file() and ('token' in path.name or path.name in ('config.json', 'generation_config.json'))
    )
    author_source = Path('/root/devin-worker/repos/GSQ-author/src/data/dataset.py')
    provenance = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'started_utc': started,
        'argv': sys.argv,
        'qvq_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'dense': str(args.dense.resolve()),
        'dataset': args.dataset,
        'dataset_subset': args.subset,
        'dataset_revision': revision,
        'streaming': True,
        'shuffle_seed': args.seed,
        'shuffle_buffer_size': args.shuffle_buffer_size,
        'construction': 'author make_concat_chunks: tokenize each shuffled text, concatenate, no EOS, fixed chunks',
        'train_rows': len(rows),
        'train_samples': len(chunks),
        'token_cap': args.sequence_length,
        'calibration_tokens': args.train_samples*args.sequence_length,
        'source_tokens_consumed': tokens_consumed,
        'discarded_tail_tokens': len(token_buffer),
        'heldout_source': str(args.heldout_inputs.resolve()),
        'heldout_rows': len(heldout),
        'author_repository': '/root/devin-worker/repos/GSQ-author',
        'author_commit': subprocess.check_output(
            ['git', '-C', '/root/devin-worker/repos/GSQ-author', 'rev-parse', 'HEAD'], text=True).strip(),
        'files': {},
    }
    files = [Path(__file__).resolve(), author_source, inputs_path, rows_path, args.heldout_inputs, *tokenizer_files]
    provenance['files'] = {str(path.resolve()): digest(path) for path in files}
    write_json(args.output/'provenance.json', provenance)
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    main()
