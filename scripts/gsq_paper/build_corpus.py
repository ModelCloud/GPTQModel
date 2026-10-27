# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Build the GSQ paper's Llama calibration geometry from FineWeb-Edu.

The paper (Dadgarnia et al., arXiv 2604.18556, App. G.1) uses 4096 training,
128 validation and 512 GPTQ-initialization sequences of 4096 tokens each,
disjoint by source document. Documents are tokenized, concatenated back to
back and cut into fixed 4096-token chunks, which is the reference
implementation's ``make_concat_chunks`` behavior.

The three splits are written as separate ``datasets`` directories so a run can
never leak validation or GPTQ-init text into training:

    <out>/train      4096 sequences
    <out>/validation  128 sequences
    <out>/gptq        512 sequences
    <out>/meta.json  provenance

Because the chunks are cut from one shuffled stream, the split is disjoint by
source *document* but not by *position*; ``meta.json`` records the exact seed
and shuffle buffer so a rebuild is reproducible.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

SEQUENCE_LENGTH = 4096
TRAIN_SEQUENCES = 4096
VALIDATION_SEQUENCES = 128
GPTQ_SEQUENCES = 512


def build(*, model, out, seed=42, shuffle_buffer=100_000, source='HuggingFaceFW/fineweb-edu',
          subset='sample-10BT', progress_every=2000):
    total = TRAIN_SEQUENCES + VALIDATION_SEQUENCES + GPTQ_SEQUENCES
    tokenizer = AutoTokenizer.from_pretrained(model)
    stream = load_dataset(source, subset, split='train', streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)

    buffer, chunks, documents = [], [], 0
    for row in stream:
        documents += 1
        buffer.extend(tokenizer(row['text'], return_tensors=None)['input_ids'])
        while len(buffer) >= SEQUENCE_LENGTH:
            chunks.append(buffer[:SEQUENCE_LENGTH])
            del buffer[:SEQUENCE_LENGTH]
            if len(chunks) >= total:
                break
        if len(chunks) >= total:
            break
        if progress_every and documents % progress_every == 0:
            print(f'documents={documents} chunks={len(chunks)}/{total}', flush=True)

    if len(chunks) != total:
        raise RuntimeError(f'stream exhausted after {len(chunks)} of {total} chunks')

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    array = np.asarray(chunks, dtype=np.int32)
    splits = {
        'train': (0, TRAIN_SEQUENCES),
        'validation': (TRAIN_SEQUENCES, TRAIN_SEQUENCES + VALIDATION_SEQUENCES),
        'gptq': (TRAIN_SEQUENCES + VALIDATION_SEQUENCES, total),
    }
    for name, (start, stop) in splits.items():
        Dataset.from_dict({'input_ids': array[start:stop].tolist()}).save_to_disk(str(out / name))

    meta = {
        'sequence_length': SEQUENCE_LENGTH,
        'train': TRAIN_SEQUENCES,
        'validation': VALIDATION_SEQUENCES,
        'gptq': GPTQ_SEQUENCES,
        'source': f'{source}/{subset}',
        'seed': seed,
        'shuffle_buffer': shuffle_buffer,
        'source_documents_consumed': documents,
        'total_tokens': total * SEQUENCE_LENGTH,
        'tokenizer': model,
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B-Instruct',
                        help='tokenizer source; must match the model under test')
    parser.add_argument('--out', required=True, help='dataset root to create')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shuffle-buffer', type=int, default=100_000)
    parser.add_argument('--source', default='HuggingFaceFW/fineweb-edu')
    parser.add_argument('--subset', default='sample-10BT')
    args = parser.parse_args()
    build(model=args.model, out=args.out, seed=args.seed,
          shuffle_buffer=args.shuffle_buffer, source=args.source, subset=args.subset)


if __name__ == '__main__':
    main()
