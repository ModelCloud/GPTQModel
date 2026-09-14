"""Lock real NM documents and audit them against the held-out tasks."""

import argparse
import hashlib
import json
from pathlib import Path
import random

import pyarrow.parquet as pq

from scripts.check_calibration_disjointness import normalize, question_text
from scripts.validate_qvq_gsq_layers import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-inputs', type=Path, required=True)
    parser.add_argument('--platinum', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=128)
    args = parser.parse_args()
    if args.samples < 128:
        raise ValueError('Use at least 128 calibration documents')
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path('/monster/data/model/dataset/nm-calibration/llm.parquet')
    dense = Path('/monster/data/model/Llama-3.2-1B-Instruct')
    audit = Path('dataset/calibration-fisher-scaling-v2/yaqa182_nm10000.disjointness.json')
    historical = json.loads(audit.read_text())
    binding = next(b for b in historical['calibration_bindings'] if b['path'] == str(source))
    if historical['status'] != 'pass' or digest(source) != binding['sha256']:
        raise ValueError('NM source does not match the audited corpus')
    rows = pq.read_table(source).to_pylist()
    selected = random.Random(7).sample(range(len(rows)), args.samples)
    heldout_source = Path('dataset/divergence300-v1/divergence300-locked.jsonl')
    heldout_questions = {normalize(question_text(json.loads(line)['messages']))
                         for line in heldout_source.read_text().splitlines()}
    platinum_questions = {normalize(json.loads(line)['question']) for line in args.platinum.read_text().splitlines()}
    questions = [normalize(question_text(rows[i]['messages'])) for i in selected]
    if len(set(questions)) != args.samples or set(questions) & (heldout_questions | platinum_questions):
        raise ValueError('Calibration duplicates or held-out overlap')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(dense, local_files_only=True)
    train = []
    for i in selected:
        ids = tokenizer.apply_chat_template(rows[i]['messages'], tokenize=True,
                                            add_generation_prompt=False, return_dict=False)[:2048]
        train.append(dict(source_row=i, split='train', source_name='nm', input_ids=ids,
                          token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
    if len({row['token_sha256'] for row in train}) != args.samples or min(len(row['input_ids']) for row in train) < 2:
        raise ValueError('Invalid or duplicate token sequences')
    previous = json.loads((args.reference_inputs/'inputs.json').read_text())
    write_json(args.output/'inputs.json', dict(train=train, heldout=previous['heldout']))
    write_json(args.output/'selected-source.json', [dict(source_row=i, **rows[i]) for i in selected])
    provenance = dict(dense=str(dense), seed=7, train_rows=args.samples, eval_rows=len(previous['heldout']), token_cap=2048,
                      train_tokens=sum(len(row['input_ids']) for row in train),
                      min_train_tokens=min(len(row['input_ids']) for row in train),
                      max_train_tokens=max(len(row['input_ids']) for row in train),
                      selection='random.Random(7).sample without replacement; native NM documents, no padding',
                      weighting='unweighted NM documents', source_rows=selected,
                      source_hashes={str(p.resolve()): digest(p) for p in
                                     (source, audit, heldout_source, args.platinum)},
                      inputs_sha256=digest(args.output/'inputs.json'),
                      question_disjointness='pass against all locked D300 and all 1209 Platinum questions',
                      script_sha256=digest(__file__))
    write_json(args.output/'provenance.json', provenance)
    print(json.dumps({k: v for k, v in provenance.items() if k != 'source_rows'}, indent=2))


if __name__ == '__main__':
    main()
