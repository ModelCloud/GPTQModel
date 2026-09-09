"""Validate matching Platinum prompts and report paired task accuracy."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.validate_qvq_gsq_layers import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--calibration-inputs', type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration_inputs.read_text())
    calibration_count = len(calibration['train'])
    calibration_sha = digest(args.calibration_inputs)
    output = args.root/'comparison.json'
    if output.exists():
        raise ValueError('Do not overwrite a comparison')
    arms = ('dense', 'baseline', 'staged')
    runs, samples, scores = {}, {}, {}
    from transformers import AutoTokenizer
    prompt_ids = None
    reference_settings = None
    quantization_recipe = None
    for arm in arms:
        path = args.root/(arm+'-full')
        run = json.loads((path/'run.json').read_text())
        raw = json.loads((path/'raw.json').read_text())
        if run['state'] != 'complete' or digest(path/'raw.json') != run['raw_sha256']:
            raise ValueError('Incomplete or changed evaluation')
        if arm != 'dense':
            quantization = json.loads((Path(run['model']).parent/'report.json').read_text())
            recipe = {key: quantization[key] for key in ('bits', 'group_size', 'source_model')}
            if quantization_recipe is None:
                quantization_recipe = recipe
            elif recipe != quantization_recipe:
                raise ValueError('Quantized model rate, group size or source differs between arms')
            if calibration_sha not in quantization['source_hashes'].values():
                raise ValueError('Model was not quantized using the specified calibration inputs')
        settings = {key: run[key] for key in ('model_args', 'suite_kwargs', 'batch_size', 'task',
                    'apply_chat_template', 'qvq_commit', 'source_sha256', 'dataset_hashes')}
        settings['versions'] = raw['versions']
        settings['task_metadata'] = raw['tests'][0]['metadata']
        if reference_settings is None:
            reference_settings = settings
        elif settings != reference_settings:
            raise ValueError('Evaluation configuration or source identity differs between arms')
        rows = raw['tests'][0]['samples']
        if len(rows) != 1209 or [row['index'] for row in rows] != list(range(1209)):
            raise ValueError('Requires complete native-order Platinum test')
        tokenizer = AutoTokenizer.from_pretrained(run['model'], local_files_only=True)
        ids = tokenizer([row['prompt'] for row in rows], add_special_tokens=False)['input_ids']
        if prompt_ids is None:
            prompt_ids = ids
        elif prompt_ids != ids:
            raise ValueError('Input token IDs differ between arms')
        if samples and [(r['prompt'], r['target']) for r in rows] != [
                (r['prompt'], r['target']) for r in samples['dense']]:
            raise ValueError('Prompt or target mismatch')
        samples[arm] = rows
        scores[arm] = np.array([r['scores']['acc,num'] for r in rows])
        runs[arm] = dict(correct=int(scores[arm].sum()), total=len(rows), accuracy=float(scores[arm].mean()),
                         invalid=sum(r['extracted']['numeric-extract'] == '[invalid]' for r in rows),
                         run_sha256=digest(path/'run.json'), raw_sha256=digest(path/'raw.json'),
                         seconds=run['seconds'], model=run['model'])
    delta = scores['staged']-scores['baseline']
    rng = np.random.default_rng(7)
    draws = np.array([delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(10000)])
    result = dict(arms=runs, quantization_recipe=quantization_recipe,
                  paired_staged_minus_baseline=float(delta.mean()),
                  calibration_samples=calibration_count, calibration_inputs_sha256=calibration_sha,
                  paired_bootstrap_ci95=np.quantile(draws, [.025, .975]).tolist(),
                  bootstrap_seed=7, bootstrap_draws=10000,
                  baseline_only_correct=int(((scores['baseline'] == 1) & (scores['staged'] == 0)).sum()),
                  staged_only_correct=int(((scores['baseline'] == 0) & (scores['staged'] == 1)).sum()),
                  exact_prompts_targets_and_input_ids=True,
                  prompt_ids_sha256=hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest())
    with gzip.open(args.root/'prompt-input-ids.json.gz', 'wt') as stream:
        json.dump(prompt_ids, stream)
    write_json(output, result)
    lines = ['# Matched GSM8K Platinum: Llama 3.2 1B Instruct', '',
             'All 1,209 test questions, eight-shot CoT, chat template, greedy generation, 256 new-token limit,',
             'FP16/eager, batch size 32, seed 7. Exact rendered prompts, targets and input IDs match.', '',
             '| Arm | Correct | Accuracy | Invalid answers |', '|---|---:|---:|---:|']
    for arm, r in runs.items():
        lines.append(f'| {arm} | {r["correct"]}/{r["total"]} | {r["accuracy"]:.4%} | {r["invalid"]} |')
    lines += ['', f'The W{quantization_recipe["bits"]}/group{quantization_recipe["group_size"]} control uses '
              'the staged-path GPTQ initializer with GSQ disabled.',
              f'The treatment adds staged Lion training and learned scales. Both use {calibration_count} calibration',
              'documents and are experimental; this is not the package-default GPTQ recipe or full paper reproduction.',
              '', '```json', json.dumps(result, indent=2), '```', '']
    (args.root/'comparison.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
