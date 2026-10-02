# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Validate one saved full-model NVFP4 accuracy evidence bundle."""

import hashlib
import json
from pathlib import Path

from .w4a_quality_regression import compare_adapted


def validate_saved_nvfp4_acceptance(frozen_manifest: Path, native_result: Path,
                                   activation_result: Path, consumer_audit: Path) -> dict:
    frozen = json.loads(frozen_manifest.read_text())
    if (frozen.get('task') != 'gsm8k_platinum_cot' or frozen.get('rows') != 1209
            or frozen.get('correct') != 524):
        raise ValueError('Expected the frozen original 524/1209 native reference')
    reference = Path(frozen['result'])
    required = {str(reference)} | {str(Path(frozen['model']) / name) for name in (
        'model.safetensors', 'quantize_config.json', 'chat_template.jinja',
        'tokenizer.json', 'tokenizer_config.json')}
    if not required <= frozen.get('sha256', {}).keys():
        raise ValueError('Frozen reference hashes are incomplete')
    for name in sorted(required):
        with Path(name).open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        if actual != frozen['sha256'][name]:
            raise ValueError(f'Frozen reference artifact changed: {name}')

    quant = json.loads(activation_result.read_text())
    checkpoint = Path(quant['model']['path']).resolve()
    audit = json.loads(consumer_audit.read_text())
    if Path(audit['checkpoint']).resolve() != checkpoint:
        raise ValueError('Consumer audit belongs to a different activation checkpoint')
    if (audit.get('variant') != 'w4a_nvfp4' or audit.get('activation_version') != 4
            or audit.get('full_coverage') is not True
            or audit.get('handoffs_checked') != 381
            or audit.get('independent_decode_checks') != 720
            or audit.get('independent_gemm_checks') != 336
            or audit.get('counts', {}).get('decoder_layer') != 16
            or audit.get('counts', {}).get('w4a_linear') != 112
            or audit.get('selected_layers') != [f'model.layers.{i}' for i in range(16)]):
        raise ValueError('Expected the complete version-4 NVFP4 consumer audit')
    config = json.loads((checkpoint / 'quantize_config.json').read_text())
    if config.get('activation', {}).get('mode') != 'w4a_nvfp4' or config['activation'].get('version') != 4:
        raise ValueError('Saved accuracy acceptance requires a version-4 NVFP4 checkpoint')
    from gptqmodel.quantization.activation_floatx import normalize_nvfp4_recipe

    if normalize_nvfp4_recipe(audit.get('activation_recipe')) != normalize_nvfp4_recipe(
            config['activation'].get('recipe', 'least_squares')):
        raise ValueError('Consumer audit recipe differs from the checkpoint policy')
    report = compare_adapted(reference, native_result, activation_result, allowed_drop_pp=2.)
    if report['frozen_native_correct'] != 524:
        raise ValueError('Frozen reference result disagrees with its declared point score')
    if not report['accepted']:
        raise AssertionError(
            f'Full GSM8K acceptance failed: native={report["candidate_native_correct"]}/1209, '
            f'A4={report["candidate_float_correct"]}/1209; '
            f'same-weight={report["same_weight_activation"]["verdict"]}, '
            f'frozen-reference={report["frozen_reference_activation"]["verdict"]}')
    return report
