# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""CPU checks preventing incompatible full-row paired evaluations."""

import json

import pytest

from tests.models.w4a_quality_regression import paired_eval_batch_size


@pytest.fixture
def paired_paths(tmp_path):
    reference, checkpoint = tmp_path / "reference", tmp_path / "checkpoint"
    reference.mkdir()
    checkpoint.mkdir()
    for name in ("model.safetensors", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json"):
        (reference / name).write_text(name)
        (checkpoint / name).symlink_to(reference / name)
    result = tmp_path / "baseline.json"
    result.write_text(json.dumps({
        "tests": [{"name": "gsm8k_platinum_cot", "samples": [{}] * 1209}],
        "engine": {"seed": 42, "dtype": "bfloat16", "padding_side": "left",
                   "max_new_tokens": 256, "batch_size": 32},
        "model": {"path": str(reference)},
    }))
    return checkpoint, result


def test_inherits_frozen_batch_size(paired_paths):
    assert paired_eval_batch_size(*paired_paths, "gsm8k_platinum_cot") == 32
    assert paired_eval_batch_size(*paired_paths, "gsm8k_platinum_cot", 32) == 32


def test_rejects_batch_mismatch(paired_paths):
    with pytest.raises(ValueError, match="differs from frozen baseline"):
        paired_eval_batch_size(*paired_paths, "gsm8k_platinum_cot", 8)


@pytest.mark.parametrize("field,value", [
    ("seed", 1), ("dtype", "float16"), ("padding_side", "right"),
    ("max_new_tokens", 128), ("batch_size", 0), ("batch_size", True),
])
def test_rejects_engine_mismatch(paired_paths, field, value):
    checkpoint, result = paired_paths
    baseline = json.loads(result.read_text())
    baseline["engine"][field] = value
    result.write_text(json.dumps(baseline))
    with pytest.raises(ValueError):
        paired_eval_batch_size(checkpoint, result, "gsm8k_platinum_cot")


@pytest.mark.parametrize("change", ["partial", "task", "multiple"])
def test_rejects_wrong_coverage(paired_paths, change):
    checkpoint, result = paired_paths
    baseline = json.loads(result.read_text())
    if change == "partial":
        baseline["tests"][0]["samples"].pop()
    elif change == "task":
        baseline["tests"][0]["name"] = "arc_challenge"
    else:
        baseline["tests"].append(baseline["tests"][0])
    result.write_text(json.dumps(baseline))
    with pytest.raises(ValueError):
        paired_eval_batch_size(checkpoint, result, "gsm8k_platinum_cot")


@pytest.mark.parametrize("name", [
    "model.safetensors", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
])
def test_rejects_changed_artifacts(paired_paths, name):
    checkpoint, result = paired_paths
    (checkpoint / name).unlink()
    (checkpoint / name).write_text("different")
    with pytest.raises(ValueError):
        paired_eval_batch_size(checkpoint, result, "gsm8k_platinum_cot")


@pytest.fixture
def adapted_results(tmp_path):
    paths = []
    for name, backend in [("frozen", "gptq_triton"), ("native", "gptq_triton"),
                           ("float", "gptq_w4a_nvfp4")]:
        model = tmp_path / name
        model.mkdir()
        if name == "float":
            (model / "model.safetensors").symlink_to(tmp_path / "native" / "model.safetensors")
        else:
            (model / "model.safetensors").write_text(name)
        cfg = {"bits": 4, "pack_dtype": "int32"}
        if name == "float":
            cfg["activation"] = {"mode": "w4a_nvfp4", "version": 4}
        (model / "quantize_config.json").write_text(json.dumps(cfg))
        samples = [{"index": i, "prompt": f"prompt {i}", "target": str(i),
                    "scores": {"acc,num": int(i < 524)},
                    "extracted": {"numeric-extract": str(i) if i < 524 else "wrong"}}
                   for i in range(1209)]
        result = {"model": {"path": str(model)}, "versions": {},
                  "engine": {"backend": backend, "execution": {"quantized_backend": backend,
                             "effective_attn_implementation": "sdpa", "runtime_format": "gptq_v2"},
                             "seed": 42, "batch_size": 32, "dtype": "bfloat16",
                             "padding_side": "left", "max_new_tokens": 256},
                  "tests": [{"name": "gsm8k_platinum_cot", "metadata": {"num_fewshot": 8},
                             "metrics": {"acc,num": 524 / 1209}, "samples": samples}]}
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(result))
        paths.append(path)
    return paths


def _change_result(path, change):
    result = json.loads(path.read_text())
    change(result)
    path.write_text(json.dumps(result))


def test_adapted_acceptance_retains_same_weight_and_frozen_gates(adapted_results):
    from tests.models.w4a_quality_regression import compare, compare_adapted

    report = compare_adapted(*adapted_results)
    assert report['accepted'] and report['native_point_floor_met']
    assert report['frozen_native_correct'] == report['candidate_native_correct'] == 524
    assert report['same_weight_activation']['paired_95pct_ci_pp'] == [0., 0.]
    assert len(report['result_sha256']) == 3
    # The original comparison must still reject different weight files.
    with pytest.raises(ValueError, match='same packed-weight'):
        compare(adapted_results[0], adapted_results[2], 'gsm8k_platinum_cot', 2.)


def test_adapted_acceptance_rejects_degrading_both_lanes(adapted_results):
    from tests.models.w4a_quality_regression import compare_adapted

    def degrade(run):
        for row in run['tests'][0]['samples'][523:524]:
            row['scores']['acc,num'] = 0
        run['tests'][0]['metrics']['acc,num'] = 523 / 1209
    for path in adapted_results[1:]:
        _change_result(path, degrade)
    report = compare_adapted(*adapted_results)
    assert report['same_weight_activation']['verdict'] == 'within_budget'
    assert not report['native_point_floor_met'] and not report['accepted']
    assert report['frozen_reference_activation']['verdict'] == 'within_budget'


@pytest.mark.parametrize('change', ['duplicate', 'partial', 'aggregate', 'nonbinary', 'prompt',
                                   'seed', 'attention', 'versions', 'backend'])
def test_adapted_acceptance_rejects_incompatible_or_inconsistent_results(adapted_results, change):
    from tests.models.w4a_quality_regression import compare_adapted

    def mutate(run):
        test = run['tests'][0]
        if change == 'duplicate': test['samples'][-1]['index'] = 0
        elif change == 'partial': test['samples'].pop()
        elif change == 'aggregate': test['metrics']['acc,num'] = 1.
        elif change == 'nonbinary': test['samples'][0]['scores']['acc,num'] = float('nan')
        elif change == 'prompt': test['samples'][0]['prompt'] = 'different'
        elif change == 'seed': run['engine']['seed'] = 7
        elif change == 'attention': run['engine']['execution']['effective_attn_implementation'] = 'eager'
        elif change == 'versions': run['versions']['torch'] = 'different'
        elif change == 'backend': run['engine']['execution']['quantized_backend'] = 'gptq_triton'
    _change_result(adapted_results[2], mutate)
    with pytest.raises((ValueError, AssertionError)):
        compare_adapted(*adapted_results)


@pytest.fixture
def saved_acceptance_bundle(adapted_results, tmp_path):
    import hashlib
    from pathlib import Path

    frozen, native, quant = adapted_results
    model = Path(json.loads(frozen.read_text())['model']['path'])
    for name in ('chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json'):
        (model / name).write_text(name)
    required = [frozen] + [model / name for name in (
        'model.safetensors', 'quantize_config.json', 'chat_template.jinja',
        'tokenizer.json', 'tokenizer_config.json')]
    manifest = tmp_path / 'frozen_manifest.json'
    manifest.write_text(json.dumps({'task': 'gsm8k_platinum_cot', 'rows': 1209,
        'correct': 524, 'result': str(frozen), 'model': str(model),
        'sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in required}}))
    audit = tmp_path / 'consumer_audit.json'
    audit.write_text(json.dumps({'checkpoint': json.loads(quant.read_text())['model']['path'],
        'variant': 'w4a_nvfp4', 'activation_version': 4, 'activation_recipe': 'least_squares',
        'full_coverage': True, 'handoffs_checked': 381, 'independent_decode_checks': 720,
        'independent_gemm_checks': 336, 'counts': {'decoder_layer': 16, 'w4a_linear': 112},
        'selected_layers': [f'model.layers.{i}' for i in range(16)]}))
    return manifest, native, quant, audit


def test_saved_acceptance_validates_the_complete_bundle(saved_acceptance_bundle):
    from tests.models.w4a_saved_acceptance import validate_saved_nvfp4_acceptance

    assert validate_saved_nvfp4_acceptance(*saved_acceptance_bundle)['accepted']


def test_saved_acceptance_normalizes_legacy_recipe_names(saved_acceptance_bundle):
    from tests.models.w4a_saved_acceptance import validate_saved_nvfp4_acceptance

    audit = saved_acceptance_bundle[3]
    data = json.loads(audit.read_text())
    data['activation_recipe'] = 'lsq'
    audit.write_text(json.dumps(data))
    assert validate_saved_nvfp4_acceptance(*saved_acceptance_bundle)['accepted']


@pytest.mark.parametrize('field,value', [
    ('full_coverage', False), ('checkpoint', '/different/checkpoint'),
    ('independent_gemm_checks', 14), ('activation_version', 3),
    ('selected_layers', ['model.layers.15']), ('activation_recipe', 'four_six'),
])
def test_saved_acceptance_rejects_wrong_consumer_audit(saved_acceptance_bundle, field, value):
    from tests.models.w4a_saved_acceptance import validate_saved_nvfp4_acceptance

    audit = saved_acceptance_bundle[3]
    data = json.loads(audit.read_text())
    data[field] = value
    audit.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='audit'):
        validate_saved_nvfp4_acceptance(*saved_acceptance_bundle)


def test_saved_acceptance_rejects_changed_frozen_artifact(saved_acceptance_bundle):
    from pathlib import Path
    from tests.models.w4a_saved_acceptance import validate_saved_nvfp4_acceptance

    data = json.loads(saved_acceptance_bundle[0].read_text())
    (Path(data['model']) / 'tokenizer.json').write_text('changed tokenizer')
    with pytest.raises(ValueError, match='Frozen reference artifact changed'):
        validate_saved_nvfp4_acceptance(*saved_acceptance_bundle)


def test_saved_acceptance_rejects_failing_downstream_scores(saved_acceptance_bundle):
    from tests.models.w4a_saved_acceptance import validate_saved_nvfp4_acceptance

    def degrade(run):
        for row in run['tests'][0]['samples'][399:524]:
            row['scores']['acc,num'] = 0
        run['tests'][0]['metrics']['acc,num'] = 399 / 1209
    _change_result(saved_acceptance_bundle[2], degrade)
    with pytest.raises(AssertionError, match='Full GSM8K acceptance failed'):
        validate_saved_nvfp4_acceptance(*saved_acceptance_bundle)


def test_nvfp4_full_gsm8k_saved_acceptance():
    """Opt-in e2e evidence gate; missing artifacts are never reported as a pass."""
    import os
    from pathlib import Path
    from tests.models.w4a_saved_acceptance import validate_saved_nvfp4_acceptance
    from tests.models.w4a_dtype_audit import _verify_checkpoint_fingerprint

    names = ('GPTQMODEL_W4A_FROZEN_MANIFEST', 'GPTQMODEL_W4A16_FULL_RESULT',
             'GPTQMODEL_W4A4_FULL_RESULT', 'GPTQMODEL_W4A4_CONSUMER_AUDIT')
    values = [os.environ.get(name) for name in names]
    if not any(values):
        pytest.skip('Full NVFP4 accuracy evidence was not supplied; quality remains unverified')
    missing = [name for name, value in zip(names, values, strict=True) if not value]
    if missing:
        pytest.fail(f'Incomplete full NVFP4 accuracy evidence: {missing}')
    audit = json.loads(Path(values[3]).read_text())
    if not audit.get('checkpoint_sha256'):
        pytest.fail('Consumer audit has no checkpoint file hashes; rerun the saved-checkpoint audit')
    _verify_checkpoint_fingerprint(Path(audit['checkpoint']), audit['checkpoint_sha256'])
    validate_saved_nvfp4_acceptance(*(Path(value) for value in values))


@pytest.mark.parametrize('mutation', ['none', 'changed_weights', 'missing_hashes'])
def test_full_saved_entrypoint_requires_current_checkpoint_hashes(saved_acceptance_bundle, monkeypatch, mutation):
    from pathlib import Path
    from tests.models.w4a_dtype_audit import _checkpoint_fingerprint

    audit_path = saved_acceptance_bundle[3]
    audit = json.loads(audit_path.read_text())
    checkpoint = Path(audit['checkpoint'])
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        (checkpoint / name).write_text('{}')
    audit['checkpoint_sha256'] = _checkpoint_fingerprint(checkpoint)
    if mutation == 'missing_hashes':
        del audit['checkpoint_sha256']
    audit_path.write_text(json.dumps(audit))
    if mutation == 'changed_weights':
        (checkpoint / 'model.safetensors').write_bytes(b'changed')
    names = ('GPTQMODEL_W4A_FROZEN_MANIFEST', 'GPTQMODEL_W4A16_FULL_RESULT',
             'GPTQMODEL_W4A4_FULL_RESULT', 'GPTQMODEL_W4A4_CONSUMER_AUDIT')
    for name, value in zip(names, saved_acceptance_bundle, strict=True):
        monkeypatch.setenv(name, str(value))
    if mutation == 'changed_weights':
        with pytest.raises(ValueError, match='Checkpoint files changed'):
            test_nvfp4_full_gsm8k_saved_acceptance()
    elif mutation == 'missing_hashes':
        with pytest.raises(pytest.fail.Exception, match='no checkpoint file hashes'):
            test_nvfp4_full_gsm8k_saved_acceptance()
    else:
        test_nvfp4_full_gsm8k_saved_acceptance()
