import pytest

from scripts.analyze_gsq_gsm8k import validate_evaluation_settings


def test_dense_reuse_allows_only_recorded_qvq_revision_difference():
    dense = dict(qvq_commit='old', source_sha256='same-script', batch_size=32,
                 model_args={'dtype': 'float16'}, dataset_hashes={'test': 'locked'})
    current = dict(dense, qvq_commit='new')
    with pytest.raises(ValueError):
        validate_evaluation_settings(current, dense)
    validate_evaluation_settings(current, dense, reuse_dense=True)
    assert dense['qvq_commit'] == 'old' and current['qvq_commit'] == 'new'
    for key, value in [('source_sha256', 'different'), ('batch_size', 16),
                       ('model_args', {'dtype': 'bfloat16'}), ('dataset_hashes', {'test': 'changed'})]:
        with pytest.raises(ValueError):
            validate_evaluation_settings(dict(current, **{key: value}), dense, reuse_dense=True)


def test_disabled_baseline_optimizer_is_unused_but_initializer_must_match():
    from scripts.analyze_gsq_full_llama import validate_matched_recipe

    baseline = dict(source_model='same', bits=4, group_size=128, train_precision='float32',
                    export_precision='float16', calibration_samples=128, calibration_tokens=1000,
                    weighting='unweighted', gsq_training=dict(enabled=False, initializer='gptq_signed', epochs=5))
    staged = dict(baseline, gsq_training=dict(enabled=True, initializer='gptq_signed', epochs=10, optimizer='adamw'))
    validate_matched_recipe(baseline, staged)
    for change in ({'initializer': 'gptq'}, {'optimizer': 'unknown'}, {'batch_size': 64}):
        with pytest.raises(ValueError):
            validate_matched_recipe(baseline, dict(staged, gsq_training=dict(staged['gsq_training'], **change)))
    with pytest.raises(ValueError):
        validate_matched_recipe(baseline, dict(staged, bits=2))
