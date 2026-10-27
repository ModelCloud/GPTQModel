# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU lifecycle regressions for collective failures and checkpoint ancestry."""

import copy
import json
import shutil
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gptqmodel.looper import gsq_training_model as staged
from gptqmodel.quantization import GSQTrainingConfig
from gptqmodel.quantization.gsq_distributed import GSQDistributed


def _model(layers=3):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=layers)
    config._attn_implementation = 'sdpa'
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(419)
        return LlamaForCausalLM(config).float().eval()


def _documents():
    return [{'input_ids': list(range(start, start + 4))} for start in range(1, 5)]


def _options():
    return {'bits': 3, 'group_size': 32,
            'gsq': GSQTrainingConfig(enabled=True, epochs=1, qk_steps=1, batch_size=2, microbatch_size=1),
            'initialization_documents': [{'input_ids': [64, 65, 66, 67]}],
            'validation_documents': [{'input_ids': [32, 33, 34, 35]}, {'input_ids': [36, 37, 38, 39]}]}


@pytest.fixture(scope='module', autouse=True)
def _cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.fixture(scope='module')
def checkpoints(tmp_path_factory):
    """Real packed checkpoints; failure tests change only their dependency chain."""
    directory = tmp_path_factory.mktemp('gsq-checkpoint-ancestry')
    result = staged.quantize_llama_gsq_model(_model(), _documents(), checkpoint_directory=directory, **_options())
    assert result['state'] == 'complete'
    assert sorted(path.name for path in directory.glob('*.pt')) == [f'layer-{index:02d}.pt' for index in range(3)]
    return directory


def _copy_checkpoints(source, destination, *, legacy=False):
    shutil.copytree(source, destination)
    if legacy:
        for path in destination.glob('*.pt'):
            saved = torch.load(path, map_location='cpu', weights_only=True)
            saved['signature'].pop('packed_prefix', None)
            torch.save(saved, path)
    return destination


def _replace_packed_predecessor(directory):
    path = directory / 'layer-00.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    signature = copy.deepcopy(saved['signature'])
    name = next(name for name in saved['state_dict'] if name.endswith('.qweight'))
    value = saved['state_dict'][name]
    original = value.clone()
    value.reshape(-1)[0].bitwise_xor_(1)
    assert not torch.equal(value, original)
    torch.save(saved, path)
    # This must defeat a check that looks only at dense model/corpus metadata.
    assert torch.load(path, map_location='cpu', weights_only=True)['signature'] == signature


def _resume_options(directory, destination=None):
    options = _options()
    options['checkpoint_directory'] = directory if destination is None else destination
    if destination is not None:
        options['resume_prefix_directory'] = directory
    return options


def _assert_rejected_before_training(options, *, context=None, match=r'(?i)(checkpoint|prefix|predecessor|gap)'):
    model = _model()
    if context is not None:
        options = dict(options, distributed=context)
    with patch.object(staged, 'capture_llama_gsq_inputs',
                      side_effect=AssertionError('checkpoint rejection must precede capture')) as capture, \
            patch.object(staged, 'quantize_llama_gsq_capture',
                         side_effect=AssertionError('checkpoint rejection must precede training')) as train:
        with pytest.raises((ValueError, RuntimeError), match=match) as caught:
            staged.quantize_llama_gsq_model(model, _documents(), **options)
        capture.assert_not_called()
        train.assert_not_called()
    return str(caught.value)


def _assert_resume_matches_source(source, options):
    model = _model()
    with patch.object(staged, 'capture_llama_gsq_inputs',
                      side_effect=AssertionError('completed prefix must not be captured')) as capture, \
            patch.object(staged, 'quantize_llama_gsq_capture',
                         side_effect=AssertionError('completed prefix must not be retrained')) as train:
        result = staged.quantize_llama_gsq_model(model, _documents(), **options)
        capture.assert_not_called()
        train.assert_not_called()
    assert result['state'] == 'complete' and len(result['blocks']) == 3
    for index, layer in enumerate(model.model.layers):
        expected = torch.load(source / f'layer-{index:02d}.pt', map_location='cpu', weights_only=True)['state_dict']
        assert layer.state_dict().keys() == expected.keys()
        for name, value in layer.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    return result


def _capture_failure(context, kind):
    """Inject one boundary failure while the other capture/replay paths succeed."""
    model = _model(layers=2)
    captures, cleaned, advanced, trained = {}, [], [], []
    original_capture = staged.capture_llama_gsq_inputs
    original_advance = staged.advance_llama_gsq_inputs
    marker = ('injected initialization advancement failure' if kind == 'advance'
              else 'injected validation capture failure')

    def capture(model, documents, **kwargs):
        first_token = documents[0]['input_ids'][0]
        label = 'initialization' if first_token >= 64 else 'validation' if first_token >= 32 else 'train'
        if kind == 'validation' and label == 'validation':
            raise RuntimeError(marker)
        result = original_capture(model, documents, **kwargs)
        captures[label] = result
        cleanup = result.cleanup

        def release():
            cleaned.append(label)
            cleanup()

        result.cleanup = release
        return result

    def quantize(layer, cache, **kwargs):
        # Quantization arithmetic has independent coverage. Keep real Llama
        # captures and forwards, and isolate the lifecycle after a fitted block.
        trained.append(layer)
        return copy.deepcopy(layer), {'stages': {}, 'initializer_metadata': {}}

    def advance(layer, cache, **kwargs):
        label = next(label for label, value in captures.items() if value is cache)
        if label == 'initialization':
            assert context is None or context.is_primary
            raise RuntimeError(marker)
        result = original_advance(layer, cache, **kwargs)
        advanced.append(label)
        return result

    options = dict(_options(), reuse_capture=True)
    if context is not None:
        options['distributed'] = context
    with patch.object(staged, 'capture_llama_gsq_inputs', side_effect=capture), \
            patch.object(staged, 'quantize_llama_gsq_capture', side_effect=quantize), \
            patch.object(staged, 'advance_llama_gsq_inputs', side_effect=advance), \
            pytest.raises(RuntimeError, match=marker) as caught:
        staged.quantize_llama_gsq_model(model, _documents(), **options)
    assert model.gsq_training_runs[-1]['state'] == 'failed'
    assert set(captures) <= set(cleaned), f'captures missing cleanup: {set(captures) - set(cleaned)}'
    for cache in captures.values():
        assert cache.layer_inputs == [] and cache.src_inputs == []
    primary = context is None or context.is_primary
    assert ('initialization' in captures) == primary
    if kind == 'advance':
        assert len(trained) == 1, 'a peer trained the next block after initialization replay failed'
        assert 'train' in advanced
        if not primary:
            assert 'validation' in advanced
    else:
        assert not trained
        assert 'validation' not in captures
    return str(caught.value)


def _failure_worker(rank, rendezvous, directory, case, source):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    try:
        context = GSQDistributed(device='cpu')
        if case in ('advance', 'validation'):
            message = _capture_failure(context, case)
        else:
            options = _resume_options(Path(source), Path(directory) / 'imported')
            message = _assert_rejected_before_training(options, context=context)
        # Record each result independently: another collective could itself hang
        # if the regression has already broken collective ordering.
        (Path(directory) / f'rank-{rank}.json').write_text(json.dumps({'error': message}), encoding='utf-8')
    finally:
        dist.destroy_process_group()


def _spawn_failure(tmp_path, case, source=None):
    mp.spawn(_failure_worker, args=((tmp_path / 'rendezvous').as_uri(), str(tmp_path), case,
                                    None if source is None else str(source)), nprocs=2, join=True)
    reports = [json.loads((tmp_path / f'rank-{rank}.json').read_text(encoding='utf-8')) for rank in range(2)]
    assert reports[0]['error'] == reports[1]['error']


@pytest.mark.timeout(120)
@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='requires CPU Gloo')
@pytest.mark.parametrize('case', ['advance', 'validation'])
def test_capture_failures_propagate_and_clean_up_on_every_rank(tmp_path, case):
    _spawn_failure(tmp_path, case)


def test_initialization_cache_is_cleaned_when_validation_capture_fails():
    _capture_failure(None, 'validation')


def test_new_checkpoints_record_packed_prefix_and_resume(checkpoints):
    signatures = [torch.load(checkpoints / f'layer-{index:02d}.pt', map_location='cpu', weights_only=True)['signature']
                  for index in range(3)]
    assert all('packed_prefix' in signature for signature in signatures)
    assert signatures[1]['packed_prefix'] != signatures[2]['packed_prefix']
    _assert_resume_matches_source(checkpoints, _resume_options(checkpoints))


@pytest.mark.parametrize('missing', [0, 1], ids=['missing-first', 'interior-gap'])
@pytest.mark.parametrize('migration', [False, True], ids=['resume', 'import'])
def test_checkpoint_gaps_are_rejected_before_training(tmp_path, checkpoints, missing, migration):
    source = _copy_checkpoints(checkpoints, tmp_path / 'source')
    (source / f'layer-{missing:02d}.pt').unlink()
    options = _resume_options(source, tmp_path / 'destination' if migration else None)
    _assert_rejected_before_training(options)
    assert not (source / f'layer-{missing:02d}.pt').exists()


@pytest.mark.parametrize('migration', [False, True], ids=['resume', 'import'])
def test_replaced_packed_predecessor_invalidates_later_checkpoint(tmp_path, checkpoints, migration):
    source = _copy_checkpoints(checkpoints, tmp_path / 'source')
    _replace_packed_predecessor(source)
    options = _resume_options(source, tmp_path / 'destination' if migration else None)
    _assert_rejected_before_training(options)


@pytest.mark.timeout(120)
@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='requires CPU Gloo')
def test_checkpoint_predecessor_rejection_reaches_every_rank(tmp_path, checkpoints):
    source = _copy_checkpoints(checkpoints, tmp_path / 'source')
    _replace_packed_predecessor(source)
    _spawn_failure(tmp_path, 'checkpoint', source)


@pytest.mark.parametrize('migration', [False, True], ids=['resume', 'import'])
def test_contiguous_legacy_checkpoints_remain_usable(tmp_path, checkpoints, migration):
    source = _copy_checkpoints(checkpoints, tmp_path / 'legacy', legacy=True)
    options = _resume_options(source, tmp_path / 'destination' if migration else None)
    _assert_resume_matches_source(source, options)
    for path in source.glob('*.pt'):
        assert 'packed_prefix' not in torch.load(path, map_location='cpu', weights_only=True)['signature']


def test_legacy_import_resumes_after_partial_migration(tmp_path, checkpoints):
    source = _copy_checkpoints(checkpoints, tmp_path / 'legacy', legacy=True)
    destination = tmp_path / 'destination'
    options = _resume_options(source, destination)
    _assert_resume_matches_source(source, options)
    # Retain the real migration provenance and only the completed first block.
    (destination / 'layer-01.pt').unlink()
    (destination / 'layer-02.pt').unlink()
    _assert_resume_matches_source(source, options)


def test_unverified_legacy_stitching_rejects_different_packed_predecessor(tmp_path, checkpoints):
    source = _copy_checkpoints(checkpoints, tmp_path / 'legacy', legacy=True)
    destination = tmp_path / 'destination'
    destination.mkdir()
    shutil.copy2(source / 'layer-00.pt', destination / 'layer-00.pt')
    _replace_packed_predecessor(destination)
    _assert_rejected_before_training(_resume_options(source, destination))
