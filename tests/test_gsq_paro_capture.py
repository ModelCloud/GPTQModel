import pytest
import torch

from gptqmodel.quantization.paroquant.gsq_capture import capture_module_inputs


def test_capture_preserves_batch_invocation_and_copies_inputs():
    module = torch.nn.Identity()
    batch = [3]
    value = torch.ones(2, 4)
    calls = []
    existing = module.register_forward_pre_hook(lambda *_: calls.append(True))
    with capture_module_inputs({'q': module}, lambda: batch[0]) as captured:
        module(value)
        value.add_(1)
        module(value)
        batch[0] = 4
        module(value)
    assert [(b, i) for b, i, _ in captured['q']] == [(3, 0), (3, 1), (4, 0)]
    assert torch.equal(captured['q'][0][2], torch.ones(2, 4))
    assert len(module._forward_pre_hooks) == 1
    module(value)
    assert len(calls) == 4
    existing.remove()


@pytest.mark.parametrize('failure', ['replay', 'missing_batch'])
def test_capture_failure_discards_partial_data_and_removes_hooks(failure):
    module = torch.nn.Identity()
    batch = [0]
    with pytest.raises((RuntimeError, ValueError)):
        with capture_module_inputs({'q': module}, lambda: batch[0]) as captured:
            module(torch.ones(2, 4))
            if failure == 'replay':
                raise RuntimeError('failed replay')
            batch[0] = None
            module(torch.ones(2, 4))
    assert captured == {'q': []}
    assert not module._forward_pre_hooks


def test_pair_alignment_requires_exact_invocations_and_shapes():
    from gptqmodel.quantization.paroquant.gsq_capture import align_module_inputs

    x = torch.ones(2, 4)
    clean = [(1, 0, x), (0, 0, x)]
    paired = align_module_inputs(clean, list(reversed(clean)))
    assert [key for key, _, _ in paired] == [(0, 0), (1, 0)]
    for bad, match in ((clean[:1], 'IDs do not match'),
                       (clean + clean[:1], 'duplicate'),
                       ([(1, 0, x[:1]), (0, 0, x)], 'shapes')):
        with pytest.raises(ValueError, match=match):
            align_module_inputs(clean, bad)


def test_processor_pristine_context_commits_only_successful_capture():
    import threading
    from types import SimpleNamespace
    from gptqmodel.looper.paroquant_processor import ParoQuantProcessor

    processor = object.__new__(ParoQuantProcessor)
    # Exercise collector infrastructure; public noisy-input GSQ stays guarded.
    processor.qcfg = SimpleNamespace(gsq=SimpleNamespace(enabled=True), opt_scope='layer',
                                     opt_train_on_noisy_inputs=True)
    processor._layer_states = {}
    processor._layer_states_lock = threading.Lock()
    processor.current_batch_index = lambda: 0
    layer = torch.nn.Sequential(torch.nn.Linear(4, 4))
    processor.receive_pristine_layer_module(layer_index=0, layer_module=layer)
    with processor.pristine_quant_input_capture(layer_index=0):
        layer(torch.ones(2, 4))
    saved = processor._get_layer_state(0).gsq_clean_inputs
    assert saved['0'][0][:2] == (0, 0)
    processor.receive_pristine_layer_module(layer_index=0, layer_module=layer)
    with pytest.raises(RuntimeError, match='replay failed'):
        with processor.pristine_quant_input_capture(layer_index=0):
            layer(torch.ones(2, 4))
            raise RuntimeError('replay failed')
    assert processor._get_layer_state(0).gsq_clean_inputs is saved
    assert not layer[0]._forward_pre_hooks


def test_noisy_hook_and_clean_capture_split_preserve_pair_alignment():
    import threading
    from types import SimpleNamespace
    from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
    from gptqmodel.quantization.config import GSQConfig

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(gsq=GSQConfig(enabled=True), opt_scope='layer',
                                     opt_train_on_noisy_inputs=True, opt_train_samples=8, opt_validation_samples=8)
    processor.lock = threading.Lock()
    processor.tasks = {}
    processor._has_explicit_validation_calibration = True
    processor._train_calibration_batch_count = 1
    processor._validation_calibration_batch_count = 1
    batch = [1]
    processor.current_batch_index = lambda: batch[0]
    processor._record_input_feature('q', torch.full((3, 4), 20.))
    batch[0] = 0
    processor._record_input_feature('q', torch.full((2, 4), 10.))
    state = SimpleNamespace(modules={'q': SimpleNamespace(full_name='q_proj')}, gsq_clean_inputs={
        'q': [(0, 0, torch.ones(2, 4)), (1, 0, torch.full((3, 4), 2.))]})
    processor._layer_input_features(state)
    entry = processor.tasks['q']
    assert torch.equal(entry['train_inputs'], entry['gsq_teacher_train_inputs'] * 10)
    assert torch.equal(entry['validation_inputs'], entry['gsq_teacher_validation_inputs'] * 10)
    assert entry['train_inputs'].shape == (1, 2, 4)
    assert entry['validation_inputs'].shape == (1, 3, 4)


def test_paired_fitter_samples_the_same_clean_and_noisy_rows(monkeypatch):
    from types import SimpleNamespace
    from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
    from gptqmodel.quantization.config import ParoConfig
    import gptqmodel.quantization.gsq_paro as adapter

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = ParoConfig(gsq={'enabled': True}, group_size=16, opt_train_samples=4, opt_validation_samples=2)
    noisy = torch.arange(96).float().reshape(6, 16)
    clean = noisy + 100
    validation = torch.ones(3, 16)
    seen = []

    def fit(result, **kwargs):
        assert torch.equal(kwargs['teacher_inputs'], kwargs['inputs'] + 100)
        assert len(kwargs['inputs']) == 4
        seen.append(True)
        return {'before': 1., 'after': 1., 'history': [1.]}

    monkeypatch.setattr(adapter, 'refine_paro_export', fit)
    module = SimpleNamespace(full_name='q_proj', state={})
    result = SimpleNamespace(train_loss=2., val_loss=3.)
    processor._refine_gsq_export(module, result, torch.ones(8, 16), noisy, validation,
                                 teacher_inputs=clean, teacher_validation_inputs=validation + 100)
    processor._refine_gsq_export(module, result, torch.ones(8, 16), noisy, None, teacher_inputs=clean)
    assert seen == [True, True]
    assert module.state['gsq_diagnostics']['objective'] == 'asymmetric_transformed_quadratic_without_constant'


def test_capture_uses_thread_local_batch_ids_under_parallel_replay():
    from concurrent.futures import ThreadPoolExecutor
    from threading import local

    module = torch.nn.Identity()
    context = local()

    def replay(batch):
        context.batch = batch
        for _ in range(3):
            module(torch.full((2, 4), float(batch)))

    with capture_module_inputs({'q': module}, lambda: context.batch) as captures:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(replay, range(12)))
    assert len(captures['q']) == 36
    for batch in range(12):
        records = [(index, value) for key, index, value in captures['q'] if key == batch]
        assert sorted(index for index, _ in records) == [0, 1, 2]
        assert all(torch.equal(value, torch.full((2, 4), float(batch))) for _, value in records)
    assert not module._forward_pre_hooks


@pytest.mark.parametrize('scope', ['layer', 'compute_block'])
def test_paired_capture_to_group_export_lifecycle(scope, monkeypatch):
    import threading
    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
    from gptqmodel.quantization.config import ParoConfig
    from gptqmodel.quantization.paroquant.optimization import ParoQuantOptimizationResult

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = ParoConfig(group_size=16, opt_scope=scope, offload_to_disk=False,
                                opt_train_samples=64, opt_validation_samples=16,
                                gsq={'enabled': True, 'steps': 80, 'learning_rate': .2})
    # Internal integration probe while the public configuration remains guarded.
    processor.qcfg.opt_train_on_noisy_inputs = True
    processor.lock = threading.Lock()
    processor._layer_states_lock = threading.Lock()
    processor._layer_states = {}
    processor.tasks = {}
    processor.fallback = False
    processor.calculate_w_wq_diff = False
    processor._has_explicit_validation_calibration = True
    processor._train_calibration_batch_count = 1
    processor._validation_calibration_batch_count = 1
    batch = [0]
    processor.current_batch_index = lambda: batch[0]
    layer = torch.nn.Sequential(torch.nn.Linear(32, 8, bias=False, dtype=torch.float16))
    layer[0].weight.data.fill_(.125)
    module = NamedModule(layer[0], '0', 'model.layers.0.q_proj', 0)
    module.state['module_tree_flags'] = frozenset({'q'})
    rng = torch.Generator().manual_seed(7)
    noisy = [torch.randn(1, rows, 32, generator=rng).half() for rows in (64, 16)]
    clean = [x * 1.2 for x in noisy]
    processor.receive_pristine_layer_module(layer_index=0, layer_module=layer)
    with processor.pristine_quant_input_capture(layer_index=0):
        for index, x in enumerate(clean):
            batch[0] = index
            layer(x)
    state = processor._get_layer_state(0)
    state.modules = {'0': module}
    state.layer_inputs = [[x] for x in noisy]
    state.layer_outputs = [[layer(x)] for x in clean]
    processor.tasks['0'] = {'inputs': [], 'batch_indices': [], 'layer_index': 0}
    hook = layer[0].register_forward_hook(processor.pre_process_fwd_hook('0'))
    try:
        for index, x in enumerate(noisy):
            batch[0] = index
            layer(x)
    finally:
        hook.remove()
    baseline = ParoQuantOptimizationResult(
        pseudo_weight=torch.zeros(8, 32), pack_weight=torch.zeros(8, 32).half(),
        q_scales=torch.full((8, 2), .125).half(), q_zeros=torch.full((8, 2), 8.),
        pairs=torch.empty(0, 32, dtype=torch.int16), theta=torch.empty(0, 16),
        channel_scales=torch.ones(32), train_loss=3., val_loss=4., used_identity=True)
    monkeypatch.setattr(processor, '_optimize_group', lambda state, modules: ({'0': baseline}, 4.))
    monkeypatch.setattr(processor, '_log_quant_result', lambda *args: None)
    original = module.weight.detach().float().clone()
    processor._quantize_layer(0, state)
    stats = module.state['gsq_diagnostics']
    assert stats['after'] < stats['before']
    assert stats['objective'] == 'asymmetric_transformed_quadratic_without_constant'
    actual = noisy[0].float() @ module.weight.float().T
    target = clean[0].float() @ original.T
    assert (actual-target).square().sum() < target.square().sum()
    assert state.quantized and state.gsq_clean_inputs is None
    assert not any(key.startswith('gsq_') for key in processor.tasks['0'])
    assert not layer[0]._forward_pre_hooks and not layer[0]._forward_hooks
