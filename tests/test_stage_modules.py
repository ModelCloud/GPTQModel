import sys
import threading
import types
from typing import Dict

import torch

import gptqmodel.looper.stage_subset as stage_subset_module
from gptqmodel.looper.forward_executor import ForwardExecutor
from gptqmodel.looper.loop_processor import ExecutionConfig
from gptqmodel.looper.module_looper import FinalizeProgressInfo, ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
from gptqmodel.looper.stage_layer import (
    _capture_pristine_group_context,
    _processor_needs_pristine_group_clone,
    _replay_layer_outputs,
    _should_drain_finalize_futures_synchronously,
    _should_empty_cache_after_sync_finalize,
    run_layer_stage,
)
from gptqmodel.looper.stage_subset import CalibrationCoveragePolicy, SubsetPlan, SubsetStageResult
from gptqmodel.models.base import BaseQModel
from gptqmodel.quantization.config import QuantizeConfig


class _DummyQModel:
    def __init__(self):
        self.support_batch_quantize = False
        self.quantize_config = types.SimpleNamespace(
            device=None,
            dense_vram_strategy="exclusive",
            dense_vram_strategy_devices=None,
            moe_vram_strategy="exclusive",
            moe_vram_strategy_devices=None,
            moe_routing_bypass=lambda: False,
        )
        self.layer_callback = None


def _make_looper():
    processors = [types.SimpleNamespace(layer_count=0, pb=None)]
    return ModuleLooper(model=_DummyQModel(), processors=processors)


def test_cache_inputs_delegates_to_stage_capture(monkeypatch):
    looper = _make_looper()
    sentinel = object()
    captured = {}

    class FakeStage:
        def __init__(self, looper_arg, logger):
            captured["looper"] = looper_arg
            captured["logger"] = logger

        def cache_inputs(self, **kwargs):
            captured["kwargs"] = kwargs
            return sentinel

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.StageInputsCapture",
        FakeStage,
    )

    layers = [object()]
    data = [{"hidden_states": torch.zeros(1, 2, 2)}]
    result = looper.cache_inputs(layers=layers, calibration_data=data, use_cache=False)

    assert result is sentinel
    assert captured["looper"] is looper
    assert captured["kwargs"]["layers"] == layers
    assert captured["kwargs"]["calibration_data"] is data


def test_assign_quant_device_prefers_balanced_hint():
    looper = _make_looper()
    looper._quant_devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    looper._module_device_map = {}
    looper._quant_device_rr = 0

    named = NamedModule(
        torch.nn.Linear(4, 4, bias=False),
        name="mlp.experts.1.gate_proj",
        full_name="model.layers.0.mlp.experts.1.gate_proj",
        layer_index=0,
    )
    named.state["preferred_quant_device"] = torch.device("cuda:1")

    target = looper._assign_quant_device_for_module(
        named,
        fallback_device=torch.device("cuda:0"),
    )

    assert target == torch.device("cuda:1")
    assert looper._module_device_map[named.full_name] == torch.device("cuda:1")
    assert looper._quant_device_rr == 0


def test_module_looper_runtime_telemetry_reports_gil_and_split_pools(monkeypatch):
    emitted = []
    info_logs = []
    warn_logs = []
    module_looper_module = sys.modules[ModuleLooper.__module__]

    monkeypatch.setattr(
        module_looper_module,
        "emit_device_telemetry",
        lambda event, **fields: emitted.append((event, fields)),
    )
    monkeypatch.setattr(module_looper_module, "has_gil_control", lambda: True)
    monkeypatch.setattr(module_looper_module, "has_gil_disabled", lambda: True)
    monkeypatch.setattr(module_looper_module.os, "environ", {"PYTHON_GIL": "0"})
    monkeypatch.setattr(module_looper_module.log, "info", lambda *args, **kwargs: info_logs.append(args))
    monkeypatch.setattr(module_looper_module.log, "warn", lambda *args, **kwargs: warn_logs.append(args))

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = types.SimpleNamespace(dynamic_expert_index=object())
    looper._dense_quant_devices = [torch.device("cuda:0")]
    looper._moe_quant_devices = [torch.device("cuda:1"), torch.device("cuda:2")]
    looper._dense_vram_strategy = "exclusive"
    looper._moe_vram_strategy = "balanced"
    looper.moe_routing_override = 256
    looper.moe_routing_bypass = False

    looper._emit_moe_parallel_quant_runtime()

    assert info_logs
    assert not warn_logs
    assert len(emitted) == 1
    event, fields = emitted[0]
    assert event == "moe_parallel_quant_runtime"
    assert fields["dense_devices"] == ["cuda:0"]
    assert fields["moe_devices"] == ["cuda:1", "cuda:2"]
    assert fields["routing_override"] == 256
    assert fields["python_gil_env"] == "0"
    assert fields["python_gil_disabled"] is True
    assert fields["free_threaded_parallel_quant_eligible"] is True


class _TinyLayer(torch.nn.Module):
    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        return hidden_states


class _TinyModel(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.config = types.SimpleNamespace(model_type="llama")
        self.visual_tokenizer = types.SimpleNamespace(dtype=torch.float32)

    def forward(self, *, hidden_states, attention_mask=None, position_ids=None, use_cache=False, **kwargs):
        return self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )


class _TinyGptqModel:
    ATTENTION_MASKS_REQUIRED_FOR_INPUT = False
    ATTENTION_MASKS_DTYPE = torch.long
    INPUT_EMBEDDING_EXTRA_ARGS = {}
    finalize_input_capture_example = BaseQModel.finalize_input_capture_example
    capture_first_layer_positional_inputs = BaseQModel.capture_first_layer_positional_inputs
    capture_first_layer_input_kwargs = BaseQModel.capture_first_layer_input_kwargs
    move_input_capture_example = BaseQModel.move_input_capture_example
    prepare_layer_replay_kwargs = BaseQModel.prepare_layer_replay_kwargs
    run_input_capture = BaseQModel.run_input_capture

    def __init__(self):
        self.layer = _TinyLayer()
        self.model = _TinyModel(self.layer)
        self.quantize_config = types.SimpleNamespace(
            device=torch.device("cpu"),
            calibration_data_device=None,
        )
        self._hook_started = False
        self._hook_finished = False

    def shell_module_materialize(self, target_submodule, device, **kwargs):
        del kwargs
        target_submodule.to(device)
        return target_submodule

    def get_base_modules(self, model):
        return []

    def pre_quantize_generate_hook_start(self):
        self._hook_started = True

    def pre_quantize_generate_hook_end(self):
        self._hook_finished = True


class _TinyLooper:
    def __init__(self, gptq_model):
        self.gptq_model = gptq_model

    def _batch_row_count(self, batch_inputs):
        if not batch_inputs:
            return 0
        tensor = batch_inputs[0]
        return int(tensor.shape[0]) if tensor.ndim > 0 else int(tensor.numel())


class _TinyExecutorLayer(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states


class _RecordingCtx:
    def __init__(self, sink):
        self.sink = sink

    def __enter__(self):
        self.sink.append("enter")
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyForwardProcessor:
    num_batches = None

    def _set_current_batch_index(self, _idx):
        return None


class _ImmediateFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _ImmediateThreadPool:
    def submit(self, _device, fn, *args, **kwargs):
        return _ImmediateFuture(fn(*args, **kwargs))

    def submit_serial(self, _device, fn, *args, **kwargs):
        return _ImmediateFuture(fn(*args, **kwargs))


def _make_forward_executor_looper(
    *,
    override_entries=None,
    lifecycle_entries=None,
    moe_routing_override=None,
    moe_routing_bypass=False,
    should_use_moe_lifecycle=False,
):
    def _override_context(*_args, **_kwargs):
        if override_entries is None:
            raise AssertionError("override should stay disabled")
        return _RecordingCtx(override_entries)

    def _lifecycle_context(*_args, **_kwargs):
        if lifecycle_entries is None:
            raise AssertionError("lifecycle should stay disabled")
        return _RecordingCtx(lifecycle_entries)

    return types.SimpleNamespace(
        _resolve_batch_total=lambda _num_batches, layer_inputs: len(layer_inputs),
        _collect_row_counts=lambda layer_inputs: [int(batch[0].shape[0]) for batch in layer_inputs],
        _set_processor_mask=lambda _processor, _mask: None,
        _batch_row_count=lambda batch_inputs: int(batch_inputs[0].shape[0]),
        support_batch_quantize=False,
        gptq_model=types.SimpleNamespace(
            quantize_config=types.SimpleNamespace(
                calibration_data_device=None,
                compute_device_filter=None,
            ),
            prepare_layer_replay_kwargs=lambda layer, layer_input, additional_inputs, target_device: additional_inputs,
        ),
        moe_routing_override=moe_routing_override,
        moe_routing_bypass=moe_routing_bypass,
        MoERoutingOverrideContext=_override_context,
        MoELifecycleContext=_lifecycle_context,
        _should_use_moe_lifecycle=lambda *_args, **_kwargs: should_use_moe_lifecycle,
        _current_subset=None,
    )


def _run_executor_single(executor, processor, *, apply_moe_config):
    return executor.run_single(
        module=_TinyExecutorLayer(),
        processor=processor,
        layer_inputs=[[torch.zeros(1, 1, 1)]],
        layer_input_kwargs=[{}],
        position_ids=[],
        attention_masks=[None],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        need_outputs=True,
        reuse_kv=False,
        apply_moe_config=apply_moe_config,
    )


def _run_executor_parallel(executor, processor, *, apply_moe_config):
    def clone_module_for_devices_fn(module, devices, progress_callback=None):
        del progress_callback
        return dict.fromkeys(devices, module)

    def forward_batch_worker_fn(
        _replica,
        _processor,
        batch_idx,
        _batch_inputs,
        _batch_kwargs,
        _attention_mask,
        _position_ids,
        **_kwargs,
    ):
        return batch_idx, torch.zeros(1, 1, 1), None

    return executor.run_parallel(
        module=_TinyExecutorLayer(),
        processor=processor,
        layer_inputs=[[torch.zeros(1, 1, 1)], [torch.zeros(1, 1, 1)]],
        layer_input_kwargs=[{}, {}],
        position_ids=[None, None],
        attention_masks=[None, None],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        need_outputs=True,
        reuse_kv=False,
        devices=[torch.device("cuda:0"), torch.device("cuda:1")],
        apply_moe_config=apply_moe_config,
        clone_module_for_devices_fn=clone_module_for_devices_fn,
        forward_batch_worker_fn=forward_batch_worker_fn,
        device_thread_pool=_ImmediateThreadPool(),
    )


def test_stage_layer_forces_sync_finalizers_for_paroquant():
    looper = types.SimpleNamespace(
        gptq_model=types.SimpleNamespace(
            quantize_config=QuantizeConfig(
                bits=4,
                group_size=128,
                wait_for_submodule_finalizers=False,
            )
        )
    )
    paro_processor = object.__new__(ParoQuantProcessor)

    assert _should_drain_finalize_futures_synchronously(
        looper,
        finalize_tasks=[(paro_processor, None, None, None, None)],
    ) is True


def test_stage_layer_keeps_async_finalizers_for_non_paroquant_when_unset():
    looper = types.SimpleNamespace(
        gptq_model=types.SimpleNamespace(
            quantize_config=QuantizeConfig(
                bits=4,
                group_size=128,
                wait_for_submodule_finalizers=False,
            )
        )
    )

    assert _should_drain_finalize_futures_synchronously(
        looper,
        finalize_tasks=[(types.SimpleNamespace(), None, None, None, None)],
    ) is False


def test_stage_layer_empties_cache_after_sync_paroquant_finalize_only_with_offload():
    looper = types.SimpleNamespace(
        gptq_model=types.SimpleNamespace(
            quantize_config=QuantizeConfig(
                bits=4,
                group_size=128,
                offload_to_disk=True,
            )
        )
    )
    paro_processor = object.__new__(ParoQuantProcessor)

    assert _should_empty_cache_after_sync_finalize(
        looper,
        finalize_tasks=[(paro_processor, None, None, None, None)],
    ) is True

    looper.gptq_model.quantize_config.offload_to_disk = False
    assert _should_empty_cache_after_sync_finalize(
        looper,
        finalize_tasks=[(paro_processor, None, None, None, None)],
    ) is False

def test_stage_layer_paroquant_layer_scope_skips_pristine_group_clone():
    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = types.SimpleNamespace(opt_scope="layer")

    assert _processor_needs_pristine_group_clone(processor) is False


def test_stage_layer_paroquant_compute_block_scope_keeps_pristine_group_clone():
    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = types.SimpleNamespace(opt_scope="compute_block")

    assert _processor_needs_pristine_group_clone(processor) is True


def test_stage_subset_flush_stays_local_when_work_stays_on_cur_layer_device():
    cur_layer_device = torch.device("cuda:0")

    assert (
        stage_subset_module._resolve_cache_flush_device(cur_layer_device, [torch.device("cuda:0")])
        == cur_layer_device
    )


def test_stage_subset_flush_goes_global_when_work_fans_out_across_devices():
    cur_layer_device = torch.device("cuda:0")

    assert stage_subset_module._resolve_cache_flush_device(
        cur_layer_device,
        [torch.device("cuda:0"), torch.device("cuda:1")],
    ) is None


def test_stage_inputs_capture_collects_real_inputs():
    gptq_model = _TinyGptqModel()
    looper = _TinyLooper(gptq_model)
    stage = StageInputsCapture(looper, logger=None)

    hidden = torch.ones(1, 2, 3)
    attention = torch.ones(1, 2)
    position_ids = torch.arange(2).unsqueeze(0)
    extra = torch.tensor([5.0])

    dataset = [
        {
            "hidden_states": hidden.clone(),
            "attention_mask": attention.clone(),
            "position_ids": position_ids.clone(),
            "extra": extra.clone(),
        }
    ]

    cache = stage.cache_inputs(layers=[gptq_model.layer], calibration_data=dataset, use_cache=False)

    assert len(cache.layer_inputs) == 1
    assert torch.equal(cache.layer_inputs[0][0], hidden)
    assert torch.equal(cache.attention_masks[0], attention.long())
    assert torch.equal(cache.position_ids[0], position_ids)
    assert torch.equal(cache.layer_input_kwargs[0]["extra"], extra.unsqueeze(0))
    assert gptq_model._hook_started is True
    assert gptq_model._hook_finished is True


def test_forward_executor_run_single_can_skip_moe_routing_override_for_replay():
    """Replay must skip top-k override, while quant-time forward still enables it."""

    override_entries = []
    looper = _make_forward_executor_looper(
        override_entries=override_entries,
        lifecycle_entries=[],
        moe_routing_override=256,
    )
    executor = ForwardExecutor(looper)
    processor = _DummyForwardProcessor()

    # Replay path: do not install any MoE routing override context.
    outputs = _run_executor_single(executor, processor, apply_moe_config=False)

    assert len(outputs) == 1
    assert override_entries == []

    override_entries.clear()
    outputs = _run_executor_single(executor, processor, apply_moe_config=True)

    assert len(outputs) == 1
    assert override_entries == ["enter"]


def test_forward_executor_run_single_can_skip_moe_lifecycle_for_replay():
    """Replay must also skip bypass/lifecycle hooks, not just routing override."""

    lifecycle_entries = []
    looper = _make_forward_executor_looper(
        lifecycle_entries=lifecycle_entries,
        moe_routing_bypass=True,
        should_use_moe_lifecycle=True,
    )
    executor = ForwardExecutor(looper)
    processor = _DummyForwardProcessor()

    # Replay path: bypass routing stays off, so lifecycle hooks must not run.
    outputs = _run_executor_single(executor, processor, apply_moe_config=False)

    assert len(outputs) == 1
    assert lifecycle_entries == []

    outputs = _run_executor_single(executor, processor, apply_moe_config=True)

    assert len(outputs) == 1
    assert lifecycle_entries == ["enter"]


def test_forward_executor_run_parallel_can_skip_moe_config_for_replay():
    """Parallel replay must skip the same MoE config that serial replay skips."""

    override_entries = []
    looper = _make_forward_executor_looper(
        override_entries=override_entries,
        lifecycle_entries=[],
        moe_routing_override=8,
        moe_routing_bypass=True,
        should_use_moe_lifecycle=True,
    )
    executor = ForwardExecutor(looper)
    processor = _DummyForwardProcessor()

    # Replay path: each replica should stay on the model's native router.
    outputs = _run_executor_parallel(executor, processor, apply_moe_config=False)

    assert len(outputs) == 2
    assert override_entries == []

    # Quant-time path: replicas should still install the quant-time MoE context.
    outputs = _run_executor_parallel(executor, processor, apply_moe_config=True)

    assert len(outputs) == 2
    assert override_entries == ["enter", "enter"]


def test_run_layer_stage_invokes_subset_stage(monkeypatch):
    calls = []

    def fake_run_subset_stage(looper, **kwargs):
        calls.append(kwargs["plan"].subset_index)
        return SubsetStageResult(
            processed_subset={},
            layer_inputs=kwargs["layer_inputs"],
            plan=kwargs["plan"],
        )

    monkeypatch.setattr("gptqmodel.looper.stage_layer.run_subset_stage", fake_run_subset_stage)
    monkeypatch.setattr("gptqmodel.looper.stage_layer.find_modules", lambda *_, **__: {})

    class DummyPB:
        def __init__(self, iterable):
            self._iterable = list(iterable)
            self.current_iter_step = 0

        def __iter__(self):
            return iter(self._iterable)

        def __len__(self):
            return len(self._iterable)

        def manual(self):
            return self

        def set(self, **kwargs):
            return self

        def title(self, *_):
            return self

        def subtitle(self, *_):
            return self

        def draw(self):
            return self

        def next(self):
            return self

        def close(self):
            return self

    class DummyLogger:
        def pb(self, iterable):
            return DummyPB(iterable)

        def info(self, *_, **__):
            return None

        def debug(self, *_, **__):
            return None

        def warning(self, *_, **__):
            return None

        warn = warning

        def error(self, *_, **__):
            return None

    class DummyProcessor:
        def __init__(self):
            tensor = torch.zeros(1, 1, 1)
            self.execution_config = ExecutionConfig(
                require_fwd=True,
                fwd_replay_after_process=False,
                fwd_all_modules_in_single_pass=False,
            )
            self.inputs_cache = types.SimpleNamespace(
                layer_inputs=[[tensor]],
                layer_input_kwargs=[{}],
                position_ids=[],
                attention_masks=[],
            )
            self.calibration_dataset = []
            self.log = []
            self.tasks = {}

        def collect_memory_info(self, *_):
            return None

        def pre_process_fwd_hook(self, *_):
            return lambda *a, **k: None

        def process(self, *_, **__):
            return None

        def clear_cache_data(self):
            return None

        def receive_layer_inputs(self, inputs):
            self.inputs_cache.layer_inputs = inputs

        def set_fwd_time(self, *_):
            return None

        def name(self):
            return "dummy"

        def submodule_finalize(self, *_, **__):
            return None

        def finalize(self, *_, **__):
            return None

        def log_plotly(self):
            return None

    class DummyGptqModel:
        def __init__(self):
            self.model = torch.nn.Module()
            self.model.config = types.SimpleNamespace(model_type="llama")
            self.quantize_config = QuantizeConfig(
                bits=4,
                group_size=128,
                offload_to_disk=False,
            )
            self.lm_head = None

        def pre_quantize(self, module):
            return module

        def post_quantize(self, module):
            return module

        def lm_head_pre_quantize_generate_hook(self, value):
            return value

    class DummyLooper:
        def __init__(self):
            self.gptq_model = DummyGptqModel()
            self.processors = [DummyProcessor()]
            self._quant_devices = [torch.device("cpu")]
            self._module_device_map = {}
            self._quant_device_lock = threading.Lock()
            self._moe_subset_threshold = 16
            self._dense_quant_devices = [torch.device("cpu")]
            self._moe_quant_devices = [torch.device("cpu")]
            self._dense_vram_strategy = types.SimpleNamespace()
            self._moe_vram_strategy = types.SimpleNamespace()
            self._dense_vram_strategy_explicit = False
            self._moe_vram_strategy_explicit = False
            self._layer_events = []

        def _check_loop_stop(self):
            return False

        def _is_attention_module_name(self, _name):
            return False

        def _extract_moe_group_key(self, _name):
            return None

        def _resolve_batch_total(self, _num_batches, layer_inputs):
            return len(layer_inputs)

        def _collect_row_counts(self, layer_inputs):
            return [1 for _ in layer_inputs]

        def _emit_layer_complete(self, *, layer_idx, submodule_finalized, raise_in_place):
            self._layer_events.append((layer_idx, submodule_finalized, raise_in_place))

        def _request_loop_stop(self, exc):
            self._stop_exc = exc

        def _subset_event_dispatch(self, *kwargs):
            pass

        def create_named_modules(self, module, full, is_lm_head_module, layer_index, layers_prefix, names, processor,
                                 fallback, layer_module=None) -> Dict[str, NamedModule]:
            subset = {}
            name = "self_attn.q_proj"
            subset[name] = NamedModule(module, name=name, full_name=full, layer_index=layer_index)
            return subset

    looper = DummyLooper()
    processor = looper.processors[0]
    pb = DummyPB(range(1))
    processor.layer_count = 1
    processor.pb = pb

    layers = [torch.nn.Linear(in_features=64, out_features=64)]
    layer_modules = [["foo"]]
    logger = DummyLogger()

    run_layer_stage(
        looper,
        layers=layers,
        layer_modules=layer_modules,
        planning_layer_modules=layer_modules,
        layers_prefix="model.layers",
        fallback=True,
        shared_kv_cache_dict={},
        pb=pb,
        layer_count=1,
        region_timer=None,
        finalize_progress_cls=FinalizeProgressInfo,
        logger=logger,
    )

    assert calls == [0]


def test_run_layer_stage_stops_after_last_quantized_layer(monkeypatch):
    calls = []

    def fake_run_subset_stage(looper, **kwargs):
        calls.append(kwargs["layer_index"])
        return SubsetStageResult(
            processed_subset={},
            layer_inputs=kwargs["layer_inputs"],
            plan=kwargs["plan"],
        )

    monkeypatch.setattr("gptqmodel.looper.stage_layer.run_subset_stage", fake_run_subset_stage)
    monkeypatch.setattr("gptqmodel.looper.stage_layer.find_modules", lambda *_, **__: {})

    class DummyPB:
        def __init__(self, iterable):
            self._iterable = list(iterable)
            self.current_iter_step = 0
            self.close_calls = 0

        def __iter__(self):
            return iter(self._iterable)

        def __len__(self):
            return len(self._iterable)

        def manual(self):
            return self

        def set(self, **kwargs):
            return self

        def title(self, *_):
            return self

        def subtitle(self, *_):
            return self

        def draw(self):
            return self

        def next(self):
            return self

        def close(self):
            self.close_calls += 1
            return self

    class DummyLogger:
        def pb(self, iterable):
            return DummyPB(iterable)

        def info(self, *_, **__):
            return None

        def debug(self, *_, **__):
            return None

        def warning(self, *_, **__):
            return None

        warn = warning

        def error(self, *_, **__):
            return None

    class DummyProcessor:
        def __init__(self):
            tensor = torch.zeros(1, 1, 1)
            self.execution_config = ExecutionConfig(
                require_fwd=True,
                fwd_replay_after_process=False,
                fwd_all_modules_in_single_pass=False,
            )
            self.inputs_cache = types.SimpleNamespace(
                layer_inputs=[[tensor]],
                layer_input_kwargs=[{}],
                position_ids=[],
                attention_masks=[],
            )
            self.calibration_dataset = []
            self.log = []
            self.tasks = {}

        def collect_memory_info(self, *_):
            return None

        def pre_process_fwd_hook(self, *_):
            return lambda *a, **k: None

        def process(self, *_, **__):
            return None

        def clear_cache_data(self):
            return None

        def receive_layer_inputs(self, inputs):
            self.inputs_cache.layer_inputs = inputs

        def set_fwd_time(self, *_):
            return None

        def name(self):
            return "dummy"

        def submodule_finalize(self, *_, **__):
            return None

        def finalize(self, *_, **__):
            return None

        def log_plotly(self):
            return None

    class DummyGptqModel:
        def __init__(self):
            self.model = torch.nn.Module()
            self.model.config = types.SimpleNamespace(model_type="llama")
            self.quantize_config = QuantizeConfig(
                bits=4,
                group_size=128,
                offload_to_disk=False,
                dynamic={
                    r"-:^model\.layers\.1\.foo$": {},
                    r"-:^model\.layers\.2\.foo$": {},
                },
            )
            self.lm_head = None

        def pre_quantize(self, module):
            return module

        def post_quantize(self, module):
            return module

        def lm_head_pre_quantize_generate_hook(self, value):
            return value

    class DummyLooper:
        def __init__(self):
            self.gptq_model = DummyGptqModel()
            self.processors = [DummyProcessor()]
            self._quant_devices = [torch.device("cpu")]
            self._module_device_map = {}
            self._quant_device_lock = threading.Lock()
            self._moe_subset_threshold = 16
            self._dense_quant_devices = [torch.device("cpu")]
            self._moe_quant_devices = [torch.device("cpu")]
            self._dense_vram_strategy = types.SimpleNamespace()
            self._moe_vram_strategy = types.SimpleNamespace()
            self._dense_vram_strategy_explicit = False
            self._moe_vram_strategy_explicit = False
            self._layer_events = []
            self.named_module_layers = []

        def _check_loop_stop(self):
            return False

        def _is_attention_module_name(self, _name):
            return False

        def _extract_moe_group_key(self, _name):
            return None

        def _resolve_batch_total(self, _num_batches, layer_inputs):
            return len(layer_inputs)

        def _collect_row_counts(self, layer_inputs):
            return [1 for _ in layer_inputs]

        def _emit_layer_complete(self, *, layer_idx, submodule_finalized, raise_in_place):
            self._layer_events.append((layer_idx, submodule_finalized, raise_in_place))

        def _request_loop_stop(self, exc):
            self._stop_exc = exc

        def _subset_event_dispatch(self, *kwargs):
            pass

        def create_named_modules(self, module, full, is_lm_head_module, layer_index, layers_prefix, names, processor,
                                 fallback, layer_module=None) -> Dict[str, NamedModule]:
            self.named_module_layers.append(layer_index)
            return {
                "self_attn.q_proj": NamedModule(
                    module,
                    name="self_attn.q_proj",
                    full_name=full,
                    layer_index=layer_index,
                )
            }

    looper = DummyLooper()
    processor = looper.processors[0]
    pb = DummyPB(range(3))
    processor.layer_count = 3
    processor.pb = pb

    run_layer_stage(
        looper,
        layers=[torch.nn.Linear(64, 64) for _ in range(3)],
        layer_modules=[["foo"]],
        planning_layer_modules=[["foo"]],
        layers_prefix="model.layers",
        fallback=True,
        shared_kv_cache_dict={},
        pb=pb,
        layer_count=3,
        region_timer=None,
        finalize_progress_cls=FinalizeProgressInfo,
        logger=DummyLogger(),
    )

    assert calls == [0]
    assert looper.named_module_layers == [0]
    assert pb.close_calls == 1


def test_run_layer_stage_reuses_subset_plan_for_replay(monkeypatch):
    tensor = torch.zeros(1, 1, 1)
    replay_modules = {
        "self_attn.q_proj": NamedModule(
            torch.nn.Linear(1, 1, bias=False),
            name="self_attn.q_proj",
            full_name="model.layers.0.self_attn.q_proj",
            layer_index=0,
        )
    }
    replay_plan = SubsetPlan(
        modules=replay_modules,
        subset_index=0,
        subset_total=1,
        execute_forward=True,
        replay_after_process=True,
        forward_mode="serial",
        batch_count=2,
        forward_row_counts=[2, 3],
        forward_total_rows=5,
        moe_groups={},
        forward_device_map={"self_attn.q_proj": torch.device("cuda:0")},
        calibration_coverage_policy=CalibrationCoveragePolicy(
            validate_input_coverage=False,
            fallback_enabled=True,
            prune_uncovered_modules=False,
            record_dynamic_exclusions=False,
        ),
        module_chunks=[replay_modules],
    )

    def fake_build_layer_subset_plans(*_args, **_kwargs):
        return [replay_plan]

    def fake_run_subset_stage(looper, **kwargs):
        return SubsetStageResult(
            processed_subset={},
            layer_inputs=kwargs["layer_inputs"],
            plan=kwargs["plan"],
        )

    monkeypatch.setattr("gptqmodel.looper.stage_layer.build_layer_subset_plans", fake_build_layer_subset_plans)
    monkeypatch.setattr("gptqmodel.looper.stage_layer.run_subset_stage", fake_run_subset_stage)
    monkeypatch.setattr("gptqmodel.looper.stage_layer.find_modules", lambda *_, **__: {})

    class DummyPB:
        def __init__(self, iterable):
            self._iterable = list(iterable)
            self.current_iter_step = 0

        def __iter__(self):
            return iter(self._iterable)

        def __len__(self):
            return len(self._iterable)

        def manual(self):
            return self

        def set(self, **kwargs):
            return self

        def title(self, *_):
            return self

        def subtitle(self, *_):
            return self

        def draw(self):
            return self

        def next(self):
            return self

        def close(self):
            return self

    class DummyLogger:
        def pb(self, iterable):
            return DummyPB(iterable)

        def info(self, *_, **__):
            return None

        def debug(self, *_, **__):
            return None

        def warning(self, *_, **__):
            return None

        warn = warning

        def error(self, *_, **__):
            return None

    class DummyProcessor:
        def __init__(self):
            self.execution_config = ExecutionConfig(
                require_fwd=True,
                fwd_replay_after_process=True,
                fwd_all_modules_in_single_pass=False,
            )
            self.inputs_cache = types.SimpleNamespace(
                layer_inputs=[[tensor]],
                layer_input_kwargs=[{}],
                position_ids=[],
                attention_masks=[],
            )
            self.calibration_dataset = []
            self.log = []
            self.tasks = {}

        def collect_memory_info(self, *_):
            return None

        def clear_cache_data(self):
            return None

        def receive_layer_inputs(self, inputs):
            self.inputs_cache.layer_inputs = inputs

        def name(self):
            return "dummy"

        def submodule_finalize(self, *_, **__):
            return None

        def finalize(self, *_, **__):
            return None

        def log_plotly(self):
            return None

    class DummyGptqModel:
        def __init__(self):
            self.model = torch.nn.Module()
            self.model.config = types.SimpleNamespace(model_type="llama")
            self.quantize_config = QuantizeConfig(
                bits=4,
                group_size=128,
                offload_to_disk=False,
                wait_for_submodule_finalizers=True,
            )
            self.lm_head = None

        def pre_quantize(self, module):
            return module

        def post_quantize(self, module):
            return module

        def lm_head_pre_quantize_generate_hook(self, value):
            return value

    class DummyLooper:
        def __init__(self):
            self.gptq_model = DummyGptqModel()
            self.processors = [DummyProcessor()]
            self._quant_devices = [torch.device("cpu")]
            self._module_device_map = {}
            self._quant_device_lock = threading.Lock()
            self._moe_subset_threshold = 16
            self._dense_quant_devices = [torch.device("cpu")]
            self._moe_quant_devices = [torch.device("cpu")]
            self._dense_vram_strategy = types.SimpleNamespace()
            self._moe_vram_strategy = types.SimpleNamespace()
            self._dense_vram_strategy_explicit = False
            self._moe_vram_strategy_explicit = False
            self.forward_replay_calls = []

        def _run_forward_batches(self, **kwargs):
            self.forward_replay_calls.append(kwargs)
            return [[tensor]]

        def _apply_forward_device_overrides(self, modules, forward_device_map, fallback_modules=None):
            self.forward_override_modules = modules
            self.forward_override_map = forward_device_map
            return {"self_attn.q_proj": torch.device("cpu")}

        def _restore_forward_device_overrides(self, modules, previous_devices, fallback_modules=None):
            self.restored_override_modules = modules
            self.restored_previous_devices = previous_devices

        def _check_loop_stop(self):
            return False

        def _emit_layer_complete(self, *, layer_idx, submodule_finalized, raise_in_place):
            return None

        def _request_loop_stop(self, exc):
            self._stop_exc = exc

        def _subset_event_dispatch(self, *kwargs):
            return None

        def register_dangling_thread(self, thread):
            return None

    looper = DummyLooper()
    processor = looper.processors[0]
    pb = DummyPB(range(2))
    processor.layer_count = 2
    processor.pb = pb

    run_layer_stage(
        looper,
        layers=[torch.nn.Linear(1, 1, bias=False) for _ in range(2)],
        layer_modules=[["self_attn.q_proj"]],
        planning_layer_modules=[["self_attn.q_proj"]],
        layers_prefix="model.layers",
        fallback=True,
        shared_kv_cache_dict={},
        pb=pb,
        layer_count=2,
        region_timer=None,
        finalize_progress_cls=FinalizeProgressInfo,
        logger=DummyLogger(),
    )

    assert len(looper.forward_replay_calls) == 1
    assert looper.forward_replay_calls[0]["force_serial"] is True
    assert looper.forward_replay_calls[0]["preserve_module_devices"] is True
    assert looper.forward_replay_calls[0]["progress_rows_per_batch"] == [2, 3]
    assert looper.forward_replay_calls[0]["progress_total_rows"] == 5
    assert looper.forward_override_modules is replay_modules
    assert looper.forward_override_map == {"self_attn.q_proj": torch.device("cuda:0")}
    assert looper.restored_override_modules is replay_modules


def test_replay_layer_outputs_without_plan_uses_generic_progress():
    """Untouched-layer replay should use generic progress and disable MoE config."""

    input_tensor = torch.ones(2, 1, 1)
    expected_output = input_tensor + 3.0
    timer_records = []

    class DummyPB:
        def manual(self):
            return self

        def set(self, **kwargs):
            return self

        def title(self, *_):
            return self

        def subtitle(self, *_):
            return self

        def draw(self):
            return self

        def close(self):
            return self

    class DummyLogger:
        def pb(self, iterable):
            return DummyPB()

    class DummyTimer:
        def record(self, *args, **kwargs):
            timer_records.append((args, kwargs))

    class DummyLooper:
        def __init__(self):
            self._current_subset = "not-cleared"
            self.forward_calls = []

        def _resolve_batch_total(self, _num_batches, layer_inputs):
            return len(layer_inputs)

        def _collect_row_counts(self, layer_inputs):
            return [int(batch[0].shape[0]) for batch in layer_inputs]

        def _run_forward_batches(self, **kwargs):
            self.forward_calls.append(kwargs)
            return [[expected_output.clone()]]

        def _apply_forward_device_overrides(self, *args, **kwargs):
            raise AssertionError("untouched-layer replay should not install device overrides")

        def _restore_forward_device_overrides(self, *args, **kwargs):
            raise AssertionError("untouched-layer replay should not restore device overrides")

    looper = DummyLooper()
    processor = types.SimpleNamespace(num_batches=None)

    outputs = _replay_layer_outputs(
        looper,
        module=torch.nn.Identity(),
        processor=processor,
        layer_inputs=[[input_tensor]],
        layer_input_kwargs=[{}],
        position_ids=[],
        attention_masks=[],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        layer_descriptor="model.layers.0",
        full={},
        log=DummyLogger(),
        region_timer=DummyTimer(),
        replay_plan=None,
    )

    assert len(looper.forward_calls) == 1
    assert looper.forward_calls[0]["progress_rows_per_batch"] == [2]
    assert looper.forward_calls[0]["progress_total_rows"] == 2
    assert looper.forward_calls[0]["force_serial"] is False
    assert looper.forward_calls[0]["preserve_module_devices"] is False
    assert looper.forward_calls[0]["apply_moe_config"] is False
    assert looper._current_subset is None
    assert len(outputs) == 1
    assert len(outputs[0]) == 1
    assert torch.allclose(outputs[0][0], expected_output)
    assert timer_records[0][1]["source"] == "model.layers.0:untouched"


def test_replay_layer_outputs_with_plan_uses_plan_metadata_and_device_overrides():
    """Subset-driven replay should keep its plan metadata but still disable MoE config."""

    tensor = torch.zeros(1, 1, 1)
    replay_modules = {
        "self_attn.q_proj": NamedModule(
            torch.nn.Linear(1, 1, bias=False),
            name="self_attn.q_proj",
            full_name="model.layers.0.self_attn.q_proj",
            layer_index=0,
        )
    }
    replay_plan = SubsetPlan(
        modules=replay_modules,
        subset_index=0,
        subset_total=1,
        execute_forward=True,
        replay_after_process=True,
        forward_mode="serial",
        batch_count=2,
        forward_row_counts=[2, 3],
        forward_total_rows=5,
        moe_groups={},
        forward_device_map={"self_attn.q_proj": torch.device("cuda:0")},
        calibration_coverage_policy=CalibrationCoveragePolicy(
            validate_input_coverage=False,
            fallback_enabled=True,
            prune_uncovered_modules=False,
            record_dynamic_exclusions=False,
        ),
        module_chunks=[replay_modules],
    )
    timer_records = []

    class DummyPB:
        def manual(self):
            return self

        def set(self, **kwargs):
            return self

        def title(self, *_):
            return self

        def subtitle(self, *_):
            return self

        def draw(self):
            return self

        def close(self):
            return self

    class DummyLogger:
        def pb(self, iterable):
            return DummyPB()

    class DummyTimer:
        def record(self, *args, **kwargs):
            timer_records.append((args, kwargs))

    class DummyLooper:
        def __init__(self):
            self._current_subset = replay_modules
            self.forward_calls = []

        def _run_forward_batches(self, **kwargs):
            self.forward_calls.append(kwargs)
            return [[tensor]]

        def _apply_forward_device_overrides(self, modules, forward_device_map, fallback_modules=None):
            self.forward_override_modules = modules
            self.forward_override_map = forward_device_map
            self.forward_override_fallback = fallback_modules
            return {"self_attn.q_proj": torch.device("cpu")}

        def _restore_forward_device_overrides(self, modules, previous_devices, fallback_modules=None):
            self.restored_override_modules = modules
            self.restored_previous_devices = previous_devices
            self.restored_override_fallback = fallback_modules

    looper = DummyLooper()
    processor = types.SimpleNamespace(num_batches=None)

    outputs = _replay_layer_outputs(
        looper,
        module=torch.nn.Linear(1, 1, bias=False),
        processor=processor,
        layer_inputs=[[tensor]],
        layer_input_kwargs=[{}],
        position_ids=[],
        attention_masks=[],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        layer_descriptor="model.layers.0",
        full={},
        log=DummyLogger(),
        region_timer=DummyTimer(),
        replay_plan=replay_plan,
    )

    assert len(looper.forward_calls) == 1
    assert looper.forward_calls[0]["progress_rows_per_batch"] == [2, 3]
    assert looper.forward_calls[0]["progress_total_rows"] == 5
    assert looper.forward_calls[0]["force_serial"] is True
    assert looper.forward_calls[0]["preserve_module_devices"] is True
    assert looper.forward_calls[0]["apply_moe_config"] is False
    assert looper._current_subset is None
    assert outputs == [[tensor]]
    assert looper.forward_override_modules is replay_modules
    assert looper.forward_override_map == {"self_attn.q_proj": torch.device("cuda:0")}
    assert looper.forward_calls[0]["apply_moe_config"] is False
    assert looper.restored_override_modules is replay_modules
    assert looper.restored_previous_devices == {"self_attn.q_proj": torch.device("cpu")}
    assert timer_records[0][1]["source"] == "model.layers.0:subset1/1"


def test_replay_layer_outputs_with_plan_can_skip_override_restore():
    """Replay should honor plans that intentionally keep module overrides installed."""

    tensor = torch.zeros(1, 1, 1)
    replay_modules = {
        "self_attn.q_proj": NamedModule(
            torch.nn.Linear(1, 1, bias=False),
            name="self_attn.q_proj",
            full_name="model.layers.0.self_attn.q_proj",
            layer_index=0,
        )
    }
    replay_plan = SubsetPlan(
        modules=replay_modules,
        subset_index=0,
        subset_total=1,
        execute_forward=True,
        replay_after_process=True,
        forward_mode="serial",
        batch_count=1,
        forward_row_counts=[1],
        forward_total_rows=1,
        moe_groups={},
        forward_device_map={"self_attn.q_proj": torch.device("cuda:0")},
        calibration_coverage_policy=CalibrationCoveragePolicy(
            validate_input_coverage=False,
            fallback_enabled=True,
            prune_uncovered_modules=False,
            record_dynamic_exclusions=False,
        ),
        module_chunks=[replay_modules],
        restore_forward_device_overrides=False,
    )

    class DummyPB:
        def manual(self):
            return self

        def set(self, **kwargs):
            return self

        def title(self, *_):
            return self

        def subtitle(self, *_):
            return self

        def draw(self):
            return self

        def close(self):
            return self

    class DummyLogger:
        def pb(self, iterable):
            return DummyPB()

    class DummyLooper:
        def __init__(self):
            self._current_subset = replay_modules
            self.forward_calls = []

        def _run_forward_batches(self, **kwargs):
            self.forward_calls.append(kwargs)
            return [[tensor]]

        def _apply_forward_device_overrides(self, modules, forward_device_map, fallback_modules=None):
            self.forward_override_modules = modules
            self.forward_override_map = forward_device_map
            return {"self_attn.q_proj": torch.device("cpu")}

        def _restore_forward_device_overrides(self, modules, previous_devices, fallback_modules=None):
            raise AssertionError("restore should be skipped when replay_plan disables it")

    looper = DummyLooper()
    processor = types.SimpleNamespace(num_batches=None)

    outputs = _replay_layer_outputs(
        looper,
        module=torch.nn.Linear(1, 1, bias=False),
        processor=processor,
        layer_inputs=[[tensor]],
        layer_input_kwargs=[{}],
        position_ids=[],
        attention_masks=[],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        layer_descriptor="model.layers.0",
        full={},
        log=DummyLogger(),
        region_timer=None,
        replay_plan=replay_plan,
    )

    assert outputs == [[tensor]]
    assert looper.forward_override_modules is replay_modules
    assert looper.forward_override_map == {"self_attn.q_proj": torch.device("cuda:0")}


def test_replay_layer_outputs_with_multi_device_plan_skips_moe_config():
    """Multi-device replay should disable MoE config without changing override install."""

    tensor = torch.zeros(1, 1, 1)
    replay_modules = {
        "self_attn.q_proj": NamedModule(
            torch.nn.Linear(1, 1, bias=False),
            name="self_attn.q_proj",
            full_name="model.layers.0.self_attn.q_proj",
            layer_index=0,
        )
    }
    replay_plan = SubsetPlan(
        modules=replay_modules,
        subset_index=0,
        subset_total=1,
        execute_forward=True,
        replay_after_process=True,
        forward_mode="serial",
        batch_count=2,
        forward_row_counts=[2, 3],
        forward_total_rows=5,
        moe_groups={},
        forward_device_map={
            "self_attn.q_proj": torch.device("cuda:0"),
            "mlp.experts.0.gate_proj": torch.device("cuda:1"),
        },
        calibration_coverage_policy=CalibrationCoveragePolicy(
            validate_input_coverage=False,
            fallback_enabled=True,
            prune_uncovered_modules=False,
            record_dynamic_exclusions=False,
        ),
        module_chunks=[replay_modules],
        restore_forward_device_overrides=False,
    )
    timer_records = []

    class DummyPB:
        def manual(self):
            return self

        def set(self, **kwargs):
            return self

        def title(self, *_):
            return self

        def subtitle(self, *_):
            return self

        def draw(self):
            return self

        def close(self):
            return self

    class DummyLogger:
        def pb(self, iterable):
            return DummyPB()

    class DummyTimer:
        def record(self, *args, **kwargs):
            timer_records.append((args, kwargs))

    class DummyLooper:
        def __init__(self):
            self._current_subset = replay_modules
            self.forward_calls = []
            self.override_calls = []

        def _run_forward_batches(self, **kwargs):
            self.forward_calls.append(kwargs)
            return [[tensor]]

        def _apply_forward_device_overrides(self, modules, forward_device_map, fallback_modules=None):
            self.override_calls.append((modules, forward_device_map, fallback_modules))
            return {}

        def _restore_forward_device_overrides(self, modules, previous_devices, fallback_modules=None):
            raise AssertionError("restore should be skipped when replay_plan disables it")

    looper = DummyLooper()
    processor = types.SimpleNamespace(num_batches=None)

    outputs = _replay_layer_outputs(
        looper,
        module=torch.nn.Linear(1, 1, bias=False),
        processor=processor,
        layer_inputs=[[tensor]],
        layer_input_kwargs=[{}],
        position_ids=[],
        attention_masks=[],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        layer_descriptor="model.layers.0",
        full={},
        log=DummyLogger(),
        region_timer=DummyTimer(),
        replay_plan=replay_plan,
    )

    assert outputs == [[tensor]]
    assert looper.override_calls == [
        (
            replay_modules,
            {
                "self_attn.q_proj": torch.device("cuda:0"),
                "mlp.experts.0.gate_proj": torch.device("cuda:1"),
            },
            {},
        )
    ]
    assert len(looper.forward_calls) == 1
    assert looper.forward_calls[0]["progress_rows_per_batch"] == [2, 3]
    assert looper.forward_calls[0]["progress_total_rows"] == 5
    assert looper.forward_calls[0]["force_serial"] is True
    assert looper.forward_calls[0]["preserve_module_devices"] is True
    assert looper.forward_calls[0]["apply_moe_config"] is False
    assert timer_records[0][1]["source"] == "model.layers.0:subset1/1"


class _ToySelfAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(1, 1, bias=False)
        self.k_proj = torch.nn.Linear(1, 1, bias=False)
        self.v_proj = torch.nn.Linear(1, 1, bias=False)
        self.o_proj = torch.nn.Linear(1, 1, bias=False)
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            torch.nn.init.constant_(proj.weight, 1.0)

    def forward(self, hidden_states):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        return self.o_proj(q + k + v)


class _ToyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(1, 1, bias=False)
        self.up_proj = torch.nn.Linear(1, 1, bias=False)
        self.down_proj = torch.nn.Linear(1, 1, bias=False)
        for proj in (self.gate_proj, self.up_proj, self.down_proj):
            torch.nn.init.constant_(proj.weight, 1.0)

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(gate + up)


class _ToyLlamaDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.self_attn = _ToySelfAttention()
        self.post_attention_layernorm = torch.nn.Identity()
        self.mlp = _ToyMLP()
        self.forward_inputs = []

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        self.forward_inputs.append(hidden_states.detach().clone())
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states


def test_run_layer_stage_replays_untouched_layer_outputs_when_all_modules_skipped(monkeypatch):
    observed_layer_inputs = []

    def fake_run_subset_stage(looper, **kwargs):
        observed_layer_inputs.append(
            (
                kwargs["layer_index"],
                kwargs["plan"].subset_index,
                kwargs["layer_inputs"][0][0].detach().clone(),
            )
        )
        return SubsetStageResult(
            processed_subset={},
            layer_inputs=kwargs["layer_inputs"],
            plan=kwargs["plan"],
        )

    monkeypatch.setattr("gptqmodel.looper.stage_layer.run_subset_stage", fake_run_subset_stage)
    monkeypatch.setattr("gptqmodel.looper.stage_layer.find_modules", lambda *_, **__: {})

    class DummyPB:
        def __init__(self, iterable):
            self._iterable = list(iterable)
            self.current_iter_step = 0

        def __iter__(self):
            return iter(self._iterable)

        def __len__(self):
            return len(self._iterable)

        def manual(self):
            return self

        def set(self, **kwargs):
            return self

        def title(self, *_):
            return self

        def subtitle(self, *_):
            return self

        def draw(self):
            return self

        def next(self):
            return self

        def close(self):
            return self

    class DummyLogger:
        def pb(self, iterable):
            return DummyPB(iterable)

        def info(self, *_, **__):
            return None

        def debug(self, *_, **__):
            return None

        def warning(self, *_, **__):
            return None

        warn = warning

        def error(self, *_, **__):
            return None

    class DummyProcessor:
        def __init__(self, initial_inputs):
            self.execution_config = ExecutionConfig(
                require_fwd=True,
                fwd_replay_after_process=True,
                fwd_all_modules_in_single_pass=False,
                subset_forward_early_stop=True,
            )
            self.inputs_cache = types.SimpleNamespace(
                layer_inputs=initial_inputs,
                layer_input_kwargs=[{}],
                position_ids=[],
                attention_masks=[],
            )
            self.calibration_dataset = []
            self.log = []
            self.tasks = {}

        def collect_memory_info(self, *_):
            return None

        def clear_cache_data(self):
            self.tasks = {}
            self.inputs_cache.layer_inputs = []

        def receive_layer_inputs(self, inputs):
            self.inputs_cache.layer_inputs = inputs

        def set_fwd_time(self, *_):
            return None

        def name(self):
            return "GPTQProcessor"

        def submodule_finalize(self, *_, **__):
            return None

        def finalize(self, *_, **__):
            return None

        def log_plotly(self):
            return None

    class DummyGptqModel:
        def __init__(self):
            self.model = torch.nn.Module()
            self.model.config = types.SimpleNamespace(model_type="llama")
            self.quantize_config = QuantizeConfig(
                bits=4,
                group_size=128,
                offload_to_disk=False,
                wait_for_submodule_finalizers=True,
                dynamic={
                    r"-:^model\.layers\.0\.": {},
                },
            )
            self.lm_head = None

        def pre_quantize(self, module):
            return module

        def post_quantize(self, module):
            return module

        def lm_head_pre_quantize_generate_hook(self, value):
            return value

    class DummyLooper:
        def __init__(self, layers, initial_inputs):
            self.gptq_model = DummyGptqModel()
            self.processors = [DummyProcessor(initial_inputs)]
            self._quant_devices = [torch.device("cpu")]
            self._module_device_map = {}
            self._quant_device_lock = threading.Lock()
            self._moe_subset_threshold = 16
            self._dense_quant_devices = [torch.device("cpu")]
            self._moe_quant_devices = [torch.device("cpu")]
            self._dense_vram_strategy = types.SimpleNamespace()
            self._moe_vram_strategy = types.SimpleNamespace()
            self._dense_vram_strategy_explicit = False
            self._moe_vram_strategy_explicit = False
            self._current_subset = None
            self.support_batch_quantize = False
            self.moe_routing_override = None
            self.moe_routing_bypass = False
            self.forward_layer_indices = []
            self.layers = layers

        def _run_forward_batches(self, **kwargs):
            self.forward_layer_indices.append(kwargs["layer_index"])
            outputs = []
            for batch_inputs in kwargs["layer_inputs"]:
                hidden_states = batch_inputs[0]
                output = kwargs["module"](
                    hidden_states=hidden_states,
                    attention_mask=None,
                    position_ids=None,
                )
                outputs.append([output])
            return outputs

        def _check_loop_stop(self):
            return False

        def _is_attention_module_name(self, name):
            return name.startswith("self_attn.")

        def _extract_moe_group_key(self, _name):
            return None

        def _resolve_batch_total(self, _num_batches, layer_inputs):
            return len(layer_inputs)

        def _collect_row_counts(self, layer_inputs):
            return [int(batch[0].shape[0]) for batch in layer_inputs]

        def _emit_layer_complete(self, *, layer_idx, submodule_finalized, raise_in_place):
            return None

        def _request_loop_stop(self, exc):
            self._stop_exc = exc

        def _subset_event_dispatch(self, *kwargs):
            return None

        def register_dangling_thread(self, thread):
            return None

        def create_named_modules(
            self,
            module,
            full,
            is_lm_head_module,
            layer_index,
            layers_prefix,
            names,
            processor,
            fallback,
            layer_module=None,
        ) -> Dict[str, NamedModule]:
            subset = {}
            for name in names:
                full_name = f"{layers_prefix}.{layer_index}.{name}"
                if self.gptq_model.quantize_config.dynamic_get(layer_name=full_name) is False:
                    continue
                subset[name] = NamedModule(
                    module.get_submodule(name),
                    name=name,
                    full_name=full_name,
                    layer_index=layer_index,
                )
            return subset

    input_tensor = torch.tensor([[[2.0]]])
    layers = [_ToyLlamaDecoderLayer(), _ToyLlamaDecoderLayer()]
    looper = DummyLooper(layers, initial_inputs=[[input_tensor.clone()]])
    processor = looper.processors[0]
    pb = DummyPB(range(2))
    processor.layer_count = 2
    processor.pb = pb

    run_layer_stage(
        looper,
        layers=layers,
        layer_modules=[
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ["self_attn.o_proj"],
            ["mlp.gate_proj", "mlp.up_proj"],
            ["mlp.down_proj"],
        ],
        planning_layer_modules=[
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ["self_attn.o_proj"],
            ["mlp.gate_proj", "mlp.up_proj"],
            ["mlp.down_proj"],
        ],
        layers_prefix="model.layers",
        fallback=True,
        shared_kv_cache_dict={},
        pb=pb,
        layer_count=2,
        region_timer=None,
        finalize_progress_cls=FinalizeProgressInfo,
        logger=DummyLogger(),
    )

    layer1_inputs = [
        layer_input
        for layer_idx, _subset_idx, layer_input in observed_layer_inputs
        if layer_idx == 1
    ]
    expected_layer0_output = input_tensor * 6.0

    assert looper.forward_layer_indices == [0]
    assert len(layers[0].forward_inputs) == 1
    assert torch.allclose(layers[0].forward_inputs[0], input_tensor)
    assert layer1_inputs
    assert all(torch.allclose(layer_input, expected_layer0_output) for layer_input in layer1_inputs)


def test_capture_pristine_group_context_preserves_untouched_layer_io(monkeypatch):
    observed = {}
    sentinel_outputs = [[torch.randn(1, 1, 1)]]

    def fake_replay_layer_outputs(*_args, **kwargs):
        observed["replay_kwargs"] = kwargs
        return sentinel_outputs

    monkeypatch.setattr("gptqmodel.looper.stage_layer._replay_layer_outputs", fake_replay_layer_outputs)

    class DummyProcessor:
        def uses_grouped_optimization(self):
            return True

        def receive_layer_forward_context(self, **kwargs):
            observed["receive_kwargs"] = kwargs

    tensor = torch.randn(1, 1, 1)
    subset_plan = SubsetPlan(
        modules={},
        subset_index=0,
        subset_total=1,
        execute_forward=True,
        replay_after_process=True,
        forward_mode="serial",
        batch_count=1,
        forward_row_counts=[1],
        forward_total_rows=1,
        moe_groups={},
        forward_device_map={},
        calibration_coverage_policy=CalibrationCoveragePolicy(
            validate_input_coverage=False,
            fallback_enabled=True,
            prune_uncovered_modules=False,
            record_dynamic_exclusions=False,
        ),
        module_chunks=[{}],
    )

    _capture_pristine_group_context(
        looper=types.SimpleNamespace(),
        processor=DummyProcessor(),
        module=torch.nn.Identity(),
        pristine_module=None,
        subset_plans=[subset_plan],
        layer_inputs=[[tensor]],
        layer_input_kwargs=[{}],
        position_ids=[],
        attention_masks=[],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        layer_descriptor="model.layers.0",
        full={},
        log=None,
        region_timer=None,
    )

    assert observed["replay_kwargs"]["replay_plan"] is None
    assert observed["receive_kwargs"]["layer_outputs"] is sentinel_outputs
    assert observed["receive_kwargs"]["layer_inputs"] == [[tensor]]
    assert observed["receive_kwargs"]["layer_input_kwargs"] == [{}]
    assert observed["receive_kwargs"]["subset_total"] == 1


def test_masked_hook_wrapper_trims_left_padded_inputs_before_add_batch():
    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = types.SimpleNamespace(quant_region_timer=None)

    class _FakeTask:
        def __init__(self):
            self.add_batch_input = None

        def add_batch(self, inp, out, batch_index=None):
            self.add_batch_input = inp

    processor = types.SimpleNamespace()
    task = _FakeTask()

    input_ids = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [30.0, 30.0], [40.0, 40.0]],
            [[3.0, 3.0], [4.0, 4.0], [50.0, 50.0], [60.0, 60.0]],
        ],
        dtype=torch.float32,
    )

    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    looper._set_processor_mask(processor, attention_mask)

    def inner_hook(module, hook_inputs, hook_output):
        task.add_batch(hook_inputs[0], torch.empty(0))
        return module, hook_inputs, hook_output

    wrapped_hook = looper._masked_hook_wrapper(processor, inner_hook, "test")
    wrapped_hook(
        None,
        (input_ids,),
        torch.empty((2, 4, 2)),
    )

    assert task.add_batch_input is not None
    assert task.add_batch_input.shape == (4, 2)
    assert torch.equal(
        task.add_batch_input,
        torch.tensor(
            [
                [30.0, 30.0],
                [40.0, 40.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ],
            dtype=torch.float32,
        ),
    )
