import sys
import threading
import types
import weakref
from typing import Dict

import torch

import gptqmodel.looper.stage_subset as stage_subset_module
from gptqmodel import QuantizeEmbed, QuantizeEmbedConfig
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.forward_executor import ForwardExecutor
from gptqmodel.looper.gptq_processor import GPTQProcessor
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
        self.model = torch.nn.Module()
        self.model.config = types.SimpleNamespace(tie_word_embeddings=False)
        self.layer_callback = None


def _make_looper():
    processors = [types.SimpleNamespace(layer_count=0, pb=None)]
    return ModuleLooper(model=_DummyQModel(), processors=processors)


def test_module_looper_preserves_embedding_only_setting():
    looper = ModuleLooper(
        model=_DummyQModel(),
        processors=[],
        embed_quant_config=QuantizeEmbedConfig(
            embed_quant_mode=QuantizeEmbed.INPUT,
            embed_only=False,
        ),
    )

    assert looper.embed_only is False


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


def test_deferred_boundary_prefix_materializes_once_before_publication():
    looper = _make_looper()
    model = looper.gptq_model
    processors = looper.processors
    calls = []

    class Boundary:
        def materialize_deferred_prefix(self, **kwargs):
            calls.append(kwargs)

    looper._materialize_deferred_boundary_prefix(Boundary())

    assert calls == [{"model": model, "processors": processors}]


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


def test_forward_preparation_allows_processor_to_materialize_deferred_weight():
    target = torch.nn.Linear(4, 4, bias=False, device="meta")
    named = NamedModule(
        target,
        name="mlp.experts.1.gate_proj",
        full_name="model.layers.0.mlp.experts.1.gate_proj",
        layer_index=0,
    )
    calls = []

    class Processor:
        def prepare_runtime_weight_for_forward(self, **kwargs):
            calls.append(kwargs)
            kwargs["module"].module.weight = torch.nn.Parameter(
                torch.ones((4, 4)), requires_grad=False
            )
            return kwargs["module"].module

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.processors = [Processor()]
    prepared = looper._prepare_named_module_for_forward(
        named_module=named,
        fallback_device=torch.device("cpu"),
    )

    assert prepared is target
    assert prepared.weight.device.type == "cpu"
    assert calls == [
        {"module": named, "target_device": torch.device("cpu")}
    ]
    assert named.target_device == torch.device("cpu")


def test_forward_preparation_materializes_directly_on_preferred_device():
    target = torch.nn.Linear(4, 4, bias=False)
    named = NamedModule(
        target,
        name="mlp.experts.1.gate_proj",
        full_name="model.layers.0.mlp.experts.1.gate_proj",
        layer_index=0,
    )
    named.state["preferred_quant_device"] = torch.device("cuda:1")
    calls = []

    class Model:
        @staticmethod
        def shell_module_materialize(**kwargs):
            calls.append(kwargs)
            return kwargs["target_submodule"]

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.processors = []
    looper.gptq_model = Model()

    prepared = looper._prepare_named_module_for_forward(
        named_module=named,
        fallback_device=torch.device("cuda:0"),
    )

    assert prepared is target
    assert calls == [
        {
            "target_submodule": target,
            "device": torch.device("cuda:1"),
            "role": "forward",
            "named_module": named,
        }
    ]
    assert named.target_device == torch.device("cuda:1")
    assert target.target_device == torch.device("cuda:1")


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
            prepare_layer_replay_keep_mask=lambda **kwargs: kwargs["keep_mask"],
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
    def clone_module_for_devices_fn(
        module, devices, progress_callback=None, root_module=None
    ):
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


def test_stage_layer_forces_sync_finalizers_for_durable_boundary():
    looper = types.SimpleNamespace(
        gptq_model=types.SimpleNamespace(
            quantize_config=QuantizeConfig(
                bits=4,
                group_size=128,
                wait_for_submodule_finalizers=False,
            ),
            quantization_layer_boundary_checkpoint=object(),
        ),
        _quant_devices=[torch.device("cuda:0")],
    )

    assert _should_drain_finalize_futures_synchronously(
        looper,
        finalize_tasks=[(types.SimpleNamespace(), None, None, None, None)],
    ) is True
def test_stage_layer_forces_sync_finalizers_for_w4a():
    for activation in ("w4afp8", "w4a_nvfp4"):
        looper = types.SimpleNamespace(
            gptq_model=types.SimpleNamespace(
                quantize_config=QuantizeConfig(
                    bits=4,
                    group_size=128,
                    activation=activation,
                    wait_for_submodule_finalizers=False,
                )
            ),
            _quant_devices=[torch.device("cuda:0")],
        )

        assert _should_drain_finalize_futures_synchronously(
            looper,
            finalize_tasks=[(types.SimpleNamespace(), None, None, None, None)],
        ) is True


def test_stage_layer_releases_cache_after_w4a_finalize():
    for activation in ("w4afp8", "w4a_nvfp4"):
        looper = types.SimpleNamespace(
            gptq_model=types.SimpleNamespace(
                quantize_config=QuantizeConfig(
                    bits=4,
                    group_size=128,
                    activation=activation,
                    offload_to_disk=False,
                )
            )
        )

        assert _should_empty_cache_after_sync_finalize(
            looper,
            finalize_tasks=[(types.SimpleNamespace(), None, None, None, None)],
        ) is True


def test_stage_layer_forces_sync_finalizers_for_multi_device_generic_processor():
    looper = types.SimpleNamespace(
        gptq_model=types.SimpleNamespace(
            quantize_config=QuantizeConfig(
                bits=4,
                group_size=128,
                wait_for_submodule_finalizers=False,
            )
        ),
        _quant_devices=[torch.device("cuda:0"), torch.device("cuda:1")],
    )

    assert _should_drain_finalize_futures_synchronously(
        looper,
        finalize_tasks=[(types.SimpleNamespace(), None, None, None, None)],
    ) is True


def test_stage_layer_forces_sync_finalizers_for_multi_device_gptq():
    looper = types.SimpleNamespace(
        gptq_model=types.SimpleNamespace(
            quantize_config=QuantizeConfig(
                bits=4,
                group_size=128,
                wait_for_submodule_finalizers=False,
            )
        ),
        _quant_devices=[torch.device("cuda:0"), torch.device("cuda:1")],
    )
    gptq_processor = object.__new__(GPTQProcessor)

    assert _should_drain_finalize_futures_synchronously(
        looper,
        finalize_tasks=[(gptq_processor, None, None, None, None)],
    ) is True


def test_stage_layer_keeps_async_finalizers_for_single_device_gptq():
    looper = types.SimpleNamespace(
        gptq_model=types.SimpleNamespace(
            quantize_config=QuantizeConfig(
                bits=4,
                group_size=128,
                wait_for_submodule_finalizers=False,
            )
        ),
        _quant_devices=[torch.device("cuda:0")],
    )
    gptq_processor = object.__new__(GPTQProcessor)

    assert _should_drain_finalize_futures_synchronously(
        looper,
        finalize_tasks=[(gptq_processor, None, None, None, None)],
    ) is False


def test_stage_layer_forces_sync_finalizers_for_multi_device_awq():
    looper = types.SimpleNamespace(
        gptq_model=types.SimpleNamespace(
            quantize_config=QuantizeConfig(
                bits=4,
                group_size=128,
                wait_for_submodule_finalizers=False,
            )
        ),
        _quant_devices=[torch.device("cuda:0"), torch.device("cuda:1")],
    )
    awq_processor = object.__new__(AWQProcessor)

    assert _should_drain_finalize_futures_synchronously(
        looper,
        finalize_tasks=[(awq_processor, None, None, None, None)],
    ) is True


def test_stage_layer_keeps_async_finalizers_for_single_device_awq():
    looper = types.SimpleNamespace(
        gptq_model=types.SimpleNamespace(
            quantize_config=QuantizeConfig(
                bits=4,
                group_size=128,
                wait_for_submodule_finalizers=False,
            )
        ),
        _quant_devices=[torch.device("cuda:0")],
    )
    awq_processor = object.__new__(AWQProcessor)

    assert _should_drain_finalize_futures_synchronously(
        looper,
        finalize_tasks=[(awq_processor, None, None, None, None)],
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


def test_subset_pass_without_forward_skips_forward_device_overrides(monkeypatch):
    class DummyProcessor:
        execution_config = ExecutionConfig(require_fwd=False)
        tasks = {}

        def set_fwd_time(self, *_):
            return None

    class DummyLooper:
        gptq_model = types.SimpleNamespace(
            quantize_config=QuantizeConfig(bits=4, group_size=128),
        )

        def _apply_forward_device_overrides(self, *_, **__):
            raise AssertionError("a no-forward pass must not install forward device overrides")

    monkeypatch.setattr(stage_subset_module, "torch_sync", lambda: None)

    plan = SubsetPlan(
        modules={},
        subset_index=0,
        subset_total=1,
        execute_forward=False,
        replay_after_process=False,
        forward_mode="parallel",
        batch_count=0,
        forward_row_counts=[],
        forward_total_rows=1,
        moe_groups={},
        forward_device_map={"mlp.experts.0.gate_proj": torch.device("cuda:0")},
        calibration_coverage_policy=CalibrationCoveragePolicy(
            validate_input_coverage=False,
            fallback_enabled=True,
            prune_uncovered_modules=False,
            record_dynamic_exclusions=False,
        ),
        module_chunks=[{}],
    )

    processed, outputs, used_data_parallel = stage_subset_module._run_single_subset_pass(
        looper=DummyLooper(),
        processor=DummyProcessor(),
        module=torch.nn.Identity(),
        plan=plan,
        layer_inputs=[],
        layer_input_kwargs=[],
        position_ids=[],
        attention_masks=[],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        layer_descriptor="model.layers.0",
        layer_title="Layer 0",
        layer_index=0,
        full={},
        fallback=None,
        shared_kv_cache_dict={},
        pb=None,
        logger=types.SimpleNamespace(),
        is_awq_processor=False,
        execute_forward=False,
    )

    assert processed == {}
    assert outputs is None
    assert used_data_parallel is False


def test_subset_pass_restored_capture_skips_forward(monkeypatch):
    class DummyProcessor:
        execution_config = ExecutionConfig(
            require_fwd=True,
            fwd_replay_after_process=True,
        )
        tasks = {}

        def restore_subset_capture_frontier(self, **kwargs):
            self.restored = kwargs
            return True

        def set_fwd_time(self, *_):
            return None

    class DummyLooper:
        gptq_model = types.SimpleNamespace(
            quantize_config=QuantizeConfig(bits=4, group_size=128),
        )

        def _prepare_layer_direct_state_for_forward(self, *_, **__):
            raise AssertionError("restored capture must skip the subset forward")

    monkeypatch.setattr(stage_subset_module, "torch_sync", lambda: None)
    processor = DummyProcessor()
    plan = SubsetPlan(
        modules={},
        subset_index=2,
        subset_total=5,
        execute_forward=True,
        replay_after_process=True,
        forward_mode="parallel",
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
    processed, outputs, _ = stage_subset_module._run_single_subset_pass(
        looper=DummyLooper(),
        processor=processor,
        module=torch.nn.Identity(),
        plan=plan,
        layer_inputs=[],
        layer_input_kwargs=[],
        position_ids=[],
        attention_masks=[],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        layer_descriptor="model.layers.3",
        layer_title="Layer 3",
        layer_index=3,
        full={},
        fallback=None,
        shared_kv_cache_dict={},
        pb=None,
        logger=types.SimpleNamespace(),
        is_awq_processor=False,
    )
    assert processed == {}
    assert outputs is None
    assert processor.restored["layer_index"] == 3
    assert processor.restored["subset_index"] == 2


def test_subset_pass_finishes_zero_route_recovery_before_quant_fanout(monkeypatch):
    events = []

    class DummyPB:
        def manual(self):
            return self

        def set(self, **_kwargs):
            return self

        def title(self, *_args):
            return self

        def subtitle(self, *_args):
            return self

        def draw(self):
            return self

        def close(self):
            return self

    class DummyLogger:
        def pb(self, _iterable):
            return DummyPB()

        def error(self, *_args, **_kwargs):
            return None

        def isEnabledFor(self, _level):
            return False

    class ImmediateFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class ImmediatePool:
        def submit(self, _device, callback, *args):
            events.append("submit")
            return ImmediateFuture(callback(*args))

    class RecoveryContext:
        def __enter__(self):
            events.append("recovery_enter")

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback
            events.append("recovery_exit")

    projection = NamedModule(
        torch.nn.Linear(4, 4, bias=False),
        name="mlp.experts.65.gate_proj",
        full_name="mtp.1.mlp.experts.65.gate_proj",
        layer_index=1,
    )

    class DummyProcessor:
        execution_config = ExecutionConfig(
            require_fwd=True,
            fwd_replay_after_process=True,
        )
        tasks = {projection.name: object()}

        def begin_shared_input_capture(self, *args, **kwargs):
            return {}

        def end_shared_input_capture(self, subset_names):
            return {}

        def pre_process_fwd_hook(self, _name):
            return lambda *_args, **_kwargs: None

        def plan_subset_zero_route_recovery(self, *, subset, layer_module):
            assert subset == {projection.name: projection}
            assert isinstance(layer_module, torch.nn.Identity)
            events.append("census")
            return (projection.name,)

        def finish_subset_zero_route_recovery(self, *, subset, task_names):
            assert subset == {projection.name: projection}
            assert task_names == (projection.name,)
            events.append("recovery_finish")

        def validate_subset_capture_readiness(self, *, subset, layer_module):
            assert subset == {projection.name: projection}
            assert isinstance(layer_module, torch.nn.Identity)
            events.append("readiness")

        def commit_subset_capture_frontier(self, **_kwargs):
            events.append("capture_commit")

        def set_fwd_time(self, _value):
            return None

        def process(self, **_kwargs):
            events.append("process")

    class DummyQModel:
        quantize_config = QuantizeConfig(bits=4, group_size=128)

        def zero_route_recovery_context(self, **_kwargs):
            return RecoveryContext()

    class DummyLooper:
        gptq_model = DummyQModel()

        def _masked_hook_wrapper(self, _processor, hook, _source):
            return hook

        def _masked_pre_hook_wrapper(self, _processor, hook, _source):
            return hook

        def _prepare_layer_direct_state_for_forward(self, *_args, **_kwargs):
            events.append("forward_prepare")

        def _prepare_named_module_for_forward(self, **_kwargs):
            return None

        def _run_forward_batches(self, **kwargs):
            events.append(kwargs["progress_stage"])
            return None

        def _prepare_named_module_for_quantization(self, **_kwargs):
            events.append("quant_prepare")
            return torch.device("cpu")

    monkeypatch.setattr(stage_subset_module, "DEVICE_THREAD_POOL", ImmediatePool())
    monkeypatch.setattr(stage_subset_module, "torch_sync", lambda: None)
    monkeypatch.setattr(
        stage_subset_module,
        "_emit_moe_parallel_quant_subset_telemetry",
        lambda **_kwargs: None,
    )
    plan = SubsetPlan(
        modules={projection.name: projection},
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
            fallback_enabled=False,
            prune_uncovered_modules=False,
            record_dynamic_exclusions=False,
        ),
        module_chunks=[{projection.name: projection}],
    )

    processed, outputs, _ = stage_subset_module._run_single_subset_pass(
        looper=DummyLooper(),
        processor=DummyProcessor(),
        module=torch.nn.Identity(),
        plan=plan,
        layer_inputs=[[torch.zeros(1, 1, 4)]],
        layer_input_kwargs=[{}],
        position_ids=[],
        attention_masks=[],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        layer_descriptor="mtp.1",
        layer_title="MTP 1",
        layer_index=1,
        full={projection.name: projection},
        fallback=None,
        shared_kv_cache_dict={},
        pb=None,
        logger=DummyLogger(),
        is_awq_processor=False,
    )

    assert outputs is None
    assert processed == {projection.name: projection}
    assert events == [
        "forward_prepare",
        "Forward",
        "census",
        "recovery_enter",
        "Zero-route recovery",
        "recovery_exit",
        "recovery_finish",
        "capture_commit",
        "readiness",
        "quant_prepare",
        "submit",
        "process",
    ]


def test_subset_pass_restored_recovery_frontier_is_validated_before_fanout(
    monkeypatch,
):
    events = []
    projection = NamedModule(
        torch.nn.Linear(4, 4, bias=False),
        name="mlp.experts.65.gate_proj",
        full_name="mtp.1.mlp.experts.65.gate_proj",
        layer_index=1,
    )

    class ImmediateFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class ImmediatePool:
        def submit(self, _device, callback, *args):
            events.append("submit")
            return ImmediateFuture(callback(*args))

    class DummyProcessor:
        execution_config = ExecutionConfig(
            require_fwd=True,
            fwd_replay_after_process=True,
        )
        tasks = {projection.name: object()}

        def restore_subset_capture_frontier(self, **_kwargs):
            events.append("restore")
            return True

        def validate_subset_capture_readiness(self, *, subset, layer_module):
            assert subset == {projection.name: projection}
            assert isinstance(layer_module, torch.nn.Identity)
            events.append("readiness")

        def set_fwd_time(self, _value):
            return None

        def process(self, **_kwargs):
            events.append("process")

    class DummyLooper:
        gptq_model = types.SimpleNamespace(
            quantize_config=QuantizeConfig(bits=4, group_size=128),
        )

        def _prepare_layer_direct_state_for_forward(self, *_args, **_kwargs):
            raise AssertionError("a restored frontier must not replay forward")

        def _prepare_named_module_for_quantization(self, **_kwargs):
            events.append("quant_prepare")
            return torch.device("cpu")

    monkeypatch.setattr(stage_subset_module, "DEVICE_THREAD_POOL", ImmediatePool())
    monkeypatch.setattr(stage_subset_module, "torch_sync", lambda: None)
    monkeypatch.setattr(
        stage_subset_module,
        "_emit_moe_parallel_quant_subset_telemetry",
        lambda **_kwargs: None,
    )
    plan = SubsetPlan(
        modules={projection.name: projection},
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
            fallback_enabled=False,
            prune_uncovered_modules=False,
            record_dynamic_exclusions=False,
        ),
        module_chunks=[{projection.name: projection}],
    )

    processed, outputs, _ = stage_subset_module._run_single_subset_pass(
        looper=DummyLooper(),
        processor=DummyProcessor(),
        module=torch.nn.Identity(),
        plan=plan,
        layer_inputs=[],
        layer_input_kwargs=[],
        position_ids=[],
        attention_masks=[],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        layer_descriptor="mtp.1",
        layer_title="MTP 1",
        layer_index=1,
        full={projection.name: projection},
        fallback=None,
        shared_kv_cache_dict={},
        pb=None,
        logger=types.SimpleNamespace(),
        is_awq_processor=False,
    )

    assert outputs is None
    assert processed == {projection.name: projection}
    assert events == ["restore", "readiness", "quant_prepare", "submit", "process"]


def test_forward_device_override_materializes_meta_fallback_named_module():
    original = NamedModule(
        torch.nn.Linear(4, 4, bias=False, device="meta"),
        name="mlp.experts.0.gate_proj",
        full_name="model.layers.0.mlp.experts.0.gate_proj",
        layer_index=0,
    )
    calls = []

    class DummyQModel:
        def shell_module_materialize(self, **kwargs):
            calls.append(kwargs)
            return torch.nn.Linear(4, 4, bias=False, device=kwargs["device"])

    looper = object.__new__(ModuleLooper)
    looper.gptq_model = DummyQModel()

    previous = looper._apply_forward_device_overrides(
        subset={},
        device_map={original.name: torch.device("cpu")},
        fallback_modules={original.name: original},
    )

    assert previous == {}
    assert len(calls) == 1
    assert calls[0]["role"] == "forward"
    assert calls[0]["named_module"] is original
    assert calls[0]["device"] == torch.device("cpu")
    assert original.module.weight.device.type == "cpu"


def test_lazy_forward_materializes_direct_state_but_not_projection_weights():
    class PackedProjection(torch.nn.Module):
        QUANT_TYPE = "test-packed"

        def __init__(self):
            super().__init__()
            self.register_buffer("trellis", torch.empty(2, device="meta"))

    class LazyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.empty(2, device="meta"))
            self.proj = torch.nn.Linear(2, 2, bias=False, device="meta")
            self.packed = PackedProjection()

    layer = LazyLayer()
    projection = NamedModule(
        layer.proj,
        name="proj",
        full_name="model.layers.0.proj",
        layer_index=0,
    )
    calls = []

    class DummyQModel:
        def shell_direct_meta_materialize(self, *, target_submodule, device):
            calls.append((target_submodule, device))
            target_submodule.scale = torch.nn.Parameter(torch.ones(2, device=device))

    looper = object.__new__(ModuleLooper)
    looper.gptq_model = DummyQModel()

    count = looper._prepare_layer_direct_state_for_forward(
        layer,
        torch.device("cpu"),
        projection_modules={"proj": projection},
    )

    assert count == 1
    assert calls == [(layer, torch.device("cpu"))]
    assert layer.scale.device.type == "cpu"
    assert layer.proj.weight.device.type == "meta"
    assert layer.packed.trellis.device.type == "meta"


def test_rehome_processor_task_rebinds_dict_capture_to_quant_source():
    named = NamedModule(
        torch.nn.Linear(4, 4, bias=False, device="meta"),
        name="mlp.experts.0.gate_proj",
        full_name="model.layers.0.mlp.experts.0.gate_proj",
        layer_index=0,
    )
    quant_source = torch.nn.Linear(4, 4, bias=False)
    named.state["quant_source_module"] = quant_source
    capture = types.SimpleNamespace(module=named.module)
    processor = types.SimpleNamespace(tasks={named.name: {"capture": capture}})

    looper = object.__new__(ModuleLooper)
    looper._rehome_processor_task(processor, named, torch.device("cpu"))

    assert capture.module is quant_source


def test_quant_prepare_materializes_meta_source_when_capture_skips_forward():
    named = NamedModule(
        torch.nn.Linear(4, 4, bias=False, device="meta"),
        name="mlp.experts.0.gate_proj",
        full_name="model.layers.0.mlp.experts.0.gate_proj",
        layer_index=0,
    )
    dense_source = torch.nn.Linear(4, 4, bias=False)
    calls = []

    class DummyQModel:
        def shell_module_materialize(self, **kwargs):
            calls.append(kwargs)
            named.state["quant_source_module"] = dense_source
            return dense_source

    looper = object.__new__(ModuleLooper)
    looper.gptq_model = DummyQModel()
    looper._assign_quant_device_for_module = lambda *_, **__: torch.device("cpu")
    processor = types.SimpleNamespace(tasks={})

    target = looper._prepare_named_module_for_quantization(
        processor=processor,
        named_module=named,
        fallback_device=torch.device("cpu"),
    )

    assert target == torch.device("cpu")
    assert named.module is dense_source
    assert dense_source.weight.device.type == "cpu"
    assert calls[0]["role"] == "quant_source"
    assert calls[0]["named_module"] is named


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


def test_forward_executor_run_single_streams_outputs_through_model_writer():
    calls = []

    class Writer:
        def __init__(self):
            self.outputs = {}

        def put(self, index, value):
            self.outputs[index] = value

        def finalize(self):
            calls.append(("finalize", sorted(self.outputs)))
            return ("disk-sequence", self.outputs)

    writer = Writer()
    looper = _make_forward_executor_looper()

    def create_writer(**kwargs):
        calls.append(("create", kwargs))
        return writer

    looper.gptq_model.create_quantization_layer_output_writer = create_writer
    executor = ForwardExecutor(looper)
    result = _run_executor_single(
        executor,
        _DummyForwardProcessor(),
        apply_moe_config=False,
    )

    assert result == ("disk-sequence", writer.outputs)
    assert calls[0] == (
        "create",
        {
            "layer_index": 0,
            "expected_batches": 1,
            "progress_stage": None,
            "apply_moe_config": False,
        },
    )
    assert calls[1] == ("finalize", [0])
    assert torch.equal(writer.outputs[0][0], torch.zeros(1, 1, 1))


def test_forward_executor_run_single_skips_durable_output_batch():
    class Writer:
        committed_indices = frozenset({0})

        def put(self, _index, _value):
            raise AssertionError("durable replay batch was rewritten")

        def finalize(self):
            return "recovered"

    class NoForward(torch.nn.Module):
        def forward(self, *_args, **_kwargs):
            raise AssertionError("durable replay batch was executed")

    looper = _make_forward_executor_looper()
    looper.gptq_model.create_quantization_layer_output_writer = (
        lambda **_kwargs: Writer()
    )
    result = ForwardExecutor(looper).run_single(
        module=NoForward(),
        processor=_DummyForwardProcessor(),
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
        apply_moe_config=False,
    )
    assert result == "recovered"


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
    previous_result = None

    def fake_run_subset_stage(looper, **kwargs):
        nonlocal previous_result
        if previous_result is not None:
            assert previous_result() is None
        calls.append(kwargs["layer_index"])
        result = SubsetStageResult(
            processed_subset={},
            layer_inputs=kwargs["layer_inputs"],
            plan=kwargs["plan"],
        )
        previous_result = weakref.ref(result)
        return result

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

        def should_quantize_layer(self, *_args):
            return True

        def post_quantize(self, module):
            return module

        def lm_head_pre_quantize_generate_hook(self, value):
            return value

    class DummyLooper:
        def __init__(self):
            self.gptq_model = DummyGptqModel()
            self.processors = [DummyProcessor()]
            self.input_embeddings_name = None
            self.output_embeddings_name = None
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
    pb = DummyPB(range(2))
    processor.layer_count = 2
    processor.pb = pb

    layers = [
        torch.nn.Linear(in_features=64, out_features=64),
        torch.nn.Linear(in_features=64, out_features=64),
    ]
    layer_modules = [["foo"]]
    logger = DummyLogger()

    run_layer_stage(
        looper,
        layers=layers,
        layer_modules=layer_modules,
        planning_layer_modules=layer_modules,
        layer_names=["model.layers.0", "model.layers.1"],
        fallback=True,
        shared_kv_cache_dict={},
        pb=pb,
        layer_count=2,
        region_timer=None,
        finalize_progress_cls=FinalizeProgressInfo,
        embed_quant_mode=QuantizeEmbed.OUTPUT,
        embed_only=False,
        logger=logger,
    )

    assert calls == [0, 1]


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

        def should_quantize_layer(self, *_args):
            return True

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
        layer_names=["model.layers.0", "model.layers.1", "model.layers.2"],
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

        def should_quantize_layer(self, *_args):
            return True

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
        layer_names=["model.layers.0", "model.layers.1"],
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

        def should_quantize_layer(self, *_args):
            return True

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
            self.direct_state_layers = []
            self.prepared_native_modules = []
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

        def _prepare_layer_direct_state_for_forward(
            self, module, fallback_device, *, projection_modules=None
        ):
            self.direct_state_layers.append(
                (module, fallback_device, projection_modules)
            )
            for relative_name, candidate in list(projection_modules.items()):
                projection_modules[relative_name] = NamedModule(
                    candidate,
                    name=relative_name,
                    full_name=f"model.layers.0.{relative_name}",
                    layer_index=0,
                )
            return 0

        def _prepare_named_module_for_forward(
            self, *, named_module, fallback_device
        ):
            self.prepared_native_modules.append(
                (named_module, fallback_device)
            )
            return named_module.module

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
                full_name = f"{layers_prefix}.{name}"
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
        layer_names=["model.layers.0", "model.layers.1"],
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
    assert len(looper.direct_state_layers) == 1
    assert {
        named.full_name for named, _device in looper.prepared_native_modules
    } == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    }
    assert all(
        not isinstance(named.module, NamedModule)
        for named, _device in looper.prepared_native_modules
    )
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


def test_run_layer_stage_catches_up_packed_layer_with_original_forward_topology(
    monkeypatch,
):
    sentinel_inputs = [[torch.tensor([[[2.0]]])]]
    sentinel_outputs = [[torch.tensor([[[7.0]]])]]
    replay_calls = []
    subset_calls = []

    def fake_replay(*_args, **kwargs):
        replay_calls.append(kwargs)
        return sentinel_outputs

    monkeypatch.setattr(
        "gptqmodel.looper.stage_layer._replay_layer_outputs", fake_replay
    )
    monkeypatch.setattr(
        "gptqmodel.looper.stage_layer.run_subset_stage",
        lambda *args, **kwargs: subset_calls.append((args, kwargs)),
    )

    class DummyPB:
        def __iter__(self):
            return iter([0])

        def __len__(self):
            return 1

        def title(self, *_args, **_kwargs):
            return self

        def subtitle(self, *_args, **_kwargs):
            return self

        def draw(self):
            return self

        def close(self):
            return None

    class DummyLogger:
        def debug(self, *_args, **_kwargs):
            return None

        def info(self, *_args, **_kwargs):
            return None

        def isEnabledFor(self, *_args, **_kwargs):
            return False

    class DummyProcessor:
        def __init__(self):
            self.inputs_cache = types.SimpleNamespace(
                layer_inputs=sentinel_inputs,
                layer_input_kwargs=[{}],
                position_ids=[],
                attention_masks=[],
            )
            self.log_call_count = 0

        def collect_memory_info(self, _layer_index):
            return None

        def clear_cache_data(self):
            self.inputs_cache.layer_inputs = []

        def receive_layer_inputs(self, layer_inputs):
            self.inputs_cache.layer_inputs = layer_inputs

    class Boundary:
        def __init__(self, processor):
            self.processor = processor
            self.prepared = []
            self.finalized = []
            self.committed = []

        def is_catchup_layer(self, layer_index):
            return layer_index == 0

        def prepare_catchup_layer(self, **kwargs):
            self.prepared.append(kwargs)
            return self.processor

        def finalize_catchup_layer(self, **kwargs):
            self.finalized.append(kwargs)

        def commit_layer(self, **kwargs):
            self.committed.append(kwargs)

    class DummyGptqModel:
        def __init__(self, boundary):
            self.model = torch.nn.Module()
            self.model.config = types.SimpleNamespace(model_type="test")
            self.quantize_config = QuantizeConfig(bits=4, group_size=128)
            self.quantization_layer_boundary_checkpoint = boundary
            self.lm_head = None

        def should_quantize_layer(self, *_args):
            return True

        def pre_quantize(self, module):
            return module

        def post_quantize(self, module):
            raise AssertionError("packed catch-up used generic post_quantize")

    class DummyLooper:
        def __init__(self):
            self.processors = [DummyProcessor()]
            self.boundary = Boundary(self.processors[0])
            self.gptq_model = DummyGptqModel(self.boundary)
            self.events = []
            self.direct_state_calls = []
            self.native_prepare_calls = []

        def _check_loop_stop(self):
            return False

        def _emit_layer_complete(self, **kwargs):
            self.events.append(kwargs)

        def _prepare_layer_direct_state_for_forward(self, *args, **kwargs):
            self.direct_state_calls.append((args, kwargs))

        def _prepare_named_module_for_forward(self, **kwargs):
            self.native_prepare_calls.append(kwargs)
            return kwargs["named_module"].module

    looper = DummyLooper()
    layers = [torch.nn.Linear(1, 1, bias=False)]
    run_layer_stage(
        looper,
        layers=layers,
        layer_modules=[["weight"]],
        planning_layer_modules=[["weight"]],
        layer_names=["model.layers.0"],
        fallback=True,
        shared_kv_cache_dict={},
        pb=DummyPB(),
        layer_count=1,
        region_timer=None,
        finalize_progress_cls=FinalizeProgressInfo,
        logger=DummyLogger(),
    )

    assert len(looper.boundary.prepared) == 1
    assert len(replay_calls) == 1
    assert replay_calls[0]["force_serial"] is False
    assert not subset_calls
    assert len(looper.direct_state_calls) == 1
    assert len(looper.native_prepare_calls) == 1
    assert looper.processors[0].inputs_cache.layer_inputs is sentinel_outputs
    assert len(looper.boundary.finalized) == 1
    assert len(looper.boundary.committed) == 1
    assert looper.boundary.committed[0]["layer_index"] == 0
    assert [event["submodule_finalized"] for event in looper.events] == [False, True]


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
