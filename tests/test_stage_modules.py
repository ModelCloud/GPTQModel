import threading
import types
from typing import Dict

import torch

from gptqmodel.looper.module_looper import FinalizeProgressInfo, ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.loop_processor import ExecutionConfig
from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
from gptqmodel.looper.stage_layer import run_layer_stage
from gptqmodel.looper.stage_subset import CalibrationCoveragePolicy, SubsetPlan, SubsetStageResult
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.pause_resume import PauseResumeController


class _DummyQModel:
    def __init__(self):
        self.support_batch_quantize = False
        self.quantize_config = types.SimpleNamespace(device=None, vram_strategy="exclusive", moe_routing_bypass=lambda : False)
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

    def __init__(self):
        self.layer = _TinyLayer()
        self.model = _TinyModel(self.layer)
        self.quantize_config = types.SimpleNamespace(device=torch.device("cpu"))
        self._hook_started = False
        self._hook_finished = False

    def shell_module_materialize(self, target_submodule, device):
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
                fwd_after_process=False,
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
            self._vram_strategy = types.SimpleNamespace()
            self._layer_events = []
            self.pause_controller = PauseResumeController()

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
                fwd_after_process=False,
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
            self._vram_strategy = types.SimpleNamespace()
            self._layer_events = []
            self.named_module_layers = []
            self.pause_controller = PauseResumeController()

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
                fwd_after_process=True,
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
            self._vram_strategy = types.SimpleNamespace()
            self.pause_controller = PauseResumeController()
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
