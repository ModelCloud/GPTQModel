# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4349: pruned MoE experts must restore a dense source.

`auto_dtype_cast="auto_nf12"` keeps the calibration source packed in `nf12_*`
buffers and only the quant-source preparation step turns it back into a dense
module. Uncovered experts pruned by `prune_uncovered_modules` never reached that
step, so their packed buffers survived into `save()` and the writer's ownership
check aborted the export.
"""

import types

import pcre
import pytest
import torch

import gptqmodel.looper.stage_subset as stage_subset_module
from gptqmodel.looper.loop_processor import ForwardConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.stage_subset import CalibrationCoveragePolicy, SubsetPlan
from gptqmodel.models.writer import _validate_nf12_payload_ownership
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.auto_dtype_cast import SourceModuleSpec, restore_source_module


class _PackedSource(torch.nn.Module):
    """Stand-in for the AUTO_NF12 wrapper installed during a calibration forward."""

    def __init__(self, dense: torch.nn.Linear):
        super().__init__()
        self.dense = dense
        self.in_features = dense.in_features
        self.out_features = dense.out_features
        self.register_buffer("nf12_packed", torch.zeros(8, dtype=torch.int32))
        self.register_buffer("nf12_escapes", torch.zeros(2, dtype=torch.uint8))
        self.register_buffer("nf12_exp_base", torch.zeros(1, dtype=torch.uint8))

    def dequantize_weight(self, *, device, dtype):
        return self.dense.weight.detach().to(device=device, dtype=dtype)


class _ImmediateFuture:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class _ImmediatePool:
    def submit(self, _device, callback, *args):
        return _ImmediateFuture(callback(*args))


class _Logger:
    def error(self, *_args, **_kwargs):
        return None

    def debug(self, *_args, **_kwargs):
        return None

    def isEnabledFor(self, *_args, **_kwargs):
        return False


def _make_packed_model(names):
    """Build a tiny model tree with one packed AUTO_NF12 source per name."""

    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.experts = torch.nn.ModuleList([torch.nn.Module() for _ in names])

    named_modules = {}
    for name in names:
        _, _, leaf = name.rpartition(".")
        parent_path = f"mlp.{name.rpartition('.')[0]}"
        dense = torch.nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
        packed = _PackedSource(dense)
        model.get_submodule(parent_path).register_module(leaf, packed)
        named_module = NamedModule(
            packed,
            name=name,
            full_name=f"mlp.{name}",
            layer_index=0,
        )
        named_module.state["auto_dtype_cast_source_spec"] = SourceModuleSpec(
            kind="linear",
            dtype=torch.bfloat16,
            shape=(16, 64),
            bias=False,
        )
        named_module.state["auto_dtype_cast_was_applied"] = True
        named_modules[name] = named_module
    return model, named_modules


class _RestoringLooper:
    """Minimal looper whose quant-source step restores the dense module."""

    def __init__(self, model):
        self.model = model
        self.gptq_model = types.SimpleNamespace(
            quantize_config=QuantizeConfig(bits=4, group_size=128, fallback=None),
            quant_region_timer=None,
        )
        self.prepared = []

    def _prepare_named_module_for_quantization(self, *, processor, named_module, fallback_device):
        self.prepared.append(named_module.name)
        spec = named_module.state.pop("auto_dtype_cast_source_spec", None)
        if spec is not None:
            restored = restore_source_module(named_module.module, spec, device=fallback_device)
            parent_path, _, leaf = named_module.full_name.rpartition(".")
            self.model.get_submodule(parent_path).register_module(leaf, restored)
            named_module.module = restored
        return fallback_device


class _Processor:
    forward = ForwardConfig(required=False, replay_after_process=False)

    def name(self):
        return "fake"

    def __init__(self, names, uncovered):
        self.tasks = {name: object() for name in names}
        self.qcfg = QuantizeConfig(bits=4, group_size=128, fallback=None)
        self._uncovered = set(uncovered)

    def has_captured_input_ids(self, name):
        return name not in self._uncovered

    def process(self, **_kwargs):
        return None

    def set_fwd_time(self, _value):
        return None


def _plan(named_modules):
    names = list(named_modules)
    return SubsetPlan(
        modules=named_modules,
        subset_index=0,
        subset_total=1,
        execute_forward=False,
        replay_after_process=False,
        forward_mode="serial",
        batch_count=0,
        forward_row_counts=[],
        forward_total_rows=0,
        moe_groups={},
        forward_device_map={},
        calibration_coverage_policy=CalibrationCoveragePolicy(
            validate_input_coverage=True,
            fallback_enabled=False,
            prune_uncovered_modules=True,
            record_dynamic_exclusions=True,
        ),
        module_chunks=[named_modules],
        ordered_module_names=names,
    )


def _run_pass(monkeypatch, model, named_modules, uncovered):
    looper = _RestoringLooper(model)
    processor = _Processor(list(named_modules), uncovered)
    monkeypatch.setattr(stage_subset_module, "DEVICE_THREAD_POOL", _ImmediatePool())
    monkeypatch.setattr(stage_subset_module, "torch_sync", lambda: None)
    monkeypatch.setattr(
        stage_subset_module,
        "_emit_moe_parallel_quant_subset_telemetry",
        lambda **_kwargs: None,
    )
    processed, _outputs, _used_data_parallel = stage_subset_module._run_single_subset_pass(
        looper=looper,
        processor=processor,
        module=torch.nn.Identity(),
        plan=_plan(named_modules),
        layer_inputs=[],
        layer_input_kwargs=[],
        position_ids=[],
        attention_masks=[],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        layer_descriptor="model.layers.0",
        layer_title="Layer 0",
        layer_index=0,
        full=dict(named_modules),
        fallback=None,
        shared_kv_cache_dict={},
        pb=None,
        logger=_Logger(),
        is_awq_processor=False,
        execute_forward=False,
    )
    return looper, processor, processed


def _nf12_buffer_modules(model):
    return [
        name
        for name, module in model.named_modules()
        if any(
            buffer_name in getattr(module, "_buffers", {})
            for buffer_name in ("nf12_packed", "nf12_escapes", "nf12_exp_base")
        )
    ]


def test_pruned_uncovered_experts_restore_dense_source_before_save(monkeypatch):
    names = ["experts.0.gate_proj", "experts.1.gate_proj", "experts.2.gate_proj"]
    model, named_modules = _make_packed_model(names)
    uncovered = {"experts.1.gate_proj", "experts.2.gate_proj"}

    looper, processor, processed = _run_pass(monkeypatch, model, named_modules, uncovered)

    # Every packed source must pass through the quant-source step, including
    # the experts pruned for lacking calibration traffic.
    assert set(looper.prepared) == set(names)
    assert set(processed) == {"experts.0.gate_proj"}
    assert _nf12_buffer_modules(model) == []

    # Pruned experts keep the existing dynamic-exclusion semantics.
    assert set(processor.qcfg.dynamic) == {
        f"-:{pcre.escape('mlp.experts.1.gate_proj')}",
        f"-:{pcre.escape('mlp.experts.2.gate_proj')}",
    }

    # `save()` runs this check; it must not abort once the sources are dense.
    owner = types.SimpleNamespace(model=model, quantize_config=processor.qcfg)
    _validate_nf12_payload_ownership(owner)


def test_leftover_nf12_buffer_is_still_rejected_even_when_dynamically_excluded():
    names = ["experts.0.gate_proj"]
    model, named_modules = _make_packed_model(names)
    qcfg = QuantizeConfig(bits=4, group_size=128, fallback=None)
    qcfg.dynamic = {f"-:{pcre.escape('mlp.experts.0.gate_proj')}": {}}

    owner = types.SimpleNamespace(model=model, quantize_config=qcfg)

    # A dynamic exclusion only skips quantization; it does not excuse packed
    # scratch buffers in the checkpoint, so the ownership check must fail.
    with pytest.raises(ValueError, match="AUTO_NF12 source storage is still installed"):
        _validate_nf12_payload_ownership(owner)
    assert named_modules["experts.0.gate_proj"].module is not None
