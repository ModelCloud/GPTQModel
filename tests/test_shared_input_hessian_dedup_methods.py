# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Shared-input Hessian dedup for QQQ, VPTQ, MACaQ, EXL3, GPTAQ and FOEM.

Each quantizer keeps its own statistics layout, so every method proves two
things independently: the follower's capture is skipped (``add_batch`` count
drops to one per group), and the adopted statistics are bitwise equal to an
independent capture of the same batches.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Dict, List

import pytest
import torch
from torch import nn

from gptqmodel.looper.exllamav3_processor import EXL3Processor
from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig, LoopProcessor
from gptqmodel.looper.macaq_processor import MacaqProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY
from gptqmodel.looper.qqq_processor import QQQProcessor
from gptqmodel.looper.shared_input import (
    SharedInputCaptureMixin,
    SharedInputCaptureProcessor,
)
from gptqmodel.looper.vptq_processor import VPTQProcessor
from gptqmodel.models.shared_input import SharedInputPlan, build_shared_input_plan
from gptqmodel.quantization.config import (
    EXL3Config,
    FOEMConfig,
    GPTAQConfig,
    GPTQConfig,
    MacaqConfig,
    QQQConfig,
    QuantizeConfig,
    VPTQConfig,
)
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.macaq import Macaq
from gptqmodel.quantization.qqq import QQQ
from gptqmodel.quantization.vptq import VPTQ


_TREE = [
    "model",
    "layers",
    "#",
    {
        "self_attn": ("q:0:in=x", "k:0:in=x", "v:0:in=x", "o:1"),
        "mlp": ("g:0:in=x", "u:0:in=x", "d:1"),
    },
]
_LAYER_MODULES = [
    ["self_attn.q", "self_attn.k", "self_attn.v"],
    ["self_attn.o"],
    ["mlp.g", "mlp.u"],
    ["mlp.d"],
]
_QKV = ["self_attn.q", "self_attn.k", "self_attn.v"]


def _plan() -> SharedInputPlan:
    return build_shared_input_plan(_TREE, _LAYER_MODULES)


def _fake_model(plan: SharedInputPlan, qcfg):
    def shared_input_plan(model_config=None, quantize_config=None, is_awq_quantize: bool = False):
        del model_config, quantize_config, is_awq_quantize
        return plan

    return SimpleNamespace(
        shared_input_plan=shared_input_plan,
        model=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=1)),
        quantize_config=qcfg,
    )


def _named_linear(
    name: str,
    in_features: int = 64,
    out_features: int = 8,
    seed: int = 0,
) -> NamedModule:
    torch.manual_seed(seed)
    return NamedModule(
        nn.Linear(in_features, out_features, bias=False),
        name=name,
        full_name=f"model.layers.0.{name}",
        layer_index=0,
    )


def _batches(in_features: int = 64, n: int = 3, tokens: int = 5, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(1, tokens, in_features, generator=generator) for _ in range(n)]


def _feed(processor, modules: Dict[str, NamedModule], names: List[str], batches):
    hooks = {name: processor.pre_process_fwd_hook(name) for name in names}
    for batch_index, x in enumerate(batches):
        processor._set_current_batch_index(batch_index)
        for name in names:
            module = modules[name].module
            out = module(x)
            hooks[name](module, (x,), out)
    processor._set_current_batch_index(None)


def _preprocess_all(processor, names: List[str]) -> Dict[str, NamedModule]:
    modules = {}
    for index, name in enumerate(names):
        named = _named_linear(name, seed=index)
        processor.preprocess(named)
        modules[name] = named
    return modules


def _assert_group_capture_skipped(processor, names: List[str], batches):
    tasks = {name: processor.tasks[name] for name in names}
    assert tasks[names[0]].fwd_counter == len(batches)
    for name in names[1:]:
        assert tasks[name].fwd_counter == 0, name
    processor.end_shared_input_capture(names)
    for name in names[1:]:
        assert tasks[name].fwd_counter == len(batches), name
    assert processor.shared_input_dedup_count == len(names) - 1
    return tasks


# ---------------------------------------------------------------------------
# QQQ
# ---------------------------------------------------------------------------


def test_qqq_processor_adopts_one_hessian_per_group():
    qcfg = QQQConfig(bits=4, group_size=8, desc_act=False)
    processor = QQQProcessor(tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig())
    model = _fake_model(_plan(), qcfg)
    modules = _preprocess_all(processor, _QKV)
    batches = _batches()

    leaders = processor.begin_shared_input_capture(model, _QKV)
    assert leaders == {"self_attn.k": "self_attn.q", "self_attn.v": "self_attn.q"}
    _feed(processor, modules, _QKV, batches)

    tasks = _assert_group_capture_skipped(processor, _QKV, batches)

    reference = QQQ(module=_named_linear("ref", seed=1), qcfg=copy.deepcopy(qcfg))
    for x in batches:
        reference.add_batch(x, None)
    reference_H = reference.materialize_hessian()
    follower_H = tasks["self_attn.k"].materialize_hessian()
    assert torch.equal(follower_H, reference_H)
    assert follower_H.data_ptr() != tasks["self_attn.q"].materialize_hessian().data_ptr()

    reference.quantizer.configure(4, perchannel=True, sym=qcfg.sym, mse=False, norm=100, groupsize=8)
    reference_weight, *_ = reference.quantize()
    follower_weight, *_ = tasks["self_attn.k"].quantize()
    assert torch.equal(follower_weight, reference_weight)


# ---------------------------------------------------------------------------
# VPTQ
# ---------------------------------------------------------------------------


def _vptq_config() -> VPTQConfig:
    return VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 4],
        num_res_centroids=[-1, 2],
        kmeans_iters=8,
        block_size=64,
    )


def test_vptq_processor_adopts_one_hessian_per_group():
    qcfg = _vptq_config()
    processor = VPTQProcessor(tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig())
    model = _fake_model(_plan(), qcfg)
    modules = _preprocess_all(processor, _QKV)
    batches = _batches()

    leaders = processor.begin_shared_input_capture(model, _QKV)
    assert leaders == {"self_attn.k": "self_attn.q", "self_attn.v": "self_attn.q"}
    _feed(processor, modules, _QKV, batches)

    tasks = _assert_group_capture_skipped(processor, _QKV, batches)

    reference = VPTQ(module=_named_linear("ref", seed=1), qcfg=copy.deepcopy(qcfg))
    for x in batches:
        reference.add_batch(x, None)
    reference_H = reference.materialize_hessian()
    follower_H = tasks["self_attn.k"].materialize_hessian()
    assert torch.equal(follower_H, reference_H)

    reference_result, _, _ = reference.quantize()
    follower_result, _, _ = tasks["self_attn.k"].quantize()
    assert torch.equal(follower_result.quantized_weight, reference_result.quantized_weight)


# ---------------------------------------------------------------------------
# MACaQ
# ---------------------------------------------------------------------------


def _macaq_config() -> MacaqConfig:
    return MacaqConfig(
        rotation="none",
        pre_mode="none",
        n_flip_iters=4,
        fast_flip=False,
        compile_flip=False,
        exp_screen=True,
    )


def test_macaq_processor_adopts_one_hessian_per_group():
    qcfg = _macaq_config()
    processor = MacaqProcessor(tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig())
    model = _fake_model(_plan(), qcfg)
    modules = _preprocess_all(processor, _QKV)
    batches = _batches()

    leaders = processor.begin_shared_input_capture(model, _QKV)
    assert leaders == {"self_attn.k": "self_attn.q", "self_attn.v": "self_attn.q"}
    _feed(processor, modules, _QKV, batches)

    tasks = _assert_group_capture_skipped(processor, _QKV, batches)

    reference = Macaq(module=_named_linear("ref", seed=1), qcfg=copy.deepcopy(qcfg))
    for x in batches:
        reference.add_batch(x, None)
    reference_H = reference.materialize_hessian()
    follower_H = tasks["self_attn.k"].materialize_hessian()
    assert torch.equal(follower_H, reference_H)

    reference_result, _, _ = reference.quantize()
    follower_result, _, _ = tasks["self_attn.k"].quantize()
    assert torch.equal(follower_result.W_packed, reference_result.W_packed)
    assert torch.equal(follower_result.W_scales, reference_result.W_scales)


# ---------------------------------------------------------------------------
# EXL3 (capture tasks are plain GPTQ)
# ---------------------------------------------------------------------------


def _exl3_capture(module: NamedModule, qcfg: GPTQConfig) -> GPTQ:
    capture = GPTQ(module=module, qcfg=qcfg)
    capture.expected_nsamples = None
    capture.quantizer.configure(perchannel=True)
    return capture


def test_exl3_processor_adopts_one_hessian_per_group():
    qcfg = EXL3Config()
    processor = EXL3Processor(tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig())
    model = _fake_model(_plan(), qcfg)
    capture_cfg = GPTQConfig(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        offload_to_disk=False,
        device="cpu",
    )

    modules = {}
    for index, name in enumerate(_QKV):
        named = _named_linear(name, seed=index)
        processor.tasks[name] = {
            "capture": _exl3_capture(named, capture_cfg),
            "qcfg": qcfg,
        }
        modules[name] = named
    batches = _batches()

    leaders = processor.begin_shared_input_capture(model, _QKV)
    assert leaders == {"self_attn.k": "self_attn.q", "self_attn.v": "self_attn.q"}
    _feed(processor, modules, _QKV, batches)

    captures = {name: processor.tasks[name]["capture"] for name in _QKV}
    assert captures["self_attn.q"].fwd_counter == len(batches)
    assert captures["self_attn.k"].fwd_counter == 0
    assert captures["self_attn.v"].fwd_counter == 0

    processor.end_shared_input_capture(_QKV)
    assert captures["self_attn.k"].fwd_counter == len(batches)
    assert captures["self_attn.v"].fwd_counter == len(batches)
    assert processor.shared_input_dedup_count == 2

    reference = GPTQ(module=_named_linear("ref", seed=1), qcfg=copy.deepcopy(capture_cfg))
    reference.quantizer.configure(perchannel=True)
    for batch_index, x in enumerate(batches):
        reference.add_batch(x, None, batch_index=batch_index)
    reference.materialize_global_hessian()
    follower = captures["self_attn.k"]
    follower.materialize_global_hessian()
    assert torch.equal(follower.H, reference.H)
    assert follower.H.data_ptr() != captures["self_attn.q"].H.data_ptr()


# ---------------------------------------------------------------------------
# GPTAQ / FOEM
# ---------------------------------------------------------------------------


def _gptq_processor_qcfg(kind: str) -> QuantizeConfig:
    if kind == "gptaq":
        return QuantizeConfig(bits=4, group_size=8, gptaq=GPTAQConfig())
    return QuantizeConfig(bits=4, group_size=8, foem=FOEMConfig(alpha=0.25))


def _gptaq_foem_tasks(kind: str, batches, seed_offset: int = 0):
    qcfg = _gptq_processor_qcfg(kind)
    processor = GPTQProcessor(tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig())
    model = _fake_model(_plan(), qcfg)
    modules = {}
    for index, name in enumerate(_QKV):
        named = _named_linear(name, in_features=16, seed=seed_offset + index)
        named.state[NATIVE_INPUTS_STATE_KEY] = [
            (batch_index, x) for batch_index, x in enumerate(batches)
        ]
        processor.preprocess(named)
        modules[name] = named
    return qcfg, processor, model, modules


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_gptaq_foem_processor_adopts_h_and_dxxt(kind: str):
    batches = _batches(in_features=16, n=3, tokens=4)
    qcfg, processor, model, modules = _gptaq_foem_tasks(kind, batches)

    leaders = processor.begin_shared_input_capture(model, _QKV)
    assert leaders == {"self_attn.k": "self_attn.q", "self_attn.v": "self_attn.q"}
    _feed(processor, modules, _QKV, batches)

    tasks = {name: processor.tasks[name] for name in _QKV}
    assert tasks["self_attn.q"].fwd_counter == len(batches)
    assert tasks["self_attn.k"].fwd_counter == 0
    assert tasks["self_attn.v"].fwd_counter == 0

    processor.end_shared_input_capture(_QKV)
    follower = tasks["self_attn.k"]
    leader = tasks["self_attn.q"]
    assert follower.fwd_counter == len(batches)
    assert processor.shared_input_dedup_count == 2

    reference_named = _named_linear("ref", in_features=16, seed=1)
    reference_named.state[NATIVE_INPUTS_STATE_KEY] = [
        (batch_index, x) for batch_index, x in enumerate(batches)
    ]
    reference = (
        GPTAQ(reference_named, copy.deepcopy(qcfg))
        if kind == "gptaq"
        else FOEM(reference_named, copy.deepcopy(qcfg))
    )
    reference.quantizer.configure(perchannel=True)
    for batch_index, x in enumerate(batches):
        reference.add_batch(x, None, batch_index=batch_index)

    if kind == "gptaq":
        leader._finalize_gptaq_hessian()
    else:
        leader._finalize_foem_hessian()

    assert torch.equal(follower.H, reference.H)
    assert follower.H.data_ptr() != leader.H.data_ptr()
    assert follower.dXXT is not None
    assert torch.equal(follower.dXXT, reference.dXXT)
    assert follower.dXXT.data_ptr() != leader.dXXT.data_ptr()
    assert follower.nsamples == reference.nsamples == leader.nsamples
    assert follower.native_inps == []
    if kind == "gptaq":
        assert follower._gptaq_batch_request_count == reference._gptaq_batch_request_count


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_gptaq_foem_quantize_after_adopt_matches_independent(kind: str):
    batches = _batches(in_features=16, n=4, tokens=8)
    qcfg, processor, model, modules = _gptaq_foem_tasks(kind, batches)

    processor.begin_shared_input_capture(model, _QKV)
    _feed(processor, modules, _QKV, batches)
    processor.end_shared_input_capture(_QKV)

    reference_named = _named_linear("ref", in_features=16, seed=1)
    reference_named.state[NATIVE_INPUTS_STATE_KEY] = [
        (batch_index, x) for batch_index, x in enumerate(batches)
    ]
    reference = (
        GPTAQ(reference_named, copy.deepcopy(qcfg))
        if kind == "gptaq"
        else FOEM(reference_named, copy.deepcopy(qcfg))
    )
    reference.quantizer.configure(perchannel=True)
    for batch_index, x in enumerate(batches):
        reference.add_batch(x, None, batch_index=batch_index)

    follower = processor.tasks["self_attn.k"]
    reference_weight, *_ = reference.quantize()
    follower_weight, *_ = follower.quantize()
    assert torch.equal(follower_weight, reference_weight)


# ---------------------------------------------------------------------------
# gate/up groups are declared by the same plan
# ---------------------------------------------------------------------------


_GATE_UP = ["mlp.g", "mlp.u"]


def _capture_pair(processor, model, names, batches, seed_offset):
    modules = {}
    for index, name in enumerate(names):
        named = _named_linear(name, seed=seed_offset + index)
        processor.preprocess(named)
        modules[name] = named
    leaders = processor.begin_shared_input_capture(model, names)
    assert leaders == {names[1]: names[0]}
    _feed(processor, modules, names, batches)
    tasks = {name: processor.tasks[name] for name in names}
    assert tasks[names[0]].fwd_counter == len(batches)
    assert tasks[names[1]].fwd_counter == 0
    processor.end_shared_input_capture(names)
    assert tasks[names[1]].fwd_counter == len(batches)
    return tasks


@pytest.mark.parametrize("method", ["qqq", "vptq", "macaq"])
def test_gate_up_group_dedups(method: str):
    batches = _batches()
    if method == "qqq":
        qcfg = QQQConfig(bits=4, group_size=8, desc_act=False)
        processor = QQQProcessor(tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig())
        task_cls = QQQ
    elif method == "vptq":
        qcfg = _vptq_config()
        processor = VPTQProcessor(tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig())
        task_cls = VPTQ
    else:
        qcfg = _macaq_config()
        processor = MacaqProcessor(tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig())
        task_cls = Macaq
    model = _fake_model(_plan(), qcfg)

    tasks = _capture_pair(processor, model, _GATE_UP, batches, seed_offset=3)

    reference = task_cls(module=_named_linear("ref", seed=4), qcfg=copy.deepcopy(qcfg))
    for x in batches:
        reference.add_batch(x, None)
    assert torch.equal(
        tasks["mlp.u"].materialize_hessian(),
        reference.materialize_hessian(),
    )


def test_exl3_gate_up_group_dedups():
    qcfg = EXL3Config()
    processor = EXL3Processor(tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig())
    model = _fake_model(_plan(), qcfg)
    capture_cfg = GPTQConfig(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        offload_to_disk=False,
        device="cpu",
    )
    batches = _batches()
    modules = {}
    for index, name in enumerate(_GATE_UP):
        named = _named_linear(name, seed=3 + index)
        processor.tasks[name] = {
            "capture": _exl3_capture(named, capture_cfg),
            "qcfg": qcfg,
        }
        modules[name] = named

    leaders = processor.begin_shared_input_capture(model, _GATE_UP)
    assert leaders == {"mlp.u": "mlp.g"}
    _feed(processor, modules, _GATE_UP, batches)
    assert processor.tasks["mlp.g"]["capture"].fwd_counter == len(batches)
    assert processor.tasks["mlp.u"]["capture"].fwd_counter == 0
    processor.end_shared_input_capture(_GATE_UP)

    reference = GPTQ(module=_named_linear("ref", seed=4), qcfg=copy.deepcopy(capture_cfg))
    reference.quantizer.configure(perchannel=True)
    for batch_index, x in enumerate(batches):
        reference.add_batch(x, None, batch_index=batch_index)
    reference.materialize_global_hessian()
    follower = processor.tasks["mlp.u"]["capture"]
    follower.materialize_global_hessian()
    assert torch.equal(follower.H, reference.H)


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_gptaq_foem_gate_up_group_dedups(kind: str):
    batches = _batches(in_features=16)
    qcfg, processor, model, modules = _gptaq_foem_tasks(kind, batches, seed_offset=3)
    # `_gptaq_foem_tasks` seeds only the q/k/v names; rebuild the gate/up tasks.
    processor.tasks.clear()
    modules = {}
    for index, name in enumerate(_GATE_UP):
        named = _named_linear(name, in_features=16, seed=3 + index)
        named.state[NATIVE_INPUTS_STATE_KEY] = [
            (batch_index, x) for batch_index, x in enumerate(batches)
        ]
        processor.preprocess(named)
        modules[name] = named

    leaders = processor.begin_shared_input_capture(model, _GATE_UP)
    assert leaders == {"mlp.u": "mlp.g"}
    _feed(processor, modules, _GATE_UP, batches)
    assert processor.tasks["mlp.g"].fwd_counter == len(batches)
    assert processor.tasks["mlp.u"].fwd_counter == 0
    processor.end_shared_input_capture(_GATE_UP)
    follower = processor.tasks["mlp.u"]
    assert follower.fwd_counter == len(batches)

    reference_named = _named_linear("ref", in_features=16, seed=4)
    reference_named.state[NATIVE_INPUTS_STATE_KEY] = [
        (batch_index, x) for batch_index, x in enumerate(batches)
    ]
    reference = (
        GPTAQ(reference_named, copy.deepcopy(qcfg))
        if kind == "gptaq"
        else FOEM(reference_named, copy.deepcopy(qcfg))
    )
    reference.quantizer.configure(perchannel=True)
    for batch_index, x in enumerate(batches):
        reference.add_batch(x, None, batch_index=batch_index)
    if kind == "gptaq":
        processor.tasks["mlp.g"]._finalize_gptaq_hessian()
    else:
        processor.tasks["mlp.g"]._finalize_foem_hessian()
    assert torch.equal(follower.H, reference.H)
    assert torch.equal(follower.dXXT, reference.dXXT)


# ---------------------------------------------------------------------------
# Base-class resolution for the shared-input capture lifecycle
# ---------------------------------------------------------------------------


_PROCESSOR_CONFIGS = [
    (QQQProcessor, QQQConfig(bits=4, group_size=8, desc_act=False)),
    (VPTQProcessor, _vptq_config()),
    (MacaqProcessor, _macaq_config()),
    (EXL3Processor, EXL3Config()),
]


@pytest.mark.parametrize(("processor_cls", "qcfg"), _PROCESSOR_CONFIGS)
def test_shared_input_capture_methods_resolve_to_single_shared_base(processor_cls, qcfg):
    # The four calibrated processors inherit one intermediate base, so the
    # mixin/loop-processor conflict is resolved in exactly one place.
    assert issubclass(processor_cls, SharedInputCaptureProcessor)
    assert processor_cls.begin_shared_input_capture is SharedInputCaptureProcessor.begin_shared_input_capture
    assert processor_cls.end_shared_input_capture is SharedInputCaptureProcessor.end_shared_input_capture
    assert processor_cls.begin_shared_input_capture is not LoopProcessor.begin_shared_input_capture
    assert processor_cls.end_shared_input_capture is not LoopProcessor.end_shared_input_capture

    processor = processor_cls(tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig())
    model = _fake_model(_plan(), qcfg)
    if processor_cls is EXL3Processor:
        # EXL3's capture tasks are plain GPTQ accumulators and preprocess()
        # rejects the 64-wide fixture geometry (it requires 128-aligned input).
        capture_cfg = GPTQConfig(
            bits=4,
            group_size=-1,
            desc_act=False,
            sym=True,
            offload_to_disk=False,
            device="cpu",
        )
        modules = {}
        for index, name in enumerate(_QKV):
            named = _named_linear(name, seed=index)
            processor.tasks[name] = {"capture": _exl3_capture(named, capture_cfg), "qcfg": qcfg}
            modules[name] = named
    else:
        modules = _preprocess_all(processor, _QKV)
    batches = _batches()

    leaders = processor.begin_shared_input_capture(model, _QKV)
    assert leaders == {"self_attn.k": "self_attn.q", "self_attn.v": "self_attn.q"}
    _feed(processor, modules, _QKV, batches)

    telemetry = processor.end_shared_input_capture(_QKV)
    assert telemetry["enabled"] is True
    assert telemetry["expected_followers"] == 2
    assert telemetry["adopted_followers"] == 2
    assert telemetry["status"] == "verified"


def test_shared_input_capture_wrapper_delegates_to_mixin(monkeypatch):
    calls = []
    original_begin = SharedInputCaptureMixin.begin_shared_input_capture
    original_end = SharedInputCaptureMixin.end_shared_input_capture

    def spy_begin(self, *args, **kwargs):
        calls.append("begin")
        return original_begin(self, *args, **kwargs)

    def spy_end(self, *args, **kwargs):
        calls.append("end")
        return original_end(self, *args, **kwargs)

    monkeypatch.setattr(SharedInputCaptureMixin, "begin_shared_input_capture", spy_begin)
    monkeypatch.setattr(SharedInputCaptureMixin, "end_shared_input_capture", spy_end)

    qcfg = QQQConfig(bits=4, group_size=8, desc_act=False)
    processor = QQQProcessor(tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig())
    model = _fake_model(_plan(), qcfg)
    modules = _preprocess_all(processor, _QKV)

    assert processor.begin_shared_input_capture(model, _QKV) == {
        "self_attn.k": "self_attn.q",
        "self_attn.v": "self_attn.q",
    }
    _feed(processor, modules, _QKV, _batches())
    telemetry = processor.end_shared_input_capture(_QKV)
    assert calls == ["begin", "end"]
    assert telemetry["adopted_followers"] == 2


def test_loop_processor_without_mixin_keeps_noop_defaults():
    proc = LoopProcessor.__new__(LoopProcessor)
    assert type(proc).begin_shared_input_capture is LoopProcessor.begin_shared_input_capture
    assert type(proc).end_shared_input_capture is LoopProcessor.end_shared_input_capture
    assert proc.begin_shared_input_capture(object(), ["a", "b"]) == {}
    assert proc.end_shared_input_capture(["a", "b"]) is None
