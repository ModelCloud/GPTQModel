# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Looper-side Hessian dedup for explicit `:in=<tag>` shared-input groups (CPU only)."""

from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest
import torch
from torch import nn

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.loop_processor import LoopProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.shared_input import SharedInputPlan, build_shared_input_plan
from gptqmodel.quantization import GPTQ
from gptqmodel.quantization.config import METHOD, HessianConfig, QuantizeConfig


_IN = 8
_OUT = 6
_QKV_TREE = ["model", "layers", "#", {"self_attn": ("q:0:in=x", "k:0:in=x", "v:0:in=x", "o:1"), "mlp": ("g:0:in=x", "u:0:in=x", "d:1")}]
_LAYER_MODULES = [["self_attn.q", "self_attn.k", "self_attn.v"], ["self_attn.o"], ["mlp.g", "mlp.u"], ["mlp.d"]]


def _named_linear(name: str, in_features: int = _IN, out_features: int = _OUT, seed: int = 0) -> NamedModule:
    torch.manual_seed(seed)
    return NamedModule(
        nn.Linear(in_features, out_features, bias=False),
        name=name,
        full_name=f"model.layers.0.{name}",
        layer_index=0,
    )


def _processor(qcfg: Optional[QuantizeConfig] = None) -> GPTQProcessor:
    return GPTQProcessor(
        tokenizer=None,
        qcfg=qcfg or QuantizeConfig(method=METHOD.GPTQ, bits=4, group_size=4, desc_act=False),
        calibration=None,
        prepare_dataset_func=None,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )


def _fake_model(plan: SharedInputPlan, qcfg: QuantizeConfig):
    calls: List[tuple] = []

    def shared_input_plan(model_config=None, quantize_config=None, is_awq_quantize: bool = False):
        calls.append((model_config, quantize_config))
        return plan

    model = SimpleNamespace(
        shared_input_plan=shared_input_plan,
        model=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=1)),
        quantize_config=qcfg,
        _calls=calls,
    )
    return model


def _batches(n: int = 3, tokens: int = 5, seed: int = 1) -> List[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, tokens, _IN, generator=g) for _ in range(n)]


def _expected_H(batches: List[torch.Tensor]) -> torch.Tensor:
    x = torch.cat([b.reshape(-1, _IN) for b in batches], dim=0).to(torch.float32)
    return 2.0 * (x.T @ x) / x.shape[0]


def _setup(processor: GPTQProcessor, names: List[str]) -> Dict[str, NamedModule]:
    modules = {}
    for i, name in enumerate(names):
        nm = _named_linear(name, seed=i)
        processor.preprocess(nm)
        modules[name] = nm
    return modules


def _feed(processor: GPTQProcessor, modules: Dict[str, NamedModule], names: List[str], batches: List[torch.Tensor]):
    hooks = {name: processor.pre_process_fwd_hook(name) for name in names}
    for idx, x in enumerate(batches):
        processor._set_current_batch_index(idx)
        for name in names:
            out = modules[name].module(x)
            hooks[name](modules[name].module, (x,), out)
    processor._set_current_batch_index(None)


# ---------------------------------------------------------------------------
# GPTQ.adopt_hessian_from
# ---------------------------------------------------------------------------


class TestAdoptHessian:
    def test_follower_gets_equal_but_independent_hessian(self):
        qcfg = QuantizeConfig(method=METHOD.GPTQ)
        leader = GPTQ(module=_named_linear("q"), qcfg=qcfg)
        follower = GPTQ(module=_named_linear("k", seed=1), qcfg=qcfg)
        batches = _batches()
        for i, x in enumerate(batches):
            leader.add_batch(x, None, batch_index=i)

        assert leader._hessian_dirty and leader.H is None
        follower.adopt_hessian_from(leader)

        expected = _expected_H(batches)
        assert torch.allclose(follower.H, expected, atol=1e-5)
        assert torch.allclose(leader.H, expected, atol=1e-5)
        assert follower.nsamples == leader.nsamples == sum(b.shape[1] for b in batches)
        assert follower.fwd_counter == leader.fwd_counter == len(batches)
        assert not follower._hessian_dirty and not follower._device_hessian_partials
        assert follower.H.data_ptr() != leader.H.data_ptr()

        follower.H.fill_(0)
        assert torch.allclose(leader.H, expected, atol=1e-5)

    def test_adopt_is_idempotent_for_materialized_leader(self):
        qcfg = QuantizeConfig(method=METHOD.GPTQ)
        leader = GPTQ(module=_named_linear("q"), qcfg=qcfg)
        a = GPTQ(module=_named_linear("k"), qcfg=qcfg)
        b = GPTQ(module=_named_linear("v"), qcfg=qcfg)
        batches = _batches()
        for i, x in enumerate(batches):
            leader.add_batch(x, None, batch_index=i)
        a.adopt_hessian_from(leader)
        b.adopt_hessian_from(leader)
        assert torch.equal(a.H, b.H)
        assert a.nsamples == b.nsamples == leader.nsamples

    def test_adopt_from_uncalled_leader_yields_zero_hessian(self):
        qcfg = QuantizeConfig(method=METHOD.GPTQ)
        leader = GPTQ(module=_named_linear("q"), qcfg=qcfg)
        follower = GPTQ(module=_named_linear("k"), qcfg=qcfg)
        follower.adopt_hessian_from(leader)
        assert follower.H.shape == (_IN, _IN)
        assert torch.count_nonzero(follower.H) == 0
        assert follower.nsamples == 0 and follower.fwd_counter == 0

    def test_adopt_self_is_noop(self):
        qcfg = QuantizeConfig(method=METHOD.GPTQ)
        task = GPTQ(module=_named_linear("q"), qcfg=qcfg)
        task.add_batch(_batches(1)[0], None, batch_index=0)
        task.adopt_hessian_from(task)
        assert task._hessian_dirty and task.H is None

    def test_column_mismatch_raises(self):
        qcfg = QuantizeConfig(method=METHOD.GPTQ)
        leader = GPTQ(module=_named_linear("q"), qcfg=qcfg)
        follower = GPTQ(module=_named_linear("k", in_features=_IN * 2), qcfg=qcfg)
        with pytest.raises(ValueError, match="cannot share Hessian"):
            follower.adopt_hessian_from(leader)

    def test_quantize_after_adopt_matches_independent_collection(self):
        qcfg = QuantizeConfig(method=METHOD.GPTQ, bits=4, group_size=4, desc_act=False)
        batches = _batches(n=4, tokens=16)

        independent = GPTQ(module=_named_linear("k", seed=7), qcfg=qcfg)
        independent.quantizer.configure(perchannel=True)
        for i, x in enumerate(batches):
            independent.add_batch(x, None, batch_index=i)
        wq_independent, *_ = independent.quantize()

        leader = GPTQ(module=_named_linear("q", seed=3), qcfg=qcfg)
        leader.quantizer.configure(perchannel=True)
        for i, x in enumerate(batches):
            leader.add_batch(x, None, batch_index=i)
        follower = GPTQ(module=_named_linear("k", seed=7), qcfg=qcfg)
        follower.quantizer.configure(perchannel=True)
        follower.adopt_hessian_from(leader)
        wq_follower, *_ = follower.quantize()
        wq_leader, *_ = leader.quantize()

        assert torch.equal(wq_follower, wq_independent)
        assert wq_leader.shape == wq_follower.shape


# ---------------------------------------------------------------------------
# GPTQProcessor begin/end_shared_input_capture
# ---------------------------------------------------------------------------


class TestProcessorLeaderElection:
    def test_loop_processor_default_is_noop(self):
        proc = LoopProcessor.__new__(LoopProcessor)
        assert proc.begin_shared_input_capture(object(), ["a", "b"]) == {}
        assert proc.end_shared_input_capture(["a", "b"]) is None

    def test_elects_first_member_per_explicit_group(self):
        proc = _processor()
        plan = build_shared_input_plan(_QKV_TREE, _LAYER_MODULES)
        model = _fake_model(plan, proc.qcfg)
        names = ["self_attn.q", "self_attn.k", "self_attn.v"]
        _setup(proc, names)

        leaders = proc.begin_shared_input_capture(model, names)
        assert leaders == {"self_attn.k": "self_attn.q", "self_attn.v": "self_attn.q"}
        assert proc.shared_input_leader("self_attn.k") == "self_attn.q"
        assert proc.shared_input_leader("self_attn.q") is None

    def test_plan_is_derived_once_per_model_class(self):
        proc = _processor()
        plan = build_shared_input_plan(_QKV_TREE, _LAYER_MODULES)
        model = _fake_model(plan, proc.qcfg)
        names = ["self_attn.q", "self_attn.k", "self_attn.v"]
        _setup(proc, names)
        proc.begin_shared_input_capture(model, names)
        proc.begin_shared_input_capture(model, names)
        assert len(model._calls) == 1
        assert model._calls[0] == (model.model.config, proc.qcfg)

    def test_subset_order_decides_leader(self):
        proc = _processor()
        plan = build_shared_input_plan(_QKV_TREE, _LAYER_MODULES)
        model = _fake_model(plan, proc.qcfg)
        names = ["self_attn.v", "self_attn.q", "self_attn.k"]
        _setup(proc, names)
        assert proc.begin_shared_input_capture(model, names) == {
            "self_attn.q": "self_attn.v",
            "self_attn.k": "self_attn.v",
        }

    def test_singletons_and_untagged_never_dedup(self):
        proc = _processor()
        tree = ["model", "layers", "#", {"self_attn": ("q:0", "k:0", "v:0", "o:1")}]
        plan = build_shared_input_plan(tree, [["self_attn.q", "self_attn.k", "self_attn.v"], ["self_attn.o"]])
        model = _fake_model(plan, proc.qcfg)
        names = ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o"]
        _setup(proc, names)
        assert proc.begin_shared_input_capture(model, names) == {}

    def test_only_members_present_in_subset_dedup(self):
        proc = _processor()
        plan = build_shared_input_plan(_QKV_TREE, _LAYER_MODULES)
        model = _fake_model(plan, proc.qcfg)
        _setup(proc, ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o"])

        assert proc.begin_shared_input_capture(model, ["self_attn.q", "self_attn.o"]) == {}
        assert proc.begin_shared_input_capture(model, ["self_attn.k", "self_attn.v"]) == {"self_attn.v": "self_attn.k"}
        assert proc.begin_shared_input_capture(model, ["self_attn.q", "self_attn.k", "self_attn.v"]) == {
            "self_attn.k": "self_attn.q",
            "self_attn.v": "self_attn.q",
        }

    def test_members_without_task_are_ignored(self):
        proc = _processor()
        plan = build_shared_input_plan(_QKV_TREE, _LAYER_MODULES)
        model = _fake_model(plan, proc.qcfg)
        _setup(proc, ["self_attn.q", "self_attn.k"])
        names = ["self_attn.q", "self_attn.k", "self_attn.v"]
        assert proc.begin_shared_input_capture(model, names) == {"self_attn.k": "self_attn.q"}

    def test_disabled_by_hessian_config(self):
        proc = _processor(QuantizeConfig(method=METHOD.GPTQ, hessian=HessianConfig(dedup_shared_inputs=False)))
        plan = build_shared_input_plan(_QKV_TREE, _LAYER_MODULES)
        model = _fake_model(plan, proc.qcfg)
        names = ["self_attn.q", "self_attn.k", "self_attn.v"]
        _setup(proc, names)
        assert proc.begin_shared_input_capture(model, names) == {}
        assert model._calls == []

    def test_disabled_per_module_by_dynamic_override(self):
        qcfg = QuantizeConfig(
            method=METHOD.GPTQ,
            dynamic={r".*\.self_attn\.v$": {"hessian": {"dedup_shared_inputs": False}}},
        )
        proc = _processor(qcfg)
        plan = build_shared_input_plan(_QKV_TREE, _LAYER_MODULES)
        model = _fake_model(plan, proc.qcfg)
        names = ["self_attn.q", "self_attn.k", "self_attn.v"]
        _setup(proc, names)
        assert proc.begin_shared_input_capture(model, names) == {"self_attn.k": "self_attn.q"}

    def test_lm_head_never_dedups(self):
        proc = _processor()
        plan = build_shared_input_plan(_QKV_TREE, _LAYER_MODULES)
        model = _fake_model(plan, proc.qcfg)
        names = ["self_attn.q", "self_attn.k", "self_attn.v"]
        _setup(proc, names)
        assert proc.begin_shared_input_capture(model, names, is_lm_head_module=True) == {}

    def test_model_without_plan_api(self):
        proc = _processor()
        names = ["self_attn.q", "self_attn.k"]
        _setup(proc, names)
        assert proc.begin_shared_input_capture(SimpleNamespace(), names) == {}

    def test_non_plain_gptq_tasks_excluded(self):
        class NotPlainGPTQ(GPTQ):
            pass

        proc = _processor()
        plan = build_shared_input_plan(_QKV_TREE, _LAYER_MODULES)
        model = _fake_model(plan, proc.qcfg)
        names = ["self_attn.q", "self_attn.k", "self_attn.v"]
        modules = _setup(proc, names)
        proc.tasks["self_attn.k"] = NotPlainGPTQ(module=modules["self_attn.k"], qcfg=proc.qcfg)
        assert proc.begin_shared_input_capture(model, names) == {"self_attn.v": "self_attn.q"}

    def test_column_mismatch_within_group_is_not_shared(self):
        proc = _processor()
        plan = build_shared_input_plan(_QKV_TREE, _LAYER_MODULES)
        model = _fake_model(plan, proc.qcfg)
        proc.preprocess(_named_linear("self_attn.q"))
        proc.preprocess(_named_linear("self_attn.k", in_features=_IN * 2))
        proc.preprocess(_named_linear("self_attn.v"))
        names = ["self_attn.q", "self_attn.k", "self_attn.v"]
        assert proc.begin_shared_input_capture(model, names) == {"self_attn.v": "self_attn.q"}

    def test_begin_resets_previous_election(self):
        proc = _processor()
        plan = build_shared_input_plan(_QKV_TREE, _LAYER_MODULES)
        model = _fake_model(plan, proc.qcfg)
        names = ["self_attn.q", "self_attn.k", "self_attn.v"]
        _setup(proc, names)
        proc.begin_shared_input_capture(model, names)
        assert proc.begin_shared_input_capture(model, ["self_attn.o"]) == {}
        assert proc.shared_input_leader("self_attn.k") is None


class TestProcessorCaptureAndAdopt:
    def test_followers_skip_capture_then_adopt_leader_hessian(self):
        proc = _processor()
        plan = build_shared_input_plan(_QKV_TREE, _LAYER_MODULES)
        model = _fake_model(plan, proc.qcfg)
        names = ["self_attn.q", "self_attn.k", "self_attn.v"]
        modules = _setup(proc, names)
        batches = _batches()

        proc.begin_shared_input_capture(model, names)
        _feed(proc, modules, names, batches)

        leader, k, v = (proc.tasks[n] for n in names)
        assert leader.fwd_counter == len(batches)
        assert k.fwd_counter == 0 and v.fwd_counter == 0
        assert not k._device_hessian_partials and not v._device_hessian_partials
        assert not proc.has_captured_input_ids("self_attn.k")

        proc.end_shared_input_capture(names)

        expected = _expected_H(batches)
        for task in (leader, k, v):
            assert torch.allclose(task.H, expected, atol=1e-5)
            assert task.nsamples == sum(b.shape[1] for b in batches)
            assert task.fwd_counter == len(batches)
        assert k.H.data_ptr() != leader.H.data_ptr() != v.H.data_ptr()
        assert proc.has_captured_input_ids("self_attn.k") and proc.has_captured_input_ids("self_attn.v")
        assert proc.shared_input_dedup_count == 2
        assert proc.shared_input_leader("self_attn.k") is None

    def test_hooks_after_end_capture_normally_again(self):
        proc = _processor()
        plan = build_shared_input_plan(_QKV_TREE, _LAYER_MODULES)
        model = _fake_model(plan, proc.qcfg)
        names = ["self_attn.q", "self_attn.k"]
        modules = _setup(proc, names)
        proc.begin_shared_input_capture(model, names)
        proc.end_shared_input_capture(names)

        x = _batches(1)[0]
        k = modules["self_attn.k"].module
        proc.pre_process_fwd_hook("self_attn.k")(k, (x,), k(x))
        assert proc.tasks["self_attn.k"].fwd_counter == 1

    def test_dedup_matches_independent_collection(self):
        batches = _batches(n=4, tokens=7)
        names = ["mlp.g", "mlp.u"]

        base = _processor(QuantizeConfig(method=METHOD.GPTQ, hessian=HessianConfig(dedup_shared_inputs=False)))
        plan = build_shared_input_plan(_QKV_TREE, _LAYER_MODULES)
        base_modules = _setup(base, names)
        assert base.begin_shared_input_capture(_fake_model(plan, base.qcfg), names) == {}
        _feed(base, base_modules, names, batches)
        base.end_shared_input_capture(names)
        for task in base.tasks.values():
            task.materialize_global_hessian()

        dedup = _processor()
        dedup_modules = _setup(dedup, names)
        assert dedup.begin_shared_input_capture(_fake_model(plan, dedup.qcfg), names) == {"mlp.u": "mlp.g"}
        _feed(dedup, dedup_modules, names, batches)
        dedup.end_shared_input_capture(names)

        for name in names:
            assert torch.allclose(dedup.tasks[name].H, base.tasks[name].H, atol=1e-6), name
            assert dedup.tasks[name].nsamples == base.tasks[name].nsamples

    def test_keep_mask_path_is_skipped_for_followers(self):
        proc = _processor()
        plan = build_shared_input_plan(_QKV_TREE, _LAYER_MODULES)
        model = _fake_model(plan, proc.qcfg)
        names = ["self_attn.q", "self_attn.k"]
        modules = _setup(proc, names)
        proc.begin_shared_input_capture(model, names)

        x = torch.randn(2, 4, _IN)
        keep = torch.tensor([[True, True, False, False], [True, False, False, False]])
        proc._mask_tls = SimpleNamespace(value=keep)
        proc._set_current_batch_index(0)
        for name in names:
            proc.pre_process_fwd_hook(name)(modules[name].module, (x,), modules[name].module(x))
        proc._set_current_batch_index(None)
        proc.end_shared_input_capture(names)

        kept = torch.cat([x[0, :2], x[1, :1]], dim=0)
        expected = 2.0 * (kept.T @ kept) / kept.shape[0]
        assert proc.tasks["self_attn.q"].nsamples == 3
        assert torch.allclose(proc.tasks["self_attn.k"].H, expected, atol=1e-5)

    def test_end_without_begin_is_noop(self):
        proc = _processor()
        _setup(proc, ["self_attn.q"])
        proc.end_shared_input_capture(["self_attn.q"])
        assert proc.shared_input_dedup_count == 0

    def test_end_tolerates_pruned_tasks(self):
        proc = _processor()
        plan = build_shared_input_plan(_QKV_TREE, _LAYER_MODULES)
        model = _fake_model(plan, proc.qcfg)
        names = ["self_attn.q", "self_attn.k"]
        _setup(proc, names)
        proc.begin_shared_input_capture(model, names)
        proc.tasks.pop("self_attn.q")
        proc.end_shared_input_capture(names)
        assert proc.shared_input_dedup_count == 0
        assert proc.tasks["self_attn.k"].H is None
