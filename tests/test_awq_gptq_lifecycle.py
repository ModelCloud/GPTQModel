# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Micro-tests for processor lifecycle boundaries touched by AWQ MoE capture."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor, _compute_awq_weight_mean
from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import (
    FORMAT,
    METHOD,
    ExpertsRoutingBypass,
    HessianConfig,
    MoEConfig,
    QuantizeConfig,
)


def _named_linear(name: str, in_features: int = 3, out_features: int = 2) -> NamedModule:
    return NamedModule(
        nn.Linear(in_features, out_features, bias=False),
        name=name,
        full_name=f"model.layers.0.{name}",
        layer_index=0,
    )


def _awq_processor(qcfg: QuantizeConfig) -> AWQProcessor:
    model = SimpleNamespace(
        rotary_embedding=None,
        qlinear_kernel=None,
        awq_input_feature_aggregation=lambda name: (
            {"mode": "token_rows", "capture_root": True}
            if name == "mlp"
            else {"mode": "token_rows"}
            if name.startswith("mlp.")
            else None
        ),
    )
    return AWQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=None,
        prepare_dataset_func=None,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
        gptq_model=model,
        model=nn.Module(),
    )


def test_awq_capture_is_pre_forward_and_gptq_does_not_receive_the_hook():
    events = []

    class Root(nn.Module):
        def forward(self, hidden_states):
            events.append(("forward", hidden_states))
            return hidden_states + 1

    root = Root()
    handles = []
    probe = _awq_processor(
        QuantizeConfig(method=METHOD.AWQ, format=FORMAT.GEMM)
    )
    # Keep the assertion focused on hook timing; the real AWQ processor's
    # recorder is exercised separately below and in the integration test.
    probe.record_moe_root_input_feature = lambda name, hidden_states: events.append(
        ("capture", name, hidden_states)
    )
    assert probe.register_moe_root_capture_hook(root, "mlp", handles) is True

    hidden_states = torch.randn(1, 2, 3)
    root(hidden_states)
    assert [event[0] for event in events] == ["capture", "forward"]
    assert events[0][1] == "mlp"
    assert events[0][2] is hidden_states

    handles[0].remove()
    events.clear()
    root(hidden_states)
    assert [event[0] for event in events] == ["forward"]

    # GPTQ inherits the generic no-op lifecycle method; it must not receive an
    # AWQ root hook even when it processes the same MoE module.
    gptq_probe = GPTQProcessor(
        tokenizer=None,
        qcfg=QuantizeConfig(method=METHOD.GPTQ),
        calibration=None,
        prepare_dataset_func=None,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )
    assert gptq_probe.register_moe_root_capture_hook(root, "mlp", []) is False


def test_gptq_lifecycle_collects_normalized_hessian_without_awq_feature_state():
    qcfg = QuantizeConfig(
        method=METHOD.GPTQ,
        hessian=HessianConfig(staging_dtype=torch.float32),
        moe=MoEConfig(routing=ExpertsRoutingBypass()),
    )
    processor = GPTQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=None,
        prepare_dataset_func=None,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    named_module = _named_linear("mlp.experts.0.gate_proj")
    named_module.module.to(device)
    processor.preprocess(named_module)
    task = processor.tasks[named_module.name]

    inputs = torch.tensor(
        [[[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]], [[-1.0, 1.0, 2.0], [0.5, 2.0, -2.0]]]
    ).to(device)
    outputs = torch.zeros((2, 2, 2), device=device)
    processor._set_current_batch_index(4)
    processor.pre_process_fwd_hook(named_module.name)(named_module.module, (inputs,), outputs)
    processor._set_current_batch_index(None)

    assert qcfg.moe_routing_bypass() is True
    assert task.fwd_counter == 1
    assert task.H is None
    assert task._device_hessian_partials

    task.finalize_hessian()
    flattened = inputs.reshape(-1, inputs.shape[-1])
    expected = 2.0 / flattened.shape[0] * flattened.to(torch.float32).T @ flattened.to(torch.float32)
    torch.testing.assert_close(task.H, expected, atol=0, rtol=0)
    assert not hasattr(task, "_feature_stats")


def test_awq_lifecycle_uses_weight_mean_and_deduplicates_bypass_root_capture():
    qcfg = QuantizeConfig(
        method=METHOD.AWQ,
        format=FORMAT.GEMM,
        group_size=2,
        moe=MoEConfig(routing=ExpertsRoutingBypass()),
    )
    processor = _awq_processor(qcfg)
    named_module = _named_linear("mlp.experts.0.gate_proj", in_features=4, out_features=2)
    processor.preprocess(named_module)

    assert qcfg.moe_routing_bypass() is True
    assert "mlp" in processor.tasks

    processor._set_current_batch_index(2)
    root_input = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    processor.record_moe_root_input_feature("mlp", root_input)
    processor.record_moe_root_input_feature("mlp", root_input + 100)
    processor._set_current_batch_index(None)

    assert len(processor.tasks["mlp"]["inputs"]) == 1
    assert torch.equal(processor.tasks["mlp"]["inputs"][0], root_input)

    captured_input = torch.randn(1, 3, 4)
    processor.pre_process_fwd_hook(named_module.name)(named_module.module, (captured_input,), None)
    assert torch.equal(processor.tasks[named_module.name]["inputs"][0], captured_input)
    assert "H" not in processor.tasks[named_module.name]

    weight = named_module.module.weight.detach().abs().reshape(2, 2, 2)
    weight.div_(weight.amax(dim=2, keepdim=True).add_(1e-6))
    expected = weight.reshape(2, 4).sum(dim=0) / 2
    actual = _compute_awq_weight_mean([named_module.module], group_size=2)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.cuda
def test_awq_weight_mean_gpu_matches_cpu_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the AWQ GPU math path")

    torch.manual_seed(17)
    cpu_layers = [nn.Linear(4, 3, bias=False), nn.Linear(4, 2, bias=False)]
    gpu_layers = [nn.Linear(4, 3, bias=False).cuda(), nn.Linear(4, 2, bias=False).cuda()]
    for cpu_layer, gpu_layer in zip(cpu_layers, gpu_layers):
        gpu_layer.load_state_dict(cpu_layer.state_dict())

    cpu_mean = _compute_awq_weight_mean(cpu_layers, group_size=2)
    gpu_mean = _compute_awq_weight_mean(gpu_layers, group_size=2).cpu()
    torch.testing.assert_close(gpu_mean, cpu_mean, atol=1e-6, rtol=1e-6)
