# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The default KernelConfig must drive AWQ Triton dispatch without an explicit plan."""

import pytest
import torch

import gptqmodel.nn_modules.qlinear.gemm_awq_triton as awq_triton_module
from gptqmodel.backend import AwqTritonConfig
from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear
from gptqmodel.quantization.awq.modules.triton.scheduler import (
    AwqTritonPlan,
    clear_awq_triton_rules,
    default_schedule_mode,
    legacy_plan,
    register_awq_triton_rule,
)
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.utils.kernel import KernelConfig, configure_kernel, get_kernel_config


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]

    assert unpacked.shape[1] % pack_factor == 0
    packed = torch.zeros(
        (unpacked.shape[0], unpacked.shape[1] // pack_factor),
        dtype=torch.int32,
    )
    for col in range(unpacked.shape[1] // pack_factor):
        for i, order in enumerate(order_map):
            value = unpacked[:, col * pack_factor + order].to(torch.int32)
            packed[:, col] |= value << (i * bits)
    return packed


def _make_packed_buffers(bits: int, in_features: int, out_features: int, group_size: int):
    groups = in_features // group_size
    int_weight = torch.randint(0, 2**bits, size=(in_features, out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, out_features), dtype=torch.int32)
    scales = (torch.rand(groups, out_features, dtype=torch.float16) * 0.5) + 0.75
    return (
        _pack_awq_tensor(int_weight, bits),
        _pack_awq_tensor(zero_points, bits),
        scales,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AWQ Triton forward")
def test_default_kernel_config_forward_uses_configured_schedule_mode():
    pytest.importorskip("triton")
    torch.manual_seed(0)

    previous = get_kernel_config()
    configure_kernel(KernelConfig())
    try:
        assert default_schedule_mode() == "auto"

        bits = 4
        in_features = 512
        out_features = 512
        group_size = 128
        qweight, qzeros, scales = _make_packed_buffers(
            bits, in_features, out_features, group_size
        )

        # No kernel/schedule argument: the module must follow the process-wide
        # default KernelConfig instead of treating its mode as an explicit plan.
        module = AwqGEMMTritonLinear(
            bits=bits,
            group_size=group_size,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            register_buffers=True,
        ).cuda()
        module.qweight.copy_(qweight.cuda())
        module.qzeros.copy_(qzeros.cuda())
        module.scales.copy_(scales.cuda())
        module.post_init()
        module.eval()

        x = torch.randn(1, 1, in_features, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            actual = module(x)

        assert actual.shape == (1, 1, out_features)
        assert torch.isfinite(actual).all()

        dequant_weight = dequantize_gemm(
            qweight=module.qweight,
            qzeros=module.qzeros,
            scales=module.scales,
            bits=bits,
            group_size=group_size,
        ).to(device=x.device, dtype=x.dtype)
        expected = torch.matmul(x.reshape(-1, in_features), dequant_weight).reshape_as(actual)
        abs_diff = (actual - expected).abs()
        assert abs_diff.max().item() <= 1.0
        assert abs_diff.mean().item() <= 0.02

        # The configured value is a dispatch mode; it must stay in the mode
        # slot and never be materialized as an explicit schedule plan.
        assert module.awq_triton_schedule is None
        assert module.awq_triton_schedule_mode == "auto"
    finally:
        configure_kernel(previous)


_IN = 512
_OUT = 512
_GROUP = 128
_ROWS = 16


def _build_module(**kwargs):
    # Construction runs the kernel's validate_once, which requires Triton.
    pytest.importorskip("triton")
    return AwqGEMMTritonLinear(
        bits=4,
        group_size=_GROUP,
        sym=True,
        desc_act=False,
        in_features=_IN,
        out_features=_OUT,
        bias=False,
        register_buffers=True,
        **kwargs,
    )


# (constructor kwargs, expected mode slot, expected explicit plan slot)
_CONFIG_CASES = [
    pytest.param({}, "auto", None, id="process-default"),
    pytest.param({"kernel": KernelConfig()}, "auto", None, id="kernel-default"),
    pytest.param({"kernel": KernelConfig(awq_triton_schedule="legacy")}, "legacy", None, id="kernel-legacy"),
    pytest.param({"backend": AwqTritonConfig()}, "auto", None, id="backend-default"),
    pytest.param({"backend": AwqTritonConfig(schedule="legacy")}, "legacy", None, id="backend-legacy"),
    pytest.param(
        {"backend": AwqTritonConfig(schedule="legacy"), "schedule_mode": "auto"},
        "auto", None, id="explicit-mode-overrides-config",
    ),
]


@pytest.mark.parametrize("kwargs, mode, schedule", _CONFIG_CASES)
def test_config_schedule_is_a_mode_never_a_plan(kwargs, mode, schedule):
    previous = get_kernel_config()
    configure_kernel(KernelConfig())
    try:
        module = _build_module(**kwargs)
    finally:
        configure_kernel(previous)
    assert module.awq_triton_schedule_mode == mode
    assert module.awq_triton_schedule is schedule


def test_explicit_plan_keeps_plan_slot():
    plan = AwqTritonPlan("fused", 16, 64, 64, 1, 4, 2)
    module = _build_module(backend=AwqTritonConfig(schedule="legacy"), schedule=plan)
    assert module.awq_triton_schedule == plan
    assert module.awq_triton_schedule_mode == "legacy"


_DENSE = AwqTritonPlan(path="dense")
_EXPLICIT = AwqTritonPlan("fused", 16, 64, 64, 1, 4, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AWQ Triton forward")
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        # "auto" follows the measured exact-shape rule registered below.
        pytest.param({}, _DENSE, id="process-default"),
        pytest.param({"backend": AwqTritonConfig()}, _DENSE, id="backend-default"),
        # A per-load legacy policy must win over the process-wide "auto".
        pytest.param({"kernel": KernelConfig(awq_triton_schedule="legacy")},
                     legacy_plan(_ROWS), id="kernel-legacy"),
        pytest.param({"backend": AwqTritonConfig(schedule="legacy")},
                     legacy_plan(_ROWS), id="backend-legacy"),
        pytest.param({"backend": AwqTritonConfig(schedule="legacy"), "schedule_mode": "auto"},
                     _DENSE, id="explicit-mode"),
        pytest.param({"schedule": _EXPLICIT}, _EXPLICIT, id="explicit-plan"),
    ],
)
def test_config_schedule_mode_drives_forward(monkeypatch, kwargs, expected):
    pytest.importorskip("triton")
    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device())

    previous = get_kernel_config()
    configure_kernel(KernelConfig())
    clear_awq_triton_rules()
    try:
        # Make "auto" observable on every device: an exact measured rule
        # selects the dense path, while "legacy" keeps the fused 32-tile plan.
        register_awq_triton_rule(
            device, _DENSE, M=_ROWS, N=_OUT, K=_IN, group_size=_GROUP,
            input_dtype=torch.float16, output_dtype=torch.float16,
            compute_dtype=torch.float16, fp32_accum=True,
        )

        selected = []
        select = awq_triton_module.select_awq_triton_plan

        def spy(**plan_kwargs):
            plan = select(**plan_kwargs)
            selected.append(plan)
            return plan

        monkeypatch.setattr(awq_triton_module, "select_awq_triton_plan", spy)

        qweight, qzeros, scales = _make_packed_buffers(4, _IN, _OUT, _GROUP)
        module = _build_module(**kwargs).to(device)
        module.qweight.copy_(qweight.to(device))
        module.qzeros.copy_(qzeros.to(device))
        module.scales.copy_(scales.to(device))
        module.post_init()
        module.eval()

        x = torch.randn(_ROWS, _IN, device=device, dtype=torch.float16)
        with torch.inference_mode():
            actual = module(x)

        assert selected == [expected]
        dequant_weight = dequantize_gemm(
            qweight=module.qweight, qzeros=module.qzeros, scales=module.scales,
            bits=4, group_size=_GROUP,
        ).to(device=device, dtype=x.dtype)
        expected_out = torch.matmul(x, dequant_weight)
        abs_diff = (actual - expected_out).abs()
        assert torch.isfinite(actual).all()
        assert abs_diff.max().item() <= 1.0
        assert abs_diff.mean().item() <= 0.02
    finally:
        clear_awq_triton_rules()
        configure_kernel(previous)
