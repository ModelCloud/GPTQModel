# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.module_looper import _restrict_quant_devices_for_method
from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig
from gptqmodel.quantization.paroquant.optimization import (
    GroupLinearQuantizer,
    _ParoQuantOptimLinear,
    build_random_rotation_buffers,
    optimize_paroquant_linear,
    pseudo_quantize_dequant,
)
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import get_kernel_for_backend
from gptqmodel.utils.paroquant import (
    apply_paroquant_rotation_reference,
    build_identity_rotation_buffers,
)


def test_paroquant_quantize_config_dispatches_constructor():
    cfg = QuantizeConfig(
        quant_method=METHOD.PAROQUANT,
        format=FORMAT.PAROQUANT,
        bits=4,
        group_size=128,
        krot=8,
    )

    assert cfg.quant_method == METHOD.PAROQUANT
    assert cfg.format == FORMAT.PAROQUANT
    assert cfg.krot == 8
    assert cfg.export_quant_method() == METHOD.PAROQUANT


def test_paroquant_quantize_config_from_external_payload_round_trips():
    cfg = QuantizeConfig.from_quant_config(
        {
            "quant_method": "paroquant",
            "bits": 4,
            "group_size": 128,
            "krot": 8,
            "meta": {
                "opt_rotation_epochs": 10,
                "opt_finetune_epochs": 10,
                "opt_train_samples": 2048,
                "opt_validation_samples": 64,
                "opt_batch_size": 16,
                "opt_rotation_lr": 0.05,
                "opt_weight_lr": 1e-5,
                "opt_quantizer_lr": 1e-6,
                "opt_pair_ratio": 0.5,
                "opt_seed": 0,
            },
        }
    )

    assert cfg.quant_method == METHOD.PAROQUANT
    assert cfg.format == FORMAT.PAROQUANT
    assert cfg.krot == 8
    assert cfg.opt_rotation_epochs == 10
    assert cfg.opt_finetune_epochs == 10
    assert cfg.opt_train_samples == 2048
    assert cfg.opt_validation_samples == 64
    assert cfg.opt_batch_size == 16
    assert cfg.opt_rotation_lr == 0.05
    assert cfg.opt_weight_lr == 1e-5
    assert cfg.opt_quantizer_lr == 1e-6
    assert cfg.opt_pair_ratio == 0.5
    assert cfg.opt_seed == 0


def test_paroquant_kernel_mapping_uses_paroquant_backend():
    from gptqmodel.nn_modules.qlinear.paroquant import ParoQuantQuantLinear

    assert (
        get_kernel_for_backend(BACKEND.PAROQUANT_CUDA, METHOD.PAROQUANT, FORMAT.PAROQUANT)
        is ParoQuantQuantLinear
    )


def test_paroquant_kernel_mapping_uses_paroquant_triton_backend():
    from gptqmodel.nn_modules.qlinear.paroquant_triton import ParoQuantTritonQuantLinear

    assert (
        get_kernel_for_backend(BACKEND.PAROQUANT_TRITON, METHOD.PAROQUANT, FORMAT.PAROQUANT)
        is ParoQuantTritonQuantLinear
    )


def test_paroquant_identity_rotation_buffers_preserve_input():
    x = torch.randn(3, 128, dtype=torch.float16)
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=128,
        group_size=128,
        krot=8,
        dtype=torch.float16,
    )

    rotated = apply_paroquant_rotation_reference(
        x,
        pairs,
        theta,
        scales=channel_scales,
        group_size=128,
    )

    torch.testing.assert_close(rotated, x, atol=0, rtol=0)


def test_paroquant_processor_is_not_awq_subclass():
    assert not issubclass(ParoQuantProcessor, AWQProcessor)


def test_paroquant_quant_device_selection_forces_single_gpu():
    cuda_devices = [torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2")]
    mixed_devices = [torch.device("cpu"), torch.device("cuda:3"), torch.device("cuda:4")]

    assert _restrict_quant_devices_for_method(METHOD.PAROQUANT, cuda_devices) == [torch.device("cuda:0")]
    assert _restrict_quant_devices_for_method(METHOD.PAROQUANT, mixed_devices) == [torch.device("cuda:3")]
    assert _restrict_quant_devices_for_method(METHOD.GPTQ, cuda_devices) == cuda_devices


def test_paroquant_optimizer_improves_over_identity_quantization():
    in_features = 128
    out_features = 12
    group_size = 128
    bits = 4
    seed = 11
    pair_ratio = 1.0 / group_size

    pairs, mask = build_random_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=1,
        pair_ratio=pair_ratio,
        seed=seed,
        device=torch.device("cpu"),
    )

    theta = torch.zeros((1, in_features // 2), dtype=torch.float32)
    theta[~mask] = 0.65
    channel_scales_opt = torch.linspace(0.75, 1.25, steps=in_features, dtype=torch.float32).view(1, -1)
    transformed_weight = (torch.randint(-7, 8, (out_features, in_features), dtype=torch.int32).to(torch.float32)) * 0.25

    original_weight = apply_paroquant_rotation_reference(
        transformed_weight,
        pairs.flip(0),
        -theta.flip(0),
        scales=None,
        group_size=group_size,
    ) / channel_scales_opt

    inputs = torch.randn(256, in_features, dtype=torch.float32)
    targets = F.linear(inputs, original_weight)

    baseline_weight = pseudo_quantize_dequant(
        original_weight,
        bits=bits,
        group_size=group_size,
        sym=False,
        use_ste=False,
    )
    baseline_loss = F.smooth_l1_loss(F.linear(inputs, baseline_weight), targets)

    result = optimize_paroquant_linear(
        weight=original_weight,
        bias=None,
        inputs=inputs,
        bits=bits,
        group_size=group_size,
        sym=False,
        krot=1,
        pair_ratio=pair_ratio,
        train_rows=192,
        val_rows=64,
        batch_size=32,
        rotation_epochs=24,
        finetune_epochs=16,
        rotation_lr=0.05,
        weight_lr=5e-4,
        quantizer_lr=5e-4,
        seed=seed,
    )

    optimized_loss = F.smooth_l1_loss(F.linear(inputs, result.pseudo_weight), targets)
    assert optimized_loss < baseline_loss


def test_paroquant_exported_runtime_state_matches_paper_contract():
    in_features = 128
    out_features = 10
    group_size = 128
    bits = 4

    weight = torch.randn(out_features, in_features, dtype=torch.float32) * 0.3
    bias = torch.randn(out_features, dtype=torch.float32) * 0.1
    pairs, theta_mask = build_random_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=2,
        pair_ratio=2.0 / group_size,
        seed=7,
        device=torch.device("cpu"),
    )

    model = _ParoQuantOptimLinear(
        weight,
        bias,
        bits=bits,
        group_size=group_size,
        sym=False,
        pairs=pairs,
        theta_mask=theta_mask,
    )

    with torch.no_grad():
        model.theta.uniform_(-0.35, 0.35)
        model.reset_masked_angles()
        model.channel_scales_opt.copy_(torch.linspace(0.8, 1.2, steps=in_features))
        model.weight.add_(torch.linspace(-0.05, 0.05, steps=in_features).view(1, -1))

    model.quantizer = GroupLinearQuantizer(
        model.transformed_weight().detach(),
        bits=bits,
        group_size=group_size,
        sym=False,
    )
    with torch.no_grad():
        model.quantizer.scale.mul_(1.05)
        model.quantizer.zero_point_float.add_(0.15)

    transformed = apply_paroquant_rotation_reference(
        model.weight.detach() * model.channel_scales_opt.detach().view(1, -1),
        model.pairs,
        model.theta.detach(),
        scales=None,
        group_size=group_size,
    )
    quantized_transformed = pseudo_quantize_dequant(
        transformed,
        bits=bits,
        group_size=group_size,
        sym=False,
        scale=model.quantizer.scale.detach(),
        zero_point_float=model.quantizer.zero_point_float.detach(),
        use_ste=False,
    )
    expected_pseudo_weight = apply_paroquant_rotation_reference(
        quantized_transformed,
        model.pairs.flip(0),
        -model.theta.detach().flip(0),
        scales=None,
        group_size=group_size,
    ) / model.channel_scales_opt.detach().view(1, -1)

    torch.testing.assert_close(model.pseudo_weight().detach(), expected_pseudo_weight, atol=1e-5, rtol=1e-5)

    pack_weight, _q_scales, _q_zeros, theta, runtime_channel_scales = model.export_pack_state()
    inputs = torch.randn(32, in_features, dtype=torch.float32)
    runtime_outputs = F.linear(
        apply_paroquant_rotation_reference(
            inputs,
            model.pairs,
            theta,
            scales=runtime_channel_scales,
            group_size=group_size,
        ),
        pack_weight,
        bias,
    )

    torch.testing.assert_close(
        runtime_outputs,
        F.linear(inputs, model.pseudo_weight().detach(), bias),
        atol=1e-5,
        rtol=1e-5,
    )
