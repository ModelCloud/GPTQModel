# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig
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
        }
    )

    assert cfg.quant_method == METHOD.PAROQUANT
    assert cfg.format == FORMAT.PAROQUANT
    assert cfg.krot == 8


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
