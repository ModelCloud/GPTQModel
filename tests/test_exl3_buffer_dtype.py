# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""EXL3 sign vectors must stay fp16 across a model-wide dtype cast.

Background
----------
``LinearEXL3`` asserts that the unpacked sign vectors ``suh``/``svh`` are
``torch.half`` because the Hadamard kernels are typed for fp16. They are stored
that way in the checkpoint, but ``nn.Module._apply`` casts *every*
floating-point buffer to the model dtype, so loading an EXL3 checkpoint with
``dtype=bfloat16`` turned them into bf16 and the first forward died with::

    AssertionError: suh is wrong datatype

The quantizer happily writes bf16 checkpoints, so the failure only appeared on
reload -- which is exactly when a saved artifact is supposed to be trustworthy.
``ExllamaV3Linear`` now restores the kernel dtypes after every cast.
"""

from __future__ import annotations

import torch

from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear


def _module(dtype: torch.dtype = torch.float16) -> ExllamaV3Linear:
    return ExllamaV3Linear(
        in_features=256,
        out_features=256,
        name="model.layers.0.self_attn.q_proj",
        tensors={
            # 48 columns / 16 == 3 bits per weight
            "trellis": torch.zeros((16, 16, 48), dtype=torch.int16),
            "suh": torch.ones(256, dtype=dtype),
            "svh": torch.ones(256, dtype=dtype),
            "su": torch.zeros(256, dtype=torch.int16),
            "sv": torch.zeros(256, dtype=torch.int16),
            "bias": None,
            "mcg": None,
            "mul1": None,
        },
    )


def test_bfloat16_cast_keeps_sign_vectors_in_fp16():
    module = _module()
    module.bfloat16()

    assert module.suh.dtype == torch.half
    assert module.svh.dtype == torch.half


def test_float32_cast_keeps_sign_vectors_in_fp16():
    module = _module()
    module.float()

    assert module.suh.dtype == torch.half
    assert module.svh.dtype == torch.half


def test_cast_preserves_packed_int16_vectors():
    module = _module()
    module.bfloat16()

    assert module.su.dtype == torch.int16
    assert module.sv.dtype == torch.int16
    assert module.trellis.dtype == torch.int16


def test_cast_preserves_sign_values():
    module = _module()
    before = module.suh.clone()
    module.bfloat16()

    assert torch.equal(module.suh, before)
