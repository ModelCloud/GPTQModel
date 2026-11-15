# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``model.optimize()`` must reach every CPU kernel's ``optimize`` override.

``BaseQModel.optimize`` calls ``module.optimize(fullgraph=..., backend=...,
mode=...)``. Five CPU kernels overrode ``optimize(self)`` with no parameters,
so compiling a CPU AUTO model raised ``TypeError`` before the model-level
compile even started (#4068).
"""

import torch.nn as nn

from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear
from gptqmodel.nn_modules.qlinear.torch_int8 import TorchInt8Linear
from gptqmodel.quantization import FORMAT


_KERNEL_KWARGS = {
    "bits": 4,
    "group_size": 32,
    "sym": True,
    "desc_act": False,
    "in_features": 32,
    "out_features": 32,
    "bias": False,
    "register_buffers": True,
}


def _cpu_kernels():
    return [
        TorchAtenLinear(**_KERNEL_KWARGS),
        TorchFusedLinear(**_KERNEL_KWARGS),
        TorchFusedAwqLinear(checkpoint_format=FORMAT.GEMM, **_KERNEL_KWARGS),
        TorchAtenAwqLinear(checkpoint_format=FORMAT.GEMM, **_KERNEL_KWARGS),
        TorchInt8Linear(**{**_KERNEL_KWARGS, "bits": 8}),
    ]


def test_cpu_model_optimize_accepts_the_compile_options():
    model = object.__new__(BaseQModel)
    nn.Module.__init__(model)
    model.quantized = True
    model.compiled = False
    modules = _cpu_kernels()
    model.model = nn.Sequential(*modules)

    model.optimize(backend="eager", mode=None, fullgraph=False)

    assert all(module.optimized for module in modules)
