# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization import METHOD
from gptqmodel.utils.importer import select_quant_linear
from gptqmodel.utils.rocm import IS_ROCM
from gptqmodel.utils.torch import HAS_CUDA, HAS_MPS, HAS_XPU


def _iter_kernel_classes():
    seen = set()
    stack = list(BaseQuantLinear.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        if "SUPPORTS_FORMATS" in cls.__dict__:
            yield cls


def _infer_quant_methods(cls):
    supported = getattr(cls, "SUPPORTS_METHODS", None)
    if supported is None:
        raise ValueError(f"{cls.__name__} is missing SUPPORTS_METHODS.")
    return [
        METHOD(method) if isinstance(method, METHOD) else METHOD(str(method).lower())
        for method in supported
    ]


def _pick_device(cls):
    devices = getattr(cls, "SUPPORTS_DEVICES", [])
    if DEVICE.ALL in devices:
        return DEVICE.CPU
    if DEVICE.CPU in devices:
        return DEVICE.CPU
    if DEVICE.CUDA in devices and HAS_CUDA:
        return DEVICE.CUDA
    if DEVICE.ROCM in devices and IS_ROCM:
        return DEVICE.ROCM
    if DEVICE.XPU in devices and HAS_XPU:
        return DEVICE.XPU
    if DEVICE.MPS in devices and HAS_MPS:
        return DEVICE.MPS
    return None


def _pick_group_size(cls):
    group_sizes = list(getattr(cls, "SUPPORTS_GROUP_SIZE", []))
    for candidate in group_sizes:
        if candidate != -1:
            return candidate
    return group_sizes[0] if group_sizes else 128


CASES = []
for kernel_cls in sorted(_iter_kernel_classes(), key=lambda cls: cls.__name__):
    for method in _infer_quant_methods(kernel_cls):
        for fmt in kernel_cls.SUPPORTS_FORMATS:
            CASES.append((kernel_cls, method, fmt))


@pytest.mark.parametrize("kernel_cls,method,fmt", CASES)
def test_select_quant_linear_smoke(kernel_cls, method, fmt):
    device = _pick_device(kernel_cls)
    if device is None:
        pytest.skip(f"No supported device available for {kernel_cls.__name__}.")

    ok, err = kernel_cls.cached_validate_once()
    if not ok:
        pytest.skip(f"{kernel_cls.__name__} unavailable: {err}")

    pack_dtype = kernel_cls.SUPPORTS_PACK_DTYPES[0]
    bits = kernel_cls.SUPPORTS_BITS[0]
    group_size = _pick_group_size(kernel_cls)
    desc_act = kernel_cls.SUPPORTS_DESC_ACT[0]
    sym = kernel_cls.SUPPORTS_SYM[0]

    qlinear_cls = select_quant_linear(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        device=device,
        backend=kernel_cls.SUPPORTS_BACKEND,
        format=fmt,
        quant_method=method,
        pack_dtype=pack_dtype,
    )

    assert qlinear_cls is kernel_cls
