# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Per-channel (``group_size=-1``) GEMM AWQ must fall back to the portable kernel.

``AwqGEMMLinear.SUPPORTS_GROUP_SIZE`` advertises ``-1``, so the model-wide
kernel stays ``AwqGEMMLinear`` for ``AWQConfig(group_size=-1)``.  Packing a
per-channel module normalizes the group to ``in_features``, so
``AwqGEMMLinear._validate`` additionally requires ``out_features %
in_features == 0``.  GQA ``k_proj``/``v_proj`` (896->128) and ``down_proj``
(4864->896) never satisfy that, and the per-module ``AwqTorchLinear`` fallback
used to look only at the static divisibility attributes, so it never fired and
``create_quant_module`` raised after the whole layer had been calibrated
(#5391).
"""

from types import SimpleNamespace

from gptqmodel.looper.awq_processor import (
    AWQProcessor,
    _kernel_shape_supported,
)
from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import AWQConfig


def _processor_stub(*, group_size=-1, model_kernel=AwqGEMMLinear, bits=4):
    class Stub:
        qcfg = AWQConfig(bits=bits, group_size=group_size, sym=True)
        format = FORMAT.GEMM
        gptq_model = SimpleNamespace(qlinear_kernel=model_kernel, lm_head="lm_head")
        qlinear_kernel = model_kernel

        _module_bits = AWQProcessor._module_bits
        _module_group_size = AWQProcessor._module_group_size
        _module_format = AWQProcessor._module_format
        _resolve_qlinear_kernel = AWQProcessor._resolve_qlinear_kernel
        _portable_kernel_for_shape = AWQProcessor._portable_kernel_for_shape
        _select_qlinear_kernel_for_format = AWQProcessor._select_qlinear_kernel_for_format
        _quant_device_is_npu = AWQProcessor._quant_device_is_npu

    return Stub()


def test_kernel_shape_supported_applies_the_gemm_group_rule():
    # The static contract (out_features % 64 == 0) accepts both GQA shapes.
    assert 128 % 64 == 0 and 896 % 64 == 0

    assert not _kernel_shape_supported(AwqGEMMLinear, 896, 128, -1)
    assert not _kernel_shape_supported(AwqGEMMLinear, 4864, 896, -1)
    # A square per-channel projection still packs on the CUDA GEMM kernel.
    assert _kernel_shape_supported(AwqGEMMLinear, 896, 896, -1)
    # The portable kernel packs every shape the format admits.
    assert _kernel_shape_supported(AwqTorchLinear, 896, 128, -1)
    assert _kernel_shape_supported(AwqTorchLinear, 4864, 896, -1)


def test_kernel_shape_supported_applies_the_gemm_group_rule_for_positive_groups():
    # 192 is a multiple of the static 64-wide output contract but not of the
    # configured 128-wide group, so the GEMM kernel cannot pack it either.
    assert not _kernel_shape_supported(AwqGEMMLinear, 4096, 192, 128)
    assert _kernel_shape_supported(AwqGEMMLinear, 4096, 256, 128)
    assert _kernel_shape_supported(AwqTorchLinear, 4096, 192, 128)


def test_resolve_qlinear_kernel_falls_back_for_perchannel_gqa():
    stub = _processor_stub()

    assert stub._resolve_qlinear_kernel("k_proj", in_features=896, out_features=128) is AwqTorchLinear
    assert (
        stub._resolve_qlinear_kernel("down_proj", in_features=4864, out_features=896)
        is AwqTorchLinear
    )
    # The shape the GEMM kernel can pack keeps the native kernel.
    assert stub._resolve_qlinear_kernel("o_proj", in_features=896, out_features=896) is AwqGEMMLinear


def test_resolve_qlinear_kernel_falls_back_without_a_model_wide_kernel():
    # `_select_qlinear_kernel_for_format` returns AwqGEMMLinear for GEMM before
    # any module shape is known; that path needs the same shape fallback.
    stub = _processor_stub(model_kernel=None)

    assert stub._resolve_qlinear_kernel("k_proj", in_features=896, out_features=128) is AwqTorchLinear
    assert stub._resolve_qlinear_kernel("o_proj", in_features=896, out_features=896) is AwqGEMMLinear


def test_resolve_qlinear_kernel_falls_back_for_positive_group_size_remainder():
    stub = _processor_stub(group_size=128)

    assert (
        stub._resolve_qlinear_kernel("up_proj", in_features=4096, out_features=192)
        is AwqTorchLinear
    )
    assert stub._resolve_qlinear_kernel("gate_proj", in_features=4096, out_features=256) is AwqGEMMLinear


def test_portable_kernel_keeps_the_requested_kernel_when_no_kernel_packs_the_shape():
    class Unpackable:
        @staticmethod
        def _validate_shape_constraints(*, in_features, out_features):
            return False, NotImplementedError("no")

    stub = _processor_stub()
    # 100 is not a multiple of the portable kernel's 8-wide output contract, so
    # neither kernel packs the shape and the requested one is kept.
    assert (
        stub._portable_kernel_for_shape(
            Unpackable, "proj", in_features=896, out_features=100, group_size=-1
        )
        is Unpackable
    )


def test_resolve_qlinear_kernel_still_returns_the_portable_kernel_for_non_4bit():
    stub = _processor_stub(bits=3)

    assert stub._resolve_qlinear_kernel(
        "k_proj", in_features=896, out_features=128
    ) is AwqTorchLinear
