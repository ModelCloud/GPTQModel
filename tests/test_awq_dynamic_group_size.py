# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.quantization.config import AWQConfig
from gptqmodel.utils.model import create_quant_module


class ShapeOnlyAwqGemm:
    """AwqGEMMLinear's packing contract without the CUDA-extension platform gate.

    Deliberately not a ``BaseQuantLinear`` subclass: the kernel registry
    enumerates every subclass, and a second ``AWQ_GEMM``/``GEMM`` declaration
    makes ``get_kernel_for_backend`` reject the real kernel as ambiguous when
    another test file loads it in the same process.
    """

    SUPPORTS_GROUP_SIZE = AwqGEMMLinear.SUPPORTS_GROUP_SIZE

    @staticmethod
    def validate(**kwargs):
        kwargs["check_platform"] = False
        return AwqGEMMLinear._validate(**kwargs)


def _stub(bits=4, dynamic=None):
    class Stub:
        qcfg = AWQConfig(bits=bits, group_size=-1, sym=True, dynamic=dynamic)
        _module_bits = AWQProcessor._module_bits
        _module_group_size = AWQProcessor._module_group_size
        pseudo_quantize_tensor = AWQProcessor.pseudo_quantize_tensor
        _pseudo_quantize_tensor_into = AWQProcessor._pseudo_quantize_tensor_into

    return Stub()


def test_module_group_size_follows_the_dynamic_override():
    stub = _stub(dynamic={"model.layers.0.mlp.down_proj": {"group_size": 16}})
    stub.qcfg.group_size = 128

    group_size = stub._module_group_size(
        "model.layers.0.mlp.down_proj"
    )
    weight = torch.randn(8, 128, dtype=torch.float16)
    _, scales, zeros = stub.pseudo_quantize_tensor(
        weight,
        bits=4,
        group_size=group_size,
    )

    assert group_size == 16
    assert scales.shape == (8, 8)
    assert zeros.shape == (8, 8)


def test_dynamic_group_size_selects_compatible_kernel_and_packs():
    config = AWQConfig(
        bits=4,
        group_size=128,
        sym=True,
        dynamic={"proj": {"group_size": 16}},
    )

    class Stub:
        qcfg = config
        format = FORMAT.GEMM
        gptq_model = SimpleNamespace(qlinear_kernel=AwqGEMMLinear)
        _module_bits = AWQProcessor._module_bits
        _module_group_size = AWQProcessor._module_group_size
        _module_format = AWQProcessor._module_format
        _resolve_qlinear_kernel = AWQProcessor._resolve_qlinear_kernel
        _select_qlinear_kernel_for_format = AWQProcessor._select_qlinear_kernel_for_format
        _quant_device_is_npu = AWQProcessor._quant_device_is_npu
        pseudo_quantize_tensor = AWQProcessor.pseudo_quantize_tensor

    stub = Stub()
    kernel_cls = stub._resolve_qlinear_kernel("proj")
    assert kernel_cls is AwqTorchLinear

    source = torch.nn.Linear(128, 8, bias=False, dtype=torch.float16)
    with torch.no_grad():
        source.weight.copy_(torch.linspace(-0.5, 0.5, 128).repeat(8, 1))
    staged, scales, zeros = stub.pseudo_quantize_tensor(
        source.weight.detach(), group_size=16
    )
    with torch.no_grad():
        source.weight.copy_(staged)

    model = torch.nn.Module()
    model.proj = source
    create_quant_module(
        name="proj",
        linear_cls=kernel_cls,
        bits=4,
        desc_act=False,
        dynamic=config.dynamic,
        group_size=config.group_size,
        module=model,
        submodule=source,
        sym=True,
        device=DEVICE.CPU,
        lm_head_name="lm_head",
        pack_dtype=torch.int32,
        format=FORMAT.GEMM,
        register_buffers=False,
        dtype=torch.float16,
    )
    packed = model.proj
    assert isinstance(packed, AwqTorchLinear)
    assert packed.group_size == 16
    packed.pack(source, scales, zeros)
    decoded = dequantize_gemm(
        packed.qweight, packed.qzeros, packed.scales, 4, 16
    )
    torch.testing.assert_close(decoded.T, staged, rtol=2e-3, atol=2e-3)


def test_awq_retunes_group_size_before_calibration_for_non_divisible_in_features():
    """A 448-wide projection must not fail after calibration with group_size 128.

    Phi-tiny-MoE has ``intermediate_size=448``; 448 is not a multiple of 128, so
    the AWQ weight-mean step raised only after the whole calibration pass. The
    processor now resolves a supported divisor (64) up front and records it as a
    per-module ``dynamic`` override.
    """

    import pytest

    from gptqmodel.looper.awq_processor import AWQProcessor, _accumulate_awq_weight_mean

    class Stub:
        qcfg = AWQConfig(bits=4, group_size=128, sym=True)
        format = FORMAT.GEMM
        gptq_model = SimpleNamespace(qlinear_kernel=ShapeOnlyAwqGemm, lm_head="lm_head")

        def __init__(self):
            self.model = torch.nn.Module()
            # Phi-tiny-MoE expert down_proj: in_features=448, out_features=4096.
            self.model.proj = torch.nn.Linear(448, 4096, bias=False, dtype=torch.float16)

        _module_group_size = AWQProcessor._module_group_size
        _module_format = AWQProcessor._module_format
        _module_bits = AWQProcessor._module_bits
        _resolve_module_kernel = AWQProcessor._resolve_module_kernel
        _kernel_group_size_supported = AWQProcessor._kernel_group_size_supported
        _resolve_qlinear_kernel = AWQProcessor._resolve_qlinear_kernel
        _select_qlinear_kernel_for_format = AWQProcessor._select_qlinear_kernel_for_format
        _quant_device_is_npu = AWQProcessor._quant_device_is_npu
        _largest_supported_group_divisor = AWQProcessor._largest_supported_group_divisor
        _ensure_group_size_divisibility = AWQProcessor._ensure_group_size_divisibility
        qlinear_kernel = ShapeOnlyAwqGemm

    stub = Stub()
    weight = stub.model.proj.weight.detach()

    # The original configuration fails inside the weight-mean step.
    with pytest.raises(ValueError, match="divisible by group_size"):
        _accumulate_awq_weight_mean([stub.model.proj], [128])

    stub._ensure_group_size_divisibility()

    assert stub._module_group_size("proj") == 64
    assert stub.qcfg.dynamic["+:^proj$"] == {"group_size": 64}
    # The step that used to raise now succeeds with the resolved divisor.
    w_sum, row_count = _accumulate_awq_weight_mean([stub.model.proj], [64])
    assert w_sum.shape == (448,)
    assert row_count == weight.shape[0]


def test_awq_retunes_group_size_for_non_divisible_out_features():
    """Phi-tiny-MoE expert gate/up: in_features divides 128, out_features=448 does not.

    ``AwqGEMMLinear`` requires both axes to be group multiples. The previous
    retune only checked ``in_features``, so a 4096->448 expert projection kept
    group_size 128 and raised in ``create_quant_module`` after calibration.
    """

    from gptqmodel.looper.awq_processor import AWQProcessor

    class Stub:
        qcfg = AWQConfig(bits=4, group_size=128, sym=True)
        format = FORMAT.GEMM
        gptq_model = SimpleNamespace(qlinear_kernel=ShapeOnlyAwqGemm, lm_head="lm_head")

        def __init__(self):
            self.model = torch.nn.Module()
            # Phi-tiny-MoE expert gate_proj/up_proj: in_features=4096, out_features=448.
            self.model.proj = torch.nn.Linear(4096, 448, bias=False, dtype=torch.float16)

        _module_group_size = AWQProcessor._module_group_size
        _module_format = AWQProcessor._module_format
        _module_bits = AWQProcessor._module_bits
        _resolve_module_kernel = AWQProcessor._resolve_module_kernel
        _kernel_group_size_supported = AWQProcessor._kernel_group_size_supported
        _resolve_qlinear_kernel = AWQProcessor._resolve_qlinear_kernel
        _select_qlinear_kernel_for_format = AWQProcessor._select_qlinear_kernel_for_format
        _quant_device_is_npu = AWQProcessor._quant_device_is_npu
        _largest_supported_group_divisor = AWQProcessor._largest_supported_group_divisor
        _ensure_group_size_divisibility = AWQProcessor._ensure_group_size_divisibility
        qlinear_kernel = ShapeOnlyAwqGemm

    stub = Stub()
    assert stub._module_group_size("proj") == 128

    # The input axis divides 128, so the old predicate accepted this module.
    assert stub.model.proj.in_features % 128 == 0
    # The kernel rejects it because 448 is not a multiple of 128.
    assert not stub._kernel_group_size_supported(
        ShapeOnlyAwqGemm,
        "proj",
        4096,
        448,
        128,
        dtype=torch.float16,
    )

    stub._ensure_group_size_divisibility()

    assert stub._module_group_size("proj") == 64
    assert stub.qcfg.dynamic["+:^proj$"] == {"group_size": 64}
    assert stub._kernel_group_size_supported(
        ShapeOnlyAwqGemm,
        "proj",
        4096,
        448,
        64,
        dtype=torch.float16,
    )


def test_awq_leaves_unpackable_module_untouched():
    """Phi-tiny-MoE router (4096 -> 16) packs at no group size; do not rewrite it.

    The router is declared non-quantized with ``router:!``. Retuning it to
    per-channel would add a dynamic override that is itself invalid, so the
    processor must leave the configured size alone.
    """

    from gptqmodel.looper.awq_processor import AWQProcessor

    class Stub:
        qcfg = AWQConfig(bits=4, group_size=128, sym=True)
        format = FORMAT.GEMM
        gptq_model = SimpleNamespace(qlinear_kernel=ShapeOnlyAwqGemm, lm_head="lm_head")

        def __init__(self):
            self.model = torch.nn.Module()
            self.model.proj = torch.nn.Linear(4096, 16, bias=False, dtype=torch.float16)

        _module_group_size = AWQProcessor._module_group_size
        _module_format = AWQProcessor._module_format
        _module_bits = AWQProcessor._module_bits
        _resolve_module_kernel = AWQProcessor._resolve_module_kernel
        _kernel_group_size_supported = AWQProcessor._kernel_group_size_supported
        _resolve_qlinear_kernel = AWQProcessor._resolve_qlinear_kernel
        _select_qlinear_kernel_for_format = AWQProcessor._select_qlinear_kernel_for_format
        _quant_device_is_npu = AWQProcessor._quant_device_is_npu
        _largest_supported_group_divisor = AWQProcessor._largest_supported_group_divisor
        _ensure_group_size_divisibility = AWQProcessor._ensure_group_size_divisibility
        qlinear_kernel = ShapeOnlyAwqGemm

    stub = Stub()
    assert not stub._kernel_group_size_supported(
        ShapeOnlyAwqGemm, "proj", 4096, 16, -1, dtype=torch.float16
    )

    stub._ensure_group_size_divisibility()

    assert stub._module_group_size("proj") == 128
    assert stub.qcfg.dynamic is None
