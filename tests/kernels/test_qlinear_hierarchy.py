# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
import torch

from gptqmodel.nn_modules.qlinear import AWQuantLinear, BaseQuantLinear, GPTQQuantLinear, GroupedQuantLinear
from gptqmodel.nn_modules.qlinear.bitblas_awq import AWQBitBlasKernel
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8QuantLinear
from gptqmodel.nn_modules.qlinear.gemm_hf_kernel_awq import HFKernelAwqLinear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchQuantLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqQuantLinear


def test_quant_linear_hierarchy_splits_grouped_and_weight_only_kernels():
    assert issubclass(TorchQuantLinear, GPTQQuantLinear)
    assert issubclass(AwqTorchQuantLinear, AWQuantLinear)
    assert issubclass(AwqTorchQuantLinear, GroupedQuantLinear)
    assert not issubclass(TorchFP8QuantLinear, GroupedQuantLinear)
    assert not issubclass(GGUFTorchQuantLinear, GroupedQuantLinear)


def test_awq_hybrid_kernels_do_not_inherit_gptq_only_base_state():
    for cls in (TorchFusedAwqQuantLinear, HFKernelAwqLinear):
        assert issubclass(cls, AWQuantLinear)
        assert not issubclass(cls, GPTQQuantLinear)
        assert not hasattr(cls, "qzero_format")

    assert issubclass(AWQBitBlasKernel, GroupedQuantLinear)
    assert not issubclass(AWQBitBlasKernel, GPTQQuantLinear)
    assert not hasattr(AWQBitBlasKernel, "qzero_format")


def test_base_quant_linear_init_is_method_agnostic():
    params = inspect.signature(BaseQuantLinear.__init__).parameters

    assert "group_size" not in params
    assert "desc_act" not in params
    assert "sym" not in params
    assert "pack_dtype" not in params


def test_grouped_kernels_keep_grouped_runtime_state():
    gptq = TorchQuantLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=32,
        bias=False,
        register_buffers=False,
    )
    awq = AwqTorchQuantLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=32,
        bias=False,
        register_buffers=False,
    )

    for module in (gptq, awq):
        assert module.group_size == 32
        assert module.desc_act is False
        assert module.sym is True
        assert module.pack_dtype == torch.int32
        assert module.smooth_block_size() == 32

    assert gptq.qzero_format() == 1
    assert not hasattr(awq, "_qzeros_format")


def test_weight_only_kernels_do_not_store_grouped_runtime_state():
    ok, err = TorchFP8QuantLinear.validate_once()
    if not ok:
        pytest.skip(f"FP8 unavailable: {err}")

    fp8 = TorchFP8QuantLinear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=32,
        bias=False,
        register_buffers=False,
        weight_scale_method="block",
        weight_block_size=(16, 16),
    )
    gguf = GGUFTorchQuantLinear(
        bits="q4_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=32,
        bias=False,
        register_buffers=False,
    )

    for module in (fp8, gguf):
        assert not hasattr(module, "group_size")
        assert not hasattr(module, "desc_act")
        assert not hasattr(module, "sym")
        assert not hasattr(module, "pack_dtype")

    assert fp8.smooth_block_size() == 16
    assert gguf.smooth_block_size() == gguf.gguf_block_size


def test_weight_only_kernels_do_not_declare_grouped_support_metadata():
    for cls in (TorchFP8QuantLinear, GGUFTorchQuantLinear):
        assert not hasattr(cls, "SUPPORTS_GROUP_SIZE")
        assert not hasattr(cls, "SUPPORTS_DESC_ACT")
        assert not hasattr(cls, "SUPPORTS_SYM")
