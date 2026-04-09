# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
import torch

import gptqmodel.nn_modules.qlinear.bitblas as bitblas_module
from gptqmodel.nn_modules.qlinear import (
    AWQuantLinear,
    BaseQuantLinear,
    GPTQQuantLinear,
    GroupedQuantLinear,
    PackedGroupedQuantLinear,
)
from gptqmodel.nn_modules.qlinear.bitblas import BitblasLinear
from gptqmodel.nn_modules.qlinear.bitblas_awq import AWQBitBlasKernel
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.qqq import QQQLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear


def test_quant_linear_hierarchy_splits_grouped_and_weight_only_kernels():
    assert issubclass(TorchLinear, GPTQQuantLinear)
    assert issubclass(TorchLinear, PackedGroupedQuantLinear)
    assert issubclass(AwqTorchLinear, AWQuantLinear)
    assert issubclass(AwqTorchLinear, PackedGroupedQuantLinear)
    assert issubclass(AwqTorchLinear, GroupedQuantLinear)
    assert not issubclass(TorchFP8Linear, GroupedQuantLinear)
    assert not issubclass(GGUFTorchLinear, GroupedQuantLinear)


def test_awq_hybrid_kernels_do_not_inherit_gptq_only_base_state():
    for cls in (TorchFusedAwqLinear, TorchAtenAwqLinear):
        assert issubclass(cls, AWQuantLinear)
        assert not issubclass(cls, GPTQQuantLinear)
        assert not hasattr(cls, "qzero_format")

    assert issubclass(AWQBitBlasKernel, GroupedQuantLinear)
    assert not issubclass(AWQBitBlasKernel, GPTQQuantLinear)
    assert not issubclass(AWQBitBlasKernel, PackedGroupedQuantLinear)
    assert not issubclass(AWQBitBlasKernel, BitblasLinear)
    assert not hasattr(AWQBitBlasKernel, "qzero_format")
    assert not hasattr(AWQBitBlasKernel, "repack_from_gptq")


def test_bitblas_gptq_kernel_keeps_gptq_only_repack_surface():
    assert issubclass(BitblasLinear, GroupedQuantLinear)
    assert not issubclass(BitblasLinear, PackedGroupedQuantLinear)
    assert hasattr(BitblasLinear, "repack_from_gptq")
    assert not hasattr(BitblasLinear, "repack_from_awq")


def test_base_quant_linear_init_is_method_agnostic():
    params = inspect.signature(BaseQuantLinear.__init__).parameters

    assert "group_size" not in params
    assert "desc_act" not in params
    assert "sym" not in params
    assert "pack_dtype" not in params


def test_grouped_kernels_keep_grouped_runtime_state():
    gptq = TorchLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=32,
        bias=False,
        register_buffers=False,
    )
    awq = AwqTorchLinear(
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


def _install_dummy_bitblas(monkeypatch):
    class _DummyConfig:
        A_dtype = "float16"
        W_dtype = "uint4"
        out_dtype = "float16"
        accum_dtype = "float32"
        group_size = 32

    class _DummyMatmul:
        def __init__(self, config):
            self.config = config
            self.lib = object()
            self.weight_transform = None

        @staticmethod
        def retrieve_weight_shape():
            return (1, 1)

    def _fake_get_or_create(self, config, enable_tuning):
        del enable_tuning
        return _DummyMatmul(config)

    def _fake_configure(self, infeatures, outfeatures, params_dtype, enable_tuning, bias, layout, bits):
        del infeatures, outfeatures, params_dtype, enable_tuning, bias, layout, bits
        self.bitblas_matmul = _DummyMatmul(_DummyConfig())

    monkeypatch.setattr(bitblas_module, "BITBLAS_AVAILABLE", True)
    monkeypatch.setattr(bitblas_module, "import_bitblas", lambda: None)
    monkeypatch.setattr(BitblasLinear, "_get_or_create_bitblas_operator", _fake_get_or_create)
    monkeypatch.setattr(AWQBitBlasKernel, "_get_or_create_bitblas_operator", _fake_get_or_create)
    monkeypatch.setattr(BitblasLinear, "_configure_bitblas_matmul", _fake_configure)
    monkeypatch.setattr(AWQBitBlasKernel, "_configure_bitblas_matmul", _fake_configure)
    BitblasLinear.cached_validate_once.cache_clear()
    AWQBitBlasKernel.cached_validate_once.cache_clear()


def test_grouped_nonpacked_kernels_do_not_store_packed_runtime_state(monkeypatch):
    _install_dummy_bitblas(monkeypatch)
    monkeypatch.setattr(QQQLinear, "cached_validate_once", classmethod(lambda cls: (True, None)))

    gptq_bitblas = BitblasLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=32,
        bias=False,
    )
    awq_bitblas = AWQBitBlasKernel(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=32,
        bias=False,
    )
    qqq = QQQLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=128,
        bias=False,
        register_buffers=False,
    )

    for module in (gptq_bitblas, awq_bitblas):
        assert module.group_size == 32
        assert module.desc_act is False
        assert module.sym is True
        assert module.smooth_block_size() == 32
        assert not hasattr(module, "pack_dtype")
        assert not hasattr(module, "pack_dtype_bits")
        assert not hasattr(module, "pack_factor")
        assert not hasattr(module, "pack_np_dtype")
        assert not hasattr(module, "pack_np_math_dtype")
        assert not hasattr(module, "maxq")

    assert qqq.group_size == 128
    assert qqq.desc_act is False
    assert qqq.sym is True
    assert qqq.smooth_block_size() == 128
    assert not hasattr(qqq, "pack_dtype")
    assert not hasattr(qqq, "pack_dtype_bits")
    assert not hasattr(qqq, "pack_factor")
    assert not hasattr(qqq, "pack_np_dtype")
    assert not hasattr(qqq, "pack_np_math_dtype")
    assert qqq.maxq == 15


def test_weight_only_kernels_do_not_store_grouped_runtime_state():
    ok, err = TorchFP8Linear.validate_once()
    if not ok:
        pytest.skip(f"FP8 unavailable: {err}")

    fp8 = TorchFP8Linear(
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
    gguf = GGUFTorchLinear(
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
    for cls in (TorchFP8Linear, GGUFTorchLinear):
        assert not hasattr(cls, "SUPPORTS_GROUP_SIZE")
        assert not hasattr(cls, "SUPPORTS_DESC_ACT")
        assert not hasattr(cls, "SUPPORTS_SYM")
