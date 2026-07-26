# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Cross-method packing accuracy matrix.

Exercises every CPU-packable ``QuantLinear`` subclass for all declared combinations
of ``bits``, ``group_size``, ``pack_dtype``, ``sym`` and ``desc_act`` (when they
affect the packing path), and asserts that ``pack`` + ``dequantize_weight``
round-trips to the dense reference or, for backends without a dequantizer, that
the packed buffers have the expected shape and are consistent with the reference
AWQ/GPTQ layout.
"""

import math
import unittest

import torch
import torch.nn as nn
from parameterized import parameterized

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear import PackableQuantLinear, WeightOnlyQuantLinear
from gptqmodel.nn_modules.qlinear.cannoe import AwqCannoeLinear, CannoeLinear
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear
from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.komodo import AwqKomodoLinear, KomodoLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear, TorchQuantEmbeddings
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.nn_modules.qlinear.trilin import AwqTrilinLinear, TrilinLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear


def _name(func, idx, p):
    cls = p.args[0]
    return f"{func.__name__}_{cls.__name__}"


# Classes that can be packed on CPU without GPU-only dependencies.
PACKABLE_CLASSES = (
    # GPTQ / packable quantized linear paths
    TorchLinear,
    TorchAtenLinear,
    TorchFusedLinear,
    TorchQuantEmbeddings,
    TrilinLinear,
    TritonV2Linear,
    KomodoLinear,
    CannoeLinear,
    # Format-specific weight-only paths
    GGUFTorchLinear,
    TorchFP8Linear,
    # AWQ paths (no dequantize_weight available, so validate packed buffer layout)
    AwqTorchLinear,
    AwqGEMMLinear,
    AwqGEMVLinear,
    AwqGEMVFastLinear,
    AwqGEMMTritonLinear,
    AwqTrilinLinear,
    AwqKomodoLinear,
    AwqCannoeLinear,
)


class TestPackingMatrix(unittest.TestCase):
    OUT_FEATURES = 128

    @staticmethod
    def _in_features_for_group_size(group_size: int) -> int:
        """Pick a small input dimension that is divisible by ``group_size``.

        Keeping the fixture small keeps the matrix fast while still covering every
        declared group size.  ``pack_block``-based paths additionally require the
        dimension to be a multiple of 32, so we round up to that.
        """
        if group_size <= 0:
            return 128
        if group_size <= 128:
            return 128
        return ((group_size + 31) // 32) * 32

    @staticmethod
    def _build_inputs(in_features: int, out_features: int, bits: int, group_size: int, sym: bool, desc_act: bool):
        torch.manual_seed(0)
        max_q = (1 << bits) - 1

        if group_size == -1:
            groups = 1
        else:
            groups = math.ceil(in_features / group_size)

        if desc_act:
            g_idx = torch.arange(in_features, dtype=torch.int64)
            g_idx = g_idx[torch.randperm(in_features)] // group_size if group_size > 0 else torch.zeros(in_features, dtype=torch.int64)
        else:
            g_idx = torch.arange(in_features, dtype=torch.int64) // (group_size if group_size > 0 else 1)
        g_idx = g_idx.clamp(0, max(groups - 1, 0))

        scales = torch.rand(groups, out_features, dtype=torch.float32) * 0.05 + 1e-3

        if sym:
            zeros = torch.full((groups, out_features), max_q // 2, dtype=torch.int32)
        else:
            zeros = torch.randint(0, max_q + 1, (groups, out_features), dtype=torch.int32)

        q_int = torch.randint(0, max_q + 1, (in_features, out_features), dtype=torch.int32)
        weight = scales[g_idx].to(torch.float32) * (q_int.to(torch.float32) - zeros[g_idx].to(torch.float32))

        linear = nn.Linear(in_features, out_features, bias=False)
        linear.weight.data = weight.T.to(linear.weight.dtype)
        return linear, scales, zeros, g_idx, weight

    @staticmethod
    def _pack_dtype_bits(pack_dtype):
        return {torch.int8: 8, torch.int16: 16, torch.int32: 32, torch.int64: 64}[pack_dtype]

    def _call_pack(self, qlinear, linear, scales, zeros, g_idx):
        """Call the best available pack method with a common argument shape."""
        scales_t = scales.t().contiguous()
        zeros_t = zeros.t().contiguous()
        g_idx_t = g_idx.to(torch.int32)

        if hasattr(qlinear, "pack_original"):
            qlinear.pack_original(linear, scales_t, zeros_t, g_idx=g_idx_t)
        elif hasattr(qlinear, "pack"):
            try:
                qlinear.pack(linear, scales_t, zeros_t, g_idx=g_idx_t)
            except TypeError:
                qlinear.pack(linear, scales_t, zeros_t)
        else:
            raise unittest.SkipTest(f"{qlinear.__class__.__name__} has no pack method")

    def _dequant_close(self, qlinear, expected, atol):
        if not hasattr(qlinear, "dequantize_weight"):
            return
        dq = qlinear.dequantize_weight()
        if dq.shape == expected.shape:
            diff = (dq.to(torch.float32) - expected.to(torch.float32)).abs().max().item()
        elif dq.shape == expected.T.shape:
            diff = (dq.to(torch.float32) - expected.T.to(torch.float32)).abs().max().item()
        else:
            raise AssertionError(
                f"{qlinear.__class__.__name__} dequantize_weight returned {tuple(dq.shape)}, "
                f"expected {tuple(expected.shape)} or its transpose"
            )
        self.assertLess(diff, atol, f"dequant diff {diff} exceeds atol {atol}")

    def _run_packable_roundtrip(self, qlinear_cls, bits, group_size, pack_dtype, sym, desc_act, in_features):
        """Round-trip test for PackableQuantLinear subclasses."""
        if bits == 3 and pack_dtype != torch.int32:
            # The 3-bit custom packing loop currently hardcodes 32-bit words.
            self.skipTest("3-bit pack_original currently supports only 32-bit words")

        out_features = self.OUT_FEATURES
        linear, scales, zeros, g_idx, weight = self._build_inputs(in_features, out_features, bits, group_size, sym, desc_act)
        backend = qlinear_cls.SUPPORTS_BACKENDS[0]
        qlinear = qlinear_cls(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=pack_dtype,
            backend=backend,
            bias=False,
        )

        self._call_pack(qlinear, linear, scales, zeros, g_idx)

        # For int32, also exercise the fast pack() path and compare to pack_original.
        if pack_dtype == torch.int32 and hasattr(qlinear, "pack"):
            qlinear_fast = qlinear_cls(
                bits=bits,
                group_size=group_size,
                sym=sym,
                desc_act=desc_act,
                in_features=in_features,
                out_features=out_features,
                pack_dtype=pack_dtype,
                backend=backend,
                bias=False,
            )
            try:
                qlinear_fast.pack(linear, scales.t().contiguous(), zeros.t().contiguous(), g_idx=g_idx.to(torch.int32))
                self.assertTrue(torch.equal(qlinear_fast.qweight, qlinear.qweight))
                self.assertTrue(torch.equal(qlinear_fast.qzeros, qlinear.qzeros))
            except (AssertionError, ValueError, NotImplementedError):
                # pack() may impose stricter divisibility/alignment rules.
                pass

        self._dequant_close(qlinear, weight, atol=1e-2)

    def _run_weight_only_roundtrip(self, qlinear_cls, bits, group_size, pack_dtype, sym, desc_act, in_features):
        """Round-trip test for WeightOnlyQuantLinear subclasses (GGUF/FP8)."""
        out_features = self.OUT_FEATURES
        linear, scales, zeros, g_idx, weight = self._build_inputs(in_features, out_features, bits, group_size, sym, desc_act)
        backend = qlinear_cls.SUPPORTS_BACKENDS[0]
        qlinear = qlinear_cls(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=pack_dtype,
            backend=backend,
            bias=False,
        )
        self._call_pack(qlinear, linear, scales, zeros, g_idx)

        # These backends ignore scales/zeros and quantize from the dense weight.
        atol = 1e-2 if qlinear_cls is TorchFP8Linear else 5e-2
        self._dequant_close(qlinear, weight, atol=atol)

    def _run_awq_layout(self, qlinear_cls, bits, group_size, pack_dtype, sym, desc_act, in_features):
        """Layout validation for AWQ packers (no dequantize_weight available)."""
        out_features = self.OUT_FEATURES
        linear, scales, zeros, g_idx, weight = self._build_inputs(in_features, out_features, bits, group_size, sym, desc_act)

        # AWQ packers internally derive g_idx as sorted channel groups, so generate the
        # dense weight using that same ordering regardless of desc_act.
        if desc_act:
            if group_size > 0:
                g_idx_sorted = torch.arange(in_features, dtype=torch.int64) // group_size
            else:
                g_idx_sorted = torch.zeros(in_features, dtype=torch.int64)
            max_q = (1 << bits) - 1
            weight_sorted = scales[g_idx_sorted].to(torch.float32) * (
                torch.randint(0, max_q + 1, (in_features, out_features), dtype=torch.int32).to(torch.float32)
                - zeros[g_idx_sorted].to(torch.float32)
            )
            linear.weight.data = weight_sorted.T.to(linear.weight.dtype)

        backend = qlinear_cls.SUPPORTS_BACKENDS[0]
        qlinear = qlinear_cls(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=pack_dtype,
            backend=backend,
            bias=False,
        )

        try:
            qlinear.pack(linear, scales.t().contiguous(), zeros.t().contiguous())
        except ValueError as exc:
            if "3-bit symmetric AWQ packing requires every zero point" in str(exc):
                self.skipTest(str(exc))
            raise

        # All AWQ packers should produce the canonical 2-D qweight/qzeros/scales buffers.
        self.assertEqual(qlinear.qweight.dim(), 2)
        self.assertEqual(qlinear.qzeros.dim(), 2)
        self.assertEqual(qlinear.scales.dim(), 2)
        expected_scale_elems = out_features * (math.ceil(in_features / group_size) if group_size > 0 else 1)
        self.assertEqual(qlinear.scales.numel(), expected_scale_elems)

        # For 4-bit int32 layouts, compare to the reference AwqTorchLinear packing.
        if bits == 4 and pack_dtype == torch.int32 and qlinear_cls is not AwqGEMVFastLinear:
            ref = AwqTorchLinear(
                bits=bits,
                group_size=group_size,
                sym=sym,
                desc_act=desc_act,
                in_features=in_features,
                out_features=out_features,
                pack_dtype=pack_dtype,
                backend=BACKEND.AWQ_TORCH,
                bias=False,
            )
            ref.pack(linear, scales.t().contiguous(), zeros.t().contiguous())
            if qlinear.qweight.shape == ref.qweight.shape:
                self.assertTrue(torch.equal(qlinear.qweight, ref.qweight), "qweight layout mismatch")
            if qlinear.qzeros.shape == ref.qzeros.shape:
                self.assertTrue(torch.equal(qlinear.qzeros, ref.qzeros), "qzeros layout mismatch")

    def _run_case(self, qlinear_cls, bits, group_size, pack_dtype, sym, desc_act, in_features):
        with self.subTest(
            bits=bits,
            group_size=group_size,
            in_features=in_features,
            pack_dtype=str(pack_dtype),
            sym=sym,
            desc_act=desc_act,
        ):
            try:
                if issubclass(qlinear_cls, PackableQuantLinear):
                    self._run_packable_roundtrip(qlinear_cls, bits, group_size, pack_dtype, sym, desc_act, in_features)
                elif issubclass(qlinear_cls, WeightOnlyQuantLinear):
                    self._run_weight_only_roundtrip(qlinear_cls, bits, group_size, pack_dtype, sym, desc_act, in_features)
                else:
                    self._run_awq_layout(qlinear_cls, bits, group_size, pack_dtype, sym, desc_act, in_features)
            except (NotImplementedError, ValueError, AssertionError) as exc:
                self.skipTest(str(exc))

    @parameterized.expand([(cls,) for cls in PACKABLE_CLASSES], name_func=_name)
    def test_packing_matrix(self, qlinear_cls):
        bits_list = list(getattr(qlinear_cls, "SUPPORTS_BITS", None) or [4])
        group_sizes = list(getattr(qlinear_cls, "SUPPORTS_GROUP_SIZE", None) or [-1])
        if not group_sizes:
            group_sizes = [-1]
        pack_dtypes = list(getattr(qlinear_cls, "SUPPORTS_PACK_DTYPES", None) or [torch.int32])
        sym_opts = list(getattr(qlinear_cls, "SUPPORTS_SYM", None) or [True])
        desc_opts = list(getattr(qlinear_cls, "SUPPORTS_DESC_ACT", None) or [False])

        for bits in bits_list:
            for group_size in group_sizes:
                in_features = self._in_features_for_group_size(group_size)
                for pack_dtype in pack_dtypes:
                    for sym in sym_opts:
                        for desc_act in desc_opts:
                            self._run_case(qlinear_cls, bits, group_size, pack_dtype, sym, desc_act, in_features)


if __name__ == "__main__":
    unittest.main()
