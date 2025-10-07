# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402


# isort: off
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
# isort: on
from parameterized import parameterized  # noqa: E402

from gptqmodel import BACKEND  # noqa: E402
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear  # noqa: E402


def gen_quant(k: int, n: int, groupsize: int, bits: int):
    """
    Generalized version of gen_quant4 for bits ∈ {2,3,4,8}.
    Produces a Linear layer with dequantized reference weights and per-column scales.
    """
    assert bits in (2, 3, 4, 8), "bits must be one of {2,3,4,8}"
    maxq = (1 << bits) - 1
    w = torch.randn((k, n), dtype=torch.half, device="cpu")
    original_w = w.clone()

    # Grouped view for scale computation when groupsize != -1
    if groupsize != -1:
        w_g = w.reshape((-1, groupsize, n)).permute(1, 0, 2).reshape((groupsize, -1))
    else:
        w_g = w

    # Per-column max-abs scale (same logic as original, generalized for bits)
    s = torch.max(torch.abs(w_g), 0, keepdim=True)[0]
    s *= 2.0 / maxq

    # Quantize to integers
    q = torch.round(w_g / s).int()
    # Unsigned shift & clamp
    q += (maxq + 1) // 2
    q = torch.clamp(q, 0, maxq)

    # Dequant reference
    ref = (q - (maxq + 1) // 2).half() * s

    if groupsize != -1:
        def _reshape(x):
            x = x.reshape((groupsize, -1, n)).permute(1, 0, 2).reshape((k, n)).contiguous()
            return x
        ref = _reshape(ref)
        q = _reshape(q)

    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(k, n, bias=False)
    linear.weight.data = ref.t()  # Linear expects [out,in] = [n,k]

    return original_w, linear, s


class TestRepacking(unittest.TestCase):
    QLINEAR_DICT = {
        # BACKEND.TRITON: TritonV2QuantLinear,
        BACKEND.TORCH: TorchQuantLinear,
        # BACKEND.TORCH_FUSED: TorchFusedQuantLinear,
    }

    # Dimensions (match your original scale)
    k = 2048
    n = 1024 * 100
    pack_dtype = torch.int32

    # Grid over bits and group sizes
    TEST_GRID = []
    for bits in (2, 3, 4, 8):
        for gs in (32, 64, 128):
            for backend in QLINEAR_DICT:
                TEST_GRID.append((bits, gs, backend))

    def _make_gidx(self, k: int, group_size: int) -> torch.Tensor:
        """Map each input channel to its group id, length k, dtype long."""
        return (torch.arange(k, dtype=torch.long) // group_size).contiguous()

    def _pack_one(self, qlinearCls, backend, bits, group_size, linear, s, zeros, g_idx):
        """Instantiate a quant linear and pack with provided tensors."""
        qlinear = qlinearCls(
            bits=bits,
            group_size=group_size,
            sym=True,
            desc_act=True,
            in_features=self.k,
            out_features=self.n,
            pack_dtype=self.pack_dtype,
            backend=backend,
            bias=False,
        )
        # Note: pack expects scales/zeros transposed in your code path
        qlinear.pack_block(linear, s.T, zeros.T, g_idx=g_idx.to(torch.int32))
        return qlinear

    @parameterized.expand(TEST_GRID)
    def test_packing_variants(self, bits: int, group_size: int, backend):
        # Build per-case inputs
        _, linear, s = gen_quant(self.k, self.n, group_size, bits)

        # zeros per original shape: [k//group_size, n], fill with midpoint
        maxq = (1 << bits) - 1
        mid = (maxq + 1) // 2
        zeros = torch.full((self.k // group_size, self.n), mid, dtype=torch.int32)
        g_idx = self._make_gidx(self.k, group_size)

        # Primary backend under test
        PrimaryQL = self.QLINEAR_DICT[backend]
        try:
            primary = self._pack_one(PrimaryQL, backend, bits, group_size, linear, s, zeros, g_idx)
            primary.post_init()
        except (NotImplementedError, ValueError) as e:
            self.skipTest(f"{backend} does not support bits={bits}, group_size={group_size}: {e}")

        # Torch reference packer (compare against Torch CPU packer)
        try:
            torch_linear = self._pack_one(TorchQuantLinear, BACKEND.TORCH, bits, group_size, linear, s, zeros, g_idx)
            torch_linear.post_init()
        except (NotImplementedError, ValueError) as e:
            self.skipTest(f"Torch backend does not support bits={bits}, group_size={group_size}: {e}")

        # Generic dequant (works for all bits): compare to the reference Linear weight
        # NOTE: dequantize_weight() returns shape [in, out], so transpose to [out, in]
        w_primary = primary.dequantize_weight().T.to(torch.float16)
        self.assertTrue(torch.equal(w_primary, linear.weight), "Primary backend dequant != reference weight")

        w_torch = torch_linear.dequantize_weight().T.to(torch.float16)
        self.assertTrue(torch.equal(w_torch, linear.weight), "Torch backend dequant != reference weight")

        # Cross-backend consistency checks for all bit-widths
        self.assertTrue(torch.allclose(primary.qweight, torch_linear.qweight), "qweight mismatch")
        self.assertTrue(torch.allclose(primary.scales,  torch_linear.scales),  "scales mismatch")
        self.assertTrue(torch.allclose(primary.qzeros,  torch_linear.qzeros),  "qzeros mismatch")


if __name__ == "__main__":
    unittest.main()
