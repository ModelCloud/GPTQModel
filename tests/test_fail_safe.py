import unittest

import torch
import torch.nn as nn

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


class TestGPTQHessianSimilarity(unittest.TestCase):
    """
    This test verifies that Hessian-based GPTQ produces quantized weights
    that remain numerically close to RTN fallback, while still introducing
    minimal corrective differences.

    The test intentionally checks *similarity*, not equality.
    """

    def _run_test(self, device: str):
        torch.manual_seed(0)

        # Large dimensions are intentionally used to:
        # - avoid degenerate small-layer behavior
        # - amplify Hessian effects in a stable manner
        in_features = 1024
        out_features = 2048
        batch = 4
        seq = 16

        inp = torch.randn(batch, seq, in_features, device=device)
        linear = nn.Linear(in_features, out_features, bias=False).to(device)

        qcfg = QuantizeConfig(
            bits=4,
            group_size=128,
            failsafe_with_rtn=False,
        )

        # ============================================================
        # Hessian-based GPTQ (use_hessian = True)
        # ============================================================
        gptq_h = GPTQ(linear, qcfg)
        gptq_h.quantizer.configure(perchannel=True)
        gptq_h.failsafe_with_rtn = False

        # Accumulate Hessian via the public API
        gptq_h.add_batch(inp, None)

        Q_h, scale_h, zero_h, gidx_h, *_ = gptq_h.quantize()

        # ============================================================
        # RTN fallback (use_hessian = False)
        # ============================================================
        qcfg.failsafe_with_rtn=True
        gptq_r = GPTQ(linear, qcfg)
        gptq_r.quantizer.configure(perchannel=True)
        gptq_r.failsafe_with_rtn = True

        # IMPORTANT:
        # We intentionally do NOT call add_batch here,
        # so nsamples == 0 and the code falls back to RTN-style quantization.
        Q_r, scale_r, zero_r, gidx_r, *_ = gptq_r.quantize()

        # ============================================================
        # Assertions
        # ============================================================

        # ------------------------------------------------------------
        # 1. Quantized weights should remain numerically close
        #
        # GPTQ is designed to minimally perturb RTN quantization
        # while reducing global error via Hessian-based correction.
        # ------------------------------------------------------------
        quant_step = torch.mean(scale_r).item()

        close_mask = torch.isclose(Q_h, Q_r, atol=quant_step)
        close_ratio = close_mask.float().mean().item()

        self.assertGreater(
            close_ratio,
            0.95,
            msg="More than 95% of quantized values should remain within one quantization bin",
        )

        # ------------------------------------------------------------
        # 2. Quantized weights must NOT be exactly identical
        #
        # At least some discrete corrections are expected when
        # Hessian-based error propagation is active.
        # ------------------------------------------------------------
        self.assertFalse(
            torch.equal(Q_h, Q_r),
            msg="Quantized weights should not be exactly identical",
        )

        self.assertGreater(
            torch.count_nonzero(Q_h != Q_r).item(),
            0,
            msg="At least one quantized element should differ due to Hessian correction",
        )

        # ------------------------------------------------------------
        # 3. Group indices must be identical
        #
        # Group assignment depends only on group_size and ordering,
        # and must NOT be affected by Hessian usage.
        # ------------------------------------------------------------
        self.assertTrue(
            torch.equal(gidx_h, gidx_r),
            msg="Group indices (g_idx) must be identical regardless of Hessian usage",
        )

        # ------------------------------------------------------------
        # 4. Scale tensors: shape must match and values must remain stable
        #
        # Scale is allowed to change slightly due to redistribution
        # of weights, but should remain within a small relative bound.
        # ------------------------------------------------------------
        self.assertEqual(scale_h.shape, scale_r.shape)
        self.assertEqual(zero_h.shape, zero_r.shape)

        scale_rel_diff = torch.mean(
            torch.abs(scale_h - scale_r) / scale_r
        ).item()

        self.assertLess(
            scale_rel_diff,
            0.05,
            msg="Relative scale deviation should remain below 5%",
        )

        # ------------------------------------------------------------
        # 5. Zero-points may shift slightly, but the shift must be bounded
        #
        # A bounded zero-point shift corresponds to less than one
        # quantization bin and is expected behavior.
        # ------------------------------------------------------------
        zero_diff = torch.mean(torch.abs(zero_h - zero_r)).item()

        self.assertLess(
            zero_diff * quant_step,
            quant_step,
            msg="Zero-point shift should correspond to less than one quantization bin",
        )

    def test_cpu(self):
        self._run_test("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda(self):
        self._run_test("cuda")
