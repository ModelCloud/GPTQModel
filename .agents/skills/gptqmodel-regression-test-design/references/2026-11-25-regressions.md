<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Regression fixtures from the Nov 23–25 audit

37 PRs merged into main between 2026-11-23 01:02 and 2026-11-25 01:02 UTC,
including two earlier skill updates. The [Nov 24 audit](2026-11-24-regressions.md)
covers the overlapping finite-grid and candidate-ranking fixes.
[Audited revision](https://github.com/ModelCloud/GPTQModel/commit/376196a547e6954ce75cfef95d2d49b63df9b60d)

## Importing a fixture can change another test's kernel

In a fresh process, resolve `(BACKEND.GPTQ_TORCH, METHOD.GPTQ, FORMAT.GPTQ)`
to `TorchLinear`. Import the qzeros test module, then repeat the lookup in the
same process. Before #6041 the import alone added `_V1RuntimeTorchLinear` as
a second match. The AUTO map already filtered out-of-package classes, so an
AUTO-only assertion missed this. Extend the existing fresh-process registry
test when changing discovery. Keep test-only subclasses out of selection even
when the fixture needs to declare the same formats as the production class.
[Fixture fix #6041](https://github.com/ModelCloud/GPTQModel/pull/6041) ·
[qzeros fixture](../../../../tests/test_qzeros_assign_load_4304.py) ·
[registry tests](../../../../tests/test_lazy_kernel_registry_import.py) ·
[proposed membership contract #6057](https://github.com/ModelCloud/GPTQModel/issues/6057)

A CPU oracle can also depend on which accelerator packages the host has
installed. For a Nemotron-H AWQ fold, run the small model's forward inside
`eager_torch_ssm_kernels()` and compare its logits before and after the fold.
For a CPU checkpoint round trip, pass `device="cpu"` on each public load
because `BACKEND.TORCH` alone permits accelerator placement. Keep native dispatch
tests outside the eager reference context.
[SSM fixture #6039](https://github.com/ModelCloud/GPTQModel/pull/6039) ·
[fold test](../../../../tests/module_tree/test_nemotron_h_local_port.py) ·
[CPU load fixtures #6040](https://github.com/ModelCloud/GPTQModel/pull/6040) ·
[Qwen2.5-VL round trips](../../../../tests/test_qwen2_5_vl_local_port.py)

## Packed fields that a scale-at-byte-zero test misses

- **Overflow.** The Q4_0/Q8_0 guard preceded the all-format check. Build the
  field inventory independently of the validator and account for every
  supported format, including those with no FP16 field. In IQ1_M put nibbles
  `0, 0, C, 7` into the high four bits of the four trailing scale words of
  block 1. They encode FP16 infinity. Require an error naming IQ1_M, its
  scale and block 1 of 2. Encode `0x3c00` in those positions as the accepted
  control. Exercise export on an out-of-range source and assert that the
  destination never acquires the rejected qweight.
  [First guard #5999](https://github.com/ModelCloud/GPTQModel/pull/5999) ·
  [all-format fix #6025](https://github.com/ModelCloud/GPTQModel/pull/6025) ·
  [header and export tests](../../../../tests/test_gguf_all_formats.py)
- **Underflow.** Pack 32 Q8_0 values of magnitude `1e-7` with opposite signs
  in the first two lanes. Read the first two bytes as little-endian FP16 and
  multiply by the signed code bytes. Require a negative and a positive result.
  A finite all-zero decode passed the overflow guard. Include an actual zero
  block, a normal block whose bytes survive a neighbouring repair, and a
  `1e-8` block whose codes all round to zero and must be rejected.
  [Repair #6027](https://github.com/ModelCloud/GPTQModel/pull/6027) ·
  [mixed-block and decode tests](../../../../tests/test_gguf_all_formats.py)

## Run the numerical fixture through every implementation of the decision

The NVFP4 replay fix left the Triton activation packer ranking candidates in
absolute FP32 coordinates. Scale the BF16 block and its global scale by
`2**124`. Compare all packed codes and swizzled scale bytes with replay,
then check the chosen local scale against the recipe's independent FP64
objective. In QQQ, the generic affine span fix missed the method's own fit.
Use BF16 endpoints `±2**127` and an ordinary-range control across the main
fit, MSE grid and fallback. Their final scales can fit even when subtracting
the endpoints in FP32 overflows.
[Replay #6008](https://github.com/ModelCloud/GPTQModel/pull/6008) ·
[Triton #6028](https://github.com/ModelCloud/GPTQModel/pull/6028) ·
[native parity and oracle tests](../../../../tests/kernels/test_w4a_nvfp4_triton_scale_invariance.py) ·
[QQQ #6024](https://github.com/ModelCloud/GPTQModel/pull/6024) ·
[QQQ tests](../../../../tests/test_qqq_quantizer.py)

**Historical check.** On the [parent of #6027](https://github.com/ModelCloud/GPTQModel/commit/fb1af1bc77ec74c2f467f20af15b57c5d30a46a5)
the tiny Q8_0 block decoded to signed zeros, the IQ1_M infinity field passed
validation, and explicit lookup raised after importing the qzeros fixture.
The unit-amplitude Q8_0 and finite IQ1_M controls passed. All five CPU probes
passed on the audited revision.
