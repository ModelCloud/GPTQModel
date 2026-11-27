<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Regression fixtures from the Nov 25–27 audit

26 PRs landed on main in the 48 hours ending Nov 27 at 01:00 UTC. The
[previous audit](2026-11-26-regressions.md) covers the overlapping feature
fixes. [Audited revision](https://github.com/ModelCloud/GPTQModel/commit/ccc7868a12043be67de16e7b06eb1d9954a6d9e9)

## A legal initial fit can lead to an illegal refinement

For both scalar GSQ refiners, start with an FP16 INT4 row at code 1, scale
8192 and zero point 8. The row replays as −57344. Set the FP32 target to
−62000. Code 0 decodes to −65536 and has **lower squared error**, but becomes
−inf in FP16. The packer can recover 0 from that −inf and report an improved
finite score. Assert the two competing errors and the overflow before running
the search. Require the original finite row back. In a second case start
at −65504. Regenerating that finite initializer from its own code would lose it.
[Fix #6083](https://github.com/ModelCloud/GPTQModel/pull/6083) ·
[scalar GSQ tests](../../../../tests/test_gsq_scalar.py)

Pair rejection with a row that *can* improve. For learned AWQ scales, pack
the returned tensors, unpack the integer lanes and compute the objective with
the stored scales and offsets for that layout. Compare it with `after`.
This catches a search that scores scales before their final rounding, and a
search that always returns the initializer. Reuse the same fixture with two
row budgets. [Earlier scale fix #4162](https://github.com/ModelCloud/GPTQModel/pull/4162) ·
[packed-export tests](../../../../tests/test_gsq_awq_code_recovery.py) ·
[GSQ acceptance proposal #6092](https://github.com/ModelCloud/GPTQModel/issues/6092)

## Follow the grid into each representation that consumes it

| Path | Trigger and observable result |
| --- | --- |
| AWQ asymmetric 8-bit | Put −0.134765625 and 1.7109375 in the first two columns of an otherwise zero group. The quantizer selects 255 for the second value. Recovering codes from BF16 replay gives 254. Compute expected packed words from the original grid, then call `apply_quant` and `pack_module` and compare the actual words. FP16 retains 255. Use both Linear and rectangular Conv1D to exercise the transpose at handoff. [AWQ #6085](https://github.com/ModelCloud/GPTQModel/pull/6085) · [processor and packer tests](../../../../tests/test_awq_per_module_bits.py) · [earlier GPTAQ/FOEM fix #5872](https://github.com/ModelCloud/GPTQModel/pull/5872) |
| QQQ and Paro FP16 | Extend the [RTN/GPTQ endpoint fixture](2026-11-24-regressions.md) through QQQ's grouped solve, channel-scale normalization, pack and one-hot forward. Check the packed codes, nonzero group scales and finite output. For Paro, supply learned scale 8192 and weight −65504 directly to eager and straight-through pseudo-quantization. With signed INT4, −8 would reconstruct −65536. An ordinary random row seldom selects that endpoint. [QQQ #6078](https://github.com/ModelCloud/GPTQModel/pull/6078) · [QQQ tests](../../../../tests/test_qqq_quantizer.py) · [Paro #6075](https://github.com/ModelCloud/GPTQModel/pull/6075) · [Paro tests](../../../../tests/test_paro.py) |
| Sign-only GGUF | In each of Q1_0 and Q1_0_g128 set two lanes to ±1.28e−7 and the rest to zero. The scale is the **mean** magnitude. Check the unclamped FP16 mean rounds to zero, read the repaired header as 2⁻²⁴ and verify both signs in the decode. A uniformly tiny block misses dilution by the other lanes. [Earlier simple-scale repair #6027](https://github.com/ModelCloud/GPTQModel/pull/6027) · [sign-only fix #6074](https://github.com/ModelCloud/GPTQModel/pull/6074) · [GGUF tests](../../../../tests/test_gguf_all_formats.py) |

## Exercise the grouped solver before testing its fallback

G2PTQ with samples but no block statistics can complete via plain GPTQ.
Use `hf_quantize` to check the missing-statistics warning, and keep a
zero-sample control where that warning would be misleading. For recovery,
install grouped Hessians and a weight gradient, run real compensation, and
inject NaN into its loss terms. With fallback disabled expect the NaN error.
With recovery configured require mock retry and a finite result. For a block
OOM, fail after a column has been processed on the accelerator and compare
the recovered weight, scales, zeros and group indices with a fresh CPU solve.
[Missing statistics #6071](https://github.com/ModelCloud/GPTQModel/pull/6071) ·
[grouped recovery #6077](https://github.com/ModelCloud/GPTQModel/pull/6077) ·
[solver tests](../../../../tests/test_g2ptq_processor.py)

**Historical CPU check.** On the [revision before #6083 and #6085](https://github.com/ModelCloud/GPTQModel/commit/d3bd7873776bafa625b9ea5a822541f210c2364c),
both GSQ endpoint tests failed on nonfinite replay. All four BF16 AWQ cases
failed on the packed words, while all four FP16 controls passed. The same ten
tests passed on the audited revision.
