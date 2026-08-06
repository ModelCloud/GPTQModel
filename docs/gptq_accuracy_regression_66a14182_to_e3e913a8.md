# GPTQ math/accuracy regression analysis: 66a14182 to e3e913a8

Date: 2026-08-06 UTC

## Verdict

The range cannot be validated as free of GPTQ math/accuracy regressions. Two independent regressions were reproduced:

1. Commit `e789272d` changes the non-static, grouped, legacy-compatible GPTQ path from `blocksize=128` to
   `min(blocksize, group_size)` even when adaptive feedback is disabled. This changes `qweight`, scale, and zero
   selection for group sizes below 128. The worst micro-case increased held-out output relative RMSE by 7.52%.
2. Adaptive damping and adaptive clipping are enabled by default at tip. Across ten deterministic dense-reference
   cases, the tip default increased held-out output relative RMSE by 9.32% on average and regressed 7/10 cases. The
   worst case increased relative output RMSE by 46.62%. Adaptive clipping was the largest isolated contributor.

Packing, planar pack/unpack, Torch dequantization, and Torch forward reference tests were clean. The earliest failing
boundary is therefore quantizer/GPTQ math before packing, not serialization or inference reconstruction.

## Scope and environment

```text
+----------------------+--------------------------------------------------------------------------+
| Field                | Value                                                                    |
+----------------------+--------------------------------------------------------------------------+
| Baseline             | 66a14182868e587394c6d48ffe33c0e7a0f5f4fd                               |
| Candidate            | e3e913a813a693fe88418822f852713694d01cb2                               |
| Range size           | 56 commits; 187 files; +37,551/-990 lines                              |
| Repository state     | main, clean, tracking origin/main                                       |
| Physical GPU         | 7                                                                        |
| PCI bus / UUID       | 00000000:E4:00.0 / GPU-724ea08e-67c3-c0ce-29bb-e6c48e7dde28            |
| GPU                  | NVIDIA PG506-230, sm_80, 124 SMs, 98,304 MiB                            |
| Idle gate            | 3/3 samples: 0 MiB, 0% utilization, no foreign compute processes         |
| Driver               | 610.43.02                                                                |
| Python               | 3.14.5 free-threaded                                                     |
| PyTorch / CUDA       | 2.13.0+cu130 / 13.0                                                     |
| Transformers/Triton  | 5.15.0.dev0 / 3.7.1                                                     |
| Arithmetic reference | FP64 host recomputation from dense FP32 weights and held-out activations  |
| CUDA mapping         | CUDA_DEVICE_ORDER=PCI_BUS_ID; CUDA_VISIBLE_DEVICES set to GPU UUID        |
+----------------------+--------------------------------------------------------------------------+
```

The micro-tests are correctness tests, not warmed performance benchmarks. Timing is included for context only; first
call compilation makes it unsuitable for performance conclusions.

## Static analysis

```text
+----------+----------------------+---------------------------------------------+------------------------------+
| Severity | Area / commit        | Static finding                              | Runtime evidence             |
+----------+----------------------+---------------------------------------------+------------------------------+
| Critical | e789272d, GPTQ loop  | Non-adaptive grouped path unconditionally    | Legacy tip differs in 5/10;  |
|          |                      | uses min(blocksize, group_size). This changes| causal one-line control      |
|          |                      | sequential update partition and FP order.   | restores 10/10 bitwise parity|
+----------+----------------------+---------------------------------------------+------------------------------+
| Critical | 266eb83e, clipping   | Adaptive clipping defaults enabled and skips| 8/10 clip-only cases regress;|
|          |                      | the previous mse/scale_search path for      | worst output error +43.47%   |
|          |                      | grouped weights.                            |                              |
+----------+----------------------+---------------------------------------------+------------------------------+
| High     | 53571a36/e789272d,   | Adaptive spectral damping, online residual  | 6/10 damp-only cases regress;|
|          | damping              | feedback, and group-size prior default on.  | worst output error +13.95%   |
+----------+----------------------+---------------------------------------------+------------------------------+
| Medium   | quantizer CPU path   | Adaptive clipping diverts batched search    | CPU extension/eager parity   |
|          |                      | from the compiled scale-search contract.    | test fails                   |
+----------+----------------------+---------------------------------------------+------------------------------+
| Medium   | grouped scale search | HESSIAN/HYBRID tests expect full group      | 2 focused tests fail: actual |
|          |                      | Hessian [1,4,4], receive diagonal [1,4].    | shape [1,4]                  |
+----------+----------------------+---------------------------------------------+------------------------------+
| Low      | formatting           | git diff --check reports one new blank line | No numerical impact          |
|          |                      | at EOF in swordfish repack CUDA source.      |                              |
+----------+----------------------+---------------------------------------------+------------------------------+
```

Relevant candidate locations:

- `gptqmodel/quantization/config.py:3934-3950`: default adaptive damping/clipping configuration.
- `gptqmodel/quantization/config.py:1410-1466`: default-on damping feedback and group-size prior.
- `gptqmodel/quantization/quantizer.py:763-782`: adaptive clipping replaces existing scale search.
- `gptqmodel/quantization/gptq.py:2606-2616`: unconditional grouped block partition.

## A/B method

Ten deterministic cases used identical weights, calibration activations, held-out activations, config, and seeds at
both revisions. The matrix covers bits 2/3/4/8; groups 32/64/128/-1; symmetric/asymmetric quantization; Gaussian,
outlier, skewed, and ill-conditioned inputs; `desc_act`; and static groups. Each output is compared with dense FP64
matrix multiplication using held-out activations. KLD uses FP64 softmax distributions.

Variants:

```text
+------------+--------------------------------------------------------------------+
| Variant    | Definition                                                         |
+------------+--------------------------------------------------------------------+
| baseline   | 66a14182 defaults                                                  |
| legacy     | tip with adaptive_damping.enabled=false and clipping.enabled=false |
| default    | tip defaults                                                       |
| damp_only  | tip adaptive damping on, adaptive clipping off                     |
| clip_only  | tip adaptive damping off, adaptive clipping on                     |
+------------+--------------------------------------------------------------------+
```

## Default-to-default dense-reference results

`W-rel` is weight relative RMSE. `Y-rel` is held-out output relative RMSE. Positive delta means worse than baseline.

```text
+-------------------------+------+-------+------+---------+---------+---------+---------+---------+----------+----------+-------+-------+---------+--------+
| Case                    | Bits | Group | Sym  | Base W  | Tip W   | Base Y  | Tip Y   | Y delta | Base KLD | Tip KLD  | B top | T top | B ms    | T ms   |
+-------------------------+------+-------+------+---------+---------+---------+---------+---------+----------+----------+-------+-------+---------+--------+
| w2_g32_sym_gaussian     | 2    | 32    | yes  | 0.47464 | 0.71149 | 0.46878 | 0.68734 | +46.62% | 0.17958  | 0.39448  | 0.500 | 0.391 | 1593.66 | 385.76 |
| w3_g32_asym_outlier     | 3    | 32    | no   | 0.29220 | 0.33715 | 0.28388 | 0.32758 | +15.40% | 0.29053  | 0.41526  | 0.594 | 0.562 |   18.41 |  30.31 |
| w4_g64_sym_gaussian     | 4    | 64    | yes  | 0.14004 | 0.15268 | 0.14108 | 0.15620 | +10.72% | 0.01580  | 0.01871  | 0.875 | 0.828 |   49.07 |  28.96 |
| w4_g64_sym_outlier      | 4    | 64    | yes  | 0.32723 | 0.38514 | 0.38329 | 0.43577 | +13.69% | 0.43821  | 0.58408  | 0.531 | 0.562 |    4.33 |  29.98 |
| w4_g64_asym_skew        | 4    | 64    | no   | 0.17163 | 0.18837 | 0.17316 | 0.18999 |  +9.73% | 0.02452  | 0.03030  | 0.750 | 0.641 |    5.53 |  29.31 |
| w4_g128_sym_illcond     | 4    | 128   | yes  | 0.23655 | 0.26336 | 0.08245 | 0.08126 |  -1.44% | 0.01618  | 0.01609  | 0.875 | 0.891 |  402.80 | 291.08 |
| w4_g128_sym_desc        | 4    | 128   | yes  | 0.15074 | 0.15506 | 0.15424 | 0.15572 |  +0.97% | 0.03376  | 0.03459  | 0.844 | 0.781 |    5.31 |  51.63 |
| w4_g32_sym_static       | 4    | 32    | yes  | 0.25097 | 0.27659 | 0.24985 | 0.27653 | +10.68% | 0.25131  | 0.27930  | 0.625 | 0.578 |   39.65 |  32.49 |
| w4_tensor_sym           | 4    | -1    | yes  | 0.53393 | 0.49789 | 0.55413 | 0.51439 |  -7.17% | 1.27700  | 1.13862  | 0.453 | 0.422 |   45.75 |  44.74 |
| w8_g128_asym            | 8    | 128   | no   | 0.01263 | 0.01200 | 0.01359 | 0.01277 |  -6.02% | 0.00021  | 0.00018  | 1.000 | 0.969 |   86.77 |  50.88 |
+-------------------------+------+-------+------+---------+---------+---------+---------+---------+----------+----------+-------+-------+---------+--------+
```

## Isolation summary

```text
+-----------+--------------+----------------+-------------+----------+-----------+----------------+--------------+
| Variant   | Mean Y delta | Median Y delta | Worst delta | Improved | Regressed | Mean KLD delta | Top-1 delta  |
+-----------+--------------+----------------+-------------+----------+-----------+----------------+--------------+
| legacy    |       +1.53% |         +0.00% |      +7.52% | 1        | 4         |         +2.79% |     -0.94 pp |
| default   |       +9.32% |        +10.68% |     +46.62% | 3        | 7         |        +22.66% |     -4.22 pp |
| damp_only |       +2.58% |         +6.19% |     +13.95% | 4        | 6         |         +6.34% |     -0.16 pp |
| clip_only |       +8.29% |         +6.89% |     +43.47% | 1        | 8         |        +20.40% |     -4.38 pp |
+-----------+--------------+----------------+-------------+----------+-----------+----------------+--------------+
```

All 50 baseline/tip outputs had finite quantized weights, scales, and zeros, and source weights were not mutated.
Finiteness is therefore not sufficient to catch these accuracy regressions.

## Legacy-path causal control

A detached temporary tip worktree changed only the non-adaptive block selection back to the baseline behavior:

```text
candidate: if group_size > 0: effective_block = min(blocksize, group_size)
control:   if Hinv is None and group_size > 0: effective_block = group_size
```

```text
+-------------------------+------------------+-------------+------------+------------+
| Case                    | qweight bit-equal| scale equal | zero equal | g_idx equal|
+-------------------------+------------------+-------------+------------+------------+
| w2_g32_sym_gaussian     | yes              | yes         | yes        | yes        |
| w3_g32_asym_outlier     | yes              | yes         | yes        | yes        |
| w4_g64_sym_gaussian     | yes              | yes         | yes        | yes        |
| w4_g64_sym_outlier      | yes              | yes         | yes        | yes        |
| w4_g64_asym_skew        | yes              | yes         | yes        | yes        |
| w4_g128_sym_illcond     | yes              | yes         | yes        | yes        |
| w4_g128_sym_desc        | yes              | yes         | yes        | yes        |
| w4_g32_sym_static       | yes              | yes         | yes        | yes        |
| w4_tensor_sym           | yes              | yes         | yes        | yes        |
| w8_g128_asym            | yes              | yes         | yes        | yes        |
+-------------------------+------------------+-------------+------------+------------+
```

This establishes causality for the legacy-path divergence; it is not merely correlated with the larger range.

## Repository test results

```text
+--------------------------------------------------+---------+---------+---------+-----------------------------------------------+
| Test group                                       | Passed  | Failed  | Skipped | Notes                                         |
+--------------------------------------------------+---------+---------+---------+-----------------------------------------------+
| GPTQ + adaptive focused run (interrupted later)  | 639     | 3       | 2       | Stopped after failures in 11,268 collection  |
| Standard pack parity partial                     | 29      | 0       | 2       | 24 subtests; stopped before unrelated AWQ JIT|
| Planar + Torch reference accuracy                | 146     | 0       | 0       | Includes tiny quantize/save/load lifecycles  |
+--------------------------------------------------+---------+---------+---------+-----------------------------------------------+
```

The focused failures were:

```text
+--------------------------------------------------------------+-----------------------------------------------+
| Test                                                         | Failure                                       |
+--------------------------------------------------------------+-----------------------------------------------+
| grouped scale search [hessian]                               | expected Hessian [1,4,4], actual [1,4]        |
| grouped scale search [hybrid]                                | expected Hessian [1,4,4], actual [1,4]        |
| find_params_batched CPU extension matches eager              | scale tensors differ exactly                  |
+--------------------------------------------------------------+-----------------------------------------------+
```

Standard packing parity verified zero differences among original, block CPU, and GPU packing for 2/3/4/8-bit and
group sizes -1/32/64/128 before the unrelated packing matrix began compiling AWQ extensions.

## Exact validation commands

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-724ea08e-67c3-c0ce-29bb-e6c48e7dde28 \
PYTHONPATH=<baseline-or-tip> /root/vm314t-devin-one/bin/python \
  /tmp/gptq_accuracy_ab.py --revision-label <revision> --variant <variant> \
  --device cuda --output <artifact.pt>

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-724ea08e-67c3-c0ce-29bb-e6c48e7dde28 \
PYTHONPATH=. /root/vm314t-devin-one/bin/python -m pytest -q \
  tests/test_adaptive_damping.py tests/test_adaptive_clipping.py tests/test_gptq.py

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-724ea08e-67c3-c0ce-29bb-e6c48e7dde28 \
PYTHONPATH=. /root/vm314t-devin-one/bin/python -m pytest -q \
  tests/test_planar_bits_567.py tests/test_planar_format_gptq_p.py tests/test_torch_kernel_accuracy.py
```

Raw local artifacts: `/tmp/gptq_ab_baseline.pt`, `/tmp/gptq_ab_tip.pt`, and
`/tmp/gptq_ab_tip_block_control.pt`.

## Recommended acceptance gate

Do not accept the range as accuracy-neutral in its current state. At minimum:

1. Restore baseline block partitioning whenever online group feedback is disabled; retain group-sized blocks only for
   the adaptive path that requires them.
2. Make adaptive damping/clipping opt-in until full, disjoint-calibration model evaluations establish non-regression
   across supported bit widths, group sizes, symmetry modes, and model families.
3. Add a baseline-vs-tip dense-reference regression test that asserts legacy-compatible `qweight`, scales, zeros, and
   `g_idx` are bitwise stable for group sizes below the block size.
4. Add held-out output/KLD gates for adaptive clipping. Existing tests mainly prove finiteness or local candidate
   selection and do not prevent the reproduced distribution-level regressions.
5. Resolve the three focused unit failures before acceptance.

## Limitations

No production checkpoint or calibration/evaluation dataset was specified. Therefore this analysis does not claim a
model-level task-score delta, perplexity delta, or lm-eval result. The dense-reference micro-tests are sufficient to
reject the stronger claim that there are no mathematical/accuracy regressions, and they localize the earliest failing
boundary. A final release decision should additionally run matched full-model quantization with identical calibration
rows and disjoint held-out evaluation (including logits KLD and paired task flips) on representative 4-bit and lower-bit
models.

## Remediation validation

The proposed fix restores the legacy/static GPTQ block partition, makes adaptive damping and clipping explicit opt-ins,
and preserves legacy scalar damping overrides. An identical seeded 10-case CPU rerun against `66a14182` produced
bitwise-identical `qweight`, scale, zero, `g_idx`, reconstructed dense weights, and held-out inputs in every case.

```text
+-----------------------------------------+----------------------------------------------+
| Validation                              | Result                                       |
+-----------------------------------------+----------------------------------------------+
| Quantization cases                      | 10 / 10 completed                            |
| Packed and auxiliary tensors            | 60 / 60 bitwise identical                    |
| Dense-reference accuracy metrics        | Identical across all 10 cases                |
| Adaptive damping/clipping tests         | 101 passed                                   |
| Planar and Torch kernel accuracy tests  | 146 passed                                   |
| Focused prior failures and guards       | 10 passed                                    |
| Ruff and diff whitespace checks         | Passed                                       |
+-----------------------------------------+----------------------------------------------+
```

GPU 7 remained externally saturated during final branch validation (92,388 MiB used of 98,304 MiB and 100% compute,
with no process ownership visible to this container). The earlier GPU 7 isolation control for the block-partition fix
was bitwise exact in all 10 cases; the final branch additionally received the full CPU bitwise A/B above.
