# QVQ all-linear Block-LDLQ and YAQA-512 sweep (2026-08-17)

This matched sweep compares canonical V2 and V2B2-P32 with Block-LDLQ and YAQA across every supported
half-step rate from W1 through W3.5. Unlike the earlier Q/K/V/O-only sweep, this run quantizes all 112
decoder linear modules: Q/K/V/O plus gate/up/down in all 16 Llama 3.2 1B layers. Embeddings and the LM head
remain dense.

## Matched configuration

- Model: local `Llama-3.2-1B-Instruct` checkpoint under `/monster/data/model`.
- Dataset: local NeuralMagic calibration dataset.
- Calibration: rows 0--511, 512 independent full rows, batch 1, no concatenation or length limit,
  188,256 valid tokens.
- Evaluation: disjoint rows 512--1023, 512 independent full rows, batch 1, no concatenation or length limit,
  172,367 valid tokens.
- YAQA Sketch-B: disjoint rows 1024--1535, 512 full rows, batch 8, 163,324 valid tokens.
- YAQA factors: FP32; 15,334,375,424 bytes of factor storage reported by the harness.
- MLP bit-rate ladder and fail-closed MLP acceptance: disabled; every module uses the stated fixed rate.
- Seed: 18240; YAQA seed: 0.
- Runtime: Python 3.14.6 free-threaded, PyTorch 2.13.0+cu130, CUDA 13.0.
- Source reports: `/monster/data/model/qvq_v4_sweep_results/all_linear_v2_b2_modes_20260817/reports`.
- Code revision after result collection and report finalization: `24816f6d`.

## Complete scores

Lower is better for Relative L2 and KL. Higher is better for Top-1/5/10 agreement. Top-5 and Top-10 are
set-overlap ratios from the CUDA diagnostic evaluator.

| Rounding | Rate | Codec | Rel L2 | Local KL | Live KL | Layer KL | Final KL | Top-1 | Top-5 | Top-10 | Arm time (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Block-LDLQ | W1 | V2 | 0.615282 | 0.087385 | 1.207866 | 0.975189 | 2.914563 | 34.01% | 35.46% | 35.81% | 381.49 |
| Block-LDLQ | W1 | V2B2-P32 | 0.601869 | 0.080849 | 1.430552 | 0.930279 | 2.783337 | 35.04% | 35.80% | 36.21% | 1372.70 |
| Block-LDLQ | W1.5 | V2 | 0.440301 | 0.026398 | 0.568802 | 0.438939 | 0.846402 | 66.43% | 58.69% | 57.93% | 414.13 |
| Block-LDLQ | W1.5 | V2B2-P32 | 0.429972 | 0.024757 | 0.481857 | 0.410739 | 0.786050 | 67.35% | 60.00% | 59.02% | 1362.80 |
| Block-LDLQ | W2 | V2 | 0.315526 | 0.010515 | 0.249496 | 0.186168 | 0.318296 | 79.50% | 71.61% | 70.74% | 358.47 |
| Block-LDLQ | W2 | V2B2-P32 | 0.307657 | 0.009692 | 0.243337 | 0.175065 | 0.309662 | 79.89% | 71.84% | 71.22% | 1178.33 |
| Block-LDLQ | W2.5 | V2 | 0.225740 | 0.004600 | 0.126192 | 0.085838 | 0.147373 | 86.16% | 78.99% | 78.67% | 369.27 |
| Block-LDLQ | W2.5 | V2B2-P32 | 0.219804 | 0.004140 | 0.120720 | 0.081493 | 0.143049 | 86.05% | 79.51% | 79.06% | 1186.27 |
| Block-LDLQ | W3 | V2 | 0.162025 | 0.002196 | 0.066099 | 0.042954 | 0.074880 | 89.82% | 84.44% | 84.22% | 392.65 |
| Block-LDLQ | W3 | V2B2-P32 | 0.157558 | 0.002072 | 0.065584 | 0.040478 | 0.073343 | 90.01% | 84.74% | 84.43% | 1187.14 |
| Block-LDLQ | W3.5 | V2 | 0.116592 | 0.001105 | 0.035062 | 0.022166 | 0.040567 | 92.15% | 88.31% | 88.16% | 375.64 |
| Block-LDLQ | W3.5 | V2B2-P32 | 0.113285 | 0.001019 | 0.031684 | 0.020183 | 0.036464 | 92.67% | 88.74% | 88.54% | 1317.67 |
| YAQA-512 | W1 | V2 | 0.752992 | 0.613671 | 1.235108 | 0.909567 | 2.326737 | 42.52% | 42.36% | 42.21% | 1893.62 |
| YAQA-512 | W1 | V2B2-P32 | 0.738404 | 0.597931 | 1.218070 | 0.802997 | 2.119864 | 46.28% | 44.11% | 43.69% | 7091.10 |
| YAQA-512 | W1.5 | V2 | 0.562102 | 0.167916 | 0.515339 | 0.361860 | 0.680847 | 71.32% | 62.94% | 62.01% | 1849.44 |
| YAQA-512 | W1.5 | V2B2-P32 | 0.549967 | 0.140412 | 0.458138 | 0.328301 | 0.615066 | 72.45% | 64.40% | 63.31% | 6810.12 |
| YAQA-512 | W2 | V2 | 0.414082 | 0.050158 | 0.223337 | 0.151484 | 0.254286 | 82.26% | 74.21% | 73.52% | 1830.05 |
| YAQA-512 | W2 | V2B2-P32 | 0.404558 | 0.044812 | 0.211028 | 0.140135 | 0.238473 | 83.03% | 74.89% | 74.18% | 6739.28 |
| YAQA-512 | W2.5 | V2 | 0.301521 | 0.020260 | 0.110877 | 0.070301 | 0.111683 | 88.24% | 81.54% | 81.00% | 2004.69 |
| YAQA-512 | W2.5 | V2B2-P32 | 0.293953 | 0.019083 | 0.105685 | 0.065686 | 0.105500 | 88.50% | 81.82% | 81.36% | 7403.02 |
| YAQA-512 | W3 | V2 | 0.218276 | 0.009600 | 0.056138 | 0.034271 | 0.053334 | 91.43% | 86.31% | 85.96% | 1849.65 |
| YAQA-512 | W3 | V2B2-P32 | 0.212395 | 0.009249 | 0.052759 | 0.032762 | 0.050171 | 91.60% | 86.58% | 86.39% | 6756.79 |
| YAQA-512 | W3.5 | V2 | 0.157744 | 0.005022 | 0.029166 | 0.017509 | 0.026740 | 94.07% | 89.74% | 89.69% | 1824.88 |
| YAQA-512 | W3.5 | V2B2-P32 | 0.153295 | 0.004710 | 0.027580 | 0.016580 | 0.025609 | 93.85% | 90.13% | 89.93% | 6979.81 |

## Main observations

- V2B2-P32 improves Final KL over the matched codec at every rate under both Block-LDLQ and YAQA.
- V2B2-P32+YAQA also improves Top-5 and Top-10 at every rate. Top-1 improves through W3, while W3.5
  regresses by 0.21 percentage points despite better Final KL and Top-5/10.
- YAQA substantially improves propagated quality over Block-LDLQ despite increasing weight Relative L2 and
  Local KL. This confirms that local weight/projection metrics alone do not rank the best propagated artifact.
- The cost is excessive: V2B2-P32+YAQA arm time is 6,739--7,403 seconds, roughly 3.7--3.9x the matched
  V2+YAQA arm and about 5.7--6.2x the Block-LDLQ V2B2-P32 arm. The W2.5 result motivates dedicated
  quantization-versus-evaluation profiling before further quality sweeps.

The attempted output-alignment queue is excluded because the comparison harness did not execute alignment;
those jobs failed or remained pending and produced no valid scores.
