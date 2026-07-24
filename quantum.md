# Quantum-assisted ultra-low-bit quantization log

Last updated: 2026-07-24 01:23:51 UTC

## Goal

Test whether quantum/QUBO methods can improve GPTQ-style ultra-low-bit quantization, then implement
the strongest classical AdjacentExact formulation for 2/3/4/8-bit groups of 32, 64, and 128.
Quality is the priority; runtime and optimization cost are secondary.

The bounded experiment fixes the affine quantization scale and zero point, then assigns one binary
decision to each weight: choose its adjacent lower or upper quantization code. For weights `w`,
dequantized lower endpoints `l`, scale `s`, and activation Hessian `H`, the objective is:

```text
minimize over z in {0,1}^32: (w - l - s z)^T H (w - l - s z)
```

This is a 32-variable QUBO and therefore maps to exactly 32 qubits for both 2-bit and 3-bit
codebooks. It does not permit an arbitrary code choice: unrestricted binary encodings would need
64 qubits at 2-bit and 96 qubits at 3-bit.

## Guardrails

- Physical GPU index 6 only, resolved to a UUID before setting `CUDA_VISIBLE_DEVICES`.
- FP64 CUDA-Q simulation for the accuracy-first test.
- Compare against round-to-nearest and an exact classical binary optimizer.
- Validate direct Hessian cost, QUBO cost, and Ising energy algebra before GPU optimization.
- Keep this as an isolated research harness, not a GPT-QModel production backend.

## Confirmed environment and capacity

Physical GPU 6 is UUID `GPU-737e2423-874a-23a4-1126-dfbe3e77c294`, NVIDIA `PG506-230`,
compute capability 8.0, 124 SMs, and 98,304 MiB total memory. The observed software stack is driver
610.43.02, CUDA 13.0, PyTorch 2.13.0+cu130, and CUDA-Q 0.15.0.

With CUDA-Q host-memory spill disabled:

| Precision | Largest executed state vector | Raw state size | Observed GPU allocation |
| --- | ---: | ---: | ---: |
| complex FP32 | 33 qubits | 64 GiB | 66,049 MiB |
| complex FP64 | 32 qubits | 64 GiB | 66,049 MiB |

The next qubit doubles the state vector to 128 GiB and does not fit this 96 GiB GPU.

## Progress

- 2026-07-23: Installed CUDA-Q 0.15.0 in the isolated `venv/cudaq` environment.
- 2026-07-23: Confirmed GPU-6-only UUID isolation and measured the FP32/FP64 capacity boundaries.
- 2026-07-23: Validated the QUBO-to-Ising mapping on an exhaustive 8-variable toy problem to
  `3.331e-16` maximum absolute error.
- 2026-07-23: Added CPU tests for the 2-bit and 3-bit affine codebooks, direct/QUBO/Ising energy
  equivalence, and the exact classical reference. All six focused tests pass.
- 2026-07-23: Ran a 32-qubit complex-FP64 smoke test on physical GPU 6. The final CUDA-Q sample
  allocated a raw 64 GiB state vector and completed successfully.
- 2026-07-23: Completed the accuracy-first depth-5 QAOA comparison for 2-bit and 3-bit group-size-32
  rounding on GPU 6.
- 2026-07-23: Repeated the seeded comparison with end-to-end wall-clock instrumentation and added
  time to the reported solution/candidate.
- 2026-07-23: Implemented the solver-agnostic Torch adjacent-QUBO core, row-group quantization
  callback, exact and greedy references, Ising conversion, and JSON bridge to the isolated CUDA-Q
  environment.
- 2026-07-23: Added and passed 16 focused tests spanning existing Quantizer equivalence, symmetric
  and asymmetric 2/3-bit codebooks, energy mappings, real `GPTQ.add_batch` Hessians, packable codes,
  and Torch-to-NumPy coefficient transport.
- 2026-07-23: Completed the depth-5 real-GPTQ-to-CUDA-Q comparison on physical GPU 6.
- 2026-07-23: Added 4-bit and 8-bit adjacent-codebook coverage, including production RTN
  equivalence for symmetric and asymmetric quantization.
- 2026-07-23: Implemented and validated the FP64 AdjacentExact CUDA Gray-code exhaustive kernel,
  active-decision compression, and exact disconnected-component factorization.
- 2026-07-23: Completed 28 real-GPTQ comparisons across 2/3/4/8 bits and group sizes 8–32, with
  five-run time-to-solution distributions and held-out output-error checks.
- 2026-07-23: Completed 32 additional `sym=True` comparisons at group sizes 64 and 128 using
  diagonal controls and exactly factorable 8-, 16-, and 32-decision correlation blocks.
- 2026-07-23: Implemented a native FP64 CUDA branch-and-bound path for fully coupled 33–128
  decision QUBOs. It preserves every cross-weight Hessian term, carries the state in two 64-bit
  words, returns a candidate plus a lower bound under a finite node budget, and reports exact only
  when every prefix subtree is certified.
- 2026-07-23: Completed 24 dense `sym=True` group-64/128 comparisons at 2/3/4/8 bits on physical
  GPU 6 with 12.8 million branch-and-bound nodes per run and five timing repeats.
- 2026-07-23: Final validation passed 20 focused CUDA tests on GPU 6, 29 CPU/reference tests, one
  targeted extension-registry test, Ruff, shell syntax checks, JSON invariant checks, and
  `git diff --check`.
- 2026-07-23: Completed a serial seed-898 whole-model A/B on Llama 3.2 1B Instruct: dense BF16,
  Classic GPTQ, and the guarded Adjacent hybrid, followed by ARC Challenge and GSM8K Platinum COT.
- 2026-07-23: Added runtime-only Marlin K/N padding for 32-aligned GPTQ checkpoints, validated the
  288-to-384 K and 288-to-320 N paths numerically, and reloaded the 288-wide tiny-model checkpoint
  with all 42 quantized modules using Marlin.
- 2026-07-24: Completed the serial Qwen3-8B dense/Classic/Adjacent A/B and downstream ARC
  Challenge plus GSM8K Platinum COT evaluation. The guard selected zero Adjacent rows, so its
  scores exactly matched Classic GPTQ while quantization took 3.172x as long.
- 2026-07-24: Integrated and validated conservative whole-module CPU candidate offload. The
  measured Qwen layer route keeps Q/K/V/O on CUDA and moves gate/up/down to CPU, projecting a
  1.270x candidate-phase and 1.258x full-Adjacent layer speedup with exact hybrid output.
- 2026-07-24: Promoted the runtime hook to `QuantizeConfig.adjacent_model`, renamed candidate
  placement to `executor`, documented the post-GPTQ, full-row guarded selection lifecycle, and
  passed 60 focused CPU/CUDA tests on physical GPU 0.
- 2026-07-24: Documented the mathematical relationship between GPTQ's damped-inverse-Hessian
  sequential compensation and AdjacentExact's undamped-Hessian binary floor/ceiling QUBO, including
  the row-level no-regression proof and its limits.
- 2026-07-24: Audited AWQ applicability. The fixed-grid binary QUBO transfers mathematically, but
  AWQ needs a separate post-scaling/post-clipping lifecycle integration and currently rejects the
  unsupported `adjacent_model` option.

## Result

The fixed experiment uses seed `20260724`, a `float64` Hessian with condition number `148.216350`,
and four independent dense 8-by-8 covariance blocks inside one 32-weight group. This block structure
is intentional: it preserves correlated Hessian rounding while permitting a certified optimum by
exhaustively checking all 256 assignments in each block, or 1,024 states per bit width.

The affine range endpoints make two qubits inert, so each problem has 30 active adjacent decisions
but is still executed as an exactly 32-qubit circuit. The QUBO/Ising mapping maximum absolute errors
over 4,096 random full-group states were `9.237e-14` at 2-bit and `1.243e-14` at 3-bit.

CUDA-Q used complex FP64, QAOA depth 5, six COBYLA restarts of at most 300 evaluations per block,
20,000 samples per block, and 20,000 samples from the combined 32-qubit circuit.

| Bits | Method | Hessian-weighted error | Reduction from RTN | Time to solution/candidate |
| ---: | --- | ---: | ---: | ---: |
| 2 | classic round-to-nearest | 5.022661497 | 0.000% | 0.055 ms |
| 2 | classical greedy bit flips | 1.055991953 | 78.975% | 1.673 ms |
| 2 | certified classical optimum | 0.961950056 | 80.848% | 0.556 ms |
| 2 | CUDA-Q blockwise postselection | 0.961950056 | 80.848% | 66.197 s |
| 2 | best combined 32-qubit sample | 1.189852891 | 76.310% | 72.052 s |
| 3 | classic round-to-nearest | 0.661716915 | 0.000% | 0.048 ms |
| 3 | classical greedy bit flips | 0.281958940 | 57.390% | 0.634 ms |
| 3 | certified classical optimum | 0.165139241 | 75.044% | 0.523 ms |
| 3 | CUDA-Q blockwise postselection | 0.165139241 | 75.044% | 63.719 s |
| 3 | best combined 32-qubit sample | 0.200120105 | 69.757% | 69.428 s |

These are single-run wall-clock measurements after the common QUBO was built, not steady-state
kernel benchmarks. Classical rows include the complete listed solve. CUDA-Q blockwise time includes
kernel construction, 7,200 optimizer evaluations, and four 20,000-shot block samples. Full32 time
is end-to-end and adds the final 20,000-shot 32-qubit sample. “Candidate” is used because the
combined full-state sample did not reach the certified optimum.

Every independently sampled 8-qubit block contained its exact optimum. Concatenating those four
samples therefore matched the certified group optimum for both bit widths. This is valid only
because the reference Hessian is block diagonal.

The measured per-block optimum probabilities were:

| Bits | Block 0 | Block 1 | Block 2 | Block 3 | Joint probability |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.330% | 3.630% | 5.040% | 3.140% | `7.640e-7` |
| 3 | 1.510% | 2.430% | 5.840% | 3.050% | `6.536e-7` |

Only about 1.5% at 2-bit and 1.3% at 3-bit was expected probability of seeing at least one joint
optimum in 20,000 full-group shots, so neither combined run sampled it. The best full-state results
still improved substantially over RTN. The 32-qubit sampling phases took 5.855 seconds at 2-bit and
5.709 seconds at 3-bit in the timed rerun.

## GPTQ-integrated implementation result

The integrated probe uses an actual one-row `torch.nn.Linear(32, 1)` task. GPT-QModel accumulates
the float32 Hessian through `GPTQ.add_batch` and fixes each bit width's asymmetric scale and zero
point through `Quantizer.find_params`. The calibration matrix is deliberately block structured so
that the adjacent optimum remains certifiable, but the QUBO coefficients are now those emitted from
the implemented GPTQ tensor path rather than a parallel NumPy generator.

The captured Hessian condition number was `430.281911`. Both bit widths had 30 active decisions and
two inert endpoint qubits. Cross-environment direct/QUBO/Ising mapping errors over 4,096 random
states were `7.816e-14` at 2-bit and `2.487e-14` at 3-bit.

| Bits | Method | Hessian-weighted error | Reduction from RTN | Time to solution/candidate |
| ---: | --- | ---: | ---: | ---: |
| 2 | round-to-nearest | 3.843299390 | 0.000% | 0.056 ms |
| 2 | classical greedy bit flips | 1.738507918 | 54.765% | 0.901 ms |
| 2 | classic GPTQ | 1.789156808 | 53.447% | 147.419 ms |
| 2 | certified adjacent optimum | 1.405059805 | 63.441% | 0.597 ms |
| 2 | CUDA-Q blockwise postselection | 1.405059805 | 63.441% | 69.790 s |
| 2 | best combined 32-qubit sample | 1.559596550 | 59.420% | 75.626 s |
| 3 | round-to-nearest | 1.488330530 | 0.000% | 0.064 ms |
| 3 | classical greedy bit flips | 0.313493568 | 78.937% | 1.678 ms |
| 3 | classic GPTQ | 0.212253659 | 85.739% | 10.915 ms |
| 3 | certified adjacent optimum | 0.236763813 | 84.092% | 0.546 ms |
| 3 | CUDA-Q blockwise postselection | 0.236763813 | 84.092% | 65.071 s |
| 3 | best combined 32-qubit sample | 0.294160467 | 80.236% | 70.714 s |

All four blockwise CUDA-Q samples contained their exact adjacent optimum at both bit widths. The
measured block optimum probabilities were 1.775%, 1.470%, 8.995%, and 1.235% at 2-bit; and 2.460%,
1.375%, 13.265%, and 1.330% at 3-bit. The corresponding joint probabilities were only `2.899e-7`
and `5.968e-7`, so neither 20,000-shot full32 sample reached the joint optimum.

At 2-bit, the adjacent optimum reduced error by a further 21.47% relative to classic GPTQ. At 3-bit,
it was 11.55% worse than classic GPTQ. This is not an algebra failure: classic GPTQ's sequential
error feedback modifies the remaining working weights and can choose codes that are not one of an
original weight's adjacent floor/ceiling pair. The one-qubit-per-weight encoding cannot represent
those farther moves.

## Classical AdjacentExact CUDA kernels

The adjacent QUBO is now implemented as a managed GPT-QModel CUDA JIT extension rather than only a
Torch or CUDA-Q research solver. It supports the fixed-codebook adjacent problem at 2, 3, 4, and
8 bits. The bit width changes the scale and endpoints but not the binary search dimension: every
active weight still chooses between its lower and upper code.

The original fast path exhausts connected components of at most 32 active decisions:

- removes endpoint decisions whose lower and upper values are identical;
- splits only mathematically disconnected components whose QUBO coupling is exactly zero;
- assigns a contiguous Gray-code interval to every worker warp;
- maintains the flip field in FP64, so moving to the next state costs one interaction update per
  lane instead of recomputing the full quadratic form;
- recomputes the full FP64 field and energy every 64 states to bound incremental drift;
- derives its default worker count from the live SM count (`SM count * 64`, or 7,936 warps on this
  run), rather than embedding a GPU model or fixed device index; and
- reevaluates every warp's winning state directly before the final reduction.

The new native 33–128 path is a different CUDA kernel:

- represents one full assignment with two 64-bit words, so a dense group of 64 or 128 is not
  truncated or split into artificial 32-wide subproblems;
- gauges the QUBO around the best of RTN and deterministic multi-start local-search incumbents, then
  orders decisions by unary-plus-coupling impact;
- assigns disjoint high-impact prefixes to CUDA threads and performs deterministic depth-first
  branch-and-bound inside each prefix;
- preserves all dense FP64 pair couplings and shares improved incumbents through an atomic FP64
  global minimum;
- takes the maximum of four admissible lower bounds: independent negative edges, split negative
  edges, a positive-edge McCormick envelope, and both mixed combinations; and
- returns the best state, lower bound, visited-node count, and completion state. A bounded search
  is never labeled exact merely because it found a strong candidate.

“Exact” on the 32-bit path means exhaustive enumeration. On the 64/128-bit path it means every
branch-and-bound prefix was pruned or exhausted within the configured FP64 certificate tolerance.
Neither is a symbolic rational proof. Passing a finite node budget can return a high-quality
uncertified candidate; the default exact API instead raises `AdjacentExactIncompleteError` unless
`require_optimal=False`. A zero node budget means unlimited search, which can still take
astronomical time for a hard dense 128-variable instance.

Focused CUDA tests compare both CUDA paths against Torch enumeration across all four bit widths and
dense sizes through 16, exercise a deliberately incomplete node budget and its lower-bound
contract, recover a planted full-high-word optimum for dense sizes 64 and 128, retain exact
component factorization on a non-default CUDA stream, and reject more than 128 active decisions.
All CUDA work was run with physical GPU 6 resolved to UUID
`GPU-737e2423-874a-23a4-1126-dfbe3e77c294`.

### 2/3/4/8-bit GPTQ comparison matrix

The comparison uses actual `GPTQ.add_batch` Hessians and the exact scale/zero emitted by
`Quantizer.find_params`. All 28 cases use one row and one group, `desc_act=False`, no scale search,
float32 weights/calibration, and an FP64 QUBO/objective. `Adj/GPTQ` is the raw Hessian-error ratio;
less than one favors AdjacentExact. Times are warmed median wall-clock time to a solution over five
runs. Every GPTQ timing sample uses a separately reconstructed, pre-calibrated task so `quantize()`
starts from pristine weights and Hessian.

| Scenario | Bits | Sym | Group | Active / components | RTN error | Greedy error | AdjacentExact | Classic GPTQ | Adj/GPTQ | AdjacentExact ms | GPTQ ms |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| diagonal | 2 | no | 8 | 7 / 7 | 1.5061e-1 | 1.5061e-1 | 1.5061e-1 | 1.5061e-1 | 1.0000 | 4.350 | 3.914 |
| diagonal | 3 | no | 8 | 7 / 7 | 3.0517e-2 | 3.0517e-2 | 3.0517e-2 | 3.0517e-2 | 1.0000 | 4.240 | 4.138 |
| diagonal | 4 | no | 8 | 7 / 7 | 5.6968e-3 | 5.6968e-3 | 5.6968e-3 | 5.6968e-3 | 1.0000 | 4.259 | 3.906 |
| diagonal | 8 | no | 8 | 7 / 7 | 4.3008e-5 | 4.3008e-5 | 4.3008e-5 | 4.3008e-5 | 1.0000 | 4.217 | 3.876 |
| dense | 2 | no | 12 | 10 / 1 | 1.8623e-1 | 1.8623e-1 | 1.5088e-1 | 5.9780e-1 | 0.2524 | 1.268 | 5.649 |
| dense | 3 | no | 12 | 11 / 1 | 7.2671e-2 | 6.5518e-2 | 2.7577e-2 | 5.9518e-2 | 0.4633 | 1.211 | 4.885 |
| dense | 4 | no | 12 | 11 / 1 | 2.7306e-2 | 4.6225e-3 | 2.7949e-3 | 1.5242e-2 | 0.1834 | 1.215 | 5.065 |
| dense | 8 | no | 12 | 11 / 1 | 5.1165e-5 | 3.7164e-5 | 1.4593e-5 | 2.4648e-5 | 0.5921 | 1.949 | 5.683 |
| ill-conditioned | 2 | yes | 16 | 14 / 1 | 2.5796e-1 | 2.2698e-1 | 2.2698e-1 | 2.2988e-1 | 0.9874 | 1.236 | 6.475 |
| ill-conditioned | 3 | yes | 16 | 15 / 1 | 3.9717e-2 | 3.8238e-2 | 3.8238e-2 | 3.8638e-2 | 0.9896 | 1.248 | 6.319 |
| ill-conditioned | 4 | yes | 16 | 15 / 1 | 1.0690e-2 | 1.0481e-2 | 1.0481e-2 | 1.0534e-2 | 0.9950 | 1.258 | 6.212 |
| ill-conditioned | 8 | yes | 16 | 15 / 1 | 3.4227e-5 | 3.3941e-5 | 3.3941e-5 | 3.4227e-5 | 0.9916 | 1.266 | 6.307 |
| signed correlation | 2 | yes | 20 | 18 / 1 | 2.8763e+0 | 1.5610e+0 | 1.1098e+0 | 1.5610e+0 | 0.7110 | 1.406 | 7.303 |
| signed correlation | 3 | yes | 20 | 19 / 1 | 2.2864e+0 | 2.3217e-1 | 2.0824e-1 | 3.2536e-1 | 0.6400 | 1.409 | 7.404 |
| signed correlation | 4 | yes | 20 | 19 / 1 | 8.5277e-1 | 4.8271e-2 | 4.2632e-2 | 4.4037e-2 | 0.9681 | 1.462 | 7.341 |
| signed correlation | 8 | yes | 20 | 19 / 1 | 2.8372e-4 | 2.8372e-4 | 1.5672e-4 | 1.9401e-4 | 0.8078 | 1.387 | 7.432 |
| block correlation | 2 | no | 32 | 31 / 4 | 4.3940e+0 | 1.9394e+0 | 1.3686e+0 | 2.9087e+0 | 0.4705 | 3.173 | 10.822 |
| block correlation | 3 | no | 32 | 31 / 4 | 3.5527e+0 | 4.7899e-1 | 3.2184e-1 | 5.3989e-1 | 0.5961 | 3.106 | 11.161 |
| block correlation | 4 | no | 32 | 31 / 4 | 1.5617e-1 | 1.1222e-1 | 8.3320e-2 | 1.1942e-1 | 0.6977 | 3.067 | 10.599 |
| block correlation | 8 | no | 32 | 31 / 4 | 3.0006e-3 | 2.7013e-4 | 1.5167e-4 | 2.0250e-4 | 0.7490 | 3.077 | 10.607 |
| dense correlation | 2 | no | 32 | 31 / 1 | 1.9424e+1 | 4.0431e+0 | 9.9970e-1 | 5.3353e+0 | 0.1874 | 268.254 | 10.781 |
| dense correlation | 3 | no | 32 | 31 / 1 | 3.2570e+0 | 7.7534e-1 | 2.0559e-1 | 7.0220e-1 | 0.2928 | 268.209 | 10.525 |
| dense correlation | 4 | no | 32 | 31 / 1 | 6.2158e-1 | 1.9283e-1 | 3.4157e-2 | 1.1678e-1 | 0.2925 | 267.963 | 10.303 |
| dense correlation | 8 | no | 32 | 31 / 1 | 2.3482e-3 | 5.6751e-4 | 1.2365e-4 | 8.1640e-4 | 0.1515 | 268.095 | 10.402 |
| prior GPTQ-win reference | 2 | no | 32 | 30 / 4 | 3.8433e+0 | 1.7385e+0 | 1.4051e+0 | 1.7892e+0 | 0.7853 | 3.060 | 10.507 |
| prior GPTQ-win reference | 3 | no | 32 | 30 / 4 | 1.4883e+0 | 3.1349e-1 | 2.3676e-1 | 2.1225e-1 | 1.1155 | 3.048 | 10.208 |
| prior GPTQ-win reference | 4 | no | 32 | 30 / 4 | 3.5692e-1 | 5.7933e-2 | 4.1520e-2 | 6.2405e-2 | 0.6653 | 2.984 | 10.186 |
| prior GPTQ-win reference | 8 | no | 32 | 31 / 4 | 6.0828e-4 | 1.4979e-4 | 1.3034e-4 | 1.6890e-4 | 0.7717 | 2.963 | 10.207 |

The fixed scale and zero matched those returned by the classic GPTQ run in all 28 cases, and
AdjacentExact never exceeded RTN error. Mean AdjacentExact/GPTQ error ratios across the seven
scenarios were `0.6277`, `0.7282`, `0.6860`, and `0.7234` for 2, 3, 4, and 8 bits respectively.
The only material loss was the expected 3-bit reference case, where AdjacentExact was 11.55% worse.
The diagonal 3-bit row also selected GPTQ by a negligible `1.6e-7` relative difference caused by
float32 dequantization; the methods represent the same independent rounding solution.

The dense group-32 rows had 31 active coupled decisions, a logical state space of
`2,147,483,648`. Their five-run median AdjacentExact times were 267.963–268.254 ms, with observed
minima of 267.670–268.077 ms and maxima of 268.079–284.725 ms. Classic GPTQ medians were
10.303–10.781 ms. Exact block factorization reduced group-32 AdjacentExact time to about 3.1 ms.
Runtime is therefore exponential in the largest connected component, not in the nominal group
size alone.

Held-out output MSE favored AdjacentExact in 23 of 28 rows. Small calibration-Hessian improvements
did not always transfer: three ill-conditioned rows and one signed-correlation row were up to 2.66%
worse on held-out MSE despite lower calibration error. The 3-bit GPTQ-win reference was also 29.66%
worse on held-out MSE. A model-level quality decision therefore needs representative calibration
and evaluation data; the exact group objective is not itself a guarantee of lower perplexity.

The complete machine-readable results, including five-run min/median/max timing summaries, weight
MSE, held-out MSE, Hessian condition numbers, build flags, software versions, and hardware metadata,
are in
`scripts/quantum_quantization/results/adjacent_exact_benchmark.json`.

### Symmetric group-64/128 sweep

The added large-group suite concentrates entirely on `sym=True`. It uses the same real
`GPTQ.add_batch`/`Quantizer.find_params` path, fixed scale and zero point, `desc_act=False`, no
scale search, float32 weights and calibration, and an FP64 QUBO/objective. Each group size has a
diagonal control and correlated Hessians with exact block widths 8, 16, and 32. Cross-block
couplings are exactly zero by construction, so independently exhausting every connected component
certifies the global full-group optimum.

`Active / components / max` reports the active binary decisions, number of exact interaction
components, and largest component. `Adj/GPTQ` below one favors AdjacentExact. Times are warmed
five-run median wall-clock milliseconds to a solution.

| Scenario | Bits | Group | Coupled block | Active / components / max | RTN error | AdjacentExact | Classic GPTQ | Adj/GPTQ | AdjacentExact ms | GPTQ ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| diagonal | 2 | 64 | 1 | 63 / 63 / 1 | 3.7788e-1 | 3.7788e-1 | 3.7788e-1 | 1.0000 | 32.789 | 19.582 |
| diagonal | 3 | 64 | 1 | 64 / 64 / 1 | 6.8157e-2 | 6.8157e-2 | 6.8157e-2 | 1.0000 | 33.168 | 19.369 |
| diagonal | 4 | 64 | 1 | 64 / 64 / 1 | 1.3711e-2 | 1.3711e-2 | 1.3711e-2 | 1.0000 | 33.521 | 19.373 |
| diagonal | 8 | 64 | 1 | 64 / 64 / 1 | 5.5469e-5 | 5.5469e-5 | 5.5469e-5 | 1.0000 | 33.589 | 19.713 |
| block correlation | 2 | 64 | 8 | 63 / 8 / 8 | 1.2640e+1 | 2.0241e+0 | 3.2167e+0 | 0.6292 | 5.479 | 19.543 |
| block correlation | 3 | 64 | 8 | 63 / 8 / 8 | 2.0259e+0 | 4.1877e-1 | 5.2986e-1 | 0.7903 | 5.492 | 19.567 |
| block correlation | 4 | 64 | 8 | 63 / 8 / 8 | 5.8407e-1 | 8.8708e-2 | 1.4515e-1 | 0.6112 | 5.993 | 20.302 |
| block correlation | 8 | 64 | 8 | 63 / 8 / 8 | 2.0414e-3 | 2.2431e-4 | 2.6891e-4 | 0.8342 | 5.768 | 20.421 |
| block correlation | 2 | 64 | 16 | 64 / 4 / 16 | 5.5016e+1 | 5.2135e+0 | 8.6967e+0 | 0.5995 | 3.462 | 19.693 |
| block correlation | 3 | 64 | 16 | 64 / 4 / 16 | 1.5061e+1 | 9.6023e-1 | 1.9473e+0 | 0.4931 | 3.419 | 19.606 |
| block correlation | 4 | 64 | 16 | 64 / 4 / 16 | 5.6238e+0 | 2.3080e-1 | 4.2706e-1 | 0.5404 | 3.495 | 20.347 |
| block correlation | 8 | 64 | 16 | 64 / 4 / 16 | 2.2764e-2 | 8.0858e-4 | 1.4139e-3 | 0.5719 | 3.520 | 20.254 |
| block correlation | 2 | 64 | 32 | 64 / 2 / 32 | 6.9204e+1 | 4.0056e+0 | 5.6230e+0 | 0.7124 | 2343.729 | 19.851 |
| block correlation | 3 | 64 | 32 | 64 / 2 / 32 | 6.1397e+0 | 8.0520e-1 | 1.4045e+0 | 0.5733 | 2344.307 | 19.838 |
| block correlation | 4 | 64 | 32 | 64 / 2 / 32 | 1.0738e+0 | 1.6133e-1 | 2.5817e-1 | 0.6249 | 2345.268 | 19.826 |
| block correlation | 8 | 64 | 32 | 64 / 2 / 32 | 6.1630e-3 | 6.0128e-4 | 1.2996e-3 | 0.4627 | 2346.312 | 19.212 |
| diagonal | 2 | 128 | 1 | 121 / 121 / 1 | 2.6720e-1 | 2.6720e-1 | 2.6720e-1 | 1.0000 | 62.394 | 37.275 |
| diagonal | 3 | 128 | 1 | 125 / 125 / 1 | 4.2513e-2 | 4.2513e-2 | 4.2513e-2 | 1.0000 | 64.861 | 37.964 |
| diagonal | 4 | 128 | 1 | 127 / 127 / 1 | 9.6719e-3 | 9.6719e-3 | 9.6719e-3 | 1.0000 | 66.181 | 37.901 |
| diagonal | 8 | 128 | 1 | 127 / 127 / 1 | 3.0902e-5 | 3.0902e-5 | 3.0902e-5 | 1.0000 | 66.474 | 37.563 |
| block correlation | 2 | 128 | 8 | 128 / 16 / 8 | 9.8776e+0 | 2.0845e+0 | 2.5220e+0 | 0.8265 | 10.466 | 37.536 |
| block correlation | 3 | 128 | 8 | 128 / 16 / 8 | 3.5479e+0 | 3.7604e-1 | 6.4395e-1 | 0.5840 | 10.497 | 38.299 |
| block correlation | 4 | 128 | 8 | 128 / 16 / 8 | 8.4319e-1 | 9.8001e-2 | 1.2967e-1 | 0.7558 | 10.443 | 38.703 |
| block correlation | 8 | 128 | 8 | 128 / 16 / 8 | 3.4938e-3 | 2.4873e-4 | 3.6190e-4 | 0.6873 | 10.478 | 38.223 |
| block correlation | 2 | 128 | 16 | 125 / 8 / 16 | 1.1364e+1 | 2.7607e-1 | 5.8016e-1 | 0.4759 | 49.577 | 276.356 |
| block correlation | 3 | 128 | 16 | 126 / 8 / 16 | 1.0493e+0 | 5.2362e-2 | 8.6841e-2 | 0.6030 | 10.932 | 45.184 |
| block correlation | 4 | 128 | 16 | 127 / 8 / 16 | 3.5414e-1 | 1.0930e-2 | 1.6621e-2 | 0.6576 | 7.304 | 44.611 |
| block correlation | 8 | 128 | 16 | 128 / 8 / 16 | 1.5040e-3 | 4.0879e-5 | 7.6718e-5 | 0.5329 | 6.847 | 39.467 |
| block correlation | 2 | 128 | 32 | 127 / 4 / 32 | 1.3867e+2 | 7.9366e+0 | 1.6092e+1 | 0.4932 | 3799.121 | 37.964 |
| block correlation | 3 | 128 | 32 | 127 / 4 / 32 | 1.0325e+2 | 1.9938e+0 | 3.3370e+0 | 0.5975 | 3786.922 | 38.204 |
| block correlation | 4 | 128 | 32 | 127 / 4 / 32 | 3.4478e+1 | 4.2672e-1 | 9.1840e-1 | 0.4646 | 3785.808 | 37.953 |
| block correlation | 8 | 128 | 32 | 127 / 4 / 32 | 8.5713e-2 | 1.4435e-3 | 2.2585e-3 | 0.6391 | 3786.368 | 37.980 |

All 32 scale/zero pairs matched Classic GPTQ, and AdjacentExact never exceeded RTN or GPTQ on the
raw Hessian objective. The eight diagonal controls tied as expected. Across the 24 correlated rows,
the AdjacentExact/GPTQ error ratio ranged from `0.4627` to `0.8342` and averaged `0.6150`, a mean
38.50% error reduction. AdjacentExact also beat Classic GPTQ on held-out output MSE in all 24
correlated rows; the held-out ratios ranged from `0.4010` to `0.9224`.

This sweep predates the native branch-and-bound kernel. Its certificates still depend on exact zero
cross-block couplings, and the Gray-code exhaustive path still stops at 32 active decisions. The
new path below removes that implementation-width restriction for dense 64/128 groups, but it does
not remove their exponential worst-case complexity.

The machine-readable sweep, including component sizes, five-run timing summaries, held-out errors,
and full hardware/software metadata, is in
`scripts/quantum_quantization/results/adjacent_exact_sym_large_benchmark.json`.

### Native dense group-64/128 branch-and-bound sweep

This sweep uses fully dense Hessians: it does not zero cross-block terms or solve 32-wide
subgroups. All 24 cases use `sym=True`, actual `GPTQ.add_batch` Hessians and
`Quantizer.find_params` scales/zeros, one row, `desc_act=False`, no scale search, float32
weights/calibration, and an FP64 QUBO/objective. The fixed scales and zeros matched Classic GPTQ in
all cases.

Each native run used split depth 8 and 50,000 nodes for each of 256 prefix workers, for a maximum of
12.8 million visited nodes. Times are warmed five-run median wall-clock milliseconds to the
returned candidate or certificate. The candidate and node count were identical in all five repeats.
`Cert.` means all prefix subtrees were completed and the result is optimal within the configured
`1e-12` FP64 certificate tolerance. `no` means the row is a bounded high-quality candidate, not a
claim of exact optimality.

| Scenario | Bits | Group | Active | RTN error | Native B&B | Classic GPTQ | Native/GPTQ | Cert. | Nodes | Native ms | GPTQ ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| dense | 2 | 64 | 62 | 1.6757e+01 | 2.1867e+00 | 4.3546e+00 | 0.5022 | no | 12800000 | 657.917 | 19.772 |
| dense | 3 | 64 | 63 | 5.3847e+00 | 4.4114e-01 | 8.8597e-01 | 0.4979 | no | 12800000 | 760.543 | 19.824 |
| dense | 4 | 64 | 63 | 9.5679e-01 | 9.1048e-02 | 2.1682e-01 | 0.4199 | no | 12800000 | 740.260 | 19.171 |
| dense | 8 | 64 | 63 | 2.5224e-03 | 3.0506e-04 | 5.0528e-04 | 0.6037 | no | 12800000 | 896.600 | 20.163 |
| signed correlated | 2 | 64 | 59 | 5.8611e+00 | 5.1071e+00 | 6.0288e+00 | 0.8471 | no | 12800000 | 600.682 | 19.472 |
| signed correlated | 3 | 64 | 63 | 3.1201e+00 | 1.1802e+00 | 1.3009e+00 | 0.9072 | no | 12800000 | 560.373 | 19.530 |
| signed correlated | 4 | 64 | 63 | 1.4717e+00 | 2.4678e-01 | 3.1897e-01 | 0.7737 | no | 12800000 | 566.861 | 19.647 |
| signed correlated | 8 | 64 | 63 | 2.3388e-03 | 8.2981e-04 | 8.3568e-04 | 0.9930 | no | 12800000 | 633.773 | 19.629 |
| ill-conditioned | 2 | 64 | 61 | 6.4826e-01 | 6.2606e-01 | 6.3493e-01 | 0.9860 | yes | 350 | 129.974 | 19.247 |
| ill-conditioned | 3 | 64 | 63 | 1.4951e-01 | 1.4664e-01 | 1.4854e-01 | 0.9872 | yes | 376 | 144.176 | 19.537 |
| ill-conditioned | 4 | 64 | 63 | 6.6009e-02 | 6.1320e-02 | 6.3201e-02 | 0.9702 | yes | 466 | 134.746 | 19.194 |
| ill-conditioned | 8 | 64 | 63 | 1.6572e-04 | 1.6035e-04 | 1.6104e-04 | 0.9957 | yes | 372 | 135.460 | 19.107 |
| dense | 2 | 128 | 126 | 6.5405e+01 | 6.2471e+00 | 6.7877e+00 | 0.9204 | no | 12800000 | 946.652 | 37.246 |
| dense | 3 | 128 | 127 | 6.7354e+00 | 9.8561e-01 | 1.3797e+00 | 0.7144 | no | 12800000 | 1034.827 | 36.321 |
| dense | 4 | 128 | 127 | 1.7587e+00 | 2.2497e-01 | 3.2819e-01 | 0.6855 | no | 12800000 | 884.841 | 40.562 |
| dense | 8 | 128 | 127 | 5.0159e-03 | 6.5633e-04 | 6.7058e-04 | 0.9788 | no | 12800000 | 1012.847 | 37.906 |
| signed correlated | 2 | 128 | 125 | 4.6207e+01 | 1.8810e+01 | 2.4513e+01 | 0.7673 | no | 12800000 | 584.786 | 37.815 |
| signed correlated | 3 | 128 | 125 | 7.7562e+00 | 4.0020e+00 | 3.8858e+00 | 1.0299 | no | 12800000 | 568.680 | 36.996 |
| signed correlated | 4 | 128 | 127 | 3.4702e+00 | 1.0261e+00 | 9.8180e-01 | 1.0452 | no | 12800000 | 747.078 | 37.286 |
| signed correlated | 8 | 128 | 127 | 2.0349e-02 | 3.2088e-03 | 3.8923e-03 | 0.8244 | no | 12800000 | 719.774 | 38.469 |
| ill-conditioned | 2 | 128 | 126 | 1.9463e+00 | 1.7761e+00 | 1.7965e+00 | 0.9886 | yes | 4428 | 344.931 | 39.116 |
| ill-conditioned | 3 | 128 | 128 | 3.7975e-01 | 3.3800e-01 | 3.4581e-01 | 0.9774 | yes | 27024 | 571.267 | 38.398 |
| ill-conditioned | 4 | 128 | 128 | 8.4017e-02 | 7.5549e-02 | 7.7719e-02 | 0.9721 | yes | 10040 | 408.390 | 38.770 |
| ill-conditioned | 8 | 128 | 128 | 3.0194e-04 | 2.7317e-04 | 2.8415e-04 | 0.9614 | yes | 13020 | 456.651 | 39.689 |

Native AdjacentExact beat RTN in all 24 rows and Classic GPTQ in 22 of 24. The mean
Native/GPTQ Hessian-error ratio was `0.8479`, or a 15.21% mean reduction. Eight
ill-conditioned rows were certified optimal, including four dense 128-variable searches. The other
16 rows exhausted the finite budget and retain an explicit lower bound and non-optimal status.
Held-out output MSE favored the native candidate in 13 of 24 rows, reinforcing that the calibration
Hessian objective is not a model-level quality guarantee.

The two losses were the 3-bit and 4-bit signed-correlated group-128 cases, at ratios `1.0299` and
`1.0452`. This is expected to remain possible even with a certified adjacent optimum: Classic GPTQ
can reach non-adjacent codes through sequential error feedback, while AdjacentExact deliberately
chooses only each original weight's floor or ceiling code. The quality-first integration rule should
therefore evaluate both candidates and retain the lower objective, rather than replacing GPTQ
unconditionally.

The complete machine-readable artifact includes all five raw timing/cost/lower-bound/node samples,
held-out and weight MSE, build flags, exact GPU UUID, software versions, and certificate status:
`scripts/quantum_quantization/results/adjacent_native_benchmark.json`.

### Integration decision

AdjacentExact should complement GPTQ rather than replace it. A quality-first group path can produce
both candidates and select the lower Hessian error. This guarantees the hybrid is no worse than
either candidate on the measured objective and preserves GPTQ for cases where sequential error
feedback reaches non-adjacent codes. The whole-model integration is opt-in through the runtime-only
`QuantizeConfig.adjacent_model` field. It runs after Classic GPTQ and before packing, keeps GPTQ's
scales/zeros/group index, and substitutes only rows that strictly improve the original full-Hessian
objective. It does not alter checkpoint serialization, CPU fallbacks, or normal GPTQ behavior when
the field is unset.

Mathematically, GPTQ greedily quantizes a compensated sequence of weights using a Cholesky factor
of the damped inverse Hessian. AdjacentExact freezes GPTQ's scale and zero point but jointly chooses
each original weight's floor or ceiling code. For `q(z) = a + Dz`, it minimizes
`(w - q(z))ᵀH(w - q(z))`; the off-diagonal Hessian entries become binary pair couplings, allowing
coordinated flips that RTN and a greedy ordering can miss. Conversely, GPTQ compensation can reach
codes outside that adjacent set, so neither method dominates the other.

The whole-model implementation constructs the alternative with group Hessian blocks, then evaluates
both completed rows using the original full Hessian. Selecting the lower-cost row makes the hybrid
no worse than Classic GPTQ on that calibration objective, but not necessarily on perplexity or
downstream accuracy. A full derivation, the cross-group coupling limitation, and the Llama/Qwen
evidence are recorded in `adjacent_exact.md`.

The same binary floor/ceiling math can complement AWQ after AWQ finishes activation-aware rescaling
and clipping. AWQ already has captured inputs and final affine scales/zeros, so group Hessians can
be formed on demand and the final guard can use direct calibration-output error. This is not wired
today: AWQ has no GPTQ Hessian task, mutates weights and inputs during scaling, and needs a preserved
pre-clipping reference for a valid no-regression comparison. Until that separate lifecycle is
implemented and tested, `adjacent_model` is explicitly GPTQ-only.

## Whole-model Llama 3.2 1B A/B

The requested “Llama 3.1 1B” target does not exist in the local CI recipe; the repository test
target is `/monster/data/model/Llama-3.2-1B-Instruct`. The A/B therefore used that model and the
CI-compatible settings: 4-bit, group size 128, `sym=True`, `desc_act=False`,
`act_group_aware=True`, damp 0.05, 512 calibration rows, concatenation length 2048, descending
calibration sort, quant batch size 1, evaluation batch size 64, chat templates, and seed 898.

Dense, Classic, and Adjacent ran in separate serial processes on the same physical GPU 6 UUID.
Classic and Adjacent each started from the dense checkpoint and identical calibration inputs. The
Adjacent run optimized all 7,602,176 row-groups with deterministic multi-start adjacent coordinate
search, attempted 448 bounded native CUDA refinements, visited 14,068,065 native branch-and-bound
nodes, and certified 4 refinements. It selected the Adjacent candidate for only 27 of 376,832 full
rows; every other row retained its run-local Classic GPTQ candidate.

| Model | ARC acc | ARC normalized | GSM8K Platinum COT | Quant wall | Peak process GPU | Peak Torch allocated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense BF16 | 0.315700 | 0.354096 | 0.464020 | n/a | n/a | n/a |
| Classic GPTQ | 0.314846 | 0.345563 | 0.449959 | 168.485 s | 4,266 MiB | 2,821.458 MiB |
| Adjacent hybrid | 0.311433 | 0.349829 | 0.432589 | 387.604 s | 6,122 MiB | 3,141.458 MiB |

Relative to Classic GPTQ, Adjacent changed ARC accuracy by `-0.003413`, ARC normalized accuracy by
`+0.004266`, and GSM8K Platinum by `-0.017370`. Quantization took `2.3005x` as long and used
1,856 MiB more sampled process GPU memory.

The direct full-Hessian replay objective fell from `387.51606880444825` to
`387.51604458496814`, only a `0.00000625%` reduction. The unguarded all-adjacent candidate was much
worse at `670.8444793143836`; the full-row Classic-versus-Adjacent acceptance gate prevented that
regression. This is the central whole-model result: a tiny calibration-Hessian gain did not
reliably transfer to downstream metrics, and notably harmed GSM8K. The current Adjacent objective
is therefore not strong enough to justify replacing Classic GPTQ or enabling this research mode by
default.

The complete stage outputs, raw Evalution results, per-module Adjacent statistics, memory samples,
and summary are under
`scripts/quantum_quantization/results/llama32_adjacent_ab/20260723T213505Z/`.

## Whole-model Qwen3-8B A/B

The larger serial A/B used `/monster/data/model/Qwen3-8B`, physical GPU 6 UUID
`GPU-737e2423-874a-23a4-1126-dfbe3e77c294`, and seed 898. Quantization used
4-bit group-size-128 `sym=True`, `desc_act=False`, `act_group_aware=True`,
damp 0.05, 512 calibration rows, concatenation length 2048, descending
calibration sort, and quant batch size 1. Evaluation used Marlin, batch size
16, no chat template, ARC Challenge, and GSM8K Platinum COT.

| Model | ARC acc | ARC normalized | GSM8K Platinum COT | Quant wall | Peak process GPU | Peak Torch allocated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense BF16 | 0.558020 | 0.558874 | 0.924731 | n/a | n/a | n/a |
| Classic GPTQ | 0.546075 | 0.547782 | 0.911497 | 723.215 s | 8,986 MiB | 6,221.643 MiB |
| Adjacent hybrid | 0.546075 | 0.547782 | 0.911497 | 2,293.886 s | 11,390 MiB | 6,993.784 MiB |

Across 252 modules and 1,400,832 full rows, the Adjacent phase optimized
54,263,808 row-groups, attempted 1,008 native refinements, visited 22,213,991
branch-and-bound nodes, certified 60 refinements, and improved 547 local-group
candidates. Nevertheless, the unguarded Adjacent candidate's full-Hessian
error was `32467.778694544693`, 2.083x the Classic replay error of
`15583.444859792653`. The full-row acceptance guard therefore selected zero
Adjacent rows. The guarded checkpoint was exactly Classic GPTQ, so all three
reported downstream metrics are exactly equal.

This is a verified negative result for the current 4-bit/group-128 formulation:
local adjacent-group improvements and bounded native refinements did not
produce one row that beat Classic GPTQ on its full-Hessian objective.
Quantization took 3.172x as long and sampled 2,404 MiB more process GPU memory,
with no quality gain. The run predates the later execution-only CUDA
optimizations, so its wall time is not a post-optimization benchmark; those
changes are candidate-equivalent and do not alter the zero-row quality result.

Complete stage outputs and the machine-readable summary remain local under
`scripts/quantum_quantization/results/qwen3_8b_adjacent_ab/20260723T222838Z/`.

## Marlin padding result

Llama 3.2 1B already has Marlin-compliant 2048/8192 projection widths. The padding change targets
otherwise valid GPTQ checkpoints whose K/N dimensions are packed in 32-value units but do not meet
Marlin's 64-column N tile or complete-K-group requirements, such as the 288-wide lifecycle model.

The checkpoint retains its logical serialized shapes. During `MarlinLinear.post_init`, K is padded
to a complete quantization group, N is padded to a 64-column boundary, inputs gain zero K tails,
and outputs are sliced back to logical N. The initial safe contract supports symmetric 4/8-bit GPTQ
with 32-aligned K/N. K padding with `desc_act=True` and any padded module with an adapter are
rejected explicitly. The generic Torch backend remains the fallback.

Focused CUDA tests on GPU 6 passed 7/7 cases, including two-dimensional and batched
three-dimensional numerical comparisons against dense FP16. The actual 6-layer, 42-quantized-module
checkpoint exercised these padded shapes:

| Logical K x N | Marlin runtime K x N |
| ---: | ---: |
| 288 x 288 | 384 x 320 |
| 288 x 768 | 384 x 768 |
| 768 x 288 | 768 x 320 |

After three warmups, ten serial greedy 16-token generations gave:

| Backend | Median wall time | Range | Tokens/s | Padded modules |
| --- | ---: | ---: | ---: | ---: |
| GPTQ Torch | 241.316 ms | 223.755-277.119 ms | 66.303 | 0 |
| Marlin padded | 182.978 ms | 178.614-193.909 ms | 87.442 | 42 |

Marlin was `1.319x` faster and produced an identical 21-token output sequence. The machine-readable
benchmark is
`scripts/quantum_quantization/results/llama32_adjacent_ab/20260723T213505Z/marlin_padding_tiny.json`.

## Qwen3-8B Adjacent CPU versus GPU path timing

The production-shaped timing study used physical GPU 0 UUID
`GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2`, a 96 GiB PG506-230 with compute
capability 8.0 and 124 SMs. The isolated build target was
`-gencode=arch=compute_80,code=sm_80`; Torch was `2.13.0+cu130`, the CUDA runtime
was 13.0, and the driver was 610.43.02. Python 3.14's GIL was explicitly disabled.
Torch intra-op parallelism was fixed at one thread, and the parallel CPU path
used four independent outer tasks. The host exposed 96 logical CPUs during the
run.

This comparison times the dominant `_adjacent_group_candidate` phase, including
the four `nearest/zero/one/linear` starts, at most 32 coordinate flips per start,
rebasing every eight flips, and the production scalar statistics. CPU and GPU
received identical FP64 Hessians and identical real Qwen3-8B layer-0 weight
slices. The weights and projection dimensions are real; the correlated
positive-definite 128-by-128 Hessian is deterministic synthetic calibration
data, so this is a path benchmark rather than a second model-quality experiment.
Each row reports the median of three full task-count runs at 4-bit, group size
128, and `sym=True`.

`GPU/CPU-1` and `GPU/CPU-4` are explicitly CPU baseline time divided by GPU wall
time. Values above one mean the GPU is faster; values below one mean the CPU is
faster.

| Module | Shape | Candidate tasks | CPU-1 | CPU-4 | GPU wall | GPU CUDA | GPU/CPU-1 | GPU/CPU-4 | CPU-4/CPU-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q_proj | 4096 x 4096 | 64 | 21.578 s | 5.203 s | 4.016 s | 4.016 s | 5.372x | 1.295x | 4.147x |
| k_proj | 1024 x 4096 | 32 | 5.265 s | 1.336 s | 1.941 s | 1.941 s | 2.712x | 0.689x | 3.939x |
| v_proj | 1024 x 4096 | 32 | 5.103 s | 1.298 s | 1.925 s | 1.925 s | 2.651x | 0.674x | 3.931x |
| o_proj | 4096 x 4096 | 64 | 20.989 s | 5.181 s | 3.940 s | 3.940 s | 5.327x | 1.315x | 4.051x |
| gate_proj | 12288 x 4096 | 192 | 65.789 s | 16.065 s | 11.620 s | 11.620 s | 5.662x | 1.383x | 4.095x |
| up_proj | 12288 x 4096 | 192 | 63.096 s | 15.418 s | 11.900 s | 11.900 s | 5.302x | 1.296x | 4.092x |
| down_proj | 4096 x 12288 | 192 | 68.161 s | 15.252 s | 11.904 s | 11.904 s | 5.726x | 1.281x | 4.469x |
| one Qwen layer total | seven projections | 768 | 249.982 s | 59.753 s | 47.247 s | 47.247 s | 5.291x | 1.265x | 4.184x |

Four CPU workers scaled almost linearly and beat the sequential GPU path for the
smaller K/V projections, but GPU 0 remained 1.265x faster across a complete
seven-projection layer. The earlier thread sweep's baseline was one CPU worker,
not the GPU; this table supplies the missing matched GPU baseline. A later
granularity sweep split the same work into 512-row tasks and reached 0.443
seconds on `q_proj` with 64 CPU workers. That follow-up is recorded with its
optimized 0.776-second GPU baseline in `adjacent_exact.md`; it supersedes the
coarse-task inference that scaling past four workers is inherently harmful.

CPU and GPU candidates were bit-identical for all seven module types, convergence
and flip counts matched exactly, and the maximum absolute FP64 cost difference
was `8.674e-19`. Thus the speed comparison does not trade away the adjacent
objective. An idealized scheduler that overlaps GPU work with four CPU workers
would reduce the summed candidate phase from 47.247 to about 26.216 seconds per
layer if both measured throughputs remain independent; that `1.80x` projection
is not yet a measured hybrid result.

Native 126-128-active-variable branch-and-bound has no matched CPU result: the
classical CPU reference enumerates at most 20 decisions, whereas a real group
has 126-128. On GPU 0 the bounded production configuration (split depth 6, 500
nodes per worker) took 0.123-0.144 seconds per refinement, visited exactly
32,000 nodes, and did not certify any of these hard groups. Four refinements per
module project to about 3.58 seconds per seven-projection layer, so the native
exact tail is meaningful but remains much smaller than the coordinate candidate
phase.

The reproducible script, all raw timing samples, checksums, correctness deltas,
native node counts, certificate flags, and hardware metadata are in
`scripts/quantum_quantization/benchmark_adjacent_model_cpu_gpu.py` and
`scripts/quantum_quantization/results/adjacent_model_cpu_gpu/20260723_gpu0/results.json`.

### Post-optimization whole-module CPU offload

A follow-up corrected the cache-reuse bias in the repeated-prototype timing by
loading each complete Qwen3-8B layer-0 projection and including CUDA-to-CPU
inputs, a newly constructed 64-worker pool, CPU-to-CUDA output, and the CUDA
full-Hessian guard. Across independently measured module medians, all-CUDA
candidate time sums to 9.079 seconds. The conservative automatic route keeps
Q/K/V/O on CUDA and uses CPU for the 393,216-row-group gate/up/down modules,
reducing that sum to 7.147 seconds, or 1.270x. Including objective replay gives
9.343 versus 7.426 seconds, or 1.258x.

All seven CPU/CUDA pairs returned exactly equal hybrid tensors and discrete
search statistics. The auto route requires GIL-disabled Python, enough
runtime-probed CPU affinity for the requested workers, and at least 393,216
row-groups in the measured 4-bit/group-128 regime; otherwise it retains CUDA.
This optimization changes execution only, not the Qwen A/B quality conclusion
above. Full methodology and the per-module table are in `adjacent_exact.md`.

## Conclusion

The experiments confirm that 2/3/4/8-bit group-size-32 adjacent rounding has the same one-binary-
decision-per-weight structure and that Hessian-aware joint rounding can greatly reduce the chosen
error metric. The solver boundary produces valid GPTQ codes. It does not show a quantum advantage:
the deterministic classical CUDA solver reaches the adjacent optimum directly, while CUDA-Q QAOA
only samples candidates.

A real GPTQ Hessian is generally dense, so quantum blockwise proof and postselection do not carry
over automatically. The classical CUDA implementation now represents and optimizes all couplings
in a native dense group of up to 128 active decisions. It exhausts up to 32 decisions and uses
branch-and-bound above 32, returning an exact certificate only when every subtree completes.
Supporting 128 variables does not make every `2^128` worst case tractable; hard bounded runs are
explicitly reported as candidates rather than optima.

A dense 32-variable QAOA is simulatable, but joint good-state probability remains the quality
bottleneck. Arbitrary code selection still does not fit the quantum simulator: it needs 64 qubits
for 2-bit or 96 qubits for 3-bit. The strongest practical result remains the classical
AdjacentExact/GPTQ hybrid candidate rule, now with a native full-coupling 64/128 CUDA candidate and
an honest certificate contract, not a production CUDA-Q quantizer.
