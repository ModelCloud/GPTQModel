# Exact-tile mask folding

Experimental device revision `dd298d97`; preceding device specialization
`c387f365`. Production is unchanged. Overall target remains >=1.5x across all
364 cases versus c89459e3; the last production result is still 36/364.

## Proof and checks

For the exact grid `ceil(M/BM), ceil(N/BN)`, when M is divisible by BM every
launched row index is strictly below M. The analogous statement holds for N.
The kernel now removes activation masks only for divisible M and weight masks
only for divisible N. Stores are unmasked only when both dimensions divide
exactly. K divisibility remains a static assertion. Mixed and full tails retain
their original masks. This is exact predicate folding, not a math/precision change.

`--fused-keep-masks` retains the preceding control behavior. All 96 focused
cases return exactly equal masked/pruned outputs, including full tiles,
M-only/N-only/both tails, FP16/FP32 output, K256/5120/6144, tiles64/128, and
both accumulator modes. Output canaries pass. After profiling, the combined
suite passes 446 tests, 90 warnings, in 16.41 seconds. Ruff and diff checks pass.

## Matched generated-code / executed-instruction comparison

M1024,K5120,N17408; BM64,BN128,BK64, four waves, interleaved FP32 accumulator.
Four rates share the same specialization. All candidate counter samples agree.

```text
metric                  masked          pruned
issued VALU          106362880        77778944
issued SALU           29558784         1723392
issued LDS            21028864        21028864
issued MFMA           22282240        22282240
static instructions       583             503
VGPR metadata             164             138
SGPR metadata              32              30
AGPR metadata               0               0
LDS bytes               32768           32768
scratch / spills            0               0
```

SALU falls 94.17%, VALU 26.87%; static instructions fall 13.72%. The SSA
unconditional loads survive lowering: the shape-tail predicate and repeated
execution-mask save/restore work disappear. Remaining comparisons implement
lane/layout control and the K-loop exit, not tensor tails. The MFMA stream,
shared-memory layout conversions, and barriers remain. No arithmetic term,
rounding boundary, dtype, or activation/weight stream was removed.

The ISA JSON stores old/new binary hashes, JIT keys, complete static opcode
deltas, actual counter distributions, and exact raw CSV paths. Occupancy,
bank conflicts, scheduler eligibility, stall reasons, and achieved bandwidth
were not measured; register reduction is not a measured occupancy claim.

## Post-profile benchmark

The raw gate/up folded-residual ceiling still bypasses production guards and
preprocessing. It is an optimistic experimental operator, not promoted dispatch.
All 12 canonical FP32 checks pass (max absolute error0.0018806458), as do all
12 stream, graph, fresh-input, and output-ownership checks. The synthetic
fixtures are not model-quality evidence.

```text
M        speedup vs retained production, geomean over four rates
1024     1.00602x
2048     0.84438x
4096     0.73299x
```

The M1024 result is near parity, not a demonstrated meaningful production win.
These values are paired against production34c406a8, not directly paired against
the earlier masked experiment. Do not derive a precise masked-to-pruned latency
speedup by dividing results from separate runs. No complete 364-case sweep was
rerun for this still-rejected experimental backend.

## Reproduction / next action

Post-profile command uses the existing benchmark with `--butterfly none
--fused-correction --fused-interleave --fused-block-m 64 --fused-block-n 128
--fused-block-k 64 --baseline-amd-commit 34c406a8 --full-sweep --shapes mlp_gate_up
--m-values 1024 2048 4096 --folded-residual-ceiling --warmup 20 --iterations 50`.
Output: `/tmp/qvq-fused-mask-post/report.json`. Strict idle preflight and
pre-timing recheck passed on physical GPU0 MI355X VF gfx950, BDF0000:83:00.0,
unique0x333ef6e01ec019b3,256CUs. Software/config/source hashes are in the JSON.

After commit, sudo rocprofv3 captured M1024, warmup1/iterations2, mangled names,
CSV output and SQ_INSTS_VALU/SALU/LDS/MFMA, filtered to the fused kernel and
Cijk library kernels. Raw capture `/tmp/qvq-fused-mask-profile/raw`; compiler
artifacts `/tmp/qvq-fused-mask-profile-cache`. The preceding matched capture
is `/tmp/qvq-fused-single-profile/raw/ubuntu2404-mi350x/808060_counter_collection.csv`.

Next pursue explicit global/local prefetch and direct-to-LDS operand preparation.
The ROCm gfx950 Gluon tutorials HEAD was reverified at
`4d7d632a320b25a789bbdb7a9ba8a8683dce2142`; relevant source is
`kernels/gemm/intra_wave/a16w16/v4_global_prefetch/matmul_kernel.py`, followed
by v5_local_prefetch. This pass located those sources but did not implement
their pipeline. Keep this exact mask folding in that experimental path and
recheck its generated form after any pipeline rewrite.
