# MFMA operand-orientation experiment

Device revision `728a9dd8`, preceding experimental kernel `bcb6d1de`.
Paired production baseline `216054fa` (production unchanged from preceding audit).
Only the experimental prefetch MFMA layout's transposed flag changes.
This does not change production dispatch or meet the overall 1.5x goal.

## Hypothesis and numerical contract

Test whether transposed MFMA operand ownership, as used by the inspected
ROCm tutorial, reduces the compatible padded layout's approximately 10%
LDSBankConflict metric. Preserve FP16 input/high/low values, FP32 accumulators,
both correction terms, interleaved K-tile evaluation, FP16 output, and slot
reuse barriers. This changes representation/ownership, not precision.
Multiplication operand exchange is not used as an untested rounding proof:
the focused suite checks exact equality to the synchronous control at matched
geometry, in addition to the independent canonical maximum-error gate.

Before committing, all 196 focused tests pass (392 Triton compilation
deprecation warnings). After both executed profiles and the warmed timing
run, all 546 AMD/experimental tests pass (14 warnings, 11.56 seconds).
No threshold or test was weakened. Canonical/tail/mask/canary, FP16/FP32,
separate/interleaved accumulators and repeated one/odd ring cases remain covered.

## Post-profile timings

Raw folded-residual gate/up ceiling: M1024/2048/4096,K5120,N17408,
rates2/2.5/3/3.5, BM128,BN64,BK64, four waves, warmup20/iterations50.
Both the strict idle preflight and pre-timing checks passed on physical GPU0,
MI355X VF gfx950, BDF0000:83:00.0, unique0x333ef6e01ec019b3,256CUs.

| M | Candidate median ms, geomean | Speedup vs paired production, geomean |
|---|---:|---:|
| 1024 | 0.441435 | 1.44895x |
| 2048 | 0.723949 | 1.27703x |
| 4096 | 1.346373 | 1.17390x |

All 12 canonical checks pass, max absolute error0.001880645751953125,
including fresh-input checks. All 12 graph/stream/previous-output ownership
checks pass. The preceding orientation reported1.46702/1.26824/1.16874x
in a separate paired run; these close results do not establish an incremental
latency win. This is not a direct interleaved orientation A/B experiment.
Raw ceilings bypass guards/preprocessing and are not production speedups or
real-model quality evidence. No new 364-case production sweep is claimed.

## Executed counters and generated-code reduction

Matched M1024,K5120,N17408,BM128,BN64,BK64 counters; repeated issued counts
within each capture agree:

| Metric | bcb6d1de | 728a9dd8 |
|---|---:|---:|
| VALU | 25,015,296 | 24,580,096 |
| SALU | 20,959,232 | 20,924,416 |
| LDS | 11,141,120 | 11,141,120 |
| MFMA | 22,282,240 | 22,282,240 |
| Static body instructions | 401 | 324 |
| Compiler VGPR | 114 | 106 |
| Compiler SGPR | 27 | 26 |
| Compiler LDS bytes | 67,488 | 67,488 |
| Scratch / spills | 0 / 0 | 0 / 0 |

Source-correlated AMDGCN shows the output ownership now supports eight
`global_store_dwordx2` instructions instead of16 `global_store_short` plus
16 `global_store_short_d16_hi`. The corresponding 64-bit shift/add address
instructions fall32 to8; vector arithmetic right shifts fall17 to5 and
vector adds23 to10. FP32-to-FP16 packed conversions remain16, preserving
the narrowing boundary. Static savings are mostly epilogue work rather
than the repeatedly executed MFMA loop, explaining why a19.2% static-body
reduction is not a19.2% executed-work or latency reduction.

MFMA count, LDS reads, eight asynchronous hot-loop copies, four static
barriers, and non-draining vmcnt(8)/final vmcnt(0) remain. No copy-routing
`ds_bpermute_b32` or `s_and_saveexec_b64` returns. The interleaved constexpr
still eliminates the separate correction accumulator/add. Shared weight
offsets and one X tile serve both products. Lane address setup is hoisted;
ring-slot/base arithmetic and resource-descriptor advances remain necessary
under the current pipeline. No barrier is removed speculatively.

## The bank-conflict hypothesis was not confirmed

Candidate derived-counter means:

| Metric | bcb6d1de | 728a9dd8 |
|---|---:|---:|
| LDSBankConflict | 10.01960 | 9.97094 |
| MeanOccupancyPerCU | 6.68577 | 6.67008 |
| SQ_WAIT_INST_LDS | 22,593,995.29 | 37,339,184.71 |

Orientation does not materially remove the measured bank-conflict metric;
LDS issue-wait count increases. Lower register use does not translate into
a measured occupancy improvement. These are separate captures, not a causal
proof of every timing difference. Occupancy is waves/CU; LDS issue-wait is
in four-wave-cycle units and not a normalized stall percentage. The installed
gfx950 LDSBankConflict formula is documented in the prior prefetch artifact;
it is not conflicts/access. Scheduler eligibility, full stall breakdown,
achieved bandwidth and actual overlap percentage remain unmeasured.

Retain as an experimental epilogue-layout option, not a new performance
winner or production candidate. Next test padded-LDS spacing and read
scheduling with both orientation controls; do not infer a bank fix from the
tutorial's flag alone. Require matched executed counters, correctness and
latency before selecting a winner.

## Artifacts and reproduction

`orientation_728a9dd8_post.json` records all12 rows, distribution statistics,
configuration, fingerprints and hardware/software identity.
`orientation_728a9dd8_profile.json` records raw CSV hashes, both candidate and
retained-main-GEMM distributions, JIT key, binary hash and full opcode counts.
Compiler fields and rounded profiler allocation fields are kept separate.

Raw roots: `/tmp/qvq-orientation-profile/raw`,
`/tmp/qvq-orientation-metrics/raw`, `/tmp/qvq-orientation-post/report.json`.
Compiler cache: `/tmp/qvq-orientation-profile-cache`.
All three processes exited zero. Profiler recipes follow the previous audit:
sudo rocprofv3 with process-local ROCm SDK library path, mangled kernels, CSV,
regex `folded_residual_prefetch_kernel.*|Cijk_Alik_Bljk_HSS.*`, then separate
SQ_INSTS_VALU/SALU/LDS/MFMA and MeanOccupancyPerCU/LDSBankConflict/SQ_WAIT_INST_LDS
captures. Use M1024,warmup1/iterations2 and the same tile/raw-ceiling flags.
Post timing uses baseline216054fa, M1024/2048/4096,warmup20/iterations50.

Overall retained production evidence is unchanged:36/364 reach1.5x versus
c89459e3;352 canonical passes plus12 unchanged large gate/up fallback cases
that are exact-baseline-equal but exceed the canonical0.002 gate.
