# Compatible async-load/LDS layout audit

Experimental device revision `bcb6d1de`; preceding prefetch `aeb9206c`;
paired production baseline `98afd5f0`. No production dispatch changes.
Overall target remains 1.5x on all 364 cases versus `c89459e3`.

## Post-profile timing and correctness

Raw folded-residual gate/up ceiling, K5120,N17408, FP16 input/output,
interleaved FP32 accumulation, BK64, four waves. These measurements bypass
production guards/preprocessing and are not end-to-end production speedups.
Each entry is the geometric mean over rates 2/2.5/3/3.5.

| M | BM128 BN64 | BM128 BN128 |
|---|---:|---:|
| 1024 | 1.46702x | 1.27344x |
| 2048 | 1.26824x | 1.18372x |
| 4096 | 1.16874x | 1.10203x |

Both 12-case reports pass canonical maximum absolute error <=0.002;
maximum observed error, including fresh-input checks, is 0.001880645751953125.
All 24 stream, graph and previous-output ownership checks pass. One tall
M1024 row reaches 1.53007x, but its four-rate aggregate remains below 1.5x.
After the profiles, all 546 focused AMD/experimental GPU tests pass
(14 warnings, 11.62 seconds). Tests cover tails, both accumulator/output
modes, identical-geometry synchronous controls, mask pruning and repeated
one/odd-tile ring use. Synthetic checks are not model-quality evidence.

The first square post-profile attempt exited before Torch initialization:
three idle samples reported 2% utilization despite no KFD PID. It produced
no valid timing result. After telemetry returned to zero, a fresh retry
passed the unchanged strict gate. No process was killed or device reset.
Both retained reports passed their idle and pre-timing checks.

## Math, SSA and generated ISA

This is exact representation/layout reuse, not a precision reduction.
The same FP16 X/high/low terms are consumed once per K tile; FP32 MFMA order,
rounding boundaries, tail behavior and explicit slot-reuse barrier remain.
Global DistributedLinearLayout bases now match the vector/lane/wave order
of PaddedSharedLayout. The latter adds 16 elements per 512-element interval.
The supported experimental domain is explicitly BK64, BM/BN64 or128.

TTGIR retains one FP32 accumulator loop in interleaved mode: the unused
separate correction accumulator and final addition disappear. Exact-shape
tail predicates fold away. High and low share the same weight offsets;
X is loaded once for both MFMA products. Invariant lane offsets are built
outside the loop; descriptor pointers advance by 0x80 bytes per K tile.
The modulo-two slot selection and padded slot-base arithmetic remain.
Do not remove the cross-wave reuse barrier without a synchronization proof.

Crucially, source-correlated hot-loop wait_group(1) now emits
`s_waitcnt vmcnt(8)`, rather than the preceding `vmcnt(0)`; the epilogue
still drains with vmcnt(0). The tall body has no `ds_bpermute_b32` or
`s_and_saveexec_b64`: copy-routing permutations and EXEC masking are gone.
Eight global-to-LDS vector loads can remain outstanding at the wait.
This demonstrates non-draining lowering, not a measured overlap percentage.

Matched M1024,K5120,N17408,BM128,BN64,BK64 executed counts:

| Metric | aeb9206c | bcb6d1de |
|---|---:|---:|
| VALU | 48,768,512 | 25,015,296 |
| SALU | 39,028,736 | 20,959,232 |
| LDS | 16,711,680 | 11,141,120 |
| MFMA | 22,282,240 | 22,282,240 |
| Static body instructions | 596 | 401 |
| Compiler VGPR / AGPR | 114 / 0 | 114 / 0 |
| Compiler LDS bytes | 65,536 | 67,488 |
| Scratch / spills | 0 / 0 | 0 / 0 |

Square bcb6d1de uses 614 static instructions, 198 VGPR, 0 AGPR,
101,280 LDS bytes, no scratch/spills. It is slower in all three aggregates.
Compiler resource counts above are not substituted for rounded profiler
allocation fields; both are preserved separately in the profile JSON.

## Measured remaining bottleneck

Tall M1024 derived-counter means:

| Metric | Previous prefetch | Compatible layout | Retained main GEMM |
|---|---:|---:|---:|
| MeanOccupancyPerCU | 6.57260 | 6.68577 | 3.20259 |
| LDSBankConflict | 0 | 10.01960 | 0 |
| SQ_WAIT_INST_LDS | 38,971,618.07 | 22,593,995.29 | 5,798,181.06 |

Occupancy is mean waves/CU, not percent. LDSBankConflict uses the installed
gfx950 derived formula recorded in the preceding profile artifact; it is
not conflicts per LDS access. SQ_WAIT_INST_LDS is a raw waiting count in
units of four wave-cycles, not a stall percentage. The candidate computes
two products while the retained main GEMM computes one. General scheduler
eligibility, full stall breakdown, achieved bandwidth and actual overlap
percentage remain unmeasured.

Next bounded experiment: match the MFMA operand orientation to this LDS
layout (the inspected tutorial uses transposed=True; this kernel uses False),
then measure bank conflicts and timing again. Treat this as a hypothesis,
not a proven cause. Preserve FP32 accumulation, independent accuracy tests,
and ring barriers. Expanding BM256 is a separate layout/resource experiment.

## Reproduction and scope

Post-profile benchmark flags: `--butterfly none --fused-correction
--fused-prefetch --fused-interleave --fused-block-m 128 --fused-block-n {64,128}
--fused-block-k 64 --baseline-amd-commit 98afd5f0 --full-sweep
--shapes mlp_gate_up --m-values 1024 2048 4096 --folded-residual-ceiling
--warmup 20 --iterations 50`. Copied JSON reports retain exact configuration,
hardware/software identity, source hashes, per-case metrics and gate results.
Physical GPU0: MI355X VF gfx950, BDF0000:83:00.0,
unique0x333ef6e01ec019b3,256CUs.

Executed rocprofv3 instruction captures: `/tmp/qvq-linear-{tall,square}-profile/raw`;
derived metrics: `/tmp/qvq-linear-tall-metrics/raw`. All exited zero.
Compiler cache: `/tmp/qvq-linear-profile-cache`. Exact CSV paths/hashes,
specialization keys, opcode distributions and resources are preserved in
`linear_prefetch_bcb6d1de_profile.json`. Commands follow the preceding audit's
sudo rocprofv3 recipe with the above tile flags and baseline, M1024,
warmup1/iterations2. Counters: SQ_INSTS_VALU/SALU/LDS/MFMA, then
MeanOccupancyPerCU,LDSBankConflict,SQ_WAIT_INST_LDS.

No new production full sweep is claimed. The latest production sweep remains
36/364 meeting 1.5x, geometric mean 1.16688x: 352 canonical passes plus
12 unchanged large gate/up fallback cases that are exactly baseline-equal
but fail the canonical 0.002 gate. This raw candidate is not promoted.
