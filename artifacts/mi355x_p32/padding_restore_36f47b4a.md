# Padding sweep conclusion and restored-kernel audit

The padding investigation is complete for these controls, not the overall
1.5x goal. No production behavior changes. Device36f47b4a restores[[512,16]]
after the [[512,32]]85230490 and [[512,8]]83e3c901 experiments. The earlier
uncommitted[[256,16]] control failed LLVM translation and was never executable.
All failure and measurement evidence is preserved in the corresponding audits.

## Decision

Retain16. At matched M1024,K5120,N17408,BM128,BN64,BK64,transposed MFMA,
the executed evidence is:

| Padding | LDSBankConflict mean | SQ_WAIT_INST_LDS mean | LDS bytes |
|---|---:|---:|---:|
| 16, original728a9dd8 | 9.97094 | 37,339,184.71 | 67,488 |
| 32 | 19.91522 | 53,492,186.61 | 69,440 |
| 8 | 19.88716 | 53,526,830.21 | 66,512 |
| 16, restored36f47b4a | 9.93389 | 37,348,985.29 | 67,488 |

Padding is not monotonically related to measured conflicts. Both alternate
amounts roughly double this metric; the restored repeat reproduces the better
control. The metric is the installed gfx950 derived formula, not conflicts
per access; raw LDS waiting is in four-wave-cycle units, not a normalized
stall percent. Restored mean occupancy6.66183waves/CU, versus6.67060 for8.
Scheduler eligibility, achieved bandwidth, actual overlap and full stalls
remain unmeasured. No quantitative overlap claim is made from vmcnt alone.

## Per-commit restored ISA / SSA check

An executed rocprofv3 capture at36f47b4a collects all seven instruction and
derived counters in one successful pass. Restored issued counts match the
original16 control exactly: VALU24,580,096,SALU20,924,416,LDS11,141,120,
MFMA22,282,240. Compared with preceding8, VALU falls8,704 and the remaining
issued counts stay. Static body returns325 to324 instructions;106VGPR,
26SGPR,no scratch/spills. The extra setup instruction from8 disappears.

The complete restored HSACO SHA256 is exactly equal to the728a9dd8 binary:
`c8491c1be62928f1463da7cd3c11c0b299caf67bc10d554b5f4e38009d2cdc42`.
Full opcode distributions also match. This independently confirms restoration
of the audited layout address shifts/offsets, hoisted lane offsets, ring-slot
selection, descriptor advances, interleaved FP32 correction and wide output
stores. Source and generated code preserve both products, tail predicates
where needed and synchronization. Hot vmcnt(8)/final vmcnt(0), no routing
bpermute/EXEC masking and no separate interleaved correction add survive.
This is a verified restoration, not a newly discovered algebraic speedup.

## Post-profile validation

Raw gate/up ceiling, FP16 input/output, FP32 accumulation,
K5120,N17408,BM128,BN64,BK64,four waves,four rates2/2.5/3/3.5.
Warmup20/iterations50; paired production baselinef41acda8.

| M | Restored median ms, geomean | Speedup vs production, geomean |
|---|---:|---:|
| 1024 | 0.441696 | 1.45722x |
| 2048 | 0.723934 | 1.28392x |
| 4096 | 1.346021 | 1.17171x |

All12 canonical/graph/stream/output-ownership checks pass, maxerror
0.001880645751953125. Both strict idle and pre-timing checks pass. Raw
ceilings omit production guards/preprocessing and are not production speedups.
Synthetic checks do not establish model quality. After profiling and timing,
all546 focused AMD/experimental tests pass (14warnings,11.59seconds).
Ruff and git diff checks pass. No test or numerical gate was weakened.

PhysicalGPU0 MI355XVF gfx950,BDF0000:83:00.0,unique0x333ef6e01ec019b3,
256CUs. Source/hardware/software fingerprints, all12 latency distributions
and checks are in `padrestore_36f47b4a_post.json`; CSV hashes, full counter
distributions, compiler key/resources and binary hash are in
`padrestore_36f47b4a_profile.json`. Raw `/tmp/qvq-padrestore-profile/raw`,
compiler `/tmp/qvq-padrestore-profile-cache`, timing
`/tmp/qvq-padrestore-post/report.json`; all exited0. Reproduction follows
the pad32 commands, with padrestore paths and all seven PMCs in one capture:
SQ_INSTS_VALU/SALU/LDS/MFMA,MeanOccupancyPerCU,LDSBankConflict,SQ_WAIT_INST_LDS.

## Next experiment and full objective

Stop tuning padding amounts on this basis. Investigate explicit LDS-to-register
prefetch/read scheduling so high and low operands become available without
lengthening dependent MFMA chains. Preserve the ring-reuse barrier until
cross-wave ownership proves it removable; do not trade correctness for a
shorter static body. Compare matched counters and latency before selection.

No new production full sweep is claimed: latest retained evidence remains
36/364 meeting1.5x versusc89459e3,352 canonical passes plus12 unchanged
large gate/up fallback cases exact-baseline-equal but above canonical0.002.
The full M/N/K/rate objective remains open; these controls only eliminate
unsuccessful local hypotheses.
