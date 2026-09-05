# Explicit-prefetch correction GEMM: pipeline lowering audit

Experimental device revision `aeb9206c`; production baseline `8a4246a3`.
Production is unchanged. The 1.5x goal remains over all 364 cases versus
c89459e3; the latest production result is still 36/364 meeting it.

## Post-profile result

Both candidates are raw gate/up folded-residual ceilings, K5120,N17408,
four rates, BM/BN as below, BK64, four waves, interleaved FP32 accumulation.
They bypass production guards and preprocessing and are not promoted dispatch.

```text
M       wide BM64 BN128  tall BM128 BN64  (speedup vs production)
1024      0.93606x          1.05094x
2048      0.78644x          0.89232x
4096      0.69949x          0.79077x
```

Each value is the geometric mean over rates2/2.5/3/3.5. All 24 canonical
checks pass, max absolute error0.0018806458, with all stream, graph,
fresh-input and output-ownership checks passing. These are synthetic kernel
checks, not model-quality evidence. The small raw M1024 win does not meet the
target and does not license production use. No new 364-case sweep is claimed.

After all profiles, 546 focused tests pass, 14 warnings, in11.64 seconds.
The prefetch subset covers both accumulator modes and output dtypes, square
tiles64/128 with BK64, mixed/full tails, exact equality to the synchronous
kernel at identical tested geometry, and masked/unmasked equality. Dedicated
repeated all-ones tests cover one K tile and odd three-tile ring-buffer use.
Canaries pass. Ruff and diff checks pass.

## Pipeline and correctness design

The implementation adapts the MIT-licensed ROCm gfx950 Gluon tutorial v4 at
4d7d632a320b25a789bbdb7a9ba8a8683dce2142; attribution/license is retained in
the new experimental source. It adds separate high/low shared buffers,
FP32 correction accumulation, general tails, and explicit buffer-reuse barriers.

Prologue loads tile0. Each iteration issues tile(t+1) into the alternate LDS
slot, waits for the previous group, consumes tile(t) for both high/low MFMA,
and synchronizes before that slot can be overwritten. The epilogue drains
the final tile. There are no speculative out-of-range K prefetches. A is
loaded once for both products. Both accumulate modes preserve the preceding
Gluon evaluation order at matched geometry; tests independently check that.
FP16/FP32 output is determined by the existing output-pointer contract.

## Crucial SSA / ISA finding

TTGIR retains `async_wait {num = 1}` and the double-buffer index. AMDGCN
actually contains `buffer_load_dwordx4 ... lds`, so global-to-LDS transfers
are present. However, the hot-loop source-correlated `wait_asyncmark(1)`
lowers to **`s_waitcnt vmcnt(0)`**, draining all pending loads before LDS
reads/MFMA. Thus the intended overlap is not demonstrated and is serialized
at this boundary. Do not describe source-level double buffering as an achieved
latency-hiding pipeline.

The copy lowering also introduces `ds_bpermute_b32` address routing plus
`v_lshrrev_b64`, comparisons, `s_and_saveexec_b64`, and execution-mask branches
around transfers. TTGIR tensor-tail masks are constant true for these exact
tiles; the new execution-mask handling belongs to layout/copy lowering, not
unremoved M/N tails. This is precisely why instruction audits must be repeated
after a pipeline change: earlier mask savings do not imply mask handling
cannot reappear elsewhere.

## Executed instructions at M1024

All repeated instruction samples of each candidate agree.

```text
configuration          VALU       SALU       LDS       MFMA
synchronous64x128   77778944    1723392   21028864   22282240
prefetch64x128      54069248   45295616   20889600   22282240
prefetch128x64      48768512   39028736   16711680   22282240
```

The synchronous row is the preceding matched dd298d97 capture. The tall row
changes tile geometry too; it is not an isolated one-variable comparison.

```text
configuration       static ISA  VGPR  AGPR  LDS bytes  scratch/spills
prefetch64x128          625      130     0      81920       0
prefetch128x64          596      114     0      65536       0
```

Static counts cover complete function bodies, not dynamic iterations. Binary
hashes, exact JIT keys, opcode distributions (including loads with the `lds`
modifier), resource metadata, counter distributions and raw paths are in the
profile JSON. MFMA arithmetic is unchanged from the synchronous folded
candidate; it still performs two products versus the retained main GEMM's one.

## Additional measured bottleneck counters

The installed rocprofiler config.yaml explicitly defines these for gfx950;
its descriptions, formulas and hash are stored with the capture. M1024 tall
candidate and the matched retained main GEMM report:

```text
metric                          retained GEMM       prefetch128x64
MeanOccupancyPerCU mean              3.18771              6.57260
LDSBankConflict min/mean/max         0/0/0                0/0/0
SQ_WAIT_INST_LDS mean          5797328.1875         38971618.0714
```

Occupancy is the tool's mean wave count per CU, not percent occupancy.
SQ_WAIT_INST_LDS reports wave-cycles waiting for LDS issue in units of four
cycles; these raw counts are not a normalized stall percentage. Candidate
and retained operator perform different amounts of work. Nonetheless this
capture does not support bank conflicts or lower mean occupancy as the sole
explanation. General scheduler eligibility, achieved bandwidth and the full
stall breakdown remain unmeasured. The generated routing and wait-zero
instructions give a concrete next investigation instead.

## Reproduction / next action

Post-profile commands use the existing benchmark with `--butterfly none
--fused-correction --fused-prefetch --fused-interleave --fused-block-k 64`,
BM/BN from the table, baseline8a4246a3, full-sweep shape filtermlp_gate_up,
M1024/2048/4096, folded-residual-ceiling, warmup20/iterations50. Reports:
`/tmp/qvq-prefetch-{wide,tall}-post/report.json`.
Both strict idle preflight and pre-timing recheck passed. Physical GPU0 is
MI355X VF gfx950, BDF0000:83:00.0, unique0x333ef6e01ec019b3,256CUs.
Software/config/source fingerprints are embedded, including the new prefetch file.

Post-commit instruction captures use sudo rocprofv3, process-local ROCm/profiler
LD_LIBRARY_PATH, mangled names, CSV, SQ_INSTS_VALU/SALU/LDS/MFMA, regex
`folded_residual_prefetch_kernel.*|Cijk.*`, M1024, warmup1/iterations2.
Raw roots: `/tmp/qvq-prefetch-{wide,tall}-profile/raw`.
Additional capture `/tmp/qvq-prefetch-tall-metrics/raw` requests
MeanOccupancyPerCU,LDSBankConflict,SQ_WAIT_INST_LDS and filters the candidate
plus Cijk_Alik_Bljk_HSS main GEMM. Compiler artifacts:
`/tmp/qvq-prefetch-profile-cache`.

Next adapt a compatible distributed-linear load layout and padded LDS layout
from the inspected ROCm v4/v5 examples to eliminate address routing and make
async group accounting tractable. Verify the resulting wait count and issued
instructions rather than assuming overlap. Do not remove reuse barriers
speculatively; retain the one/odd/tail and exact-control tests through the rewrite.
