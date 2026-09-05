# Delayed low-correction LDS read

Device0164f98e, preceding device36f47b4a; paired production baselineeec1279f.
Only `_consume` changes: load X/high, issue high MFMA, then load low and
issue its MFMA. Previously low was loaded before high MFMA. FP16 values,
FP32 accumulation, high-then-low K-tile evaluation, padding[[512,16]], tails,
output rounding and ring barriers are unchanged. This is value/lifetime reuse,
not approximate arithmetic. Production never imports this experiment.

196 focused tests pass before commit; after executed profiling and warmed
timing,546 AMD/experimental GPU tests pass (14warnings,11.62seconds).
Canonical maximum-error, exact synchronous control, mask equality, both
accumulator/output modes, tails/canaries and one/odd ring cases are retained.

## Matched M1024,K5120,N17408,BM128,BN64,BK64 profile

| Metric | preceding36f47b4a | delayed0164f98e |
|---|---:|---:|
| Compiler VGPR | 106 | 92 |
| Static body instructions | 324 | 316 |
| Static s_nop | 9 | 3 |
| Static s_waitcnt | 15 | 13 |
| LDSBankConflict mean | 9.93389 | 9.93947 |
| MeanOccupancyPerCU | 6.66183 | 6.67238 |
| SQ_WAIT_INST_LDS mean | 37,348,985.29 | 21,039,134.00 |

Issued VALU24,580,096,SALU20,924,416,LDS11,141,120,MFMA22,282,240
are unchanged; every repeated sample agrees. Compiler LDS67,488bytes,
SGPR26 and no scratch/spills remain. Static wait/nop savings do not imply
a decrease in the selected issued instruction counters.

Source-correlated AMDGCN shows high operands in v76:91, followed by all
high MFMAs; low LDS loads then reuse v76:91 for correction. X remains live
for both products. This changes the schedule and register lifetime, not the
number of products or LDS reads. High loads are no longer overlapped with
simultaneously live low operands in the same way. The late low reads still
precede lgkmcnt(0) and the explicit reuse barrier, followed by low MFMAs.
Thus some waiting is moved onto the later dependency path; a lower LDS
issue-wait count is not by itself lower total critical-path latency.

The full opcode delta is six fewer nops and two fewer waits, with register
renaming/scheduling changes. Hoisted lane offsets, shared high/low weight
offsets,0x80-byte K descriptor advances and modulo-two slot state remain.
No repeated decode, conversion or final correction add is introduced;
interleaved mode retains one accumulator. Wide output stores, hot vmcnt(8),
final vmcnt(0), four static barriers and absence of routing bpermute/EXEC
masking survive. There is no evidence to remove the ring barrier.

Occupancy is mean waves/CU, not percent. LDSBankConflict is the installed
gfx950 derived metric, not conflicts/access. SQ_WAIT_INST_LDS uses four-wave-
cycle units, not normalized stalls. Scheduler eligibility, achieved bandwidth,
actual overlap and full stall breakdown remain unmeasured.

## Post-profile raw ceiling timing

Gate/up,K5120,N17408,BM128,BN64,BK64,four waves, FP16 input/output,
FP32 accumulation. Four rates2/2.5/3/3.5,warmup20/iterations50.

| M | Median ms, geomean | Speedup vs paired production, geomean |
|---|---:|---:|
| 1024 | 0.444058 | 1.46210x |
| 2048 | 0.729235 | 1.27076x |
| 4096 | 1.358821 | 1.16405x |

All12 canonical/graph/stream/ownership checks pass, maxerror
0.001880645751953125. Both strict idle and pre-timing checks pass. These
raw ceilings omit production guards/preprocessing. Results are close to the
preceding separate run and do not establish an incremental latency speedup.
Synthetic tests are not model-quality evidence. Retain as an experimental
register-lifetime simplification, not a promoted performance winner.

Raw `/tmp/qvq-late-low-profile/raw` collects all seven PMCs in one successful
sudo rocprofv3 capture. Use the restored-padding audit command with late-low
paths, baselineeec1279f,M1024,warmup1/iterations2; all other flags unchanged.
Compiler `/tmp/qvq-late-low-profile-cache`; post timing
`/tmp/qvq-late-low-post/report.json`,M1024/2048/4096,warmup20/iterations50.
All processes exited0. Checked-in profile/post JSONs preserve exact commands,
source/hardware fingerprints, CSV/binary hashes, opcodes, distributions and
all12 per-case checks. PhysicalGPU0 MI355XVF gfx950,BDF0000:83:00.0,
unique0x333ef6e01ec019b3,256CUs.

Next use the reduced live-register footprint to investigate explicit
LDS-to-register prefetch across K iterations. The installed Gluon
`amd/warp_pipeline.py` also exposes stage markers, but their effect has not
been benchmarked; do not assume source clustering enforces a faster schedule.
Preserve ownership/barriers and compare actual generated ordering.

Overall goal remains unchanged: all364 M/N/K/rate cases must reach1.5x
versusc89459e3. Latest production evidence is still36/364,352 canonical
passes plus12 unchanged large gate/up fallback cases exact-baseline-equal
but exceeding canonical0.002. No new full production sweep is claimed.
