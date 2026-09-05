# Square-tile reuse and occupancy controls

Source7ac7061c throughout, production baseline7ac7061c. No device source or
production dispatch changes in this phase. These are executed profiles of
new launch geometries, not assumed reuse of earlier tile evidence.
Compare BM128,BN128,BK64 with register prefetch on/off; preceding BM128,BN64
captures are explicitly different geometry controls.

## Matched square M1024,K5120,N17408 counters

| Metric | no register prefetch | paired register prefetch |
|---|---:|---:|
| Executed VALU | 24,088,320 | 23,048,192 |
| Executed SALU | 12,203,008 | 8,334,080 |
| Executed LDS | 8,355,840 | 8,355,840 |
| Executed MFMA | 22,282,240 | 22,282,240 |
| Compiler VGPR / SGPR | 144 / 27 | 272 / 47 |
| Compiler LDS bytes | 101,280 | 101,280 |
| Static body instructions | 514 | 762 |
| LDSBankConflict mean | 0 | 0 |
| MeanOccupancyPerCU | 3.21763 | 3.20801 |
| SQ_WAIT_INST_LDS mean | 13,794,635.39 | 14,908,359.54 |

No scratch/spills in either kernel; repeated instruction counts agree.
Compared with the tall paired tile (67,488LDS bytes,170VGPR,
6.62242mean waves/CU,11,141,120issued LDS), widening BN improves reuse and
removes the measured bank-conflict metric but roughly halves mean waves/CU.
The square no-register control has nearly half the vector registers yet
the same occupancy. Thus register count alone is not supported as the
explanation; the common larger LDS allocation is the next hypothesis.
This is not a measured complete occupancy-resource limiter breakdown.

Occupancy means waves/CU, not percent. Conflict is the installed gfx950
derived metric, not conflicts/access. LDS issue-wait is in four-wave-cycle
units rather than normalized stalls. General scheduler eligibility, full
stalls, achieved bandwidth and actual overlap remain unmeasured.

## ISA / SSA / algebra check

Both geometries preserve FP16 X/high/low values, FP32 high-then-low products,
K coverage, masks, output narrowing and ring barriers. Doubling BN reuses
each X tile across twice as many output columns; issued LDS falls25% versus
the tall tile while issued MFMA is unchanged. This is reuse, not dropping
correction or changing precision. The larger per-CTA accumulator and shared
tiles are the resource tradeoff, measured above.

Both main loops contain zero v_mov_b32 operand handoff copies; the paired
rewrite's elimination survives this geometry. The paired loop has128 static
MFMA instructions per two-iteration body versus64 in the single body,
without changing dynamic products. Static output stores are16 dwordx2 and
packed FP32-to-FP16 conversions32 in each. Interleaved mode still removes
the separate correction accumulator/add. Exact-shape predicates fold away;
no routing bpermute/EXEC masking reappears. Both hot loops wait vmcnt(12)
for twelve next-tile vector copies, with final vmcnt(0). No claim of an
actual overlap percentage follows. Hoisted addresses, paired constant slot
selection and buffer-reuse synchronization remain intact.

## Post-profile raw ceiling timings

Gate/up,K5120,N17408,BM128,BN128,BK64,four waves, FP16 input/output,
FP32 accumulation, rates2/2.5/3/3.5,warmup20/iterations50.

| M | no-reg median ms | paired median ms | no-reg speedup | paired speedup |
|---|---:|---:|---:|---:|
| 1024 | 0.512261 | 0.511947 | 1.24823x | 1.25727x |
| 2048 | 0.802895 | 0.817422 | 1.14728x | 1.14772x |
| 4096 | 1.441711 | 1.479250 | 1.07808x | 1.05154x |

Entries are geometric means over rates, relative to each run's paired
production timings; variants ran separately, not a randomized interleaved
A/B. Neither beats the earlier tall observations. All24 canonical/graph/
stream/output-ownership checks pass, maxerror0.001880645751953125. All
strict idle and pre-timing gates pass. After both profiles and timing runs,
656 AMD/experimental tests pass (14warnings,12.11seconds).

Raw folded-residual ceilings omit production guards/preprocessing; synthetic
checks are not model-quality evidence. No promotion or full-sweep gain is
claimed. Square tiles are not selected from these results.

Next test BK32 while retaining BM128/BN128, to reduce shared-tile storage.
This requires extending the compatible load/shared-layout domain and
validating it, not just bypassing the BK64 assertion. Extra K iterations
may raise bookkeeping cost. Profile the actual footprint/occupancy and
instructions, then rerun accuracy/timing before selecting a configuration.

## Reproduction / artifacts

Raw `/tmp/qvq-square-{paired,retained}-profile/raw`; compiler caches with
the same prefixes plus `-profile-cache`; post
`/tmp/qvq-square-{paired,retained}-post/report.json`. All exited0. Checked-in
square_paired/square_retained profile and post JSONs retain CSV/binary hashes,
full opcodes/resources/counter distributions, per-case results, configuration
and hardware/software/source fingerprints. PhysicalGPU0 MI355XVF gfx950,
BDF0000:83:00.0,unique0x333ef6e01ec019b3,256CUs.

Use the prior regpair audit commands with BN128, baseline7ac7061c and
these prefixes; paired adds --fused-register-prefetch, retained omits it.
Each sudo rocprofv3 capture collects all seven PMCs at M1024,warmup1/iterations2.
Post M1024/2048/4096,warmup20/iterations50. Geometry/math-changing phases
remain subject to the repository's executed-profile and post-test gates.

Overall target remains1.5x across364 cases versusc89459e3. No new production
sweep: latest retained evidence remains36/364,352 canonical passes plus12
unchanged large gate/up fallback cases exact-baseline-equal but above0.002.
