# Cross-iteration register prefetch: loop handoff cost

Device6df7146e, preceding experimental0164f98e, production baseline18fec75c.
The new `--fused-register-prefetch` option requires `--fused-prefetch`; invalid
use exits2 before GPU initialization. The option defaults false. Neither
production nor the retained experimental default is replaced.

## Pipeline / math proof and tests

For T=K/BK>1, prime LDS tiles0/1, wait for0, load its operands to registers
and synchronize all waves. Iteration t holds tilet in registers, copies
t+2 into the now-consumed LDS slot, waits for tilet+1, loads its operands,
and evaluates high then low products for tilet. The next operands become
the loop-carried values. A barrier protects their shared slot before reuse.
After T-2 iterations, drain and load T-1, evaluate T-2 and T-1 exactly once.
T=2 has no main loop; T=1 uses the retained one-tile path. No out-of-bounds
K prefetch or dropped correction is permitted.

This changes value lifetimes, not precision or evaluation order: identical
FP16 X/high/low, FP32 high-then-low MFMA accumulation and output narrowing.
Independent canonical comparisons and exact synchronous-control comparisons
validate the implementation. The new constexpr adds a second specialization
only to the prefetch kernel; the synchronous family is not multiplied.

Before commit,300 focused tests pass (80deselected,1188 compilation
deprecation warnings,61.90seconds). The matrix adds96 register-prefetch
cases over K256/5120/6144, tiles64/128, FP16/FP32 outputs, both accumulator
modes and mixed/full tails. Ring tests now cover K64/128/192, both modes
and prefetch controls, ten exact all-ones repeats with canaries. After
executed profiling and warmed timing,650 AMD/experimental tests pass
(14warnings,11.98seconds). Ruff/diff checks pass; no gate was weakened.

## Matched M1024,K5120,N17408,BM128,BN64,BK64 capture

| Metric | retained0164f98e | register-prefetch6df7146e |
|---|---:|---:|
| Executed VALU | 24,580,096 | 67,978,240 |
| Executed SALU | 20,924,416 | 20,811,264 |
| Executed LDS | 11,141,120 | 11,141,120 |
| Executed MFMA | 22,282,240 | 22,282,240 |
| Static body instructions | 316 | 455 |
| Static v_mov_b32 | 33 | 97 |
| Compiler VGPR | 92 | 170 |
| LDSBankConflict mean | 9.93947 | 10.06483 |
| MeanOccupancyPerCU | 6.67238 | 6.61477 |
| SQ_WAIT_INST_LDS mean | 21,039,134.00 | 40,352,694.32 |

Compiler SGPR26,LDS67,488bytes,no scratch/spills unchanged. Counts are
exact across repeated instruction samples. Occupancy is mean waves/CU,
not percent; conflict is the installed gfx950 derived metric, not
conflicts/access; LDS wait uses four-wave-cycle units, not stall percentage.
General scheduler eligibility, achieved bandwidth, actual overlap and full
stall breakdown remain unmeasured.

## ISA / SSA diagnosis and next rewrite

The intended loop-carried operands appear in generated code, but physical
register assignment materializes64 `v_mov_b32_e32` copies at the loop
handoff. Source-correlated code intersperses these copies with the current
MFMA operations: for example v90<-v154 and v42:45<-v102:105. The full move
sequence closes before the loop branch. This redundant register movement
explains most of the extra issued VALU work; it is not extra arithmetic
terms. Holding two operand tiles also raises register demand substantially.

There are still exactly two products per K tile (issued MFMA unchanged),
one X tile serves both products, high/low share global offsets, and the
interleaved constexpr eliminates the separate correction accumulator/add.
No decode/mask/conversion duplication is introduced. Wider static MFMA/LDS
counts reflect expanded prologue/epilogue, not more executed products/reads.
Exact-shape predicates fold away; output stores remain wide. Hot waits
retain vmcnt(8), final drain vmcnt(0), with no copy-routing bpermute/EXEC
masking. Ring reuse remains synchronized; do not remove barriers to disguise
the handoff cost.

Next test a two-iteration unroll or explicit alternating register groups
so the next iteration can consume the physical registers already filled,
instead of copying back into a single loop-header assignment. This is a
hypothesis requiring a new generated-code check. The installed range API
supports loop_unroll_factor; merely requesting unrolling is not evidence
that register copies disappear. Cover odd T, T1/T2 and masked cases again.

## Post-profile raw ceiling timing

FP16 gate/up, four rates2/2.5/3/3.5,warmup20/iterations50. Same tile and
K/N as above, M1024/2048/4096. All12 canonical/graph/stream/ownership checks
pass, maxerror0.001880645751953125. Both strict idle and pre-timing checks
pass. Results are raw folded-residual ceilings without production guards
or preprocessing, not production speedups or model-quality evidence.

| M | Candidate median ms, geomean | Speedup vs paired production |
|---|---:|---:|
| 1024 | 0.451978 | 1.43171x |
| 2048 | 0.773173 | 1.19487x |
| 4096 | 1.471627 | 1.07132x |

This candidate is not selected: it has worse generated work and slower
observed latencies than the preceding separate run. No incremental speedup
claim or production promotion follows from correctness alone.

Reproduce with the late-low audit command plus `--fused-register-prefetch`,
baseline18fec75c, regprefetch paths. One successful sudo rocprofv3 capture
collects all seven PMCs, M1024,warmup1/iterations2. Raw
`/tmp/qvq-regprefetch-profile/raw`; compiler
`/tmp/qvq-regprefetch-profile-cache`; post
`/tmp/qvq-regprefetch-post/report.json`. All processes exited0. Profile/post
JSONs preserve exact configuration, CSV/HSACO hashes, opcode/resource
distributions and all12 rows. PhysicalGPU0 MI355XVF gfx950,
BDF0000:83:00.0,unique0x333ef6e01ec019b3,256CUs.

Full objective remains1.5x for all364 cases versusc89459e3. No new production
sweep: latest retained evidence remains36/364,352 canonical passes plus12
unchanged large gate/up fallback cases exact-baseline-equal but above0.002.
