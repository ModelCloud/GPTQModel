# Register-protected single LDS slot

Device24cc49c1; production baselinebe57f26a. New opt-in
--fused-single-buffer requires --fused-register-prefetch (invalid CLI use
was checked to exit2 before hardware initialization). Existing double-slot
and production defaults remain unchanged.

## Lifetime / numerical proof

Prime tile0 into the sole LDS slot, wait, load all current X/high/low operands
to registers, then synchronize all waves. Each iteration issues the next
tile into LDS, computes both products from the current register values,
drains copies, loads the next operands, and synchronizes before another
overwrite. The epilogue computes the last resident tile. K coverage is exact,
with no speculative next-tile load. The all-wave barrier is essential because
the producer may overwrite values consumed by another wave.

This preserves the preceding same-BK arithmetic order, FP16 operands,
FP32 accumulators, both correction terms and output narrowing. It is a
storage/lifetime change, not a precision reduction. BK32 and BK64 remain
separate floating-evaluation configurations, each checked canonically.
The independent synchronous control, exact same-geometry comparisons,
tails/masks/canaries and repeated one/two/three/five-tile cases remain.
244 targeted single-slot/boundary tests pass before commit, including random
odd-remainder comparisons at both BK values. After the profiles and warmed
timing, all1074 AMD/experimental GPU tests pass (3086 deprecation warnings
including JIT compilation,133.16seconds). Ruff/diff checks pass. No numerical
gate was weakened.

## Executed BK64 M1024,K5120,N17408 profiles

| Metric | double square | single square | double tall | single tall |
|---|---:|---:|---:|---:|
| BM / BN | 128 / 128 | 128 / 128 | 128 / 64 | 128 / 64 |
| Executed VALU | 23,048,192 | 23,035,136 | 23,178,752 | 23,196,160 |
| Executed SALU | 8,334,080 | 8,290,560 | 13,821,952 | 13,761,024 |
| Executed LDS | 8,355,840 | 8,355,840 | 11,141,120 | 11,141,120 |
| Executed MFMA | 22,282,240 | 22,282,240 | 22,282,240 | 22,282,240 |
| Compiler LDS bytes | 101,280 | 50,592 | 67,488 | 33,696 |
| Compiler VGPR / SGPR | 272 / 47 | 172 / 41 | 170 / 40 | 106 / 36 |
| Static instructions | 762 | 744 | 478 | 467 |
| LDSBankConflict mean | 0 | 0 | 9.63998 | 9.54903 |
| MeanOccupancyPerCU | 3.20801 | 5.51876 | 6.62242 | 12.36134 |
| SQ_WAIT_INST_LDS mean | 14,908,359.54 | 21,286,249.61 | 29,789,331.29 | 36,191,496.29 |

Double controls are the preceding paired-register variants at matched shapes,
not the retained no-register production kernel. Repeated issued counts agree;
no scratch/spills in either new specialization. Occupancy is waves/CU,
not percent. Conflict is the installed gfx950 derived metric, not conflicts
per access. LDS waiting uses four-wave-cycle units rather than normalized
stalls. General scheduler eligibility, full stalls, achieved bandwidth and
actual overlap percentage remain unmeasured.

## ISA / SSA finding

The shared-memory footprint and register demand fall as intended. The paired
loop still has zero operand-handoff v_mov instructions; tile values are
consumed directly from their assigned physical registers. There is no new
decode, mask, conversion or correction-add work. Shared high/low offsets,
X reuse, hoisted addresses and wide output stores survive. With a constant
single slot, alternating slot/base bookkeeping is no longer needed.

The new protocol correctly uses vmcnt(0), not the double-buffered vmcnt(12)
or vmcnt(8), before reading the refill. But source-correlated ISA places the
first drain after only13 of the square tile's64 current MFMA instructions,
or11 of the tall tile's32, within the paired main body. More current MFMA
instructions follow the drain/barrier. Copy loads and some MFMA are interleaved;
the source statement order does not guarantee the full current tile computes
before that wait. This is an observed schedule, not proof it is globally
suboptimal or a measured overlap percentage. Do not remove correctness waits
or barriers to manufacture an overlap claim.

Next test explicit compiler-supported stage boundaries around refill/current
compute/drain to determine whether later draining helps, while keeping
all-wave reuse synchronization. Verify the generated order and timing rather
than assuming a source grouping improves the schedule.

## Post-profile raw ceiling timing

Four rates2/2.5/3/3.5, FP16 input/output, FP32 accumulation, BK64,
K5120,N17408,warmup20/iterations50; geometric means over rates:

| M | square median ms | tall median ms | square speedup | tall speedup |
|---|---:|---:|---:|---:|
| 1024 | 0.519197 | 0.456244 | 1.22118x | 1.40658x |
| 2048 | 0.766807 | 0.727364 | 1.19812x | 1.27780x |
| 4096 | 1.346311 | 1.352900 | 1.15661x | 1.16151x |

All24 canonical/graph/stream/ownership checks pass, maxerror
0.001880645751953125. Strict idle/pre-timing gates pass. Neither variant
beats the retained tall observations; do not promote. These raw ceilings
bypass production guards/preprocessing and are not model-quality evidence.
Earlier variants were measured in separate runs, not randomized interleaved A/B.

Raw `/tmp/qvq-single-{square,tall}-profile/raw`; compiler caches with the
same prefixes plus -profile-cache; post
`/tmp/qvq-single-{square,tall}-post/report.json`. All exited0. Use the square
audit commands with --fused-single-buffer --fused-register-prefetch,BK64,
BN128 or64, baselinebe57f26a. Each successful sudo rocprofv3 capture collects
all seven PMCs at M1024,warmup1/iterations2; postM1024/2048/4096,warmup20/
iterations50. Checked-in JSONs preserve exact configurations, CSV/binary
hashes, full opcodes/resources/counters and all24 rows. PhysicalGPU0
MI355XVF gfx950,BDF0000:83:00.0,unique0x333ef6e01ec019b3,256CUs.

Overall goal remains1.5x across364 cases versusc89459e3. No new production
sweep: latest retained evidence remains36/364,352 canonical passes plus12
unchanged large gate/up fallback cases exact-baseline-equal but above0.002.
