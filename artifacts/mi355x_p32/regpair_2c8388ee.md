# Paired register-prefetch loop: handoff copies eliminated

Device2c8388ee, preceding register-prefetch6df7146e, paired production
baselinea857aa26. Register prefetch remains opt-in; production and the
retained experimental default do not change.

## Change, boundary checks and initial failure

Use Triton's range iterator with loop_unroll_factor=2 in the register-prefetch
branch. This pairs two adjacent K iterations without changing their order,
FP16 operands, FP32 accumulation, correction terms, output rounding or
ring-reuse synchronization. The first attempt used gl.range, which the
installed Gluon module does not export:100 register-prefetch cases failed
with AttributeError,200 controls passed. No executable candidate or timing
was produced by that attempt (`/tmp/qvq-regpair-tests.log`). Inspection of
the installed compiler's loop handler confirmed recognition of triton.language.range;
importing that iterator fixed the frontend failure.

With the corrected import,300 focused cases pass. Added K320/five-tile
coverage exercises a paired main loop plus an odd remainder: all18 ring
and random-remainder cases pass, including canonical FP32 and exact retained
control comparisons in both accumulator modes. After the executed profile
and warmed timing,656 AMD/experimental GPU tests pass (14warnings,11.87s).
Ruff/diff checks pass. No numerical test or gate was weakened.

## Matched M1024,K5120,N17408,BM128,BN64,BK64 evidence

| Metric | single iteration6df7146e | paired2c8388ee |
|---|---:|---:|
| Executed VALU | 67,978,240 | 23,178,752 |
| Executed SALU | 20,811,264 | 13,821,952 |
| Executed LDS | 11,141,120 | 11,141,120 |
| Executed MFMA | 22,282,240 | 22,282,240 |
| Static instructions | 455 | 478 |
| Static v_mov_b32 | 97 | 33 |
| Compiler VGPR / SGPR | 170 / 26 | 170 / 40 |
| LDSBankConflict mean | 10.06483 | 9.63998 |
| MeanOccupancyPerCU | 6.61477 | 6.62242 |
| SQ_WAIT_INST_LDS mean | 40,352,694.32 | 29,789,331.29 |

Repeated issued counts agree. LDS67,488bytes and no scratch/spills remain.
Occupancy is waves/CU, conflict is the installed gfx950 derived metric
rather than conflicts/access, LDS waiting uses four-wave-cycle units rather
than normalized stalls. Full stalls, scheduler eligibility, achieved bandwidth
and actual overlap remain unmeasured.

## ISA / SSA and algebra pass

TTGIR confirms the main loop step is2, with two operand-load/MFMA bodies.
AMDGCN now has no operand-copy sequence at the loop handoff; all33 vector
moves are setup work before the loop. Alternating physical register groups
consume the values in place. This removes the preceding64 per-iteration
register copies rather than changing the computed terms. The larger static
body is expected from unrolling and is not more executed MFMA/LDS work.

Modulo-two slot assignments become alternating constants in the machine
loop: repeated s_xor slot selection and s_mul slot-stride operations
disappear; base/copy offsets are hoisted and reused. The loop branch covers
two original iterations. This reduces scalar bookkeeping as confirmed by
executed SALU. Wide output stores and16 packed FP32-to-FP16 conversions
remain, as do both high/low products and shared X reuse. The interleaved
constexpr still removes the separate correction accumulator/add.
No routing bpermute/EXEC masks return. vmcnt(8) survives in the paired body,
with final vmcnt(0); cross-wave barriers remain. Unrolling exposes scheduling
opportunities but does not prove a particular overlap percentage.

## Post-profile raw ceiling timing

Gate/up FP16 input/output, FP32 accumulation, same tile/K/N, four rates
2/2.5/3/3.5,warmup20/iterations50. All12 canonical/graph/stream/ownership
checks pass, maxerror0.001880645751953125. Strict idle/pre-timing gates pass.

| M | Candidate median ms, geomean | Speedup vs paired production |
|---|---:|---:|
| 1024 | 0.450957 | 1.42177x |
| 2048 | 0.733351 | 1.26970x |
| 4096 | 1.358670 | 1.15754x |

The paired version recovers much of the register-prefetch experiment's
observed large-M slowdown, but has not beaten the retained no-register-prefetch
path. Comparisons with earlier variants are separate runs, not a randomized
interleaved A/B. Raw ceilings bypass production preprocessing/guards and
are not production speedups or real-model quality evidence. Do not promote.

Next compare tile reuse/register pressure now that handoff copies are gone,
including a wider tile and the retained no-register-prefetch control.
The still-high170VGPR and unchanged mean occupancy limit what can be inferred
from instruction savings alone. Any new geometry requires its own profile,
numerical checks and timing before selection.

Reproduction: preceding regprefetch audit command, regpair paths,
baselinea857aa26. All seven PMCs collected in one successful sudo rocprofv3
capture at M1024,warmup1/iterations2. Raw `/tmp/qvq-regpair-profile/raw`,
compiler `/tmp/qvq-regpair-profile-cache`, post
`/tmp/qvq-regpair-post/report.json`,M1024/2048/4096,warmup20/iterations50.
All processes exited0. Checked-in profile/post JSONs preserve configuration,
source/software/hardware fingerprints, CSV/HSACO hashes, complete opcodes,
resources, counter distributions and all12 result rows. PhysicalGPU0
MI355XVF gfx950,BDF0000:83:00.0,unique0x333ef6e01ec019b3,256CUs.

Full objective remains1.5x across364 cases versusc89459e3. No new production
sweep: latest retained evidence remains36/364,352 canonical passes plus12
unchanged large gate/up fallback cases exact-baseline-equal but above0.002.
