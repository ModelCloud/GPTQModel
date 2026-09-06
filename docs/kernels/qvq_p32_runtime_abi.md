# QVQ P32 framework-neutral runtime ABI

The canonical Ampere P32 CUDA runtime lives under
`gptqmodel_ext/qvq/p32/`. It accepts caller-owned device buffers, workspace,
and a CUDA stream without depending on PyTorch, PJRT, XLA, or Zig.

Framework integrations own graph construction and tuning policy. They may use
native split reduction or request split partials so a graph compiler can keep
the reduction and neighboring operations visible. No ABI entry point allocates
device memory, changes the current stream, or synchronizes the device.

The public header is the canonical operation and launch contract. It defines
the mathematical operation version, C ABI version, kernel version, SM target,
tile geometry, supported transition-bit range, launch limits, and enum values.
The ABI and kernel versions are also exported by the shared object. Consumers
must include both values in transient autotune keys and reject incompatible
libraries during initialization.

The initial library target is deliberately SM80-only. Additional architecture
libraries must use distinct targets and runtime capability gates rather than a
single implicit device assumption.

`runtime_smoke` validates the shared object through this public ABI on SM80. It
uses nonzero deterministic inputs and checks that native split reduction equals
the caller-visible sum of split partials.

## Standard large-M partial output

Kernel contract v11 accepts partial-output mode for M>16. The product kernels
retain their global `[S,M,N]` split-plane strides; the host launch helpers leave
the final reduction to the caller in partial mode. Native mode still launches
its reducer after the product succeeds. For S=1 the product writes `output`;
for S>1 it writes `partial_output`. A consumer returning only `[S,M,N]` may pass
that result's address to both pointers because only the selected destination
is written.

The SM80 smoke passes 472 standard configurations at K80/N16/S3: rates 4–7,
M1/17/31/32/64/128/256, threads 64/128/256, stages 1–4, automatic grouping,
and supported explicit row groups 2/4/8. Caller-summed partials are compared
with native row_groups=1, requiring finite values and maximum absolute drift
<=0.002 in every case. Ten additional cases cover restricted row_groups=16: M1024/K5120/N10240
at stages2/3 across rates4–7, M1024/K5120/N1024 at stage3/rate5, and
M4096/K5120/N6144 at stage4/rate4. All use S3 and the same per-case gate,
bringing the standard total to482. These synthetic checks do not establish
model quality. Evidence: `/tmp/qvq-expanded-standard-partials.log` and
`/tmp/qvq-rowgroup16-partials.log`.

## Grouped partial-output mode (in development)

Grouped dispatch and launch-plan creation accept `QVQ_P32_REDUCTION_PARTIALS`.
This mode executes only the grouped product kernel; the plan contains one
descriptor and no reduction descriptors. The count remains 336 grouped device
specializations: scalar `4 rates * 4 rows * 3 thread counts *
4 stages = 192`, plus block `4 rates * 3 row variants * 3 thread counts *
4 stages = 144`. Reduction ownership is a host decision, not a new template axis.

The caller retains FP16 inputs/codebook and FP32 accumulation/output semantics.
For group `g` with width `N_g` and split count `S_g`:

- If `S_g == 1`, its output is written to its columns of row-major `[M,total_N]`.
- Otherwise, it writes `[S_g,M,N_g]` to packed partial storage, in group order.
  The offset is the sum of `S_h*M*N_h` for preceding groups with `S_h > 1`.
- Split-group final-output columns and unused partial storage are untouched.
  Consumers must not read them. Both allocations must outlive their consumers.

The caller reconstructs split outputs with an ordered FP32 sum; unsplit groups
need no reduction. Native mode retains the original CUDA reduction sequence.
The extended smoke compares both partial dispatch and partial launch plans with
native output, verifies poisoned regions remain untouched, and compares grouped
output with independently packed standard projections. Its matrix includes
every M1–16, rates4–7, threads64/128/256, stages1–4, two/three groups, and four
split patterns including uneven three-way K partitioning. The numerical limit
is maximum absolute drift <= 2e-3 per case; synthetic tests establish kernel
correctness only, not model quality or a performance claim.

Kernel contract v11 also corrects the generic partial-row block path's direct
output addressing to the documented row-major layout. Previously M5–7 and
M9–15 used group-major offsets for unsplit groups while split reducers used
row-major offsets. An independent M5/K64/N80, rate4, threads64, stage1,
two-group unsplit case reproduced absolute drift 0.0275879 before correction.
No product arithmetic, reduction order, or template axis changes in this fix.
The fixed SM80 runtime passes all 6,144 configurations with zero drift for
native-versus-partial dispatch, native-versus-partial descriptors, and grouped
versus independent standard projections. Twelve invalid cases are rejected.
Three initial idle samples and continuous foreign-GPU-process exclusion passed.
Evidence: `/tmp/qvq-grouped-partials-fixed-runtime-v2.log`; runtime SHA256
`72d950c2e2b0591e49aae4099f7ee29256b62e700a99258ac328b487a4fbd053`.
Broader-shape inference, graph integration, and full-path performance gates
remain open.

The final large-M-capable v11 binary was reprofiled at the same grouped
M5/K64/N80/rate4/threads64/stage1 configuration against the retained v10
baseline. Executed warp instructions are 6018 versus6003, registers remain80,
static shared memory remains1.312KB, and local spilling requests remain0.
The shared-conflict metric remains31; achieved occupancy is3.07% versus3.36%
on this deliberately tiny grid. Source-correlated instructions use the
segment's N-tile start and total-N row stride for direct stores, replacing
the obsolete group-major offset; packed decode and MMA arithmetic are unchanged.
This is a correctness/addressing audit, not a timing win: single profiled
launches and such low occupancy do not establish representative performance.
Artifacts: `/tmp/qvq-v11-final-grouped.ncu-rep`,
`/tmp/qvq-v11-final-grouped-metrics.csv`, and
`/tmp/qvq-v11-final-grouped-source-sass.txt`. Binary SHA256:
`220e5182ecb0483be422f611eba174a5310cb927f712b330d0a92fcaf479fd20`.
The post-profile run passes all482 standard cases, 6144 grouped cases, and
12 rejection cases under exclusive GPU isolation:
`/tmp/qvq-v11-post-profile-correctness.log`.
