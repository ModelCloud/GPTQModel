# Warp-stage scheduling experiment: compilation rejected

Starting revision: 09f284ab; retained device source: 24cc49c1.
Target: single-slot register-prefetched Gluon GEMM on MI355XVF gfx950.
Compiler: Triton 3.8.0+git4cff872c.rocm10.0.0.

The preceding single-slot ISA audit observed a copy drain before most current
MFMA instructions. This experiment tested explicit copy/compute/refill stage
boundaries. It did not remove waits, reuse barriers, residual products, or
change operand/accumulator precision.

## Attempts and evidence

All commands selected `single or paired_odd` in
`tests/test_qvq_p32_amd_butterfly_experiment.py`, physical GPU 0,
`CUDA_DEVICE_ORDER=PCI_BUS_ID HIP_VISIBLE_DEVICES=0`, cache
`TRITON_CACHE_DIR=/tmp/qvq-triton-butterfly-tests`, pytest cache
`PYTEST_ADDOPTS='-o cache_dir=/tmp/qvq-pytest-cache'`.

| Attempt | Result | Diagnostic |
|---|---|---|
| Wait/barrier inside refill stage, unroll 2 | 208 failed, 36 passed | barrier or wait op cannot appear inside a warp_pipeline_stage region |
| Wait/barrier between stages, unroll 2 | 168 failed, 76 passed | unexpected op inside pipelined_for body |
| Wait/barrier between stages, no unrolling | 204 failed, 40 passed | same unexpected-op diagnostic |

Each run deselected 560 tests and exited 1. Logs respectively:
`/tmp/qvq-stages-tests.log`, `/tmp/qvq-stages-boundaries-tests.log`, and
`/tmp/qvq-stages-loop-tests.log`. Passing subsets do not validate the failed
loop specializations. Removing unrolling did not resolve the failure; the
initial tentative attribution to the unroller was incorrect.

## Pinned compiler diagnosis

Primary sources inspected through authenticated `gh api`, revision 4cff872c:

- [WarpPipeliner.cpp](https://github.com/ROCm/triton/blob/4cff872c/third_party/amd/lib/TritonAMDGPUTransforms/WarpPipeliner.cpp)
- [ConvertWarpPipeline.cpp](https://github.com/ROCm/triton/blob/4cff872c/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertWarpPipeline.cpp)

The first pass's `canSitBetweenStages` accepts `triton::gpu::BarrierOp`.
The conversion pass's `isWarpPipelineIgnorableBarrier` does not, although its
separate `isIntraPipelineGlue` helper explicitly accepts that type. The failed
IR contains `ttg.barrier all` at the required refill/reuse boundary; the later
validator reports the unexpected-op error there. This is evidence of a
barrier-type mismatch between the installed compiler passes, not a numerical
failure or proof that unrolling is unsupported.

The conversion also builds phase-shifted warp groups with four waves per
group and describes two groups. Our current launch has four waves total.
Thus this API cannot be assumed to provide simple scheduling fences for the
existing launch even after the validator mismatch is repaired. Eight-wave
layout/group support would require independent design and validation.
No compiler replacement, validator bypass, or barrier deletion was performed.

## SSA / algebra and profiling disposition

Current X/high/low tile values must reach registers in every wave before the
next iteration overwrites their shared storage. The all-wave reuse barrier
therefore remains necessary. Both high and low MFMA products retain their
existing evaluation order and FP32 accumulators. Stage markers do not justify
algebraic reassociation or dropping dependencies.

The hot loop candidate failed compilation: no new hot-kernel ISA, executed
counter profile, occupancy/register delta, or warmed timing exists. This is a
rejected compilation experiment, not a completed performance phase or win.
The prior executed single-slot evidence remains in
`single_slot_24cc49c1.md` and its linked JSON artifacts.

All experimental edits were restored exactly to HEAD. Source SHA256:
`d964c2547f98de4ccde991d80a8f3ee459d8c62f2f2c92561aae1cc2f67edc71`.
`git diff --exit-code -- scripts/qvq_p32_amd_prefetch_experiment.py` passed.
After restoration, strict idle checks passed (three 0% samples, no resident
foreign KFD process, 285.7 MiB VRAM, BDF 0000:83:00.0), followed by all
244 targeted GPU tests in 2.98 seconds, terminal exit 0. Restoration log:
`/tmp/qvq-stages-restored-tests.log`. No device change is retained by this audit.

## Decision

Do not promote warp stages or remove correctness synchronization. The next
scheduling experiment must use a mechanism compatible with the installed
compiler and preserve the existing launch/dependency contract; alternatively
return to backend dispatch for the unchanged shape groups. Any executable
candidate still requires matched executed profiles, ISA/SSA inspection and
post-profile correctness/timing before promotion.

The full goal remains 1.5x on all 364 cases versus c89459e3. Production evidence
is unchanged: 36/364 reach the target, 352 canonical checks pass, and 12 unchanged
large gate/up fallbacks match the baseline exactly but exceed 0.002 canonical
error. No new full sweep or model-quality result is claimed.
