# Kernel integration workflow

Apply the shared [NVIDIA A100+ architecture contract](nvidia-a100-plus.md) to CUDA performance work.

## Repository path

| Concern | Location |
| --- | --- |
| Central JIT extension registry | `gptqmodel/extension.py` |
| Fingerprinting, caching, compiler flags | `gptqmodel/utils/cpp.py` |
| Python wrappers | `gptqmodel/utils/` and `gptqmodel/nn_modules/qlinear/` |
| CUDA/C++ sources | `gptqmodel_ext/` |
| Focused kernel tests | `tests/kernels/` and `tests/test_*_jit.py` |
| Timed experiments | `scripts/benchmark_*.py` |

Inspect `gptqmodel/utils/marlin.py`, `gptqmodel/utils/grasshopper.py`, and their matching extension directories for representative patterns. Reuse their loading and error-reporting conventions; do not import private compiled namespaces directly from unrelated modules.

## Correctness sequence

1. Define a deterministic reference computation and seed.
2. Generate legal packed inputs through the real packer when possible; random integers alone may miss layout errors.
3. Test accumulation-sensitive cases, including long reduction dimensions and adversarial signs/magnitudes.
4. Exercise the last partial tile independently in M, N, and K.
5. Check inputs that are non-contiguous only if the public contract accepts them; otherwise assert a clear rejection.
6. Verify device/stream behavior and that temporary workspace lifetime covers asynchronous execution.
7. Run the kernel through the QuantLinear/backend path, not only as a raw operator.

## Performance sequence

1. Finish JIT compilation and warmup before collecting samples.
2. Use CUDA events for device time or synchronize immediately around host timing.
3. Record median and at least one tail statistic over enough iterations for stable results.
4. Separate decode-like small-M and prefill-like larger-M cases.
5. Compare matched outputs and configs; never compare different precision, layout, or effective work without labeling it.
6. Inspect occupancy, memory traffic, and instruction mix only after end-to-end timing identifies the bottleneck.
7. For each commit/phase that changes generated GPU instructions, capture the
   affected kernel from that exact committed revision with Nsight Compute (or an
   equivalent executed-instruction profiler) and export source-correlated SASS.
8. Compare against the preceding committed binary at the same workload. Audit
   math/algebra and data movement for folding, common-subexpression elimination,
   deduplicated decode/address generation, value reuse, and redundant
   mask/shift/conversion/permutation/store-load sequences. Recheck opcode
   families removed in earlier phases because compiler changes can reintroduce
   them.
9. Record instruction/opcode deltas, registers, spills, bank conflicts,
   occupancy, scheduler/stalls, exact revisions and report paths; then repeat
   correctness and warmed CUDA-event timing before promotion.


## A100+ architecture checks before promotion

For every CUDA hot path, record and verify:

1. **Global access:** lane-to-lane addresses, vector natural alignment, sectors/request, and useful bytes transferred. Tensor contiguity alone is not proof of coalescing.
2. **Shared access:** write the bank equation for the issued warp instruction. Distinguish 32-bit words from FP16 halfwords and broadcasts from true multi-address conflicts.
3. **Async pipeline:** identify the physical producer and consumer (`cp.async`, TMA, ordinary threads, WGMMA/TCGen05) and prove commit/wait/barrier ownership including tails.
4. **Synchronization scope:** same warp -> `__syncwarp`; cross-warp CTA handoff -> CTA barrier/mbarrier; cross-CTA -> kernel/cooperative-grid/cluster mechanism. Never use `__syncwarp` as a cross-warp fence.
5. **Streams:** launch on the caller's current stream and make every cross-stream dependency explicit with events. Do not add device-wide synchronization to make a race disappear.
6. **Resources:** compare registers, spills/local memory, static+dynamic shared memory, active blocks, active/eligible warps, and pipeline depth. Occupancy is evidence, not the objective.
7. **Architecture:** compile and run the exact gated target: sm_80 for A100, sm_90/sm_90a as appropriate for Hopper, and the exact supported Blackwell target for TCGen05/TMEM. Do not assume architecture-accelerated binaries are forward compatible.
