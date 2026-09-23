# CUDA failure workflow

Use this order for illegal memory access, misaligned address, launch failure, hangs, or silent corruption.

## Preserve the first failure

Because CUDA errors surface asynchronously, restart the process after an illegal access. Later stack traces may point at an innocent synchronization.

At the launch boundary, record only reproducible metadata:

- operator/backend and architecture path;
- device index, compute capability, current stream, and capture state;
- tensor shape, dtype, device, stride, storage offset, alignment, and contiguity;
- quantization bits, group size, symmetry, activation ordering, and pack dtype;
- launch grid/block, dynamic shared memory, tile configuration, and relevant compiler flags.

Do not dump model weights or tokens unless the test data is known safe and minimal.

## Localize

1. Reproduce with the smallest shape and one GPU.
2. Compare against the reference before and after each suspected boundary: packing, wrapper normalization, launch, and epilogue.
3. Temporarily run the isolated reproducer with `CUDA_LAUNCH_BLOCKING=1` to improve stack attribution. Do not commit this as a runtime default.
4. Run `compute-sanitizer --tool memcheck` first. Then use `--tool racecheck` for shared-memory hazards (including Ampere+ async-copy handoffs), `--tool synccheck` for invalid barrier/warp synchronization, and `--tool initcheck` for uninitialized global/shared reads when the symptom fits.
5. Rebuild with debug line information only for diagnosis, then use `cuda-gdb` or tightly bounded device assertions/printf if sanitizer output is insufficient.
6. Check graph capture separately: allocation, JIT compilation, host callbacks, and stream changes may be illegal during capture even when eager execution works.

## Common quantized-kernel causes

- non-contiguous input tensors that the kernel assumes are contiguous, causing illegal reads or silent fallback to a slow reference (see `$gptqmodel-contiguous-memory`);
- bit-packed tail reads crossing allocation bounds;
- scale/zero or group-index dimensions inconsistent with the weight layout;
- vectorized load alignment assumed but not validated;
- shared-memory byte count mismatched with the selected tile;
- integer overflow in offsets or accumulation;
- workspace freed before asynchronous completion;
- compiling a specialized architecture path into a binary that runs on a different compute capability.

After fixing the cause, add the minimal reproducer as a regression test and remove temporary synchronization and debug output.


## Alignment, bank, and handoff triage on A100+

When the symptom is "misaligned", unexpectedly slow, or nondeterministic rather than a clean OOB:

- Validate the alignment of the **actual per-lane address**, not just the allocation base. A `uint4` cast after a sliced/storage-offset view can violate the 16-byte contract even when the original allocation was aligned.
- Print or unit-test the lane address formula for one warp. A naturally aligned vector load can still be uncoalesced when adjacent lanes are separated by a large stride.
- For shared memory, reduce the byte address to `floor(addr/4) % 32`. Remember that two adjacent FP16 values share one bank word; use Nsight shared wavefront/conflict data instead of assuming "lane N -> bank N".
- Do not repair a cross-warp race with `__syncwarp()`; it synchronizes one warp only. Use a CTA-scoped barrier/mbarrier for shared producer/consumer handoff.
- For `cp.async`/TMA, verify commit/wait or barrier transaction counts before changing arithmetic. A stale stage can look like numerical noise.
- Reproduce graph and non-default-stream paths separately. Hidden default-stream ordering can mask a missing event dependency.
- After a hang involving barriers, run bounded sanitizer tests and inspect inactive/tail paths for missing arrivals before increasing timeouts.
