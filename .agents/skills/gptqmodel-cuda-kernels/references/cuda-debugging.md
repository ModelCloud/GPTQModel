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
4. Run `compute-sanitizer --tool memcheck` on the minimal reproducer; use race or initialization tools when the symptom fits.
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
