# Ampere reference notes

Source of architectural limits: NVIDIA's [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html). Query the live device and CUDA API before allocating resources.

## A100-class sm_80 facts

- Compute capability: 8.0.
- Combined L1/shared-memory capacity: 192 KiB per SM.
- Shared-memory capacity: up to 164 KiB per SM and 163 KiB per block, subject to opt-in and runtime limits.
- Asynchronous global-to-shared copy can reduce register pressure and overlap memory movement with computation.
- BF16 and TF32 Tensor Core modes are available; numerical tolerances must reflect the mode actually used.

## Audit-host snapshot

On 2026-07-20, the host exposed eight `PG506-230/232` devices in PCI order. PyTorch reported compute capability 8.0, 124 SMs, and approximately 96 GiB per device. This snapshot may change between runs and must not appear in launch constants or device-index assumptions.

## Build and validation checklist

- Confirm the generated code contains the intended `sm_80` path.
- Test a tail shape and an accumulation-heavy shape as well as the tuned shape.
- Measure both decode and prefill regimes when the kernel serves both.
- Compare against a matched dense/dequantized result and the closest production backend.
- Verify fallback or explicit rejection on non-Ampere devices.
- Record PyTorch, CUDA runtime, NVCC, driver, GPU properties, build flags, and JIT cache/rebuild state.
