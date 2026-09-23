---
name: gptqmodel-ampere-kernels
description: Optimize, review, or validate GPT-QModel CUDA and Triton quantized kernels for NVIDIA Ampere GPUs, especially A100-class sm_80 devices. Use for launch geometry, occupancy, cp.async, Tensor Core paths, shared-memory limits, build targets, and A100 benchmarking; combine with the general CUDA-kernel skill.
---

# GPT-QModel Ampere kernels

Use this skill with `$gptqmodel-cuda-kernels`. Probe the live device first, then tune for its compute capability and resources. Treat the repository's current host inventory as a snapshot, never as a stable mapping from CUDA index to GPU type.

Read [references/ampere-notes.md](references/ampere-notes.md) and the shared [NVIDIA A100+ architecture contract](../gptqmodel-cuda-kernels/references/nvidia-a100-plus.md) before setting architecture flags, staging layouts, or shared-memory sizes.

## Probe before choosing a path

Capture:

```bash
nvidia-smi --query-gpu=index,pci.bus_id,name,memory.total,compute_cap --format=csv
python -c 'import torch; print(torch.__version__, torch.version.cuda); [print(i, torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())]'
```

In code, branch on runtime device properties. Do not recognize A100 by product string, assume 108 or 124 SMs, or assign a role to a fixed index or PCI bus ID.

## Target Ampere deliberately

1. Use `sm_80` for A100-class code. Keep `sm_86` and later architectures separate when an instruction or resource assumption differs.
2. Set `TORCH_CUDA_ARCH_LIST=8.0` for an isolated A100-only JIT experiment. Preserve the project's broader architecture list for distributable builds.
3. Use Ampere features only behind appropriate compilation and runtime gates: asynchronous global-to-shared copies, Tensor Core MMA/`ldmatrix`, BF16/FP16, TF32 where numerically intended, and native integer tensor operations.
4. Do not route Ampere through Hopper-only TMA, thread-block clusters, distributed shared memory, WGMMA, or native FP8 assumptions.
5. Opt in explicitly when a block needs more than the default dynamic shared-memory allowance, and validate the requested bytes against the live device.

## A100 execution rules

- Treat `cp.async`/CUDA asynchronous copy as a pipeline with an explicit wait contract, not as a faster spelling of a load/store pair. Prefer 16-byte-aligned copies when the layout permits.
- Compute shared-memory banks from 4-byte words. FP16 lane-contiguous values share bank words in pairs; prove the issued access pattern before padding or swizzling.
- A100 global coalescing is about compact warp sectors, not "32 floats = one transaction". Inspect sectors/request and useful bytes.
- Use `ldmatrix`/MMA layouts that match fragment ownership; do not transpose in registers with a large shuffle network unless it beats a load-time/shared layout.
- `__syncwarp` is same-warp only. Cross-warp staging requires CTA synchronization.
- Do not chase occupancy by register caps without checking spills and eligible-warps/cycle.

## Tune from the bottleneck


- For decode-like small-M quantized GEMV/GEMM, measure launch overhead, weight bandwidth, dequantization cost, and occupancy before adding stages.
- For prefill-like larger-M GEMM, measure Tensor Core utilization, global-to-shared pipeline efficiency, reuse, and epilogue cost.
- Derive grid size from work and live SM count. Benchmark persistent launch factors rather than embedding the current host's 124-SM observation.
- Check registers, active blocks, and shared memory together; a larger tile can reduce effective bandwidth by collapsing occupancy.
- Keep accumulation precision explicit and compare long-K error with the dense reference.

Use existing A100-oriented scripts such as `scripts/benchmark_marlin_a100.py` and the prefill/decode kernel benchmarks when they match the operation. Report the full hardware/software/configuration table alongside results.

## Hidden Ampere/CUDA performance cliffs

Review the bullet list in [references/ampere-notes.md](references/ampere-notes.md) for shape/bank/occupancy/launch/spill/coalescing facts that are not visible in source code or compiler warnings but regularly dominate kernel performance.

## See also

- [Curated GPU performance engineering resources](references/wafer-gpu-perf-resources.md) — External reading list from wafer-ai's performance engineering index.
