---
name: gptqmodel-hopper-kernels
description: Optimize, review, port, or validate GPT-QModel CUDA and Triton quantized kernels for NVIDIA Hopper H100/H200 sm_90 GPUs. Use for CUTLASS/CuTe, TMA, thread-block clusters, distributed shared memory, WGMMA, FP8, Hopper resource limits, sm_90 versus sm_90a builds, or Hopper benchmarks; combine with the general CUDA-kernel skill.
---

# GPT-QModel Hopper kernels

Use this skill with `$gptqmodel-cuda-kernels`. Separate portable Hopper code generation from architecture-accelerated code, and separate build evidence from runtime evidence.

Read [references/hopper-notes.md](references/hopper-notes.md) and the shared [NVIDIA A100+ architecture contract](../gptqmodel-cuda-kernels/references/nvidia-a100-plus.md) before choosing `sm_90` or `sm_90a`, TMA layouts, WGMMA ownership, or shared-memory swizzles.

## Establish available evidence

Probe the same live properties used by the Ampere workflow. If no compute-capability 9.0 device is available:

- inspect and compile Hopper paths when the installed toolchain supports them;
- run fallback and architecture-gate tests on available hardware;
- mark Hopper runtime tests skipped with the detected capability;
- make no H100 correctness, latency, throughput, occupancy, or speedup claim.

The 2026-07-20 audit host had no Hopper GPU. An actual H100 run is therefore required before merging a performance claim produced solely on this host.

## Choose the target correctly

1. Use `sm_90` for code intended to remain compatible across compute-capability 9.0 implementations.
2. Use `sm_90a` only for architecture-accelerated features whose generated code is not forward- or backward-compatible. Isolate that binary/path and retain a generic or older-architecture fallback.
3. Gate TMA, WGMMA, thread-block clusters, and distributed shared memory in both build selection and runtime dispatch.
4. Treat FP8 format, scaling, accumulation, saturation, and output dtype as explicit quantization contracts. Native FP8 throughput does not remove the need for accuracy and save/load tests.
5. Derive SM count, cluster limits, shared-memory opt-in, and launch bounds from the live device rather than a canonical H100 SKU.

## Hopper memory/Tensor Core rules

- TMA is most valuable when it deletes per-thread address/load instructions and overlaps a sufficiently large tile. Do not infer a win from "asynchronous" alone.
- Use TMA swizzle/CuTe shared layouts for structured operands when they reduce measured wavefronts; do not swizzle data-dependent LUTs blindly.
- WGMMA source ownership is a design axis: register-source can save SMEM traffic but raise register/decode pressure; shared-source can enable reuse but can force one CTA/SM.
- Keep WGMMA fragment lifetime and `warpgroup_arrive/commit/wait` ordering explicit. Reusing a fragment before the wait is a correctness bug.
- Hopper's 227-KiB per-block SMEM ceiling makes "eliminate all bank conflicts by replication" especially dangerous for large reuse tiles. Compare whole-CTA residency and end-to-end latency.
- A low DRAM percentage can be healthy for compressed/reuse-heavy kernels. Check scheduler eligibility, long scoreboard, shared wavefronts, integer decode instructions, and WGMMA issue before chasing HBM GB/s.

## Tune Hopper features only when justified


- Use TMA for multidimensional transfers whose descriptors and alignment are stable enough to amortize setup.
- Use clusters and distributed shared memory only when cross-block cooperation outweighs scheduling constraints and reduced residency.
- Tune warp-group MMA pipelines with register pressure, stage count, shared memory, and producer/consumer synchronization considered together.
- Compare specialized Hopper paths to the repository's production alternatives, including Machete or Marlin where their method/format contract matches.
- Measure decode and prefill separately; large Hopper GEMM gains do not imply small-M latency gains.

### CUTLASS and CuTe primitive audit

Before hand-writing a Hopper mechanism, identify the repository's vendored CUTLASS release and compare it with the current supported release. Verify the selected headers and compile flags actually instantiate the expected `sm_90a` operation. Treat the Python CUTLASS DSL package and the vendored C++ headers as separate dependencies; installing one does not update the other.

For each hot producer, decode, MMA, and epilogue step, check whether current CuTe exposes a suitable primitive or layout abstraction, including:

- `cute::make_tma_atom`, `cute::tma_partition`, TMA descriptor prefetch/cache policy, and multicast when reuse crosses CTAs;
- `cutlass::PipelineTmaAsync` and matching producer/consumer barrier accounting;
- `cute::GMMA::rs_op_selector` or shared-sourced WGMMA selected for the actual operand ownership;
- `cute::warpgroup_arrive`, `warpgroup_commit_batch`, `warpgroup_wait`, operand fences, and supported register reconfiguration;
- CuTe shared-memory layout/swizzle selectors instead of accumulating manual padding/permutations;
- fixed warp data-movement primitives such as `movmatrix` only when their lane mapping matches the mathematical tensor ownership.

Use these primitives to remove address generation, redundant transfers, or synchronization—not merely to replace syntax. Confirm the generated SASS, TMA/WGMMA activity, register/shared footprint, and event-timed latency. A newer primitive is not a win when the CTA has too little work to amortize its producer/barrier machinery.

### TMA and warp specialization

TMA copies can be initiated by one elected thread and are executed asynchronously by the hardware unit. Other warps can perform independent work while the transfer is in flight. This is one instance of warp specialization, but the handoff scope must be correct: `__syncwarp()` synchronizes only one warp; producer/consumer warps in the same CTA need the matching CTA/mbarrier pipeline, and cross-CTA cooperation needs a kernel/cooperative-grid boundary or cluster-scoped primitive. The TMA transaction-byte count, stage ownership, async-proxy completion, and buffer-reuse point are part of the correctness proof.

See the modern SIMT/warp-specialization note in `$gptqmodel-cuda-kernels` for the Volta+ independent-thread-scheduling caveats that affect cross-warp handoffs, including why legacy assumptions that “all threads in a warp move together” can break.

Retain a deterministic dense/dequantized reference and test architecture dispatch independently from kernel math.

## See also

- [Curated GPU performance engineering resources](references/wafer-gpu-perf-resources.md) — External reading list from wafer-ai's performance engineering index.
