# NVIDIA A100+ agent-skill audit

Date: 2026-09-24

Scope: all repository agent skills were inventoried. CUDA/performance skills were
reviewed for NVIDIA A100-and-newer correctness and tuning guidance. Skills whose
domain is CPU packing, tokenizer/model registration, AMD/ROCm, Apple/Metal,
evaluation, provenance, or generic workflow were intentionally not polluted with
NVIDIA-specific rules unless they directly participate in CUDA execution.

## Primary target policy

- NVIDIA performance target: A100/Ampere (cc 8.0) and newer.
- Hopper and Blackwell architecture-specific paths must keep explicit runtime and
  build gates.
- Pre-Ampere optimization is out of scope unless requested.
- Existing CPU/non-target fallbacks remain supported; AMD/Metal optimization
  stays in their dedicated skills.

## Deficiencies corrected

1. **No shared modern CUDA contract.** Added one A100+ reference covering global
   coalescing, vector alignment, shared banks, async copy, TMA/WGMMA/TCGen05,
   synchronization scope, streams/graphs, occupancy and SASS proof.
2. **No Blackwell skill.** Added a dedicated Blackwell workflow for cc 10.x/12.x,
   TCGen05, Tensor Memory, TMA, architecture-family targets and block-scaled
   narrow formats.
3. **Cross-warp synchronization ambiguity.** Explicitly states that
   `__syncwarp()` synchronizes only one warp; cross-warp shared handoffs need
   CTA barriers/mbarriers and cross-CTA dependencies need a wider primitive.
4. **Shared-memory folklore.** Standardized 32 banks / 4-byte granularity,
   highlighted FP16 pair ownership, separated regular padding/swizzle from
   arbitrary LUT replication, and made whole-CTA SMEM pressure part of the gate.
5. **Global-memory folklore.** Removed "32 floats = one transaction" reasoning;
   agents must inspect lane addresses, 32-byte sector accounting, useful bytes
   and natural vector alignment.
6. **Tensor Core shape overgeneralization.** Removed blanket FP16/INT8 divisibility
   claims; prove the exact library/instruction path from generated code.
7. **Contiguity overuse.** `.contiguous()` is no longer the default answer.
   Agents compare stride-aware kernels, prepared repacks and materialization cost.
8. **Occupancy cargo culting.** Removed automatic register-cap advice. Occupancy
   is treated as latency-hiding capacity; spills, eligible warps and reuse decide.
9. **Low bandwidth misdiagnosis.** Low HBM utilization is explicitly not an
   alignment diagnosis for compressed/decode-heavy kernels.
10. **Architecture-specific pipeline gaps.** Added Ampere `cp.async`, Hopper
    TMA/WGMMA, and Blackwell TCGen05/TMEM ownership/barrier rules.
11. **Stream/graph ambiguity.** Current-stream ownership, cross-stream events,
    prepared descriptors/repacks, and replay lifetimes are now explicit.
12. **External tuning identity.** QVQ/ZML cache/ABI guidance now includes
    architecture target, swizzle/layout, stage count, CTA/cluster geometry and
    related code-generation axes.

## Skills materially enhanced

- `gptqmodel-cuda-kernels` and its kernel/debug workflow references
- `gptqmodel-ampere-kernels`
- `gptqmodel-hopper-kernels`
- new `gptqmodel-blackwell-kernels`
- `gptqmodel-contiguous-memory`
- `gptqmodel-inference-fusion`
- `gptqmodel-mega-kernels` and its playbook
- `gptqmodel-gpu-profiling`
- `gptqmodel-gpu-testing`
- `graph-safe-kernels`
- `perf-workload-profiling`
- `perf-nsight-compute-analysis` memory/bottleneck guidance
- `qvq-zml-kernel-contract`
- root `AGENTS.md` routing

## Intentionally unchanged domains

The audit found no reason to inject A100+ kernel mechanics into quantization
algorithm math, CPU packing, tokenizer normalization, model adapters, checkpoint
format semantics, AMD/ROCm-specific skills, Metal profiling, evaluation,
provenance, git synchronization, or generic telemetry skills. Those remain
separate domains and should call the CUDA architecture skills only when actual
NVIDIA kernel work enters their scope.
