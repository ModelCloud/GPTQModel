# Clang and LLVM: practical SSA analysis of CUDA

## Primary sources

- [CUDA compilation with Clang](https://llvm.org/docs/CompileCudaWithLLVM.html)
  and [command-line reference](https://clang.llvm.org/docs/ClangCommandLineReference.html).
- [LLVM analysis and transform passes](https://llvm.org/docs/Passes.html),
  [opt](https://llvm.org/docs/CommandGuide/opt.html) and
  [new pass manager](https://llvm.org/docs/NewPassManager.html).
- [MemorySSA](https://llvm.org/docs/MemorySSA.html) and
  [NVPTX backend](https://llvm.org/docs/NVPTXUsage.html).

Clang supplies a CUDA frontend, and LLVM's NVPTX backend lowers device IR to PTX.
LLVM has analysis passes for value/loop/memory reasoning and transform passes
for simplification. MemorySSA represents memory dependencies; it is not a
GPU-wide race detector. Pass availability should be queried from the selected
`opt` build rather than inferred from an old pass list.

## Which analyses address QVQ's question?

| Facility | Proposed question |
|---|---|
| InstCombine / EarlyCSE / GVN | Are masks, conversions, expressions or safe redundant loads removable? |
| SCCP / SimplifyCFG | Do fixed specialization values eliminate branches and computations? |
| ScalarEvolution / loop analyses | Are offsets and induction expressions repeated or reducible? |
| Alias analysis + MemorySSA | Can a load be reused given intervening writes and calls? |
| LICM | Can invariant work move outside a loop without changing effects? |
| Optimization remarks and IR diffs | Which candidate simplifications occurred or were missed? |

These facilities already participate in optimized compilation. Running an extra
pass does not automatically discover a missed optimization or improve code.

## Proposed analysis-only recipe

Use a matched, pinned Clang/LLVM pair with NVPTX support and a CUDA toolkit it
accepts. The following is a starting command template, **not a successful QVQ
compile from this survey**. It targets the audited SM80 translation unit:

```bash
mkdir -p artifacts/static-analysis
clang++ -x cuda --cuda-device-only --cuda-gpu-arch=sm_80 \
  --cuda-path=/usr/local/cuda -std=c++17 -O3 \
  -S -emit-llvm \
  -I gptqmodel_ext/qvq/p32 \
  gptqmodel_ext/qvq/p32/qvq_p32_cuda.cu \
  -o artifacts/static-analysis/p32.device.ll

opt --print-passes
opt -passes='function(print<scalar-evolution>,print<memoryssa>)' \
  -disable-output artifacts/static-analysis/p32.device.ll \
  2> artifacts/static-analysis/p32.analysis.txt

opt -passes='function(instcombine,early-cse,gvn,simplifycfg)' \
  -S artifacts/static-analysis/p32.device.ll \
  -o artifacts/static-analysis/p32.simplified.ll
```

Choose the real CUDA path and target for each audited build; do not bypass a
toolchain incompatibility by silently removing headers or CUDA operations.
Match relevant macros, include paths, numeric flags and instantiated templates.
Query pass names before using this template. Retain diagnostics if it fails.

The first file is already optimized IR. The second pipeline is an exploratory
comparison, not the complete production O3 pipeline or a guaranteed improvement.
For attribution, capture relevant before/after pass IR or optimization remarks
from the same compile; an O0 comparison alone cannot identify a specific pass.

## Inline PTX and production-build limits

The [current source](https://github.com/ModelCloud/QvQ/blob/66565c27ed8a42639c0c2bbe55fdb4a8e677dca0/gptqmodel_ext/qvq/p32/qvq_p32_cuda.cu) includes volatile inline assembly for
matrix fragments, MMA and asynchronous transfers. LLVM sees the operands and
declared effects, not ordinary arithmetic corresponding to the assembly body.
An intrinsic is analyzable only to the extent that the pass or validator models
its semantics.

Never erase volatile, convergent, memory or synchronization effects to make a
candidate appear optimizable. Inspect incomplete declarations as correctness
questions, not optimization opportunities.

The [production target](https://github.com/ModelCloud/QvQ/blob/66565c27ed8a42639c0c2bbe55fdb4a8e677dca0/gptqmodel_ext/qvq/p32/BUILD.bazel) uses NVCC, so Clang IR does not prove
what NVCC/ptxas emits. Keep the existing build as the runtime baseline. First
inspect pure integer helpers or small extracted specializations; quantify what
fraction of the intended operation the analysis actually covers.

## What counts as evidence?

Save toolchain versions, exact command, source SHA, target, specialization,
numeric flags, IR hashes and function-level changes. A smaller IR can still
increase registers, spills or instruction latency after lowering.

Use [Alive2/Z3](rewrite-verification.md) where their supported semantics match
the rewrite. Then check the production [SASS](ssa-sass.md) and
[Nsight](nsight-profiling.md) evidence. Static load reuse analysis does not
authorize changing barrier protocols; a successful `opt` run is neither a
whole-kernel proof nor a performance result.
