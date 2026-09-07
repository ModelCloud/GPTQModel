# PTQ recovery research

This directory records scientific findings that affect QVQ quantization, recovery,
and inference. Read the relevant note before changing the corresponding contract.
It is a research index, not a claim that every referenced method is implemented,
validated on QVQ, or enabled in production.

## Reading map

Start with the [supported-method and enhancement inventory](supported-methods.md)
for enum/config coverage, dispatch boundaries, and implementation status.

| Topic | Note | Role |
|---|---|---|
| GPTQ | [GPTQ](gptq.md) | Second-order weight rounding and compensation |
| AWQ | [AWQ](awq.md) | Activation-aware channel scaling for weight quantization |
| QQQ | [QQQ](qqq.md) | W4A8 smoothing and Hessian compensation |
| ParoQuant | [ParoQuant](paroquant.md) | Learned pairwise rotations and transformed-domain quantization |
| EXL3 | [EXL3](exl3.md) | ExLlamaV3 quantization integration and exclusions |
| RTN | [RTN](rtn.md) | Calibration-free baseline/fallback; not a separate METHOD |
| FP8 | [FP8](fp8.md) | Weight format versus activation/cache policy |
| bitsandbytes | [FP4/NF4 integration](bitsandbytes.md) | Blockwise storage and runtime; distinct from QLoRA training |
| GGUF | [GGUF](gguf.md) | Container, tensor types and supported consumers |
| MXFP4 | [MXFP4](mxfp4.md) | Microscaling and CPU module integration |
| GPTAQ / former GPTQv2 | [GPTAQ](gptaq.md) | Asymmetric calibration for upstream error |
| FOEM | [FOEM](foem.md) | First-order weight-error compensation |
| GAR | [GAR](gar.md) | Group-aware reordering |
| SLQ | [SLQ](slq.md) | Statistical fidelity and nonuniform allocation |
| Rotation / smoothing | [Rotation and smoothing](rotation-and-smoothing.md) | Equivalent transforms versus lossy preprocessing |
| QVQ enhancements | [Implementation findings](qvq-enhancements.md) | SwiGLU, alignment, propagation, candidate search and exact pruning |
| NVFP4 W4A4 and FP8 KV calibration | [NVFP4 hybrid PTQ](nvfp4-hybrid-ptq.md) | Numeric representation, fusion correctness, calibration |
| EoRA and output-residual fitting | [EoRA recovery](eora.md) | Additive low-rank compensation |
| QTIP | [Trellis quantization](qtip.md) | Weight quantizer and parallel-decodable representation |
| YAQA (requested as “VAQA”) | [Model-preserving rounding](yaqa.md) | Full-model-sensitive weight rounding |
| QVQ V2B2-P32 | [P32 and lossless windows](p32.md) | Repository format and runtime contract |
| Google DeepMind Recirculation | [Recirculation](recirculation.md) | Inference-time state intervention; PTQ benefit unproven |

“VAQA” is interpreted here as **YAQA**, consistent with the repository's
`qvq_yaqa.py` and the linked paper. No separate VAQA publication is asserted.
P32 is documented as a QVQ implementation, not an independently identified paper.

## Compiler and GPU research

These are compiler, execution and measurement topics, not additional quantization
METHOD members. “stablehalo” is interpreted as **StableHLO**. SSA and SASS refer
to different stages of compilation.

| Topic | Note | Role |
|---|---|---|
| ZML | [ZML](zml.md) | Symbolic model construction and device execution |
| XLA | [XLA](xla.md) | Graph optimization, backend lowering and tuning boundaries |
| StableHLO | [StableHLO](stablehlo.md) | Operation semantics and compiler portability |
| SSA / SASS | [Analysis and folding](ssa-sass.md) | IR simplification versus emitted GPU instructions |
| Deployable static-analysis tools | [Open-source tool assessment](cuda-static-analysis-tools.md) | Priorities, licenses, source snapshots and CUDA coverage limits |
| LLVM CUDA inspection | [Clang/LLVM analysis](llvm-cuda-analysis.md) | Device IR, analysis passes and an untested starting recipe |
| Exact rewrite checking | [Alive2, Z3 and Souper](rewrite-verification.md) | Refinement, bit-vector proofs and integer rewrite search |
| Numerical analysis | [Daisy, FPTaylor and Herbie](floating-point-analysis-tools.md) | Error bounds versus candidate generation |
| Structural algebra | [MLIR, Polygeist, Polly and symbolic tools](mlir-algebraic-analysis.md) | Graph/loop analysis and rewrite exploration |
| Concurrency verification | [GPUVerify](gpuverify.md) | Race and barrier-divergence checks; modern CUDA coverage unverified |
| Nsight | [Profiling workflow](nsight-profiling.md) | System timeline, kernel counters and replay limits |
| LUT costs | [CUDA lookup tradeoffs](cuda-lut-tradeoffs.md) | Bank conflicts, dependency latency, footprint and existing QVQ evidence |
| Metric interpretation | [Metrics versus performance](cuda-metrics-and-performance.md) | No single counter is a sufficient speedup criterion |
| Pipeline throughput | [Useful work and overlap](cuda-pipeline-throughput.md) | Resource bottlenecks, stage ownership and critical waits |
| Efficient LUT quantization | [FLUTE](flute.md) | Bank-aware/vectorized lookup design |
| Layout and write-back | [QUICK](quick.md) | Quantization-aware interleaving to avoid shared write-back |
| LUT-based computation | [LUT-GEMM](lut-gemm.md) | Algorithm-level lookup reuse; distinct from a P32 decoder LUT |
| CUDA execution | [SMs, Tensor Cores, TMA, async and barriers](cuda-execution.md) | Architecture capabilities and pipeline correctness |
| SSA foundations | [Cytron et al.](ssa-paper.md) | Program representations for optimization |
| Equality saturation | [egg](egg.md) | Exploring equivalent expressions with valid rewrite rules |
| Performance modeling | [Roofline](roofline.md) | Operational intensity and compute/bandwidth bounds |
| Asynchronous attention | [FlashAttention-3](flashattention-3.md) | Hopper overlap and low-precision attention research |
| Task scheduling | [Task-Based Tensor Computations](task-based-tensor-computations.md) | Coordination of asynchronous GPU units |

The compiler/GPU notes reference official documentation reviewed on 2026-09-06
and QVQ's P32 ABI at merged main
[`263ed4b`](https://github.com/ModelCloud/QvQ/tree/263ed4baf7be5e9547b4e731c9031bef5f48cf69).
They distinguish proposed integration from implemented behavior; no new compiler,
GPU profiling or model-quality experiments were run for these notes.

The static-analysis tool survey and LUT/metric/pipeline follow-up use main
[`66565c2`](https://github.com/ModelCloud/QvQ/tree/66565c27ed8a42639c0c2bbe55fdb4a8e677dca0).
The tool survey records source-reviewed candidates, not installed integrations.
The LUT note links existing experiment results and preserves their unresolved
causal attribution; no new analyzer run or GPU measurement is claimed.

## How the pieces relate

Quantizer choice, rounding objective, numeric scales, storage layout, additive
recovery, and recurrence are different axes. A lossless layout conversion cannot
recover quantization error. A fitted low-rank correction cannot replace a required
NVFP4 scale. An inference intervention can improve a task while moving away from
the original teacher; report those outcomes separately.

For proposed W4A4 recovery, distinguish:

- Calibration: estimating quantization parameters from representative inputs.
- Reconstruction fitting: solving for weights or factors against a specified target.
- QAT/distillation: optimizing with a quantized forward path and a training objective.
- Runtime quantization: producing input-dependent codes and block scales.

Calling all four “training scales” loses the distinction needed for implementation.

## Evidence convention

Each note separates **source findings**, **repository evidence**, and **QVQ
implications or proposed experiments**. Paper numbers are author-reported unless
a linked repository report explicitly reproduces them. Cite the paper version and
section, pin implementation evidence to a commit, and record scope and limitations.
Do not copy entire papers or treat an abstract's headline as a universal guarantee.

The initial QVQ source audit is pinned to
[`4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac`](https://github.com/ModelCloud/QvQ/tree/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac).
Verify current code before acting on an implementation statement. Source review
for these notes did not run GPU or model-quality experiments.

## Experiment records

Keep run-specific results in the existing `docs/experiments/` and `artifacts/`
locations; link the evidence here when a durable finding is established. Record
model/checkpoint revision, exact operator and activation domain, W/A/KV precision,
scale convention, correction rank/dtype, calibration/evaluation split, hardware,
backend, and source revision.

Use matched-input kernel references for correctness and disjoint real-model
evaluation for quality. Preserve the acceptance rules in [AGENTS.md](../AGENTS.md).
Measure conversion, transforms, correction, and addition in full-operator timing;
report prefill and decode separately. Include scale, bank, padding, and correction
storage in effective BPW. A note or successful reference fit does not promote a
runtime default.

Agent entry point:
[PTQ recovery research skill](../.agents/skills/qvq-ptq-recovery-research/SKILL.md).
