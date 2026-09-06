# Native ROCm YAQA quantization: baseline and port boundary

User request: native QVQ/YAQA quantization on this AMD ROCm GPU, targeting
100x versus the equivalent Torch implementation. This is quantization work,
not the previous inference1.5x task. Treat `vaqa` as the repository's YAQA
path unless clarified. A core-only win does not establish the end-to-end
YAQA target. No speedup is achieved or claimed by this initial phase.

## Source and hardware

PR116 was confirmed merged at41ef31b6. Origin has no master branch; its
default is main. New branch codex/qvq-rocm-quantization-wip starts from
origin/main bc36a5ba0f5123192e4796168da7430aa875ea38.
Guard/baseline implementation7beba298. Previous unrelated artifacts remain
untouched. MI355XVF gfx950,256CUs, physicalGPU0, BDF0000:83:00.0,
unique0x333ef6e01ec019b3. Exact software/source hashes are in the JSON.

The unmodified public banked Viterbi call failed on ROCm: HIP reports device
capability(9,5), the generic `device.type == cuda && capability >= (8,0)`
guard selected the NVIDIA extension, and that extension rejected ROCm.
The guard now explicitly excludes HIP before probing NVIDIA capability.
Strict required-pruning policies still reject rather than silently falling
back. This enables the existing eager banked reference only; it does not
claim native ROCm quantization or full YAQA support. Other YAQA CUDA-specific
family/feedback/tail-biting dispatch sites still require auditing/porting.

## Independent correctness and eager timing

Eight real-GPU zero-tie cases cover W2/2.5/3/3.5 and FP16/FP32 codebooks:
all states, reconstructed values, loss and eight P32 segment banks are zero.
A patched capability probe raises if touched, proving HIP does not use that
NVIDIA selection rule. Two additional cases preserve strict pruning errors.
The initial test mistakenly expected four segments; P32 is32 scalar weights,
or16V2 steps, so128steps contain eight segments. Only that test expectation
was corrected; the recurrence/output was not changed.

All51 focused/pruning-config tests passed after profiling,14existing
deprecation warnings,4.94seconds. New benchmark/test lint and whitespace
checks pass. Baseline fixtures are synthetic and are not model-quality data.

The eager reference includes validation, emissions, exact min-sum recurrence,
bank boundaries, traceback and returned values. Shapes: [B,128,2] FP32
sequences and[2,65536,2] FP32 codebooks. Rates and seed are recorded per row.
Warmup1, three synchronized event/wall samples; exact repeated states,
banks, loss and values pass. No candidate/oracle mutation or precision
relaxation is introduced.

| B tiles | W2 ms | W2.5 ms | W3 ms | W3.5 ms |
|---|---:|---:|---:|---:|
| 1 | 17.1151 | 18.1149 | 17.1384 | 16.9950 |
| 4 | 17.3430 | 17.1848 | 17.1162 | 17.1726 |

All eight Torch traces contain2350 GPU launches per call. Summed profiled
GPU durations are7.03..9.96ms; these perturbed trace sums are not warmed
latencies or an overlap percentage. B1W2 includes127 prefix-min reductions,
128 two-component dot GEMMs,256 runtime copies and numerous pointwise/index
kernels. End-to-end timing is also affected by launch/host gaps. No new
generated AMD kernel was introduced, so this phase has operator mapping,
not a native instruction-counter or ISA-speedup claim.

## Exact native recurrence design to test next

Let P=2^(2*bits), S=65536/P, F_t[b,s] be the original cost after emission
E_t[b,s], and R_t[b,q]=min_p F_t[b,p*S+q]. The existing transition uses
R_(t-1)[b,s>>log2(P)] inside a segment. At a boundary use the minimum over
prior banks, preserving lexicographic(bank,prefix) tie order.

Candidate representation: retain R rather than materializing all F between
steps; compute each next R by evaluating the same emissions/cost additions
and reducing over p. For W2, two banks require8192 FP32 compressed costs
(32KiB) per buffer versus131072 full costs (512KiB). Two compressed buffers
can fit in gfx950 LDS. This is a design hypothesis, not an implemented or
measured native kernel. Register demand, inter-wave exchange, occupancy,
batch-parallel scheduling and traceback bandwidth still need profiling.

Crucial correctness details:

- Preserve the Torch emission expression and its actual rounding/FMA
  boundaries; FP32 alone does not prove identical paths.
- Save predecessor prefixes and boundary-bank decisions needed for traceback.
- Endpoint argmin must compare original(bank,state) order, not compressed
  suffix order. Lowest suffix is not necessarily the lowest original state.
- Preserve initial/final overlap restrictions, weighted emissions, all-zero
  ties, finite/range checks and deterministic packed-word/metadata output.
- A changed quantization decision cannot be excused by inference tolerances.
- Do not copy implementation code from GPL reference projects; derive from
  repository mathematics and test against the independent eager reference.

Port stages: exact banked Viterbi/emission/traceback first; tail-biting and
family selection second; YAQA Hessian-feedback schedule third; then real
weight/calibration quantize-save-reload and full endpoint timing. Benchmark
the same inputs, rates, sample strategy, family mode, iteration counts and
calibration policy. Do not claim a core100x result as full quantization100x.

Artifacts: yaqa_rocm_7beba298_baseline.json and
yaqa_rocm_7beba298_mapping.json. Raw traces:
/tmp/qvq-yaqa-rocm-baseline-traces. Logs:
/tmp/qvq-yaqa-rocm-baseline.log and
/tmp/qvq-yaqa-amd-post-profile-tests.log. Both strict idle and per-timing
PID checks passed; no unrelated process was terminated. The accepted
benchmark completed with exit0.

Reproduce with scripts/benchmark_qvq_yaqa_amd.py --batch-sizes1 4
--bits2 2.5 3 3.5 --warmup1 --iterations3 --profile-dir<path> --output<path>,
using spaces between flags/values and CUDA_DEVICE_ORDER=PCI_BUS_ID,
HIP_VISIBLE_DEVICES=0. Larger batches, other semantic cases, native
profiling and end-to-end YAQA benchmarks remain outstanding.
