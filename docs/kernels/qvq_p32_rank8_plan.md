# Unified P32 window, rank8 and external geometry plan

Goal: complete the user's full phased window + rank8 + BM/BN plan. The
separate-reference implementation and one fused epilogue are intermediate
steps, not a redefinition of that goal. Baseline includes PR #137 at
`0cbb57c7`; its A100 crossover and timing results are not Hopper policies.

| Phase | Required deliverable | Authoritative current state | Still required |
|---|---|---|---|
| 0 | Versioned off/on numerical contract, optional per module/graph, FP32 initial accumulation | `qvq_rank8.py`, local off/on and graph tests | Full supported precision/device/TP matrix |
| 1 | Normal P32 → lossless window → original-teacher output residual → two output-aware fits in one job, document-disjoint validation | Public `quantize(rank8_capture=...)` for materialized or explicitly lazy teachers; bounded document capture and deterministic memory-capped output-range fitting, module finalizer and tiny-Llama integration; atomic SwiGLU defers fitting until the selected serialized triplet and appends factors to that payload; output-aligned modules defer fitting until final SU/SV state; real H200 Llama first-layer Q/gate fit and four-document audit are recorded; a bounded H200 run fits the full 16-layer eligible P32 set (94/94 modules) with the same disjoint document contract; exporter promotion now requires independent audit confirmation and rolls back rejected factors | Broader model families and full-model propagated quality evaluation |
| 2 | One deployment package, first-class A/B, metadata/hashes, byte/BPW reporting | Standard module buffers plus unified exporter/loader; selected-module weighted window/recovered BPW, rank8 delta and actual serialized container bytes; public tiny-Llama save/reload preserves accepted factors, validates hashes and reproduces corrected module output; applied kernel choices persist as versioned advisory metadata bound to P32/factor hashes and are revalidated on load; CUDA package loads now use explicit window-only ownership with CPU planar compatibility reconstruction | Real-model checkpoint matrix |
| 3 | Production/direct × off/on, integrated API, same transformed input, eager/graphs | H200 tests for auto/M16/row-reuse; independent factors and disabled poison checks; prepared native graph replay is now used by the ZML adapter; native transform-free composite K/N ABI path is exercised at Qwen K5120/N17408 | Explicit nested ZML user-capture replay through the public executable API and full shape/rate/repetition matrix |
| 4 | Existing Hopper WGMMA + optional rank8; H100/H200 BM/BN/warp-group/stage candidates | Existing WGMMA retained; explicit BM32/64/128, BN64/128, BK256/stages2; H200 off/on sweep; selected BN now also controls the rank8 projection/epilogue warp width and graph-warm identity | BN32 and additional stage/warp candidates, grouped explicit controls, integrate rank8 into consumer pipeline, both-device sweep |
| 5 | Shared input producer, decode producer and consumer; project rank8 alongside WGMMA | Explicit shared SU/H + rank8 producer for power-of-two K; folded composite K producer for input_hadamard=false; grouped independent factors tested; graph-safe `concurrent_reference` auxiliary-stream candidate with exact reference arithmetic and shape warm guards | Composite Hadamard input widths; SIMT/half2 vs padded-TC and FP8-factor sweep; resource-aware concurrent scheduling beyond the measured contended stream candidate; B staging/cache sweep |
| 6 | Expansion + FP32 add + output H/SV + final store; defined rounding order | `00a71f4b`: fused Triton expansion/add/H/SV/bias, independent graph-safe reference; folded composite N epilogue for output_hadamard=false; direct FP16 store; prepared FP32 A/B factor residency removes replay conversion allocations | Composite Hadamard output widths; full native pipeline fusion |
| 7 | Grouped gate/up and QKV, independent flags, optional SiLU*up, preserve Q/K norm/RoPE/TP | Grouped separate and fused output epilogues tested on H200; fused MLP now carries rank8 through gate/up, SiLU/product and completed FP32 down correction; child-local flags; checkpointed shared-input P32 groups now add each child's rank8 correction in the shared transformed-input domain before child-local output recovery, with graph replay coverage | Composite-width producer, fused SiLU/down composition beyond the reference path, real-model grouped/TP validation |
| 8 | ZML tunes eligible whole operators over geometry, transforms, quality and grouping; product-specific caches | Python static policy, shape-specific candidate enumeration, strict external controls, correctness-gated native/external executable runner and persistent revalidated caches; hash-manifested native-friendly artifact writer with descriptor-level `payload_sha256` plus typed Zig manifest loader (format/version, names, dtypes, shapes, byte counts, per-file SHA-256 and Python-compatible base/factor semantic hashes checked before upload); prepared native graph-backed ZML off/on custom call with repeated-handle replay; allocation-free `selectFastest` winner selection plus pre-capture `benchmarkExecutable` timing for direct ZML callers; the executable ZML verifier now compiles, warms, validates and measures every BM/BN candidate before capture, records matched correction-off/on overhead, applies an optional manifest-declared recovery budget, and can persist the complete off/on sweep plus selected geometry as versioned JSON bound to `artifact_payload_sha256`; the QvQ-to-ZML fixture exporter now carries that same exact package binding into `manifest.json`; gfx950 exposes immutable `QVQAMDLaunchConfig` candidates and the unified `P32WindowConfig(algorithm="amd_gfx950")`/window tuner now forwards each launch choice to the fused consumer, with cold-cache capture rejected; SM80 exposes measured-plus-probe split candidates through `qvq_p32_window_ampere_kernel_candidates`, and grouped SM80/ZML APIs expose independent per-child split tuples, `p32GroupedSelectFastest`, explicit `p32WindowMatmulGroupedWithConfig` launch control, and direct `p32GroupedAutotune` executable measurement; SM90 Python grouping now exposes shape-valid `hopper_m16` child split tuples and consumes selected splits in the grouped payload plan; native ABI now admits 256-wide transform-free composite dimensions through N17408 while retaining Hadamard guards | Non-Hopper grouped candidates, broader shape/device/quality matrix |
| 9 | fast/balanced/quality graphs, mode changes only at request boundaries | Request-owned single-GPU stateless model graphs, transactional three-mode capture, state invalidation, stream ordering and retained payload/output lifetime; tiny and real Llama graph/eager checks | Stateful generation/KV and TP scheduler integration, request overhead and graph residency measurements, broader model/shape validation |
| 10 | Combined scorecard, teacher/tail/logit/PPL/ARC/GSM8K, prefill/decode, TP, VRAM/BPW/overhead | Preparation-time rank candidate sweep (2/4/6/8/12) now records output-aware solver and serialized-FP16 scores; propagation reports now include teacher CE/PPL, temperature-one and low-temperature KL, margins, top-1/top-5/top-10/top-32 agreement and paired document bootstrap; local tests and synthetic H200 full-operator benchmark plus Nsight artifacts | Runtime dynamic-rank kernels, real disjoint model scorecard; H100/H200, M128–8192, batches1–64, TP1/2/4/8; ARC/GSM8K integration and statistical quality gates |
| 11 | Promote and ship only verified complete operator/model results | No default promotion; explicit fused mode only | Off equivalence, meaningful teacher gain, <=3–5% marginal recovery cost, no small-M regression, competitive large-M, no credible model-quality regression |

The current physical inventory provides one H200. H100 SXM/PCIe, H200 SXM
SKU comparison and multi-GPU TP performance cannot be claimed from it. That
limits those validation rows; it does not block the remaining implementation,
local tests, real-model single-GPU fitting, or H200 performance work.

Graph safety is now an explicit requirement for all QvQ kernel code and all
kernel-library integrations, not only the rank8 branch. The QvQ and zml-ultra
agent guides route these changes through `graph-safe-kernels`. Completion must
cover preparation, allocation/pool ownership, non-default streams, grouped child
lifetimes, mode/shape invalidation, repeated replay, native FFI and actual ZML
command buffers. Existing Python and native graph tests prove only their tested
paths. The managed CUDA extension, standalone Hopper WGMMA, Ampere
window/grouped operators, rank8 Triton producer/epilogue, and YAQA Triton
projector now fail closed on cold JIT/operator registration during capture;
callers must prewarm them. The managed CUDA path also requires its PGC16 level
table, bank selector validation and core GEMV/v4/Hadamard handles to be ready
before a graph is created. Host-validated CUDA Viterbi/telemetry helpers reject
capture explicitly. The ROCm YAQA trusted recurrence now applies the same
guard; its prepared internal graph remains the supported repeated path.
The native/ZML prepared-graph registry now detaches duplicate and LRU entries
under the map lock but performs CUDA graph destruction after unlock while the
entry lock is held, so teardown cannot serialize unrelated replay lanes.
Quantization kernel integrations now fail closed at the YAQA Sketch-B
collection and Gram-materialization boundary as well as at the underlying
CUDA projection/Viterbi helpers; quantization is preparation work and must
finish before capture. TP/KV execution and remaining capture-sensitive lazy
paths still require an explicit audit.

The Python tuner now supports `measure_recovery_candidates=True`, retaining
matched correction-off/on medians and marginal overhead for every eligible
geometry. The ZML adapter exposes the equivalent `benchmarkRecoveryPair` API.
Neither path lets latency override audit or arithmetic-signature eligibility.
Promotion jobs may additionally pass `max_recovery_overhead_percent` to reject
every measured geometry above the explicit 3--5% budget; the default remains
report-only because current modules do not all meet that budget.

The matched benchmark CLI exposes the same policy for reproducible scorecards:
`scripts/benchmark_qvq_window_rank8.py --autotune
--measure-recovery-candidates --max-recovery-overhead-percent 5`. The report
records the requested gate and each candidate's correction-off/on medians;
without the explicit gate, timing remains diagnostic and cannot change quality
or arithmetic-signature eligibility.

The native prepared-graph API owns its temporary allocation pool and can insert
the existing window/rank8 operator as a child of an enclosing CUDA capture. The
ZML adapter now retains one such handle per executable buffer set, stream and
static configuration. Its first eager call prepares the handle and later calls
reuse it; the adapter advertises command-buffer compatibility for this prepared
path and rejects capture-before-warmup. The normal ZML executable
command-buffer path has been exercised by the StableHLO verifier with two calls
per correction mode. Explicit nested user capture remains an additional
validation item, so the adapter evidence does not claim that case is complete.

The registry binds addresses together with byte sizes, element dtypes, the PJRT
device ordinal and stream, uses yielding/per-entry locks, bounds residency with
LRU eviction, and destroys evicted handles before releasing their private pools.
Eviction probes entry locks without waiting under the global map lock; when all
entries are busy it fails with a bounded resource error. Runtime-owned handles
must still outlive all enclosing executables and graphs.

That external verifier was rerun against the current ZML source build
`8d67a352` with the CUDA PJRT/StableHLO bundle on the H200. Its release build
produced the GPU PJRT plugin and runtime sandbox, and the real M33/K=N2048
fixture remained bit-exact for both rank8-off and rank8-on executable calls.
This strengthens the public ZML command-buffer evidence without changing the
scope: explicit user-owned nested capture, TP/KV execution and H100 coverage
remain open.

The prepared native graph passed 350 focused H200 tests, including 168 graph
cases across four rates, M1/33/128, M16 and all six BM/BN choices, correction
off/on, disabled invalid factor pointers, repeated changed inputs, allocator
pressure and enclosing stream dependencies. The real Q projection remains
bit-exact. The matched profile retains the same 16 kernels and resources.
See `results/p32_window_native_graphs.json` for the exact tested scope and
remaining ownership obligations.

```text
H200 physical 0, real Q, FP16, M33 K2048 N2048, W2, BM64 BN64, rank8 on
Native ordinary entry      81.42 us median
Native prepared graph      66.39 us median
Enclosing CUDA graph       66.66 us median
```

These timings include the same full native operator. They are not ZML, model,
large-M recovery-overhead or additional-fusion measurements.

The explicit `recovery_projection="input_fused"` candidate now publishes the
exact FP16 transformed activation and up to three child projections from one
CTA per row. The window consumer still reads that published activation in a
separate launch; this is not the final concurrent WGMMA pipeline. It passed
116 focused H200 tests, including grouped independent flags, graph replay,
non-default stream and the deployed CUDA input overflow boundary.

Matched H200 post-profile measurements are recorded in
`results/p32_rank8_h200_producer.json`. For K=N=2048 W3, the shared producer
improves the recovered full operator at M16/128 but regresses M1/512/2048.
It stays explicitly selectable, with no default promotion. At M16 it removes
five launches and reduces source-correlated executed instructions from
2,486,932 to 1,615,856. The producer has 40 registers, no spills, and substantial
shared exchange/reduction work; instruction reduction does not establish a
large-M latency improvement.

The Ampere continuous-window operator now participates in the same
`P32WindowConfig` and `window_kernel_candidates` policy. On SM80, the
shape-specific split-wave enumerator publishes explicit `ampere_window`
candidates, preparation prewarms the operator, and replay passes the selected
`split_k` without re-autotuning. `grouped_window_kernel_candidates` exposes the
same shape policy as tuples of explicit child configs, so the grouped tuner can
select one complete launch while retaining independent split waves. Grouped
runtimes consume those child policies through the existing segmented/fused SM80
dispatcher; their packed payload and PGC16 level table are prepared once before
capture and retained by the grouped runtime. Mixed grouped consumers fail
closed rather than silently ignoring a child policy. The candidate API remains
pure and graph-safe; ZML's existing split-count tuner can consume the same
explicit shape policy. On SM90, grouped candidates now use `hopper_m16` with
shape-valid child-local split tuples; the grouped payload builder consumes
those selected splits, while unsupported single-child BM/BN controls fail
closed instead of being ignored.

The six existing Hopper geometry choices are now externally selectable,
and their H200 off/on sweep is recorded in `results/p32_window_h200_geometry.json`.
BM128/BN128 improves M2048 but loses at smaller M. The API retains every
supported geometry rather than pruning it based on another shape. Next:
connect the tuning runner to the native ZML executable and extend KV/TP graph
ownership and backend/grouped candidate coverage; improve the producer's
shared-memory exchange and factor reuse, then evaluate padded Tensor Core
projection and integration with the WGMMA producer/consumer pipeline. The
full phase requirements above remain open.

Explicit native and 4096-row-chunked FP16 policies now cover M through 8192;
the retained `results/p32_window_h200_m8192.json` contains the 80-candidate
H200 run and matched post-profile instruction/timing comparison. Rank-16-padded
Tensor Core projection is also implemented as an explicit candidate, with
independent FP32 projection references and grouped/single graph tests. This
remains a separate projection launch, not concurrent recovery within WGMMA.
`results/p32_rank8_h200_tensor_core.json` records its post-profile comparison,
executed HMMA/resource audit and expanded 126-candidate tuning run. Synthetic
M8192 correction overhead is 0.6% for the measured K=N=2048 W3 fixture, while
M2048 remains 7.4%; this does not close the full-model promotion gate.

Real first-layer Q/gate fitting and a separate four-document audit are now
recorded in `results/p32_rank8_llama_first_layer.json`. Aggregate teacher MSE
improves 11.16%/5.23%; individual maximum-error regressions remain visible.
Real-factor/activation M8192 timing gives 0.61% overhead for square Q and
5.73% for wide gate. These results advance module evidence only; whole-model
quality and the broad performance/promotion requirements remain open.

The direct final FP16 store is now implemented for eligible single/grouped
fused epilogues and passes 405 post-profile tests (86 skips). The C4 validation
subset covers 128 documents/47,550 predictions with fixed first-layer Q/gate
corrections. It shows a small loss benefit over correction off; the padded
Tensor Core projection loses some of the reference correction's benefit,
while the reference-projection/fused-epilogue ablation preserves it. Keep
both implementations explicit. Broader model quality, ZML executable tuning, KV/TP graph ownership and other phase rows
remain open; this subset is not a full-model promotion scorecard.

Shared correction-off/on transform/store fusion is now implemented and
validated with 414 passing post-profile tests (86 skips), poisoned disabled
factor cases and bit-exact off logits on all 128 C4 subset documents. The
fair matched M8192 marginal costs are Q 6.07% and gate 3.27%, superseding
the earlier comparisons with a less optimized off epilogue. The retained
`results/p32_window_h200_shared_epilogue.json` also contains a 144-candidate
real-Q tuning run: production/shared-input wins at M128, BM128/BN128 with
Tensor Core projection at M2048. Local numerical acceptance does not waive
the measured Tensor Core projection model-quality difference. No universal
geometry or quality implementation is promoted.

The request graph owner now captures fast/balanced/quality as independent CUDA
graphs with externally supplied per-module geometry. A four-document real
Llama checkpoint check preserves eager logits exactly in every mode. This
advances phase 9; it does not complete KV/generation, TP or serving performance
requirements. Full evidence and the captured-mask contract are recorded in
`results/p32_window_llama_graphs.json` and the runtime documentation.

An initial native C ABI and actual ZML StableHLO custom-call adapter now execute
the existing Hopper window operator with optional reference rank8. The real Q
fixture is bit-exact through ZML in both correction states, with two calls per
mode reusing the prepared native graph handle. All six existing BM/BN choices
are exposed as native controls. This is not the final fused or TP-aware ZML
runtime: explicit nested ZML user capture and latency tuner remain open. The
native artifact loader now validates typed descriptors, payload/per-file hashes,
and Python-compatible base/factor semantic hashes before upload. The M33 native-entry timing beats Python eager but loses to the
captured Python operator. See `results/p32_window_native_zml.json` and
`../../integrations/zml/README.md` for the supported contract and reproducible
build/execution commands.

The current native rank8 ABI now combines FP32 base and rank8 expansion with one `addmm` epilogue after the explicit FP16 hidden boundary. A fresh H200 ZML verifier run compiled, warmed, correctness-checked, and timed every M16/BM/BN candidate against the current library; the complete off/on report is recorded in `results/p32_window_native_zml_addmm.json`. The selected M33 rank8 path measured 3.848% overhead for M16, while BM/BN choices ranged above and below that value, confirming that recovery cost must remain a shape-specific tuning and promotion gate.
A second verifier run passed `--max-recovery-overhead-percent=5`; its selected candidate and rejected over-budget rows are recorded in `results/p32_window_native_zml_addmm_budget5.json`. The gate is applied before winner selection and leaves quantizer quality and arithmetic eligibility unchanged.

The current ZML verifier now emits schema-versioned arithmetic policy in every
tuning report. The H200 rerun at ZML `6265dd5` records `fast` selection for the
correction-off graph and `quality` selection for correction-on, with every
candidate carrying its arithmetic signature. The complete 14-row report is
`results/p32_window_native_zml_policy.json`; it is evidence that timing and
recovery-budget gates remain subordinate to the declared numerical contract.

After widening the native ABI to admit transform-free composite dimensions,
the fresh-library H200 rerun is recorded in
`results/p32_window_native_zml_current.json`. It includes the real
K5120/N17408 decoder compared with the QvQ reference, plus the updated
StableHLO rank8-off/on replay.

Atomic SwiGLU selection now defers rank8 fitting until the complete
gate/up/down candidate triplet has been selected. The fit consumes the
selected serialized payload and immutable dense teacher snapshot, then adds
`rank8_A`, `rank8_B`, and `rank8_metadata` to that same payload before
staging. This prevents candidate-zero factors from being paired with a
different selected bank arm. Output-aligned modules consume the final aligned
SU/SV payload before fitting. The full eligible-module H200 fit/audit evidence
is now recorded separately; broader model families and propagated full-model
quality evaluation remain open.

A fresh current-code Q-projection replay scorecard is recorded in
`results/p32_rank8_qproj_h200_m128_8192_current.json`. It uses a newly accepted
four-document-audit package and fixed audit activation buffers at M=128, 2048,
and 8192. The separate-reference correction costs 64.04%, 24.09%, and 23.43%
respectively; the fused epilogue costs 44.81%, 16.92%, and 17.01%. These are
complete graph-replay timings on the same serialized factors, not synthetic
algebra. They show that large-M rank8 is not generally free for this Q module
and that the 3--5% promotion target still requires concurrent producer/consumer
fusion and shape-specific tuning.

The full eligible-module H200 run is now recorded in
`results/p32_rank8_llama_full_model.json`. It fits 94/94 P32 modules across
all 16 layers using eight train, four selection, and four audit documents,
with 64 retained rows per document, a 2 GiB capture bound, and a 256 MiB
per-module solver bound. Mean audit MSE/MAE/tail error falls by 5.71%/3.29%/
2.71% over the window baseline. The 94 packages contain 349,869,578 tensor
bytes (3.11 window BPW and 3.20 recovered BPW); serialized package bytes are
reported separately. W4 output projections and the W4 up projections in
layers 6 and 8 are outside the P32/A16 contract and were intentionally
omitted. This is fit provenance, not a full-model perplexity/task or promotion
scorecard.
