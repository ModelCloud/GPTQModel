# Unified P32 window, rank8 and external geometry plan

Goal: complete the user's full phased window + rank8 + BM/BN plan. The
separate-reference implementation and one fused epilogue are intermediate
steps, not a redefinition of that goal. Baseline includes PR #137 at
`0cbb57c7`; its A100 crossover and timing results are not Hopper policies.

| Phase | Required deliverable | Authoritative current state | Still required |
|---|---|---|---|
| 0 | Versioned off/on numerical contract, optional per module/graph, FP32 initial accumulation | `qvq_rank8.py`, local off/on and graph tests | Full supported precision/device/TP matrix |
| 1 | Normal P32 → lossless window → original-teacher output residual → two output-aware fits in one job, document-disjoint validation | `quantize_qvq_linear(rank8_calibration=...)` and processor attachment; actual tiny quantize/fit/save/load smoke | Automatic document capture in public model quantization, bounded scalable solver, alignment/atomic finalization integration, real-model fit evidence |
| 2 | One deployment package, first-class A/B, metadata/hashes, byte/BPW reporting | Standard module buffers plus unified window exporter, loader, SHA256 checks and byte inventory | Public model writer/loader end-to-end, whole-model weighted BPW and all serialized bytes |
| 3 | Production/direct × off/on, integrated API, same transformed input, eager/graphs | H200 tests for auto/M16/row-reuse; independent factors and disabled poison checks | Native ABI3 operator surface and full shape/rate/repetition matrix |
| 4 | Existing Hopper WGMMA + optional rank8; H100/H200 BM/BN/warp-group/stage candidates | Existing WGMMA retained; explicit BM32/64/128, BN64/128, BK256/stages2; H200 off/on sweep | BN32 and additional stage/warp candidates, grouped explicit controls, integrate rank8 into consumer pipeline, both-device sweep |
| 5 | Shared input producer, decode producer and consumer; project rank8 alongside WGMMA | Explicit shared SU/H + rank8 producer for power-of-two K; grouped independent factors tested | Composite input widths; SIMT/half2 vs padded-TC and FP8-factor sweep; concurrent scheduling; B staging/cache sweep |
| 6 | Expansion + FP32 add + output H/SV + final store; defined rounding order | `00a71f4b`: fused Triton expansion/add/H/SV/bias, independent graph-safe reference | Integrate input projection, direct final FP16 store, composite output widths, full native pipeline fusion |
| 7 | Grouped gate/up and QKV, independent flags, optional SiLU*up, preserve Q/K norm/RoPE/TP | Grouped separate and fused output epilogues tested on H200; child-local flags | Composite-width producer, fused SiLU/down composition, real-model grouped/TP validation |
| 8 | ZML tunes eligible whole operators over geometry, transforms, quality and grouping; product-specific caches | Python static policy, shape-specific candidate enumeration, strict external controls, correctness-gated native/external executable runner and persistent revalidated caches; H200 90-candidate run | Actual ZML/StableHLO/native ABI adapter, non-Hopper backend candidate providers, grouped candidates, broader shape/device/quality matrix |
| 9 | fast/balanced/quality graphs, mode changes only at request boundaries | Prepared module policies; separate captured on/off graphs tested | Model-wide graph manager and request-boundary ownership, atomic policy changes and cache invalidation |
| 10 | Combined scorecard, teacher/tail/logit/PPL/ARC/GSM8K, prefill/decode, TP, VRAM/BPW/overhead | Local tests and synthetic H200 full-operator benchmark plus Nsight artifacts | Real disjoint model scorecard; H100/H200, M128–8192, batches1–64, TP1/2/4/8; statistical quality gates |
| 11 | Promote and ship only verified complete operator/model results | No default promotion; explicit fused mode only | Off equivalence, meaningful teacher gain, <=3–5% marginal recovery cost, no small-M regression, competitive large-M, no credible model-quality regression |

The current physical inventory provides one H200. H100 SXM/PCIe, H200 SXM
SKU comparison and multi-GPU TP performance cannot be claimed from it. That
limits those validation rows; it does not block the remaining implementation,
local tests, real-model single-GPU fitting, or H200 performance work.

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

The six existing Hopper geometry choices are now externally selectable,
and their H200 off/on sweep is recorded in `results/p32_window_h200_geometry.json`.
BM128/BN128 improves M2048 but loses at smaller M. The API retains every
supported geometry rather than pruning it based on another shape. Next:
connect the now-tested tuning runner to actual ZML lowering and model graph
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
