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
| 4 | Existing Hopper WGMMA + optional rank8; H100/H200 BM/BN/warp-group/stage candidates | Existing WGMMA retained; explicit consumer/split selection | Expose supported geometry, add missing justified candidates, integrate rank8 into consumer pipeline, both-device sweep |
| 5 | Shared input producer, decode producer and consumer; project rank8 alongside WGMMA | Reference shares X-prime; grouped paths share the padded operand | SU/H/input-staging + rank8 projection producer; SIMT/half2 vs padded-TC and FP8-factor sweep; concurrent scheduling; B staging/cache sweep |
| 6 | Expansion + FP32 add + output H/SV + final store; defined rounding order | `00a71f4b`: fused Triton expansion/add/H/SV/bias, independent graph-safe reference | Integrate input projection, direct final FP16 store, composite output widths, full native pipeline fusion |
| 7 | Grouped gate/up and QKV, independent flags, optional SiLU*up, preserve Q/K norm/RoPE/TP | Grouped separate and fused output epilogues tested on H200; child-local flags | Shared producer projection for multiple children, fused SiLU/down composition, real-model grouped/TP validation |
| 8 | ZML tunes eligible whole operators over geometry, transforms, quality and grouping; product-specific caches | Python static policy and device/product/UUID/shape/TP/build key builder | StableHLO/native ABI adapter, correctness-filtered eligible candidate runner, persistent measured caches; do not key only on SM90 |
| 9 | fast/balanced/quality graphs, mode changes only at request boundaries | Prepared module policies; separate captured on/off graphs tested | Model-wide graph manager and request-boundary ownership, atomic policy changes and cache invalidation |
| 10 | Combined scorecard, teacher/tail/logit/PPL/ARC/GSM8K, prefill/decode, TP, VRAM/BPW/overhead | Local tests and synthetic H200 full-operator benchmark plus Nsight artifacts | Real disjoint model scorecard; H100/H200, M128–8192, batches1–64, TP1/2/4/8; statistical quality gates |
| 11 | Promote and ship only verified complete operator/model results | No default promotion; explicit fused mode only | Off equivalence, meaningful teacher gain, <=3–5% marginal recovery cost, no small-M regression, competitive large-M, no credible model-quality regression |

The current physical inventory provides one H200. H100 SXM/PCIe, H200 SXM
SKU comparison and multi-GPU TP performance cannot be claimed from it. That
limits those validation rows; it does not block the remaining implementation,
local tests, real-model single-GPU fitting, or H200 performance work.

Next kernel step: move rank8 projection into the shared input-transform
producer. The M16 baseline profile shows FP32 activation/factor conversions,
a narrow SIMT GEMM and split reduction still consume launches after output
fusion. Then expose existing Hopper geometry in the unified policy and
measure each candidate with correction both off and on.
