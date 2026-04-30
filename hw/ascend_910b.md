# Ascend 910B Hardware and Kernel Notes

Last scanned: 2026-04-30

Scope: public sources only. Do not treat this as a disclosure of private or non-public
Huawei data. Ascend 910B has several variants and public sources conflict, so CANN
kernels should query the target at runtime whenever possible.

## Executive Summary

- NVIDIA "SM" is not the right unit here. The practical Ascend equivalent is the
  AI Core, which is split into matrix/Cube work and vector work.
- Public sources put Ascend 910B active AI Cores in the 20-25 range. This host's
  `npu-smi` reports `910B1` with 25 AI Cores per NPU.
- The kernel-relevant per-core structure for Atlas A2/A3 `__NPU_ARCH__=220x`
  is: AIC/Cube and AIV/Vector are separated, each has Scalar control, Vector
  uses UB, Cube uses L0A/L0B/L0C, and AIC/AIV exchange via GM/L2 on 910B-class
  220x devices.
- High-confidence public SRAM facts for A2/A3 220x:
  - UB is 192 KB, split into 16 bank groups, 3 banks per group, 4 KB per bank,
    128 rows per bank, 32 B per row.
  - L0C is 128 KB.
  - BiasTable is 1 KB.
  - ICache and DCache hardware limits are 32 KB each.
- Do not hardcode L1/L0A/L0B/L2/HBM sizes in CANN tiling. Use
  `PlatformAscendC::GetCoreMemSize()` for `L0_A`, `L0_B`, `L0_C`, `L1`, `L2`,
  `UB`, and `HBM`.
- For matmul, start from TCubeTiling's recommended `(baseM, baseN, baseK) =
  (128, 256, 64)`, enable L0A/L0B double buffering, and then tune for the real
  shape and SRAM returned by CANN.

## Public Spec Table

| Item | Best working value | Confidence | Notes |
| --- | --- | --- | --- |
| Architecture family | Atlas A2/A3, `__NPU_ARCH__=220x` | High | Huawei CANN docs explicitly map 220x to Atlas A2/A3. |
| AI Core count | 20-25 active cores | Medium | CSET reports 20-25 across 910B variants. Local 910B1 reports 25. Some blogs say 24. |
| Matrix/tensor-like unit | Cube/AIC | High | Huawei docs describe Cube as the matrix unit accessing L0A/L0B/L0C. |
| Vector units | Usually 2 AIV per AI Core on 910B | Medium | CSET says second-gen 910B adds a second vector unit per core; OpenReview says one Cube and usually two Vector cores. |
| HBM capacity | 64 GB for 910B1/2/3, 32 GB for 910B4 | Medium | CSET appendix notes 4 x 16 GB for B1/B2/B3 and 2 x 16 GB for B4. Local 910B1 is 64 GB. |
| HBM bandwidth | 1600 GB/s for 910B1/2/3, 800 GB/s for 910B4 | Medium | CSET cites technical docs. WareDB lists 1200 GB/s for a generic 910B, lower confidence. |
| Clock | 1850 MHz nominal on local 910B1 | Local only | `npu-smi info -t common -i 0` reports 1850 MHz AICore frequency and 800 MHz current idle frequency. |
| Peak compute | Up to about 400 FP16 TFLOPS in CSET analysis | Medium | Useful for roofline estimates, not tiling. WareDB lists 256 FP16 TFLOPS, 512 INT8 TOPS, 1024 INT4 TOPS; lower confidence. |
| UB | 192 KB per 220x AIV | High | Huawei CANN bank-conflict docs. |
| L0C | 128 KB | High | Huawei CANN hardware-constraint docs. |
| BiasTable | 1 KB | High | Huawei CANN hardware-constraint docs. |
| ICache/DCache | 32 KB each | High | Huawei CANN hardware-constraint docs. |

## Local Host Snapshot

Command results from this machine, not public web data:

- `npu-smi 25.5.2`, eight NPUs, all named `910B1`.
- Each NPU reports 64 GB HBM. Device 0 reported `HBM Capacity(MB): 65536`.
- Device 0 reported `Aicore Count: 25`, `Aicore Freq(MHZ): 1850`, idle
  `Aicore curFreq(MHZ): 800`.
- Topology is full HCCS between all eight NPUs. CPU affinities:
  - NPU0/NPU2: CPUs 144-167
  - NPU1/NPU3: CPUs 0-23
  - NPU4/NPU6: CPUs 96-119
  - NPU5/NPU7: CPUs 48-71

This matters for benchmarking: bind data-loader and host-side CANN work to the
matching CPU affinity, and prefer HCCS-aware collectives for multi-NPU kernels.

## Memory Hierarchy and Data Movement

For 220x devices, Huawei's CANN architecture docs describe:

- Host CPU plus NPU Device execution.
- AI Core split into AIC/Cube and AIV/Vector components.
- Vector input/output data lives in Unified Buffer and must be 32 B aligned.
- Cube uses:
  - L0A for the left matrix.
  - L0B for the right matrix.
  - L0C for results and accumulators.
- L1 cache is recommended in `FRACTAL_NZ`; L0A/L0B/L0C use fractal layouts:
  - 220x L0A: `FRACTAL_ZZ`
  - 220x L0B: `FRACTAL_ZN`
  - 220x L0C: `FRACTAL_NZ`
- UB has no required data format, but it has bank conflicts.
- AIC and AIV communicate through Global Memory on 220x, so fusing AIC and AIV
  phases is not the same as sharing CUDA shared memory. Avoid ping-ponging
  between vector and cube phases unless the saved work exceeds the GM/L2 traffic.
- HCCS physical links are supported for cross-card `DataCopy` on Atlas A2.

Practical alignment rules from CANN docs:

- UB: 32 B alignment.
- L1: 32 B alignment.
- L0A/L0B: 512 B alignment.
- L0C: 64 B alignment.
- BiasTable: 64 B alignment.
- Fixpipe: 64 B in architecture docs, 128 B in the hardware-constraint page.
  Use the stricter 128 B unless a specific CANN API says otherwise.
- GM allocation is normally 512 B aligned. Make inner-axis movement 128/256/512 B
  aligned where possible because CANN says GM moves are split into those granularities.
- Single GM move lengths of at least 16 KB are recommended for bandwidth.
- `DataCopy` stride fields are bounded by 65535 units of 32 B in several paths;
  split very-strided movement into multiple instructions.

## UB Bank Conflicts

The UB layout for 220x is:

- Total: 192 KB.
- 48 banks.
- 16 bank groups.
- 3 banks per bank group.
- Bank size: 4 KB.
- Bank row: 32 B.

Conflict cases:

- Read/write conflict: source and destination hit the same bank.
- Write/write conflict: writes hit the same bank group.
- Read/read conflict: multiple reads hit the same bank group.

Tuning tricks from Huawei docs:

- Prefer continuous read plus strided write over strided read plus continuous write
  when a vector repeat would otherwise read many blocks from one bank group.
- Deliberately pad UB allocations. Huawei's example pads one tensor by 256 B and
  spaces the next tensor so x/y/z do not collide in the same repeat.
- Use msProf `ResourceConflictRatio` when a vector kernel looks compute-light but
  still stalls.

## Matmul Tiling Constraints

CANN's `TCubeTiling` is the most actionable public source for custom GEMM-style
kernels. A valid base tile must satisfy:

```text
usedCoreNum <= aiCoreCnt
baseM * baseK * sizeof(A_type) * dbL0A < l0a_size
baseN * baseK * sizeof(B_type) * dbL0B < l0b_size
baseM * baseN * sizeof(int32_t) * dbL0C < l0c_size
baseN * sizeof(Bias_type) < biasT_size
baseM * baseK * depthA1 * sizeof(A_type)
  + baseN * baseK * depthB1 * sizeof(B_type) <= L1_size
```

Alignment:

- `baseM` and `baseN` must be 16-element aligned.
- `baseK` must be aligned to C0:
  - FP16/BF16: 16
  - FP32: 8
  - INT8: 32
  - INT4: 64

Recommended starting point from CANN:

```text
baseM, baseN, baseK = 128, 256, 64
dbL0A = 2
dbL0B = 2
depthA1 / (stepM * stepKa) = 2
depthB1 / (stepN * stepKb) = 2
Choose stepKa/stepKb first to fully load K, then expand M/N reuse.
```

Tiling guidance:

- Fill L0C first. With 128 KB L0C and int32/float accumulators, the output tile
  ceiling is `128 KB / 4 B = 32768` elements before double buffering. Examples:
  `128 x 256`, `256 x 128`, `64 x 512`, `512 x 64`.
- If enabling L0C double buffering, halve the safe C tile footprint. A common
  full tile becomes about `128 x 128` for int32/float accumulators.
- L0A/L0B double buffering is usually worth it; L0C double buffering is only worth
  it when the output block remains large enough to keep Cube busy.
- If all M/N tiles for a K slice fit in L1, favor `stepKa=stepKb=1` and make
  `stepM`/`stepN` cover the resident M/N reuse.
- If L1 cannot hold that, tune a more cube-like L1 tile and keep K reuse high.
- Keep shape-dependent code small. CANN notes ICache is 32 KB and recommends
  splitting tiling keys or using templates to reduce code size.
- For small shapes, do not launch all cores blindly. CANN notes multi-core access
  to the same ICache address can serialize; small shape kernels may be faster
  with fewer active cores.
- MTE1 and MMAD instruction queues have depth 32. CANN recommends Load3D over
  Load2D where applicable because Load2D can emit 32 instructions and fill queues.
- For scalar writes to GM, use `DataCacheCleanAndInvalid` where consistency matters.

Runtime query example for tiling code:

```cpp
auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
uint64_t ub_size = 0, l1_size = 0, l0a_size = 0, l0b_size = 0, l0c_size = 0;
platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a_size);
platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b_size);
platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c_size);
```

## Quantization and Fixpipe Notes

Huawei's 220x docs state Fixpipe can harden typical Cube-side post-processing:

- Quantization/dequantization paths including S32 to FP16/S32/S4/S8/S16 and FP32
  to FP16/BF16/S8/S4/FP32.
- Relu-style activation.
- Format conversion, including channel merge/split and `NZ2ND`.

Implications for GPTQ/weight-only kernels:

- Use Fixpipe for output conversion/quantization when the exact conversion is
  supported, instead of writing extra AIV conversion passes.
- For INT4 weights, obey C0=64 and output-channel multiples that allow channel
  merge to pack cleanly.
- W4A16 is difficult on Ascend because mixed INT4-to-FP16 GEMM is not native in
  the obvious way. A 2026 W4A16 Ascend paper reports a practical design using
  vector cores for on-the-fly INT4-to-FP16 dequantization, Cube cores for GEMM,
  and Split-K parallelism. It also reports that extra global-memory movement of
  weights, not dequant compute alone, is the main bottleneck.
- For 910B 220x, AIC/AIV exchange through GM/L2. That makes "dequant on AIV,
  GEMM on AIC" expensive unless the dequantized stream is carefully staged and
  reused. Prefer formats that CANN/Torch-NPU kernels already support natively
  before writing a split AIV/AIC custom op.

## Framework Kernel Scan

Local repositories scanned under `/tmp/ascend_repo_scan`:

- `vllm-project/vllm-ascend` at `d1f6684`.
- `vllm-project/vllm` at `10558f5`.
- `sgl-project/sglang` at `aa74911`.
- `sgl-project/sgl-kernel-npu` at `69ab269`.
- `deepseek-ai/DeepSeek-V3` at `9b4e978`.
- `deepseek-ai/DeepEP` at `b306af0`.
- `deepseek-ai/DeepGEMM` at `891d57b`.
- `deepseek-ai/FlashMLA` at `9241ae3`.

### vLLM

Mainline vLLM does not carry the real Ascend kernel implementation. It documents
the hardware-plugin mechanism and points users to `vllm-ascend` as the official
Ascend device plugin.

### vLLM-Ascend

`vllm-ascend` has actual Ascend C and Torch-NPU kernel paths.

Kernel groups found in `csrc`:

- Attention and KV cache:
  - `sparse_flash_attention`
  - `mla_preprocess`
  - `transpose_kv_cache_by_block`
  - `reshape_and_cache_bnsd`
  - `hamming_dist_top_k`
  - `lightning_indexer_quant`
  - `lightning_indexer_vllm`
- MoE:
  - `moe_init_routing_custom`
  - `moe_gating_top_k`
  - `moe_grouped_matmul`
  - `moe_dispatch_normal`
  - `moe_combine_normal`
  - `dispatch_layout`
  - `dispatch_ffn_combine`
  - `dispatch_ffn_combine_bf16`
  - `dispatch_ffn_combine_w4_a8`
  - `dispatch_gmm_combine_decode`
  - `grouped_matmul_swiglu_quant_weight_nz_tensor_list`
- Fusions:
  - `add_rms_norm_bias`
  - `matmul_allreduce_add_rmsnorm`
  - `apply_top_k_top_p_custom`
  - `batch_matmul_transpose`
  - `causal_conv1d`
  - `recurrent_gated_delta_rule`
- LoRA-like kernels:
  - `bgmv_expand`, `bgmv_shrink`, `sgmv_expand`, `sgmv_shrink` under `csrc/kernels`.

How they optimize Ascend usage:

- Attention uses Torch-NPU/ATB primitives such as `npu_fused_infer_attention_score`,
  `_npu_flash_attention_unpad`, `npu_scatter_pa_kv_cache`, ATB paged cache load,
  and custom sparse/MLA preprocessing kernels. The goal is to keep paged KV in
  NPU-friendly layout, reduce host-side reshape/cache overhead, and avoid generic
  PyTorch attention paths.
- vLLM-Ascend FAQ notes small-batch performance issues in the default fused infer
  attention path when flash decoding is absent. It provides scripts to install
  `fused_infer_attention_score` variants for A2/A3.
- MLA preprocess fuses projection/dequant, RMSNorm, RoPE, and KV cache writes into
  one operator path. That reduces GM round trips and Python launches.
- `transpose_kv_cache_by_block` is an Ascend C fused op for GQA KV-cache transfer.
- MoE paths use custom init routing/top-k, grouped matmul, SwigLU quant, and
  dispatch/combine fusion. These reduce per-expert launch overhead, keep expert
  tokens in contiguous/grouped layouts, and use NZ tensor-list weights for Cube.
- Norm/quant fusion replaces `add + rmsnorm + quantize` with
  `npu_add_rms_norm_quant` or related custom ops to save one or more HBM trips.
- Weight prefetch uses `torch_npu.npu_prefetch` and vLLM prefetch hooks on separate
  streams. It caps prefetch size and uses token thresholds to avoid prefetch cost
  dominating tiny workloads.
- ACL/NPU graph paths require care: vLLM-Ascend docs mention query-per-KV-head
  restrictions for some DeepSeek MLA graph cases and stream/workspace limits.

### SGLang

SGLang's NPU backend forces the attention backend to `ascend`, sets a default
page size of 128, enables Torch-NPU internal format, and disables JIT compile
mode. Its default memory policy separates 32 GB devices (`910B4`) from 64 GB
devices (`910B1/2/2C/3` and related `910_93xx` models).

NPU-specific optimization paths found:

- `npu_format_cast` casts aligned weights to `FRACTAL_NZ`.
  - BF16/FP16: K and N divisible by 16.
  - INT8: K divisible by 16 and N by 32.
  - INT4 packed as uint8/int32: K divisible by 16 and N by 64.
  - FP4: both dims divisible by 64.
- CMO weight prefetch launches `torch_npu.npu_prefetch` on a separate NPU stream
  to overlap matmul weight movement with AIV or communication kernels.
- Fused MoE top-k uses `npu_moe_gating_top_k_softmax` and
  `npu_moe_gating_top_k`, with fallback to torch native only for unsupported
  routing cases.
- MLA preprocess is gated by `SGLANG_NPU_USE_MLAPO` and requires the NZ FIA path.
  It uses `npu_interleave_rope`, `npu_kv_rmsnorm_rope_cache`, and a custom
  `torch.ops.npu.mla_preprocess` path.
- Attention backend uses `npu_sparse_flash_attention`,
  `npu_fused_infer_attention_score`, `_npu_flash_attention_qlens`,
  `_npu_paged_attention`, `_npu_paged_attention_mla`, and ATB `npu_ring_mla`.
- NPUGraph capture/replay is implemented for static-ish decode paths. This reduces
  launch overhead but inherits CANN graph shape and stream restrictions.

### SGLang-Kernel-NPU

This is the most direct open-source Ascend kernel library found in the scan.

Major components:

- DeepEP-Ascend:
  - Ascend implementation of DeepSeek's DeepEP concept.
  - Supports Atlas A2 and A3.
  - Normal mode: high-throughput dispatch/combine for training and prefill,
    up to 8192 tokens/batch on A2 and 65536 tokens/batch on A3.
  - Low-latency mode: small-batch production inference around 128 tokens/batch,
    with README tables reporting sub-150 us latencies on A3 HCCS setups.
  - Uses A2 intranode HCCS plus internode RDMA in hierarchical mode; A3 uses
    HCCS for intra/inter-node paths.
  - Supports INT8/FP8/BF16 dispatch/combine modes to reduce memory bandwidth.
- Other Ascend C kernels:
  - `mla_preprocess`
  - `lightning_indexer`
  - `batch_matmul_transpose`
  - `causal_conv1d` and `causal_conv1d_update`
  - `recurrent_gated_delta_rule`
  - `lora`
  - `tri_inv`
  - `assign_cache_op`, `cache_location_assign`, `transfer_kv_dim_exchange`
  - `build_tree` and `apply_token_bitmask` for speculative decoding
  - `catlass` Ascend-side matmul-related kernels

How it optimizes Ascend usage:

- Provides kernels already written in Ascend C, with explicit `DataCopy`,
  `LocalTensor`, `InitBuffer`, event/flag sync, UB staging, and per-core tiling.
- DeepEP-Ascend moves MoE dispatch/combine off generic collectives and into
  HCCS/RDMA-aware kernels, then fuses routing with MoE compute where possible.
- MLA preprocessing explicitly fuses `RMSNorm -> Dequant -> MatMul -> RoPE ->
  ReshapeAndCache`, which is exactly the kind of fusion needed on 910B because
  AIV/AIC intermediate exchange through GM is costly.

### DeepSeek Repos

DeepSeek-V3:

- README says Huawei Ascend NPU can run DeepSeek-V3 in INT8 and BF16.
- The recommended Ascend path is MindIE for BF16.
- The repo itself does not contain Ascend C kernels. Its demo inference code is
  CUDA/Triton oriented and uses `device="cuda"`.

DeepEP:

- CUDA/NVIDIA-only in the scanned repo. It depends on CUDA/NCCL/NVSHMEM-style
  infrastructure and describes SM90/SM100 performance.
- Conceptual ideas still apply:
  - Dispatch/combine should overlap with compute.
  - Communication kernels should reserve minimal compute cores.
  - Use low-precision dispatch payloads and BF16 combine when accuracy permits.
  - Maintain per-expert contiguous or masked grouped layouts for the following GEMM.
- For Ascend, use SGLang-Kernel-NPU's DeepEP-Ascend implementation rather than
  trying to port DeepEP CUDA directly.

DeepGEMM:

- CUDA/NVIDIA-only in the scanned repo. It uses `CUDAExtension`, CUDA Toolkit,
  SM90/SM100, TMA-aligned layouts, JIT, and CUDA tensor cores.
- Useful ideas for Ascend:
  - Separate transpose/cast overhead from the GEMM kernel, or fuse it into the
    preceding kernel.
  - Keep grouped MoE GEMMs aligned at group boundaries.
  - Use masked grouped GEMM for decode when expert token counts are graph-static
    but logically variable.
  - Cache JIT/tiling choices by shape.

FlashMLA:

- CUDA/NVIDIA-only in the scanned repo.
- It provides useful attention-design ideas:
  - FP8 KV cache format uses 656 B per token: 512 FP8 NoPE bytes, 16 B of four
    FP32 scales, and 128 B of BF16 RoPE.
  - It dequantizes FP8 KV to BF16 inside the kernel and computes BF16 attention.
  - The deep dive identifies dequantization as the bottleneck in sparse FP8 decode
    and uses Hopper CTA-cluster distributed shared memory to share dequantized KV.
- Ascend translation: there is no direct Hopper DSM equivalent on 910B 220x.
  Instead, minimize AIV/AIC GM handoffs, fuse KV dequant with attention where
  possible, and choose NZ/paged cache layouts that let Cube reuse K/V without
  repeated vector-side conversion.

## Practical CANN Kernel Checklist for 910B

1. Query hardware:
   - `GetCoreMemSize()` for all SRAM/HBM sizes.
   - CANN platform core count, and cross-check with `npu-smi` during bringup.
2. Pick data format:
   - BF16/FP16 Cube: make M/N multiples of 16 and K multiple of 16.
   - INT8 Cube: K multiple of 32 for C0.
   - INT4 Cube/packed weights: K multiple of 64 and prefer N multiples of 64.
   - Use `FRACTAL_NZ` for L1/cacheable weights where possible.
3. Build the pipeline:
   - `GM -> L1/UB -> L0A/L0B -> Cube -> L0C -> GM`.
   - Use `TPipe` queues with buffer count 2 for CopyIn/Compute/CopyOut overlap.
   - Double-buffer L0A/L0B first.
4. Avoid bank conflicts:
   - Keep UB tensors 32 B aligned.
   - Pad UB allocations when vector repeats collide on bank groups.
   - Measure with msProf.
5. Tune core usage:
   - Large batches: use all reported AI Cores.
   - Small decode or small M/N: test fewer cores to avoid ICache and GM same-address
     serialization.
6. Prefer existing kernels first:
   - `torch_npu.npu_grouped_matmul`
   - `torch_npu.npu_grouped_matmul_swiglu_quant`
   - `torch_npu.npu_dynamic_quant`
   - `torch_npu.npu_moe_gating_top_k`
   - `torch_npu.npu_fused_infer_attention_score`
   - `torch_npu.npu_prefetch`
   - `sgl-kernel-npu` DeepEP/MLA kernels when using SGLang.
7. For GPTQ/Komodo-like weight-only matmul:
   - Keep packed weight layout aligned to Cube C0 and channel-merge constraints.
   - Avoid vector dequant writing a full FP16 matrix to GM unless reused many times.
   - For tiny decode batches, Split-K can help latency hiding, but it increases
     accumulator/reduction traffic; benchmark per shape.
   - If dequant must be vector-side, dequant a K tile, immediately feed Cube, and
     reuse that tile across as many M/N outputs as L1/L0 allow.

## Sources

Hardware and CANN:

- CSET, "Pushing the Limits: Huawei's AI Chip Tests U.S. Export Controls":
  https://cset.georgetown.edu/publication/pushing-the-limits-huaweis-ai-chip-tests-u-s-export-controls/
- Huawei CANN 220x architecture docs:
  https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html
- Huawei CANN hardware constraints:
  https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00048.html
- Huawei CANN UB bank-conflict best practices:
  https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_00019.html
- Huawei CANN `TCubeTiling` docs:
  https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0673.html
- Huawei CANN `GetCoreMemSize` docs:
  https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha001/API/ascendcopapi/atlasascendc_api_07_1034.html
- Arthur Chiao, Ascend 910B notes:
  https://arthurchiao.art/blog/gpu-advanced-notes-2-zh/
- WareDB Ascend 910B page, lower confidence because values conflict with CSET
  and local `npu-smi`:
  https://www.waredb.com/processor/ascend-910b
- OpenReview, "Parallel Scan on Ascend AI Accelerators":
  https://openreview.net/pdf?id=wPepcNWMhs
- arXiv, "W4A16 Mixed-Precision Matrix Multiplication on Decoupled Architecture":
  https://arxiv.org/abs/2601.16536
- arXiv, "AscendCraft: Automatic Ascend NPU Kernel Generation via DSL-Guided
  Transcompilation":
  https://arxiv.org/abs/2601.22760

Frameworks and kernels:

- vLLM main:
  https://github.com/vllm-project/vllm
- vLLM-Ascend:
  https://github.com/vllm-project/vllm-ascend
- vLLM-Ascend FAQ:
  https://docs.vllm.ai/projects/ascend/en/v0.18.0/faqs.html
- SGLang Ascend feature docs:
  https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu_support_features
- SGLang:
  https://github.com/sgl-project/sglang
- SGLang-Kernel-NPU:
  https://github.com/sgl-project/sgl-kernel-npu
- DeepSeek-V3:
  https://github.com/deepseek-ai/DeepSeek-V3
- DeepEP:
  https://github.com/deepseek-ai/DeepEP
- DeepGEMM:
  https://github.com/deepseek-ai/DeepGEMM
- FlashMLA:
  https://github.com/deepseek-ai/FlashMLA
- FlashMLA FP8 sparse decode deep dive:
  https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md
