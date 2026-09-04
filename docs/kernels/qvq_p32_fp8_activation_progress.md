# QVQ P32 FP8 activation execution plan

Status date: 2026-09-04. Validation GPU: one exclusive NVIDIA H200 (SM90).

## Non-negotiable contracts

- Standard `qvq_v2b2_p32` checkpoint tensors and their planar serialization do
  not change. FP8 is a transient execution operand, not a new P32 weight format.
- `activation_quantization=None` is W2--W3.5A16 and preserves the historical
  path. `QVQActivationConfig` is opt-in A8.
- A8 always installs the fail-closed E4M3 KV cache. A dense Transformers cache
  is rejected instead of silently changing the requested mode.
- Unsupported devices, shapes, training, or adapters retain the established
  exact fallback. No FP8 label may be emitted unless telemetry proves an FP8
  payload and an FP8 hardware instruction path.

## Lessons adopted from the Together NVFP4 branch

The design follows the useful separation demonstrated by commits `96cc7062`
and `16422dd1` on `users/sgambhira/nvfp4`: calibrate against the values the
deployed kernel actually consumes, keep activation scale ownership explicit,
share scales at shared input sites, and fail loudly when serving metadata cannot
reproduce calibration. P32 does not copy NVFP4's weight format or two-level
weight scales.

## Phases and gates

### Phase 0 — freeze modes and observability (complete)

The comparison modes remain dense BF16, W3.5A16, and W3.5A8. Existing A8
calibration applies dynamic per-token E4M3 fake quantization before Hessian and
YAQA statistics. FP8 KV telemetry proves payload dtype, scale dtype, byte count,
sequence length, and absence of a full-precision residual.

### Phase 1 — remove avoidable A8 work (complete)

Inference now creates the E4M3 payload and row scale without eagerly creating a
dequantized tensor. Portable CPU/non-native paths dequantize only when needed;
the straight-through training path is unchanged.

Gate: activation unit tests and the H200 native fused-Hadamard path must remain
numerically identical to the previous A8 contract.

### Phase 2 — share activation preparation across P32 siblings (complete)

Grouped QKV and gate/up launches accept A8 only when every sibling has identical
activation-quantization state. One shared input is quantized once and reused by
the segmented P32 launch. H100-only specialized input paths are not selected for
A8 until independently validated. Telemetry adds:

- `grouped_a8_launches`
- `shared_fp8_quantizations`

Gate: on H200, M=1/2/4/8/16 grouped A8 is bit-exact with independent A8 child
execution and reports exactly one shared FP8 quantization.

### Phase 3 — prove the native FP8 WGMMA atom (complete)

The H200 compile-and-run smoke uses the SM90A register/shared
`m64n16k32` E4M3 x E4M3 WGMMA with FP32 accumulation. It verifies 16 FP8
register-A values per lane. This is the shape required to combine two adjacent
P32 K16 decoded tiles without changing the P32 stream.

### Phase 4 — integrate optional FP8 P32 operands (next)

1. Decode two adjacent P32 K16 tiles into one K32 E4M3 register-A fragment.
2. Write the transformed activation operand to the WGMMA shared-memory layout
   as E4M3, carrying one explicit scale per logical row.
3. Quantize the 256-entry PGC16 level table to E4M3 with an explicit transient
   kernel scale. Apply activation and level scales to the FP32 accumulator before
   the existing output recovery.
4. Add an opt-in dispatch policy. The default remains the current FP16 WGMMA
   until quality and speed gates pass; unsupported cases fall back to FP16.
5. Add counters for requested/eligible/executed FP8 WGMMA and every fallback
   reason. "A8" alone is not evidence that FP8 WGMMA executed.

Calibration gate: add the extra P32-operand quantization error to the calibration
objective. The Hessian and quality report must describe the exact scaled E4M3
operand grid used by the kernel, following the NVFP4 branch's deployed-operand
principle.

### Phase 5 — native FP8 KV attention (pending)

The current cache is true FP8 storage but SDPA dequantizes the retained prefix to
BF16 for attention. Replace that with an H200 FP8-cache-aware attention consumer
and a static/paged allocation strategy so decode no longer requantizes and
dequantizes the full growing prefix.

Gate: cache payload stays E4M3 end to end, no BF16 residual cache exists, and
Nsight/kernel telemetry proves the attention consumer reads FP8 payloads.

### Phase 6 — end-to-end acceptance (pending after Phases 4 and 5)

On the same exclusive H200 and matched prompts, publish one table for dense,
W3.5A16, and W3.5A8 containing PPL, KLD, top-1/5/10 agreement, prefill tokens/s,
decode tokens/s, latency percentiles, allocated/reserved/driver peak VRAM, model
bytes, KV payload/scale bytes, cache ratio, and all P32/FP8 dispatch counters.

Required workload points are batch 1 with logical M=1/2/4/8/16 for decode and
M=32/64/128/256/512/1024/2048/4096 for prefill. Logical M>16 continues to tile
over the native M16 P32 operator until dedicated larger-M kernels win their own
accuracy and performance gates.
