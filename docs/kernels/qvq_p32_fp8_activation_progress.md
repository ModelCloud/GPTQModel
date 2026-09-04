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

## Comparison quantization configurations

Dense BF16 is the unmodified source checkpoint:

```json
{
  "quantization_config": null,
  "weights": "bfloat16",
  "activations": "bfloat16",
  "kv_cache": "bfloat16"
}
```

W3.5A16 uses P32 weights with native BF16 model activations and KV cache:

```json
{
  "method": "qvq",
  "bits": 3.5,
  "format": "qvq_v2b2_p32",
  "vector_size": 2,
  "trellis_window": 16,
  "bank_count": 2,
  "rounding": "block_ldlq",
  "activation_quantization": null
}
```

W3.5A8 uses the same P32 weight format and opts into the exact deployed FP8
operand, replay, and KV-cache contracts:

```json
{
  "method": "qvq",
  "bits": 3.5,
  "format": "qvq_v2b2_p32",
  "vector_size": 2,
  "trellis_window": 16,
  "bank_count": 2,
  "rounding": "block_ldlq",
  "activation_quantization": {
    "bits": 8,
    "format": "float8_e4m3fn",
    "scale_method": "dynamic_per_token",
    "target": "p32_operand",
    "kernel_mode": "require",
    "replay_passes": 1,
    "replay_max_rows": 2048,
    "replay_validation_fraction": 0.125
  }
}
```

`kernel_mode=require` is deliberate for acceptance checkpoints: a run cannot be
labeled W3.5A8 if native FP8 P32 execution is unavailable. Developers may use
`auto` for an observable fallback or `disable` for the exact FP8-emulation
control. `replay_passes=0` disables the FP8-targeted second encode.

## Phases and gates

### Phase 0 — freeze modes and observability (complete)

The comparison modes remain dense BF16, W3.5A16, and W3.5A8. The recommended
A8 target is now `p32_operand`: model-visible BF16/FP16 input remains native,
and dynamic per-token E4M3 quantization occurs only after SU/Hadamard at the
actual WGMMA operand boundary. `target=linear_input` retains the old pre-linear
fake-quantization experiment. FP8 KV telemetry proves payload dtype, scale
dtype, byte count, sequence length, and absence of a full-precision residual.

### Phase 1 — remove avoidable A8 work (complete)

Inference now creates the E4M3 payload and row scale without eagerly creating a
dequantized tensor. Portable CPU/non-native paths dequantize only when needed;
the straight-through training path is unchanged.

Gate: activation unit tests and the H200 native fused-Hadamard path must remain
numerically identical to the previous A8 contract.

### Phase 2 — share activation preparation across P32 siblings (complete)

Grouped QKV and gate/up launches accept A8 only when every sibling has identical
activation-quantization state. `target=linear_input` retains one shared input
quantization. For `target=p32_operand`, the current correctness implementation
shares SU/Hadamard preparation and invokes each child's proven native FP8 P32
kernel; a future grouped K32 kernel will share the final E4M3 conversion.
Telemetry adds:

- `grouped_a8_launches`
- `fp8_independent_child_launches`
- `shared_fp8_quantizations`

Gate: on H200, FP16 and BF16 M=1/2/4/8/16 grouped A8 is bit-exact with
independent A8 child execution and reports three native child launches with no
fallback.

The quantization lifecycle now derives ordinary QKV and gate/up topology from
the same `module_tree` roles used by runtime fusion, then assigns one layer-local
input sign seed to each sibling group. Previously only architecture-specific
`qvq_grouped_p32_candidates` received shared SU signs; ordinary Llama P32
checkpoints therefore could not fuse even though inference discovered their
roles. Existing checkpoints with different sibling SU tensors remain valid but
cannot be retrofitted without requantization.

### Phase 3 — prove the native FP8 WGMMA atom (complete)

The H200 compile-and-run smoke uses the SM90A register/shared
`m64n16k32` E4M3 x E4M3 WGMMA with FP32 accumulation. It verifies 16 FP8
register-A values per lane. This is the shape required to combine two adjacent
P32 K16 decoded tiles without changing the P32 stream.

### Phase 4 — integrate optional FP8 P32 operands (complete)

1. Decode two adjacent P32 K16 tiles into one K32 E4M3 register-A fragment.
2. Write the transformed activation operand to the WGMMA shared-memory layout
   as E4M3, carrying one explicit scale per logical row.
3. Quantize the 256-entry PGC16 level table to E4M3 with an explicit transient
   kernel scale. Apply activation and level scales to the FP32 accumulator before
   the existing output recovery.
4. Add `kernel_mode=auto|require|disable`. `require` rejects any ineligible or
   failed launch, `auto` records and falls back, and `disable` is the explicit
   emulation control.
5. Add counters for requested/eligible/executed FP8 WGMMA and every fallback
   reason. "A8" alone is not evidence that FP8 WGMMA executed.

The P32 checkpoint format is unchanged. The new operator decodes the standard
continuous P32 window directly into an E4M3 register-A fragment and uses the
SM90 `m64n16k32` E4M3 x E4M3 WGMMA with FP32 accumulation. It adds four device
specializations (transition widths 4/5/6/7) behind one host launch site. Each
K32 partial is scaled in FP32 before summation to prevent raw FP8 accumulator
growth from weakening the strict 2e-3 exact-deployed-operand gate.

The tensor-core atom remains M16, but one native two-dimensional CUDA grid now
covers every logical M through 4096. The wrapper tail-pads once and launches
once; it no longer invokes one CUDA operator per M16 tile. H200 coverage passed
W2/W2.5/W3/W3.5 at M=1/16/17, and W3.5 at
M=1/2/4/8/16/17/32/64/128/256/512/1024/2048/4096 across three seeds. The new
grid is exact to the deployed E4M3 operand/weight reference at every point.
Porting the FP16 M32/M64 decoded-weight reuse optimization to the E4M3 atom is
an optional future throughput optimization, not a format or correctness gap.

### Phase 4.5 — recommended FP8-targeted quantization replay (complete)

The calibration objective now describes the exact scaled E4M3 operand grid
used by the deployed kernel, following the NVFP4 branch's deployed-operand
principle.

Use a teacher-targeted two-pass solve rather than recursively quantizing P32
weights:

1. Preserve the original BF16/FP16 teacher weights and capture dense teacher
   outputs `Y`.
2. Produce the first P32 encoding from that immutable original dense weight
   using the pristine native-input Hessian.
3. Instantiate the exact first-pass artifact, require the real H200 FP8 kernel,
   and form the deployed post-SU/Hadamard operand `Z` using the same row scales
   and E4M3 rounding as inference.
4. Invert bias/SV/output-Hadamard on the native teacher output to obtain the
   matching inner target `T`, then accumulate normalized `G = Z^T Z` and
   `C = Z^T T` in FP32. A `G`-only objective cannot compensate the activation
   error between the native teacher input and deployed `Z`.
5. Solve `(G + lambda I) W* = C + lambda W0`, where `W0` is the immutable
   original dense weight mapped into the frozen first-pass SU/SV inner basis.
   This prior-centered ridge preserves null-space directions when replay rows
   are fewer than K. Perform exactly one new P32 encode of `W*`; reconstructed
   P32 weights are never recursively quantized.
6. Run both serialized candidates through `kernel_mode=require` on disjoint
   held-out native teacher rows. Select the re-encode only when its full module
   output MSE is no worse. Persist row counts, G/C shapes, damping, both losses,
   selection, source provenance, and native-kernel execution counters.

The flow is optional: `replay_passes=0` disables it, while
`target=linear_input` preserves the legacy calibration contract. A focused H200
test passes the entire native-teacher/first-encode/FP8-replay/re-encode/held-out
sequence and proves both candidates execute the native kernel.

The fresh Llama-3.2-1B W3.5A8 checkpoint recorded 112/112 first-candidate and
112/112 second-candidate native H200 executions. The held-out safety gate kept
the first encoding for all 112 modules: mean validation MSE was 0.000700 for
the initial encode and 0.001931 for the correction. This is the intended safe
outcome when replay cannot improve the serialized P32 candidate.

The comparison boundary is the real FP32 WGMMA accumulator followed by the
existing output recovery and BF16 model cast. The accumulator itself is not
stored as FP8; each following linear quantizes its own actual input operand.

### Phase 5 — native FP8 KV attention (complete)

The H200 attention interface now receives opaque E4M3 cache views. It
row-quantizes Q and runs native E4M3 QK-transpose through cuBLASLt with FP32
accumulation. After softmax, it folds each per-token V scale into the
probability column, row-quantizes that operand to E4M3, and runs the second
native E4M3-by-E4M3 GEMM directly against the cached V payload. No BF16/FP16 K
or V prefix is constructed. Telemetry counts both FP8 GEMM sites,
dequantized elements, and dense-prefix materializations; acceptance requires
the latter two to remain zero.

The cache uses aligned initial allocation, page-based dynamic growth, or a
one-shot static allocation when generation provides the maximum length. K is
stored row-major and V column-major, so decode neither concatenates the prefix
nor transpose-copies V. For decode-sized query M<=16, one grouped cuBLASLt call
executes every batch/KV-head matrix at each QK and PV site. Larger prefills keep
the per-KV-head streaming path to avoid materializing a multi-gigabyte
all-head score tensor.

Gate: cache payload stays E4M3 end to end, no BF16 residual cache exists, and
Nsight/kernel telemetry proves the attention consumer reads FP8 payloads.

### Phase 6 — end-to-end acceptance (complete)

On the same exclusive H200 and matched prompts, publish one table for dense,
W3.5A16, and W3.5A8 containing PPL, KLD, top-1/5/10 agreement, prefill tokens/s,
decode tokens/s, latency percentiles, allocated/reserved/driver peak VRAM, model
bytes, KV payload/scale bytes, cache ratio, and all P32/FP8 dispatch counters.

Required workload points are batch 1 with logical M=1/2/4/8/16 for decode and
M=32/64/128/256/512/1024/2048/4096 for prefill. The FP16 P32 path uses accepted
M32/M64 row reuse through M4096. The FP8 P32 path uses a single native M-tiled
grid through M4096 while preserving its E4M3 x E4M3, FP32-accumulation atom.

The held-out quality slice used parquet rows 256--319 (18,692 shifted tokens,
maximum length 512), disjoint from calibration rows 0--127. All arms ran BF16
model compute on the same H200; A8 additionally required native E4M3 P32 and KV
attention execution.

| arm | PPL | mean KL(dense || arm) | dense top-1 | dense top-5 overlap | dense top-10 overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense BF16 | 3.9234 | 0 | 100.00% | 100.00% | 100.00% |
| W3.5A16 | 4.0151 | 0.04311 | 91.87% | 88.02% | 87.93% |
| W3.5A8 | 4.0731 | 0.06366 | 90.36% | 85.19% | 85.11% |

Ground-truth next-token accuracy on the same positions:

| arm | top-1 | top-5 | top-10 |
| --- | ---: | ---: | ---: |
| dense BF16 | 68.89% | 87.60% | 91.17% |
| W3.5A16 | 68.47% | 87.06% | 90.96% |
| W3.5A8 | 68.15% | 87.01% | 90.75% |

The matched resource run used batch 1, 4,096-token prefill, 16 decode warmup
tokens, and 64 measured decode tokens. Peak allocated/reserved are PyTorch
whole-workload peaks; driver peak is sampled per-process NVML usage.

| arm | prefill tok/s | prefill p50 / p95 ms | decode tok/s | decode p50 / p95 ms | peak alloc / reserved GiB | NVML peak MiB | model GiB | KV MiB at 4,176 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense BF16 | 183,755 | 22.29 / 23.03 | 31.84 | 31.29 / 31.82 | 2.710 / 2.795 | 3,558 | 2.303 | 130.50 |
| W3.5A16 | 32,274 | 126.92 / 127.52 | 18.66 | 53.36 / 55.41 | 1.942 / 2.115 | 2,864 | 0.900 | 130.50 |
| W3.5A8 | 3,993 | 1,025.76 / 1,028.12 | 16.59 | 60.20 / 60.78 | 3.083 / 3.416 | 4,180 | 0.900 | 72.25 |

The static A8 cache reserved 4,352 token slots for the 4,176-token logical
sequence and still used 44.64% fewer retained bytes than the BF16 cache,
including FP32 per-token scales. At an exact-capacity boundary the reduction is
46.875%. It allocated once per layer, performed zero reallocations/copies, and
executed 9,856/9,856 requested P32 FP8 calls plus 10,368 QK and 10,368 PV FP8
matrices. Grouping reduced the attention work to 1,408 QK and 1,408 PV launches.
There were zero P32 fallback/rejections, zero KV dequantized elements, and zero
dense K/V prefix materializations.

Relative to the earlier correctness baseline, M-grid launch collapsing raised
A8 prefill from 803 to 3,993 tok/s (4.97x), grouped attention raised A8 decode
from 11.28 to 16.59 tok/s (1.47x), and FP16 large-M row reuse raised A16 prefill
from 5,225 to 32,274 tok/s (6.18x). Matching dense now requires another 46.02x
for A8 prefill or 5.69x for A16 prefill. A8 decode is 1.12x short of A16 and
1.92x short of dense.
