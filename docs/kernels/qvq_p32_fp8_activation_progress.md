# QVQ P32 FP8 activation execution plan

Status date: 2026-09-04. Validation GPU: one exclusive NVIDIA H200 (SM90).

## Non-negotiable contracts

- Standard `qvq_v2b2_p32` checkpoint tensors and their planar serialization do
  not change. FP8 is a transient execution operand, not a new P32 weight format.
- `activation=None` is W2--W3.5A16 and preserves the historical
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
  "activation": null
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
  "activation": {
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
quantization. For `target=p32_operand`, runtime shares SU/Hadamard preparation
and the final dynamic per-row E4M3 conversion, then invokes each child's proven
native FP8 P32 kernel with the same payload and row scales.
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
W2/W2.5/W3/W3.5 at M=1/16/17/32/64, and W3.5 at
M=1/2/4/8/16/17/32/64/128/256/512/1024/2048/4096 across three seeds. The new
grid is exact to the deployed E4M3 operand/weight reference at every point.

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
the per-KV-head streaming path and process at most 2,048 query tokens at once.
The score allocation is reused in-place for softmax and V-scale folding. This
bounds the FP32 workspace without materializing a multi-gigabyte all-head score
tensor or a dense K/V prefix.

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
| dense BF16 | 183,811 | 22.28 / 23.06 | 31.68 | 31.39 / 32.23 | 2.710 / 2.795 | 3,558 | 2.303 | 130.50 |
| W3.5A16 | 37,435 | 109.42 / 131.48 | 20.69 | 48.02 / 49.05 | 2.052 / 2.225 | 2,976 | 0.900 | 130.50 |
| W3.5A8 | 13,178 | 310.82 / 311.70 | 36.43 | 27.35 / 28.03 | 1.799 / 1.994 | 2,724 | 0.900 | 72.25 |

The static A8 cache reserved 4,352 token slots for the 4,176-token logical
sequence and still used 44.64% fewer retained bytes than the BF16 cache,
including FP32 per-token scales. At an exact-capacity boundary the reduction is
46.875%. It allocated once per layer, performed zero reallocations/copies, and
executed 9,856/9,856 requested P32 FP8 calls plus 10,496 QK and 10,496 PV FP8
matrices. Query chunking and grouped decode produced 1,536 QK and 1,536 PV
launches.
There were zero P32 fallback/rejections, zero KV dequantized elements, and zero
dense K/V prefix materializations.

Relative to the earlier correctness baseline, M-grid launch collapsing raised
A8 prefill from 803 to 3,993 tok/s (4.97x), decoded-weight row reuse raised it
to 6,667 tok/s, and the FP8-specific allocation/launch reductions raised it to
8,871 tok/s. Paired P32 state decode and simplified bank mixing raised it to
9,926 tok/s, and vectorized E4M3 activation staging raised the final result to
12,559 tok/s. Asynchronous activation staging raised it to 13,152 tok/s, and
CTA-local row-scale caching raised the final result to 13,178 tok/s. This is
3.301x over the 3,993 target baseline and 16.41x over the 803 tok/s correctness
baseline. A8 decode rose from 11.28 to 36.43 tok/s and is now 1.15x dense
decode. Matching dense prefill still requires another 13.95x; A16 requires
2.84x.

### Phase 7 — FP8 decoded-weight row reuse (complete)

The E4M3 P32 atom now decodes each weight fragment once and issues it against
up to eight independent M16 activation tiles before committing the WGMMA batch.
Each tile retains its own FP32 accumulator and dynamic per-row activation scale,
so output-row arithmetic and the serialized P32 format are unchanged. Host
dispatch keeps independent M16 CTAs for M16/M32, selects reuse-4 from M64, and
selects reuse-8 at M256 and above when divisible by M128. Larger shapes
divisible by M32 but not M64 retain reuse-2. M32 deliberately keeps two
independent CTAs: on H200 and the Llama-3.2-1B gate/up shape, reuse-2 was 7%
slower because the smaller grid lost occupancy.

For a direct W3.5 kernel grid at K=2,048 and N=8,192, reuse-4 gave the following
improvements over the prior single-row-tile grid:

| M | prior ms | reuse ms | speedup |
| ---: | ---: | ---: | ---: |
| 64 | 0.224 | 0.160 | 1.40x |
| 128 | 0.410 | 0.188 | 2.18x |
| 256 | 0.783 | 0.332 | 2.36x |
| 512 | 1.524 | 0.615 | 2.48x |
| 1,024 | 2.968 | 1.187 | 2.50x |
| 2,048 | 5.830 | 2.314 | 2.52x |
| 4,096 | 11.529 | 4.511 | 2.56x |

Reuse-8 crosses over at M256. At M4096 it reduces the reuse-4 result from
4.511 ms to 3.415 ms (1.32x), or 3.38x versus the single-row-tile grid.

The JIT-built cubin contains all W2/W2.5/W3/W3.5 reuse-1/2/4/8 variants and
native Hopper E4M3 WGMMA instructions. W3.5 reuse-8 uses 168 registers per
thread, 5 KiB shared memory, and no local-memory spill. Exact deployed-operand
tests pass all four bit widths at M32 and M64, plus the full W3.5 M grid through
4096 across three seeds.

### Phase 8 — FP8 allocation, launch, and SSA reduction (complete)

The final pass makes sibling groups share one transformed-operand E4M3
quantization, fuses dynamic row quantization into one CUDA kernel, removes the
redundant host-synchronizing BF16 finiteness probe from required native A8,
stores recovered BF16 directly from the Hadamard kernel, and bounds attention
scores with 2,048-query chunks and in-place softmax/V-scale operations. The
fused quantizer is bit-exact to the former PyTorch chain for FP16, BF16, and
FP32 inputs, including zero rows.

NCU and SASS were rerun after instruction-changing commit `a8120b94` on the
pinned H200:

| kernel | duration | registers/thread | achieved / theoretical occupancy | compute | DRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| W3.5 P32 E4M3 reuse-8, M4096 K2048 N2048 | 969.41 us | 168 | 16.76% / 18.75% | 65.68% | 0.41% |
| fused E4M3 row quantizer, M4096 K2048 FP16 | 23.71 us | 20 | 90.20% / 100% | 48.26% | 14.81% |

The P32 SASS contains eight E4M3 WGMMA instructions, 64 FP32 multiplies, and 64
FP32 fused multiply-adds per static kernel body. Its low DRAM use and register
occupancy confirm that P32 state decoding plus per-K32 FP32 rescaling, rather
than FP8 tensor-core throughput or memory bandwidth, is the remaining limit.
Reassociating each row scale with the level scale halved the static multiplies
from 64 to 32, but changed floating-point association and measured 969.54 us,
slightly slower than the accepted 969.41 us. It was rejected.

The quantizer SASS uses one saturating E4M3 conversion and no per-element clamp:
`F2FP.SATFINITE.E4M3` already provides the required finite saturation, so two
redundant `FMNMX` operations were removed. This reduced the kernel from 24.67
to 23.71 us. The remaining reciprocal/refinement sequence implements the
data-dependent `value / row_scale`; replacing the scale calculation with IEEE
division was also rejected because it changes some E4M3 bytes by one rounding
boundary.

### Phase 9 — paired FP8 P32 state and bank algebra (complete)

The E4M3 WGMMA A-fragment maps every adjacent even/odd output-column pair to
the high/low bytes of one PGC state. The former generic coordinate loop decoded
and mixed that state twice. Commit `fecd88f6` replaces it with the explicit
Hopper lane geometry: eight unique states populate all sixteen fragment bytes,
while tile and bank metadata are resolved once per canonical K16 half. Commit
`9f7e93c5` then shares the two bank masks for each lane-local K4 group and uses
the already-proven masked affine PGC algebra.

The apples-to-apples H200 NCU sequence at M4096, K2048, N2048 is:

| revision | duration | registers/thread | achieved / theoretical occupancy | DRAM | spills |
| --- | ---: | ---: | ---: | ---: | ---: |
| `a8120b94` | 969.41 us | 168 | 16.76% / 18.75% | 0.41% | 0 |
| paired state decode | 766.30 us | 168 | 16.54% / 18.75% | 0.51% | 0 |
| paired bank/PGC algebra | 757.54 us | 168 | 16.64% / 18.75% | 0.50% | 0 |

The two accepted passes provide a cumulative 1.280x microkernel speedup without
changing the eight E4M3 WGMMAs, 64 FP32 multiplies, or 64 FP32 fused
multiply-adds. Static hot-body instruction counts changed as follows:

| instruction | `a8120b94` | `9f7e93c5` | reduction |
| --- | ---: | ---: | ---: |
| LDG | 149 | 73 | 51.0% |
| IMAD | 556 | 219 | 60.6% |
| LOP3 | 323 | 92 | 71.5% |
| SHF | 256 | 93 | 63.7% |
| PRMT | 63 | 35 | 44.4% |
| SEL | 49 | 4 | 91.8% |

Post-commit NCU/SASS was collected after each GPU-instruction-changing commit.
Both reports retain 168 registers per thread and zero local/shared spills. A
subsequent base-bit-position hoist was rejected: it only traded eight IMAD and
three IADD3 operations for three additional SHF and one LEA, with no measured
duration improvement.

At the full Llama-3.2-1B W3.5A8 boundary, the same exclusive H200 run improved
prefill from 8,870.56 to 9,925.82 tok/s (1.119x) and decode from 34.44 to
35.98 tok/s (1.045x). Peak allocation/reservation stayed exactly 1.799/1.994
GiB and sampled NVML peak stayed 2,724 MiB. The 72.25 MiB cache remained E4M3,
all 9,856 requested P32 calls executed natively, all 1,536 QK and 1,536 PV
launches used the FP8 attention backend, and there were no P32 fallbacks, KV
dequantizations, or dense-prefix materializations. The machine-readable result
is `clean-9f7e93c5-w35-a8-4096.json`.

### Phase 10 — vectorized FP8 activation staging (complete)

NCU localized the dominant remaining Phase 9 stall to the scalar E4M3
global-to-shared activation copy: each thread repeatedly issued byte-sized
loads and stores before WGMMA. The CUTLASS SM90 B-operand layout keeps each
K16 half-row contiguous and 16-byte aligned, including halves exchanged by the
row swizzle. Commit `448dc2a1` therefore replaces the byte loop with aligned
`uint4` copies. It changes only transient activation staging; P32 decoding,
E4M3 bytes, row scales, WGMMAs, and output arithmetic are unchanged.

The apples-to-apples H200 NCU result at M4096, K2048, N2048 is:

| revision | duration | speedup | registers/thread | achieved / theoretical occupancy | DRAM | spills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 9 scalar staging | 757.54 us | 1.000x | 168 | 16.64% / 18.75% | 0.50% | 0 |
| Phase 10 16-byte staging | 346.85 us | 2.184x | 168 | 16.48% / 18.75% | 1.11% | 0 |

Long-scoreboard samples fell from 21,991 to 12,756 (42.0%). The static kernel
still contains eight E4M3 WGMMAs, 64 FP32 multiplies, 64 FP32 fused
multiply-adds, 73 global loads, and seven shared stores: the gain comes from
executing the vector-copy loop sixteen times less often, not deleting the
loop-body opcodes. The post-commit NCU/SASS report exactly matches the accepted
binary (eight WGMMAs, 168 registers/thread, zero local/shared spills).

Direct W3.5 shapes confirm that the result is not specific to the NCU point:
M4096/K2048/N8192 fell from 2.569 to 1.163 ms (2.21x). Exact deployed-operand
coverage passed all W2/W2.5/W3/W3.5 variants and the full M grid through 4096
across three seeds; the focused QVQ suite passed 184 tests with 70 skipped.

At the full Llama-3.2-1B W3.5A8 boundary, exclusive H200 prefill improved from
9,925.82 to 12,558.79 tok/s (1.265x), with median/p95 latency falling from
412.66/413.51 to 326.15/326.88 ms. Decode improved from 35.98 to 36.47 tok/s
(27.35/27.74 ms median/p95). Peak allocation/reservation remained
1.799/1.994 GiB and sampled NVML peak remained 2,724 MiB. All 9,856 requested
P32 calls executed with E4M3 operands and FP32 accumulation, with no fallback
or rejection. The 72.25 MiB KV cache remained entirely E4M3 at 55.36% of its
dense-equivalent storage; all 1,536 QK and 1,536 PV launches used native FP8
attention, with zero cache dequantization or dense-prefix materialization. The
machine-readable result is `clean-be218eaa-w35-a8-4096.json`.

### Phase 11 — asynchronous FP8 activation staging (complete)

After Phase 10, 10,777 of 12,756 long-scoreboard samples were attributed to
the two vector shared stores waiting for their global loads. Commit `d4aea064`
replaces each aligned 16-byte load/store pair with Hopper `cp.async.cg`, commits
the per-thread copy group, and waits before the existing block publication
barrier. The shared layout, E4M3 bytes, WGMMA operands, and numerical order do
not change.

The apples-to-apples H200 NCU result at M4096, K2048, N2048 is:

| revision | duration | speedup | registers/thread | achieved / theoretical occupancy | no eligible | compute | DRAM | spills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 10 synchronous vector copy | 346.85 us | 1.000x | 168 | 16.48% / 18.75% | 48.06% | 50.52% | 1.11% | 0 |
| Phase 11 `cp.async` copy | 267.01 us | 1.299x | 168 | 16.58% / 18.75% | 41.36% | 56.91% | 1.50% | 0 |

Long-scoreboard samples fell from 12,756 to 6,651 (47.9%). SASS converts seven
global loads and seven shared stores into seven `LDGSTS` operations, leaving
the eight E4M3 WGMMAs, 64 FP32 multiplies, and 64 FP32 fused multiply-adds
unchanged. Direct M4096 timing improved from 0.3344 to 0.2704 ms at N2048
(1.24x), and from 1.1633 to 0.9638 ms at N8192 (1.21x). The mandatory
post-commit NCU/SASS audit reproduced seven `LDGSTS`, 168 registers/thread, and
zero local/shared spills. All-rate exact deployed-operand coverage and the
three-seed M grid through 4096 passed; the focused QVQ suite again passed 184
tests with 70 skipped.

At the full Llama-3.2-1B W3.5A8 boundary, exclusive H200 prefill improved from
12,558.79 to 13,151.51 tok/s (1.047x), with median/p95 latency falling from
326.15/326.88 to 311.45/312.27 ms. Decode remained effectively flat at
36.39 tok/s (27.30/28.44 ms median/p95). Peak allocation/reservation remained
1.799/1.994 GiB and sampled NVML peak remained 2,724 MiB. All 9,856 requested
P32 calls executed with E4M3 operands and FP32 accumulation, with no fallback
or rejection. The 72.25 MiB KV cache remained entirely E4M3 at 55.36% of its
dense-equivalent storage; all 1,536 QK and 1,536 PV launches used native FP8
attention, with zero cache dequantization or dense-prefix materialization. The
machine-readable result is `clean-d4aea064-w35-a8-4096.json`.

### Phase 12 — invariant FP8 scale caching and pipeline audit (complete)

After asynchronous activation staging, 64 of the static global-load sites were
row-scale reads inside the K32 accumulation loop. Each CTA uses only 16--128
distinct scales, invariant across all K steps. Commit `96206dc9` cooperatively
loads those scales once into at most 512 bytes of shared memory and preserves
the original `accumulator * row_scale * level_scale` FP32 evaluation order.

The apples-to-apples H200 NCU result at M4096, K2048, N2048 is:

| revision | duration | speedup | registers/thread | static shared | long scoreboard | spills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 11 global row scales | 267.01 us | 1.000x | 168 | 4.10 KiB | 6,651 | 0 |
| Phase 12 shared row scales | 261.76 us | 1.020x | 164 | 4.61 KiB | 6,232 | 0 |

Static SASS changes from 66 global loads to 35 global plus 16 shared loads and
one cooperative shared store. It retains seven activation `LDGSTS`, eight E4M3
WGMMAs, 64 FP32 multiplies, and 64 FP32 fused multiply-adds. Direct M4096
timing improves from 0.2704 to 0.2648 ms at N2048 (1.02x), and from 0.9638 to
0.9416 ms at N8192 (1.02x). Post-commit NCU/SASS reproduces 164 registers per
thread and zero local/shared spills. All-rate exact deployed-operand coverage,
the three-seed M grid through 4096, and the 184-test focused QVQ suite pass.

Three profiler-led alternatives were rejected:

- double-buffering activation shared memory improved small M but regressed the
  widest M4096/N8192 shape and reduced full-model prefill from 13,152 to 13,122
  tok/s;
- removing the post-WGMMA ownership barrier was timing-neutral and discarded
  the conservative cross-warp overwrite boundary;
- caching the randomly indexed 256-byte E4M3 level table in shared memory
  introduced bank pressure and regressed M4096/N8192 from 0.9416 to 1.1940 ms.

At the full Llama-3.2-1B W3.5A8 boundary, exclusive H200 prefill is 13,178.23
tok/s with 310.82/311.70 ms median/p95 latency. Decode is 36.43 tok/s with
27.35/28.03 ms median/p95. Peak allocation/reservation remains 1.799/1.994 GiB
and sampled NVML peak remains 2,724 MiB. All 9,856 requested P32 calls execute
with E4M3 operands and FP32 accumulation, with no fallback or rejection. The
72.25 MiB KV cache remains entirely E4M3 at 55.36% of dense-equivalent storage;
all 1,536 QK and 1,536 PV launches use native FP8 attention, with zero cache
dequantization or dense-prefix materialization. The machine-readable result is
`clean-96206dc9-w35-a8-4096.json`.
