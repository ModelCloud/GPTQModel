# EXL3 activation dispatch

`ExllamaV3Linear.forward` sends FP16 and BF16 activations to the native EXL3
kernels. FP32 keeps `ExllamaV3TorchLinear`, which computes in FP32, decodes
packed trellis weights in bounded output tiles, and rounds the result once to
the caller's dtype.

BF16 is the default dtype of most EXL3 checkpoints (Llama-3.x, Qwen2.5/3, ...).
Before #4767 every non-FP16 activation took the Torch reconstruction path, so
the default way to run those checkpoints was 20-200x slower than the native
kernel. Native EXL3 still requires FP16 operands; the bridge now supplies them.

```mermaid
flowchart TD
    A["ExllamaV3Linear.forward(x)"] --> B{"x.dtype"}
    B -->|"float32 or bfloat16 off CUDA"| H["_ensure_torch_inner"]
    B -->|"float16 or bfloat16 on CUDA"| P{"_native_input_is_finite<br/>once per module"}
    P -->|"finite and the guard owns the shape"| G["_guarded_native_forward<br/>bound in the input Hadamard,<br/>scale + bias in the epilogue"]
    G -->|accepted| YF["final-dtype result"]
    P -->|"non-finite first probe"| H
    P -->|"guard declined"| D["_native_bf16_input"]
    D --> E{"module warm and<br/>fused op present?"}
    E -->|yes| S["bf16_prescale: one launch<br/>row amax, power-of-two scale, FP16 cast"]
    E -->|no| N["first forward: isfinite probe"]
    N -->|finite| R["eager reference: vector_norm, div,<br/>frexp, ldexp, where, divide, cast"]
    N -->|non-finite| H
    S --> G2["FP16 activations plus per-row FP32 scale"]
    R --> G2
    G2 --> C["_ensure_inner"]
    C --> I["LinearEXL3.forward"]
    I --> K["Native projection with FP32 reduction and scaling"]
    K --> W{"BF16 output?"}
    W -->|no| Y2["Add bias in FP32 and cast to FP16"]
    W -->|yes| Q{"Fused BF16 epilogue present?"}
    Q -->|yes| T["bf16_rescale.raw<br/>out * scale + bias in FP32, then BF16 cast"]
    Q -->|no| U["Eager FP32 multiply and bias add, then BF16 cast"]
    H --> M["ExllamaV3TorchLinear._forward_eager"]
    M --> N2["reshape rows; convert to FP32"]
    N2 --> O["_forward_chunked: packed trellis decode and FP32 matmul"]
    O --> R2["add bias; cast to input dtype"]
```

## BF16 to FP16 conversion

BF16's 8-bit significand is a subset of FP16's 11-bit one, so every BF16
value inside FP16's finite range converts exactly. That covers ordinary activations and is
bit-exact with the plain FP16 path.

When the tensor's `amax` exceeds
`_FP16_SAFE_AMAX = 65504 / sqrt(128) ~= 5789.8`, each **row** whose own `amax`
exceeds the threshold is divided by its own power of two before the cast, and
the kernel output row is multiplied back:

- The Hadamard rotation of a 128-wide block is bounded by `sqrt(128)` times the
  row maximum, so a row maximum at or below `65504 / sqrt(128)` survives the
  rotation without overflowing FP16. This preserves #3731's guarantee: wide
  BF16 activations neither overflow nor produce NaN.
- `torch.frexp(row_amax / _FP16_SAFE_AMAX)` gives `2**exponent > row_amax /
  bound`, so the scaled maximum lies in `(bound / 2, bound]`: the smallest
  power of two that fits, not a normalization to `(0.5, 1]`. Power-of-two
  scaling is exact, so relative rounding is unchanged.
- The scale is per row because a per-tensor scale let one massive activation
  (a BOS outlier at amax ~1e3) divide every other row of the batch by the same
  factor. Ordinary rows then fell into FP16 subnormals (smallest normal
  6.1e-5) and lost most of their precision: 9.9e-2 relative error instead of
  2.7e-3 (#4870). Rows already inside the safe range keep scale 1 and convert
  bit-exactly.
- Scaling only as far as the bound also matters for the outlier row itself. A
  row that mixes a huge activation with small values (`amax` 1e3-2e4 next to
  elements around 1e-2, which is what a BOS massive-activation row looks like)
  used to be divided by its whole `amax`, so those small elements fell into
  FP16 subnormals: their norm error was 3.7e-3 at `amax` 1176 and 5.6e-2 at
  20000. Bringing the row just inside the bound keeps them normal, and in
  practice exact, because BF16 divides exactly by a power of two.
- The Hadamard bound also covers the GEMM accumulation for EXL3's packed weight
  scales, so the output row stays finite before it is rescaled.
- The BF16 bridge requests the raw projection. It rescales the FP32 output and
  adds the original checkpoint bias once before the BF16 cast.
- The decision stays on the device (#4955). A per-row `inf`-norm reduction,
  `frexp`, and a `where` give every row its own scale, and a row that needs no
  scale gets exactly 1.0, which is the identity. Nothing is read back to the
  host, so a BF16 forward no longer drains the CUDA stream once per linear:
  that `float(amax)` read was 112 synchronizations per decoded token on a 1B
  model, 4-8x slower than the same math without it, and it also prevented CUDA
  graph capture of the BF16 path. Since #5148 that is one fused launch per
  half of the mapping, against the hundreds of launches the Torch
  reconstruction performs.

Non-finite BF16 activations (`inf`/`nan`) keep the FP32 Torch path: there is no
power of two that would make them finite. `isfinite` is checked once per
module, on its first BF16 forward, instead of on every forward - a per-forward
`isfinite` is the same device-to-host sync the device-side decision removed. A
module whose first BF16 activation is non-finite keeps the FP32 reconstruction
and probes again on its next forward; once an activation is finite the module
stays on the native path.

## BF16 activation adaptation launch cost (#5148)

#4989 removed the per-forward host sync from this path, but the mapping was
still a chain of about a dozen eager kernels per linear: `vector_norm(inf)`,
`.float()`, `div`, `frexp`, `ones_like` x2, `ldexp`, `where`, `x / scale`,
`.to(fp16)`, then `result.to(fp32)`, `- bias`, `* scale`, `+ bias`, `.to(bf16)`.
At decode each is a microsecond kernel behind a full host dispatch, so the
33 us GEMM starved behind ~90 us of launches: BF16 decode ran 2.7x slower than
the same checkpoint in FP16.

Before, per BF16 linear (`gptqmodel/nn_modules/exllamav3.py` at 17a488e41,
`_native_bf16_input` + `_rescale_native_output`):

```mermaid
flowchart LR
    X["BF16 x"] --> A1["vector_norm inf"]
    A1 --> A2[".float"]
    A2 --> A3["div by bound"]
    A3 --> A4["frexp"]
    A4 --> A5["ones_like x2"]
    A5 --> A6["ldexp"]
    A6 --> A7["where"]
    A7 --> A8["x / scale"]
    A8 --> A9["to FP16"]
    A9 --> G["EXL3 GEMM"]
    G --> B1["to FP32"]
    B1 --> B2["- bias"]
    B2 --> B3["* scale"]
    B3 --> B4["+ bias"]
    B4 --> B5["to BF16"]
```

The current mapping uses two launches
(`gptqmodel_ext/exllamav3/bf16_scale.cu`, reached through
`gptqmodel/nn_modules/exllamav3_bf16_kernels.py`):

```mermaid
flowchart LR
    X["BF16 x"] --> P["bf16_prescale<br/>one launch: row amax,<br/>power-of-two scale, FP16 cast"]
    P --> G["EXL3 GEMM<br/>FP16 in, unbiased FP32 out"]
    G --> R["bf16_rescale.raw<br/>one launch: multiply, add bias,<br/>cast to BF16"]
    R --> Y["BF16 y"]
```

The pre-scale is one program per row: the row `amax` has to be complete before
that row is cast, and a launch has no grid-wide barrier, so the row stays in one
block. `frexp(amax / bound)` becomes the exponent field of the `amax / bound`
quotient (the quotient is always above 1 on the rows that use it), and the
divide is a multiply by the exact power-of-two reciprocal. The re-scale
computes `out * scale + bias` with a rounded FP32 multiply and a separate FP32
add before casting once to BF16.

**Failure contract.** The fused path declines instead of approximating:

| Situation | Path |
| --- | --- |
| Extension build without `bf16_prescale`/`bf16_rescale` (stale prebuilt wheel) | eager reference |
| Non-CUDA trellis | portable FP32 reconstruction (unchanged, #4789) |
| Non-contiguous, FP16, FP32, or empty activation | eager reference |
| First BF16 forward of a module (the one `isfinite` probe) | eager reference |
| Non-finite BF16 activation | portable FP32 reconstruction (unchanged) |

Both halves are bit-identical to the eager sequence, so a fallback is a pure
performance choice. `tests/test_exl3_bf16_fused_prescale_5148.py` compares raw
bits of the pre-scale, the re-scale (FP16/BF16/FP32/no bias, padded stored
width) and the complete forward against the fallback branch at M=1/8/512/2048
plus all-zero, denormal, `65504`, `3e38`, mixed-outlier and `inf`/`nan` rows.

Measured on an RTX 5090 (sm_120, torch 2.15.0+cu132, CUDA 13.2), eight
Llama-3.1-8B-geometry decoder layers quantized to EXL3 W4 with random weights
(no EXL3 checkpoint was available offline; the projections, kernels and shapes
are the real ones), batch 1 unless noted:

| Configuration | decode step bs1 | decode step bs8 | prefill 2048 | `generate()` 128 tok |
| --- | --- | --- | --- | --- |
| dense HF BF16 | 3.74 ms | 4.22 ms | 48.0 ms | 267 tok/s |
| EXL3 FP16 | 4.52 ms | 4.53 ms | 44.4 ms | 221 tok/s |
| EXL3 BF16, eager (before) | 8.84 ms | 8.88 ms | 54.9 ms | 113 tok/s |
| EXL3 BF16, fused (after) | 5.27 ms | 5.37 ms | 46.6 ms | 190 tok/s |

Single `mlp.gate_proj` (4096 -> 14336), M=1, CUDA-event minimum per call:
FP16 32.8 us, BF16 eager 88.2 us (2.69x), BF16 fused 35.8 us (1.09x). The
fused cost is two launches of ~2 us of GPU time plus ~6 us of host dispatch,
against ~60 us of dispatch for the eager chain.

## Folded input-range guard (#5942)

For the shapes the EXL3 GEMM owns, #5942 moves the guard into the GEMM. The
GEMM already reads the activation and `suh` for its input Hadamard, so the
same pass reduces the row bound, the narrowing cast applies the row's
power-of-two scale, and the epilogue applies the scale and the bias before
its single cast. The `bf16_prescale` + GEMM + `bf16_rescale.raw` sequence
above remains the reference and the fallback for the shapes the GEMM does
not own. Introduced in
[#5958](https://github.com/ModelCloud/GPTQModel/pull/5958).

Before:

```mermaid
flowchart LR
    X["FP16/BF16 x"] --> P["bf16_prescale<br/>row bound, scale, FP16 copy"]
    P --> G["EXL3 GEMM<br/>FP16 in, FP32 out"]
    G --> R["bf16_rescale.raw<br/>scale + bias, final cast"]
    R --> Y["final-dtype y"]
```

After:

```mermaid
flowchart LR
    X["FP16/BF16 x"] --> H["EXL3 GEMM prologue<br/>bound and scale in the input Hadamard"]
    H --> G["EXL3 GEMM<br/>FP16 in, FP32 accumulation"]
    G --> E["EXL3 GEMM epilogue<br/>row scale + bias, one final cast"]
    E --> Y["final-dtype y"]
```

The prologue zeroes the per-row `double` bound, reduces
`max(|x|, |x * suh|)` in the same warp loop that applies `2**-exponent` and
rotates, and publishes one atomic max per 128-element block. The epilogue
computes `out * 2**exponent + bias` and rounds once into a separate
final-dtype buffer; `C` stays the FP32 accumulation buffer so the split-K
partial sums keep their precision (#5331). The bound buffer is allocated per
call, so CUDA Graph capture and replay recompute it from the static input and
nothing is read back to the host.

**Precision.** FP16, and BF16 without checkpoint scales, bound, scale and
round in FP32. BF16 with `suh` uses FP64 for the bound, the scale and the
epilogue: a finite BF16 value times a finite FP16 `suh` can exceed FP32 even
when the projection is finite, and the epilogue must not round twice. This is
a deliberate trade-off: the FP64 path is taken for every BF16 checkpoint
(every EXL3 checkpoint carries `suh`), not only on rows whose FP32 product
overflows. An FP32 bound with a per-row FP64 fallback would need a second
reduction pass or a device-side flag; it is left as a follow-up. On an RTX
4090 the FP64 epilogue costs about 8 us on a 4096x4096, M=32 forward against
the unguarded FP16 GEMM.

**Failure contract.** The folded guard declines instead of approximating:

| Situation | Path |
| --- | --- |
| More than 32 rows on a non-Q4NX checkpoint (`LinearEXL3` reconstructs through HGEMM) | eager pre-scale reference |
| Q4NX without `fused_gemm` (the reconstruction dispatch) | eager pre-scale reference |
| `fp16_gemv` or `int8_gemv` policy (the GEMV writes its own epilogue) | eager pre-scale reference |
| R4NX, or a portable reconstruction with no tensor-core GEMM | eager pre-scale reference |
| Ampere Q4NX large prefill (the expanded-weight reconstruction) | eager pre-scale reference |
| Build without `bc_linear_exl3_run_guarded` (stale prebuilt extension) | eager pre-scale reference |
| Non-CUDA device, or an FP32 activation | portable FP32 reconstruction |
| First forward with a non-finite input (the one `isfinite` probe) | portable FP32 reconstruction |

The shape-independent declines are remembered on the module, so a forward
that must keep the eager reference does not pay the inner-handle signature
probe twice. The decision is cleared whenever the inner handle is rebuilt.

Measured on an RTX 4090 (torch 2.15.0+cu132, CUDA 13.2), the #5898 W4 MCG
fixture (K=512, N=256, `suh=1`, `svh=2**-10`, no bias, all-ones inputs).
Eager numbers are the minimum of five 200-forward CUDA-event trials, repeated
five times; graph numbers capture 20 forwards per graph and replay 100 times
per trial. "Unguarded" is the pre-#5898 FP16 layer forward (`_ensure_inner`
plus the native GEMM writing FP16); "eager" is the head tree with the folded
op disabled; "folded" is the head tree.

| Path | M=1 eager | M=33 eager | M=1 graph | M=33 graph |
| --- | ---: | ---: | ---: | ---: |
| Unguarded FP16 | 31.09 us | 60.71 us | 8.68 us | 6.72 us |
| FP16 eager guard (before) | 45.61 us | 81.25 us | 10.25 us | 8.32 us |
| FP16 folded guard (after) | 32.62 us | 81.11 us | 9.58 us | 8.31 us |
| BF16 eager guard (before) | 40.57 us | 81.04 us | 11.87 us | 9.96 us |
| BF16 folded guard (after) | 29.10 us | 82.28 us | 10.88 us | 9.96 us |

At M=1 the folded guard is +1.53 us (FP16) and -1.36 us (BF16) over the
unguarded FP16 layer path, against +14.52 and +10.11 us for the eager guard.
M=33 is unchanged because a non-Q4NX checkpoint reconstructs through HGEMM
above 32 rows; that path keeps the eager reference. The 4096x4096 prefill
shape follows the same split: FP16 M=32 graph replay is about +1.4 us over
the unguarded GEMM, while BF16 M=32 is about +8.1 us from the FP64 epilogue.

`tests/test_exl3_native_guard_5942.py` pins the dispatch (FP16/BF16, with and
without `suh`, padded width, Q4NX large prefill, every decline case, a stale
extension), bit-exact equality with the eager guard for FP16/BF16 at
M=1/4/32 with and without bias and with an outlier row, non-finite first
inputs, CUDA Graph replay with changed inputs, a BF16 product above FP32, and
the `_ensure_inner` decline cache.

## Dispatch contract

The row-count decision is inside `LinearEXL3.forward`. The bridge hands it a
contiguous FP16 tensor for both 32 and 33 rows, so the native class still picks
`bc.run` below 33 rows and `ext.hgemm` above it. A direct call to the
lower-level native class with BF16 or FP32 is outside this bridge contract and
still reaches the native dtype check.

| Mechanism | Before #4767 | After #4767 |
| --- | --- | --- |
| Dtype dispatch | FP16 native; BF16 and FP32 Torch | FP16 and BF16 native; FP32 Torch |
| BF16 input conversion | none (Torch, FP32 compute) | exact FP16 cast; per-row power-of-two pre-scale for rows above `65504 / sqrt(128)` |
| Wide-input safety | Torch path, FP32 | native FP16 with exact pre-scale; non-finite stays Torch |
| Native dtype contract | `gptqmodel_ext/exllamav3/quant/exl3_gemm.cu:64` and `hgemm.cu:33` require FP16 | Unchanged; the bridge satisfies it |

The wide Torch path keeps trellis, sign scales, and bias in packed checkpoint
form. `_forward_chunked` decodes at most 512 output columns at a time, then
releases that decoded weight after the forward. For a 128x128 four-bit test
layer, the packed trellis plus scales and bias occupy 8,960 bytes and the FP32
decoded weight chunk occupies 65,536 transient bytes.

## Tests

`tests/test_exl3_wide_input.py` covers the dispatch choice at 1, 32, 33, and 64
rows with a mocked native kernel, the exact FP16 cast, the power-of-two
pre-scale for magnitudes from 1e4 to 1e30, finite numerics against an FP64
reference at 1e4 and 1e5, and the non-finite fallback. The CPU oracle tests use
independently packed EXL3 symbols and a separate float64 weight reconstruction;
CUDA tests exercise the same bridge on NVIDIA when the extension is available.

`tests/test_exl3_bf16_prescale_4870.py` covers the pre-scale trigger and its
precision: rows between the old threshold and the bound are passed through
untouched, the row that carries a massive activation keeps its own small
elements (amax 260 to 20000), a wide row is scaled to just inside the bound,
the scale is per row, and a mixed-magnitude batch (64 rows of `N(0, 0.3)` with
one outlier in row 0) keeps the ordinary rows at the plain FP16 path's
accuracy.

`tests/test_exl3_bf16_fused_prescale_5148.py` pins the fused path: raw-bit
equality with the eager reference for the pre-scale, the re-scale and the whole
forward at M=1/8/512/2048 and on adversarial rows, the fallbacks (missing fused
op, non-contiguous, FP16/FP32, CPU), the absence of any retained dense weight,
and an M=1 `bf16 <= 1.3x fp16` per-linear bound. That last test measures the
eager chain in the same process with the fused ops disabled, so it fails on
17a488e41 by construction (2.7x) rather than relying on a historical number.
