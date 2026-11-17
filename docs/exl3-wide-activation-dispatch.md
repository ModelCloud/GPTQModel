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
    B -->|float16| C["_ensure_inner"]
    B -->|bfloat16 on CUDA| D["_native_bf16_input"]
    D --> E{"amax finite?"}
    E -->|no| H["_ensure_torch_inner"]
    E -->|yes| F{"amax <= 65504/sqrt(128)?"}
    F -->|yes| G["cast to FP16 (exact)"]
    F -->|no| P["divide each wide row by the smallest<br/>power of two that fits, cast to FP16"]
    G --> C
    P --> C
    C --> I["LinearEXL3.forward"]
    I --> J{"rows greater than 32?"}
    J -->|no| K["bc.run: FP16 native GEMM"]
    J -->|yes| L["ext.hgemm: FP16 native GEMM"]
    P --> Q["multiply kernel output back; bias excluded from the scale"]
    B -->|float32| H
    H --> M["ExllamaV3TorchLinear._forward_eager"]
    M --> N["reshape rows; convert to FP32"]
    N --> O["_forward_chunked: packed trellis decode and FP32 matmul"]
    O --> R["add bias; cast to input dtype"]
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
- The native kernels add the bias to the scaled accumulator. The bridge removes
  the bias, rescales, and restores it in FP32, so the bias is never scaled.
- The decision stays on the device (#4955). A per-row `inf`-norm reduction,
  `frexp`, and a `where` give every row its own scale, and a row that needs no
  scale gets exactly 1.0, which is the identity. Nothing is read back to the
  host, so a BF16 forward no longer drains the CUDA stream once per linear:
  that `float(amax)` read was 112 synchronizations per decoded token on a 1B
  model, 4-8x slower than the same math without it, and it also prevented CUDA
  graph capture of the BF16 path. That is a couple of small kernels, against
  the hundreds of launches the Torch reconstruction performs.

Non-finite BF16 activations (`inf`/`nan`) keep the FP32 Torch path: there is no
power of two that would make them finite. `isfinite` is checked once per
module, on its first BF16 forward, instead of on every forward - a per-forward
`isfinite` is the same device-to-host sync the device-side decision removed. A
module whose first BF16 activation is non-finite keeps the FP32 reconstruction
and probes again on its next forward; once an activation is finite the module
stays on the native path.

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
