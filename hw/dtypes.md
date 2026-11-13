# Dtype reference: bit layouts, weights, and arithmetic

This is the single place where GPT-QModel defines what each dtype *is*. A dtype
name appears in three different roles, and they must not be conflated:

| Role | Meaning | Example |
| --- | --- | --- |
| **Weight storage** | how a checkpoint stores the numbers on disk | int4 packed codes; `bf4_b` codes plus exponents |
| **Arithmetic** | what the multiply/accumulate unit consumes | BF16 or FP32 on Tenstorrent; FP8 on SM89+ |
| **Scale** | a multiplier stored beside the values | `E8M0` in MX; `E4M3` in NVFP4 |

A format can be one, two, or all three. `bf4_b` is storage only — the FPU never
multiplies it. `E8M0` is a scale only — it never appears as a weight value.

## 1. Naming policy

- **`bf4_b` / `bf8_b`** are this repository's names for the two Tenstorrent
  block-float formats (§8). They are the vendor spelling `bfloat4_b` /
  `bfloat8_b` with `bfloat` shortened to `bf`; the `_b` suffix is **kept**
  because it is load-bearing, not decoration (§8.5). No other provider claims
  `bf4_b`/`bf8_b` today, and the ttnn spelling lives in one place,
  `TT_BLOCKFLOAT_VENDOR_ATTRS` in `gptqmodel/utils/tt_blockfloat_quant.py`.
- **Vendor spellings** (`bfloat4_b`, `bfloat8_b`) appear only at the
  `getattr(ttnn, ...)` call site.
- **Standard names are not renamed.** `fp32`, `bf16`, `fp8_e4m3`, `mxfp4` and
  `nvfp4` mean what their specifications say.
- **`fp8` alone is ambiguous.** This repository uses it for OCP FP8 (E4M3 or
  E5M2) with a checkpoint-declared block scale. Where the scale granularity is
  part of the format, the specific name (`mxfp8`, `nvfp8`) is used.

## 2. Quick reference

`S` = sign bits, `E` = exponent bits, `M` = stored mantissa bits. Every binary
float also has one implicit leading mantissa bit unless the value is subnormal.

| name | bits | S | E | M | value range | weight role | arithmetic role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fp64` | 64 | 1 | 11 | 52 | ±1.8e308 | dense tensors | yes (CPU) |
| `fp32` | 32 | 1 | 8 | 23 | ±3.4e38 | scales, norms, accumulate | yes |
| `fp16` | 16 | 1 | 5 | 10 | ±65504 | legacy dense | yes |
| `bf16` | 16 | 1 | 8 | 7 | ±3.4e38 | dense, dequantized tiles | **yes — TT compute dtype** |
| `int8` | 8 | — | — | — | −128…127 | byte planes, codes | integer ops only |
| `int4` (GPTQ) | 4 | — | — | — | −8…7 or 0…15 | **yes — GPTQ weights** | no (expanded first) |
| `fp8_e4m3` | 8 | 1 | 4 | 3 | ±448 | FP8 checkpoints | yes on SM89+ |
| `fp8_e5m2` | 8 | 1 | 5 | 2 | ±57344 | FP8 checkpoints | yes on SM89+ |
| `e8m0` | 8 | 0 | 8 | 0 | 2^−127 … 2^127 | MX scale | no |
| `fp4_e2m1` | 4 | 1 | 2 | 1 | ±6 | NVFP4/MXFP4 values | SM100+ only |
| `nvfp4` | 4 + 8 + 32 | 1 | 2 | 1 | ±6 × scales | weights, activations | SM100+ |
| `mxfp4` | 4 + 8 | 1 | 2 | 1 | ±6 × scale | GGUF `MXFP4` | emulated |
| `nvfp8` | 8 + scale | 1 | 4 | 3 | ±448 × scale | weights | SM89+ |
| `mxfp8` | 8 + 8 | 1 | 4 | 3 | ±448 × scale | weights | emulated |
| **`bf4_b`** | 4 + 0.5 | 1 | shared | 2 | ±1.75·2ᵉ | **yes — TT block float** | no (unpacked to BF16) |
| **`bf8_b`** | 8 + 0.5 | 1 | shared | 6 | ±1.98·2ᵉ | **yes — TT block float** | no (unpacked to BF16) |
| `nf12` | 12 | 1 | 8 | 3 | bf16 range | lossless bf16 encoding | no |

## 3. IEEE binary floats

`value = (−1)^S · 1.M · 2^(E − bias)`, with subnormals at `0.M · 2^(1 − bias)`
and reserved exponent patterns for zero, infinity and NaN.

| dtype | bias | max finite | min normal | min subnormal |
| --- | --- | --- | --- | --- |
| `fp64` | 1023 | 1.797e308 | 2.2e−308 | 4.9e−324 |
| `fp32` | 127 | 3.402e38 | 1.18e−38 | 1.4e−45 |
| `fp16` | 15 | 65504 | 6.10e−5 | 5.96e−8 |
| `bf16` | 127 | 3.390e38 | 1.18e−38 | 9.2e−41 |

`bf16` is fp32 with the mantissa truncated from 23 stored bits to 7: the
exponent width and range are identical, only precision is lost. That is why it
is the safe default for a dequantized tile or an accumulator hand-off.

`fp16` makes the opposite trade — a 10-bit mantissa but a 5-bit exponent — so it
overflows at 65504. Never store a dequantized tile in `fp16` if a block scale
can push a value past that; this repository stores dense remainders at the
model's own dtype and computes TT tiles in `bf16` for exactly this reason.

## 4. Integer storage

`int8` is two's complement, 8 bits, −128…127. It is not a float and has no
exponent; any scale lives beside it.

`int4` is the GPTQ weight format: a signed 4-bit code with a per-group scale and
usually a zero point, packed two per byte. The grid is uniform and affine,
`w = (q − zero) · scale`, which is why it cannot represent a block-float value
exactly. Kernels expand int4 to BF16 or FP32 before multiplying, except where a
fused int4 kernel expands per tile.

## 5. FP8 family

| dtype | S | E | M | bias | max | inf/NaN | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fp8_e4m3` | 1 | 4 | 3 | 7 | 448 | no inf; NaN = `S.1111.111` | OCP E4M3, torch `float8_e4m3fn` |
| `fp8_e5m2` | 1 | 5 | 2 | 15 | 57344 | yes | OCP E5M2, torch `float8_e5m2` |
| `fp8_e4m3fnuz` | 1 | 4 | 3 | 8 | 240 | no; NaN = `0x80` | unsigned zero, AMD/Graphcore |
| `fp8_e5m2fnuz` | 1 | 5 | 2 | 16 | 57344 | no | unsigned zero |
| `e8m0` | 0 | 8 | 0 | 127 | 2^127 | NaN = `0xFF` | **scale only** |

`e8m0` has no sign and no mantissa: the byte *is* the biased exponent, so the
decoded value is always a power of two. `decode_e8m0_scale()` implements this,
including the all-ones byte meaning NaN rather than a huge finite scale.

## 6. FP4 (E2M1)

One sign, two exponent (bias 1), one stored mantissa. Sixteen codes, and the
finite magnitudes are exactly:

```
0, 0.5, 1, 1.5, 2, 3, 4, 6
```

`FP4_E2M1_MAX = 6.0` and the rounding table in
`gptqmodel/quantization/activation_floatx.py` (`_e2m1_nearest_values`) are the
authoritative implementation, with ties to even and saturation at 6.

## 7. Block-scaled families

Two scale granularities are in use, and they are not interchangeable.

**OCP microscaling (MX)** — one `E8M0` scale per **32** values. The scale is a
power of two, so the stored values are exact shifts of the codebook.

| name | values | scale | block |
| --- | --- | --- | --- |
| `mxfp8` | `fp8_e4m3` | `E8M0` | 32 |
| `mxfp4` | `fp4_e2m1` | `E8M0` | 32 |

GGUF's `MXFP4` tensor type is this layout: `type_size` 17 bytes per 32 values,
i.e. 16 bytes of packed E2M1 plus one `E8M0` byte.

**NVIDIA (NV)** — two levels of scale, which is the difference that matters:

| name | values | block scale | global scale | block |
| --- | --- | --- | --- | --- |
| `nvfp4` | `fp4_e2m1` | `fp8_e4m3` | `fp32`/`fp16`/`bf16` | **16** |
| `nvfp8` | `fp8_e4m3` | per checkpoint declaration | — | config |

The effective NVFP4 multiplier is `local_e4m3 · global`, so it is not a power of
two and cannot be expressed as an MX scale. A `nvfp4` checkpoint is therefore
not loadable as `mxfp4` even though both hold E2M1 values, and neither is
loadable as `bf4_b`/`bf8_b`.

`nvfp8` is dequantized by the same FP8 path as plain `fp8`; the name records the
checkpoint's declared scale semantics, not a distinct kernel.

## 8. Tenstorrent block floats: `bf4_b` and `bf8_b`

This is the format this repository quantizes *into*. It is **not** an industry
format, and the exact definition follows.

### 8.1 Layout

- Values are grouped into **blocks of 16 consecutive input features**.
- Each block shares **one plain power-of-two exponent** `e` (an `int8` per
  block), taken from the largest magnitude in the block.
- Each value stores a sign and a fixed number of mantissa bits, with one
  implicit leading bit:

| name | stored mantissa | significant bits | levels per binade | step | codes | representable range |
| --- | --- | --- | --- | --- | --- | --- |
| `bf4_b` | 2 | 3 | 4 | `2^e / 4` | 0…14 (15) | `±1.75 · 2^e` |
| `bf8_b` | 6 | 7 | 64 | `2^e / 64` | 0…254 (255) | `±1.984375 · 2^e` |

So `bf4_b` holds **4 levels inside one binade** — `0.5, 0.625, 0.75, 0.875`
times `2^e` for the binade starting at `2^e` — and `bf8_b` holds 64.

### 8.2 Exact encode / decode

```
offset = 2 · levels − 1              # 7 for bf4_b, 127 for bf8_b
levels = 2^(stored mantissa bits)     # 4 for bf4_b, 64 for bf8_b

decode:  value = (code − offset) / levels · 2^exponent
encode:  code  = clamp(round(value / 2^exponent · levels), −offset, +offset) + offset
```

Decoding is exact arithmetic and never re-quantizes. Encoding is
round-to-nearest-even, matching the unpack stage.

### 8.3 Storage cost

| name | bits per value | exponent amortized | effective bytes/weight |
| --- | --- | --- | --- |
| `bf4_b` | 4 | 8 / 16 = 0.5 | **0.5625** |
| `bf8_b` | 8 | 8 / 16 = 0.5 | **1.0625** |

### 8.4 Weights versus calculations

`bf4_b` and `bf8_b` are **storage formats only**. The Blackhole unpack stage expands
them to BF16 in the data path (`L1 → unpack → source registers → FPU`), so there
is no dequant op in the graph — but there is also no block-float arithmetic
unit. The multiply is always BF16 or FP32. This is measured, not assumed: see
`hw/tenstorrent.md` §3.

Consequences that shape kernels:

- The exponent is **per 16-value block, shared across those values**, so the
  quantizer must fix it before solving. Recomputing it per column would let it
  drift and produce values the format cannot store.
- Values are symmetric and logarithmic. There is no zero point, so an affine
  int4 checkpoint cannot be re-encoded losslessly — measured ~9.8 % normalized
  error at 4 bits.
- The grid is a *superset* of what a 4-level binade suggests: 15 codes for
  `bf4_b`, spanning `±1.75 · 2^e`. A quantizer that assumes 4 codes total will
  disagree with the hardware.

### 8.5 Naming, and why the `_b` stays

Upstream has two spellings, and only one of them reaches Python.

**C++ (`tt::DataFormat`, `tt-metal/api/tt-metalium/tt_backend_api_types.hpp`)** —
both families exist, distinguished by the exponent width:

| enum | value | shared exponent |
| --- | --- | --- |
| `Bfp8` / `Bfp4` / `Bfp2` | 2 / 3 / 11 | 5-bit (`ExpPrecision::A`) |
| `Bfp8_b` / `Bfp4_b` / `Bfp2_b` | 6 / 7 / 15 | 8-bit (`ExpPrecision::B`) |
| `Float16` / `Float16_b` | 1 / 5 | 5-bit / 8-bit |

`_b` is not a revision letter: `is_exp_b_format()` in
`tt_metal/jit_build/data_format.cpp` maps it to `ExpPrecision::B`, i.e. an
8-bit shared exponent with bfloat16's range, and the same file notes the
A-family is not shipped ("`Float16` has a 5-bit exponent that would silently cap
fp32 magnitudes"). This quantizer uses the 8-bit exponent, so `bf4`/`bf8` alone
would name the *other* format.

**Python (`ttnn`)** — only the B family is exposed:

```python
ttnn.bfloat4_b, ttnn.bfloat8_b   # present
ttnn.bfloat4,   ttnn.bfloat8     # do not exist
```

`ttnn.DataType` has `BFLOAT4_B`, `BFLOAT8_B`, `BFLOAT16`, `FLOAT32`,
`FP8_E4M3` and the integer widths. There is **no** `_a` or `_c` variant of
either format. The only `_A`/`_B` suffixed members in the enum are
`MxFp4_2x_A` (27) and `MxFp4_2x_B` (29), which are MXFP4 2x layout variants,
not block floats.

So: `bfloat8_b` at the vendor call site, `bf8_b` everywhere else. Outside that
call site `bfloat8_b` reads like an 8-bit bfloat — there is no such type — and is
easily mistaken for OCP FP8 E4M3, which it is not; hence the short form here.

`bfloat2_b` (`Bfp2_b`, BFP2) exists in the C++ enum and in Tenstorrent
documentation, but the `ttnn` build on this host does not expose it, so it is
not defined here.

## 9. `nf12`

A lossless 12-bit encoding of `bf16`: sign, the full 8-bit exponent, and the top
3 mantissa bits, with the remaining bits stored as escapes when a value is not
representable. It exists to shrink `bf16` checkpoints without changing any
value; it is not a compute dtype.

## 10. E/M variants and software emulation

Two different things are called a "variant", and the distinction matters when
reading a checkpoint's metadata:

**E/M variants** are different exponent/mantissa splits of the same bit width.
They are *different formats*, not encodings of one another:

| family | variants in this package | what changes |
| --- | --- | --- |
| FP8 | `e4m3`, `e5m2`, `e4m3fnuz`, `e5m2fnuz` | range vs precision; fnuz removes negative zero and infinity |
| FP4 | `e2m1` | the only FP4 split in use |
| FP6 | none | not supported |

**Emulation variants** are software realizations of a format the device does not
implement. They are numerically the same dtype, computed differently:

| format | native on | emulated how |
| --- | --- | --- |
| FP8 | SM89+, and via TTNN unpack for storage | widened to BF16/FP32 before the multiply |
| FP4 (E2M1) | SM100+ | unpacked to BF16, or decoded through a lookup table |
| MX/NV formats | SM100+ (NVFP4), limited elsewhere | dequantized to the target dtype on CPU or GPU |
| `bf4_b`/`bf8_b` | never (storage only) | hardware unpack to BF16 |

An emulated multiply is **not** bit-identical to a native one: the emulation
rounds to the compute dtype at the widening step, so results can differ in the
last bit from a device that multiplies the narrow format directly. Any test that
asserts exactness must therefore say which realization it ran.

## 11. Where the definitions live in code

| Concern | File |
| --- | --- |
| float8/float4 registry, E8M0 decode, FP8/FP4 dequant | `gptqmodel/quantization/dtype.py` |
| NVFP4 scale selection and E2M1 rounding | `gptqmodel/quantization/activation_floatx.py` |
| GGUF MXFP4 packing | `gptqmodel/nn_modules/qlinear/gguf.py` |
| FP8 / NVFP8 / MX checkpoint dequantization | `gptqmodel/utils/model_dequant.py` |
| `bf4_b` / `bf8_b` grid, codes, exponents | `gptqmodel/utils/tt_blockfloat_quant.py` |
| `bf4_b` / `bf8_b` modules and packing | `gptqmodel/nn_modules/qlinear/tt_blockfloat.py` |
| `nf12` codec | `gptqmodel/quantization/nf12_ad.py` |

## 12. Adding a dtype

1. Define the bit layout here first, including the weight role and the
   arithmetic role. A dtype that is storage-only must say so.
2. Add it to the registry in `gptqmodel/quantization/dtype.py` so CPU dequant,
   config normalization and checkpoint detection agree.
3. Declare it in the kernel's `SUPPORTS_*` lists; do not let a kernel accept a
   dtype by falling through a default branch.
4. State the emulation, if any, and whether results are bit-identical to native.
