# GPTQ planar checkpoint format (`gptq_p`)

`gptq_p` is a distinct GPTQ checkpoint format that stores quantized codes as
word-aligned **bit planes** ("split-plane" packing) instead of a continuous
bit stream. It is the native format for 5/6/7-bit GPTQ and an optional layout
for 2/3/4/8-bit. It is not interchangeable with the legacy continuous `gptq`
(v1) / `gptq_v2` formats.

## Why planar

The continuous layout packs codes back-to-back, so any width that does not
divide 32 produces codes straddling `int32` word boundaries, forcing
per-position carry/stitch logic in every pack, unpack, convert, and kernel
path. Planar packing splits each code into planes whose widths divide 32, so
every packed word contains whole fields and decode is branch-free shifts and
masks per plane plus an OR — the same approach GGUF uses for `q5_k`/`q6_k`.

Storage cost is identical to continuous packing — exactly
`ceil(n * bits / 32)` words — so `qweight`/`qzeros` keep their standard GPTQ
tensor names and shapes. Only the config metadata
(`checkpoint_format="gptq_p"`) distinguishes a planar checkpoint.

## Plane layouts

| bits | planes            | reconstruction                  |
|------|-------------------|---------------------------------|
| 2    | (2)               | single plane                    |
| 3    | (2) + (1)         | `q = lo \| hi << 2`             |
| 4    | (4)               | single plane                    |
| 5    | (4) + (1)         | `q = lo \| hi << 4`             |
| 6    | (4) + (2)         | `q = lo \| hi << 4`             |
| 7    | (4) + (2) + (1)   | `q = lo \| mid << 4 \| hi << 6` |
| 8    | (8)               | single plane                    |

Single-plane widths (2/4/8) produce words bit-identical to the continuous
2/4/8-bit layout, so `gptq_p` at those widths differs from `gptq_v2` only in
metadata.

## Storage contract

- Packed words are `int32`.
- Every 32 consecutive logical codes form a block stored as `bits` adjacent
  words: low-plane words first, then the higher planes.
- Within a plane of width `w`, word `i` holds codes
  `[i*(32//w), (i+1)*(32//w))` at shifts `w*j`, matching the standard
  2/4/8-bit packing convention.
- No code ever crosses a word boundary; any group of 32 codes decodes
  independently from `bits` words at a fixed offset (desc_act friendly).
- The packed dimension must be divisible by 32; misaligned shapes are
  rejected at module construction.
- `qweight` packs along rows, `qzeros` along columns.

## Format semantics

- Serialized as `checkpoint_format="gptq_p"`.
- Zero points use **v2 semantics** (true zero, no `-1` bias); planar
  checkpoints never pass through the legacy GPTQ v1 `+1` qzeros correction.
- 5/6/7-bit configurations auto-route to `gptq_p` (those widths have no
  continuous layout). Explicit `format="gptq_p"` is accepted for bits 2-8.
- Legacy `gptq`/`gptq_v2` at 2/3/4/8-bit keep their continuous layouts and
  conversion behavior unchanged; 3-bit is planar only under `gptq_p`.

## Validation summary

CPU validation covers planar pack/unpack round-trips for all widths
(including endpoint codes and misaligned-shape rejection), packer parity,
dequantization against logical-code references, planar-vs-continuous parity
(bit-identical dequant for 3-bit, bit-identical words for 2/4/8), v1/v2
zeros conversion round-trips, config routing/serialization, and whole-model
quantize -> save -> reload -> inference lifecycles for every bit width.
Measured dequantization error matches the theoretical quantization half-step
plus fp16 scale precision, with no systematic zero-point bias. GPU-native
planar kernels are a separate, later phase.
