# GPTQ planar checkpoint format (`gptq_p`)

`gptq_p` is a distinct GPTQ checkpoint format that stores quantized codes as
word-aligned **bit planes** ("split-plane" / "high-plane" packing) instead of a
continuous bit stream. It is the native format for 5/6/7-bit GPTQ and an
optional layout for 2/3/4/8-bit, and is not interchangeable with the legacy
continuous `gptq` (v1) / `gptq_v2` formats.

## Motivation

The continuous layout packs codes back-to-back, so any width that does not
divide 32 produces codes straddling `int32` word boundaries. For 3-bit this
forced the bespoke `10|1|10|1|10` layout, per-column carve-outs in every
pack/unpack/convert path, and the dedicated trilin kernel. For 5/6/7-bit the
boundary pattern is even more irregular (7-bit repeats only every 224 bits).

Planar packing removes boundary handling entirely: each code is split into
planes whose widths divide 32, so every packed word contains whole fields.
Decode is branch-free shifts/masks per plane plus an OR — the same approach
GGUF uses for `q5_k`/`q6_k`. Storage cost is identical to continuous packing:
exactly `ceil(n * bits / 32)` words, so `qweight`/`qzeros` keep their standard
GPTQ shapes.

See `docs/gptq_5_6_7_bit_plan.md` for the original feasibility analysis and
the GPU kernel roadmap.

## Plane layouts

`bits -> ((plane_width, bit_offset), ...)`, low to high
(`gptqmodel/utils/planar_packing.py`):

| bits | planes            | reconstruction                  |
|------|-------------------|---------------------------------|
| 2    | (2)               | single plane                    |
| 3    | (2) + (1)         | `q = lo \| hi << 2`             |
| 4    | (4)               | single plane                    |
| 5    | (4) + (1)         | `q = lo \| hi << 4`             |
| 6    | (4) + (2)         | `q = lo \| hi << 4`             |
| 7    | (4) + (2) + (1)   | `q = lo \| mid << 4 \| hi << 6` |
| 8    | (8)               | single plane                    |

Single-plane widths (2/4/8) produce words **bit-identical** to the continuous
2/4/8-bit layout, so `gptq_p` at those widths differs from `gptq_v2` only in
metadata.

## Storage contract

- Packed words are `int32` (`pack_dtype=torch.int32` required).
- Every 32 consecutive logical codes form a block stored as `bits` adjacent
  words: first the low-plane words, then the higher planes.
- Within a plane of width `w`, word `i` holds codes
  `[i*(32//w), (i+1)*(32//w))` at shifts `w*j` (row-major `[bits, pack_factor]`
  convention, matching the 2/4/8-bit packers).
- No code ever crosses a word boundary; any group of 32 codes decodes
  independently from `bits` words at a fixed offset (desc_act friendly).
- The packed dimension must be divisible by 32; helpers raise `ValueError`
  otherwise. Module construction rejects misaligned shapes up front with
  `NotImplementedError`. Known limitation: for quantized embeddings the
  packed dimension is the vocabulary size, so non-32-aligned vocabs
  (e.g. 50257, 32001) cannot use planar widths until padding support lands
  (planned with the GPU phase).
- `qweight` packs along rows (`planar_pack_rows`), `qzeros` along columns
  (`planar_pack_cols`).

## Format semantics

- `FORMAT.GPTQ_P = "gptq_p"`, serialized as `checkpoint_format="gptq_p"`.
- Zero points use **v2 semantics** (true zero, no `-1` bias). Planar
  checkpoints never pass through the legacy GPTQ v1 `+1` qzeros correction on
  save or load.
- `QuantizeConfig(bits=5/6/7)` auto-routes `format` from `gptq`/`gptq_v2` to
  `gptq_p` (those widths have no continuous layout). Explicit
  `format="gptq_p"` is accepted for bits 2–8.
- Legacy `gptq`/`gptq_v2` at 2/3/4/8-bit keep their continuous layouts and
  conversion behavior unchanged; 3-bit is planar only under `gptq_p`.
- `GPTQQuantLinear` receives the checkpoint format at construction and exposes
  a `planar` flag that routes `pack_block`/`pack_gpu`/`pack_original`,
  `dequantize_weight`, wf shift-buffer setup, and the Triton dequant gate.

## Execution

CPU inference runs on the merged Torch kernel (`TorchLinear`), which supports
`gptq`, `gptq_v2`, and `gptq_p` in one implementation: planar modules decode
via `planar_unpack_rows`/`planar_unpack_cols`; continuous modules keep the
existing shift-buffer paths. Thread-parallel `pack_block` (free-threaded
Python, GIL=0) is deterministic: workers write disjoint 32-aligned row blocks
of the preallocated output.

GPU-specific planar kernels are a later phase (see the plan doc): the day-1
fast path is expanding exact codes to uint8 and reusing the stock 8-bit Marlin
kernel, followed by a Triton multi-plane dequant.

## Testing and validation results

All results below are from CPU validation on free-threaded Python 3.14.6
(GIL=0 verified), torch 2.13.0+cpu, backend=TORCH.

### Test suites

| Suite | Coverage | Result |
|---|---|---|
| `tests/test_planar_bits_567.py` (27) | 5/6/7-bit round-trips, packer parity, threaded determinism, saturation, dequant reference, v1/v2 zeros conversion, tiny-model lifecycle, desc_act, sym | pass |
| `tests/test_planar_format_gptq_p.py` (41) | planar round-trips for all bits 2–8, 2/4/8 word parity vs continuous, planar-3 vs continuous-3, config routing/serialization, metadata save/load, tiny-model `gptq_p` lifecycle (bits 3 and 5) | pass |
| `tests/test_torch_kernel_accuracy.py` (69) | dequant vs logical-code reference and forward vs `x @ W_ref` for all bits and both layouts, fp16 + bf16, batched and larger shapes, desc_act, sym | pass |

### Independent math audit

Checked against independent references (pure-Python bit-level packers,
float64 forward, raw bitstream decode), not the test suites' own formulas:

| Check | Method | Result |
|---|---|---|
| Planar bit layout, bits 2–8 | tensor packer vs pure-Python bit-level reference | bit-exact |
| Continuous 3-bit layout | qweight decoded as raw little-endian 3-bit stream | bit-exact |
| Dequant element error (all bits, both formats) | max err/scale on non-clamped weights | 0.500–0.523 = theoretical half-step + fp16-scale ULP (`maxq * 2^-10`); mean bias ~1e-3, no systematic zero shift |
| Forward error (1024x256, gs=128) | module(x) vs float64 `x @ W_deq` | ~3e-4 relative = fp16 matmul noise; planar identical to continuous |
| Edge zero points (0, maxq) + desc_act | element bound audit | no violations |
| v1 <-> v2 zeros round-trip | dequant before/after | bit-equal |
| Planar-3 vs continuous-3 | dequant + forward | bit-identical |

### End-to-end

Tiny-Llama quantize -> save -> load -> generate lifecycles pass for bits
5/6/7 (auto `gptq_p`) and explicit `gptq_p` bits 3/5/6; saved `config.json`
carries `checkpoint_format="gptq_p"`, reloaded modules report `planar=True`,
and safetensor-level dequant recovers logical codes bit-exactly under v2 zero
semantics. Legacy continuous 3-bit checkpoints still save/load with v1 zero
correction, bit-exact.

### Not yet covered

- GPU kernels and GPU correctness validation (CPU-only so far).
- Expanded-8-bit Marlin fallback and Triton planar dequant (planned).
