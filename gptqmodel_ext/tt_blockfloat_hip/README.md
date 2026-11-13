# Tenstorrent bf8/bf4 emulation on ROCm

HIP kernels that execute GPT-QModel's Tenstorrent `bf8_b` / `bf4_b`
checkpoints on AMD GPUs. They exist so the formats can be tested and emulated
on hardware that is not a Tenstorrent device.

The kernels consume the checkpoint exactly as the Tenstorrent runtime does:
packed uint8 `codes` plus one int8 `exponents` value per 16 consecutive K
values. Nothing is re-quantised, converted to int4, or expanded on the host.

## Format

| format | code bits | stored mantissa | levels | value |
| --- | --- | --- | --- | --- |
| `bf8_b` | 8 | 6 | 64 | `(code - 127) / 64 * 2**exponent` |
| `bf4_b` | 4 | 2 | 4 | `(code - 7) / 4 * 2**exponent` |

`codes` are packed at their natural density: one byte per value for bf8, two
nibbles per byte for bf4 with the lower column in the low nibble.  `ttnn` spells
these `bfloat8_b` and `bfloat4_b`; that vendor spelling is used only at the
`getattr(ttnn, ...)` call site, and `hw/dtypes.md` is the bit-level reference. The exponent
is shared by a 16-value block, matching `gptqmodel.utils.tt_blockfloat_quant`.

The dequantised values are exact float32 grid points, and every bf8/bf4 value
is also exactly representable in bf16 (7 and 3 significant bits respectively),
so the emulation adds no rounding of its own.

## Kernels

All of these are in `tt_blockfloat_hip.hip` and are torch-free.

| entry point | what it does |
| --- | --- |
| `ttbf_dequant` | packed codes -> dense `f32`/`f16`/`bf16`, eight values per thread |
| `ttbf_quantize` | float32 -> codes + exponents, including the three-candidate exponent search |
| `ttbf_decompose` | already-on-grid float32 -> codes + exponents, exact |
| `ttbf_linear` | fused unpack + GEMM, `x @ W.T + bias` |
| `ttbf_embedding` | gather rows by id and dequantise only those rows |
| `ttbf_selftest` | sentinel proving the device code executed |

`ttbf_linear` has two shapes:

* **Decode (`m == 1`)** is bandwidth-bound. One CTA owns 32 output features,
  stages the activation row in shared memory once, and each lane reads exactly
  one exponent and 8 (bf4) or 16 (bf8) code bytes per K step. The packed weight
  is read once. The 16-value activation segments use a 17-float stride so the
  per-lane read hits all 32 LDS banks instead of two; 512 threads (16 warps x
  2 rows) fill the CUs' thread slots and keep enough loads in flight to
  saturate HBM.
* **Prefill (`m > 1`)** is a 32x64x32 register-tiled GEMM (256 threads, eight
  fp32 accumulators per thread). The weight tile is unpacked straight into
  shared memory as exact fp32 grid values, so the inner loop is plain FMA.
  bf16 activations on `gfx1200`/`gfx1201` instead use a WMMA kernel whose CTA
  block, warp block and K tile are all chosen at launch:

  | tokens | CTA block | warp block | K | buffers | threads |
  | --- | --- | --- | --- | --- | --- |
  | `m <= 16` | 16x64 | 16x16 | 256 | 1 | 128 |
  | `m <= 32` | 32x64 | 16x16 | 256 | 1 | 256 |
  | `m <= 64` | 64x64 | 16x16 | 128 | 1 | 256 |
  | `m > 64` | 128x128 | 32x32 | 64 | 1 | 512 |

  The M block matters because the whole `n x k` weight is streamed once per M
  block, so it sets how often the packed weight is re-read *and* how much total
  unpack work the call does. The warp block matters because a warp's WMMA needs
  one A fragment and one W fragment from shared memory per 16x16 quadrant: at
  16x16 that is 1 KiB of LDS traffic per WMMA, at 32x32 each fragment is reused
  across two quadrants and it is 512 B. `K` sets the bytes of a weight row a
  tile reads (`K/2` at bf4 density), which decides whether a warp streams a
  full 128-byte line.

No kernel writes an expanded weight to global memory. The only expanded copy
is the shared-memory tile consumed by the current block.

## Python API

```python
from gptqmodel.utils import tt_blockfloat_hip

tt_blockfloat_hip.prewarm_tt_blockfloat_hip_extension()
out = tt_blockfloat_hip.linear(x, module.codes, module.exponents, module.bits, bias)
codes, exponents = tt_blockfloat_hip.quantize(weight, bits=4)
dense = tt_blockfloat_hip.dequantize(codes, exponents, 4, weight.shape, torch.float32)
```

`TTBlockfloatLinear`, `TTBlockfloatEmbedding` and `TTBlockfloatTiedHead`
automatically use these kernels when the module is on a ROCm device; the torch
path remains the fallback everywhere else.

## Build and validate without ROCm PyTorch

The JIT extension is the production entry point, but the kernels can be
validated on any ROCm host with only `hipcc`:

```bash
hipcc --offload-arch=gfx1201 -O3 -std=c++17 \
    tt_blockfloat_hip.hip tt_blockfloat_hip_selftest.cpp -o ttbf_selftest
./ttbf_selftest --dump /tmp/ttbf_dump
python verify_against_torch.py /tmp/ttbf_dump
# throughput sweep instead of the default shapes
./ttbf_selftest --bench 32 4096 4096
```

`ttbf_selftest` compares every kernel against an independent CPU
implementation. `verify_against_torch.py` compares the same outputs against
`gptqmodel.utils.tt_blockfloat_quant` (CPU torch is sufficient), so the
checkpoint oracle is exercised even when the host has no ROCm-built PyTorch.

Measured on ROCm 7.16 / HIP 7.16.26385, Radeon AI PRO R9700 (`gfx1201`):

```
bf4 dequant f32 bit-exact                            PASS
bf4 dequant bf16 rounds correctly                    PASS
bf4 quantize matches CPU oracle                      PASS
bf4 decompose round-trips exactly                    PASS
bf4 linear m=1 norm_l1=4.84e-08                      PASS
bf4 linear m=32 norm_l1=1.48e-07                     PASS
bf4 linear bf16 m=32 (WMMA) norm_l1=0                PASS
bf8 dequant f32 bit-exact                            PASS
bf8 dequant bf16 rounds correctly                    PASS
bf8 quantize matches CPU oracle                      PASS
bf8 linear m=1 norm_l1=1.39e-07                      PASS
bf8 linear m=32 norm_l1=1.81e-07                     PASS
bf8 linear bf16 m=32 (WMMA) norm_l1=0                PASS
bf4 embedding gather bit-exact                       PASS
decode 4096x4096 bf4: ~18 us/call, ~490 GB/s on the code stream
prefill 4096x4096 bf4 bf16, by token count:
   m=16    32 us   17 TFLOP/s
   m=32    35 us   31 TFLOP/s
   m=64    43 us   50 TFLOP/s
   m=128   54 us   79 TFLOP/s
   m=256   93 us   92 TFLOP/s
   m=512  186 us   92 TFLOP/s
prefill m=32 fp32 (no matrix cores): 686 us, 1.6 TFLOP/s
```

Every one of those M values is checked for correctness in the same run, along
with `m = 15, 17, 31, 33, 100` to cover the partial-tile masking on both sides of
the 16- and 32-row boundaries. `tests/test_tt_blockfloat_hip.py` parametrises
the same set.

The Torch cross-check additionally reports `bf4/bf8 quantize weights
bit-exact`, `bf4/bf8 dequant f32 bit-exact`, the bf16 rounding check, and the
embedding gather as exact.

### Where the time goes

For a 4096x4096 matrix, bf4 storage is 8 MiB of codes plus 1 MiB of exponents
against 32 MiB for a bf16 weight, so the decode kernel moves roughly a quarter
of the bytes a dense bf16 layer would.  It sustains ~490 GB/s on the code
stream (552 GB/s counting exponents) against 607 GB/s measured for a pure
`float4` read loop on this card, i.e. it is at the roofline and further decode
gains would need fewer bytes, not a better kernel.

Prefill is a different story.  It was bounded by two things the tile shape
controls, and both are now chosen per shape instead of being fixed.

The first is how often the packed weight is re-read.  The grid is
`(n / NTILE, m / MTILE)`, so the whole `n x k` weight is streamed once per M
block.  At `MTILE = 32` and `m = 512` that is sixteen passes over the 9 MiB
weight, which no longer fits the 8 MiB L2, so the kernel was pulling it from
HBM and unpacking the same codes sixteen times.  A 128-row block cuts both to a
quarter of that; the 64-row step alone was worth 1.55x at `m = 512`.

The second is LDS traffic per WMMA.  A 16x16 warp block loads one A fragment and
one W fragment -- 1 KiB -- for a single WMMA, and at 128 B/cycle that is eight
cycles, longer than the WMMA itself, so the kernel was LDS-bandwidth bound
rather than matrix-core bound.  A 32x32 warp block reuses each fragment across
two quadrants and halves it to 512 B, which is what took `m = 512` from 58 to
92 TFLOP/s.

`K` still matters, for a different reason: a bf4 weight row is `k/2` bytes, so a
32-wide tile gives a warp 16 bytes per row and it touches eight 128-byte lines
to use 16 bytes of each, whereas `K = 256` makes the per-row run a full line.
That is why the small-M shapes keep the wide `K = 256` tile -- they are the ones
that cannot amortise a narrow row run over enough CTAs.

`m = 16` and `m = 32` are the floor of that curve and are not a tuning miss.
The call has `(m / 16) * (n / 16)` output tiles -- 256 and 512 -- and a 16x16
warp block owns one tile, so the whole card gets 8 and 16 warps per CU however
the blocks are cut.  That is a hard ceiling for these shapes.  The tradeoff is
real, not theoretical: at `m = 32` a 32x32 warp block halves LDS traffic but
leaves only four warps on the card, and it measured 58 us against 35 us.
The only way past the ceiling is to split K across CTAs, which needs a
partial-sum workspace and a reduction pass; it is not implemented.

Two things that looked plausible were measured and made no difference, so they
are not in the kernel: an `asm volatile` fence between the prefetch and the
matrix-core block (39.38 us against 39.37 us), and a barrier-free variant that
dequantises straight into the WMMA fragment instead of staging a shared tile.
The latter is 15.7 TFLOP/s rather than 31: a lane's 4-byte code load is
uncoalesced once the shared tile's row-major layout is gone.  Four accumulator
banks instead of two were also measured and are not better (40.1 us against
39.4 us at `m = 32`, measured before the staging rewrite below).

### Correctness of partial tiles

Two staging paths had to be fixed after review, and both are covered by the
harness now.

The API only promises `k % 16 == 0`, but a thread stages a `kCodes`-wide span
(16, 32, 64 or 128 depending on the tile), so a `k` that is not a multiple of
the tile width ends the last tile mid-span.  The kernel used to zero the whole
span in that case, which dropped the valid leading columns with it: at `n = 3`
a `k = 16` bf4 layer returned 0 instead of the full dot product.  Partial spans
now zero-fill past `k` and keep the columns below it.

The quantiser took the base candidate only when its squared error compared
strictly below infinity.  For inputs near `FLT_MAX` every candidate's error
overflows to infinity, so the winning codes were never initialised and the
result was zero.  The base candidate is now taken unconditionally, matching the
Torch reference, and the harness checks a `1e30` block against it.

Hoisting the per-element `w_ok` test out of the staging loop while fixing the
first one also helped the shapes with the widest spans, where the store loop is
the largest share of the work: `m = 16` went from 45 us to 32 us and `m = 32`
from 39 us to 35 us.  The other shapes are unchanged within run-to-run noise.

At large `m` the kernel sits on two hard limits rather than a tuning miss: the
64 KB of shared memory a workgroup may request, and 128 VGPRs per thread at the
three CTAs per CU that the LDS footprint allows.  A 128x128 block at `K = 64`
uses 36864 B, so a fourth CTA would need 147456 B against the 131072 B a CU
has.  Everything that trades warps for fewer LDS bytes loses, and everything
that trades registers for overlap loses:

| variant | m=512 | why it is not in the kernel |
| --- | --- | --- |
| 32x32 warp block (baseline) | 185 us | 512 B of fragments per WMMA, 48 warps/CU |
| 64x32 or 32x64 warp block | 207-208 us | 384 B per WMMA, but only 24 warps/CU |
| 64x128 CTA block | 262 us | 4 CTAs/CU, but 8 warps each |
| 128x256 CTA block | 193 us | 2 CTAs/CU |
| depth-2 register prefetch | 262 us | the second tile of staging registers pushes VGPRs past 128/thread and drops the kernel to 2 CTAs/CU |
| swap the grid axes | 184 us | neutral; the working set is above L2 either way |

Measured separately with the staging or the matrix-core block compiled out,
staging costs 15 us and the matrix core 92 us, but together they cost 184 us:
with a single shared buffer the store at the bottom of each step waits on a
load issued only one step earlier, and HBM latency is longer than one tile of
MMA.  Removing that wait needs either a second shared buffer (which does not
fit) or a second set of staging registers (which does not fit either).

Measured progression on this card (bf16 prefill, 4096x4096 bf8/bf4 weights):

| version | m=32 | m=512 |
| --- | --- | --- |
| first working | 95-104 us | - |
| double-buffered staging, vector stores | 49 us | 481 us |
| K tile tuned per shape, barrier removed | 39 us | 444 us |
| 64-row M block (half the weight re-reads) | 39 us | 296 us |
| 128x128 block, 32x32 warp tile | 39 us | 185 us |
| partial-tile fix, staging test hoisted out of the loop | 35 us | 186 us |

## Limitations

* The checkpoint layout is row-major, as GPT-QModel writes it. The Tenstorrent
  runtime still has to permute codes into 32x32 face order; that is a device
  layout concern and is not part of the dtype emulation.
* `in_features` must be a multiple of 16 (the hardware block size).  Any such
  value works, including one that is not a multiple of the tile width: the last
  tile stages a partial span and zero-fills past `k`.
* `bf2_b` is **not** implemented by these kernels.  The grid is defined in
  `gptqmodel.utils.tt_blockfloat_quant` (a sign bit and one magnitude bit, so
  `{-1, 0, +1} x 2**exponent` per 16-value block), but `ttnn` exposes no
  `bfloat2_b` attribute, so no checkpoint can contain it and there is no oracle
  to validate a kernel against.  A module with `bits == 2` takes the torch path.
* Storage with a `block` wider than 16 also stays on the torch path: the
  kernels bind exactly one exponent per 16 values.  `hip_dispatch` checks the
  block size as well as the device and dtype.
* Accumulation is fp32. On `gfx1200`/`gfx1201`, bf16 activations use the
  `v_wmma_f32_16x16x16_bf16` matrix-core path; bf8/bf4 values are exact in
  bf16, so this changes only the accumulation order. fp16 and fp32 activations
  use the register-tiled fp32 GEMM.
* `m == 1` uses the GEMV kernel; larger `m` uses the tiled GEMM, with the CTA
  block, warp block and K tile chosen by token count as in the table above.
* Small-M prefill is parallelism-limited, not arithmetic-limited.  At `m = 32`
  the call is 512 output tiles, so the card gets 16 warps per CU and reaches
  about 31 TFLOP/s, against 92 TFLOP/s at `m = 512`.  Splitting K across CTAs
  would raise the warp count but needs a partial-sum workspace and a reduction
  pass; it is not implemented.
* The WMMA path is selected at runtime from `gcnArchName`; other ROCm targets
  (CDNA, RDNA3) keep the fp32 GEMM.
