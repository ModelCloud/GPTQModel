# NF12: NyoomFloat12 12-bit BF16 weights in GPT-QModel

This directory vendors the NF12 CUDA codec and inference kernels from
[dphnAI/nf12](https://github.com/dphnAI/nf12) so GPT-QModel can quantize
weights to NF12 and run them without a separate Python package.

GPT-QModel exposes the format as `METHOD.NF12` / `FORMAT.NF12`
and the backend as `BACKEND.NF12` (all serialize as `"nf12"`). The tiled
layout implemented by the upstream `nf12` library is intentionally not
byte-compatible with the layout in the paper. See `UPSTREAM_README.md` and
the format specification below.

## Credits

NF12 (NyoomFloat12) was created by **Together AI** and introduced in
*"NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit Weight
Compression"* by Liberman, Fang, Zhang, Dao, and Fu (Together AI, 2026).

The reference implementation and all CUDA kernels in this directory were
written by **AlpinDale** (Alpin) at [dphnAI/nf12](https://github.com/dphnAI/nf12).
They are licensed under the Apache License 2.0; a copy is in `LICENSE`.

GPT-QModel (ModelCloud.ai) adapted the integration, the pure-Torch reference
codec, and the `nf12` quantizer/inference module around the unchanged
kernel math.

If you use NF12, please cite both the paper and the upstream implementation:

```bibtex
@article{liberman2026nyoomfloat12,
  title   = {NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit Weight Compression},
  author  = {Liberman, and Fang, and Zhang, and Dao, and Fu},
  year    = {2026},
  note    = {Together AI}
}
```

## Format (NF12 tiled layout)

NF12 is a lossless, fixed-length encoding of BF16 matrices. Every matrix
chooses one 8-bit biased exponent base `B` (0..120); a value is stored in a
12-byte group of 8 BF16 values when its exponent falls inside
`[2B, 2B + 15]`. Values outside that window live in a shared `int16 [E, 8]`
escape table. `decode(encode(W))` is bit-identical to `W` for every BF16
input, including `-0`, subnormals, Inf, and NaN payloads.

`nf12_decode` writes BF16. `nf12_decode_fp32` writes the same elements as
FP32, widening each register pair with a bit shift during the store, so a
BF16 matrix is never materialized for FP32 consumers. The two outputs agree
bit for bit, and the fused store moves 5.5 bytes per element instead of 9.5.

The tiled memory layout packs 16x64 super-atoms so a decoded group is exactly
the `A` fragment of `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
Constraints are `N % 16 == 0` and `K % 64 == 0`.

Serialized tensors for a weight `W`:

| tensor | dtype | shape |
| --- | --- | --- |
| `W.nf12_packed` | uint8 | `[N, K * 3 / 2]` (the packed NF12 weight, under the upstream `nf12` name) |
| `W.nf12_escapes` | int16 | `[E, 8]` |
| `W.nf12_exp_base` | uint8 | `[1]` (stored as a scalar buffer) |

`include/nf12/format.cuh` is the canonical device implementation of the bit
layout; `docs/SPEC.md` in the upstream repository is the normative
specification.

## Local modifications

The CUDA sources track [dphnAI/nf12 commit `606f882`](https://github.com/dphnAI/nf12/commit/606f8823e182f65d6f2ead368aa9e5ba965c057c),
including the volatile `movmatrix` deadlock fix, SM103 build target, and
Blackwell GEMV/direct/asynchronous small-M kernels. `reference.py` remains
at `cf66cbb`; the packed format is unchanged. `docs/SPEC.md` includes the
upstream static-weight readiness contract and GPT-QModel's elementwise error
gate. Original attribution headers are retained.

Local CUDA adaptations preserve staged-B1 projection, four-copy embedding,
and the SM80 decode/prefill tuning. Staged-B1 uses one GEMV launch per row
when ordinary B1 selects GEMV, preserving the same arithmetic. The separate
`gptqmodel_ext/nf12_hip` backend is not replaced by the upstream HIP build/ABI
refactor. See [CUDA dispatch and weight readiness](../../docs/nf12-cuda-dispatch.md).
The integration also includes:

* `gptqmodel/utils/nf12_codec.py` is an independent pure-Torch port of the
  upstream reference codec. It is used for quantization on every device and
  as the inference oracle/fallback.
* `gptqmodel_nf12` lazily builds these sources with CMake
  (`NF12_BUILD_TORCH=ON`) and exposes linear, embedding, MoE, and packed FP32
  projections without requiring a separate upstream `nf12` installation.
  `gptqmodel/utils/nf12.py` delegates legacy GPTQModel call sites to it.
* `gptqmodel/nn_modules/qlinear/nf12.py` stores only the packed NF12
  tensors. Its fallback decodes into bounded scratch for the current forward
  and never keeps a dense weight cache.
* The `nf12` quantizer casts non-BF16 weights to BF16 with
  round-to-nearest-even before encoding. That cast is the only lossy step;
  the encode itself is lossless.

## Building

There are no precompiled binaries in this repository or in the wheel. Every
install builds the extension from the sources here with CMake, targeting the
visible GPU architecture (SM80 or newer, CUDA toolkit 13.0+). Builds are on
demand and cached per source digest under
`XDG_CACHE_HOME/gptqmodel/nf12`.

The build produces two objects:

| artifact | contents |
| --- | --- |
| `libnf12.so` | the C API (`nf12_encode`, `nf12_decode`, `nf12_linear`, ...) |
| `libnf12_torch.so` | the `torch.ops.nf12` bindings; links `libnf12.so` |

The `nf12` op namespace and shared-library names match upstream's. The
`gptqmodel_nf12` package owns the source-keyed build cache.
