<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Automatic NF12 source storage

`BaseQuantizeConfig.auto_dtype_cast` controls how dense source matrices are
represented while calibration forwards run. The default, `AUTO_NF12`, stores a
supported BF16 matrix as NF12 and restores its exact BF16 values before the
quantizer consumes it.

```python
from gptqmodel import AutoDtypeCast, QuantizeConfig

config = QuantizeConfig(auto_dtype_cast=AutoDtypeCast.AUTO_NF12)
```

`AutoDtypeCast.OFF` keeps the existing dense source representation.

## Decision flow

```mermaid
flowchart TD
    A[Forward module requested] --> B{AUTO_NF12?}
    B -- No --> D[Dense source]
    B -- Yes --> C{Native floatx path validated?}
    C -- Yes --> D
    C -- No --> E{Linear or embedding with NF12 alignment?}
    E -- No --> D
    E -- Yes --> F{Source dtype}
    F -- BF16 --> G[Encode NF12]
    F -- FP16 exact BF16 round trip --> G
    F -- FP16 not exact --> D
    F -- Decoded FP8 FP4 MXFP4 MXFP8 --> G
    G --> H[Run calibration forward from packed storage]
    H --> I[Decode exact source dtype before quantization]
    I --> J[Release packed source buffers]
    J --> K[GPTQ AWQ or RTN packs the requested output format]
```

FP16 does not have a subset relationship with BF16. An FP16 matrix is therefore
eligible only when every FP16 bit pattern survives FP16 to BF16 to FP16. Signed
zero is included in that comparison. A non-exact FP16 matrix stays dense.

Native FP8, FP4, MXFP4, and MXFP8 execution keeps its validated source path.
When that path is unavailable, the existing floatx decoder materializes a dense
BF16 view and `AUTO_NF12` stores that view in NF12 for calibration.

## Quantization handoff

The processor creates its quantization task before the source is packed. The
task retains shape and operation metadata, while its dense source storage is
released. Immediately before that task runs, the exact original dtype and bit
patterns are reconstructed from NF12 and rebound to the task. The replacement
is installed first, then the now-unused packed buffers are released.

This handoff prevents a packed calibration forward from changing the input
matrix seen by GPTQ, AWQ, or RTN. It also prevents the original dense weight
from remaining live beside the packed representation.

## Module coverage

NF12 supports aligned matrices with `N % 16 == 0` and `K % 64 == 0`.

- `nn.Linear`, including GPT-QModel's calibration `HookedLinear`.
- `nn.Embedding` when the embedding quantization lifecycle selects it.
- Norm vectors and biases remain dense because NF12's matrix format does not
  represent them.

The CUDA path supports native linear and embedding operators. The RDNA4 HIP
path supports native linear and embedding operators. Embedding modules persist
the row-major lookup layout documented in
[NF12 row-major embedding storage](nf12-row-major-embedding.md); matrix modules
keep the canonical 16-row super-atom layout. Unsupported devices or shapes stay
dense.

### Hook and replay contract

`HookedLinear` and `AutoNF12HookedLinear` share
`prepare_hooked_linear_input`. The helper applies the selected W4A replay and
online Hadamard transforms exactly once. The packed wrapper then calls
`NF12Linear._forward_impl(..., apply_rotation=False)` and gives its calibration
hook the transformed operand that the GEMM consumes.

```mermaid
flowchart LR
    A[HookedLinear input] --> B[prepare_hooked_linear_input]
    B --> C[W4A replay and online Hadamard]
    C --> D[AutoNF12HookedLinear packed GEMM]
    D --> E[forward_hook sees transformed operand]
    F[restore_source_module] --> G[HookedLinear]
    G --> H[Copy hooks and replay metadata]
```

Restoring a packed hooked source therefore preserves its class, registered
hooks, online-rotation settings, and W4A replay state. Source, packed, and
restored modules share the same hook registries, so a PyTorch `RemovableHandle`
removes the hook from every replacement. Subsequent quantization replay
executes the same forward behavior as the original source.

## Previous behavior

| Mechanism | Previous behavior | New behavior |
| --- | --- | --- |
| Dense BF16 source forward | Every quantizable matrix stayed dense. | Eligible matrices use NF12 during calibration. |
| Quantizer input | The dense module was already present. | Exact source dtype and bits are restored before `process()`. |
| Native floatx source | Native or decoded path selected by `AutoModuleDecoderConfig`. | Native execution remains; decoded BF16 is eligible for NF12. |
| FP16 source | Stayed dense. | Stays dense unless the BF16 round trip is bit exact. |

## Failure and ambiguity path

An unsupported device, unaligned shape, non-exact FP16 matrix, vector, or
unsupported module remains dense. Packing or decoding errors are not treated as
successful compression. Native floatx selection takes precedence over NF12
when its capability check succeeds.

## Measured validation

The real model was `unsloth/Llama-3.2-1B-Instruct` loaded as BF16. GPTQ
calibration cast 112 projection matrices in every run. Their dense source
occupied 1,856.0 MiB and their NF12 representation occupied 1,394.532 MiB.

| Device | Runtime | AUTO_NF12 peak | OFF peak | Peak change |
| --- | --- | ---: | ---: | ---: |
| RTX 4090, SM 8.9 | Torch 2.16.0.dev20261102+cu130 | 1,229.364 MiB | 1,567.786 MiB | -21.6% |
| RTX 5090, SM 12.0 | Torch 2.16.0.dev20261102+cu130 | 1,229.364 MiB | 1,460.828 MiB | -15.8% |
| R9700, gfx1201 | Torch 2.16.0.dev20261025+rocm10.1, HIP 7.16.26385 | 1,228.664 MiB | 1,204.624 MiB | +2.0% |

The R9700 keeps the 24.9% smaller packed weights, but its calibration workspace
dominates peak allocation. A 4090 GPTQ A/B save produced 482 state tensors on
both policies; all 482 were bit-identical, including packed weights and scales.

Full dense-model inference packed 113 linear matrices. Dense allocation was
2,360.311 MiB and packed allocation was 2,275.021 MiB. The next-token IDs
matched dense execution on all three devices:

| Device | Max logits difference | Mean absolute difference |
| --- | ---: | ---: |
| RTX 4090 | 0.218750 | 0.0234022 |
| RTX 5090 | 0.2734375 | 0.0255472 |
| R9700 | 0.187500 | 0.0176995 |

The source weights are restored exactly. The logits difference comes from the
NF12 kernel's floating-point reduction order; it is not a lossy weight decode.

### Lossless storage and projection arithmetic

NF12 storage and decoding are bit exact for BF16. That property does not make
every NF12 matrix multiplication bitwise identical to a dense BF16 GEMM:
the kernels can associate the same FP32 partial sums in a different order.

On RTX 4090, all 113 aligned projection weights restored bit exactly across
1,235,746,816 elements. Representative projection comparisons were:

| Projection | Max absolute difference | Mean absolute difference | Bit exact |
| --- | ---: | ---: | --- |
| `model.layers.0.self_attn.q_proj` | 0.0009765625 | 1.7003549146465957e-07 | No |
| `model.layers.0.mlp.down_proj` | 0.03125 | 1.4519027899950743e-05 | No |
| `lm_head` | 0.0 | 0.0 | Yes |

The full model retained the same next-token IDs, with maximum logits difference
0.21875 and mean absolute difference 0.023402199131775. This is numerically
close execution of an exactly restored source, not bitwise-identical GEMM
arithmetic.

The Llama embedding table decoded bit exactly, and 512 sampled lookup rows
matched dense BF16 exactly. Its row-major packed representation occupied
394,365,297 bytes instead of 525,336,576 bytes, a 24.93% reduction.

### End-to-end quantization comparison

A separate RTX 4090 comparison used GPTQ 4-bit, group size 128, with
`AUTO_NF12` and `OFF`. Both policies used the same BF16 source model and
4,096 WikiText calibration tokens from 8 windows of 512 tokens. Model load and
checkpoint save were excluded from the quantization timer. Peak allocated
memory was reset immediately before quantization.

| Metric | `AUTO_NF12` | `OFF` |
| --- | ---: | ---: |
| Cast projection modules | 112 | 0 |
| Peak allocated memory | 1,458.342 MiB | 1,458.342 MiB |
| Warm median quantization time | 5.874719 s | 6.126948 s |

All six final checkpoints, three per policy, contained 482 state tensors, and
every corresponding tensor was bit identical. On a held-out 4,096-token
WikiText split, both quantized policies scored NLL
3.1000067591667175 and perplexity 22.198101321602206.

After warming the kernels, `AUTO_NF12` was 4.12% faster at the median. The
checkpoint and held-out score remained exact. End-to-end peak allocation was
equal because the GPTQ Hessian and inverse/damping workspace, rather than
resident source weights, determined the peak.

### Memory and timing attribution

The source representation and the quantization workspace peak at different
times:

```mermaid
flowchart TD
    A[Layer subset selected] --> B{AUTO_NF12?}
    B -- No --> C[Dense BF16 subset]
    B -- Yes --> D[Native BF16 to NF12 encode]
    D --> E[Calibration forward from packed source]
    C --> F[Calibration forward from dense source]
    E --> G[Restore exact BF16 for GPTQ]
    F --> G
    G --> H[Release packed source buffers]
    H --> I[GPTQ Hessian inverse and damping workspace]
    I --> J[End-to-end peak]
```

For the 4K calibration run, resident allocation after module preparation
peaked at 626.139 MiB with `AUTO_NF12` and 771.844 MiB with `OFF`, an 18.88%
reduction. The calibration-forward phase peak was 912.420 MiB and
920.391 MiB respectively because the cached 4K activation inputs dominated
that phase.

The full quantization peak was 1,458.342 MiB for both policies. Allocation
history attributed the peak to GPTQ's 512 MiB inverse factor, two 256 MiB
damping/inverse temporaries, 64 MiB permutation storage, and additional
quantization and activation buffers. With one 512-token block, where the
activation cache is small, the calibration-forward peak was 444.420 MiB with
`AUTO_NF12` and 681.783 MiB with `OFF`.

The packed representation therefore reduces calibration resident memory, but
cannot reduce the end-to-end peak while GPTQ's dense Hessian workspace remains
unchanged.

### Native encode path

The first warm implementation copied each GPU weight to CPU, copied it back to
the GPU for native encoding, then returned the result to CPU and copied the
packed buffers to the GPU again. It also allocated full zero NF12 buffers
before immediately overwriting them. Those round trips made 112 encodes take
about 1.8 seconds and made `AUTO_NF12` slower end to end.

The encoder now consumes the accelerator-resident BF16 source directly and
installs the native result on that device. Warm encoding took 0.118522 seconds
for all 112 modules, or 0.079967 seconds after excluding the first seven shape
specializations. Median warm encoding was 0.478 ms per module.
