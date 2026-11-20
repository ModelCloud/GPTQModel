# EXL3 routed decode

`gptqmodel.exllamav3.ext.exllamav3_ext.exl3_moe_coop` runs the gate and up
projections, activation, down projection and weighted sum of already-routed
experts on NVIDIA SM80 or newer. It accepts `3inst`, `mcg`, `mul1`, `q4nx`
and `r4nx` codebooks. The model's router supplies the expert IDs
and coefficients. Integrations call the operator explicitly.

Callers provide these tensors on the input device.

- A contiguous FP16 activation matrix, a contiguous INT64 selected-expert
  matrix and a contiguous FP16 routing-weight matrix.
- INT64 pointer tables for packed trellis, SU, SV and optional bias. Each
  table is a one-dimensional device array with an entry per local expert.
  Gate and up share a bitrate. Down may use another bitrate with the same
  codebook. Pass `q4nx=True` for Q4NX at 2.5, 3.0, or 3.5 bpw, or `r4nx=True`
  for R4NX at 2.5, 3.0, 3.5, or 4.0 bpw, with
  `mcg=False` and `mul1=False`. Keep the packed tensors and tables alive through
  execution and graph replay. Synchronize the publishing stream before consuming them.
- Caller-owned scratch for `S` slots, where `rows * topk <= S <= 256`.
  If the padded gate input, intermediate and down output widths are `H`,
  `I` and `O`, allocate FP16 `had_gate[S,H]`, `had_up[S,H]` and
  `activated[S,I]`. Use two `[S,I]` matrices of the same FP16 or FP32 dtype
  for gate and up, and FP32 `down_out[S,O]`. All three padded widths are
  positive multiples of 128. Q4NX keeps gate and up partial sums in FP32.
  It uses caller-owned FP32 scratch directly and allocates per-call FP32
  scratch when the caller supplies these buffers in FP16.
- Zero-initialized INT32 counters with at least
  `S * (I // 128) + out.shape[0] * (O // 128) + 2 * S + 3` elements.
  The native stages reset completed counters before reuse. Allocate
  independent scratch for overlapping calls on separate streams or graphs.
- A contiguous FP32 output with at least one row per input row, and width at
  most `O`. An optional FP32 shared-expert output can be added, with an
  optional FP16 shared gate vector.

Pass `-1, -1` for a common expert table or a half-open local expert range
for an expert-parallel shard. Zero-weight or nonlocal routes contribute
zero. The activation arguments follow upstream's IDs for SiLU, GELU,
ReLU-squared and SwiGLU with limit.

```mermaid
flowchart LR
    R[Router IDs and weights] --> G[Gate and up with activation]
    P[Packed EXL3 experts] --> G
    G --> D[Down projection and weighted sum]
    W[Caller-owned scratch] --> G
    W --> D
    D --> O[FP32 output]
```

The operator launches on the caller's current CUDA stream. Warm it before
CUDA graph capture to initialize device launch attributes. The caller
retains the same tensor addresses throughout graph replay.

The kernel source is from [ExLlamaV3 1.6.7](https://github.com/turboderp-org/exllamav3/blob/8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0/exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu).

## DeepSeek V4.1 Flash Q4NX

`V41MoE.forward` can use the grouped CUDA path for homogeneous Q4NX experts
on NVIDIA SM80 or newer. The loader builds the existing decoder layers from
the checkpoint and binds the Q4NX projections. The first grouped forward
publishes the packed-weight pointer tables. Device moves, buffer replacement,
and edits to versioned input scales invalidate those tables. Call
`clear_weight_cache()` on a projection after an unversioned scale edit.
Rebuild captured CUDA graphs after changing the tables or their input scale
bounds. Each launch uses its own scratch.

`routed_q4nx` prescales BF16 input with `bf16_prescale` and orders each
token's route slots by expert ID on the device. The Flash operator mode
accepts FP32 routing weights and `flash_input_scale`. It produces BF16
gate and up values before clamping, applies SiLU to the clamped gate, and
weights the activated intermediate before rounding it to BF16 for down.
It rounds each down projection to BF16 and accumulates the routed outputs in
FP32. `V41MoE` adds the shared expert before the final BF16 cast.

```mermaid
flowchart TD
    A["V41MoE.forward"] --> R["V41Router selects expert IDs and FP32 weights"]
    R --> B{"Eligible Q4NX experts on SM80+?"}
    B -->|yes| P["routed_q4nx<br/>prescale rows and order IDs on GPU"]
    P --> G["exl3_moe_coop<br/>Flash clamp, SiLU, weight and BF16 rounding"]
    B -->|no| E["V41Expert forwards<br/>FP32 index_add for routed results"]
    G --> O["Add shared expert, cast to BF16"]
    E --> O
```

The grouped path accepts unpadded, bias-free Q4NX experts with uniform
gate and up widths, bitrates, and a common positive clamp. The down projection
can use a different uniform bitrate. The input scales participate in the
gate/up prescale bound. The clamp, normalized routing weights, and down
input scales bound the activation before the down Hadamard transform.
Calls with mixed or
ineligible experts use the per-expert path. Long batches are split at
`256 // topk` rows to fit the native slots. [Flash checkpoint layout](deepseek-v41-module-inventory.md)
