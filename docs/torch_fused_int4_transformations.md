```
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
```

# Torch Fused INT4 Transformations

This note explains what `TorchFusedQuantLinear.transform_xpu` and `transform_cpu`
do to GPTQ-format tensors before calling the fused `torch.ops.aten` kernels.
The goal is to document the exact tensor shapes, the axis permutations, and the
bit packing order expected by `aten._weight_int4pack_mm_*` so you do not need to
reverse engineer the loops in `gptqmodel/nn_modules/qlinear/torch_fused.py:175-219`.

## Terminology and starting layout

Let:

* `I` – number of input features.
* `O` – number of output features.
* `B` – quantization bits (always 4 here).
* `W` – number of bits stored per lane in `pack_dtype` (`W = 32` by default).
* `pack_factor = W / B` – how many quantized values share one lane (8 when `B=4`).
* `group_size` – number of input channels that share one `(scale, zero)` pair.
* `G = ceil(I / group_size)` – number of groups (and rows in `scales`/`qzeros`).

Immediately after loading a GPTQ v2 checkpoint:

```
qweight : [I / pack_factor, O]    dtype = pack_dtype (int32)
qzeros  : [G, O / pack_factor]    dtype = pack_dtype (int32)
scales  : [G, O]                  dtype = fp16
g_idx   : [I]                     dtype = int32   (maps input channel -> group id)
```

Each entry of `qweight`/`qzeros` is a 32-bit lane that packs `pack_factor`
4-bit nibbles. Conceptually, a single column of `qweight` (one output channel)
looks like this before unpacking:

```
raw lane bits (int32) → [in_{k+7}] [in_{k+6}] … [in_{k+1}] [in_{k}]
bit positions         → 31..28    27..24          7..4      3..0
```

## `transform_xpu(dtype)`

The XPU path needs tensors that match
`aten._weight_int4pack_mm_with_scales_and_zeros`. The routine performs five
steps:

1. **Scales cast** – `self.scales = self.scales.clone().to(dtype)`. No layout changes.
2. **Unpack `qzeros`** – expand each 32-bit lane into `pack_factor` nibbles, mask
   with `0xF`, then reshape to `[G, O]`.

   ```
   Before unpack (per group g):
       qzeros[g] = [ lane_0, lane_1, … ]  (each lane holds 8 outputs)
   After unpack:
       zeros[g]  = [ z_{0}, z_{1}, …, z_{O-1} ]

       lane layout
       ┌──────────── 32 bits ────────────┐
       | z_{b+7} | … | z_{b+1} | z_{b} |
       └────────────────────────────────┘   ← reshaped into consecutive columns
   ```

3. **Unpack and reorder `qweight`** – identical nibble extraction produces a
   tensor shaped `[I, O]`. It is then re-indexed with `ret_idx` so that input
   rows follow the `g_idx` schedule used during quantization, and finally
   transposed to `[O, I]`. At this point every row corresponds to one output
   channel and every column corresponds to an *unpacked* input channel.

   ```
   weight_full (after transpose):
         input columns →
       ┌───────────────────────────────────────────┐
   out0│ w00 w01 w02 w03 w04 w05 w06 w07 … w0(I-1) │
   out1│ w10 w11 w12 w13 w14 w15 w16 w17 … w1(I-1) │
       │  ⋮                                        │
   ```

4. **Pack rows into XPU layout** – the double `for` loop rebuilds `int32`
   lanes, but now the rows are `O` (output channels) instead of packed input
   clusters. The resulting tensor has shape `[O, I / pack_factor]`.

   ```
   packed[row=j, col=k] stores inputs (8 values) = 
       weight_full[j, 8k + i]  for i = 0..7

   31..28   27..24   23..20   19..16   15..12   11..8   7..4    3..0
   [in+7]   [in+6]   [in+5]   [in+4]   [in+3]   [in+2]  [in+1]  [in+0]
   ```

5. **Finalize buffers** – `self.qweight = packed.contiguous()` (int32) and
   `self.qzeros = zeros.contiguous()` (float, `[G, O]`). These, together with
   `self.scales`, match the signature of
   `aten._weight_int4pack_mm_with_scales_and_zeros(x, qweight, group_size, scales, qzeros)`.

For XPU execution, `_fused_op_forward` also permutes activations before the
matmul:

```
x = x[:, ret_idx]
```

This applies the inverse of the group-wise reordering performed in step 3,
ensuring that column `i` of `qweight` always multiplies the same logical input
channel the calibration used.

### Visual summary (XPU)

```
             ┌─────────────┐      unpack+permute      ┌─────────────┐
raw qweight →│ I/8  ×  O   │ ───────────────────────→ │   O × I      │
             └─────────────┘                          └─────────────┘
                                                   pack rows ↓
                                                   ┌─────────────┐
                                                   │ O × (I/8)   │  int32 lanes
                                                   └─────────────┘

raw qzeros  → [G × O/8] lanes ──unpack──► zeros [G × O]
scales      → [G × O] (cast to `dtype`)
```

## `transform_cpu(dtype)`

The CPU path shares the unpack/reorder logic but delegates the final packing to
PyTorch’s helper so the layout matches
`aten._weight_int4pack_mm_for_cpu`. Steps:

1. **Scales cast** – identical to the XPU path.
2. **Unpack + reorder `qweight`** – same as step 3 above, yielding
   `weight_full = [O, I]` with 4-bit integers.
3. **Convert to int4pack** – `torch.ops.aten._convert_weight_to_int4pack_for_cpu`
   repacks that matrix into `torch.uint8` tiles of shape `[O, I * B / 8]`
   (i.e., `I/2` columns when `B=4`). Each byte stores two adjacent inputs.

   ```
   byte layout (per output row j):
   bits 7..4 → weight_full[j, 2k+1]
   bits 3..0 → weight_full[j, 2k]
   ```

   The helper currently requires both `O` and `I` to be multiples of 16; the op
   raises `_convert_weight_to_int4pack_cpu : expect N to be dividable by 16`
   otherwise.

4. **Merge scales and zeros** – The fused CPU kernel expects scale and zero
   offsets in a single tensor, so `pack_scales_and_zeros` stacks them along the
   last dimension:

   ```
   scales_and_zeros[g, o] = [ scale[g, o], zero[g, o] ]    shape = [G, O, 2]

   group g
     ┌──────── out dimension ────────┐
     │ [ s, z ]  [ s, z ]  …  [ s, z ] │
     └─────────────────────────────────┘
   ```

   The current GPTQ fused path only uses symmetric int4, so `self.qzeros` is
   zeroed before packing (`zero[g, o] = 0`). Non-symmetric per-group offsets
   would require extending this block.

5. **Buffers used at runtime** – `self.qweight` is now the `uint8`
   int4pack tensor, `self.scales_and_zeros` stores the merged metadata, and
   `_fused_op_forward` calls
   `aten._weight_int4pack_mm_for_cpu(x, qweight_uint8, group_size, scales_and_zeros)`.

### Visual summary (CPU)

```
weight_full (O × I, ints) ──_convert_weight_to_int4pack_for_cpu──►
┌──────────────┐                                             ┌──────────────┐
│  O × I       │                                             │ O × (I/2)    │ uint8
└──────────────┘                                             └──────────────┘
     ↑                                                         ↑
     └───────── unpack & transpose from raw qweight ───────────┘

scales (G × O, dtype `dtype`)
qzeros (G × O, zeroed) ──► scales_and_zeros (G × O × 2)
```

## Activation permutation and fused matmul

Both device paths rely on the same activation permutation:

1. `ret_idx` is built once from `g_idx` so that unpacked rows can be restored to
   the calibration order.
2. Before calling any fused matmul, `_fused_op_forward` applies `x = x[:, ret_idx]`.
3. The matmul then multiplies `x` with the packed `qweight`:

   * XPU: `aten._weight_int4pack_mm_with_scales_and_zeros`
     consumes `qweight[int32][O, I/8]`, `scales[G, O]`, and `qzeros[G, O]`.
   * CPU: `aten._weight_int4pack_mm_for_cpu`
     consumes `qweight[uint8][O, I/2]` and `scales_and_zeros[G, O, 2]`.

Because the same `ret_idx` is used for both the unpacked weight (during packing)
and the activation tensor (during inference), every nibble in the packed matrix
aligns with the correct logical input column.

## Comparing XPU vs CPU transformations

Although both device paths share the same unpack → reorder → transpose steps,
they diverge in how the packed tensors are laid out and what the fused matmul
expects afterward. The table below highlights the key differences for quick
debugging.

| Aspect                     | XPU (`transform_xpu`)                                          | CPU (`transform_cpu`)                                             |
|----------------------------|---------------------------------------------------------------|-------------------------------------------------------------------|
| Packed `qweight` shape     | `[O, I / 8]`, dtype `int32`                                    | `[O, I / 2]`, dtype `uint8`                                       |
| Bits per storage lane      | 32-bit lane packs 8 inputs; nibble order `[in+7 … in+0]`       | 8-bit lane packs 2 inputs; high nibble = odd, low nibble = even   |
| Packing direction          | Manual double-loop packs along **columns** of `weight_full`    | `_convert_weight_to_int4pack_for_cpu` packs along **columns** into bytes |
| Per-group zeros            | Unpacked to full `[G, O]` tensor and passed separately         | Forced to zero and merged with scales via `pack_scales_and_zeros` |
| Scale format               | One tensor per group (`scales[G, O]`)                          | Concatenated `[..., 0] = scale`, `[..., 1] = zero` (`float`)      |
| Fused kernel call          | `_weight_int4pack_mm_with_scales_and_zeros(x, qW, gsz, s, z)`  | `_weight_int4pack_mm_for_cpu(x, qW, gsz, scales_and_zeros)`       |
| Alignment requirements     | Determined by manual pack loop (only needs `I % 8 == 0`)       | Kernel enforces `I % 16 == 0` and `O % 16 == 0`                   |
| Activation permutation     | `x = x[:, ret_idx]` prior to matmul (same code path)           | Same permutation reuse                                            |

Visually, you can think of the difference as *row-major lane packing* (XPU)
versus *byte-tiling* (CPU):

```
XPU:  | int32 lane | = [w7][w6][w5][w4][w3][w2][w1][w0]
CPU:  | uint8 lane | = [w1][w0]
```

Both forms originate from the same `[O, I]` intermediate; the divergence is only
in the final storage type, accompanying metadata, and fused operator ABI.

## AWQ compatibility (`torch_fused_awq.py`)

`TorchFusedAwqQuantLinear` (`gptqmodel/nn_modules/qlinear/torch_fused_awq.py`)
reuses the CPU fused kernel while accepting checkpoints emitted by the AWQ
tooling. The module always expects `qweight` to be stored in the AWQ layout
`[in_features, out_features / pack_factor]`, meaning each row corresponds to a
single logical input channel. `transform_cpu_awq` performs a fixed shim before
the standard CPU packing runs:

1. **Unpack AWQ rows** – `unpack_awq` expands each column lane into eight
   outputs, yielding `iweight[int8][I, O]` and `izeros[int8][G, O]`. Both
   tensors are then permuted with `reverse_awq_order` (the inverse of
   `quantization.awq.utils.packing_utils.AWQ_ORDER`) so the columns match the
   logical transformer layout expected by GPTQ.
2. **Normalize zero codes** – AWQ stores integer zero points per output channel.
   `transform_cpu_awq` converts them into floating offsets compatible with the
   fused kernel using
   `zeros_fp16 = (2^{bits-1} - izeros) * scales_fp32`, keeping the result in
   `float16` so the metadata matches the original AWQ calibration statistics.
3. **Repack into GPTQ lanes** – The unpacked `iweight` matrix is reshaped to
   `[I / pack_factor, pack_factor, O]` and re-packed along the `pack_factor`
   dimension so each row once again represents eight inputs inside a 32-bit
   lane. After this step `self.qweight` is indistinguishable from a GPTQ v2
   tensor, which means the regular `transform_cpu` logic can run unchanged.
4. **Delegate to the base CPU transform** – Calling `super().transform_cpu`
   converts the temporary GPTQ-formatted `qweight` into the `[O, I/2]` `uint8`
   int4pack layout and produces `scales_and_zeros` from the (temporarily zeroed)
   metadata.
5. **Restore AWQ metadata** – Immediately afterward, the AWQ shim reinstates
   the real `float16` scales and the converted zero offsets, then rebuilds
   `scales_and_zeros = pack_scales_and_zeros(scales, zeros_fp16)`. This ensures
   `_weight_int4pack_mm_for_cpu` receives the same affine parameters the AWQ
   calibration solved for.

Because the shim runs entirely on the CPU path, `TorchFusedAwqQuantLinear`
currently raises `NotImplementedError` when asked to run the fused transform on
`xpu` devices. If the module has not been transformed yet (or fused ops are
unavailable), inference falls back to the dense AWQ matmul computed by
`awq_weight_dequantize`, which simply dequantizes the cached AWQ tensors on the fly.

## Quick reference

| Stage                          | Shape / dtype (int4)                                      | Notes                                          |
|--------------------------------|-----------------------------------------------------------|------------------------------------------------|
| Raw `qweight`                  | `[I / 8, O]`, `int32`                                     | 8 nibbles per lane                             |
| After unpack + transpose       | `[O, I]`, `int8` (values in `[0, 15]`)                     | Used by both device paths                      |
| Packed XPU `qweight`           | `[O, I / 8]`, `int32`                                     | Bits `[3:0]` hold the lowest-numbered channel  |
| Packed CPU `qweight`           | `[O, I / 2]`, `uint8`                                     | High nibble = odd input, low nibble = even     |
| `qzeros` (post-XPU transform)  | `[G, O]`, matches `scales`                                | Passed separately to the XPU fused op          |
| `scales_and_zeros` (CPU only)  | `[G, O, 2]`, float                                        | `[..., 0] = scale`, `[..., 1] = zero`          |
| Raw AWQ `qweight`              | `[I, O / 8]`, `int32`                                     | Rows are single inputs packed across outputs   |
| Unpacked AWQ weights/zeros     | `iweight[I, O]`, `izeros[G, O]`, `int8`                    | Produced by `unpack_awq` + `reverse_awq_order` |
| AWQ zero offsets (final)       | `[G, O]`, `float16`                                       | `(2^{bits-1} - izeros) * scales`; merged via `pack_scales_and_zeros` |

These details mirror the expectations of the Intel XPU and CPU fused matmul
kernels, and the ASCII layouts above describe how rows/columns line up inside
every packed tensor before the fused matmul executes.
