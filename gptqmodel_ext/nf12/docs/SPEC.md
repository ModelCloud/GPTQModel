# NF12 format and API specification

NF12 is a lossless, fixed-length 12-bit encoding of BF16 matrices, after
"NyoomFloat12" (Liberman et al., 2026). This library's bit and memory layout
is not exactly the same as the paper's.

Changing anything here requires updating every
component and the reference implementation together.

## 1. Logical matrix

- `W`: `N x K` BF16, row-major, the `torch.nn.Linear.weight` layout
  (`N` = out_features, `K` = in_features). The linear op computes
  `y[m, n] = sum_k x[m, k] * W[n, k] (+ bias[n])`.
- Constraints: `N % 16 == 0`, `K % 64 == 0`.

## 2. Value coding

A BF16 value has bits `S | E7..E0 | M6..M0` (high byte `S E7..E1`, low byte
`E0 M6..M0`). Let `e` be its 8-bit biased exponent field.

Each matrix has one **exponent base** `B` (uint8, `0 <= B <= 120`). The
**window** is `[2B, 2B + 15]`. A value is *in-window* iff `2B <= e <= 2B + 15`.
Decode reconstructs bits directly. Zeros, subnormals, Inf and NaN are
in-window or out-of-window by the same rule.

For an in-window value, with offset `o = e - 2B` (`0..15`):

- low byte: stored verbatim (`E0 M6..M0`). Because `2B` is even, `E0 == o & 1`.
- `S` and `o >> 1` (3 bits) go into the shared meta byte.

The decoded high byte is `hi = (S << 7) | (B + (o >> 1))`. `B + 7 <= 127`, so
`B + (o >> 1)` never carries into the sign bit.

The encoder chooses the `B` in `[0, 120]` that minimizes the number
of escaped groups (section 3), with ties broken by the smallest `B`.

## 3. Groups, meta and escapes

A **group** is 8 values with slots `s0..s7` (slot-to-matrix mapping in section 4).
It is encoded as 12 bytes: one meta `uint32` `M` and two low `uint32`s `L0`, `L1`.

- `M` byte `j` (bits `8j..8j+7`, `j = 0..3`) holds the pair `a = s(2j)` and `b = s(2j+1)`:
  - bit 7: `a.S`
  - bit 6: `b.S`
  - bits 5..3: `b.o >> 1`
  - bits 2..0: `a.o >> 1`
- `L0` byte `i` = low byte of slot `i` (`i = 0..3`); `L1` byte `i` = low byte of slot `4 + i`.

**Escape.** A group is escaped iff any slot is out-of-window, or the
in-window encoding would produce `M == 0` (the sentinel). An escaped group
stores `M = 0`, `L0 = escape index`, `L1 = 0`. Entry `escapes[index]` is the
group's 8 original BF16 values in slot order (16 bytes). Escape indices are
assigned in increasing **flat group id** order (section 4.3), so every encoder
produces identical bytes.

**Decoded registers.** Decoding yields four `uint32`s `R0..R3`, with
`Rj = bits(s(2j)) | bits(s(2j+1)) << 16`.

Reference decode, branch-free on the common path. `__byte_perm(x, y, sel)`
numbers bytes 0-3 from `x` and 4-7 from `y`:

```
Bx4 = B * 0x01010101
a   = (M & 0x87878787) + Bx4                                       // byte j = hi(s(2j))
b   = (((M << 1) & 0x80808080) | ((M >> 3) & 0x07070707)) + Bx4    // byte j = hi(s(2j+1))
P0  = __byte_perm(a, b, 0x5140)     // hi(s0), hi(s1), hi(s2), hi(s3)
P1  = __byte_perm(a, b, 0x7362)     // hi(s4) .. hi(s7)
R0  = __byte_perm(L0, P0, 0x5140)
R1  = __byte_perm(L0, P0, 0x7362)
R2  = __byte_perm(L1, P1, 0x5140)
R3  = __byte_perm(L1, P1, 0x7362)
if (M == 0) { R0..R3 = escapes[L0] }
```

`include/nf12/format.cuh` is the canonical device implementation.

## 4. Memory layout ("tiled")

### 4.1 Atom: 16 rows x 16 columns = 32 groups

The groups are laid out so that a decoded group *is* the `A` fragment of
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` for lane `l`. Weights
are the `A` operand ("swap-AB": rows = output features `n`), and activations
are `B` (8 tokens per MMA). For lane `l`, with `g = l >> 2` and `t = l & 3`,
the slots map to (row offset, column offset) within the atom as:

| slot | row   | col      | slot | row   | col      |
|------|-------|----------|------|-------|----------|
| s0   | g     | 2t       | s4   | g     | 2t + 8   |
| s1   | g     | 2t + 1   | s5   | g     | 2t + 9   |
| s2   | g + 8 | 2t       | s6   | g + 8 | 2t + 8   |
| s3   | g + 8 | 2t + 1   | s7   | g + 8 | 2t + 9   |

So `R0 = a0`, `R1 = a1`, `R2 = a2`, `R3 = a3` in PTX fragment order.

### 4.2 Super-atom: 16 rows x 64 columns = 4 atoms = 1536 bytes

Atom `j` (`0..3`) covers columns `16j .. 16j + 15`. Byte layout, three 512-byte
planes, each read by a warp as one coalesced 16-byte load per lane:

| plane | bytes       | lane `l` stores at                        |
|-------|-------------|-------------------------------------------|
| 0     | 0..511      | `16l + 4j`: `M` of atom `j`, `j = 0..3`   |
| 1     | 512..1023   | `512 + 16l + 8j`: `L0, L1` of atom `j`, `j = 0..1` |
| 2     | 1024..1535  | `1024 + 16l + 8(j-2)`: `L0, L1` of atom `j`, `j = 2..3` |

### 4.3 Matrix

Super-atom `(nb, kb)` with `nb in [0, N/16)` and `kb in [0, K/64)` starts at
byte `(nb * (K/64) + kb) * 1536`, so each 16-row block is contiguous along `K`.
Total size: `N * K * 3 / 2` bytes. Required alignment: 16 bytes.

Flat group id: `((nb * (K/64) + kb) * 4 + j) * 32 + l`.

### 4.4 BF16-tiled twin layout ("bf16t")

This is the same tiling stored uncompressed. It exists so that matched-BF16
baseline kernels share NF12's exact accumulation order (section 6). A super-atom
is 2048 bytes: atom `j` is the plane at `512j`, and lane `l` stores
`R0..R3` (16 bytes) at `512j + 16l`. Super-atoms are ordered as in 4.3.

## 5. Stored tensor set (serialization)

For a weight named `W`:

| name                | dtype  | shape          |
|---------------------|--------|----------------|
| `W.nf12_packed`     | uint8  | `[N, K*3/2]` (bytes in the order of 4.3; row view only) |
| `W.nf12_escapes`    | int16  | `[E, 8]` (BF16 bit patterns) |
| `W.nf12_exp_base`   | uint8  | `[1]`          |

`N` and `K` are recovered from the packed shape.

## 6. Numerics contract

1. `decode(encode(W))` equals `W` for every BF16 input
   (including -0, subnormals, Inf, NaN payloads).
2. Linear kernels accumulate in FP32 and write BF16 (round-to-nearest-even).
   For every kernel configuration, NF12 output is identical to the same
   configuration run on the bf16t twin. NF12 adds zero error.
3. Against `torch.matmul` (cuBLAS), outputs agree within normal
   BF16-accumulation-order noise. Tests use the elementwise bound
   `|got - ref| <= atol + rtol * |ref|` with `atol = rtol = 2e-3`.
   A bare `max(rel_err)` is **not** a valid gate: it diverges on near-zero
   outputs, where the absolute error is a fraction of one BF16 ULP. Report
   `max_abs`, `max_abs / ULP(scale)`, `max_rel` restricted to elements at or
   above 1% of `max|ref|`, and `rms(|err|) / rms(|ref|)` instead.
4. Deterministic: the same inputs and configuration give the same bits, so
   split-K reductions must be order-fixed. No atomics on the FP32 partials
   unless the reduction order is fixed.
5. Decode-then-GEMM routes. At large M the fused kernels re-decode every
   weight tile once per token tile. Decoding the weight once and running a
   plain BF16 GEMM is faster there. Both routes split the weight into row
   slices `[n0, n0 + rows)` (`n0`, `rows` multiples of 16; equal slices up to
   the last) and are lossless, so their contract is the same GEMM on the
   exactly decoded weights. A row slice of an NF12 weight is the view
   `{packed + n0 * K * 3/2, escapes, num_escapes, rows, K, exp_base}`: escape
   indices are absolute, so every slice decodes with the full table.
   - `nf12_linear` (C API): from a per-architecture M, each slice is decoded
     to bf16t in the caller's workspace and multiplied by the bf16t large-M
     kernel (bias slice and BF16 output as in the fused path).
     `nf12_linear_bf16t` takes the same slices from the twin, so item 2 holds.
     NF12 is identical to the twin.
   - The torch op `nf12.linear`: where `nf12_linear_decode_gemm_rows(m, n, k,
     device)` is non-zero (from a per-architecture M; on some architectures
     also for router-gate-sized weights, N <= 128, from a smaller M), each
     slice is decoded with `nf12_decode` into a
     caching-allocator buffer and multiplied by cuBLAS (`addmm` with the bias
     slice, else `mm`) into its columns of `y`. The output is identical to
     issuing the same calls on `decode(W)`, and not, in general, to the twin.
   Routing depends only on (m, n, k, architecture), never on weight values,
   and is deterministic, CUDA-graph capturable and host-sync free.

## 7. Targets

- Correctness: SM80+ (`mma.sync` m16n8k16 BF16, `prmt`, `lop3`, `vote` all exist on SM80+).
- Tuned: SM120 (RTX PRO 6000 Blackwell; GDDR7 ~1.79 TB/s, 188 SMs) and SM110
  (Jetson Thor; LPDDR5X ~273 GB/s). SM90 (H200) has a wgmma large-M and
  grouped-MoE family (register-sourced A operands decoded from NF12) and the
  small-M bulk-copy ring. SM100 (B200) runs the tcgen05 family plus a
  deterministic cluster split-K config (partials summed in rank order), with
  heuristics fitted on B200. SM80 / SM86 / SM89 (CMP 170HX, RTX A6000, RTX
  6000 Ada) run the `mma.sync` families with per-board heuristics. Without
  clusters, the small-M kernel splits K inside the CTA (warps summed in part
  order) and across CTAs (FP32 partials in the workspace, summed in rank
  order by a second kernel).

## 8. Public API

### 8.1 C API (`include/nf12/nf12.h`)

The C API takes device pointers, host-side weight descriptors and
`cudaStream_t` without framework dependencies. There is no hidden allocation.
`nf12_encode_analyze` and `nf12_encode` synchronize the stream.
Analysis outputs are host-side. Linear and decode are CUDA-graph capturable.
`nf12_linear_workspace_bytes` includes the slice buffer of the
decode-then-GEMM route (section 6.5, at most 128 MiB of decoded weight plus
the GEMM's own workspace) and, below SM90, the small-M split-K partials.
`nf12_linear_decode_gemm_rows` recommends a slice height for callers with
their own BF16 GEMM (cuBLAS), using slices of at most 256 MiB of BF16.
A return of 0 recommends the fused route. Performance depends on the caller's GEMM.

### 8.2 Python (`import nf12`)

```python
nw = nf12.encode(w)                 # w: [N, K] bf16 CUDA -> NF12Weight
w2 = nf12.decode(nw)                # -> [N, K] bf16, identical to w
y  = nf12.linear(x, nw, bias=None)  # x: [..., K] bf16 -> [..., N] bf16
layer = nf12.NF12Linear.from_linear(torch_linear)
nw.state_dict(prefix) / nf12.NF12Weight.from_state_dict(sd, prefix)

emb = nf12.embedding(ids, nw)       # ids: int32/int64 any shape -> [*ids.shape, K]; section 10
tied = nf12.NF12Embedding.from_weight(nw), nf12.NF12Linear.from_weight(nw)  # one encoded tensor

mw = nf12.encode_moe(w)             # w: [E, N, K] -> NF12MoEWeight; section 9
out = nf12.fused_moe(x, w13, w2, topk_weights, topk_ids)
sorted_ids, offsets = nf12.moe_align(topk_ids, E)  # plus moe_linear / moe_combine
```

The ops are registered as `torch.ops.nf12.*` (torch stable ABI), with fake
implementations, so they work under `torch.compile` and CUDA graphs.

## 9. Mixture-of-experts

### 9.1 Weights

`E` experts, each an `N x K` matrix with the section 1 constraints. Expert `e`
is encoded exactly as a standalone dense NF12 matrix, with its own `exp_base`
and **local** escape indices (0-based within the expert). Slicing experts for
expert parallelism is therefore byte slicing.

| tensor             | dtype  | shape              | meaning |
|--------------------|--------|--------------------|---------|
| `packed`           | uint8  | `[E, N, K*3/2]`    | expert `e`'s tiled bytes |
| `escapes`          | int16  | `[sum_e E_e, 8]`   | all experts' escape entries, expert-major |
| `escape_offsets`   | int64  | `[E + 1]`          | expert `e`'s entries are rows `[off[e], off[e+1])` |
| `exp_base`         | uint8  | `[E]`              | per-expert base |

Serialized names for weight `W` (e.g. `w13`, `w2`): `W.nf12_packed`,
`W.nf12_escapes`, `W.nf12_escape_offsets`, `W.nf12_exp_base`. bf16t twin:
uint8 `[E, N, 2K]`.

### 9.2 Routing

`A = T * topk` assignments, with assignment id `a = t * topk + j` for token `t`
and slot `j`.

- `moe_align(topk_ids [T, topk] int32, E)` produces `sorted [A] int32`, the
  assignment ids stably sorted by expert, and `expert_offsets [E + 1] int64`,
  where expert `e` owns sorted positions `[off[e], off[e+1])`. Ids outside
  `[0, E)` (e.g. `-1` for experts on another expert-parallel rank) are dropped:
  `off[E]` is the count of valid assignments, and `sorted[p]` for `p >= off[E]`
  is unspecified. Deterministic.
- `moe_linear` computes, for every sorted position `p < off[E]` with expert `e`:
  `y[out(p)] = x[in(p)] . W_e^T` (FP32 accumulate, optional `row_scale[out(p)]`
  multiply in FP32, then BF16 round-to-nearest-even), where
  - `in(p) = gather[p] / gather_div` if a gather index is given, else `p`;
  - `out(p) = scatter[p]` if a scatter index is given, else `p`.
  Rows of `y` not written by any valid `p` are untouched.
- `moe_combine(y [A, N], T, topk) -> out [T, N]`:
  `out[t] = sum_j y[t * topk + j]`, summed in FP32 in increasing `j`, then BF16.
- `moe_silu_mul(h [A, 2I]) -> out [A, I]`, `h` rows `[gate | up]` as produced by
  w13: `out[r, i] = bf16(s * u)` with `g = h[r, i]`, `u = h[r, I + i]`,
  `s = g / (1 + expf(-g))`, all FP32 (IEEE `expf`, add, division, multiply in
  that order), one BF16 round-to-nearest-even. `fused_moe` uses it as its
  activation.

A typical layer runs `align`, then `moe_linear(x, gather = sorted,
gather_div = topk)` for `w13` into sorted order. After `moe_silu_mul`,
`moe_linear(h, scatter = sorted, row_scale = topk_weights)` computes `w2`
into assignment order, followed by `combine`.

### 9.3 Contract

- No host synchronization: per-expert counts stay on the device. The host passes
  `max_assignments` (`A`) as the grid-sizing bound. CUDA-graph capturable.
- Numerics as section 6: for each kernel configuration, NF12 output is
  identical to the bf16t twin (C API; for the torch op see 9.4).
  Deterministic. An expert's rows need not match the dense `linear` on the
  same rows (configurations differ).
- Workspace contents need not be initialized (same rule as `linear`).

### 9.4 Decode-then-grouped-GEMM route (torch op)

At large T, decoding the active experts once and running a BF16 grouped GEMM
beats the grouped NF12 kernels. The torch op `nf12.moe_linear` takes this route
when `nf12_moe_linear_decode_gemm_rows(A, E, N, K, device)` is non-zero and the
call gathers its rows and has no `scatter`, no `row_scale` and no `zero_fill`
(the `w13` projection of a layer). The grouped GEMM has no FP32 epilogue,
so a row scale could not be applied before the single rounding. The measured
crossover is H200 from 384 expected rows per expert. No other architecture
uses this route.

1. `nf12_moe_grouped_operands` writes `group_ends[e]`, expert `e`'s end
   position with `expert_offsets` read exactly as `moe_linear` reads them
   (clamped to `[0, A]`, made monotone), and, with a gather index, the gathered
   rows `x[gather[p] / gather_div]` for the valid positions.
2. Per row slice `[n0, n0 + rows)` of every expert (equal slices, at most
   1 GiB of BF16 over all experts), `nf12_moe_decode` decodes the experts that
   own a position into one caching-allocator buffer, and
   `torch._grouped_mm(rows, slice^T, offs=group_ends)` multiplies them.

Rows of `y` at valid positions are identical to issuing the same grouped
GEMMs on the decoded experts. Rows of other positions are unspecified (they
are uninitialized without `zero_fill` anyway). `nf12_moe_decode` is also the
host-sync-free `NF12MoEWeight.decode()`.

## 10. Embedding lookup

An embedding table is an ordinary NF12 matrix `W` with `N = V` rows (vocabulary,
multiple of 16) and `K = H` columns (hidden, multiple of 64), encoded exactly as
in sections 2-4. The same encoded tensor serves `linear` (e.g. a tied `lm_head`)
and lookup.

- `embedding(ids [num_ids] int32 or int64, W) -> out [num_ids, H] bf16`:
  `out[i] = W[ids[i]]`. Ids outside `[0, V)` produce a row of `+0.0`.
  This serves vocab-parallel sharding, where each rank looks up only its
  own range.
- No host synchronization, CUDA-graph capturable, deterministic.
