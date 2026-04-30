# Komodo-CANN Ascend C Kernel

This directory contains the repository-owned Ascend C implementation for the
optional Komodo-CANN W4A16 custom operator. It is intentionally separate from
plain Komodo and from the `aclnnWeightQuantBatchMatmulV3` probe.

The current device kernel is a bring-up baseline for GPTQ W4A16:

- Input activations are FP16 `[M, K]`.
- Packed weights use Komodo's CANN INT4 pack, physically `int32 [K, N / 8]`.
- Scales and offsets are FP16 `[groups, N]`.
- Output is FP16 `[M, N]`.
- Supported group sizes are `0`, `32`, `64`, and `128`.

The kernel deliberately does not materialize a full dense FP16 weight matrix in
GM or L2. The current validated baseline uses up to eight logical AIV owners
over disjoint packed output-column ranges. Each owner reuses one packed INT4
word across its eight output lanes while the matching FP16 activation value
stays live, and it writes only the final output columns. This is not yet the
final high-throughput Cube-tiled design; it is the custom-op baseline needed
before replacing the scalar accumulation loop with Cube tile consumption.

The host tiler caps scalar launch ownership at `blockDim <= 8`. Earlier
multi-block experiments exposed sparse/non-contiguous physical AIV block IDs on
the local 910B runtime, so the device kernel maps `GetBlockIdx()` through
`physical_id % tiling.block_dim` and then owns a contiguous packed-column range.
Wider logical ownership such as 32 chunks left unwritten columns on this host.
If a future generated package enters a mixed AIC/AIV task layout, the scalar
fallback first normalizes AIV block IDs by `GetTaskRation()` and returns
immediately on AIC entry. The checked-in package remains AIV-only; this guard is
only preparation for the staged vector/Cube kernel.

Validated raw-op timing on NPU0 for `M=8,K=256,N=256,group_size=32,bias=True`
improved from `63.67 ms` on the initial UB dequant-tile baseline to `10.33 ms`
with the 8-lane packed-word loop, then to `9.22 ms` after hoisting scale/offset
loads to quant-group scope, and to `5.56 ms` after adding a two-row micro-tile
that reuses each dequantized packed word across two activation rows. Branchless
INT4 sign extension then reduced the same raw-op timing to `4.60 ms`, and a
four-row micro-tile reduced it further to `3.66 ms`. The current eight-row
micro-tile reaches `3.42 ms` for the same shape.

For single-row decode, the scalar baseline also has a two-packed-word micro-tile
that reuses each FP16 activation load across sixteen adjacent output channels.
On NPU0 this reduced `M=1,K=256,N=256,group_size=32` from `1.03 ms` to
`0.72 ms`, and `M=1,K=1024,N=1024,group_size=32` from `16.73 ms` to `11.42 ms`.
The two-row tail now has the same two-packed-word reuse. In a rechecked NPU0
median run it reduced `M=2,K=256,N=256,group_size=32` from `1.239 ms` to
`1.191 ms`, and `M=2,K=1024,N=1024,group_size=32` from `18.734 ms` to
`17.749 ms` while keeping `M=1` effectively unchanged.
The four-row path also reuses each FP16 activation load across two adjacent
packed weight words. Compared with the two-row commit, NPU0 medians improved
from `53.206 ms` to `51.678 ms` for `M=7,K=1024,N=1024,group_size=64` and from
`36.833 ms` to `36.688 ms` for `M=5,K=1024,N=1024,group_size=32`; the small
`M=4,K=256,N=256` median was effectively flat (`1.665 ms` to `1.678 ms`) while
the 8-NPU smoke passed on all devices.
The torch bridge now passes a true null optional bias into ACLNN instead of
allocating a synthetic zero-bias tensor and synchronizing the stream to preserve
that temporary. No-bias NPU0 medians improved from `0.827 ms` to `0.802 ms` for
`M=1,K=256,N=256,group_size=32`, from `1.118 ms` to `1.086 ms` for `M=2`, and
from `3.248 ms` to `3.213 ms` for `M=8`; bias-present timings stayed within
noise.
For symmetric GPTQ cases with at least eight rows, the planner marks the
eight-row path as zero-offset and the host tiler decodes that internal flag from
a negative `base_k` attribute while preserving the absolute tile size. This lets
the row-oct loop skip eight GM offset loads per group. NPU0 medians improved
from `3.214 ms` to `2.964 ms` for `M=8,K=256,N=256,group_size=32`, from
`6.353 ms` to `5.833 ms` for `M=16,K=256,N=256,group_size=32`, and from
`50.033 ms` to `45.781 ms` for `M=8,K=1024,N=1024,group_size=32`; M1/M2/M4
nonzero-offset paths stayed within timing noise.
With capped 8-owner AIV packed-column parallelism, the same no-dense baseline
improved by about 85-87% in an 8-NPU raw-op A/B sweep: `M=1,K=256,N=256`
from `0.713 ms` to `0.099 ms`, `M=8,K=256,N=256` from `2.895 ms` to
`0.376 ms`, symmetric `M=16,K=256,N=256` from `5.750 ms` to `0.743 ms`, and
`M=8,K=1024,N=1024` from `46.037 ms` to `5.887 ms`. Max observed error in that
sweep stayed below `0.0005`.
The M1 path now hoists the constant offset contribution out of each quant-group
K loop and applies it from a per-group activation sum. In an 8-NPU raw-op A/B
sweep this reduced `M=1,K=256,N=256,group_size=32` from `0.101 ms` to
`0.093 ms`, `M=1,K=1024,N=1024,group_size=32` from `1.493 ms` to `1.355 ms`,
and `M=1,K=1024,N=1024,group_size=64` from `1.457 ms` to `1.312 ms`.
The large down-proj check `M=1,K=17408,N=5120,group_size=32` improved from
`127.73 ms` to `116.32 ms`; max drift was `0.0625` with mean drift about
`1.0e-4`.
The same offset-hoist pattern is now applied to the two-row path. Against the
M1-hoist baseline, an 8-NPU raw-op A/B sweep improved
`M=2,K=256,N=256,group_size=32` from `0.155 ms` to `0.121 ms`,
`M=2,K=1024,N=1024,group_size=32` from `2.354 ms` to `1.809 ms`, and
`M=2,K=1024,N=1024,group_size=64` from `2.318 ms` to `1.764 ms`. Mixed tails
also improved: `M=3,K=256,N=256` from `0.242 ms` to `0.207 ms` and
`M=6,K=256,N=256` from `0.411 ms` to `0.381 ms`. Max observed drift in this
sweep was `0.00390625`; symmetric zero-offset M2 stayed exact and M1 stayed
flat.

Build from the repo root:

```bash
python scripts/build_komodo_cann_ascendc.py \
  --output /tmp/komodo_cann_w4a16_op
```

Use `--no-build` to generate and overlay the project without invoking CMake.
The helper calls `msopgen` with
`gptqmodel_ext/komodo_cann/op_ir/komodo_cann_w4a16_matmul.json`, overlays the
files in this directory, and then runs the generated `build.sh`.
