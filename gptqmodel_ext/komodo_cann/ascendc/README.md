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
GM or L2. The current validated baseline uses one writing AI Core and reuses
each packed INT4 word across its eight output lanes while the matching FP16
activation value stays live. It writes only the final output. This is not yet
the final high-throughput Cube-tiled design; it is the custom-op baseline needed
before reintroducing multi-core tile ownership and replacing the scalar
accumulation loop with Cube tile consumption.

The host tiler intentionally sets `blockDim=1` for this scalar baseline. Earlier
multi-block launch experiments exposed non-contiguous AIV block IDs on the local
910B runtime, so multi-core ownership stays disabled until the kernel has a real
partitioning scheme that does not assume block 0 is present.

Validated raw-op timing on NPU0 for `M=8,K=256,N=256,group_size=32,bias=True`
improved from `63.67 ms` on the initial UB dequant-tile baseline to `10.33 ms`
with the 8-lane packed-word loop, then to `9.22 ms` after hoisting scale/offset
loads to quant-group scope, and to `5.56 ms` after adding a two-row micro-tile
that reuses each dequantized packed word across two activation rows. Branchless
INT4 sign extension then reduced the same raw-op timing to `4.60 ms`.

Build from the repo root:

```bash
python scripts/build_komodo_cann_ascendc.py \
  --output /tmp/komodo_cann_w4a16_op
```

Use `--no-build` to generate and overlay the project without invoking CMake.
The helper calls `msopgen` with
`gptqmodel_ext/komodo_cann/op_ir/komodo_cann_w4a16_matmul.json`, overlays the
files in this directory, and then runs the generated `build.sh`.
