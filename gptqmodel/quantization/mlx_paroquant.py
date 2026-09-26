# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Native MLX packing for ParoQuant's symmetric four-bit export weights."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _paroquant_pack_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_paroquant_pack_qweight",
        input_names=["codes"],
        output_names=["packed"],
        source="""
            uint index = thread_position_in_grid.x;
            uint column = index % OUT_PACKS;
            uint row = index / OUT_PACKS;
            uint order[8] = {0, 2, 4, 6, 1, 3, 5, 7};
            uint word = 0;
            for (uint lane = 0; lane < 8; ++lane) {
                uint output = column * 8 + order[lane];
                word |= uint(codes[output * IN_FEATURES + row]) << (lane * 4);
            }
            packed[index] = as_type<int>(word);
        """,
    )


def paroquant_pack_weight_mlx(weight, scales, *, group_size: int):
    """Return ParoQuant's AWQ-layout ``(qweight, qzeros, scales)`` on MLX.

    ``weight`` is the exported, pseudo-quantized transformed weight in
    ``(out_features, in_features)`` order. ``scales`` contains its learned
    symmetric group scales in ``(out_features, groups)`` order.
    """
    import mlx.core as mx

    if weight.ndim != 2 or scales.ndim != 2:
        raise ValueError("weight and scales must be rank-2 arrays")
    out_features, in_features = weight.shape
    if out_features == 0 or in_features == 0 or out_features % 32:
        raise ValueError("weight must be nonempty with output width divisible by 32")
    if group_size == -1:
        group_size = in_features
    if (
        group_size not in (16, 32, 64, 128, in_features)
        or group_size % 2
        or in_features % group_size
    ):
        raise ValueError("invalid ParoQuant group_size for input width")
    groups = in_features // group_size
    if scales.shape != (out_features, groups):
        raise ValueError("scales must have shape (out_features, groups)")
    if weight.dtype not in (mx.float16, mx.bfloat16, mx.float32) or scales.dtype != weight.dtype:
        raise ValueError("weight and scales must share float16, bfloat16, or float32 dtype")
    # AwqTorchLinear.pack multiplies the zero point by the original scale,
    # then casts the stored runtime scale to fp16 when it starts as fp32.
    scale_zeros = scales * 8
    runtime_scales = scales.astype(mx.float16) if scales.dtype == mx.float32 else scales
    view = weight.reshape(out_features, groups, group_size)
    codes = mx.round((view + scale_zeros[:, :, None]) / runtime_scales[:, :, None])
    valid_scales = mx.all(mx.isfinite(scales) & (scales > 0))
    valid_stored_scales = mx.all(mx.isfinite(runtime_scales) & (runtime_scales > 0))
    valid_weight = mx.all(mx.isfinite(weight))
    valid_codes = mx.all(mx.isfinite(codes) & (codes >= 0) & (codes <= 15))
    mx.eval(valid_scales, valid_stored_scales, valid_weight, valid_codes)
    if not bool(valid_scales.item()):
        raise ValueError("scales must be finite and positive")
    if not bool(valid_stored_scales.item()):
        raise ValueError("stored scales must be finite and positive")
    if not bool(valid_weight.item()):
        raise ValueError("weight must be finite")
    if not bool(valid_codes.item()):
        raise ValueError("ParoQuant four-bit codes must be in [0, 15]")

    kernel = _paroquant_pack_kernel()
    out_packs = out_features // 8
    packed = kernel(
        inputs=[codes],
        grid=(in_features * out_packs, 1, 1),
        threadgroup=(min(in_features * out_packs, 256), 1, 1),
        output_shapes=[(in_features, out_packs)],
        output_dtypes=[mx.int32],
        template=[("OUT_PACKS", out_packs), ("IN_FEATURES", in_features)],
    )[0]
    qzeros = mx.full((groups, out_packs), -0x77777778, dtype=mx.int32)
    result_scales = mx.contiguous(runtime_scales.T)
    mx.eval(packed, qzeros, result_scales)
    return packed, qzeros, result_scales
