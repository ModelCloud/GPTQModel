# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Format codebooks derived from bitsandbytes (MIT; Copyright (c) Facebook, Inc. and affiliates).

"""Native MLX weight quantization for bitsandbytes checkpoint formats."""

from functools import lru_cache

import mlx.core as mx
import numpy as np

_NF4 = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
)
_FP4 = (
    0.0, 0.0052083334885537624, 0.6666666865348816, 1.0,
    0.3333333432674408, 0.5, 0.1666666716337204, 0.25,
    0.0, -0.0052083334885537624, -0.6666666865348816, -1.0,
    -0.3333333432674408, -0.5, -0.1666666716337204, -0.25,
)
_BLOCK_SIZES = {32, 64, 128, 256, 512, 1024, 2048, 4096}
# Exact float32 table bits avoid a Torch dependency and linspace rounding drift.
_DYNAMIC_CODE_HEX = (
    "33337ebf9a997abf000077bf666673bfcdcc6fbf33336cbf9a9968bf000065bf666661bfcdcc5dbf33335abf9a9956bf"
    "000053bf66664fbfcdcc4bbf333348bf9a9944bf000041bf66663dbfcdcc39bf343336bf9a9932bf00002fbf66662bbf"
    "cdcc27bf343324bf9a9920bf00001dbf666619bfcdcc15bf343312bf9a990ebf00000bbf666607bfcccc03bf333300bf"
    "3233f9be0000f2becccceabe9a99e3be6666dcbe3333d5be0000cebeccccc6be9a99bfbe6666b8be3333b1be0000aabe"
    "cccca2be9a999bbe666694be34338dbe000086be99997dbe33336fbecdcc60be666652be000044be9a9935be333327be"
    "cdcc18be66660abe0000f8bd3433dbbd85ebc9bdf728c4bd6766bebdd7a3b8bd48e1b2bdb81eadbd2a5ca7bd9a99a1bd"
    "0ad79bbd7b1496bdeb5190bd5d8f8abdcdcc84bd7b147ebd5d8f72bd3d0a67bd1f855bbd000050bde17a44bdc3f538bd"
    "a3702dbd85eb21bd676616bd48e10abd52b8febc15aee7bcd7a3d0bc9a99b9bc5d8fa2bc1f858bbcc3f568bc48e13abc"
    "643b1fbc180416bccdcc0cbc819503bc6abcf4bbd34de2bb3bdfcfbba470bdbb0d02abbb749398bbdd2486bb8a6c67bb"
    "5c8f42bb2db21dbbfca9f1ba9defa7baff6577ba3ee859ba806a3cbac1ec1eba016f01ba83e2c7b905e78cb90bd723b9"
    "4b1fbab8b3ef8ab8348037b80642b2b7ff0502b7941a5ab6b7a313b500000000b7a31335941a5a36ff0502370642b237"
    "34803738b3ef8a384b1fba380bd7233905e78c3983e2c739016f013ac1ec1e3a806a3c3a3ee8593aff65773a9defa73a"
    "fca9f13a2db21d3b5c8f423b8a6c673bdd24863b7493983b0d02ab3ba470bd3b3bdfcf3bd34de23b6abcf43b8195033c"
    "cdcc0c3c1804163c643b1f3c48e13a3cc3f5683c1f858b3c5d8fa23c9a99b93cd7a3d03c15aee73c52b8fe3c48e10a3d"
    "6766163d85eb213da3702d3dc3f5383de17a443d0000503d1f855b3d3d0a673d5d8f723d7b147e3dcdcc843d5d8f8a3d"
    "eb51903d7b14963d0ad79b3d9a99a13d2a5ca73db81ead3d48e1b23dd7a3b83d6766be3df728c43d85ebc93d3433db3d"
    "0000f83d66660a3ecdcc183e3333273e9a99353e0000443e6666523ecdcc603e33336f3e99997d3e0000863e34338d3e"
    "6666943e9a999b3ecccca23e0000aa3e3333b13e6666b83e9a99bf3eccccc63e0000ce3e3333d53e6666dc3e9a99e33e"
    "ccccea3e0000f23e3233f93e3333003fcccc033f6666073f00000b3f9a990e3f3433123fcdcc153f6666193f00001d3f"
    "9a99203f3433243fcdcc273f66662b3f00002f3f9a99323f3433363fcdcc393f66663d3f0000413f9a99443f3333483f"
    "cdcc4b3f66664f3f0000533f9a99563f33335a3fcdcc5d3f6666613f0000653f9a99683f33336c3fcdcc6f3f6666733f"
    "0000773f9a997a3f33337e3f0000803f"
)


@lru_cache(maxsize=2)
def _codebook(quant_type):
    code = np.array(_NF4 if quant_type == "nf4" else _FP4, dtype=np.float32)
    order = np.argsort(code, kind="stable")
    bounds = (code[order[:-1]] + code[order[1:]]) / np.float32(2)
    return mx.array(bounds), mx.array(order.astype(np.uint8))


@lru_cache(maxsize=1)
def _pack_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_bnb_4bit_pack",
        input_names=["weight", "absmax", "bounds", "order"],
        output_names=["packed"],
        header="""
            inline float bnb_scaled(uint bits, uint maximum_bits) {
                uint mantissa = bits & 0x7fffffffu;
                if (mantissa == 0 || maximum_bits == 0) return 0.0f;
                float inverse = maximum_bits < 0x00800000u ?
                    as_type<float>(0x7e96769au) : 1.0f / as_type<float>(maximum_bits);
                if (mantissa > 0 && mantissa < 0x00800000u) {
                    float magnitude = float(mantissa) * (0x1.0p-126f * inverse) * 0x1.0p-23f;
                    if (magnitude == 0.0f && float(mantissa) * inverse > 0.5f)
                        magnitude = 0x1.0p-126f;
                    return (bits >> 31) ? -magnitude : magnitude;
                }
                float value = as_type<float>(bits);
                if (as_type<float>(maximum_bits) > 1.0e37f)
                    return value / as_type<float>(maximum_bits);
                return value * inverse;
            }
        """,
        source="""
            uint pair = thread_position_in_grid.x;
            if (pair >= PAIRS) return;
            uint first = pair * 2;
            uint block0 = first / BLOCK;
            float a = metal::clamp(
                bnb_scaled(as_type<uint>(float(weight[first])), absmax[block0]),
                -1.0f, 1.0f);
            uint code0 = 0;
            for (uint i = 0; i < 15; ++i) code0 += a > bounds[i];
            code0 = uint(order[code0]);
            uint code1 = 0;
            if (first + 1 < SIZE) {
                uint block1 = (first + 1) / BLOCK;
                float b = metal::clamp(
                    bnb_scaled(as_type<uint>(float(weight[first + 1])), absmax[block1]),
                    -1.0f, 1.0f);
                for (uint i = 0; i < 15; ++i) code1 += b > bounds[i];
                code1 = uint(order[code1]);
            } else {
                for (uint i = 0; i < 15; ++i) code1 += 0.0f > bounds[i];
                code1 = uint(order[code1]);
            }
            packed[pair] = uchar((code0 << 4) | code1);
        """,
    )


def _block_absmax(weight, block_size, *, clamp_remainder=False):
    flat = weight.reshape(-1)
    padded = (block_size - flat.size % block_size) % block_size
    if padded:
        flat = mx.pad(flat, [(0, padded)])
    if flat.dtype == mx.float32:
        magnitude = flat.view(mx.uint32) & 0x7fffffff
        maximum = mx.max(magnitude.reshape(-1, block_size), axis=1).view(mx.float32)
    else:
        maximum = mx.max(mx.abs(flat.reshape(-1, block_size)), axis=1).astype(mx.float32)
    if padded and clamp_remainder:
        maximum = mx.concatenate([maximum[:-1], mx.maximum(maximum[-1:], 1e-38)])
    return maximum


@lru_cache(maxsize=1)
def _dynamic_lookup_np():
    code = np.frombuffer(bytes.fromhex(_DYNAMIC_CODE_HEX), dtype="<f4")
    bounds = (code[:-1] + code[1:]) / np.float32(2)
    indices = np.arange(65536, dtype=np.float32)
    values = np.float32(-1) + (indices * np.float32(2)) / np.float32(65535)
    lookup = np.searchsorted(bounds, values, side="left").astype(np.uint8)
    return lookup, code


@lru_cache(maxsize=1)
def _dynamic_codebook():
    lookup, code = _dynamic_lookup_np()
    return mx.array(lookup), mx.array(code)


def _subnormal_nested_scales(absmax):
    raw = np.asarray(absmax).astype(np.float32)
    offset = np.float32(raw.astype(np.float64).mean())
    centered = (raw - offset).astype(np.float32)
    padded = (-centered.size) % 256
    blocks = np.pad(centered, (0, padded)).reshape(-1, 256)
    nested_absmax = np.max(np.abs(blocks), axis=1)
    lookup, code = _dynamic_lookup_np()
    nested_codes = np.zeros(centered.size, dtype=np.uint8)
    for block, maximum in enumerate(nested_absmax):
        start = block * 256
        end = min(start + 256, centered.size)
        if maximum:
            scaled = np.clip(centered[start:end].astype(np.float64) / float(maximum), -1, 1)
            indices = np.floor((scaled + 1) * 32767.5 + 0.5).astype(np.int32)
            nested_codes[start:end] = lookup[indices]
    return {
        "absmax": mx.array(nested_codes),
        "nested_absmax": mx.array(nested_absmax),
        "offset": mx.array(offset),
        "nested_code": mx.array(code),
    }


@lru_cache(maxsize=1)
def _nested_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_bnb_nested_scales",
        input_names=["scales", "absmax", "lookup"],
        output_names=["codes"],
        source="""
            uint i = thread_position_in_grid.x;
            if (i >= SIZE) return;
            float scale = absmax[i / BLOCK];
            if (scale == 0.0f) {
                codes[i] = uchar(0);
                return;
            }
            float value = metal::clamp(scales[i] * (1.0f / scale), -1.0f, 1.0f);
            uint index = uint(metal::fma(value + 1.0f, 32767.5f, 0.5f));
            codes[i] = lookup[metal::min(index, 65535u)];
        """,
    )


@lru_cache(maxsize=1)
def _int8_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_bnb_int8_quant",
        input_names=["weight", "absmax"],
        output_names=["codes"],
        source="""
            uint i = thread_position_in_grid.x;
            if (i >= SIZE) return;
            float maximum = absmax[i / COLS];
            float ratio = 127.0f / maximum;
            float scaled = maximum == 0.0f ? 0.0f : float(weight[i]) * ratio;
            codes[i] = char(metal::rint(scaled));
        """,
        compile_options={"math_mode": "safe"},
    )


def quantize_4bit_weight_mlx(weight, *, quant_type="nf4", block_size=64, compress_statistics=False):
    """Return bitsandbytes packed bytes and block scales for an MLX weight.

    With compressed statistics, the second result is a dictionary containing
    ``absmax`` (8-bit codes), ``nested_absmax``, ``offset``, and ``nested_code``.
    Otherwise it is the float32 block maxima.
    """
    if weight.ndim != 2 or not all(weight.shape):
        raise ValueError("weight must be a nonempty rank-2 matrix")
    if weight.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("weight must have float16, bfloat16, or float32 dtype")
    if quant_type not in ("nf4", "fp4"):
        raise ValueError("quant_type must be 'nf4' or 'fp4'")
    if block_size not in _BLOCK_SIZES:
        raise ValueError("unsupported bitsandbytes block_size")
    if not bool(mx.all(mx.isfinite(weight)).item()):
        raise ValueError("weight must be finite")
    flat = mx.contiguous(weight.reshape(-1))
    absmax = _block_absmax(weight, block_size, clamp_remainder=True)
    bounds, order = _codebook(quant_type)
    pairs = (flat.size + 1) // 2
    packed = _pack_kernel()(
        inputs=[flat, absmax.view(mx.uint32), bounds, order],
        template=[("SIZE", flat.size), ("PAIRS", pairs), ("BLOCK", block_size)],
        grid=(pairs, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(pairs, 1)],
        output_dtypes=[mx.uint8],
    )[0]
    if not compress_statistics:
        mx.eval(packed, absmax)
        return packed, absmax
    if bool((mx.max(absmax.view(mx.uint32)) < 0x00800000).item()):
        state = _subnormal_nested_scales(absmax)
        mx.eval(packed)
        return packed, state
    anchor = mx.mean(absmax[: min(absmax.size, 256)])
    offset = anchor + mx.mean(absmax - anchor)
    centered = absmax - offset
    nested_absmax = _block_absmax(centered, 256)
    dynamic_lookup, dynamic_code = _dynamic_codebook()
    nested_codes = _nested_kernel()(
        inputs=[centered, nested_absmax, dynamic_lookup],
        template=[("SIZE", absmax.size), ("BLOCK", 256)],
        grid=(absmax.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(absmax.size,)],
        output_dtypes=[mx.uint8],
    )[0]
    mx.eval(packed, nested_codes, nested_absmax, offset)
    return packed, {
        "absmax": nested_codes,
        "nested_absmax": nested_absmax,
        "offset": offset,
        "nested_code": dynamic_code,
    }


def quantize_int8_weight_mlx(weight):
    """Return bitsandbytes LLM.int8 row codes and row maxima for an MLX weight."""
    if weight.ndim != 2 or not all(weight.shape):
        raise ValueError("weight must be a nonempty rank-2 matrix")
    if weight.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("weight must have float16, bfloat16, or float32 dtype")
    if not bool(mx.all(mx.isfinite(weight)).item()):
        raise ValueError("weight must be finite")
    matrix = weight
    absmax = mx.max(mx.abs(matrix), axis=1).astype(mx.float32)
    codes = _int8_kernel()(
        inputs=[mx.contiguous(matrix), absmax],
        template=[("SIZE", weight.size), ("COLS", weight.shape[1])],
        grid=(weight.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[weight.shape],
        output_dtypes=[mx.int8],
    )[0]
    mx.eval(codes, absmax)
    return codes, absmax
