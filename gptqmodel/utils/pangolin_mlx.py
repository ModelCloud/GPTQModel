# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Optional MLX-native Pangolin packed-weight GEMV for Apple silicon."""

from __future__ import annotations

import threading
from typing import Any

PANGOLIN_MLX_BITS = (2, 3, 4, 5, 6, 7, 8)
_KERNEL: Any | None = None
_KERNEL_M8: Any | None = None
_KERNEL_MULTIROW: Any | None = None
_KERNEL_ERRORS: dict[int, str] = {}
_KERNEL_LOCK = threading.Lock()

_HEADER = r"""
inline uint pq(device const int* p, uint k, uint n, uint N, uint b, bool rows) {
  const uint axis = rows ? k : n, block = axis >> 5, lane = axis & 31;
  const uint stride = rows ? N : 1, outer = rows ? 0 : k * ((N >> 5) * b);
  const uint base = outer + block * b * stride + (rows ? n : 0);
  uint a, c, d = 0;
  if (b == 3) { a=2; c=1; } else if (b == 5) { a=4; c=1; }
  else if (b == 6) { a=4; c=2; } else { a=4; c=2; d=1; }
  uint w0=as_type<uint>(p[base+(lane/(32/a))*stride]);
  uint w1=as_type<uint>(p[base+(a+lane/(32/c))*stride]);
  uint v=((w0>>(a*(lane%(32/a))))&((1u<<a)-1u));
  v|=((w1>>(c*(lane%(32/c))))&((1u<<c)-1u))<<a;
  if(d){uint w2=as_type<uint>(p[base+(a+c+lane/32)*stride]);v|=((w2>>(lane&31))&1u)<<(a+c);}
  return v;
}
inline uint cq(device const int* p,uint outer,uint axis,uint width,uint b,uint stride){
  uint pf=32/b; uint w=as_type<uint>(p[outer*width+(axis/pf)*stride]);
  return (w>>(b*(axis%pf)))&((1u<<b)-1u);
}
inline uint pq(constant const int* p, uint k, uint n, uint N, uint b, bool rows) {
  const uint axis=rows?k:n,block=axis>>5,lane=axis&31,stride=rows?N:1,outer=rows?0:k*((N>>5)*b),base=outer+block*b*stride+(rows?n:0);
  uint a,c,d=0;if(b==3){a=2;c=1;}else if(b==5){a=4;c=1;}else if(b==6){a=4;c=2;}else{a=4;c=2;d=1;}
  uint w0=as_type<uint>(p[base+(lane/(32/a))*stride]),w1=as_type<uint>(p[base+(a+lane/(32/c))*stride]);
  uint v=((w0>>(a*(lane%(32/a))))&((1u<<a)-1u))|(((w1>>(c*(lane%(32/c))))&((1u<<c)-1u))<<a);
  if(d){uint w2=as_type<uint>(p[base+(a+c+lane/32)*stride]);v|=((w2>>(lane&31))&1u)<<(a+c);}return v;
}
inline uint cq(constant const int* p,uint outer,uint axis,uint width,uint b,uint stride){
  uint pf=32/b,w=as_type<uint>(p[outer*width+(axis/pf)*stride]);return(w>>(b*(axis%pf)))&((1u<<b)-1u);
}
"""

_SOURCE = r"""
uint group=threadgroup_position_in_grid.x, lane=thread_index_in_simdgroup;
uint M=dims[0],K=dims[1],N=dims[2],G=dims[3],b=dims[4]; bool planar=dims[5]!=0;
if(group<M*N){uint m=group/N,n=group-m*N; float sum=0.0f; bool valid=true;
  for(uint k=lane;k<K;k+=32){int gi=gidx[k];if(gi<0)gi+=int(G);uint g=uint(gi);
    if(gi<0||gi>=int(G)){valid=false;break;}
    uint code=planar?pq(qweight,k,n,N,b,true):cq(qweight+n,0,k,K/(32/b),b,N);
    uint zero=planar?pq(qzeros,g,n,N,b,false):cq(qzeros,g,n,N/(32/b),b,1);
    sum+=float(x[m*K+k])*(float(int(code)-int(zero))*float(scales[g*N+n]));}
  valid=simd_all(valid);sum=simd_sum(sum);if(lane==0)out[group]=valid?T(sum):T(NAN);}
"""

_SOURCE_M8 = r"""
uint n=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint simd=simdgroup_index_in_threadgroup,tid=thread_index_in_threadgroup;
uint M=dims[0],K=dims[1],N=dims[2],G=dims[3],b=dims[4];bool planar=dims[5]!=0;
threadgroup float decoded[32];threadgroup atomic_uint valid;
if(tid==0)atomic_store_explicit(&valid,1u,memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);float sum=0.0f;bool active=simd<M;
for(uint base=0;base<K;base+=32){if(simd==0){uint k=base+lane;int gi=gidx[k];if(gi<0)gi+=int(G);
  if(gi<0||gi>=int(G)){atomic_store_explicit(&valid,0u,memory_order_relaxed);decoded[lane]=0.0f;}
  else{uint g=uint(gi);uint code=planar?pq(qweight,k,n,N,b,true):cq(qweight+n,0,k,K/(32/b),b,N);
    uint zero=planar?pq(qzeros,g,n,N,b,false):cq(qzeros,g,n,N/(32/b),b,1);
    decoded[lane]=float(int(code)-int(zero))*float(scales[g*N+n]);}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(active)sum+=float(x[simd*K+base+lane])*decoded[lane];
  threadgroup_barrier(mem_flags::mem_threadgroup);}
if(active){sum=simd_sum(sum);if(lane==0)out[simd*N+n]=atomic_load_explicit(&valid,memory_order_relaxed)?T(sum):T(NAN);}
"""


_SOURCE_MULTIROW = r"""
uint n=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint simd=simdgroup_index_in_threadgroup,tid=thread_index_in_threadgroup;
uint M=dims[0],K=dims[1],N=dims[2],G=dims[3],b=dims[4],stride=dims[6];bool planar=dims[5]!=0;
threadgroup float decoded[32];threadgroup atomic_uint valid;
if(tid==0)atomic_store_explicit(&valid,1u,memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;
uint r0=simd,r1=simd+stride,r2=simd+2*stride,r3=simd+3*stride;
for(uint base=0;base<K;base+=32){if(simd==0){uint k=base+lane;int gi=gidx[k];if(gi<0)gi+=int(G);
  if(gi<0||gi>=int(G)){atomic_store_explicit(&valid,0u,memory_order_relaxed);decoded[lane]=0.0f;}
  else{uint g=uint(gi);uint code=planar?pq(qweight,k,n,N,b,true):cq(qweight+n,0,k,K/(32/b),b,N);
    uint zero=planar?pq(qzeros,g,n,N,b,false):cq(qzeros,g,n,N/(32/b),b,1);
    decoded[lane]=float(int(code)-int(zero))*float(scales[g*N+n]);}}
  threadgroup_barrier(mem_flags::mem_threadgroup);float v=decoded[lane];
  if(r0<M)s0+=float(x[r0*K+base+lane])*v;if(r1<M)s1+=float(x[r1*K+base+lane])*v;
  if(r2<M)s2+=float(x[r2*K+base+lane])*v;if(r3<M)s3+=float(x[r3*K+base+lane])*v;
  threadgroup_barrier(mem_flags::mem_threadgroup);}
s0=simd_sum(s0);s1=simd_sum(s1);s2=simd_sum(s2);s3=simd_sum(s3);
if(lane==0){bool ok=atomic_load_explicit(&valid,memory_order_relaxed);
  if(r0<M)out[r0*N+n]=ok?T(s0):T(NAN);if(r1<M)out[r1*N+n]=ok?T(s1):T(NAN);
  if(r2<M)out[r2*N+n]=ok?T(s2):T(NAN);if(r3<M)out[r3*N+n]=ok?T(s3):T(NAN);}
"""


def _kernel(*, mode: int = 1):
    global _KERNEL, _KERNEL_M8, _KERNEL_MULTIROW
    if mode not in (1, 8, 32):
        raise ValueError(f"invalid Pangolin MLX kernel mode: {mode}")
    if mode == 1:
        kernel = _KERNEL
    elif mode == 8:
        kernel = _KERNEL_M8
    else:
        kernel = _KERNEL_MULTIROW
    if kernel is None:
        with _KERNEL_LOCK:
            if mode == 1:
                kernel = _KERNEL
            elif mode == 8:
                kernel = _KERNEL_M8
            else:
                kernel = _KERNEL_MULTIROW
            if kernel is None:
                if mode in _KERNEL_ERRORS:
                    raise RuntimeError(_KERNEL_ERRORS[mode])
                import mlx.core as mx

                try:
                    kernel = mx.fast.metal_kernel(
                        name={
                            1: "gptqmodel_pangolin",
                            8: "gptqmodel_pangolin_m8",
                            32: "gptqmodel_pangolin_multirow",
                        }[mode],
                        input_names=[
                            "x",
                            "qweight",
                            "scales",
                            "qzeros",
                            "gidx",
                            "dims",
                        ],
                        output_names=["out"],
                        header=_HEADER,
                        source={1: _SOURCE, 8: _SOURCE_M8, 32: _SOURCE_MULTIROW}[mode],
                        ensure_row_contiguous=True,
                    )
                except Exception as exc:
                    error = f"Pangolin MLX kernel creation failed: {exc}"
                    _KERNEL_ERRORS[mode] = error
                    raise RuntimeError(error) from exc
                if mode == 8:
                    _KERNEL_M8 = kernel
                elif mode == 32:
                    _KERNEL_MULTIROW = kernel
                else:
                    _KERNEL = kernel
    return kernel


def pangolin_mlx_gemv(
    x,
    qweight,
    scales,
    qzeros,
    g_idx,
    bits: int,
    *,
    planar: bool,
    _g_idx_validated: bool = False,
):
    """Run fused Pangolin GEMV on MLX arrays without a decoded-weight cache."""
    import mlx.core as mx

    if bits not in PANGOLIN_MLX_BITS:
        raise ValueError(f"Pangolin MLX supports bits {PANGOLIN_MLX_BITS}, got {bits}")
    if planar != (bits in (3, 5, 6, 7)):
        raise ValueError(f"invalid planar={planar} for {bits}-bit gptq_p")
    if (
        x.ndim != 2
        or qweight.ndim != 2
        or scales.ndim != 2
        or qzeros.ndim != 2
        or g_idx.ndim != 1
    ):
        raise ValueError("Pangolin MLX expects 2D x/packed/scales tensors and 1D g_idx")
    if x.dtype != mx.float16 or scales.dtype != mx.float16:
        raise TypeError("Pangolin MLX requires float16 x and scales")
    if any(v.dtype != mx.int32 for v in (qweight, qzeros, g_idx)):
        raise TypeError("Pangolin MLX requires int32 packed tensors and g_idx")
    m, k = x.shape
    groups, n = scales.shape
    if not groups or not k or not n or k % 32 or n % 32 or g_idx.shape != (k,):
        raise ValueError(
            "Pangolin MLX requires positive K/N divisible by 32 and g_idx[K]"
        )
    if (
        max(
            m,
            k,
            n,
            groups,
            m * n,
            *(v.size for v in (x, qweight, scales, qzeros, g_idx)),
        )
        > 2**32 - 1
    ):
        raise ValueError("Pangolin MLX dimensions exceed the uint32 kernel limit")
    expected_w = ((k // 32) * bits, n) if planar else (k // (32 // bits), n)
    expected_z = (groups, (n // 32) * bits) if planar else (groups, n // (32 // bits))
    if qweight.shape != expected_w or qzeros.shape != expected_z:
        raise ValueError(f"invalid packed shapes {qweight.shape}/{qzeros.shape}")
    if not m:
        return mx.empty((0, n), dtype=x.dtype)
    # MLX evaluates lazily; bounds checking through a host conversion happens
    # before the unchecked custom kernel is put on the stream.
    if not _g_idx_validated:
        lo, hi = int(mx.min(g_idx).item()), int(mx.max(g_idx).item())
        if lo < -groups or hi >= groups:
            raise ValueError("g_idx contains an out-of-bounds group index")
    row_stride = 4 if m <= 16 else 8
    dims = mx.array([m, k, n, groups, bits, int(planar), row_stride], dtype=mx.uint32)
    # MLX launch overhead differs slightly from PyTorch's: planar decode wins
    # from M=4, while the cheaper continuous formats need more row reuse.
    shared = m <= 8 and (
        (planar and m >= 4) or (bits == 2 and m >= 6) or (bits in (4, 8) and m >= 5)
    )
    mode = 32 if 9 <= m <= 32 else (8 if shared else 1)
    group_size = 128 if 9 <= m <= 16 else (256 if mode in (8, 32) else 32)
    return _kernel(mode=mode)(
        inputs=[x, qweight, scales, qzeros, g_idx, dims],
        template=[("T", mx.float16)],
        grid=(n * group_size if mode in (8, 32) else m * n * 32, 1, 1),
        threadgroup=(group_size, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float16],
    )[0]


__all__ = ["PANGOLIN_MLX_BITS", "pangolin_mlx_gemv"]
