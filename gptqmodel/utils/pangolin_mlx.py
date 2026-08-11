# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Optional MLX-native Pangolin packed-weight GEMV for Apple silicon."""

from __future__ import annotations

import functools
import threading
from typing import Any

PANGOLIN_MLX_BITS = (2, 3, 4, 5, 6, 7, 8)
_KERNEL: Any | None = None
_KERNEL_N4: Any | None = None
_KERNEL_N4_UNIFORM: Any | None = None
_KERNEL_M8: Any | None = None
_KERNEL_M8_N4: Any | None = None
_KERNEL_M8_N8: Any | None = None
_KERNEL_M16_N4: Any | None = None
_KERNEL_M16_N8: Any | None = None
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

#define PANGOLIN_VECTOR_HELPERS(address_space) \
inline uint4 load4(address_space const int* p,uint i){return as_type<uint4>(*reinterpret_cast<address_space const int4*>(p+i));} \
inline uint4 pq4(address_space const int* p,uint k,uint n,uint N,uint b,bool rows){ \
  uint w0,w1,w2=0;if(b==3){w0=2;w1=1;}else if(b==5){w0=4;w1=1;}else if(b==6){w0=4;w1=2;}else{w0=4;w1=2;w2=1;} \
  if(rows){uint block=k>>5,lane=k&31,base=block*b,pf0=32/w0,pf1=32/w1;uint4 a=load4(p,(base+lane/pf0)*N+n); \
    uint4 c=load4(p,(base+w0+lane/pf1)*N+n);uint4 v=((a>>(w0*(lane%pf0)))&((1u<<w0)-1u))|(((c>>(w1*(lane%pf1)))&((1u<<w1)-1u))<<w0); \
    if(w2){uint pf2=32/w2;uint4 d=load4(p,(base+w0+w1+lane/pf2)*N+n);v|=((d>>(w2*(lane%pf2)))&1u)<<(w0+w1);}return v;} \
  uint block=n>>5,lane=n&31,words=(N>>5)*b,base=k*words+block*b,pf0=32/w0,pf1=32/w1;uint4 lanes=uint4(lane)+uint4(0,1,2,3); \
  uint a=as_type<uint>(p[base+lane/pf0]),c=as_type<uint>(p[base+w0+lane/pf1]);uint4 v=((uint4(a)>>(w0*(lanes%pf0)))&((1u<<w0)-1u))|(((uint4(c)>>(w1*(lanes%pf1)))&((1u<<w1)-1u))<<w0); \
  if(w2){uint d=as_type<uint>(p[base+w0+w1+lane/32]);v|=((uint4(d)>>(lanes&31))&1u)<<(w0+w1);}return v;} \
inline uint4 cqw4(address_space const int* p,uint k,uint n,uint N,uint b){uint pf=32/b;uint4 w=load4(p,(k/pf)*N+n);return(w>>(b*(k%pf)))&((1u<<b)-1u);} \
inline uint4 cqz4(address_space const int* p,uint g,uint n,uint N,uint b){uint pf=32/b,words=N/pf,w=as_type<uint>(p[g*words+n/pf]);uint4 shifts=b*(uint4(n%pf)+uint4(0,1,2,3));return(uint4(w)>>shifts)&((1u<<b)-1u);}

PANGOLIN_VECTOR_HELPERS(device)
PANGOLIN_VECTOR_HELPERS(constant)
#undef PANGOLIN_VECTOR_HELPERS
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

_SOURCE_N4 = r"""
uint group=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint K=dims[1],N=dims[2],G=dims[3],b=dims[4];bool planar=dims[5]!=0;
uint groups_n=N/4,m=group/groups_n,n=(group-m*groups_n)*4;float4 sum=0.0f;bool valid=true;
for(uint k=lane;k<K;k+=32){int gi=gidx[k];if(gi<0)gi+=int(G);uint g=uint(gi);
  if(gi<0||gi>=int(G)){valid=false;break;}
  uint4 code=planar?pq4(qweight,k,n,N,b,true):cqw4(qweight,k,n,N,b);
  uint4 zero=planar?pq4(qzeros,g,n,N,b,false):cqz4(qzeros,g,n,N,b);
  float4 scale=float4(scales[g*N+n],scales[g*N+n+1],scales[g*N+n+2],scales[g*N+n+3]);
  sum+=float(x[m*K+k])*(float4(int4(code)-int4(zero))*scale);}
valid=simd_all(valid);sum.x=simd_sum(sum.x);sum.y=simd_sum(sum.y);
sum.z=simd_sum(sum.z);sum.w=simd_sum(sum.w);
if(lane==0){half4 value=valid?half4(sum):half4(NAN);out[m*N+n]=value.x;out[m*N+n+1]=value.y;
  out[m*N+n+2]=value.z;out[m*N+n+3]=value.w;}
"""

_SOURCE_N4_UNIFORM = r"""
uint group=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint K=dims[1],N=dims[2],G=dims[3],b=dims[4];bool planar=dims[5]!=0;
uint groups_n=N/4,m=group/groups_n,n=(group-m*groups_n)*4;float4 sum=0.0f;bool valid=true;
for(uint k=lane;k<K;k+=32){int gi=lane==0?gidx[k]:0;gi=simd_broadcast_first(gi);
  if(gi<0)gi+=int(G);uint g=uint(gi);if(gi<0||gi>=int(G)){valid=false;break;}
  uint4 code=planar?pq4(qweight,k,n,N,b,true):cqw4(qweight,k,n,N,b);uint4 zero=0;float4 scale=0.0f;
  if(lane==0){zero=planar?pq4(qzeros,g,n,N,b,false):cqz4(qzeros,g,n,N,b);
    scale=float4(scales[g*N+n],scales[g*N+n+1],scales[g*N+n+2],scales[g*N+n+3]);}
  zero.x=simd_broadcast_first(zero.x);zero.y=simd_broadcast_first(zero.y);
  zero.z=simd_broadcast_first(zero.z);zero.w=simd_broadcast_first(zero.w);
  scale.x=simd_broadcast_first(scale.x);scale.y=simd_broadcast_first(scale.y);
  scale.z=simd_broadcast_first(scale.z);scale.w=simd_broadcast_first(scale.w);
  sum+=float(x[m*K+k])*(float4(int4(code)-int4(zero))*scale);}
valid=simd_all(valid);sum.x=simd_sum(sum.x);sum.y=simd_sum(sum.y);
sum.z=simd_sum(sum.z);sum.w=simd_sum(sum.w);
if(lane==0){half4 value=valid?half4(sum):half4(NAN);out[m*N+n]=value.x;out[m*N+n+1]=value.y;
  out[m*N+n+2]=value.z;out[m*N+n+3]=value.w;}
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

_SOURCE_M8_N4 = r"""
uint group=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint simd=simdgroup_index_in_threadgroup,tid=thread_index_in_threadgroup;
uint M=dims[0],K=dims[1],N=dims[2],G=dims[3],b=dims[4],n=group*4;bool planar=dims[5]!=0;
threadgroup float4 decoded[32];threadgroup atomic_uint valid;
if(tid==0)atomic_store_explicit(&valid,1u,memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);float4 sum=0.0f;bool active=simd<M;
for(uint base=0;base<K;base+=32){if(simd==0){uint k=base+lane;int gi=gidx[k];if(gi<0)gi+=int(G);
  if(gi<0||gi>=int(G)){atomic_store_explicit(&valid,0u,memory_order_relaxed);decoded[lane]=0.0f;}
  else{uint g=uint(gi);uint4 code=planar?pq4(qweight,k,n,N,b,true):cqw4(qweight,k,n,N,b);
    uint4 zero=planar?pq4(qzeros,g,n,N,b,false):cqz4(qzeros,g,n,N,b);
    float4 scale=float4(scales[g*N+n],scales[g*N+n+1],scales[g*N+n+2],scales[g*N+n+3]);
    decoded[lane]=float4(int4(code)-int4(zero))*scale;}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(active)sum+=float(x[simd*K+base+lane])*decoded[lane];
  threadgroup_barrier(mem_flags::mem_threadgroup);}
if(active){sum.x=simd_sum(sum.x);sum.y=simd_sum(sum.y);sum.z=simd_sum(sum.z);sum.w=simd_sum(sum.w);
  if(lane==0){half4 value=atomic_load_explicit(&valid,memory_order_relaxed)?half4(sum):half4(NAN);
    out[simd*N+n]=value.x;out[simd*N+n+1]=value.y;out[simd*N+n+2]=value.z;out[simd*N+n+3]=value.w;}}
"""

_SOURCE_M8_N8 = r"""
uint group=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint simd=simdgroup_index_in_threadgroup,tid=thread_index_in_threadgroup;
uint M=dims[0],K=dims[1],N=dims[2],G=dims[3],b=dims[4],n=group*8;bool planar=dims[5]!=0;
threadgroup float4 d0[32],d1[32];threadgroup atomic_uint valid;
if(tid==0)atomic_store_explicit(&valid,1u,memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);float4 s0=0.0f,s1=0.0f;bool active=simd<M;
for(uint base=0;base<K;base+=32){if(simd==0){uint k=base+lane;int gi=gidx[k];if(gi<0)gi+=int(G);
  if(gi<0||gi>=int(G)){atomic_store_explicit(&valid,0u,memory_order_relaxed);d0[lane]=0.0f;d1[lane]=0.0f;}
  else{uint g=uint(gi);uint4 c0=planar?pq4(qweight,k,n,N,b,true):cqw4(qweight,k,n,N,b);
    uint4 c1=planar?pq4(qweight,k,n+4,N,b,true):cqw4(qweight,k,n+4,N,b);
    uint4 z0=planar?pq4(qzeros,g,n,N,b,false):cqz4(qzeros,g,n,N,b);
    uint4 z1=planar?pq4(qzeros,g,n+4,N,b,false):cqz4(qzeros,g,n+4,N,b);
    float4 q0=float4(scales[g*N+n],scales[g*N+n+1],scales[g*N+n+2],scales[g*N+n+3]);
    float4 q1=float4(scales[g*N+n+4],scales[g*N+n+5],scales[g*N+n+6],scales[g*N+n+7]);
    d0[lane]=float4(int4(c0)-int4(z0))*q0;d1[lane]=float4(int4(c1)-int4(z1))*q1;}}
  threadgroup_barrier(mem_flags::mem_threadgroup);if(active){float v=float(x[simd*K+base+lane]);s0+=v*d0[lane];s1+=v*d1[lane];}
  threadgroup_barrier(mem_flags::mem_threadgroup);}
if(active){s0.x=simd_sum(s0.x);s0.y=simd_sum(s0.y);s0.z=simd_sum(s0.z);s0.w=simd_sum(s0.w);
  s1.x=simd_sum(s1.x);s1.y=simd_sum(s1.y);s1.z=simd_sum(s1.z);s1.w=simd_sum(s1.w);
  if(lane==0){bool ok=atomic_load_explicit(&valid,memory_order_relaxed);half4 v0=ok?half4(s0):half4(NAN),v1=ok?half4(s1):half4(NAN);
    out[simd*N+n]=v0.x;out[simd*N+n+1]=v0.y;out[simd*N+n+2]=v0.z;out[simd*N+n+3]=v0.w;
    out[simd*N+n+4]=v1.x;out[simd*N+n+5]=v1.y;out[simd*N+n+6]=v1.z;out[simd*N+n+7]=v1.w;}}
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

_SOURCE_M16_N4 = r"""
uint group=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint simd=simdgroup_index_in_threadgroup,tid=thread_index_in_threadgroup;
uint M=dims[0],K=dims[1],N=dims[2],G=dims[3],b=dims[4],n=group*4;bool planar=dims[5]!=0;
uint r0=simd,r1=simd+8;threadgroup float4 decoded[32];threadgroup atomic_uint valid;
if(tid==0)atomic_store_explicit(&valid,1u,memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);float4 s0=0.0f,s1=0.0f;
for(uint base=0;base<K;base+=32){if(simd==0){uint k=base+lane;int gi=gidx[k];if(gi<0)gi+=int(G);
  if(gi<0||gi>=int(G)){atomic_store_explicit(&valid,0u,memory_order_relaxed);decoded[lane]=0.0f;}
  else{uint g=uint(gi);uint4 code=planar?pq4(qweight,k,n,N,b,true):cqw4(qweight,k,n,N,b);
    uint4 zero=planar?pq4(qzeros,g,n,N,b,false):cqz4(qzeros,g,n,N,b);
    float4 scale=float4(scales[g*N+n],scales[g*N+n+1],scales[g*N+n+2],scales[g*N+n+3]);
    decoded[lane]=float4(int4(code)-int4(zero))*scale;}}
  threadgroup_barrier(mem_flags::mem_threadgroup);float4 v=decoded[lane];
  s0+=float(x[r0*K+base+lane])*v;if(r1<M)s1+=float(x[r1*K+base+lane])*v;
  threadgroup_barrier(mem_flags::mem_threadgroup);}
s0.x=simd_sum(s0.x);s0.y=simd_sum(s0.y);s0.z=simd_sum(s0.z);s0.w=simd_sum(s0.w);
s1.x=simd_sum(s1.x);s1.y=simd_sum(s1.y);s1.z=simd_sum(s1.z);s1.w=simd_sum(s1.w);
if(lane==0){bool ok=atomic_load_explicit(&valid,memory_order_relaxed);half4 v0=ok?half4(s0):half4(NAN);
  out[r0*N+n]=v0.x;out[r0*N+n+1]=v0.y;out[r0*N+n+2]=v0.z;out[r0*N+n+3]=v0.w;
  if(r1<M){half4 v1=ok?half4(s1):half4(NAN);out[r1*N+n]=v1.x;out[r1*N+n+1]=v1.y;
    out[r1*N+n+2]=v1.z;out[r1*N+n+3]=v1.w;}}
"""

_SOURCE_M16_N8 = r"""
uint group=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint simd=simdgroup_index_in_threadgroup,tid=thread_index_in_threadgroup;
uint M=dims[0],K=dims[1],N=dims[2],G=dims[3],b=dims[4],n=group*8;bool planar=dims[5]!=0;
uint r0=simd,r1=simd+8;threadgroup float4 d0[32],d1[32];threadgroup atomic_uint valid;
if(tid==0)atomic_store_explicit(&valid,1u,memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);float4 s00=0.0f,s01=0.0f,s10=0.0f,s11=0.0f;
for(uint base=0;base<K;base+=32){if(simd==0){uint k=base+lane;int gi=gidx[k];if(gi<0)gi+=int(G);
  if(gi<0||gi>=int(G)){atomic_store_explicit(&valid,0u,memory_order_relaxed);d0[lane]=0.0f;d1[lane]=0.0f;}
  else{uint g=uint(gi);uint4 c0=planar?pq4(qweight,k,n,N,b,true):cqw4(qweight,k,n,N,b);
    uint4 c1=planar?pq4(qweight,k,n+4,N,b,true):cqw4(qweight,k,n+4,N,b);
    uint4 z0=planar?pq4(qzeros,g,n,N,b,false):cqz4(qzeros,g,n,N,b);
    uint4 z1=planar?pq4(qzeros,g,n+4,N,b,false):cqz4(qzeros,g,n+4,N,b);
    float4 q0=float4(scales[g*N+n],scales[g*N+n+1],scales[g*N+n+2],scales[g*N+n+3]);
    float4 q1=float4(scales[g*N+n+4],scales[g*N+n+5],scales[g*N+n+6],scales[g*N+n+7]);
    d0[lane]=float4(int4(c0)-int4(z0))*q0;d1[lane]=float4(int4(c1)-int4(z1))*q1;}}
  threadgroup_barrier(mem_flags::mem_threadgroup);float4 v0=d0[lane],v1=d1[lane];
  float x0=float(x[r0*K+base+lane]);s00+=x0*v0;s01+=x0*v1;
  if(r1<M){float x1=float(x[r1*K+base+lane]);s10+=x1*v0;s11+=x1*v1;}
  threadgroup_barrier(mem_flags::mem_threadgroup);}
s00.x=simd_sum(s00.x);s00.y=simd_sum(s00.y);s00.z=simd_sum(s00.z);s00.w=simd_sum(s00.w);
s01.x=simd_sum(s01.x);s01.y=simd_sum(s01.y);s01.z=simd_sum(s01.z);s01.w=simd_sum(s01.w);
s10.x=simd_sum(s10.x);s10.y=simd_sum(s10.y);s10.z=simd_sum(s10.z);s10.w=simd_sum(s10.w);
s11.x=simd_sum(s11.x);s11.y=simd_sum(s11.y);s11.z=simd_sum(s11.z);s11.w=simd_sum(s11.w);
if(lane==0){bool ok=atomic_load_explicit(&valid,memory_order_relaxed);
  half4 v00=ok?half4(s00):half4(NAN),v01=ok?half4(s01):half4(NAN);
  out[r0*N+n]=v00.x;out[r0*N+n+1]=v00.y;out[r0*N+n+2]=v00.z;out[r0*N+n+3]=v00.w;
  out[r0*N+n+4]=v01.x;out[r0*N+n+5]=v01.y;out[r0*N+n+6]=v01.z;out[r0*N+n+7]=v01.w;
  if(r1<M){half4 v10=ok?half4(s10):half4(NAN),v11=ok?half4(s11):half4(NAN);
    out[r1*N+n]=v10.x;out[r1*N+n+1]=v10.y;out[r1*N+n+2]=v10.z;out[r1*N+n+3]=v10.w;
    out[r1*N+n+4]=v11.x;out[r1*N+n+5]=v11.y;out[r1*N+n+6]=v11.z;out[r1*N+n+7]=v11.w;}}
"""


def _kernel(*, mode: int = 1):
    global \
        _KERNEL, \
        _KERNEL_N4, \
        _KERNEL_N4_UNIFORM, \
        _KERNEL_M8, \
        _KERNEL_M8_N4, \
        _KERNEL_M8_N8, \
        _KERNEL_M16_N4, \
        _KERNEL_M16_N8, \
        _KERNEL_MULTIROW
    if mode not in (1, 4, 5, 8, 9, 10, 11, 12, 32):
        raise ValueError(f"invalid Pangolin MLX kernel mode: {mode}")
    if mode == 1:
        kernel = _KERNEL
    elif mode == 4:
        kernel = _KERNEL_N4
    elif mode == 5:
        kernel = _KERNEL_N4_UNIFORM
    elif mode == 8:
        kernel = _KERNEL_M8
    elif mode == 9:
        kernel = _KERNEL_M8_N4
    elif mode == 10:
        kernel = _KERNEL_M16_N4
    elif mode == 11:
        kernel = _KERNEL_M8_N8
    elif mode == 12:
        kernel = _KERNEL_M16_N8
    else:
        kernel = _KERNEL_MULTIROW
    if kernel is None:
        with _KERNEL_LOCK:
            if mode == 1:
                kernel = _KERNEL
            elif mode == 4:
                kernel = _KERNEL_N4
            elif mode == 5:
                kernel = _KERNEL_N4_UNIFORM
            elif mode == 8:
                kernel = _KERNEL_M8
            elif mode == 9:
                kernel = _KERNEL_M8_N4
            elif mode == 10:
                kernel = _KERNEL_M16_N4
            elif mode == 11:
                kernel = _KERNEL_M8_N8
            elif mode == 12:
                kernel = _KERNEL_M16_N8
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
                            4: "gptqmodel_pangolin_n4",
                            5: "gptqmodel_pangolin_n4_uniform",
                            8: "gptqmodel_pangolin_m8",
                            9: "gptqmodel_pangolin_m8_n4",
                            10: "gptqmodel_pangolin_m16_n4",
                            11: "gptqmodel_pangolin_m8_n8",
                            12: "gptqmodel_pangolin_m16_n8",
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
                        source={
                            1: _SOURCE,
                            4: _SOURCE_N4,
                            5: _SOURCE_N4_UNIFORM,
                            8: _SOURCE_M8,
                            9: _SOURCE_M8_N4,
                            10: _SOURCE_M16_N4,
                            11: _SOURCE_M8_N8,
                            12: _SOURCE_M16_N8,
                            32: _SOURCE_MULTIROW,
                        }[mode],
                        ensure_row_contiguous=True,
                    )
                except Exception as exc:
                    error = f"Pangolin MLX kernel creation failed: {exc}"
                    _KERNEL_ERRORS[mode] = error
                    raise RuntimeError(error) from exc
                if mode == 4:
                    _KERNEL_N4 = kernel
                elif mode == 5:
                    _KERNEL_N4_UNIFORM = kernel
                elif mode == 8:
                    _KERNEL_M8 = kernel
                elif mode == 9:
                    _KERNEL_M8_N4 = kernel
                elif mode == 10:
                    _KERNEL_M16_N4 = kernel
                elif mode == 11:
                    _KERNEL_M8_N8 = kernel
                elif mode == 12:
                    _KERNEL_M16_N8 = kernel
                elif mode == 32:
                    _KERNEL_MULTIROW = kernel
                else:
                    _KERNEL = kernel
    return kernel


@functools.lru_cache(maxsize=128)
def _dims_array(
    m: int, k: int, n: int, groups: int, bits: int, planar: bool, row_stride: int
):
    import mlx.core as mx

    return mx.array([m, k, n, groups, bits, int(planar), row_stride], dtype=mx.uint32)


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
    _g_idx_block_uniform: bool | None = None,
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
        lo_value, hi_value = mx.min(g_idx), mx.max(g_idx)
        classify_uniform = _g_idx_block_uniform is None and planar and m <= 3
        if classify_uniform:
            blocks = g_idx.reshape(-1, 32)
            uniform_value = mx.all(blocks == blocks[:, :1])
            mx.eval(lo_value, hi_value, uniform_value)
            _g_idx_block_uniform = bool(uniform_value.item())
        else:
            mx.eval(lo_value, hi_value)
        lo, hi = int(lo_value.item()), int(hi_value.item())
        if lo < -groups or hi >= groups:
            raise ValueError("g_idx contains an out-of-bounds group index")
    if _g_idx_block_uniform is None:
        _g_idx_block_uniform = False
    row_stride = 4 if m <= 16 else 8
    dims = _dims_array(m, k, n, groups, bits, planar, row_stride)
    use_n8 = n >= 1024 and (k >= 4096 or (k >= 2048 and not planar and bits in (2, 4)))
    use_m16_n8 = not planar and k >= 4096 and n >= 4096
    mode = (
        (5 if planar and _g_idx_block_uniform else 4)
        if m <= 3
        else (
            (11 if use_n8 and (not planar or m <= 5) else 9)
            if m <= 8
            else ((12 if use_m16_n8 else 10) if m <= 16 else (32 if m <= 32 else 1))
        )
    )
    group_size = 256 if mode in (8, 9, 10, 11, 12, 32) else 32
    if mode in (8, 9, 10, 11, 12, 32):
        if mode in (11, 12):
            grid_size = (n // 8) * group_size
        else:
            grid_size = (n // 4 if mode in (9, 10) else n) * group_size
    elif mode in (4, 5):
        grid_size = m * (n // 4) * 32
    else:
        grid_size = m * n * 32
    return _kernel(mode=mode)(
        inputs=[x, qweight, scales, qzeros, g_idx, dims],
        template=[("T", mx.float16)],
        grid=(grid_size, 1, 1),
        threadgroup=(group_size, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float16],
    )[0]


__all__ = ["PANGOLIN_MLX_BITS", "pangolin_mlx_gemv"]
