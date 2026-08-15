# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fused planar QVQ inner GEMV for MLX."""

from __future__ import annotations

import functools
import operator
import threading
from dataclasses import dataclass
from typing import Any

try:
    import mlx.nn as _mlx_nn
except ModuleNotFoundError:  # pragma: no cover - MLX is optional off Apple hosts.
    _mlx_nn = None

from ..quantization.qvq_codecs import (
    PGC16_CODEBOOK_VERSION,
    PGC16_V2B4_BANK_XOR_MASKS_BY_TRANSITION_BITS,
    PGC16_V4_BANK_XOR_MASKS_BY_TRANSITION_BITS,
    pgc16_levels_for_version,
)
from ..quantization.qvq_rates import (
    QVQ_BITS,
    normalize_qvq_rate,
    qvq_transition_bits,
    qvq_words_per_tile,
)

QVQ_MLX_BITS = QVQ_BITS
_KERNEL: Any | None = None
_KERNEL_ERROR: str | None = None
_DUAL_V2_KERNEL: Any | None = None
_DUAL_V2_KERNEL_ERROR: str | None = None
_KERNEL_LOCK = threading.Lock()
_V4_KERNEL: Any | None = None
_V4_KERNEL_ERROR: str | None = None
_V4_N32_KERNELS: dict[str, Any] = {}
_V4_N32_KERNEL_ERRORS: dict[str, str] = {}
_V4_N4_KERNELS: dict[str, Any] = {}
_V4_N4_KERNEL_ERRORS: dict[str, str] = {}
_V4_MMA_KERNELS: dict[str, Any] = {}
_V4_MMA_KERNEL_ERRORS: dict[str, str] = {}
_V4_BANKED_KERNEL: Any | None = None
_V4_BANKED_KERNEL_ERROR: str | None = None
_V4_BANKED_MULTIROW_KERNELS: dict[str, Any] = {}
_V4_BANKED_MULTIROW_KERNEL_ERRORS: dict[str, str] = {}
_V4_E4_KERNEL: Any | None = None
_V4_E4_KERNEL_ERROR: str | None = None
_V4_MULTIROW_KERNEL: Any | None = None
_V4_MULTIROW_KERNEL_ERROR: str | None = None
_MULTIROW_KERNEL: Any | None = None
_MULTIROW_KERNEL_ERROR: str | None = None
_MULTIROW_N4_KERNEL: Any | None = None
_MULTIROW_N4_KERNEL_ERROR: str | None = None
_MULTIROW_N8_KERNEL: Any | None = None
_MULTIROW_N8_KERNEL_ERROR: str | None = None
_HYB_REFERENCE_KERNEL: Any | None = None
_HYB_REFERENCE_KERNEL_ERROR: str | None = None
_HYB_REFERENCE_KERNEL_LOCK = threading.Lock()
_PGC16_LEVELS_HOT: tuple[str, Any] | None = None
_VITERBI_KERNEL: Any | None = None
_VITERBI_KERNEL_ERROR: str | None = None
_FP32_KERNELS: dict[str, Any] = {}
_FP32_KERNEL_ERRORS: dict[str, str] = {}
_V2_BANKED_KERNELS: dict[tuple[str, int, bool], Any] = {}
_V2_BANKED_KERNEL_ERRORS: dict[tuple[str, int, bool], str] = {}


@dataclass(frozen=True)
class _QVQMLXPreparedCompander:
    levels: Any


def _integer_argument(value: int, name: str) -> int:
    """Reject lossy ``int(...)`` coercions at the native-kernel boundary."""

    if isinstance(value, bool):
        raise TypeError(f"QVQ MLX {name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"QVQ MLX {name} must be an integer") from exc


def _qvq_v4_bank_masks_metal(name: str) -> str:
    rows = (
        "  {" + ",".join(f"0x{mask:04x}u" for mask in PGC16_V4_BANK_XOR_MASKS_BY_TRANSITION_BITS[bits]) + "},"
        for bits in range(4, 17, 2)
    )
    return f"constant ushort {name}[7][4]={{\n" + "\n".join(rows) + "\n};"


def _qvq_v2_bank_masks_metal(name: str) -> str:
    rows = (
        "  {" + ",".join(f"0x{mask:04x}u" for mask in PGC16_V2B4_BANK_XOR_MASKS_BY_TRANSITION_BITS[bits]) + "},"
        for bits in range(2, 6)
    )
    return f"constant ushort {name}[4][4]={{\n" + "\n".join(rows) + "\n};"


_QVQ_V4_BANK_MASKS_METAL = _qvq_v4_bank_masks_metal("qbank_masks")
_QVQ_V2_BANK_MASKS_METAL = _qvq_v2_bank_masks_metal("qv2bank_masks")


_HEADER = r"""
inline uint qpw(uint remaining) {
  if(remaining>=16)return 16;if(remaining>=8)return 8;if(remaining>=4)return 4;if(remaining>=2)return 2;return 1;
}
inline uint qpt(device const int* p,uint edge,uint eb){uint block=edge>>5,lane=edge&31,base=block*eb;
  uint rem=eb,row=0,off=0,v=0;for(uint plane=0;plane<4&&rem;++plane){uint w=qpw(rem),pf=32/w;
    uint word=as_type<uint>(p[base+row+lane/pf]),code=(word>>(w*(lane%pf)))&((1u<<w)-1u);
    v|=code<<off;rem-=w;row+=w;off+=w;}return v;}
inline uint qstate(device const int* tile,uint pair,uint eb){uint count=(15+eb)/eb,first=(pair+128-count+1)&127,s=0;
  for(uint j=0;j<count;++j){uint edge=(first+j)&127;s=((s<<eb)|qpt(tile,edge,eb))&0xffffu;}return s;}
inline uint qstated(device const int* tile,uint pair,uint eb){uint chain=pair&1u,step=pair>>1;
  uint count=(15+eb)/eb,first=(step+64-count+1)&63,s=0;
  for(uint j=0;j<count;++j){uint edge=(((first+j)&63u)<<1)|chain;
    s=((s<<eb)|qpt(tile,edge,eb))&0xffffu;}return s;}
inline uint qstate4(device const int* tile,uint vec,uint eb){uint count=(15+eb)/eb,first=(vec+64-count+1)&63,s=0;
  for(uint j=0;j<count;++j){uint edge=(first+j)&63;s=((s<<eb)|qpt(tile,edge,eb))&0xffffu;}return s;}
inline uint qstate4l18(device const int* tile,uint vec,uint eb){uint count=(17+eb)/eb,first=(vec+64-count+1)&63,s=0;
  for(uint j=0;j<count;++j){uint edge=(first+j)&63;s=((s<<eb)|qpt(tile,edge,eb))&0x3ffffu;}return s;}
inline uint qstate4e4(device const int* tile,uint vec){uint first=(vec+61)&63,s=0;
  for(uint j=0;j<4;++j){uint edge=(first+j)&63,block=edge>>5,lane=edge&31;
    uint word=as_type<uint>(tile[block*4+lane/8]);s=(s<<4)|((word>>(4*(lane&7)))&15u);}return s;}
inline uint qpt4(device const int* tile,uint edge){uint block=edge>>5,lane=edge&31;
  uint word=as_type<uint>(tile[block*4+lane/8]);return(word>>(4*(lane&7)))&15u;}
inline float2 qlevels(device const half* levels,uint s){uint p=s^(s>>8);p=(p*40503u+17011u)&0xffffu;p^=p>>7;
  return float2(float(levels[p>>8]),float(levels[p&255u]));}
inline float4 qlevels4(device const half* levels,uint s){uint p0=s^(s>>8);p0=(p0*40503u+17011u)&0xffffu;p0^=p0>>7;
  uint p1=s^0xa5a5u;p1^=p1>>8;p1=(p1*40503u+17011u)&0xffffu;p1^=p1>>7;
  return float4(float(levels[p0>>8]),float(levels[p0&255u]),float(levels[p1>>8]),float(levels[p1&255u]));}
__QVQ_V4_BANK_MASKS_METAL__
__QVQ_V2_BANK_MASKS_METAL__
inline float2 qlevelsv2b(device const half* levels,uint s,uint bank,uint eb){
  uint p=s^uint(qv2bank_masks[eb-2u][bank]);p^=p>>8;p=(p*40503u+17011u)&0xffffu;p^=p>>7;
  return float2(float(levels[p>>8]),float(levels[p&255u]));}
inline uint qbank(constant const uchar* ids,uint tile){return(uint(ids[tile>>2])>>((tile&3u)<<1))&3u;}
inline uint qbank(device const uchar* ids,uint tile){return(uint(ids[tile>>2])>>((tile&3u)<<1))&3u;}
inline float4 qlevels4b(device const half* levels,uint s,uint bank,uint eb){uint p0=s^(s>>8);
  p0=(p0*40503u+17011u)&0xffffu;p0^=p0>>7;uint p1=s^uint(qbank_masks[(eb-4u)>>1][bank]);
  p1^=p1>>8;p1=(p1*40503u+17011u)&0xffffu;p1^=p1>>7;
  return float4(float(levels[p0>>8]),float(levels[p0&255u]),float(levels[p1>>8]),float(levels[p1&255u]));}
inline float4 qlevels4l18(device const half* levels,uint s,uint eb){
  return qlevels4b(levels,s&0xffffu,s>>16,eb);}
inline float2 qpair(device const int* t,device const half* levels,uint k,uint n,uint N,uint eb){
  uint tile=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),s=qstate(t+tile*(4*eb),local>>1,eb);
  return qlevels(levels,s);}
inline uint qv2b4(constant const uchar* ids,uint tile,uint pair){return(uint(ids[tile])>>((pair>>5)<<1))&3u;}
inline uint qv2b4(device const uchar* ids,uint tile,uint pair){return(uint(ids[tile])>>((pair>>5)<<1))&3u;}
inline uint qv2b2(constant const uchar* ids,constant const uchar* alt,uint tile,uint pair){
  return((uint(ids[tile])>>(pair>>4))&1u)*uint(alt[0]);}
inline uint qv2b2(device const uchar* ids,constant const uchar* alt,uint tile,uint pair){
  return((uint(ids[tile])>>(pair>>4))&1u)*uint(alt[0]);}
inline float2 qpairv2b4(device const int* t,constant const uchar* ids,device const half* levels,
    uint k,uint n,uint N,uint eb){uint tile=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),pair=local>>1;
  uint s=qstate(t+tile*(4*eb),pair,eb);return qlevelsv2b(levels,s,qv2b4(ids,tile,pair),eb);}
inline float2 qpairv2b4(device const int* t,device const uchar* ids,device const half* levels,
    uint k,uint n,uint N,uint eb){uint tile=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),pair=local>>1;
  uint s=qstate(t+tile*(4*eb),pair,eb);return qlevelsv2b(levels,s,qv2b4(ids,tile,pair),eb);}
inline float2 qpairv2b2(device const int* t,constant const uchar* ids,constant const uchar* alt,
    device const half* levels,uint k,uint n,uint N,uint eb){
  uint tile=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),pair=local>>1;
  uint s=qstate(t+tile*(4*eb),pair,eb);return qlevelsv2b(levels,s,qv2b2(ids,alt,tile,pair),eb);}
inline float2 qpairv2b2(device const int* t,device const uchar* ids,constant const uchar* alt,
    device const half* levels,uint k,uint n,uint N,uint eb){
  uint tile=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),pair=local>>1;
  uint s=qstate(t+tile*(4*eb),pair,eb);return qlevelsv2b(levels,s,qv2b2(ids,alt,tile,pair),eb);}
inline float2 qpaird(device const int* t,device const half* levels,uint k,uint n,uint N,uint eb){
  uint tile=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),s=qstated(t+tile*(4*eb),local>>1,eb);
  return qlevels(levels,s);}
inline float4 qquad(device const int* t,device const half* levels,uint k,uint n,uint N,uint eb){
  uint ti=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),pair=local>>1;
  device const int* tile=t+ti*(4*eb);uint s0=qstate(tile,pair,eb),next=pair+1;
  uint s1=((s0<<eb)|qpt(tile,next,eb))&0xffffu;
  return float4(qlevels(levels,s0),qlevels(levels,s1));}
inline float4 qquadv2b4(device const int* t,device const uchar* ids,device const half* levels,
    uint k,uint n,uint N,uint eb){uint ti=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),pair=local>>1;
  device const int* tile=t+ti*(4*eb);uint s0=qstate(tile,pair,eb),s1=((s0<<eb)|qpt(tile,pair+1,eb))&0xffffu;
  uint bank=qv2b4(ids,ti,pair);return float4(qlevelsv2b(levels,s0,bank,eb),qlevelsv2b(levels,s1,bank,eb));}
inline float4 qquadv2b4(device const int* t,constant const uchar* ids,device const half* levels,
    uint k,uint n,uint N,uint eb){uint ti=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),pair=local>>1;
  device const int* tile=t+ti*(4*eb);uint s0=qstate(tile,pair,eb),s1=((s0<<eb)|qpt(tile,pair+1,eb))&0xffffu;
  uint bank=qv2b4(ids,ti,pair);return float4(qlevelsv2b(levels,s0,bank,eb),qlevelsv2b(levels,s1,bank,eb));}
inline float4 qquadv2b2(device const int* t,device const uchar* ids,constant const uchar* alt,
    device const half* levels,uint k,uint n,uint N,uint eb){
  uint ti=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),pair=local>>1;
  device const int* tile=t+ti*(4*eb);uint s0=qstate(tile,pair,eb),s1=((s0<<eb)|qpt(tile,pair+1,eb))&0xffffu;
  uint bank=qv2b2(ids,alt,ti,pair);
  return float4(qlevelsv2b(levels,s0,bank,eb),qlevelsv2b(levels,s1,bank,eb));}
inline float4 qquadv2b2(device const int* t,constant const uchar* ids,constant const uchar* alt,
    device const half* levels,uint k,uint n,uint N,uint eb){
  uint ti=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),pair=local>>1;
  device const int* tile=t+ti*(4*eb);uint s0=qstate(tile,pair,eb),s1=((s0<<eb)|qpt(tile,pair+1,eb))&0xffffu;
  uint bank=qv2b2(ids,alt,ti,pair);
  return float4(qlevelsv2b(levels,s0,bank,eb),qlevelsv2b(levels,s1,bank,eb));}
inline float4 qquad4(device const int* t,device const half* levels,uint k,uint n,uint N,uint eb){
  uint ti=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),s=qstate4(t+ti*(2*eb),local>>2,eb);
  return qlevels4(levels,s);}
inline float4 qquad4l18(device const int* t,device const half* levels,uint k,uint n,uint N,uint eb){
  uint ti=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),s=qstate4l18(t+ti*(2*eb),local>>2,eb);
  return qlevels4l18(levels,s,eb);}
inline float4 qquad4b(device const int* t,constant const uchar* ids,device const half* levels,
    uint k,uint n,uint N,uint eb){uint ti=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15);
  uint s=qstate4(t+ti*(2*eb),local>>2,eb);return qlevels4b(levels,s,qbank(ids,ti),eb);}
inline float4 qquad4b(device const int* t,device const uchar* ids,device const half* levels,
    uint k,uint n,uint N,uint eb){uint ti=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15);
  uint s=qstate4(t+ti*(2*eb),local>>2,eb);return qlevels4b(levels,s,qbank(ids,ti),eb);}
inline void qwide4(device const int* t,device const half* levels,uint k,uint n,uint N,uint eb,
    thread float4& v0,thread float4& v1,thread float4& v2,thread float4& v3){
  uint ti=(k>>4)*(N>>4)+(n>>4),vi=((k&15)*16)>>2;device const int* tile=t+ti*(2*eb);
  uint s0=qstate4(tile,vi,eb),s1=((s0<<eb)|qpt(tile,vi+1,eb))&0xffffu;
  uint s2=((s1<<eb)|qpt(tile,vi+2,eb))&0xffffu,s3=((s2<<eb)|qpt(tile,vi+3,eb))&0xffffu;
  v0=qlevels4(levels,s0);v1=qlevels4(levels,s1);v2=qlevels4(levels,s2);v3=qlevels4(levels,s3);}
inline void qwide4e4(device const int* t,device const half* levels,uint k,uint n,uint N,
    thread float4& v0,thread float4& v1,thread float4& v2,thread float4& v3){
  uint ti=(k>>4)*(N>>4)+(n>>4),vi=((k&15)*16)>>2;device const int* tile=t+ti*8;
  uint s0=qstate4e4(tile,vi),s1=((s0<<4)|qpt4(tile,vi+1))&0xffffu;
  uint s2=((s1<<4)|qpt4(tile,vi+2))&0xffffu,s3=((s2<<4)|qpt4(tile,vi+3))&0xffffu;
  v0=qlevels4(levels,s0);v1=qlevels4(levels,s1);v2=qlevels4(levels,s2);v3=qlevels4(levels,s3);}
inline void qwide4b(device const int* t,constant const uchar* ids,device const half* levels,
    uint k,uint n,uint N,uint eb,thread float4& v0,thread float4& v1,thread float4& v2,thread float4& v3){
  uint ti=(k>>4)*(N>>4)+(n>>4),vi=((k&15)*16)>>2,bank=qbank(ids,ti);
  device const int* tile=t+ti*(2*eb);uint s0=qstate4(tile,vi,eb),s1=((s0<<eb)|qpt(tile,vi+1,eb))&0xffffu;
  uint s2=((s1<<eb)|qpt(tile,vi+2,eb))&0xffffu,s3=((s2<<eb)|qpt(tile,vi+3,eb))&0xffffu;
  v0=qlevels4b(levels,s0,bank,eb);v1=qlevels4b(levels,s1,bank,eb);
  v2=qlevels4b(levels,s2,bank,eb);v3=qlevels4b(levels,s3,bank,eb);}
inline void qwide4b(device const int* t,device const uchar* ids,device const half* levels,
    uint k,uint n,uint N,uint eb,thread float4& v0,thread float4& v1,thread float4& v2,thread float4& v3){
  uint ti=(k>>4)*(N>>4)+(n>>4),vi=((k&15)*16)>>2,bank=qbank(ids,ti);
  device const int* tile=t+ti*(2*eb);uint s0=qstate4(tile,vi,eb),s1=((s0<<eb)|qpt(tile,vi+1,eb))&0xffffu;
  uint s2=((s1<<eb)|qpt(tile,vi+2,eb))&0xffffu,s3=((s2<<eb)|qpt(tile,vi+3,eb))&0xffffu;
  v0=qlevels4b(levels,s0,bank,eb);v1=qlevels4b(levels,s1,bank,eb);
  v2=qlevels4b(levels,s2,bank,eb);v3=qlevels4b(levels,s3,bank,eb);}
inline float4 qquad4e4(device const int* t,device const half* levels,uint k,uint n,uint N){
  uint ti=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),s=qstate4e4(t+ti*8,local>>2);
  return qlevels4(levels,s);}
inline void qoctet(device const int* t,device const half* levels,uint k,uint n,uint N,uint eb,
    threadgroup float4& v0,threadgroup float4& v1){
  uint ti=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),pair=local>>1;
  device const int* tile=t+ti*(4*eb);uint s0=qstate(tile,pair,eb),next=pair+1;
  uint s1=((s0<<eb)|qpt(tile,next,eb))&0xffffu;
  ++next;uint s2=((s1<<eb)|qpt(tile,next,eb))&0xffffu;
  ++next;uint s3=((s2<<eb)|qpt(tile,next,eb))&0xffffu;
  v0=float4(qlevels(levels,s0),qlevels(levels,s1));v1=float4(qlevels(levels,s2),qlevels(levels,s3));}
inline void qoctetv2b4(device const int* t,device const uchar* ids,device const half* levels,
    uint k,uint n,uint N,uint eb,threadgroup float4& v0,threadgroup float4& v1){
  uint ti=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),pair=local>>1;
  device const int* tile=t+ti*(4*eb);uint s0=qstate(tile,pair,eb),s1=((s0<<eb)|qpt(tile,pair+1,eb))&0xffffu;
  uint s2=((s1<<eb)|qpt(tile,pair+2,eb))&0xffffu,s3=((s2<<eb)|qpt(tile,pair+3,eb))&0xffffu;
  uint bank=qv2b4(ids,ti,pair);v0=float4(qlevelsv2b(levels,s0,bank,eb),qlevelsv2b(levels,s1,bank,eb));
  v1=float4(qlevelsv2b(levels,s2,bank,eb),qlevelsv2b(levels,s3,bank,eb));}
inline void qoctetv2b4(device const int* t,constant const uchar* ids,device const half* levels,
    uint k,uint n,uint N,uint eb,threadgroup float4& v0,threadgroup float4& v1){
  uint ti=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),pair=local>>1;
  device const int* tile=t+ti*(4*eb);uint s0=qstate(tile,pair,eb),s1=((s0<<eb)|qpt(tile,pair+1,eb))&0xffffu;
  uint s2=((s1<<eb)|qpt(tile,pair+2,eb))&0xffffu,s3=((s2<<eb)|qpt(tile,pair+3,eb))&0xffffu;
  uint bank=qv2b4(ids,ti,pair);v0=float4(qlevelsv2b(levels,s0,bank,eb),qlevelsv2b(levels,s1,bank,eb));
  v1=float4(qlevelsv2b(levels,s2,bank,eb),qlevelsv2b(levels,s3,bank,eb));}
inline void qoctetv2b2(device const int* t,device const uchar* ids,constant const uchar* alt,
    device const half* levels,uint k,uint n,uint N,uint eb,threadgroup float4& v0,threadgroup float4& v1){
  uint ti=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),pair=local>>1;
  device const int* tile=t+ti*(4*eb);uint s0=qstate(tile,pair,eb),s1=((s0<<eb)|qpt(tile,pair+1,eb))&0xffffu;
  uint s2=((s1<<eb)|qpt(tile,pair+2,eb))&0xffffu,s3=((s2<<eb)|qpt(tile,pair+3,eb))&0xffffu;
  uint bank=qv2b2(ids,alt,ti,pair);v0=float4(qlevelsv2b(levels,s0,bank,eb),qlevelsv2b(levels,s1,bank,eb));
  v1=float4(qlevelsv2b(levels,s2,bank,eb),qlevelsv2b(levels,s3,bank,eb));}
inline void qoctetv2b2(device const int* t,constant const uchar* ids,constant const uchar* alt,
    device const half* levels,uint k,uint n,uint N,uint eb,threadgroup float4& v0,threadgroup float4& v1){
  uint ti=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),pair=local>>1;
  device const int* tile=t+ti*(4*eb);uint s0=qstate(tile,pair,eb),s1=((s0<<eb)|qpt(tile,pair+1,eb))&0xffffu;
  uint s2=((s1<<eb)|qpt(tile,pair+2,eb))&0xffffu,s3=((s2<<eb)|qpt(tile,pair+3,eb))&0xffffu;
  uint bank=qv2b2(ids,alt,ti,pair);v0=float4(qlevelsv2b(levels,s0,bank,eb),qlevelsv2b(levels,s1,bank,eb));
  v1=float4(qlevelsv2b(levels,s2,bank,eb),qlevelsv2b(levels,s3,bank,eb));}
inline float qhyb(device const int* t,device const half* lut,uint k,uint n,uint N,uint eb){
  uint tile=(k>>4)*(N>>4)+(n>>4),local=(k&15)*16+(n&15),s=qstate(t+tile*(4*eb),local>>1,eb);
  uint h=s*s+s,li=(h>>6)&511u;float v=float(lut[li*2+(local&1)]);
  return((local&1)&&(h&(1u<<15)))?-v:v;}
""".replace("__QVQ_V4_BANK_MASKS_METAL__", _QVQ_V4_BANK_MASKS_METAL).replace(
    "__QVQ_V2_BANK_MASKS_METAL__", _QVQ_V2_BANK_MASKS_METAL
)

_SOURCE = r"""
uint group=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint M=dims[0],K=dims[1],N=dims[2],eb=EdgeBits,pc=N>>1,m=group/pc,pair=group-m*pc,n=pair<<1;
float sum0=0.0f,sum1=0.0f;
for(uint k=lane;k<K;k+=32){float input=float(x[m*K+k]);float2 v=qpair(trellis,levels,k,n,N,eb);
  sum0+=input*v.x;sum1+=input*v.y;}
sum0=simd_sum(sum0);sum1=simd_sum(sum1);
if(lane==0){out[m*N+n]=half(sum0);out[m*N+n+1]=half(sum1);}
"""

_V4_SOURCE = r"""
uint group=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint M=dims[0],K=dims[1],N=dims[2],eb=EdgeBits,vc=N>>4,m=group/vc,vec=group-m*vc,n=vec<<4;
float4 s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;
for(uint k=lane;k<K;k+=32){float4 v0,v1,v2,v3;qwide4(trellis,levels,k,n,N,eb,v0,v1,v2,v3);
  float input=float(x[m*K+k]);s0+=input*v0;s1+=input*v1;s2+=input*v2;s3+=input*v3;}
#define QVQ_REDUCE(V) V.x=simd_sum(V.x);V.y=simd_sum(V.y);V.z=simd_sum(V.z);V.w=simd_sum(V.w);
QVQ_REDUCE(s0) QVQ_REDUCE(s1) QVQ_REDUCE(s2) QVQ_REDUCE(s3)
#undef QVQ_REDUCE
if(lane==0){*reinterpret_cast<device half4*>(out+m*N+n)=half4(s0);
  *reinterpret_cast<device half4*>(out+m*N+n+4)=half4(s1);
  *reinterpret_cast<device half4*>(out+m*N+n+8)=half4(s2);
  *reinterpret_cast<device half4*>(out+m*N+n+12)=half4(s3);}
"""

_V4_N4_SOURCE = r"""
uint group=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint M=dims[0],K=dims[1],N=dims[2],eb=EdgeBits,vc=N>>2,m=group/vc,vec=group-m*vc,n=vec<<2;
float4 sum=0.0f;
for(uint k=lane;k<K;k+=32){sum+=float(x[m*K+k])*qquad4(trellis,levels,k,n,N,eb);}
sum.x=simd_sum(sum.x);sum.y=simd_sum(sum.y);sum.z=simd_sum(sum.z);sum.w=simd_sum(sum.w);
if(lane==0){*reinterpret_cast<device half4*>(out+m*N+n)=half4(sum);}
"""

_V4_N4_E4_SOURCE = _V4_N4_SOURCE.replace("qquad4(trellis,levels,k,n,N,eb", "qquad4e4(trellis,levels,k,n,N")
_V4_N4_BANKED_SOURCE = _V4_N4_SOURCE.replace(
    "qquad4(trellis,levels,", "qquad4b(trellis,bank_ids,levels,"
)
_V4_N4_L18_SOURCE = _V4_N4_SOURCE.replace("qquad4(trellis,levels,", "qquad4l18(trellis,levels,")

_FP32_SOURCE = _SOURCE.replace(
    "out[m*N+n]=half(sum0);out[m*N+n+1]=half(sum1);",
    "out[m*N+n]=sum0;out[m*N+n+1]=sum1;",
)
_DUAL_V2_SOURCE = _SOURCE.replace("qpair(trellis,levels,", "qpaird(trellis,levels,")
_DUAL_V2_FP32_SOURCE = _FP32_SOURCE.replace("qpair(trellis,levels,", "qpaird(trellis,levels,")


def _v2_banked_source(source: str, kind: str) -> str:
    if kind == "v2b4_p64":
        return (
            source.replace("qpair(trellis,levels,", "qpairv2b4(trellis,bank_ids,levels,")
            .replace("qquad(trellis,levels,", "qquadv2b4(trellis,bank_ids,levels,")
            .replace("qoctet(trellis,levels,", "qoctetv2b4(trellis,bank_ids,levels,")
        )
    if kind == "v2b2_p32":
        return (
            source.replace(
                "qpair(trellis,levels,",
                "qpairv2b2(trellis,bank_ids,bank_alt_id,levels,",
            )
            .replace(
                "qquad(trellis,levels,",
                "qquadv2b2(trellis,bank_ids,bank_alt_id,levels,",
            )
            .replace(
                "qoctet(trellis,levels,",
                "qoctetv2b2(trellis,bank_ids,bank_alt_id,levels,",
            )
        )
    raise ValueError(f"unknown QVQ banked-V2 kernel kind: {kind}")


_V2B4_P64_SOURCE = _v2_banked_source(_SOURCE, "v2b4_p64")
_V2B2_P32_SOURCE = _v2_banked_source(_SOURCE, "v2b2_p32")
_V2B4_P64_FP32_SOURCE = _v2_banked_source(_FP32_SOURCE, "v2b4_p64")
_V2B2_P32_FP32_SOURCE = _v2_banked_source(_FP32_SOURCE, "v2b2_p32")
_V4_N4_FP32_SOURCE = _V4_N4_SOURCE.replace(
    "*reinterpret_cast<device half4*>(out+m*N+n)=half4(sum);",
    "*reinterpret_cast<device float4*>(out+m*N+n)=sum;",
)
_V4_N4_BANKED_FP32_SOURCE = _V4_N4_FP32_SOURCE.replace(
    "qquad4(trellis,levels,", "qquad4b(trellis,bank_ids,levels,"
)
_V4_N4_L18_FP32_SOURCE = _V4_N4_FP32_SOURCE.replace("qquad4(trellis,levels,", "qquad4l18(trellis,levels,")

_V4_BANKED_SOURCE = _V4_SOURCE.replace("qwide4(trellis,levels,", "qwide4b(trellis,bank_ids,levels,")

_V4_E4_SOURCE = _V4_SOURCE.replace("qwide4(trellis,levels,k,n,N,eb", "qwide4e4(trellis,levels,k,n,N")

_V4_N32_SOURCE = r"""
uint group=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint M=dims[0],K=dims[1],N=dims[2],eb=EdgeBits,vc=N>>5,m=group/vc,vec=group-m*vc,n=vec<<5;
float4 s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f,s4=0.0f,s5=0.0f,s6=0.0f,s7=0.0f;
for(uint k=lane;k<K;k+=32){float4 v0,v1,v2,v3,v4,v5,v6,v7;
  qwide4(trellis,levels,k,n,N,eb,v0,v1,v2,v3);
  qwide4(trellis,levels,k,n+16,N,eb,v4,v5,v6,v7);float input=float(x[m*K+k]);
  s0+=input*v0;s1+=input*v1;s2+=input*v2;s3+=input*v3;
  s4+=input*v4;s5+=input*v5;s6+=input*v6;s7+=input*v7;}
#define QVQ_REDUCE(V) V.x=simd_sum(V.x);V.y=simd_sum(V.y);V.z=simd_sum(V.z);V.w=simd_sum(V.w);
QVQ_REDUCE(s0) QVQ_REDUCE(s1) QVQ_REDUCE(s2) QVQ_REDUCE(s3)
QVQ_REDUCE(s4) QVQ_REDUCE(s5) QVQ_REDUCE(s6) QVQ_REDUCE(s7)
#undef QVQ_REDUCE
if(lane==0){*reinterpret_cast<device half4*>(out+m*N+n)=half4(s0);
  *reinterpret_cast<device half4*>(out+m*N+n+4)=half4(s1);
  *reinterpret_cast<device half4*>(out+m*N+n+8)=half4(s2);
  *reinterpret_cast<device half4*>(out+m*N+n+12)=half4(s3);
  *reinterpret_cast<device half4*>(out+m*N+n+16)=half4(s4);
  *reinterpret_cast<device half4*>(out+m*N+n+20)=half4(s5);
  *reinterpret_cast<device half4*>(out+m*N+n+24)=half4(s6);
  *reinterpret_cast<device half4*>(out+m*N+n+28)=half4(s7);}
"""

_V4_N32_BANKED_SOURCE = _V4_N32_SOURCE.replace(
    "qwide4(trellis,levels,", "qwide4b(trellis,bank_ids,levels,"
)
_V4_N32_E4_SOURCE = _V4_N32_SOURCE.replace(
    "qwide4(trellis,levels,k,", "qwide4e4(trellis,levels,k,"
).replace(",N,eb,v", ",N,v")

_V4_MMA_SOURCE = r"""
uint group=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint M=dims[0],K=dims[1],N=dims[2],eb=EdgeBits,nb=N>>3,row0=(group/nb)<<3,n=(group%nb)<<3;
uint qid=lane>>2,fm=(qid&4u)+((lane>>1)&3u),fn=(qid&2u)*2u+(lane&1u)*2u;
metal::simdgroup_matrix<float,8,8> C,D;C.thread_elements()[0]=0.0f;C.thread_elements()[1]=0.0f;
for(uint base=0;base<K;base+=8){metal::simdgroup_matrix<half,8,8>A,B;uint ar=row0+fm,ak=base+fn;
  A.thread_elements()[0]=ar<M?x[ar*K+ak]:half(0.0h);
  A.thread_elements()[1]=ar<M?x[ar*K+ak+1]:half(0.0h);uint bk=base+fm;
  float4 v=qquad4(trellis,levels,bk,n+(fn&~3u),N,eb);
  B.thread_elements()[0]=half((fn&2u)?v.z:v.x);B.thread_elements()[1]=half((fn&2u)?v.w:v.y);
  simdgroup_multiply_accumulate(D,A,B,C);C=D;}
if(row0+fm<M){out[(row0+fm)*N+n+fn]=half(C.thread_elements()[0]);
  out[(row0+fm)*N+n+fn+1]=half(C.thread_elements()[1]);}
"""

_V4_MMA_BANKED_SOURCE = _V4_MMA_SOURCE.replace(
    "qquad4(trellis,levels,", "qquad4b(trellis,bank_ids,levels,"
)
_V4_MMA_E4_SOURCE = _V4_MMA_SOURCE.replace(
    "qquad4(trellis,levels,bk,n+(fn&~3u),N,eb", "qquad4e4(trellis,levels,bk,n+(fn&~3u),N"
)
_V4_MMA_HEADER = "#include <metal_simdgroup_matrix>\n" + _HEADER

_V4_MULTIROW_SOURCE = r"""
uint group=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint simd=simdgroup_index_in_threadgroup;
uint M=dims[0],K=dims[1],N=dims[2],eb=EdgeBits,row_tile=dims[4],vc=N>>4;
uint row_block=group/vc,vec=group-row_block*vc,row=row_block*row_tile+simd,n=vec<<4;
threadgroup float4 d0[32],d1[32],d2[32],d3[32];float4 s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;
for(uint base=0;base<K;base+=32){uint k=base+lane;
  if(simd==0){if(k<K){float4 v0,v1,v2,v3;qwide4(trellis,levels,k,n,N,eb,v0,v1,v2,v3);
      d0[lane]=v0;d1[lane]=v1;d2[lane]=v2;d3[lane]=v3;}
    else{d0[lane]=0.0f;d1[lane]=0.0f;d2[lane]=0.0f;d3[lane]=0.0f;}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(k<K&&row<M){float input=float(x[row*K+k]);s0+=input*d0[lane];s1+=input*d1[lane];
    s2+=input*d2[lane];s3+=input*d3[lane];}
  threadgroup_barrier(mem_flags::mem_threadgroup);}
s0.x=simd_sum(s0.x);s0.y=simd_sum(s0.y);s0.z=simd_sum(s0.z);s0.w=simd_sum(s0.w);
s1.x=simd_sum(s1.x);s1.y=simd_sum(s1.y);s1.z=simd_sum(s1.z);s1.w=simd_sum(s1.w);
s2.x=simd_sum(s2.x);s2.y=simd_sum(s2.y);s2.z=simd_sum(s2.z);s2.w=simd_sum(s2.w);
s3.x=simd_sum(s3.x);s3.y=simd_sum(s3.y);s3.z=simd_sum(s3.z);s3.w=simd_sum(s3.w);
if(lane==0&&row<M){*reinterpret_cast<device half4*>(out+row*N+n)=half4(s0);
  *reinterpret_cast<device half4*>(out+row*N+n+4)=half4(s1);
  *reinterpret_cast<device half4*>(out+row*N+n+8)=half4(s2);
  *reinterpret_cast<device half4*>(out+row*N+n+12)=half4(s3);}
"""

_MULTIROW_SOURCE = r"""
uint group=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint simd=simdgroup_index_in_threadgroup;
uint M=dims[0],K=dims[1],N=dims[2],eb=EdgeBits,row_tile=dims[4],pc=N>>1;
uint row_block=group/pc,pair=group-row_block*pc,row_base=row_block*row_tile,n=pair<<1;
uint simdgroups=row_tile>>2,r0=row_base+simd,r1=r0+simdgroups,r2=r1+simdgroups,r3=r2+simdgroups;
threadgroup float2 decoded[32];float2 s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;
for(uint base=0;base<K;base+=32){uint k=base+lane;
  if(simd==0)decoded[lane]=k<K?qpair(trellis,levels,k,n,N,eb):float2(0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(k<K){float2 v=decoded[lane];if(r0<M)s0+=float(x[r0*K+k])*v;if(r1<M)s1+=float(x[r1*K+k])*v;
    if(r2<M)s2+=float(x[r2*K+k])*v;if(r3<M)s3+=float(x[r3*K+k])*v;}
  threadgroup_barrier(mem_flags::mem_threadgroup);}
s0.x=simd_sum(s0.x);s0.y=simd_sum(s0.y);s1.x=simd_sum(s1.x);s1.y=simd_sum(s1.y);
s2.x=simd_sum(s2.x);s2.y=simd_sum(s2.y);s3.x=simd_sum(s3.x);s3.y=simd_sum(s3.y);
if(lane==0){if(r0<M){half2 v=half2(s0);out[r0*N+n]=v.x;out[r0*N+n+1]=v.y;}
  if(r1<M){half2 v=half2(s1);out[r1*N+n]=v.x;out[r1*N+n+1]=v.y;}
  if(r2<M){half2 v=half2(s2);out[r2*N+n]=v.x;out[r2*N+n+1]=v.y;}
  if(r3<M){half2 v=half2(s3);out[r3*N+n]=v.x;out[r3*N+n+1]=v.y;}}
"""

_MULTIROW_N4_SOURCE = r"""
uint group=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint simd=simdgroup_index_in_threadgroup;
uint M=dims[0],K=dims[1],N=dims[2],eb=EdgeBits,row_tile=dims[4],vc=N>>2;
uint row_block=group/vc,vector=group-row_block*vc,row_base=row_block*row_tile,n=vector<<2;
uint simdgroups=row_tile>>2,r0=row_base+simd,r1=r0+simdgroups,r2=r1+simdgroups,r3=r2+simdgroups;
threadgroup float4 decoded[32];float4 s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;
for(uint base=0;base<K;base+=32){uint k=base+lane;
  if(simd==0)decoded[lane]=k<K?qquad(trellis,levels,k,n,N,eb):float4(0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(k<K){float4 v=decoded[lane];if(r0<M)s0+=float(x[r0*K+k])*v;if(r1<M)s1+=float(x[r1*K+k])*v;
    if(r2<M)s2+=float(x[r2*K+k])*v;if(r3<M)s3+=float(x[r3*K+k])*v;}
  threadgroup_barrier(mem_flags::mem_threadgroup);}
s0.x=simd_sum(s0.x);s0.y=simd_sum(s0.y);s1.x=simd_sum(s1.x);s1.y=simd_sum(s1.y);
s2.x=simd_sum(s2.x);s2.y=simd_sum(s2.y);s3.x=simd_sum(s3.x);s3.y=simd_sum(s3.y);
s0.z=simd_sum(s0.z);s0.w=simd_sum(s0.w);s1.z=simd_sum(s1.z);s1.w=simd_sum(s1.w);
s2.z=simd_sum(s2.z);s2.w=simd_sum(s2.w);s3.z=simd_sum(s3.z);s3.w=simd_sum(s3.w);
if(lane==0){if(r0<M){half4 v=half4(s0);out[r0*N+n]=v.x;out[r0*N+n+1]=v.y;out[r0*N+n+2]=v.z;out[r0*N+n+3]=v.w;}
  if(r1<M){half4 v=half4(s1);out[r1*N+n]=v.x;out[r1*N+n+1]=v.y;out[r1*N+n+2]=v.z;out[r1*N+n+3]=v.w;}
  if(r2<M){half4 v=half4(s2);out[r2*N+n]=v.x;out[r2*N+n+1]=v.y;out[r2*N+n+2]=v.z;out[r2*N+n+3]=v.w;}
  if(r3<M){half4 v=half4(s3);out[r3*N+n]=v.x;out[r3*N+n+1]=v.y;out[r3*N+n+2]=v.z;out[r3*N+n+3]=v.w;}}
"""

_MULTIROW_N8_SOURCE = r"""
uint group=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint simd=simdgroup_index_in_threadgroup;
uint M=dims[0],K=dims[1],N=dims[2],eb=EdgeBits,row_tile=dims[4],vc=N>>3;
uint row_block=group/vc,vector=group-row_block*vc,row_base=row_block*row_tile,n=vector<<3;
uint simdgroups=row_tile>>1,r0=row_base+simd,r1=r0+simdgroups;
threadgroup float4 d0[32],d1[32];float4 s00=0.0f,s01=0.0f,s10=0.0f,s11=0.0f;
for(uint base=0;base<K;base+=32){uint k=base+lane;
  if(simd==0){if(k<K)qoctet(trellis,levels,k,n,N,eb,d0[lane],d1[lane]);else{d0[lane]=0.0f;d1[lane]=0.0f;}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(k<K){float4 v0=d0[lane],v1=d1[lane];if(r0<M){float input=float(x[r0*K+k]);s00+=input*v0;s01+=input*v1;}
    if(r1<M){float input=float(x[r1*K+k]);s10+=input*v0;s11+=input*v1;}}
  threadgroup_barrier(mem_flags::mem_threadgroup);}
s00.x=simd_sum(s00.x);s00.y=simd_sum(s00.y);s00.z=simd_sum(s00.z);s00.w=simd_sum(s00.w);
s01.x=simd_sum(s01.x);s01.y=simd_sum(s01.y);s01.z=simd_sum(s01.z);s01.w=simd_sum(s01.w);
s10.x=simd_sum(s10.x);s10.y=simd_sum(s10.y);s10.z=simd_sum(s10.z);s10.w=simd_sum(s10.w);
s11.x=simd_sum(s11.x);s11.y=simd_sum(s11.y);s11.z=simd_sum(s11.z);s11.w=simd_sum(s11.w);
if(lane==0){if(r0<M){half4 v0=half4(s00),v1=half4(s01);out[r0*N+n]=v0.x;out[r0*N+n+1]=v0.y;
    out[r0*N+n+2]=v0.z;out[r0*N+n+3]=v0.w;out[r0*N+n+4]=v1.x;out[r0*N+n+5]=v1.y;
    out[r0*N+n+6]=v1.z;out[r0*N+n+7]=v1.w;}
  if(r1<M){half4 v0=half4(s10),v1=half4(s11);out[r1*N+n]=v0.x;out[r1*N+n+1]=v0.y;
    out[r1*N+n+2]=v0.z;out[r1*N+n+3]=v0.w;out[r1*N+n+4]=v1.x;out[r1*N+n+5]=v1.y;
    out[r1*N+n+6]=v1.z;out[r1*N+n+7]=v1.w;}}
"""

_V2B4_P64_MULTIROW_SOURCES = {
    2: _v2_banked_source(_MULTIROW_SOURCE, "v2b4_p64"),
    4: _v2_banked_source(_MULTIROW_N4_SOURCE, "v2b4_p64"),
    8: _v2_banked_source(_MULTIROW_N8_SOURCE, "v2b4_p64"),
}
_V2B2_P32_MULTIROW_SOURCES = {
    2: _v2_banked_source(_MULTIROW_SOURCE, "v2b2_p32"),
    4: _v2_banked_source(_MULTIROW_N4_SOURCE, "v2b2_p32"),
    8: _v2_banked_source(_MULTIROW_N8_SOURCE, "v2b2_p32"),
}

_HYB_REFERENCE_SOURCE = r"""
uint group=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
uint M=dims[0],K=dims[1],N=dims[2],eb=dims[3],m=group/N,n=group-m*N;float sum=0.0f;
for(uint k=lane;k<K;k+=32)sum+=float(x[m*K+k])*qhyb(trellis,lut,k,n,N,eb);
sum=simd_sum(sum);if(lane==0)out[m*N+n]=half(sum);
"""

_VITERBI_HEADER = r"""
inline float qvq_emit(float2 target, device const float* codebook, uint state, float weight) {
  float x=target.x,y=target.y,cx=codebook[2*state],cy=codebook[2*state+1];
  float tn=x*x+y*y,cn=cx*cx+cy*cy,dot=fma(y,cy,x*cx);
  return max(tn+cn-2.0f*dot,0.0f)*weight;
}
"""

_VITERBI_SOURCE = r"""
constexpr uint state_count=65536,prefix_count=1u<<EdgeBits,suffix_count=state_count/prefix_count;
constexpr uint overlap_mask=(1u<<(16-EdgeBits))-1u;
uint batch=threadgroup_position_in_grid.x,tid=thread_index_in_threadgroup;
uint steps=dims[1];bool constrained=dims[2]!=0,weighted=dims[3]!=0;
ulong sb=ulong(batch)*steps*2,cb=ulong(batch)*state_count;
ulong pb=ulong(batch)*(steps-1)*suffix_count,stb=ulong(batch)*steps;
uint required=constrained?overlap[batch]:0;
device float* current=costs+cb;device float* following=next_costs+cb;
float first_weight=weighted?step_weights[ulong(batch)*steps]:1.0f;
float2 first_target=float2(sequences[sb],sequences[sb+1]);
for(uint state=tid;state<state_count;state+=256){float v=qvq_emit(first_target,codebook,state,first_weight);
  if(constrained&&(state>>EdgeBits)!=required)v=INFINITY;current[state]=v;}
threadgroup_barrier(mem_flags::mem_device);
for(uint step=1;step<steps;++step){ulong target_offset=sb+ulong(step)*2;
  float2 target=float2(sequences[target_offset],sequences[target_offset+1]);
  float sw=weighted?step_weights[ulong(batch)*steps+step]:1.0f;
  for(uint suffix=tid;suffix<suffix_count;suffix+=256){float best=current[suffix];uchar bp=0;
    for(uint prefix=1;prefix<prefix_count;++prefix){float v=current[ulong(prefix)*suffix_count+suffix];
      if(v<best){best=v;bp=uchar(prefix);}}
    backpointers[pb+ulong(step-1)*suffix_count+suffix]=bp;
    for(uint edge=0;edge<prefix_count;++edge){uint state=suffix*prefix_count+edge;
      following[state]=best+qvq_emit(target,codebook,state,sw);}}
  threadgroup_barrier(mem_flags::mem_device);device float* tmp=current;current=following;following=tmp;}
threadgroup float rcost[256];threadgroup uint rstate[256];float best=INFINITY;uint best_state=state_count;
for(uint state=tid;state<state_count;state+=256){if(constrained&&(state&overlap_mask)!=required)continue;
  float v=current[state];if(v<best||(v==best&&state<best_state)){best=v;best_state=state;}}
rcost[tid]=best;rstate[tid]=best_state;threadgroup_barrier(mem_flags::mem_threadgroup);
if(tid==0){for(uint lane=1;lane<256;++lane){float v=rcost[lane];uint s=rstate[lane];
    if(v<best||(v==best&&s<best_state)){best=v;best_state=s;}}
  squared_error[batch]=best;states[stb+steps-1]=best_state;
  for(uint step=steps-1;step>0;--step){uint suffix=best_state>>EdgeBits;
    uint prefix=uint(backpointers[pb+ulong(step-1)*suffix_count+suffix]);
    best_state=prefix*suffix_count+suffix;states[stb+step-1]=best_state;}}
"""


def _kernel():
    global _KERNEL, _KERNEL_ERROR
    if _KERNEL is None:
        with _KERNEL_LOCK:
            if _KERNEL is None:
                if _KERNEL_ERROR is not None:
                    raise RuntimeError(_KERNEL_ERROR)
                import mlx.core as mx

                try:
                    _KERNEL = mx.fast.metal_kernel(
                        name="gptqmodel_qvq_planar",
                        input_names=["x", "trellis", "levels", "dims"],
                        output_names=["out"],
                        header=_HEADER,
                        source=_SOURCE,
                        ensure_row_contiguous=True,
                    )
                except Exception as exc:
                    _KERNEL_ERROR = f"QVQ MLX kernel creation failed: {exc}"
                    raise RuntimeError(_KERNEL_ERROR) from exc
    return _KERNEL


def _dual_v2_kernel():
    global _DUAL_V2_KERNEL, _DUAL_V2_KERNEL_ERROR
    if _DUAL_V2_KERNEL is None:
        with _KERNEL_LOCK:
            if _DUAL_V2_KERNEL is None:
                if _DUAL_V2_KERNEL_ERROR is not None:
                    raise RuntimeError(_DUAL_V2_KERNEL_ERROR)
                import mlx.core as mx

                try:
                    _DUAL_V2_KERNEL = mx.fast.metal_kernel(
                        name="gptqmodel_qvq_planar_dual_v2",
                        input_names=["x", "trellis", "levels", "dims"],
                        output_names=["out"],
                        header=_HEADER,
                        source=_DUAL_V2_SOURCE,
                        ensure_row_contiguous=True,
                    )
                except Exception as exc:
                    _DUAL_V2_KERNEL_ERROR = f"QVQ Dual-V2 MLX kernel creation failed: {exc}"
                    raise RuntimeError(_DUAL_V2_KERNEL_ERROR) from exc
    return _DUAL_V2_KERNEL


def _fp32_kernel(kind: str):
    """Range-preserving scalar-row kernel used by full QVQ epilogues."""

    kernel = _FP32_KERNELS.get(kind)
    if kernel is not None:
        return kernel
    with _KERNEL_LOCK:
        kernel = _FP32_KERNELS.get(kind)
        if kernel is not None:
            return kernel
        if kind in _FP32_KERNEL_ERRORS:
            raise RuntimeError(_FP32_KERNEL_ERRORS[kind])
        import mlx.core as mx

        if kind == "v2":
            input_names = ["x", "trellis", "levels", "dims"]
            source = _FP32_SOURCE
        elif kind == "dual_v2":
            input_names = ["x", "trellis", "levels", "dims"]
            source = _DUAL_V2_FP32_SOURCE
        elif kind == "v4":
            input_names = ["x", "trellis", "levels", "dims"]
            source = _V4_N4_FP32_SOURCE
        elif kind == "v4_banked":
            input_names = ["x", "trellis", "bank_ids", "levels", "dims"]
            source = _V4_N4_BANKED_FP32_SOURCE
        elif kind == "v4_l18":
            input_names = ["x", "trellis", "levels", "dims"]
            source = _V4_N4_L18_FP32_SOURCE
        else:
            raise ValueError(f"unknown QVQ MLX FP32 kernel kind: {kind}")
        try:
            kernel = mx.fast.metal_kernel(
                name=f"gptqmodel_qvq_planar_{kind}_fp32",
                input_names=input_names,
                output_names=["out"],
                header=_HEADER,
                source=source,
                ensure_row_contiguous=True,
            )
        except Exception as exc:
            error = f"QVQ MLX FP32 kernel creation failed for {kind}: {exc}"
            _FP32_KERNEL_ERRORS[kind] = error
            raise RuntimeError(error) from exc
        _FP32_KERNELS[kind] = kernel
        return kernel


def _v2_banked_kernel(kind: str, vector_width: int, *, output_fp32: bool):
    """Build one selector-aware V2 kernel without multiplying format variants."""

    if kind not in ("v2b2_p32", "v2b4_p64"):
        raise ValueError(f"unknown QVQ banked-V2 kernel kind: {kind}")
    if vector_width not in (2, 4, 8) or (output_fp32 and vector_width != 2):
        raise ValueError("QVQ banked-V2 FP32 kernels currently require vector width 2")
    key = (kind, vector_width, output_fp32)
    kernel = _V2_BANKED_KERNELS.get(key)
    if kernel is not None:
        return kernel
    with _KERNEL_LOCK:
        kernel = _V2_BANKED_KERNELS.get(key)
        if kernel is not None:
            return kernel
        if key in _V2_BANKED_KERNEL_ERRORS:
            raise RuntimeError(_V2_BANKED_KERNEL_ERRORS[key])
        import mlx.core as mx

        input_names = ["x", "trellis", "bank_ids"]
        if kind == "v2b2_p32":
            input_names.append("bank_alt_id")
        input_names.extend(("levels", "dims"))
        if output_fp32:
            source = _V2B2_P32_FP32_SOURCE if kind == "v2b2_p32" else _V2B4_P64_FP32_SOURCE
        else:
            sources = _V2B2_P32_MULTIROW_SOURCES if kind == "v2b2_p32" else _V2B4_P64_MULTIROW_SOURCES
            source = (
                _V2B2_P32_SOURCE
                if kind == "v2b2_p32" and vector_width == 2
                else _V2B4_P64_SOURCE
                if kind == "v2b4_p64" and vector_width == 2
                else sources[vector_width]
            )
        try:
            kernel = mx.fast.metal_kernel(
                name=f"gptqmodel_qvq_planar_{kind}_n{vector_width}_{'fp32' if output_fp32 else 'fp16'}",
                input_names=input_names,
                output_names=["out"],
                header=_HEADER,
                source=source,
                ensure_row_contiguous=True,
            )
        except Exception as exc:
            error = f"QVQ banked-V2 MLX kernel creation failed for {key}: {exc}"
            _V2_BANKED_KERNEL_ERRORS[key] = error
            raise RuntimeError(error) from exc
        _V2_BANKED_KERNELS[key] = kernel
        return kernel


def _v4_kernel():
    global _V4_KERNEL, _V4_KERNEL_ERROR
    if _V4_KERNEL is None:
        with _KERNEL_LOCK:
            if _V4_KERNEL is None:
                if _V4_KERNEL_ERROR is not None:
                    raise RuntimeError(_V4_KERNEL_ERROR)
                import mlx.core as mx

                try:
                    _V4_KERNEL = mx.fast.metal_kernel(
                        name="gptqmodel_qvq_planar_v4",
                        input_names=["x", "trellis", "levels", "dims"],
                        output_names=["out"],
                        header=_HEADER,
                        source=_V4_SOURCE,
                        ensure_row_contiguous=True,
                    )
                except Exception as exc:
                    _V4_KERNEL_ERROR = f"QVQ V4 MLX kernel creation failed: {exc}"
                    raise RuntimeError(_V4_KERNEL_ERROR) from exc
    return _V4_KERNEL


def _v4_banked_kernel():
    global _V4_BANKED_KERNEL, _V4_BANKED_KERNEL_ERROR
    if _V4_BANKED_KERNEL is None:
        with _KERNEL_LOCK:
            if _V4_BANKED_KERNEL is None:
                if _V4_BANKED_KERNEL_ERROR is not None:
                    raise RuntimeError(_V4_BANKED_KERNEL_ERROR)
                import mlx.core as mx

                try:
                    _V4_BANKED_KERNEL = mx.fast.metal_kernel(
                        name="gptqmodel_qvq_planar_v4_banked",
                        input_names=["x", "trellis", "bank_ids", "levels", "dims"],
                        output_names=["out"],
                        header=_HEADER,
                        source=_V4_BANKED_SOURCE,
                        ensure_row_contiguous=True,
                    )
                except Exception as exc:
                    _V4_BANKED_KERNEL_ERROR = f"QVQ banked V4 MLX kernel creation failed: {exc}"
                    raise RuntimeError(_V4_BANKED_KERNEL_ERROR) from exc
    return _V4_BANKED_KERNEL


def _v4_n32_kernel(kind: str):
    if kind not in ("generic", "e4", "banked"):
        raise ValueError(f"Unknown QVQ V4 MLX N32 kernel kind: {kind}")
    kernel = _V4_N32_KERNELS.get(kind)
    if kernel is not None:
        return kernel
    with _KERNEL_LOCK:
        kernel = _V4_N32_KERNELS.get(kind)
        if kernel is not None:
            return kernel
        error = _V4_N32_KERNEL_ERRORS.get(kind)
        if error is not None:
            raise RuntimeError(error)
        import mlx.core as mx

        source_by_kind = {
            "generic": _V4_N32_SOURCE,
            "e4": _V4_N32_E4_SOURCE,
            "banked": _V4_N32_BANKED_SOURCE,
        }
        input_names = ["x", "trellis", "levels", "dims"]
        if kind == "banked":
            input_names.insert(2, "bank_ids")
        try:
            kernel = mx.fast.metal_kernel(
                name=f"gptqmodel_qvq_planar_v4_n32_{kind}",
                input_names=input_names,
                output_names=["out"],
                header=_HEADER,
                source=source_by_kind[kind],
                ensure_row_contiguous=True,
            )
        except Exception as exc:
            message = f"QVQ V4 MLX N32 {kind} kernel creation failed: {exc}"
            _V4_N32_KERNEL_ERRORS[kind] = message
            raise RuntimeError(message) from exc
        _V4_N32_KERNELS[kind] = kernel
        return kernel


def _v4_n4_kernel(kind: str):
    if kind not in ("generic", "e4", "banked", "l18"):
        raise ValueError(f"Unknown QVQ V4 MLX N4 kernel kind: {kind}")
    kernel = _V4_N4_KERNELS.get(kind)
    if kernel is not None:
        return kernel
    with _KERNEL_LOCK:
        kernel = _V4_N4_KERNELS.get(kind)
        if kernel is not None:
            return kernel
        error = _V4_N4_KERNEL_ERRORS.get(kind)
        if error is not None:
            raise RuntimeError(error)
        import mlx.core as mx

        source_by_kind = {
            "generic": _V4_N4_SOURCE,
            "e4": _V4_N4_E4_SOURCE,
            "banked": _V4_N4_BANKED_SOURCE,
            "l18": _V4_N4_L18_SOURCE,
        }
        input_names = ["x", "trellis", "levels", "dims"]
        if kind == "banked":
            input_names.insert(2, "bank_ids")
        try:
            kernel = mx.fast.metal_kernel(
                name=f"gptqmodel_qvq_planar_v4_n4_{kind}",
                input_names=input_names,
                output_names=["out"],
                header=_HEADER,
                source=source_by_kind[kind],
                ensure_row_contiguous=True,
            )
        except Exception as exc:
            message = f"QVQ V4 MLX N4 {kind} kernel creation failed: {exc}"
            _V4_N4_KERNEL_ERRORS[kind] = message
            raise RuntimeError(message) from exc
        _V4_N4_KERNELS[kind] = kernel
        return kernel


def _v4_mma_kernel(kind: str):
    if kind not in ("generic", "e4", "banked"):
        raise ValueError(f"Unknown QVQ V4 MLX MMA kernel kind: {kind}")
    kernel = _V4_MMA_KERNELS.get(kind)
    if kernel is not None:
        return kernel
    with _KERNEL_LOCK:
        kernel = _V4_MMA_KERNELS.get(kind)
        if kernel is not None:
            return kernel
        error = _V4_MMA_KERNEL_ERRORS.get(kind)
        if error is not None:
            raise RuntimeError(error)
        import mlx.core as mx

        input_names = ["x", "trellis", "levels", "dims"]
        source = _V4_MMA_E4_SOURCE if kind == "e4" else _V4_MMA_SOURCE
        if kind == "banked":
            input_names.insert(2, "bank_ids")
            source = _V4_MMA_BANKED_SOURCE
        try:
            kernel = mx.fast.metal_kernel(
                name=f"gptqmodel_qvq_planar_v4_mma_{kind}",
                input_names=input_names,
                output_names=["out"],
                header=_V4_MMA_HEADER,
                source=source,
                ensure_row_contiguous=True,
            )
        except Exception as exc:
            message = f"QVQ V4 MLX MMA {kind} kernel creation failed: {exc}"
            _V4_MMA_KERNEL_ERRORS[kind] = message
            raise RuntimeError(message) from exc
        _V4_MMA_KERNELS[kind] = kernel
        return kernel


def _v4_banked_multirow_kernel(kind: str):
    if kind != "generic":
        raise ValueError(f"Unknown QVQ banked MLX multi-row kernel kind: {kind}")
    kernel = _V4_BANKED_MULTIROW_KERNELS.get(kind)
    if kernel is not None:
        return kernel
    with _KERNEL_LOCK:
        kernel = _V4_BANKED_MULTIROW_KERNELS.get(kind)
        if kernel is not None:
            return kernel
        error = _V4_BANKED_MULTIROW_KERNEL_ERRORS.get(kind)
        if error is not None:
            raise RuntimeError(error)
        import mlx.core as mx

        source = _V4_MULTIROW_SOURCE.replace(
            "qwide4(trellis,levels,",
            "qwide4b(trellis,bank_ids,levels,",
        )
        try:
            kernel = mx.fast.metal_kernel(
                name=f"gptqmodel_qvq_planar_v4_banked_multirow_{kind}",
                input_names=["x", "trellis", "bank_ids", "levels", "dims"],
                output_names=["out"],
                header=_HEADER,
                source=source,
                ensure_row_contiguous=True,
            )
        except Exception as exc:
            message = f"QVQ banked V4 MLX {kind} multi-row kernel creation failed: {exc}"
            _V4_BANKED_MULTIROW_KERNEL_ERRORS[kind] = message
            raise RuntimeError(message) from exc
        _V4_BANKED_MULTIROW_KERNELS[kind] = kernel
        return kernel


def _v4_e4_kernel():
    global _V4_E4_KERNEL, _V4_E4_KERNEL_ERROR
    if _V4_E4_KERNEL is None:
        with _KERNEL_LOCK:
            if _V4_E4_KERNEL is None:
                if _V4_E4_KERNEL_ERROR is not None:
                    raise RuntimeError(_V4_E4_KERNEL_ERROR)
                import mlx.core as mx

                try:
                    _V4_E4_KERNEL = mx.fast.metal_kernel(
                        name="gptqmodel_qvq_planar_v4_e4",
                        input_names=["x", "trellis", "levels", "dims"],
                        output_names=["out"],
                        header=_HEADER,
                        source=_V4_E4_SOURCE,
                        ensure_row_contiguous=True,
                    )
                except Exception as exc:
                    _V4_E4_KERNEL_ERROR = f"QVQ V4 E4 MLX kernel creation failed: {exc}"
                    raise RuntimeError(_V4_E4_KERNEL_ERROR) from exc
    return _V4_E4_KERNEL


def _v4_multirow_kernel():
    global _V4_MULTIROW_KERNEL, _V4_MULTIROW_KERNEL_ERROR
    if _V4_MULTIROW_KERNEL is None:
        with _KERNEL_LOCK:
            if _V4_MULTIROW_KERNEL is None:
                if _V4_MULTIROW_KERNEL_ERROR is not None:
                    raise RuntimeError(_V4_MULTIROW_KERNEL_ERROR)
                import mlx.core as mx

                try:
                    _V4_MULTIROW_KERNEL = mx.fast.metal_kernel(
                        name="gptqmodel_qvq_planar_v4_multirow",
                        input_names=["x", "trellis", "levels", "dims"],
                        output_names=["out"],
                        header=_HEADER,
                        source=_V4_MULTIROW_SOURCE,
                        ensure_row_contiguous=True,
                    )
                except Exception as exc:
                    _V4_MULTIROW_KERNEL_ERROR = f"QVQ V4 MLX multi-row kernel creation failed: {exc}"
                    raise RuntimeError(_V4_MULTIROW_KERNEL_ERROR) from exc
    return _V4_MULTIROW_KERNEL


def _multirow_kernel():
    global _MULTIROW_KERNEL, _MULTIROW_KERNEL_ERROR
    if _MULTIROW_KERNEL is None:
        with _KERNEL_LOCK:
            if _MULTIROW_KERNEL is None:
                if _MULTIROW_KERNEL_ERROR is not None:
                    raise RuntimeError(_MULTIROW_KERNEL_ERROR)
                import mlx.core as mx

                try:
                    _MULTIROW_KERNEL = mx.fast.metal_kernel(
                        name="gptqmodel_qvq_planar_multirow",
                        input_names=["x", "trellis", "levels", "dims"],
                        output_names=["out"],
                        header=_HEADER,
                        source=_MULTIROW_SOURCE,
                        ensure_row_contiguous=True,
                    )
                except Exception as exc:
                    _MULTIROW_KERNEL_ERROR = f"QVQ MLX multi-row kernel creation failed: {exc}"
                    raise RuntimeError(_MULTIROW_KERNEL_ERROR) from exc
    return _MULTIROW_KERNEL


def _multirow_n4_kernel():
    global _MULTIROW_N4_KERNEL, _MULTIROW_N4_KERNEL_ERROR
    if _MULTIROW_N4_KERNEL is None:
        with _KERNEL_LOCK:
            if _MULTIROW_N4_KERNEL is None:
                if _MULTIROW_N4_KERNEL_ERROR is not None:
                    raise RuntimeError(_MULTIROW_N4_KERNEL_ERROR)
                import mlx.core as mx

                try:
                    _MULTIROW_N4_KERNEL = mx.fast.metal_kernel(
                        name="gptqmodel_qvq_planar_multirow_n4",
                        input_names=["x", "trellis", "levels", "dims"],
                        output_names=["out"],
                        header=_HEADER,
                        source=_MULTIROW_N4_SOURCE,
                        ensure_row_contiguous=True,
                    )
                except Exception as exc:
                    _MULTIROW_N4_KERNEL_ERROR = f"QVQ MLX multi-row N4 kernel creation failed: {exc}"
                    raise RuntimeError(_MULTIROW_N4_KERNEL_ERROR) from exc
    return _MULTIROW_N4_KERNEL


def _multirow_n8_kernel():
    global _MULTIROW_N8_KERNEL, _MULTIROW_N8_KERNEL_ERROR
    if _MULTIROW_N8_KERNEL is None:
        with _KERNEL_LOCK:
            if _MULTIROW_N8_KERNEL is None:
                if _MULTIROW_N8_KERNEL_ERROR is not None:
                    raise RuntimeError(_MULTIROW_N8_KERNEL_ERROR)
                import mlx.core as mx

                try:
                    _MULTIROW_N8_KERNEL = mx.fast.metal_kernel(
                        name="gptqmodel_qvq_planar_multirow_n8",
                        input_names=["x", "trellis", "levels", "dims"],
                        output_names=["out"],
                        header=_HEADER,
                        source=_MULTIROW_N8_SOURCE,
                        ensure_row_contiguous=True,
                    )
                except Exception as exc:
                    _MULTIROW_N8_KERNEL_ERROR = f"QVQ MLX multi-row N8 kernel creation failed: {exc}"
                    raise RuntimeError(_MULTIROW_N8_KERNEL_ERROR) from exc
    return _MULTIROW_N8_KERNEL


def _hyb_reference_kernel():
    global _HYB_REFERENCE_KERNEL, _HYB_REFERENCE_KERNEL_ERROR
    if _HYB_REFERENCE_KERNEL is None:
        with _HYB_REFERENCE_KERNEL_LOCK:
            if _HYB_REFERENCE_KERNEL is None:
                if _HYB_REFERENCE_KERNEL_ERROR is not None:
                    raise RuntimeError(_HYB_REFERENCE_KERNEL_ERROR)
                import mlx.core as mx

                try:
                    _HYB_REFERENCE_KERNEL = mx.fast.metal_kernel(
                        name="gptqmodel_qvq_hyb_reference",
                        input_names=["x", "trellis", "lut", "dims"],
                        output_names=["out"],
                        header=_HEADER,
                        source=_HYB_REFERENCE_SOURCE,
                        ensure_row_contiguous=True,
                    )
                except Exception as exc:
                    _HYB_REFERENCE_KERNEL_ERROR = f"QVQ HYB reference MLX creation failed: {exc}"
                    raise RuntimeError(_HYB_REFERENCE_KERNEL_ERROR) from exc
    return _HYB_REFERENCE_KERNEL


def _viterbi_kernel():
    global _VITERBI_KERNEL, _VITERBI_KERNEL_ERROR
    if _VITERBI_KERNEL is None:
        with _KERNEL_LOCK:
            if _VITERBI_KERNEL is None:
                if _VITERBI_KERNEL_ERROR is not None:
                    raise RuntimeError(_VITERBI_KERNEL_ERROR)
                import mlx.core as mx

                try:
                    _VITERBI_KERNEL = mx.fast.metal_kernel(
                        name="gptqmodel_qvq_viterbi_lowrate",
                        input_names=["sequences", "codebook", "overlap", "step_weights", "dims"],
                        output_names=["costs", "next_costs", "backpointers", "states", "squared_error"],
                        header=_VITERBI_HEADER,
                        source=_VITERBI_SOURCE,
                        ensure_row_contiguous=True,
                    )
                except Exception as exc:
                    _VITERBI_KERNEL_ERROR = f"QVQ Viterbi MLX kernel creation failed: {exc}"
                    raise RuntimeError(_VITERBI_KERNEL_ERROR) from exc
    return _VITERBI_KERNEL


def qvq_mlx_viterbi(sequences, codebook, bits: float, overlap=None, step_weights=None):
    """Run the persistent W1/W1.5 Viterbi recurrence using an MLX Metal kernel."""

    import mlx.core as mx

    transition_bits = qvq_transition_bits(bits)
    if transition_bits not in (2, 3):
        raise ValueError("QVQ MLX Viterbi currently supports W1 and W1.5")
    if sequences.ndim != 3 or sequences.shape[2] != 2:
        raise ValueError("QVQ MLX Viterbi sequences must have shape [batch, steps, 2]")
    if codebook.shape != (1 << 16, 2):
        raise ValueError("QVQ MLX Viterbi codebook must have shape [65536, 2]")
    if sequences.dtype != mx.float32 or codebook.dtype != mx.float32:
        raise TypeError("QVQ MLX Viterbi sequences and codebook must use float32")
    batch, steps, _ = sequences.shape
    if batch < 1 or steps < 1:
        raise ValueError("QVQ MLX Viterbi requires a nonempty batch and sequence")
    if not bool(mx.all(mx.isfinite(sequences)).item()) or not bool(mx.all(mx.isfinite(codebook)).item()):
        raise ValueError("QVQ MLX Viterbi sequences and codebook must contain only finite values")
    constrained = overlap is not None
    weighted = step_weights is not None
    if constrained:
        if overlap.dtype != mx.uint32:
            raise TypeError("QVQ MLX Viterbi overlap must use uint32")
        if overlap.shape != (batch,):
            raise ValueError("QVQ MLX Viterbi overlap must have shape [batch]")
        overlap_limit = 1 << (16 - transition_bits)
        if bool(mx.any(overlap >= overlap_limit).item()):
            raise ValueError(f"QVQ MLX Viterbi overlap must be in [0, {overlap_limit - 1}]")
    else:
        overlap = mx.zeros((1,), dtype=mx.uint32)
    if weighted:
        if step_weights.dtype != mx.float32:
            raise TypeError("QVQ MLX Viterbi step weights must use float32")
        if step_weights.shape != (batch, steps):
            raise ValueError("QVQ MLX Viterbi step weights must have shape [batch, steps]")
        if not bool(mx.all(mx.isfinite(step_weights)).item()) or bool(mx.any(step_weights < 0).item()):
            raise ValueError("QVQ MLX Viterbi step weights must be finite and nonnegative")
    else:
        step_weights = mx.zeros((1,), dtype=mx.float32)
    suffix_count = (1 << 16) >> transition_bits
    dims = mx.array([batch, steps, int(constrained), int(weighted)], dtype=mx.uint32)
    outputs = _viterbi_kernel()(
        inputs=[sequences, codebook, overlap, step_weights, dims],
        template=[("EdgeBits", transition_bits)],
        grid=(batch * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (batch, 1 << 16),
            (batch, 1 << 16),
            (batch, steps - 1, suffix_count),
            (batch, steps),
            (batch,),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.uint8, mx.uint32, mx.float32],
    )
    return outputs[3], outputs[4]


@functools.lru_cache(maxsize=128)
def _dims_array(m: int, k: int, n: int, bits: int, row_tile: int = 0):
    import mlx.core as mx

    return mx.array([m, k, n, bits, row_tile], dtype=mx.uint32)


def _v4_row_tile(transition_bits: int, m: int, k: int, n: int) -> int:
    if k >= 8192 and n >= 8192 and transition_bits == 4:
        return 4
    if m <= 4:
        return 8 if transition_bits == 4 and k <= 2048 and n <= 2048 else 4
    if m <= 16:
        return 8
    if m < 20:
        return 4
    return 8


def _v4_independent_output_width(transition_bits: int, k: int, n: int) -> int:
    if k <= 2048 and n % 32 == 0:
        return 4 if n == 2048 and transition_bits != 4 else 32
    return 16


def _v4_use_mma(m: int, n: int) -> bool:
    return m >= (5 if n >= 8192 else 8)


@functools.lru_cache(maxsize=1)
def _cached_pgc16_levels(codebook_version: str):
    import mlx.core as mx

    levels = pgc16_levels_for_version(codebook_version)
    return mx.array(levels.tolist(), dtype=mx.float16)


def _pgc16_levels(
    codebook_version: str,
):
    global _PGC16_LEVELS_HOT
    hot = _PGC16_LEVELS_HOT
    if hot is not None and codebook_version == hot[0]:
        return hot[1]
    version = str(codebook_version).strip().lower()
    levels = _cached_pgc16_levels(version)
    _PGC16_LEVELS_HOT = (version, levels)
    return levels


def _prepare_qvq_mlx_compander(
    codebook_version: str,
) -> _QVQMLXPreparedCompander:
    return _QVQMLXPreparedCompander(levels=_pgc16_levels(codebook_version))


def _run_multirow(
    x,
    trellis,
    levels,
    transition_bits: int,
    *,
    m: int,
    k: int,
    n: int,
    vector_width: int,
):
    import mlx.core as mx

    if vector_width not in (2, 4, 8):
        raise ValueError(f"QVQ MLX multi-row vector width must be 2, 4, or 8, got {vector_width}")
    row_tile = 16 if vector_width == 8 or m <= 16 else 32
    group_size = (row_tile // (2 if vector_width == 8 else 4)) * 32
    row_blocks = (m + row_tile - 1) // row_tile
    kernels = {2: _multirow_kernel, 4: _multirow_n4_kernel, 8: _multirow_n8_kernel}
    kernel = kernels[vector_width]()
    return kernel(
        inputs=[x, trellis, levels, _dims_array(m, k, n, transition_bits, row_tile)],
        template=[("EdgeBits", transition_bits)],
        grid=(row_blocks * (n // vector_width) * group_size, 1, 1),
        threadgroup=(group_size, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float16],
    )[0]


def _multirow_vector_width(transition_bits: int, m: int, k: int, n: int) -> int:
    if m == 16:
        if transition_bits in (8, 16):
            return 8
        if (
            transition_bits in (4, 10, 12)
            or (transition_bits == 6 and k >= 8192)
            or (transition_bits == 14 and max(k, n) >= 8192)
        ):
            return 4
        return 2
    if m == 32:
        if transition_bits != 6 and (k < 8192 or transition_bits in (8, 16)):
            return 8
        return 4
    return 4 if transition_bits == 16 or m <= 8 or m > 16 else 2


def _run_v2_banked(
    x,
    trellis,
    bank_ids,
    bank_alt_id,
    levels,
    transition_bits: int,
    *,
    kind: str,
    m: int,
    k: int,
    n: int,
    output_fp32: bool,
):
    import mlx.core as mx

    vector_width = 2 if output_fp32 or m < 4 else _multirow_vector_width(transition_bits, m, k, n)
    inputs = [x, trellis, bank_ids]
    if kind == "v2b2_p32":
        inputs.append(bank_alt_id)
    inputs.extend((levels, _dims_array(m, k, n, transition_bits, 0 if m < 4 else 16)))
    if m < 4 or output_fp32:
        grid = (m * (n // vector_width) * 32, 1, 1)
        threadgroup = (32, 1, 1)
    else:
        row_tile = 16 if vector_width == 8 or m <= 16 else 32
        group_size = (row_tile // (2 if vector_width == 8 else 4)) * 32
        inputs[-1] = _dims_array(m, k, n, transition_bits, row_tile)
        grid = (((m + row_tile - 1) // row_tile) * (n // vector_width) * group_size, 1, 1)
        threadgroup = (group_size, 1, 1)
    return _v2_banked_kernel(kind, vector_width, output_fp32=output_fp32)(
        inputs=inputs,
        template=[("EdgeBits", transition_bits)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(m, n)],
        output_dtypes=[mx.float32 if output_fp32 else mx.float16],
    )[0]


def qvq_mlx_gemv(
    x,
    trellis,
    bits: float,
    *,
    out_features: int,
    codebook_version: str = PGC16_CODEBOOK_VERSION,
    vector_size: int = 2,
    trellis_window: int = 16,
    dual_v2: bool = False,
    bank_ids=None,
    v2b4_p64: bool = False,
    v2b2_p32: bool = False,
    bank_alt_id=None,
    output_fp32: bool = False,
    _prepared_compander: _QVQMLXPreparedCompander | None = None,
):
    """Multiply transformed MLX activations by planar PGC16 tiles."""

    import mlx.core as mx

    bits = normalize_qvq_rate(bits)
    if not isinstance(output_fp32, bool):
        raise TypeError("QVQ MLX output_fp32 must be a bool")
    if not isinstance(dual_v2, bool):
        raise TypeError("QVQ MLX dual_v2 must be a bool")
    if not isinstance(v2b4_p64, bool) or not isinstance(v2b2_p32, bool):
        raise TypeError("QVQ MLX banked-V2 format flags must be bools")
    if sum((dual_v2, v2b4_p64, v2b2_p32)) > 1:
        raise ValueError("QVQ MLX Dual-V2, V2B4-P64, and V2B2-P32 are mutually exclusive")
    if vector_size not in (2, 4) or (vector_size == 4 and bits > 4):
        raise ValueError("QVQ MLX vector_size must be 2, or 4 for rates W1 through W4")
    trellis_window = _integer_argument(trellis_window, "trellis_window")
    if trellis_window not in (16, 18):
        raise ValueError("QVQ MLX trellis_window must be 16 or 18")
    if trellis_window == 18 and vector_size != 4:
        raise ValueError("QVQ MLX L18 requires vector_size=4")
    if trellis_window == 18 and bits > 2.5:
        raise ValueError("QVQ MLX L18 supports only rates W1 through W2.5")
    if trellis_window == 18 and bank_ids is not None:
        raise ValueError("QVQ MLX L18 uses implicit history-selected banks and rejects bank_ids")
    if bank_ids is not None and vector_size != 4 and not (v2b4_p64 or v2b2_p32):
        raise ValueError("QVQ MLX bank selectors require vector_size=4")
    if dual_v2 and (vector_size != 2 or trellis_window != 16 or bank_ids is not None):
        raise ValueError("QVQ MLX Dual-V2 requires vector_size=2, trellis_window=16, and no bank_ids")
    if (v2b4_p64 or v2b2_p32) and (
        vector_size != 2 or trellis_window != 16 or bits > 2.5 or bank_ids is None
    ):
        raise ValueError("QVQ MLX banked-V2 formats require L16/V2, packed selectors, and W1 through W2.5")
    if v2b2_p32:
        if bank_alt_id is None or bank_alt_id.dtype != mx.uint8 or bank_alt_id.shape != (1,):
            raise ValueError("QVQ MLX V2B2-P32 requires one uint8 alternative-bank ID")
        alt_id = int(bank_alt_id.item())
        if not 1 <= alt_id <= 3:
            raise ValueError("QVQ MLX V2B2-P32 alternative-bank ID must be in [1, 3]")
    elif bank_alt_id is not None:
        raise ValueError("QVQ MLX bank_alt_id is valid only for V2B2-P32")
    transition_bits = qvq_transition_bits(bits, vector_size=vector_size)
    if x.ndim != 2 or trellis.ndim != 2:
        raise ValueError("QVQ MLX expects 2D x and trellis arrays")
    if x.dtype != mx.float16 or trellis.dtype != mx.int32:
        raise TypeError("QVQ MLX requires float16 x and int32 planar trellis words")
    m, k = x.shape
    n = _integer_argument(out_features, "out_features")
    if k <= 0 or n <= 0 or k % 16 or n % 16:
        raise ValueError(f"QVQ MLX requires positive K/N divisible by 16, got K={k}, N={n}")
    expected = (
        (k // 16) * (n // 16),
        qvq_words_per_tile(bits, vector_size=vector_size),
    )
    if trellis.shape != expected:
        raise ValueError(f"QVQ planar trellis must have shape {expected}, got {trellis.shape}")
    if bank_ids is not None:
        packed_count = expected[0] if v2b4_p64 or v2b2_p32 else (expected[0] + 3) // 4
        if bank_ids.dtype != mx.uint8:
            raise TypeError("QVQ MLX bank selectors must use packed uint8 storage")
        if bank_ids.ndim != 1 or bank_ids.size != packed_count:
            raise ValueError(f"QVQ MLX bank selectors must have packed shape {(packed_count,)}")
    levels = (
        _pgc16_levels(codebook_version) if _prepared_compander is None else _prepared_compander.levels
    )
    if not m:
        return mx.empty((0, n), dtype=mx.float32 if output_fp32 else mx.float16)
    selector_size = 0 if bank_ids is None else bank_ids.size
    if max(m, k, n, m * n, x.size, trellis.size, levels.size, selector_size) > 2**32 - 1:
        raise ValueError("QVQ MLX dimensions exceed the uint32 kernel limit")
    if output_fp32:
        if v2b4_p64 or v2b2_p32:
            return _run_v2_banked(
                x,
                trellis,
                bank_ids,
                bank_alt_id,
                levels,
                transition_bits,
                kind="v2b2_p32" if v2b2_p32 else "v2b4_p64",
                m=m,
                k=k,
                n=n,
                output_fp32=True,
            )
        kind = (
            "dual_v2"
            if dual_v2
            else "v4_l18"
            if trellis_window == 18
            else "v4_banked"
            if bank_ids is not None
            else "v4"
            if vector_size == 4
            else "v2"
        )
        inputs = [x, trellis, levels, _dims_array(m, k, n, transition_bits)]
        if bank_ids is not None:
            inputs.insert(2, bank_ids)
        return _fp32_kernel(kind)(
            inputs=inputs,
            template=[("EdgeBits", transition_bits)],
            grid=(m * (n // vector_size) * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(m, n)],
            output_dtypes=[mx.float32],
        )[0]
    if dual_v2:
        return _dual_v2_kernel()(
            inputs=[x, trellis, levels, _dims_array(m, k, n, transition_bits)],
            template=[("EdgeBits", transition_bits)],
            grid=(m * (n // 2) * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(m, n)],
            output_dtypes=[mx.float16],
        )[0]
    if v2b4_p64 or v2b2_p32:
        return _run_v2_banked(
            x,
            trellis,
            bank_ids,
            bank_alt_id,
            levels,
            transition_bits,
            kind="v2b2_p32" if v2b2_p32 else "v2b4_p64",
            m=m,
            k=k,
            n=n,
            output_fp32=False,
        )
    if trellis_window == 18:
        return _v4_n4_kernel("l18")(
            inputs=[x, trellis, levels, _dims_array(m, k, n, transition_bits)],
            template=[("EdgeBits", transition_bits)],
            grid=(m * (n // 4) * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(m, n)],
            output_dtypes=[mx.float16],
        )[0]
    if bank_ids is not None:
        if _v4_use_mma(m, n):
            return _v4_mma_kernel("banked")(
                inputs=[x, trellis, bank_ids, levels, _dims_array(m, k, n, transition_bits)],
                template=[("EdgeBits", transition_bits)],
                grid=(((m + 7) // 8) * (n // 8) * 32, 1, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(m, n)],
                output_dtypes=[mx.float16],
            )[0]
        if m >= 4:
            row_tile = _v4_row_tile(transition_bits, m, k, n)
            group_size = row_tile * 32
            row_blocks = (m + row_tile - 1) // row_tile
            return _v4_banked_multirow_kernel("generic")(
                inputs=[x, trellis, bank_ids, levels, _dims_array(m, k, n, transition_bits, row_tile)],
                template=[("EdgeBits", transition_bits)],
                grid=(row_blocks * (n // 16) * group_size, 1, 1),
                threadgroup=(group_size, 1, 1),
                output_shapes=[(m, n)],
                output_dtypes=[mx.float16],
            )[0]
        output_width = _v4_independent_output_width(transition_bits, k, n)
        if output_width == 4:
            kernel = _v4_n4_kernel("banked")
        elif output_width == 32:
            kernel = _v4_n32_kernel("banked")
        else:
            kernel = _v4_banked_kernel()
        return kernel(
            inputs=[x, trellis, bank_ids, levels, _dims_array(m, k, n, transition_bits)],
            template=[("EdgeBits", transition_bits)],
            grid=(m * (n // output_width) * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(m, n)],
            output_dtypes=[mx.float16],
        )[0]
    if m >= 4:
        if vector_size == 4:
            if _v4_use_mma(m, n):
                return _v4_mma_kernel("e4" if transition_bits == 4 else "generic")(
                    inputs=[x, trellis, levels, _dims_array(m, k, n, transition_bits)],
                    template=[("EdgeBits", transition_bits)],
                    grid=(((m + 7) // 8) * (n // 8) * 32, 1, 1),
                    threadgroup=(32, 1, 1),
                    output_shapes=[(m, n)],
                    output_dtypes=[mx.float16],
                )[0]
            row_tile = _v4_row_tile(transition_bits, m, k, n)
            group_size = row_tile * 32
            row_blocks = (m + row_tile - 1) // row_tile
            return _v4_multirow_kernel()(
                inputs=[x, trellis, levels, _dims_array(m, k, n, transition_bits, row_tile)],
                template=[("EdgeBits", transition_bits)],
                grid=(row_blocks * (n // 16) * group_size, 1, 1),
                threadgroup=(group_size, 1, 1),
                output_shapes=[(m, n)],
                output_dtypes=[mx.float16],
            )[0]
        vector_width = _multirow_vector_width(transition_bits, m, k, n)
        return _run_multirow(
            x,
            trellis,
            levels,
            transition_bits,
            m=m,
            k=k,
            n=n,
            vector_width=vector_width,
        )
    output_width = vector_size
    if vector_size == 4:
        output_width = _v4_independent_output_width(transition_bits, k, n)
        if output_width == 4:
            kernel = _v4_n4_kernel("generic")
        elif output_width == 32:
            kernel = _v4_n32_kernel("e4" if transition_bits == 4 else "generic")
        else:
            kernel = _v4_e4_kernel() if transition_bits == 4 else _v4_kernel()
    else:
        kernel = _kernel()
    return kernel(
        inputs=[x, trellis, levels, _dims_array(m, k, n, transition_bits)],
        template=[("EdgeBits", transition_bits)],
        grid=(m * (n // output_width) * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float16],
    )[0]


def _qvq_mlx_hadamard(x, hadamard_matrix=None):
    """Apply QVQ's normalized Hadamard factorization using native MLX ops."""

    import mlx.core as mx

    original_shape = x.shape
    width = original_shape[-1]
    factor = 1 if hadamard_matrix is None else hadamard_matrix.shape[0]
    work = x.reshape(-1, width, 1)
    while work.shape[1] > factor:
        work = work.reshape(work.shape[0], work.shape[1] // 2, 2, work.shape[2])
        first = work[:, :, 0, :]
        second = work[:, :, 1, :]
        work = mx.stack((first + second, first - second), axis=2).reshape(
            work.shape[0], work.shape[1], -1
        )
    if hadamard_matrix is not None:
        work = hadamard_matrix.reshape(1, factor, factor) @ work
    return (work.reshape(original_shape) / mx.sqrt(mx.array(width, dtype=mx.float32))).astype(mx.float32)


def _qvq_mlx_hadamard_matrix(width: int):
    """Create only the small non-power-of-two factor used by QVQ's transform."""

    import mlx.core as mx
    import torch

    from ..quantization.rotation.hadamard_utils import get_hadK

    matrix, _ = get_hadK(width)
    if matrix is None:
        return None
    return mx.array(matrix.to(dtype=torch.float32).cpu().numpy(), dtype=mx.float32)


def _qvq_mlx_narrow_with_row_scale(x):
    """Power-of-two row scaling for the FP16 native GEMV boundary."""

    import mlx.core as mx

    row_peak = mx.max(mx.abs(x), axis=-1, keepdims=True)
    safe_magnitude = mx.array(float(mx.finfo(mx.float16).max) / 2, dtype=mx.float32)
    required = mx.maximum(row_peak / safe_magnitude, mx.array(1, dtype=mx.float32))
    row_scale = mx.power(mx.array(2, dtype=mx.float32), mx.ceil(mx.log2(required)))
    return (x / row_scale).astype(mx.float16), row_scale


if _mlx_nn is not None:

    class QVQMLXLinear(_mlx_nn.Module):
        """Format-native MLX QVQ linear used by checkpoint model loading."""

        def __init__(
            self,
            *,
            bits: float,
            in_features: int,
            out_features: int,
            trellis,
            SU,
            SV,
            bias=None,
            codebook_version: str = PGC16_CODEBOOK_VERSION,
            vector_size: int = 2,
            trellis_window: int = 16,
            dual_v2: bool = False,
            bank_ids=None,
            v2b4_p64: bool = False,
            v2b2_p32: bool = False,
            bank_alt_id=None,
        ):
            super().__init__()
            import mlx.core as mx

            self.bits = normalize_qvq_rate(bits)
            self.in_features = _integer_argument(in_features, "in_features")
            self.out_features = _integer_argument(out_features, "out_features")
            self.codebook_version = str(codebook_version).strip().lower()
            self.vector_size = _integer_argument(vector_size, "vector_size")
            self.trellis_window = _integer_argument(trellis_window, "trellis_window")
            if not isinstance(dual_v2, bool):
                raise TypeError("QVQ MLX dual_v2 must be a bool")
            self.dual_v2 = dual_v2
            if not isinstance(v2b4_p64, bool) or not isinstance(v2b2_p32, bool):
                raise TypeError("QVQ MLX banked-V2 format flags must be bools")
            if sum((dual_v2, v2b4_p64, v2b2_p32)) > 1:
                raise ValueError("QVQ MLX Dual-V2, V2B4-P64, and V2B2-P32 are mutually exclusive")
            self.v2b4_p64 = v2b4_p64
            self.v2b2_p32 = v2b2_p32
            if self.trellis_window not in (16, 18):
                raise ValueError("QVQ MLX trellis_window must be 16 or 18")
            if self.trellis_window == 18 and self.vector_size != 4:
                raise ValueError("QVQ MLX L18 requires vector_size=4")
            if self.trellis_window == 18 and self.bits > 2.5:
                raise ValueError("QVQ MLX L18 supports only rates W1 through W2.5")
            if self.trellis_window == 18 and bank_ids is not None:
                raise ValueError("QVQ MLX L18 uses implicit history-selected banks and rejects bank_ids")
            if self.dual_v2 and (self.vector_size != 2 or self.trellis_window != 16 or bank_ids is not None):
                raise ValueError("QVQ MLX Dual-V2 requires vector_size=2, trellis_window=16, and no bank_ids")
            if (self.v2b4_p64 or self.v2b2_p32) and (
                self.vector_size != 2 or self.trellis_window != 16 or self.bits > 2.5 or bank_ids is None
            ):
                raise ValueError("QVQ MLX banked-V2 formats require L16/V2, selectors, and W1 through W2.5")
            self.trellis = trellis.astype(mx.int32)
            self.SU = SU.astype(mx.float32)
            self.SV = SV.astype(mx.float32)
            self.bias = None if bias is None else bias.astype(mx.float32)
            self.bank_ids = None if bank_ids is None else bank_ids.astype(mx.uint8)
            self.bank_alt_id = None if bank_alt_id is None else bank_alt_id.astype(mx.uint8)
            self._input_hadamard = _qvq_mlx_hadamard_matrix(self.in_features)
            self._output_hadamard = _qvq_mlx_hadamard_matrix(self.out_features)

            expected_trellis = (
                (self.in_features // 16) * (self.out_features // 16),
                qvq_words_per_tile(self.bits, vector_size=self.vector_size),
            )
            if self.trellis.shape != expected_trellis:
                raise ValueError(
                    f"QVQ MLX trellis must have shape {expected_trellis}, got {self.trellis.shape}"
                )
            if self.SU.shape != (self.in_features,) or self.SV.shape != (self.out_features,):
                raise ValueError("QVQ MLX SU/SV shapes must match the linear dimensions")
            if self.bias is not None and self.bias.shape != (self.out_features,):
                raise ValueError("QVQ MLX bias shape must match out_features")

        def __call__(self, x):
            import mlx.core as mx

            if x.shape[-1] != self.in_features:
                raise ValueError(f"QVQ MLX expected input width {self.in_features}, got {x.shape[-1]}")
            input_dtype = x.dtype
            leading_shape = x.shape[:-1]
            if x.size == 0:
                return mx.empty((*leading_shape, self.out_features), dtype=input_dtype)
            transformed = _qvq_mlx_hadamard(
                x.reshape(-1, self.in_features).astype(mx.float32) * self.SU,
                self._input_hadamard,
            )
            native_input, row_scale = _qvq_mlx_narrow_with_row_scale(transformed)
            output = qvq_mlx_gemv(
                native_input,
                self.trellis,
                self.bits,
                out_features=self.out_features,
                codebook_version=self.codebook_version,
                vector_size=self.vector_size,
                trellis_window=self.trellis_window,
                dual_v2=self.dual_v2,
                bank_ids=self.bank_ids,
                v2b4_p64=self.v2b4_p64,
                v2b2_p32=self.v2b2_p32,
                bank_alt_id=self.bank_alt_id,
                output_fp32=True,
            )
            output = _qvq_mlx_hadamard(output * row_scale, self._output_hadamard)
            output = output * self.SV
            if self.bias is not None:
                output = output + self.bias
            return output.reshape(*leading_shape, self.out_features).astype(input_dtype)

else:  # pragma: no cover - only instantiated when the optional MLX package is missing.

    class QVQMLXLinear:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise ModuleNotFoundError("QVQ MLX inference requires the optional `mlx` package")


def qvq_hyb_reference_mlx_gemv(x, trellis, lut, bits: float, *, out_features: int):
    """Run the former scalar HYB kernel as a non-loadable benchmark oracle."""

    import mlx.core as mx

    bits = normalize_qvq_rate(bits)
    transition_bits = qvq_transition_bits(bits)
    if x.ndim != 2 or trellis.ndim != 2 or lut.ndim != 2:
        raise ValueError("QVQ HYB reference expects 2D x, trellis, and LUT arrays")
    if x.dtype != mx.float16 or lut.dtype != mx.float16 or trellis.dtype != mx.int32:
        raise TypeError("QVQ HYB reference requires float16 x/LUT and int32 trellis words")
    m, k = x.shape
    n = int(out_features)
    if k <= 0 or n <= 0 or k % 16 or n % 16:
        raise ValueError(f"QVQ HYB reference requires positive aligned K/N, got K={k}, N={n}")
    expected = ((k // 16) * (n // 16), qvq_words_per_tile(bits))
    if trellis.shape != expected or lut.shape != (512, 2):
        raise ValueError("QVQ HYB reference trellis or LUT shape mismatch")
    if not m:
        return mx.empty((0, n), dtype=mx.float16)
    return _hyb_reference_kernel()(
        inputs=[x, trellis, lut, _dims_array(m, k, n, transition_bits)],
        template=[],
        grid=(m * n * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float16],
    )[0]


__all__ = ["QVQ_MLX_BITS", "QVQMLXLinear", "qvq_hyb_reference_mlx_gemv", "qvq_mlx_gemv", "qvq_mlx_viterbi"]
