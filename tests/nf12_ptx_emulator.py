# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Minimal vectorised PTX emulator for Triton/Gluon sm_80 straight-line kernels.

Executes every thread of a batch of CTAs in lockstep (numpy arrays [C, T]).
Only uniform branches are supported. FP32 FMA is correctly rounded
(float64 product is exact; TwoSum residual breaks float32 midpoint ties).
Purpose: compare the exact bits two compiled kernels produce on CPU.
"""
from __future__ import annotations

import re
import numpy as np

M32 = np.uint64(0xFFFFFFFF)


def f32(u):
    return (u & M32).astype(np.uint32).view(np.float32)


def u64(f):
    return f.astype(np.float32).view(np.uint32).astype(np.uint64)


def fma32(a, b, c):
    """Correctly rounded float32 a*b+c (the float64 product is exact)."""
    a = a.astype(np.float64); b = b.astype(np.float64); c = c.astype(np.float64)
    p = a * b
    r = p + c
    bb = r - p
    e = (p - (r - bb)) + (c - bb)  # TwoSum residual: p + c == r + e exactly
    f = r.astype(np.float32)
    fd = f.astype(np.float64)
    up = np.nextafter(f, np.float32(np.inf))
    dn = np.nextafter(f, np.float32(-np.inf))
    # Double rounding can only err when r is exactly a float32 midpoint and the
    # residual lies on the other side of it from the tie-to-even choice.
    out = np.where((r == (fd + up.astype(np.float64)) / 2) & (e > 0), up, f)
    out = np.where((r == (fd + dn.astype(np.float64)) / 2) & (e < 0), dn, out)
    return out.astype(np.float32)


class Memory:
    def __init__(self):
        self.base = 1 << 32
        self.buf = np.zeros(0, np.uint8)
        self.top = 0

    def alloc(self, array: np.ndarray) -> int:
        raw = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
        start = (self.top + 255) // 256 * 256
        need = start + raw.size
        if need > self.buf.size:
            grow = np.zeros(max(need, self.buf.size * 2) - self.buf.size, np.uint8)
            self.buf = np.concatenate([self.buf, grow])
        self.buf[start:start + raw.size] = raw
        self.top = need
        return self.base + start

    def read(self, addr: int, nbytes: int) -> np.ndarray:
        o = addr - self.base
        return self.buf[o:o + nbytes].copy()

    def gather(self, addr, width, mask):
        o = (addr.astype(np.int64) - self.base)
        o = np.where(mask, o, 0)
        assert (o >= 0).all() and (o + width <= self.buf.size).all(), "OOB global load"
        idx = o[..., None] + np.arange(width)
        return self.buf[idx]

    def scatter(self, addr, data_bytes, mask):
        o = (addr.astype(np.int64) - self.base)
        sel = mask.reshape(-1)
        o = o.reshape(-1)[sel]
        d = data_bytes.reshape(-1, data_bytes.shape[-1])[sel]
        assert (o >= 0).all() and (o + d.shape[1] <= self.buf.size).all(), "OOB global store"
        idx = o[:, None] + np.arange(d.shape[1])
        self.buf[idx] = d


_INSN = re.compile(r"^(@!?%\w+\s+)?([a-z][\w.]*)\s*(.*?);?$")


def parse(ptx: str):
    body = ptx[ptx.index("{", ptx.index(".entry")) + 1:]
    body = body[:body.index("\n}\n")]
    params = re.findall(r"\.param\s+\.\w+(?:\s+\.ptr\s+\.global\s+\.align\s+\d+)?\s+(\w+)", ptx)
    insns, labels = [], {}
    for raw in body.splitlines():
        line = raw.split("//")[0].strip()
        if not line or line.startswith((".reg", ".loc", ".file", ".section", "{", "}", ".pragma")):
            continue
        if line.endswith(":"):
            labels[line[:-1]] = len(insns)
            continue
        line = line.rstrip(";").strip()
        m = _INSN.match(line)
        if not m:
            raise ValueError(f"unparsed: {line}")
        pred, op, args = m.group(1), m.group(2), m.group(3)
        if pred:
            pred = pred.strip()[1:]
        insns.append((pred, op, args.strip()))
    return params, insns, labels


def split_args(args: str):
    out, depth, cur = [], 0, ""
    for ch in args:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip()); cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


class Emulator:
    def __init__(self, ptx: str, mem: Memory, smem_bytes: int = 65536):
        self.params, self.insns, self.labels = parse(ptx)
        self.mem = mem
        self.smem_bytes = smem_bytes

    def run(self, grid, args, threads=128, cta_chunk=64):
        ctas = [(x, y) for y in range(grid[1]) for x in range(grid[0])]
        for i in range(0, len(ctas), cta_chunk):
            self._run_chunk(ctas[i:i + cta_chunk], args, threads)

    def _run_chunk(self, ctas, args, T):
        C = len(ctas)
        shape = (C, T)
        R = {}
        P = {}
        cx = np.array([c[0] for c in ctas], np.uint64)[:, None] * np.ones(shape, np.uint64)
        cy = np.array([c[1] for c in ctas], np.uint64)[:, None] * np.ones(shape, np.uint64)
        tid = np.arange(T, dtype=np.uint64)[None, :] * np.ones(shape, np.uint64)
        special = {"%tid.x": tid, "%ctaid.x": cx, "%ctaid.y": cy, "%ctaid.z": np.zeros(shape, np.uint64),
                   "%tid.y": np.zeros(shape, np.uint64), "%tid.z": np.zeros(shape, np.uint64),
                   "%ntid.x": np.full(shape, T, np.uint64)}
        smem = np.zeros((C, self.smem_bytes), np.uint8)
        SMEM_SYM = 0x1000  # symbolic base of global_smem in the shared window
        param_val = dict(zip(self.params, args))
        ones = np.ones(shape, bool)

        def val(tok, bits=32):
            tok = tok.strip()
            if tok in special:
                return special[tok]
            if tok == "global_smem":
                return np.full(shape, SMEM_SYM, np.uint64)
            if tok.startswith("%p"):
                return P[tok]
            if tok.startswith("%"):
                return R[tok]
            if tok.startswith("0f"):
                return np.full(shape, int(tok[2:], 16), np.uint64)
            if tok.startswith("0d"):
                return np.full(shape, int(tok[2:], 16), np.uint64)
            v = int(tok.rstrip("U"), 0)
            if v < 0:
                v &= (1 << bits) - 1
            return np.full(shape, v, np.uint64)

        def mask_bits(v, bits):
            return v & np.uint64((1 << bits) - 1) if bits < 64 else v

        def sx(v, bits):
            v = mask_bits(v, bits).astype(np.int64)
            sign = np.int64(1) << np.int64(bits - 1)
            return np.where(v & sign, v - (np.int64(1) << np.int64(bits)), v) if bits < 64 else mask_bits(v, 64).view(np.int64)

        def setr(name, v, pmask):
            if pmask is None:
                R[name] = v
            else:
                old = R.get(name, np.zeros(shape, np.uint64))
                R[name] = np.where(pmask, v, old)

        def addr_of(tok):
            inner = tok.strip()[1:-1].strip()
            parts = [p.strip() for p in re.split(r"\+", inner) if p.strip()]
            base = val(parts[0], 64)
            off = sum(int(p, 0) for p in parts[1:]) if len(parts) > 1 else 0
            return base + np.uint64(off) if off >= 0 else base - np.uint64(-off)

        pc = 0
        n = len(self.insns)
        steps = 0
        while pc < n:
            steps += 1
            assert steps < 2_000_000
            pred, op, args_s = self.insns[pc]
            pc += 1
            pm = None
            if pred:
                neg = pred.startswith("!")
                pm = P[pred.lstrip("!")]
                pm = ~pm if neg else pm
            a = split_args(args_s)
            parts = op.split(".")
            base = parts[0]
            if base == "ret":
                break
            if base == "bra":
                take = pm if pm is not None else ones
                assert take.all() or (~take).all(), "divergent branch"
                if take.all():
                    pc = self.labels[a[0]]
                continue
            if base == "bar":
                continue
            if base == "ld" and parts[1] == "param":
                setr(a[0], np.full(shape, param_val[a[1].strip("[]")], np.uint64), None)
                continue
            if base == "mov":
                typ = parts[1]
                if a[0].startswith("{"):
                    dst = [d.strip() for d in a[0][1:-1].split(",")]
                    v = val(a[1])
                    w = 16 if typ == "b32" else 32
                    for i, d in enumerate(dst):
                        setr(d, (v >> np.uint64(w * i)) & np.uint64((1 << w) - 1), pm)
                    continue
                if a[1].startswith("{"):
                    srcs = [s.strip() for s in a[1][1:-1].split(",")]
                    w = 16 if typ == "b32" else 32
                    v = np.zeros(shape, np.uint64)
                    for i, s in enumerate(srcs):
                        v |= mask_bits(val(s), w) << np.uint64(w * i)
                    setr(a[0], v, pm)
                    continue
                if typ == "pred":
                    P[a[0]] = val(a[1]).astype(bool) if not a[1].startswith("%p") else P[a[1]].copy()
                    continue
                bits = {"u32": 32, "b32": 32, "s32": 32, "b64": 64, "u64": 64, "u16": 16, "b16": 16}[typ]
                setr(a[0], mask_bits(val(a[1], bits), bits), pm)
                continue
            if base == "setp":
                cmp, typ = parts[1], parts[2]
                bits = int(re.sub(r"\D", "", typ))
                x, y = val(a[1], bits), val(a[2], bits)
                if typ.startswith("s"):
                    x, y = sx(x, bits), sx(y, bits)
                elif typ.startswith("f"):
                    x, y = f32(x), f32(y)
                else:
                    x, y = mask_bits(x, bits), mask_bits(y, bits)
                r = {"eq": x == y, "ne": x != y, "lt": x < y, "le": x <= y, "gt": x > y, "ge": x >= y}[cmp]
                if pm is not None:
                    r = np.where(pm, r, P.get(a[0], np.zeros(shape, bool)))
                P[a[0]] = r
                continue
            if base in ("and", "or", "xor", "not") and parts[1] == "pred":
                if base == "not":
                    P[a[0]] = ~P[a[1]]
                else:
                    x, y = P[a[1]], P[a[2]]
                    P[a[0]] = {"and": x & y, "or": x | y, "xor": x ^ y}[base]
                continue
            if base == "selp":
                bits = int(re.sub(r"\D", "", parts[1]))
                setr(a[0], np.where(P[a[3]], mask_bits(val(a[1], bits), bits), mask_bits(val(a[2], bits), bits)), pm)
                continue
            if base in ("add", "sub", "mul", "mad", "and", "or", "xor", "shl", "shr", "min", "max", "neg", "not") and parts[-1] in ("f32",):
                if base == "add":
                    r = f32(val(a[1])) + f32(val(a[2]))
                elif base == "sub":
                    r = f32(val(a[1])) - f32(val(a[2]))
                elif base == "mul":
                    r = f32(val(a[1])) * f32(val(a[2]))
                elif base == "neg":
                    r = -f32(val(a[1]))
                else:
                    raise NotImplementedError(op)
                setr(a[0], u64(r), pm)
                continue
            if base == "fma":
                assert parts[1] == "rn" and parts[2] == "f32", op
                setr(a[0], u64(fma32(f32(val(a[1])), f32(val(a[2])), f32(val(a[3])))), pm)
                continue
            if base in ("add", "sub", "and", "or", "xor", "shl", "shr", "mul", "mad", "not", "min", "max"):
                typ = parts[-1]
                bits = int(re.sub(r"\D", "", typ))
                signed = typ.startswith("s")
                if base == "mul" or base == "mad":
                    mode = parts[1]
                    x, y = val(a[1], bits), val(a[2], bits)
                    if signed:
                        p = sx(x, bits) * sx(y, bits)
                    else:
                        p = mask_bits(x, bits).astype(np.int64) * mask_bits(y, bits).astype(np.int64)
                    if mode == "wide":
                        obits = bits * 2
                    else:
                        obits = bits
                    if base == "mad":
                        p = p + val(a[3], obits).astype(np.int64)
                    setr(a[0], mask_bits(p.astype(np.uint64), obits), pm)
                    continue
                if base == "not":
                    setr(a[0], mask_bits(~val(a[1], bits), bits), pm)
                    continue
                x, y = val(a[1], bits), val(a[2], bits)
                if base == "add":
                    r = x + y
                elif base == "sub":
                    r = x - y
                elif base == "and":
                    r = x & y
                elif base == "or":
                    r = x | y
                elif base == "xor":
                    r = x ^ y
                elif base == "shl":
                    sh = mask_bits(y, 32)
                    r = np.where(sh >= bits, 0, mask_bits(x, bits) << np.minimum(sh, np.uint64(63)))
                elif base == "shr":
                    sh = mask_bits(y, 32).astype(np.int64)
                    if signed:
                        r = (sx(x, bits) >> np.minimum(sh, bits - 1)).astype(np.uint64)
                    else:
                        r = np.where(sh >= bits, 0, mask_bits(x, bits) >> np.minimum(sh, 63).astype(np.uint64))
                elif base in ("min", "max"):
                    xs, ys = (sx(x, bits), sx(y, bits)) if signed else (mask_bits(x, bits), mask_bits(y, bits))
                    r = (np.minimum(xs, ys) if base == "min" else np.maximum(xs, ys)).astype(np.uint64)
                setr(a[0], mask_bits(r.astype(np.uint64), bits), pm)
                continue
            if base == "bfe":
                bits = int(re.sub(r"\D", "", parts[1]))
                x = mask_bits(val(a[1], bits), bits)
                pos = mask_bits(val(a[2]), 8).astype(np.int64)
                ln = mask_bits(val(a[3]), 8).astype(np.int64)
                field = (x >> pos.astype(np.uint64)) & ((np.uint64(1) << ln.astype(np.uint64)) - np.uint64(1))
                if parts[1].startswith("s"):
                    top = np.uint64(1) << (ln - 1).clip(0).astype(np.uint64)
                    field = np.where((ln > 0) & (field & top != 0),
                                     field | ~((np.uint64(1) << ln.astype(np.uint64)) - np.uint64(1)), field)
                setr(a[0], mask_bits(field, bits), pm)
                continue
            if base == "prmt":
                x = mask_bits(val(a[1]), 32); y = mask_bits(val(a[2]), 32)
                sel = int(a[3].rstrip("U"), 0) if not a[3].startswith("%") else None
                assert sel is not None and len(parts) == 2, "prmt mode"
                src = x | (y << np.uint64(32))
                r = np.zeros(shape, np.uint64)
                for i in range(4):
                    nib = (sel >> (4 * i)) & 0xF
                    byte = (src >> np.uint64(8 * (nib & 7))) & np.uint64(0xFF)
                    if nib & 8:
                        byte = np.where(byte & np.uint64(0x80), np.uint64(0xFF), np.uint64(0))
                    r |= byte << np.uint64(8 * i)
                setr(a[0], r, pm)
                continue
            if base == "cvt":
                dt, st = parts[-2], parts[-1]
                if "rn" in parts or dt.startswith("f") or st.startswith("f"):
                    raise NotImplementedError(op)
                sb = int(re.sub(r"\D", "", st)); db = int(re.sub(r"\D", "", dt))
                v = val(a[1], sb)
                v = sx(v, sb).astype(np.uint64) if st.startswith("s") else mask_bits(v, sb)
                setr(a[0], mask_bits(v, db), pm)
                continue
            if base == "shfl":
                assert parts[2] == "bfly", op
                x = val(a[1]); lm = int(a[2].rstrip("U"), 0)
                lane = np.arange(T)
                src = (lane & ~31) | ((lane & 31) ^ lm)
                setr(a[0], x[:, src], pm)
                continue
            if base in ("ld", "st"):
                space = parts[1]
                vec = 1
                rest = parts[2:]
                # strip cache qualifiers
                rest = [p for p in rest if p not in ("ca", "cg", "cs", "lu", "cv", "nc", "L1::evict_last", "L1::evict_first")]
                if rest and rest[0].startswith("v"):
                    vec = int(rest[0][1:]); rest = rest[1:]
                w = int(re.sub(r"\D", "", rest[-1])) // 8
                if base == "ld":
                    dst = [d.strip() for d in a[0].strip("{} ").split(",")]
                    addr = addr_of(a[1])
                    m = pm if pm is not None else ones
                    if space == "global":
                        raw = self.mem.gather(addr, w * vec, m)
                    else:
                        o = (addr.astype(np.int64) - SMEM_SYM)
                        o = np.where(m, o, 0)
                        idx = o[..., None] + np.arange(w * vec)
                        raw = smem[np.arange(C)[:, None, None], idx]
                    raw = raw.reshape(C, T, vec, w)
                    for i, d in enumerate(dst):
                        v = np.zeros(shape, np.uint64)
                        for b in range(w):
                            v |= raw[:, :, i, b].astype(np.uint64) << np.uint64(8 * b)
                        setr(d, v, m)
                else:
                    srcs = [s.strip() for s in a[1].strip("{} ").split(",")]
                    addr = addr_of(a[0])
                    m = pm if pm is not None else ones
                    data = np.zeros((C, T, vec * w), np.uint8)
                    for i, s in enumerate(srcs):
                        v = val(s, w * 8)
                        for b in range(w):
                            data[:, :, i * w + b] = ((v >> np.uint64(8 * b)) & np.uint64(0xFF)).astype(np.uint8)
                    if space == "global":
                        self.mem.scatter(addr, data, m)
                    else:
                        o = (addr.astype(np.int64) - SMEM_SYM)
                        for c in range(C):
                            sel = m[c]
                            idx = o[c][sel][:, None] + np.arange(vec * w)
                            smem[c][idx] = data[c][sel]
                continue
            raise NotImplementedError(f"{op} {args_s}")
