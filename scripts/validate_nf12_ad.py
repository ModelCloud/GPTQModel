#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from the nf12 reference implementation
# (https://github.com/dphnAI/nf12) by AlpinDale and the nf12 contributors.
"""NF12-AD accuracy + performance validation with full metric tables.

NF12 (NyoomFloat12) is Together AI's lossless 12-bit BF16 weight format
(Liberman, Fang, Zhang, Dao, Fu, 2026). The reference implementation and CUDA
kernels are AlpinDale's at https://github.com/dphnAI/nf12.

This script is the reproducible evidence behind the PR's accuracy and
performance claims. It never reduces a result to pass/fail: every metric is
recorded and printed.

Usage
-----
    # NVIDIA (CUDA kernels)
    python scripts/validate_nf12_ad.py --device cuda:0
    # AMD RDNA4 (HIP decode port)
    python scripts/validate_nf12_ad.py --device cuda:0 --backend hip
    python scripts/validate_nf12_ad.py --json-out /tmp/nf12_4090.json

Accuracy criterion
------------------
Comparing raw relative error is wrong near zero: for an output element whose
true value is 1e-9, an absolute error of 1e-9 is 100% "relative error" while
being numerically irrelevant. Every kernel here also rounds its output to
BF16, whose own spacing at that magnitude is far larger. The criterion used
below is therefore the standard elementwise bound

    |got - ref| <= atol + rtol * |ref|

evaluated on the full float64 difference, plus three scale-aware summaries:

* ``max_abs``      worst absolute deviation (must be <= atol).
* ``max_rel_sig``  worst relative error restricted to elements at or above 1%
                   of max|ref| -- the "meaningful" population.
* ``rms_rel``      RMS(|err|) / RMS(|ref|) -- the headline relative number;
                   it cannot be inflated by near-zero denominators.
* ``exact%``       fraction of bit-identical BF16 elements (expected high:
                   NF12 is a lossless *weight* format and decode is exact, so
                   any remaining drift is the GEMM's FP32 accumulation order).

``max_rel_all`` is still printed for transparency, but it is *not* a gate: it
diverges on near-zero elements by construction.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
import zlib

import torch


ATOL = 2e-3
RTOL = 2e-3


def banner(text: str) -> None:
    print(f"\n=== {text} ===", flush=True)


def accuracy_metrics(got: torch.Tensor, ref: torch.Tensor) -> dict:
    """Full drift report for one output tensor, computed in float64."""
    g = got.double()
    r = ref.double()
    err = (g - r).abs()

    max_abs = err.max().item() if err.numel() else 0.0
    scale = r.abs().max().item() if r.numel() else 0.0

    # Relative error, and the same restricted to the meaningful population.
    denom = r.abs().clamp_min(torch.finfo(torch.float64).tiny)
    rel = err / denom
    significant = r.abs() >= 0.01 * max(scale, torch.finfo(torch.float64).tiny)
    max_rel_sig = rel[significant].max().item() if significant.any() else 0.0

    # Scale-aware headline: RMS error over RMS signal.
    rms_ref = r.pow(2).mean().sqrt().item()
    rms_err = err.pow(2).mean().sqrt().item()
    rms_rel = rms_err / rms_ref if rms_ref > 0 else 0.0

    flat_g, flat_r = g.reshape(-1), r.reshape(-1)
    denom_norm = flat_g.norm().item() * flat_r.norm().item()
    cos = (flat_g @ flat_r).item() / denom_norm if denom_norm > 0 else 1.0

    # Bit-identical BF16 fraction. Compare as int16 so +0/-0 and NaN payloads
    # are distinguished exactly as stored.
    if got.dtype == torch.bfloat16 and ref.dtype == torch.bfloat16:
        exact = (got.view(torch.int16) == ref.view(torch.int16)).float().mean().item()
    else:
        exact = (g == r).float().mean().item()

    # The acceptance gate: elementwise combined bound, near-zero safe.
    tol = ATOL + RTOL * r.abs()
    n_violations = int((err > tol).sum().item())
    n_inf = int((~torch.isfinite(g)).sum().item())

    return {
        "max_abs": max_abs,
        "scale": scale,
        "max_rel_all": rel.max().item() if rel.numel() else 0.0,
        "max_rel_sig": max_rel_sig,
        "rms_rel": rms_rel,
        "cos": cos,
        "exact": exact,
        "ulp_of_scale": _ulp(scale),
        "max_abs_ulps": max_abs / _ulp(scale) if scale else 0.0,
        "violations": n_violations,
        "nonfinite": n_inf,
        "ok": n_violations == 0 and n_inf == 0,
    }


def _ulp(value: float) -> float:
    """Spacing between consecutive BF16 values near ``value`` (7-bit mantissa)."""
    if value == 0 or not math.isfinite(value):
        return 2.0 ** -133
    exponent = math.floor(math.log2(abs(value)))
    return 2.0 ** (exponent - 7)


def oracle(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Independent FP32 oracle: the math any correct NF12 kernel must reproduce."""
    y = torch.matmul(x.float(), w.float().t())
    if bias is not None:
        y = y + bias.float()
    return y.to(torch.bfloat16)


def make_weight(n: int, k: int, kind: str, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if kind == "normal":
        return (torch.randn(n, k, generator=generator) * 0.02).to(torch.bfloat16)
    if kind == "uniform":
        return ((torch.rand(n, k, generator=generator) - 0.5) * 0.05).to(torch.bfloat16)
    if kind == "wide_exp":
        scale = torch.logspace(-8, 3, n).unsqueeze(1)
        return (torch.randn(n, k, generator=generator) * scale).to(torch.bfloat16)
    if kind == "extremes":
        bits = torch.randint(0, 1 << 16, (n, k), generator=generator, dtype=torch.int32)
        return bits.to(torch.int16).view(torch.bfloat16)
    if kind == "zeros":
        return torch.zeros(n, k, dtype=torch.bfloat16)
    raise ValueError(kind)


def backend_module(backend: str, device: str):
    if torch.device(device).type != "cuda":
        if backend == "auto":
            return None, "torch"
        raise ValueError(f"--backend {backend} requires a CUDA/HIP device, got {device}")

    from gptqmodel.utils import nf12_ad as cuda_rt

    if backend == "hip":
        from gptqmodel.utils import nf12_ad_hip as hip

        if not hip.prewarm_nf12_ad_hip_extension():
            raise RuntimeError(f"NF12-AD HIP extension unavailable: {hip.last_error()}")
        return hip, "hip"

    if cuda_rt.load():
        return cuda_rt, "cuda"
    if backend == "cuda":
        reason = cuda_rt.last_error() or "the native extension did not load"
        raise RuntimeError(f"NF12-AD CUDA extension unavailable: {reason}")
    if torch.version.hip is not None:
        from gptqmodel.utils import nf12_ad_hip as hip

        if hip.prewarm_nf12_ad_hip_extension():
            return hip, "hip"
    return None, "torch"


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------
def run_accuracy(device: str, backend: str, quick: bool, results: dict) -> bool:
    from gptqmodel.utils import nf12_codec as codec

    banner("extension")
    rt, active = backend_module(backend, device)
    results["backend"] = active
    print(f"active backend: {active}", flush=True)
    if rt is None:
        print("  no native kernel; only the Torch codec oracle can be checked", flush=True)

    shapes = [(16, 64), (32, 128), (256, 256), (1024, 512), (4096, 1024)]
    if not quick:
        shapes += [(4096, 4096), (8192, 2048), (11008, 4096)]
    kinds = ["normal", "uniform", "wide_exp", "extremes", "zeros"]

    banner("A. codec: native encode/decode vs pure-Torch oracle")
    all_ok = True
    for n, k in shapes:
        for kind in kinds:
            # Python's hash is salted per process; a stable seed keeps accuracy
            # inputs comparable across runs and backend revisions.
            seed = zlib.crc32(f"{n}:{k}:{kind}".encode("ascii"))
            w = make_weight(n, k, kind, seed=seed)
            wc = w.to(device).contiguous()
            ref_packed, ref_esc, ref_base = codec.encode(w, None)
            row = {"test": "codec", "kind": kind, "n": n, "k": k,
                   "ref_exp_base": int(ref_base), "escapes": int(ref_esc.shape[0]),
                   "escape_pct": round(ref_esc.shape[0] / max(n * k / 8, 1) * 100, 4)}

            if rt is not None and active == "cuda":
                packed, esc, base = rt.encode(wc)
                native = rt.decode(packed, esc, base)
                row.update(
                    exp_base=int(base),
                    bytes_match=bool(torch.equal(packed.cpu().reshape(-1), ref_packed.reshape(-1))),
                    escapes_match=bool(torch.equal(esc.cpu(), ref_esc)),
                    decode_bit_exact=bool(torch.equal(native.cpu().view(torch.int16), w.view(torch.int16))),
                    packed_shape=list(packed.shape), packed_dtype=str(packed.dtype),
                )
                if not (row["bytes_match"] and row["escapes_match"] and row["decode_bit_exact"]):
                    all_ok = False
            elif rt is not None and active == "hip":
                # The HIP port ships its own analyze/scan/write encoder, so this
                # exercises the native kernel rather than reusing the Torch
                # codec's bytes. A regression here means the GPU encoder drifted
                # from the reference bit layout.
                packed, esc, base = rt.encode(wc)
                native = rt.decode(packed, esc, base)
                row.update(
                    exp_base=int(base),
                    bytes_match=bool(torch.equal(packed.cpu().reshape(-1), ref_packed.reshape(-1))),
                    escapes_match=bool(torch.equal(esc.cpu(), ref_esc)),
                    decode_bit_exact=bool(torch.equal(native.cpu().view(torch.int16), w.view(torch.int16))),
                    packed_shape=list(packed.shape), packed_dtype=str(packed.dtype),
                )
                if not (row["bytes_match"] and row["escapes_match"] and row["decode_bit_exact"]):
                    all_ok = False
            else:
                t = codec.encode(w)
                d = codec.decode(*t)
                row.update(exp_base=int(t[2]), bytes_match=True, escapes_match=True,
                           decode_bit_exact=bool(torch.equal(d.view(torch.int16), w.view(torch.int16))))
            results["accuracy"].append(row)

    if rt is None:
        return all_ok

    def encode(wc):
        if active == "cuda":
            return rt.encode(wc)
        p, e, b = codec.encode(wc.cpu(), None)
        return p.to(device), e.to(device), int(b)

    def linear(x, packed, esc, base, bias=None):
        return rt.linear(x, packed, esc, base, bias)

    banner("B. fused linear vs independent FP32 matmul oracle")
    for n, k in shapes:
        w = make_weight(n, k, "normal", seed=7)
        wc = w.to(device).contiguous()
        packed, esc, base = encode(wc)
        for m in (1, 2, 8, 16, 64):
            if quick and m not in (1, 16):
                continue
            x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.05
            met = accuracy_metrics(linear(x, packed, esc, base), oracle(x, wc))
            results["accuracy"].append({"test": "linear", "n": n, "k": k, "m": m,
                                        "exp_base": int(base), **met})
            all_ok = all_ok and met["ok"]

    banner("C. fused linear + bias")
    n, k, m = 1024, 1024, 16
    w = make_weight(n, k, "normal", seed=11).to(device).contiguous()
    packed, esc, base = encode(w)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.05
    bias = torch.randn(n, device=device, dtype=torch.bfloat16) * 0.1
    met = accuracy_metrics(linear(x, packed, esc, base, bias), oracle(x, w, bias))
    results["accuracy"].append({"test": "linear+bias", "n": n, "k": k, "m": m, **met})
    all_ok = all_ok and met["ok"]

    if active == "cuda":
        banner("D. bf16t twin")
        w = make_weight(512, 512, "normal", 13).to(device).contiguous()
        twin = torch.ops.nf12.pack_bf16t(w)
        x = torch.randn(32, 512, device=device, dtype=torch.bfloat16) * 0.05
        met = accuracy_metrics(torch.ops.nf12.linear_bf16t(x, twin), oracle(x, w))
        results["accuracy"].append({"test": "bf16t", "n": 512, "k": 512, "m": 32,
                                    "twin_ratio": round(twin.numel() / (512 * 512 * 2), 4), **met})
        all_ok = all_ok and met["ok"]

    banner("E. NF12ADLinear module end-to-end")
    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear
    from gptqmodel.quantization.nf12_ad import quantize_nf12_ad_weight

    for n, k in [(256, 256), (1024, 4096)]:
        lin = NF12ADLinear(bits=12, in_features=k, out_features=n, bias=True)
        w = make_weight(n, k, "normal", 17)
        lin.load_quant_result(quantize_nf12_ad_weight(w))
        lin = lin.to(device=device, dtype=torch.bfloat16)
        x = torch.randn(8, k, device=device, dtype=torch.bfloat16) * 0.05
        met = accuracy_metrics(lin(x), oracle(x, w.to(device), lin.bias))
        keys = sorted(lin.state_dict().keys())
        results["accuracy"].append({"test": "module", "n": n, "k": k, "m": 8,
                                    "state_keys": ",".join(keys),
                                    "has_dense_weight": "weight" in keys, **met})
        all_ok = all_ok and met["ok"]

    return all_ok


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
def timed(fn, device: str, warmup: int = 15, iters: int = 60, reps: int = 5):
    """Return (median, min) microseconds per call, timed with CUDA events.

    Timing one call between two ``torch.cuda.synchronize`` calls measures the
    launch round-trip, not the kernel: on this host that is ~30 us, which is
    larger than a 4096x4096 decode at m=16 and completely hides the difference
    between NF12 and BF16 at small m. Batch ``iters`` calls between a pair of
    recorded events instead, and take the best of several batches. This is the
    same method as ``bench/hip_linear.py``.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / iters)  # ms -> us
    return statistics.median(samples), min(samples)


def measure_bandwidth(device: str) -> float:
    """Empirical HBM read+write bandwidth from a large device-to-device copy."""
    n = 512 * 1024 * 1024 // 4
    a = torch.empty(n, dtype=torch.float32, device=device)
    b = torch.empty(n, dtype=torch.float32, device=device)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(10):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        b.copy_(a)
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - start)
    del a, b
    torch.cuda.empty_cache()
    return (2 * n * 4) / min(samples) / 1e9


def measure_cache_size(device: str, repetitions: int = 20) -> tuple[float, list]:
    """Derive the *effective* last-level cache size from a residency curve.

    ``L2_cache_size`` under-reports on RDNA4: the R9700 has a 64 MiB Infinity
    Cache (MALL) but torch reports 8 MiB, so classifying "L2-resident" from the
    property alone marks 24 MiB working sets as HBM-bound and produces >170% of
    "measured bandwidth" with no explanation. Sweeping buffer size finds the
    real knee: reads stay multi-TB/s below it and fall to HBM speed above it.
    """
    curve = []
    for mib in (4, 8, 16, 24, 32, 48, 64, 80, 96, 128):
        n = mib * 1024 * 1024 // 4
        buffer = torch.randn(n, device=device)
        for _ in range(2):
            buffer.sum()
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(8):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(repetitions):
                buffer.sum()
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - start) / repetitions)
        gbs = n * 4 / min(samples) / 1e9
        curve.append((mib, round(gbs, 1)))
        del buffer
        torch.cuda.empty_cache()

    # The knee is the largest size whose read rate still beats HBM by 1.5x.
    hbm = curve[-1][1]
    effective = 0.0
    for mib, gbs in curve:
        if gbs > 1.5 * hbm:
            effective = float(mib)
    return effective, curve


def run_perf(device: str, backend: str, quick: bool, results: dict) -> None:
    rt, active = backend_module(backend, device)
    results["backend"] = active
    if rt is None:
        results["notes"].append({"item": "perf skipped", "error": "no native kernel"})
        return

    props = torch.cuda.get_device_properties(device)
    name = torch.cuda.get_device_name(device)
    bandwidth = measure_bandwidth(device)
    l2 = getattr(props, "L2_cache_size", 0)
    effective_cache, cache_curve = measure_cache_size(device)
    results["device"] = {
        "name": name,
        "sm": f"{props.major}.{props.minor}",
        "vram_gib": round(props.total_memory / 2**30, 1),
        "sm_count": props.multi_processor_count,
        "l2_mib": round(l2 / 2**20, 1) if l2 else None,
        "effective_cache_mib": effective_cache,
        "cache_curve": cache_curve,
        "measured_bw_gbs": round(bandwidth, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "backend": results.get("backend"),
    }

    banner("F. decode throughput")
    for n, k in ([(4096, 4096)] if quick else [(2048, 2048), (4096, 4096), (8192, 2048), (11008, 4096)]):
        w = (torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.02).contiguous()
        packed, esc, base = rt.encode(w)
        packed_bytes = packed.numel() + esc.numel() * 2 + 1
        out_bytes = n * k * 2
        median_us, min_us = timed(lambda: rt.decode(packed, esc, base), device)
        total_gbs = (packed_bytes + out_bytes) / (min_us * 1e-6) / 1e9
        results["perf"].append({
            "test": "decode", "n": n, "k": k,
            "median_us": round(median_us, 2), "min_us": round(min_us, 2),
            "packed_mib": round(packed_bytes / 2**20, 2), "out_mib": round(out_bytes / 2**20, 2),
            "read_gbs": round(packed_bytes / (min_us * 1e-6) / 1e9, 1),
            "write_gbs": round(out_bytes / (min_us * 1e-6) / 1e9, 1),
            "total_gbs": round(total_gbs, 1),
            "pct_of_measured_bw": round(total_gbs / bandwidth * 100, 1),
            "footprint_mib": round((packed_bytes + out_bytes) / 2**20, 1),
            "l2_resident": bool(effective_cache and (packed_bytes + out_bytes) <= effective_cache * 2**20),
        })

    banner("G. fused linear vs BF16 torch.matmul")
    shapes = [(4096, 4096), (11008, 4096), (4096, 11008), (2048, 2048)]
    if quick:
        shapes = shapes[:2]
    for n, k in shapes:
        w = (torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.02).contiguous()
        packed, esc, base = rt.encode(w)
        packed_bytes = packed.numel() + esc.numel() * 2 + 1
        for m in (1, 2, 8, 16, 64, 128, 256):
            x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.05
            t_nf, _ = timed(lambda: rt.linear(x, packed, esc, base), device)
            t_bf, _ = timed(lambda: torch.matmul(x, w.t()), device)
            flops = 2 * m * n * k
            results["perf"].append({
                "test": "linear", "n": n, "k": k, "m": m,
                "nf12_us": round(t_nf, 2), "bf16_us": round(t_bf, 2),
                "speedup": round(t_bf / t_nf, 3),
                "nf12_gbs": round((packed_bytes + m * n * 2) / (t_nf * 1e-6) / 1e9, 1),
                "nf12_tflops": round(flops / (t_nf * 1e-6) / 1e12, 2),
                "bf16_tflops": round(flops / (t_bf * 1e-6) / 1e12, 2),
                "weight_mib": round(n * k * 2 / 2**20, 1),
            })

    banner("H. encode throughput")
    w = (torch.randn(4096, 4096, device=device, dtype=torch.bfloat16) * 0.02).contiguous()
    median_us, min_us = timed(lambda: rt.encode(w), device, warmup=5, iters=25)
    results["perf"].append({"test": "encode", "n": 4096, "k": 4096,
                            "median_us": round(median_us, 2), "min_us": round(min_us, 2),
                            "gbs": round(w.numel() * 2 / (min_us * 1e-6) / 1e9, 1)})


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def table(headers, rows) -> str:
    if not rows:
        return ""
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    lines = ["  ".join(str(h).ljust(w) for h, w in zip(headers, widths)),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(lines)


def g(x, p: int = 4) -> str:
    return "0" if x == 0 else f"{x:.{p}g}"


def print_reports(results: dict) -> None:
    device = results.get("device") or {}
    print("\n" + "=" * 118)
    if device:
        cache = device.get("effective_cache_mib") or device.get("l2_mib")
        print(f"DEVICE  {device['name']}  SM{device['sm']}  {device['vram_gib']} GiB  "
              f"{device['sm_count']} SM  effective cache {cache:g} MiB")
        print(f"        torch {device['torch']}  cuda {device['cuda']}  hip {device['hip']}  "
              f"backend {device['backend']}  measured BW {device['measured_bw_gbs']} GB/s")
    print(f"CRITERION  |got-ref| <= {ATOL:g} + {RTOL:g}*|ref|  (elementwise, float64)")
    print("=" * 118)

    codec_rows = [a for a in results["accuracy"] if a["test"] == "codec"]
    if codec_rows:
        print("\n--- A. codec bit-exactness (native vs pure-Torch oracle) ---")
        print(table(
            ["kind", "N", "K", "base", "escapes", "esc%", "bytes=", "escapes=", "decode=bits"],
            [[a["kind"], a["n"], a["k"], a["exp_base"], a["escapes"], f"{a['escape_pct']:.2f}",
              a["bytes_match"], a["escapes_match"], a["decode_bit_exact"]] for a in codec_rows]))

    linear_rows = [a for a in results["accuracy"] if a["test"] == "linear"]
    if linear_rows:
        print("\n--- B. fused linear vs independent FP32 oracle ---")
        print(table(
            ["N", "K", "M", "max_abs", "scale", "max_abs/ULP", "max_rel_sig", "rms_rel",
             "cos", "exact%", "viol", "ok"],
            [[a["n"], a["k"], a["m"], g(a["max_abs"]), g(a["scale"]), f"{a['max_abs_ulps']:.2f}",
              g(a["max_rel_sig"]), g(a["rms_rel"]), f"{a['cos']:.8f}",
              f"{a['exact'] * 100:.2f}", a["violations"], "yes" if a["ok"] else "NO"]
             for a in linear_rows]))
        worst_rel = max(linear_rows, key=lambda a: a["max_rel_all"])
        print(f"  note: max_rel_all is not a gate (near-zero denominators); worst here is "
              f"{g(worst_rel['max_rel_all'])} at N={worst_rel['n']} K={worst_rel['k']} "
              f"M={worst_rel['m']} where max_abs is only {g(worst_rel['max_abs'])} "
              f"({worst_rel['max_abs_ulps']:.2f} ULP of scale).")

    for tag, title in (("linear+bias", "C. fused linear + bias"),
                       ("bf16t", "D. bf16t twin"),
                       ("module", "E. NF12ADLinear end-to-end")):
        rows = [a for a in results["accuracy"] if a["test"] == tag]
        if not rows:
            continue
        print(f"\n--- {title} ---")
        headers = ["N", "K", "M", "max_abs", "max_abs/ULP", "max_rel_sig", "rms_rel", "cos", "exact%", "ok"]
        print(table(headers, [[a["n"], a["k"], a["m"], g(a["max_abs"]), f"{a['max_abs_ulps']:.2f}",
                               g(a["max_rel_sig"]), g(a["rms_rel"]), f"{a['cos']:.8f}",
                               f"{a['exact'] * 100:.2f}", "yes" if a["ok"] else "NO"] for a in rows]))
        if tag == "module":
            print(f"  state keys: {rows[0]['state_keys']}   dense weight present: "
                  f"{rows[0]['has_dense_weight']}")

    if device:
        print(f"\n--- F/G. performance (measured D2D bandwidth {device['measured_bw_gbs']} GB/s"
              + (f", effective cache {device['effective_cache_mib']:g} MiB"
                 if device.get("effective_cache_mib") else "") + ") ---")
        decode_rows = [p for p in results["perf"] if p["test"] == "decode"]
        if decode_rows:
            print("\n  decode (NF12 -> BF16):")
            print(table(
                ["N", "K", "in MiB", "out MiB", "med us", "min us", "read GB/s", "write GB/s",
                 "total GB/s", "%meas BW", "L2"],
                [[p["n"], p["k"], p["packed_mib"], p["out_mib"], p["median_us"], p["min_us"],
                  p["read_gbs"], p["write_gbs"], p["total_gbs"], p["pct_of_measured_bw"],
                  "yes" if p["l2_resident"] else "no"] for p in decode_rows]))

        linear_perf = [p for p in results["perf"] if p["test"] == "linear"]
        if linear_perf:
            print("\n  fused linear vs BF16 torch.matmul (speedup > 1.0 favors NF12):")
            print(table(
                ["N", "K", "M", "NF12 us", "BF16 us", "speedup", "NF12 GB/s", "NF12 TF/s",
                 "BF16 TF/s", "weight MiB"],
                [[p["n"], p["k"], p["m"], p["nf12_us"], p["bf16_us"], p["speedup"], p["nf12_gbs"],
                  p["nf12_tflops"], p["bf16_tflops"], p["weight_mib"]] for p in linear_perf]))
            wins = sum(1 for p in linear_perf if p["speedup"] >= 1.0)
            print(f"  {wins}/{len(linear_perf)} shapes at or above BF16 parity; "
                  f"best {max(p['speedup'] for p in linear_perf):.2f}x, "
                  f"worst {min(p['speedup'] for p in linear_perf):.2f}x")

        encode_rows = [p for p in results["perf"] if p["test"] == "encode"]
        if encode_rows:
            p = encode_rows[0]
            print(f"\n  encode 4096x4096: median {p['median_us']} us  min {p['min_us']} us  "
                  f"{p['gbs']} GB/s")

    for note in results["notes"]:
        print(f"\nNOTE: {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("auto", "cuda", "hip"), default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument("--perf-only", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    results = {"device": {}, "accuracy": [], "perf": [], "notes": [], "backend": None}

    ok = True
    if not args.perf_only:
        try:
            ok = run_accuracy(args.device, args.backend, args.quick, results)
        except Exception:
            traceback.print_exc()
            results["notes"].append({"item": "accuracy crashed",
                                     "trace": traceback.format_exc()[-2000:]})
            ok = False
    if not args.accuracy_only:
        try:
            run_perf(args.device, args.backend, args.quick, results)
        except Exception:
            traceback.print_exc()
            results["notes"].append({"item": "perf crashed", "trace": traceback.format_exc()[-2000:]})
            ok = False

    results["accuracy_ok"] = ok
    print_reports(results)
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(results, handle, indent=2, default=str)
        print(f"\nwrote {args.json_out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
