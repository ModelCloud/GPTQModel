#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Turn `nsys stats` CSV exports from profile_qvq_quantize_nsys.sh into Markdown attribution tables.

    python scripts/summarize_qvq_nsys_stats.py artifacts/nsys llama32_1b_full > /tmp/tables.md

Reads <prefix>_cuda_gpu_kern_sum.csv, _cuda_gpu_sum.csv, _cuda_api_sum.csv, _cuda_gpu_mem_time_sum.csv,
_cuda_gpu_mem_size_sum.csv, _nvtx_kern_sum.csv, _nvtx_pushpop_sum.csv, _nvtx_gpu_proj_sum.csv,
_cuda_kern_exec_sum.csv, _osrt_sum.csv and <prefix>_host_attribution.json.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

NS = 1e9


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        lines = [line for line in handle if not line.startswith("**")]
    return list(csv.DictReader(lines))


def short_kernel(name: str) -> str:
    name = re.sub(r"^void ", "", name)
    name = name.replace("<unnamed>::", "")
    name = re.sub(r"\(const.*$", "", name)
    name = re.sub(r"\(T\d.*$", "", name)
    return name.strip()[:110]


def fmt_s(ns: float) -> str:
    return f"{ns / NS:.2f}"


def fmt_ms(ns: float) -> str:
    return f"{ns / 1e6:.3f}"


def main() -> int:
    art = Path(sys.argv[1])
    prefix = sys.argv[2]
    p = lambda suffix: art / f"{prefix}_{suffix}.csv"  # noqa: E731
    kern = read_csv(p("cuda_gpu_kern_sum"))
    gpu_sum = read_csv(p("cuda_gpu_sum"))
    api = read_csv(p("cuda_api_sum"))
    mem_time = read_csv(p("cuda_gpu_mem_time_sum"))
    mem_size = read_csv(p("cuda_gpu_mem_size_sum"))
    nvtx_kern = read_csv(p("nvtx_kern_sum"))
    pushpop = read_csv(p("nvtx_pushpop_sum"))
    proj = read_csv(p("nvtx_gpu_proj_sum"))
    osrt = read_csv(p("osrt_sum"))
    host_json = art / f"{prefix}_host_attribution.json"
    host = json.loads(host_json.read_text()) if host_json.exists() else {}

    total_kernel_ns = sum(float(r["Total Time (ns)"]) for r in kern)
    total_gpu_ns = sum(float(r["Total Time (ns)"]) for r in gpu_sum)
    mem_ns = sum(float(r["Total Time (ns)"]) for r in mem_time)
    wall = host.get("wall_seconds_qvq_quantize_main")
    out = []
    w = out.append

    w("### Totals\n")
    w("| metric | value |\n|---|---|")
    if wall:
        w(f"| `qvq_quantize.main` wall (host, incl. load/save) | {wall:.1f} s |")
    w(f"| total CUDA kernel time (`cuda_gpu_kern_sum`) | {fmt_s(total_kernel_ns)} s |")
    w(f"| total CUDA GPU time incl. memops (`cuda_gpu_sum`) | {fmt_s(total_gpu_ns)} s |")
    w(f"| total memcpy/memset time (`cuda_gpu_mem_time_sum`) | {fmt_s(mem_ns)} s |")
    if wall:
        w(f"| kernel time / wall | {total_kernel_ns / NS / wall * 100:.1f} % |")
    w("")

    # Per NVTX range: kernel time inside range (nvtx_kern_sum) — sums kernels attributed to each range.
    range_kernel_ns: dict[str, float] = defaultdict(float)
    range_kernel_inst: dict[str, int] = defaultdict(int)
    range_kernels: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for r in nvtx_kern:
        name = r["NVTX Range"].lstrip(":")
        ns = float(r["Total Time (ns)"])
        inst = int(r["Kern Inst"])
        range_kernel_ns[name] += ns
        range_kernel_inst[name] += inst
        range_kernels[name].append((short_kernel(r["Kernel Name"]), ns, inst))
    pushpop_by = {r["Range"].lstrip(":"): r for r in pushpop}
    proj_by = {r["Range"].lstrip(":"): r for r in proj}

    w("### Viterbi / codec variants invoked by the real workload\n")
    w("Kernel time is the sum of CUDA kernels that executed inside the variant's NVTX range (`nvtx_kern_sum`); "
      "`% GPU` is relative to total kernel time. `host avg/max` are host-side durations of the op call "
      "(launch + any sync inside the op) from the Python wrapper.\n")
    w("| NVTX range (variant) | calls | kernel time (s) | % GPU kernel time | kernel avg / call (ms) | kernel launches | host avg (ms) | host max (ms) |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    ranges = sorted((n for n in set(range_kernel_ns) | set(pushpop_by) if n.startswith(("qvq_cuda.", "qvq_api.", "qvq_cpu."))),
                    key=lambda n: -range_kernel_ns.get(n, 0.0))
    host_ranges = host.get("ranges", {})
    for name in ranges:
        pp = pushpop_by.get(name)
        calls = int(pp["Instances"]) if pp else host_ranges.get(name, {}).get("calls", 0)
        kns = range_kernel_ns.get(name, 0.0)
        hr = host_ranges.get(name, {})
        w(f"| `{name}` | {calls} | {fmt_s(kns)} | {kns / total_kernel_ns * 100:.1f} | "
          f"{(kns / calls / 1e6) if calls else 0:.3f} | {range_kernel_inst.get(name, 0)} | "
          f"{hr.get('host_seconds_avg', 0) * 1e3:.3f} | {hr.get('host_seconds_max', 0) * 1e3:.1f} |")
    w("")
    all_variants = ["viterbi", "viterbi_trusted", "viterbi_tail_trusted", "viterbi_v4", "viterbi_banked",
                    "viterbi_v2_segment_banked", "viterbi_v2_segment_g", "viterbi_v2_segment_grid",
                    "viterbi_v2_segment_grid_trusted", "viterbi_v2_segment_tail_trusted",
                    "viterbi_v2_segment_midpoint_trusted", "viterbi_v2_segment_family_grid_trusted",
                    "gemv", "gemv_v4", "hadamard", "yaqa_feedback", "yaqa_feedback_update"]
    invoked = {n[len("qvq_cuda."):] for n in ranges if n.startswith("qvq_cuda.")}
    w("Registered `qvq_cuda` ops never invoked by this workload: " +
      ", ".join(f"`{v}`" for v in all_variants if v not in invoked) + "\n")

    w("### Kernels inside each variant range\n")
    for name in ranges:
        ks = sorted(range_kernels.get(name, []), key=lambda t: -t[1])[:4]
        if not ks:
            continue
        w(f"- `{name}`: " + "; ".join(f"`{k}` {fmt_s(ns)} s ×{inst}" for k, ns, inst in ks))
    w("")

    w("### Top-10 CUDA kernels overall (`cuda_gpu_kern_sum`)\n")
    w("| # | kernel | total (s) | % | instances | avg (ms) | max (ms) |\n|---:|---|---:|---:|---:|---:|---:|")
    for i, r in enumerate(kern[:10], 1):
        w(f"| {i} | `{short_kernel(r['Name'])}` | {fmt_s(float(r['Total Time (ns)']))} | {r['Time (%)']} | "
          f"{r['Instances']} | {fmt_ms(float(r['Avg (ns)']))} | {fmt_ms(float(r['Max (ns)']))} |")
    w("")

    w("### Per-stage GPU projection (`nvtx_gpu_proj_sum`, union of GPU activity under the range)\n")
    w("| stage range | instances | projected GPU time (s) | range wall (s) | GPU busy % of range |\n|---|---:|---:|---:|---:|")
    stage_rows = [r for r in proj if r["Range"].lstrip(":").startswith(("stage.", "qvq_quantize"))]
    agg: dict[str, list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
    for r in stage_rows:
        key = r["Range"].lstrip(":").split(":")[0]
        agg[key][0] += int(r["Range Instances"])
        agg[key][1] += float(r["Total Proj Time (ns)"])
        agg[key][2] += float(r["Total Range Time (ns)"])
    for key, (inst, pj, rt) in sorted(agg.items(), key=lambda kv: -kv[1][2]):
        w(f"| `{key}` | {inst} | {fmt_s(pj)} | {fmt_s(rt)} | {pj / rt * 100 if rt else 0:.1f} |")
    w("")

    w("### Memory transfers (`cuda_gpu_mem_time_sum` / `cuda_gpu_mem_size_sum`)\n")
    size_by = {r["Operation"]: r for r in mem_size}
    w("| operation | count | total time (s) | avg (us) | max (ms) | total size (MB) |\n|---|---:|---:|---:|---:|---:|")
    for r in mem_time:
        sz = size_by.get(r["Operation"], {})
        w(f"| {r['Operation']} | {r['Count']} | {fmt_s(float(r['Total Time (ns)']))} | "
          f"{float(r['Avg (ns)']) / 1e3:.1f} | {fmt_ms(float(r['Max (ns)']))} | {sz.get('Total (MB)', '?')} |")
    w("")

    w("### CUDA API (host side, `cuda_api_sum`)\n")
    w("| API | calls | total (s) | avg (us) | max (ms) |\n|---|---:|---:|---:|---:|")
    for r in api[:10]:
        w(f"| `{r['Name']}` | {r['Num Calls']} | {fmt_s(float(r['Total Time (ns)']))} | "
          f"{float(r['Avg (ns)']) / 1e3:.1f} | {fmt_ms(float(r['Max (ns)']))} |")
    w("")

    w("### OS runtime (`osrt_sum`, top 8)\n")
    w("| call | calls | total (s) | avg (us) |\n|---|---:|---:|---:|")
    for r in osrt[:8]:
        w(f"| `{r['Name']}` | {r['Num Calls']} | {fmt_s(float(r['Total Time (ns)']))} | {float(r['Avg (ns)']) / 1e3:.1f} |")
    w("")

    w("### Host-side stage timings (profile wrapper, `*_host_attribution.json`)\n")
    w("| range | calls | host total (s) | host avg (ms) | host max (ms) |\n|---|---:|---:|---:|---:|")
    for name, v in host_ranges.items():
        w(f"| `{name}` | {v['calls']} | {v['host_seconds_total']:.2f} | {v['host_seconds_avg'] * 1e3:.3f} | {v['host_seconds_max'] * 1e3:.1f} |")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
