# SPDX-License-Identifier: Apache-2.0

"""Measure projected decoder-block costs for QVQ folding arms on MLX.

The projection shapes are the real Llama 3.2 1B dimensions.  This benchmark
times each QuantLinear independently and sums medians; it is not presented as
a full-model tokens/s measurement because the transformed-basis graph rewrite
has not yet been integrated into the MLX Llama container.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.qvq import pack_qvq_binary_bank_ids
from gptqmodel.quantization.qvq_rates import qvq_transition_bits
from gptqmodel.utils.planar_packing import planar_pack_rows
from gptqmodel.utils.qvq_mlx import (
    QVQMLXLinear,
    _qvq_mlx_hadamard,
    _qvq_mlx_hadamard_matrix,
)
from gptqmodel.utils.qvq_p32_mlx import (
    qvq_mlx_p32_window_gemv,
    qvq_mlx_repack_p32_planar_to_window,
)

ROLES = {
    "q": (2048, 2048),
    "k": (2048, 512),
    "v": (2048, 512),
    "o": (2048, 2048),
    "gate": (2048, 8192),
    "up": (2048, 8192),
    "down": (8192, 2048),
}
ARM_AXES = {
    "A0": {role: (True, True) for role in ROLES},
    "A1": {
        "q": (False, True),
        "k": (False, True),
        "v": (False, False),
        "o": (False, False),
        "gate": (False, True),
        "up": (False, True),
        "down": (True, False),
    },
    "A3": {
        "q": (False, False),
        "k": (False, False),
        "v": (False, False),
        "o": (False, False),
        "gate": (False, True),
        "up": (False, True),
        "down": (True, False),
    },
    "A4": {role: (role == "down", False) for role in ROLES},
    "A6": {role: (False, False) for role in ROLES},
}
ARM_AXES["A20"] = dict(ARM_AXES["A3"])
ARM_AXES["A21"] = dict(ARM_AXES["A3"])
ARM_AXES["A22"] = {
    "q": (False, True),
    "k": (False, True),
    "v": (False, False),
    "o": (False, False),
    "gate": (False, False),
    "up": (False, False),
    "down": (True, False),
}
ARM_AXES["A23"] = {role: (False, role in {"q", "k"}) for role in ROLES}
ARM_AXES["A24"] = dict(ARM_AXES["A0"])
ARM_AXES["A24"]["gate"] = (True, False)
ARM_AXES["A24"]["up"] = (True, False)
ARM_AXES["A25"] = dict(ARM_AXES["A0"])
ARM_AXES["A25"]["v"] = (True, False)
ARM_AXES["A25"]["o"] = (False, True)
ARM_AXES["A26"] = dict(ARM_AXES["A25"])
ARM_AXES["A26"]["gate"] = (True, False)
ARM_AXES["A26"]["up"] = (True, False)
ARM_AXES["A27"] = dict(ARM_AXES["A24"])
ARM_AXES["A28"] = dict(ARM_AXES["A26"])
ARM_AXES["A29"] = dict(ARM_AXES["A24"])
ARM_AXES["A30"] = dict(ARM_AXES["A26"])
ARM_RUNTIME_ALIASES = {
    "A20": "A3",
    "A21": "A3",
    "A27": "A24",
    "A28": "A26",
    "A29": "A24",
    "A30": "A26",
}


def _payload(bits, k, n, generator):
    transition_bits = qvq_transition_bits(bits)
    tiles = (k // 16) * (n // 16)
    edges = torch.randint(
        0, 1 << transition_bits, (128, tiles), generator=generator, dtype=torch.int32
    )
    planar = planar_pack_rows(edges, transition_bits).T.contiguous()
    selectors = torch.randint(
        0, 2, (tiles * 8,), generator=generator, dtype=torch.uint8
    )
    return planar, pack_qvq_binary_bank_ids(selectors)


def _measure(function, *, warmup, samples):
    for _ in range(warmup):
        mx.eval(function())
    mx.synchronize()
    elapsed = []
    for _ in range(samples):
        started = time.perf_counter()
        mx.eval(function())
        mx.synchronize()
        elapsed.append((time.perf_counter() - started) * 1e3)
    values = np.asarray(elapsed)
    return float(np.median(values)), float(np.percentile(values, 95))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bits", type=float, default=2.0, choices=(1, 1.5, 2, 2.5, 3, 3.5)
    )
    parser.add_argument("--m", default="1,2,4,8")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=75)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT / "artifacts/qvq_rotation_runtime_m4max.json",
    )
    args = parser.parse_args()
    batches = tuple(int(item) for item in args.m.split(","))
    if any(item < 1 for item in batches) or args.warmup < 0 or args.samples < 1:
        parser.error("M and samples must be positive and warmup nonnegative")

    generator = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    payloads = {}
    for role, (k, n) in ROLES.items():
        key = (k, n)
        if key not in payloads:
            planar, selectors = _payload(args.bits, k, n, generator)
            payloads[key] = (
                mx.array(planar.numpy()),
                mx.array(selectors.numpy()),
                planar,
                selectors,
            )

    rows = []
    summaries = []
    timing_cache = {}
    for m in batches:
        for arm, axes in ARM_AXES.items():
            alias = ARM_RUNTIME_ALIASES.get(arm)
            if alias is not None:
                source_rows = [row for row in rows if row["m"] == m and row["arm"] == alias]
                if len(source_rows) != len(ROLES):
                    raise RuntimeError(f"runtime alias {arm} was evaluated before {alias}")
                cloned_rows = [{**row, "arm": arm, "runtime_alias_of": alias} for row in source_rows]
                rows.extend(cloned_rows)
                totals = {
                    key: sum(row[key] for row in cloned_rows)
                    for key in ("inner_p50_ms", "transform_p50_ms", "module_p50_ms")
                }
                summaries.append({"arm": arm, "m": m, "runtime_alias_of": alias, **totals})
                print(
                    f"M{m} {arm}={alias}: inner={totals['inner_p50_ms']:.4f} ms, "
                    f"standalone-H={totals['transform_p50_ms']:.4f} ms, "
                    f"QuantLinear={totals['module_p50_ms']:.4f} ms",
                    flush=True,
                )
                continue
            totals = {
                "inner_p50_ms": 0.0,
                "transform_p50_ms": 0.0,
                "module_p50_ms": 0.0,
            }
            for role, (k, n) in ROLES.items():
                input_h, output_h = axes[role]
                timing_key = (m, role, input_h, output_h)
                cached = timing_cache.get(timing_key)
                if cached is not None:
                    row = {**cached, "arm": arm, "runtime_reused_from": cached["arm"]}
                    rows.append(row)
                    for key in totals:
                        totals[key] += row[key]
                    continue
                planar, selectors, planar_torch, selectors_torch = payloads[(k, n)]
                bank_alt = mx.array(np.array([2], dtype=np.uint8))
                window = qvq_mlx_repack_p32_planar_to_window(planar, args.bits)
                x = mx.array(rng.standard_normal((m, k)).astype(np.float16))
                inner_x = x.astype(mx.float32)
                module = QVQMLXLinear(
                    bits=args.bits,
                    in_features=k,
                    out_features=n,
                    trellis=planar,
                    SU=mx.ones((k,), dtype=mx.float32),
                    SV=mx.ones((n,), dtype=mx.float32),
                    bank_ids=selectors,
                    v2b2_p32=True,
                    bank_alt_id=bank_alt,
                    input_hadamard=input_h,
                    output_hadamard=output_h,
                )
                mx.eval(window, module._runtime_trellis)
                inner = _measure(
                    lambda _x=inner_x, _window=window, _n=n, _selectors=selectors: (
                        qvq_mlx_p32_window_gemv(
                            _x,
                            _window,
                            args.bits,
                            out_features=_n,
                            bank_ids=_selectors,
                            bank_alt_id=2,
                        )
                    ),
                    warmup=args.warmup,
                    samples=args.samples,
                )
                transform_p50 = transform_p95 = 0.0
                if input_h:
                    matrix = _qvq_mlx_hadamard_matrix(k)
                    timing = _measure(
                        lambda _x=inner_x, _matrix=matrix: _qvq_mlx_hadamard(
                            _x, _matrix
                        ),
                        warmup=args.warmup,
                        samples=args.samples,
                    )
                    transform_p50 += timing[0]
                    transform_p95 += timing[1]
                if output_h:
                    probe = mx.array(rng.standard_normal((m, n)).astype(np.float32))
                    matrix = _qvq_mlx_hadamard_matrix(n)
                    timing = _measure(
                        lambda _probe=probe, _matrix=matrix: _qvq_mlx_hadamard(
                            _probe, _matrix
                        ),
                        warmup=args.warmup,
                        samples=args.samples,
                    )
                    transform_p50 += timing[0]
                    transform_p95 += timing[1]
                combined = _measure(
                    lambda _module=module, _x=x: _module(_x),
                    warmup=args.warmup,
                    samples=args.samples,
                )
                storage_bits = (
                    planar_torch.numel() * planar_torch.element_size() * 8
                    + selectors_torch.numel() * selectors_torch.element_size() * 8
                    + 8
                    + (k + n) * 32
                )
                row = {
                    "arm": arm,
                    "m": m,
                    "role": role,
                    "k": k,
                    "n": n,
                    "input_hadamard": input_h,
                    "output_hadamard": output_h,
                    "inner_p50_ms": inner[0],
                    "inner_p95_ms": inner[1],
                    "transform_p50_ms": transform_p50,
                    "transform_p95_ms": transform_p95,
                    "module_p50_ms": combined[0],
                    "module_p95_ms": combined[1],
                    "effective_bpw": storage_bits / (k * n),
                }
                rows.append(row)
                timing_cache[timing_key] = row
                for key in totals:
                    totals[key] += row[key]
            summary = {"arm": arm, "m": m, **totals}
            summaries.append(summary)
            print(
                f"M{m} {arm}: inner={totals['inner_p50_ms']:.4f} ms, "
                f"standalone-H={totals['transform_p50_ms']:.4f} ms, "
                f"QuantLinear={totals['module_p50_ms']:.4f} ms",
                flush=True,
            )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(
            {
                "schema": "qvq.rotation-folding.runtime.v1",
                "device": mx.device_info(),
                "bits": args.bits,
                "warmup": args.warmup,
                "samples": args.samples,
                "measurement": "sum of independent real-shape projection medians; not full-model tokens/s",
                "rows": rows,
                "summaries": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
