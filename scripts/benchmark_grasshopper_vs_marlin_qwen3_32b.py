#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from gptqmodel import extension  # noqa: E402
from gptqmodel.adapter.adapter import Lora  # noqa: E402
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear  # noqa: E402
from gptqmodel.utils import grasshopper  # noqa: E402


DEFAULT_CONFIG = Path("/monster/data/model/Qwen3-32B/config.json")


@dataclass(frozen=True)
class ProjectionCase:
    name: str
    in_features: int
    out_features: int


@dataclass(frozen=True)
class BenchResult:
    case: str
    rows: int
    in_features: int
    out_features: int
    backend: str
    mode: str
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    tflops: float
    checksum: float


def _dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower().replace("torch.", "")
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _read_qwen3_projection_cases(config_path: Path) -> list[ProjectionCase]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim") or hidden_size // num_attention_heads)
    q_out = num_attention_heads * head_dim
    kv_out = num_key_value_heads * head_dim
    return [
        ProjectionCase("q_proj", hidden_size, q_out),
        ProjectionCase("k_proj", hidden_size, kv_out),
        ProjectionCase("v_proj", hidden_size, kv_out),
        ProjectionCase("o_proj", q_out, hidden_size),
        ProjectionCase("gate_proj", hidden_size, intermediate_size),
        ProjectionCase("up_proj", hidden_size, intermediate_size),
        ProjectionCase("down_proj", intermediate_size, hidden_size),
    ]


def _parse_csv_ints(raw: str, *, name: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"--{name} produced no values.")
    return values


def _zero_int_tensor(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.empty(shape, device=device, dtype=torch.int32).random_(
        -(2**31), 2**31 - 1
    )


def _make_grasshopper_tensors(
    *,
    case: ProjectionCase,
    rows: int,
    bits: int,
    group_size: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    if case.in_features % 32 != 0:
        raise ValueError(f"{case.name}: in_features must be divisible by 32.")
    if case.in_features % group_size != 0:
        raise ValueError(f"{case.name}: in_features must be divisible by group_size.")
    if case.out_features % 32 != 0:
        raise ValueError(f"{case.name}: out_features must be divisible by 32.")

    groups = case.in_features // group_size
    qweight_rows = (case.in_features // 32) * bits
    qzero_cols = (case.out_features // 32) * bits
    x = torch.randn(
        (rows, case.in_features),
        device=device,
        dtype=dtype,
        generator=generator,
    ).contiguous()
    lora_a = (
        torch.randn(
            (case.in_features, rank),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * 0.01
    ).contiguous()
    lora_b = (
        torch.randn(
            (rank, case.out_features),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * 0.01
    ).contiguous()
    return {
        "x": x,
        "qweight": _zero_int_tensor((qweight_rows, case.out_features), device),
        "qzeros": _zero_int_tensor((groups, qzero_cols), device),
        "scales": (
            torch.rand(
                (groups, case.out_features),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * 0.02
            + 0.001
        ).contiguous(),
        "lora_a": lora_a,
        "lora_b": lora_b,
    }


def _build_marlin_module(
    *,
    case: ProjectionCase,
    bits: int,
    group_size: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> MarlinLinear:
    if case.out_features % 64 != 0:
        raise ValueError(f"{case.name}: Marlin out_features must be divisible by 64.")
    module = MarlinLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=case.in_features,
        out_features=case.out_features,
        bias=False,
        dtype=dtype,
    ).to(device)
    with torch.no_grad():
        module.qweight.copy_(
            torch.randint(
                -(2**31),
                2**31 - 1,
                module.qweight.shape,
                dtype=torch.int32,
                device=device,
                generator=generator,
            )
        )
        module.scales.copy_(
            torch.rand(
                module.scales.shape,
                dtype=dtype,
                device=device,
                generator=generator,
            )
            * 0.02
            + 0.001
        )
        module.qzeros.zero_()
        module.g_idx.zero_()
    module.eval()
    module.post_init()
    return module


def _time_cuda(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> tuple[list[float], float]:
    output = None
    with torch.inference_mode():
        for _ in range(warmup):
            output = fn()
        torch.cuda.synchronize(device)

        timings: list[float] = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = fn()
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))

    if output is None:
        raise RuntimeError("No benchmark output was produced.")
    checksum = float(output.float().mean().detach().cpu())
    return timings, checksum


def _result(
    *,
    case: ProjectionCase,
    rows: int,
    backend: str,
    mode: str,
    timings: list[float],
    checksum: float,
) -> BenchResult:
    mean_ms = statistics.fmean(timings)
    tflops = (2.0 * rows * case.in_features * case.out_features) / (mean_ms * 1.0e9)
    return BenchResult(
        case=case.name,
        rows=rows,
        in_features=case.in_features,
        out_features=case.out_features,
        backend=backend,
        mode=mode,
        mean_ms=mean_ms,
        median_ms=statistics.median(timings),
        min_ms=min(timings),
        max_ms=max(timings),
        tflops=tflops,
        checksum=checksum,
    )


def _grasshopper_base_call(
    tensors: dict[str, torch.Tensor],
    *,
    rows: int,
    group_size: int,
    bits: int,
) -> torch.Tensor:
    if rows == 1:
        return grasshopper.gemv(
            tensors["x"].reshape(-1),
            tensors["qweight"],
            tensors["scales"],
            tensors["qzeros"],
            group_size,
            accumulation_dtype=torch.float32,
            bits=bits,
        )
    return grasshopper.gemm(
        tensors["x"],
        tensors["qweight"],
        tensors["scales"],
        tensors["qzeros"],
        group_size,
        accumulation_dtype=torch.float32,
        bits=bits,
    )


def _grasshopper_lora_call(
    tensors: dict[str, torch.Tensor],
    *,
    down: torch.Tensor,
    rows: int,
    group_size: int,
    bits: int,
) -> torch.Tensor:
    if rows == 1:
        return grasshopper.gemv_lora(
            tensors["x"].reshape(-1),
            tensors["qweight"],
            tensors["scales"],
            tensors["qzeros"],
            down.reshape(-1),
            tensors["lora_b"],
            group_size,
            accumulation_dtype=torch.float32,
            bits=bits,
        )
    return grasshopper.gemm_lora(
        tensors["x"],
        tensors["qweight"],
        tensors["scales"],
        tensors["qzeros"],
        down,
        tensors["lora_b"],
        group_size,
        accumulation_dtype=torch.float32,
        bits=bits,
    )


def _bench_one(
    *,
    case: ProjectionCase,
    rows: int,
    bits: int,
    group_size: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    warmup: int,
    iters: int,
) -> list[BenchResult]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    tensors = _make_grasshopper_tensors(
        case=case,
        rows=rows,
        bits=bits,
        group_size=group_size,
        rank=rank,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    marlin_module = _build_marlin_module(
        case=case,
        bits=bits,
        group_size=group_size,
        dtype=dtype,
        device=device,
        generator=generator,
    )

    x = tensors["x"]
    lora_a = tensors["lora_a"]
    lora_b = tensors["lora_b"]
    precomputed_down = (x @ lora_a).contiguous()
    adapter = Lora(rank=rank, lora_A=lora_a, lora_B=lora_b)

    results: list[BenchResult] = []

    timings, checksum = _time_cuda(
        lambda: _grasshopper_base_call(tensors, rows=rows, group_size=group_size, bits=bits),
        warmup=warmup,
        iters=iters,
        device=device,
    )
    results.append(
        _result(
            case=case,
            rows=rows,
            backend="grasshopper",
            mode="base",
            timings=timings,
            checksum=checksum,
        )
    )

    timings, checksum = _time_cuda(
        lambda: _grasshopper_lora_call(
            tensors,
            down=precomputed_down,
            rows=rows,
            group_size=group_size,
            bits=bits,
        ),
        warmup=warmup,
        iters=iters,
        device=device,
    )
    results.append(
        _result(
            case=case,
            rows=rows,
            backend="grasshopper",
            mode="lora_precomputed_down",
            timings=timings,
            checksum=checksum,
        )
    )

    timings, checksum = _time_cuda(
        lambda: _grasshopper_lora_call(
            tensors,
            down=(x @ lora_a).contiguous(),
            rows=rows,
            group_size=group_size,
            bits=bits,
        ),
        warmup=warmup,
        iters=iters,
        device=device,
    )
    results.append(
        _result(
            case=case,
            rows=rows,
            backend="grasshopper",
            mode="lora_total",
            timings=timings,
            checksum=checksum,
        )
    )

    marlin_module.adapter = None
    timings, checksum = _time_cuda(
        lambda: marlin_module(x),
        warmup=warmup,
        iters=iters,
        device=device,
    )
    results.append(
        _result(
            case=case,
            rows=rows,
            backend="marlin",
            mode="base",
            timings=timings,
            checksum=checksum,
        )
    )

    timings, checksum = _time_cuda(
        lambda: torch.addmm(
            marlin_module(x).reshape(rows, case.out_features),
            precomputed_down.reshape(rows, rank),
            lora_b,
            beta=1.0,
            alpha=1.0,
        ),
        warmup=warmup,
        iters=iters,
        device=device,
    )
    results.append(
        _result(
            case=case,
            rows=rows,
            backend="marlin",
            mode="lora_precomputed_down",
            timings=timings,
            checksum=checksum,
        )
    )

    marlin_module.adapter = adapter
    timings, checksum = _time_cuda(
        lambda: marlin_module(x),
        warmup=warmup,
        iters=iters,
        device=device,
    )
    results.append(
        _result(
            case=case,
            rows=rows,
            backend="marlin",
            mode="lora_total",
            timings=timings,
            checksum=checksum,
        )
    )

    return results


def _format_table(results: list[BenchResult]) -> str:
    lookup = {(r.case, r.rows, r.mode, r.backend): r for r in results}
    rows: list[str] = [
        "| case | rows | k | n | mode | GrassHopper ms | Marlin ms | GH / Marlin |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for key in sorted({(r.case, r.rows, r.mode) for r in results}):
        case_name, batch_rows, mode = key
        gh = lookup[(case_name, batch_rows, mode, "grasshopper")]
        marlin = lookup[(case_name, batch_rows, mode, "marlin")]
        ratio = gh.median_ms / marlin.median_ms
        rows.append(
            f"| {case_name} | {batch_rows} | {gh.in_features} | {gh.out_features} | "
            f"{mode} | {gh.median_ms:.4f} | {marlin.median_ms:.4f} | {ratio:.3f}x |"
        )
    return "\n".join(rows)


def _format_summary(results: list[BenchResult]) -> str:
    rows: list[str] = [
        "| rows | mode | GrassHopper total ms | Marlin total ms | GH / Marlin |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for key in sorted({(r.rows, r.mode) for r in results}):
        batch_rows, mode = key
        gh_total = sum(
            r.median_ms
            for r in results
            if r.rows == batch_rows and r.mode == mode and r.backend == "grasshopper"
        )
        marlin_total = sum(
            r.median_ms
            for r in results
            if r.rows == batch_rows and r.mode == mode and r.backend == "marlin"
        )
        rows.append(
            f"| {batch_rows} | {mode} | {gh_total:.4f} | {marlin_total:.4f} | "
            f"{gh_total / marlin_total:.3f}x |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark GrassHopper GPTQ against Marlin on Qwen3-32B projection shapes."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16"))
    parser.add_argument("--bits", type=int, default=4, choices=(4, 8))
    parser.add_argument("--group-size", type=int, default=128, choices=(32, 64, 128))
    parser.add_argument("--rows", default="1,8", help="Comma-separated row counts.")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--case", default=None, help="Optional comma-separated projection names.")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be a CUDA device.")
    torch.cuda.set_device(device)
    dtype = _dtype(args.dtype)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected CUDA device does not report BF16 support.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cases = _read_qwen3_projection_cases(args.config)
    if args.case:
        selected = {part.strip() for part in args.case.split(",") if part.strip()}
        cases = [case for case in cases if case.name in selected]
    if not cases:
        raise ValueError("No projection cases selected.")

    rows_values = _parse_csv_ints(args.rows, name="rows")
    props = torch.cuda.get_device_properties(device)
    print(
        f"device={args.device}:{props.name} sm_{props.major}{props.minor} "
        f"config={args.config} dtype={args.dtype} bits={args.bits} "
        f"group={args.group_size} rank={args.rank} rows={rows_values}"
    )
    extension.load("grasshopper")
    extension.load("marlin_fp16" if dtype == torch.float16 else "marlin_bf16")

    all_results: list[BenchResult] = []
    for batch_rows in rows_values:
        for index, case in enumerate(cases):
            print(
                f"benchmark case={case.name} rows={batch_rows} "
                f"k={case.in_features} n={case.out_features}",
                flush=True,
            )
            all_results.extend(
                _bench_one(
                    case=case,
                    rows=batch_rows,
                    bits=args.bits,
                    group_size=args.group_size,
                    rank=args.rank,
                    dtype=dtype,
                    device=device,
                    seed=args.seed + 1009 * batch_rows + index,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            )

    print("\nPer-projection median latency:")
    print(_format_table(all_results))
    print("\nProjection-sum median latency:")
    print(_format_summary(all_results))

    payload: dict[str, Any] = {
        "config": str(args.config),
        "device": args.device,
        "device_name": props.name,
        "sm": f"{props.major}{props.minor}",
        "dtype": args.dtype,
        "bits": args.bits,
        "group_size": args.group_size,
        "rank": args.rank,
        "warmup": args.warmup,
        "iters": args.iters,
        "cases": [asdict(case) for case in cases],
        "results": [asdict(result) for result in all_results],
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\njson_out={args.json_out}")


if __name__ == "__main__":
    main()
