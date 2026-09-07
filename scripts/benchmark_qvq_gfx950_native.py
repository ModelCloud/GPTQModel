"""Exploratory HIP decode + rocBLAS versus a pinned QVQ AOT artifact."""

import argparse
import ctypes as c
import hashlib
import json
import os
import statistics
from pathlib import Path

from benchmark_qvq_p32_amd import _idle_preflight, _timing_recheck


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--decode-library", required=True)
    parser.add_argument("--blas-library", required=True)
    parser.add_argument("--baseline-library", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--solution-index", type=int, default=None)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--idle-memory-tolerance-mib", type=int, default=320)
    args = parser.parse_args()
    if args.output.exists() or args.iterations < 10:
        parser.error("output must be new and iterations >= 10")
    args.idle_samples, args.idle_interval, args.allow_busy = 3, 1.0, False
    hardware, _ = _idle_preflight(args)
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.physical_gpu)
    import torch

    from gptqmodel.quantization.qvq import (
        reconstruct_qvq_inner_weight,
        repack_p32_planar_to_window,
    )
    from gptqmodel.quantization.qvq_codecs import (
        PGC16_CODEBOOK_VERSION,
        pgc16_levels_for_version,
    )

    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(":")[0] == "gfx950"
    # A single owned non-default stream carries preparation, graph capture,
    # replay and timing. Native plans bind to this stream for their lifetime.
    benchmark_stream = torch.cuda.Stream()
    benchmark_stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(benchmark_stream)
    record = json.loads((args.artifact / "manifest.json").read_text())
    binary = (args.artifact / "kernel.hsaco").read_bytes()
    assert hashlib.sha256(binary).hexdigest() == record["sha256"]
    m, k, n, tb, bank = (
        record[key] for key in ("m", "k", "n", "transition_bits", "bank_alt_id")
    )
    assert 4 <= tb <= 7, "this prototype does not yet support ordinary W4"
    fields = (
        "abi_version",
        "operation_version",
        "m",
        "k",
        "n",
        "transition_bits",
        "bank_alt_id",
        "grid_x",
        "threads",
        "shared_bytes",
    )

    class Spec(c.Structure):
        _fields_ = [(name, c.c_uint32) for name in fields]

    baseline = c.CDLL(args.baseline_library)
    baseline.qvq_gfx950_prepare.argtypes = [
        c.POINTER(Spec),
        c.c_void_p,
        c.c_size_t,
        c.c_char_p,
        c.c_int,
        c.c_void_p,
        c.POINTER(c.c_void_p),
    ]
    baseline.qvq_gfx950_execute.argtypes = [c.c_void_p] * 7
    baseline.qvq_gfx950_destroy.argtypes = [c.c_void_p]
    decoder = c.CDLL(args.decode_library).qvq_gfx950_decode_window
    decoder.argtypes = [c.c_void_p] * 4 + [c.c_uint] * 4 + [c.c_void_p]
    blas = c.CDLL(args.blas_library)
    blas.qvq_gfx950_rocblas_prepare.argtypes = (
        [c.c_int] * 3 + [c.c_void_p] * 2 + [c.c_size_t, c.POINTER(c.c_void_p)]
    )
    blas.qvq_gfx950_rocblas_execute.argtypes = [c.c_void_p] * 5
    blas.qvq_gfx950_rocblas_destroy.argtypes = [c.c_void_p]

    def check(status):
        assert status == 0, f"native status {status}"

    gen = torch.Generator(device="cuda").manual_seed(950)
    planar = torch.randint(
        -(2**31),
        2**31,
        (k * n // 256, tb * 4),
        generator=gen,
        device="cuda",
        dtype=torch.int32,
    )
    window = repack_p32_planar_to_window(planar, bits=tb / 2)
    banks = torch.randint(
        0, 256, (k * n // 256,), generator=gen, device="cuda", dtype=torch.uint8
    )
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).to("cuda")
    reference_w = reconstruct_qvq_inner_weight(
        planar,
        bits=tb / 2,
        in_features=k,
        out_features=n,
        bank_ids=banks,
        v2b2_p32=True,
        bank_alt_id=torch.tensor([bank], device="cuda", dtype=torch.uint8),
    )
    x = torch.randn((m, k), generator=gen, device="cuda", dtype=torch.float16) * 0.1
    dense = torch.empty((n, k), device="cuda", dtype=torch.float16)
    y = torch.empty((m, n), device="cuda", dtype=torch.float32)
    old_y = torch.empty_like(y)
    workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    stream = torch.cuda.current_stream().cuda_stream
    old_plan, plan = c.c_void_p(), c.c_void_p()
    image = c.create_string_buffer(binary)
    spec = Spec(*(record[key] for key in fields))
    check(
        baseline.qvq_gfx950_prepare(
            c.byref(spec),
            image,
            len(binary),
            record["symbol"].encode(),
            0,
            stream,
            c.byref(old_plan),
        )
    )
    check(
        blas.qvq_gfx950_rocblas_prepare(
            m, k, n, stream, workspace.data_ptr(), workspace.numel(), c.byref(plan)
        )
    )
    if args.solution_index is not None:

        class Config(c.Structure):
            _fields_ = [
                (name, c.c_int32)
                for name in (
                    "struct_size",
                    "version",
                    "m",
                    "k",
                    "n",
                    "e",
                    "solution_index",
                    "reserved",
                )
            ]

        blas.qvq_gfx950_rocblas_prepare_config.argtypes = (
            [c.POINTER(Config)] + [c.c_void_p] * 5 + [c.c_size_t, c.POINTER(c.c_void_p)]
        )
        blas.qvq_gfx950_rocblas_get_config.argtypes = [c.c_void_p, c.POINTER(Config)]
        config = Config(c.sizeof(Config), 1, m, k, n, 1, args.solution_index, 0)
        selected = c.c_void_p()
        check(
            blas.qvq_gfx950_rocblas_prepare_config(
                c.byref(config),
                x.data_ptr(),
                dense.data_ptr(),
                y.data_ptr(),
                stream,
                workspace.data_ptr(),
                workspace.numel(),
                c.byref(selected),
            )
        )
        check(blas.qvq_gfx950_rocblas_destroy(plan))
        plan = selected
        resolved = Config(c.sizeof(Config), 1)
        check(blas.qvq_gfx950_rocblas_get_config(plan, c.byref(resolved)))
        assert bytes(resolved) == bytes(config), (
            "explicit selection changed during preparation"
        )

    def decode():
        check(
            decoder(
                window.data_ptr(),
                levels.data_ptr(),
                banks.data_ptr(),
                dense.data_ptr(),
                k,
                n,
                tb,
                bank,
                stream,
            )
        )

    def gemm():
        check(
            blas.qvq_gfx950_rocblas_execute(
                plan, x.data_ptr(), dense.data_ptr(), y.data_ptr(), stream
            )
        )

    def combined():
        decode()
        gemm()

    def original():
        check(
            baseline.qvq_gfx950_execute(
                old_plan,
                x.data_ptr(),
                window.data_ptr(),
                levels.data_ptr(),
                banks.data_ptr(),
                old_y.data_ptr(),
                stream,
            )
        )

    combined_graph = None
    original_graph = None
    try:
        combined()
        original()
        torch.cuda.synchronize()
        assert torch.equal(dense.T, reference_w.half()), (
            "decoded values differ from canonical planar oracle"
        )
        # Full-K independent FP64 reference; synthetic correctness, not model quality.
        expected = x.double() @ reference_w.half().double()
        errors = {}
        for name, output in (("native", y), ("baseline", old_y)):
            error = (output.double() - expected).abs()
            errors[name] = {
                "mean": error.mean().item(),
                "max": error.max().item(),
                "finite": bool(torch.isfinite(error).all()),
            }
            errors[name]["passed"] = (
                errors[name]["finite"]
                and errors[name]["mean"] <= 0.003
                and errors[name]["max"] <= 0.006
            )
        # Capture the two-stage chain, not just either leaf operation. All
        # buffers and handles remain owned until both graphs are destroyed.
        combined_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(combined_graph, stream=torch.cuda.current_stream()):
            combined()
        original_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(original_graph, stream=torch.cuda.current_stream()):
            original()
        original_x = x.clone()
        graph_checks = []
        for scale in (0.5, -1.0, 1.0):
            x.copy_(original_x * scale)
            combined_graph.replay()
            original_graph.replay()
            for name, output in (("native", y), ("baseline", old_y)):
                reference = x.double() @ reference_w.half().double()
                error = (output.double() - reference).abs()
                check_result = {
                    "backend": name,
                    "input_scale": scale,
                    "mean": error.mean().item(),
                    "max": error.max().item(),
                    "finite": bool(torch.isfinite(error).all()),
                }
                check_result["passed"] = (
                    check_result["finite"]
                    and check_result["mean"] <= 0.003
                    and check_result["max"] <= 0.006
                )
                graph_checks.append(check_result)
        torch.cuda.synchronize()
        timings = {}
        for name, operation in (
            ("baseline", original),
            ("decode", decode),
            ("gemm_cached", gemm),
            ("decode_plus_gemm", combined),
            ("baseline_graph", original_graph.replay),
            ("decode_plus_gemm_graph", combined_graph.replay),
        ):
            for _ in range(10):
                operation()
            torch.cuda.synchronize()
            _timing_recheck(args, {os.getpid()})
            samples = []
            for _ in range(args.iterations):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                operation()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000)
            timings[name] = {
                "p50_us": statistics.median(samples),
                "mean_us": statistics.mean(samples),
                "p95_us": sorted(samples)[int(0.95 * (len(samples) - 1))],
                "samples_us": samples,
            }
            print(
                f"M={m} K={k} N={n} tb={tb} {name}: {timings[name]['p50_us']:.3f} us",
                flush=True,
            )
        result = {
            "hardware": hardware,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "baseline_artifact": record,
            "synthetic": True,
            "errors": errors,
            "graph_checks": graph_checks,
            "solution_index": args.solution_index,
            "timings": timings,
            "dense_scratch_bytes": dense.numel() * 2,
            "blas_workspace_bytes": workspace.numel(),
            "library_sha256": {
                key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for key, path in (
                    ("decoder", args.decode_library),
                    ("blas", args.blas_library),
                    ("baseline", args.baseline_library),
                )
            },
        }
        result["combined_speedup"] = (
            timings["baseline"]["p50_us"] / timings["decode_plus_gemm"]["p50_us"]
        )
        with args.output.open("x") as file:
            json.dump(result, file, indent=2)
        print(
            json.dumps({"errors": errors, "speedup": result["combined_speedup"]}),
            flush=True,
        )
    finally:
        torch.cuda.synchronize()
        del combined_graph, original_graph
        check(blas.qvq_gfx950_rocblas_destroy(plan))
        check(baseline.qvq_gfx950_destroy(old_plan))


if __name__ == "__main__":
    main()
