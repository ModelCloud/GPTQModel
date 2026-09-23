"""Matched compressed-P32 mixed-rate M960 projection comparison: large-M2 vs WGMMA.

This is a local kernel benchmark, not a model-quality or B128 throughput gate.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

from scripts.benchmark_qvq_p32_t6_prefill_h100 import _idle_preflight, _load_arm


class RawConfig(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "abi_version", "struct_bytes", "m", "k", "n", "transition_bits",
        "split_count", "algorithm", "block_m", "block_n",
    )]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--baseline-raw", action="store_true",
                        help="compare two explicit algorithm-5 raw-ABI libraries instead of large-M2")
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--projection", choices=("q", "k", "v", "gate", "down"),
                        default="gate")
    parser.add_argument("--snapshot-dir", type=Path,
                        help="use layer-0 P32 metadata from this quantized snapshot")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--m960-block-m", type=int, choices=(64, 80), default=64,
                        help="explicit raw-ABI row geometry: BM64=four, BM80=five M16 rows/CTA")
    parser.add_argument("--m960-block-n", type=int, choices=(64, 128), default=64,
                        help="explicit raw-ABI N geometry: BN128 uses two N64 consumers/CTA")
    parser.add_argument("--baseline-block-m", type=int, choices=(64, 80),
                        help="raw-baseline BM; defaults to candidate BM")
    parser.add_argument("--baseline-block-n", type=int, choices=(64, 128),
                        help="raw-baseline BN; defaults to candidate BN")
    parser.add_argument("--profile-arm", choices=("baseline", "candidate"),
                        help="warm only this arm, then issue one traceable launch")
    parser.add_argument("--profiler-attached", action="store_true",
                        help="skip the in-process idle check; check GPU idleness before attaching a profiler")
    args = parser.parse_args()
    if args.profile_arm is None and args.rounds < 10:
        parser.error("at least 10 paired rounds are required")
    if args.profiler_attached and args.profile_arm is None:
        parser.error("--profiler-attached requires --profile-arm")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    if not args.profiler_attached:
        _idle_preflight(args.gpu_uuid)

    import torch

    if args.baseline_raw:
        baseline_library = ctypes.CDLL(str(args.baseline.resolve()), mode=ctypes.RTLD_LOCAL)
        baseline = baseline_library.qvq_p32_wgmma_raw_launch
        baseline.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_uint64,
            ctypes.POINTER(RawConfig), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        baseline.restype = ctypes.c_int
        baseline_library.qvq_p32_wgmma_raw_abi_version.restype = ctypes.c_uint32
        assert baseline_library.qvq_p32_wgmma_raw_abi_version() == 3
    else:
        baseline, baseline_error = _load_arm(args.baseline)
    library = ctypes.CDLL(str(args.candidate.resolve()), mode=ctypes.RTLD_LOCAL)
    candidate = library.qvq_p32_wgmma_raw_launch
    candidate.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_uint64,
        ctypes.POINTER(RawConfig), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    candidate.restype = ctypes.c_int
    library.qvq_p32_wgmma_raw_abi_version.restype = ctypes.c_uint32
    assert library.qvq_p32_wgmma_raw_abi_version() == 3

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (9, 0):
        raise RuntimeError(f"expected sm_90, got sm_{props.major}{props.minor}")
    shapes = {
        "q": (2048, 2048, 4), "k": (2048, 512, 5), "v": (2048, 512, 7),
        "gate": (2048, 8192, 6), "down": (8192, 2048, 6),
    }
    m = 960
    k, n, bits = shapes[args.projection]
    seed = 20260923
    generator = torch.Generator().manual_seed(seed)
    x = (torch.randn((m, k), generator=generator) * 0.02).half().to(device)
    tiles = (k // 16) * (n // 16)
    if args.snapshot_dir is None:
        trellis = torch.randint(-(2**31), 2**31 - 1, (tiles * 4 * bits,),
                                generator=generator, dtype=torch.int32).to(device)
        levels = torch.linspace(-1, 1, 256, dtype=torch.float16, device=device)
        banks = torch.randint(0, 256, (tiles,), generator=generator, dtype=torch.uint8).to(device)
        alt = torch.tensor([3], dtype=torch.uint8, device=device)
    else:
        from safetensors import safe_open

        from gptqmodel.quantization.qvq_codecs import pgc16_levels_for_version

        snapshot = args.snapshot_dir.resolve()
        family = "self_attn" if args.projection in ("q", "k", "v") else "mlp"
        prefix = f"model.layers.0.{family}.{args.projection}_proj."
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        shard = snapshot / index["weight_map"][prefix + "trellis"]
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            trellis = tensors.get_tensor(prefix + "trellis").contiguous().to(device)
            banks = tensors.get_tensor(prefix + "bank_ids").contiguous().to(device)
            alt = tensors.get_tensor(prefix + "bank_alt_id").contiguous().to(device)
        levels = pgc16_levels_for_version("pgc16-v1").contiguous().to(device)
        if trellis.numel() != tiles * 4 * bits or banks.numel() != tiles:
            raise RuntimeError("snapshot metadata does not match requested M960 shape/rate")
    baseline_output = torch.empty((m, n), dtype=torch.float32, device=device)
    candidate_output = torch.empty_like(baseline_output)
    partials = torch.empty_like(baseline_output)
    config = RawConfig(
        3, ctypes.sizeof(RawConfig), m, k, n, bits, 1, 5,
        args.m960_block_m, args.m960_block_n,
    )
    baseline_config = RawConfig(
        3, ctypes.sizeof(RawConfig), m, k, n, bits, 1, 5,
        args.baseline_block_m or args.m960_block_m,
        args.baseline_block_n or args.m960_block_n,
    )
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))

    def invoke_baseline() -> None:
        if args.baseline_raw:
            error = ctypes.create_string_buffer(4096)
            status = baseline(
                x.data_ptr(), trellis.data_ptr(), banks.data_ptr(), levels.data_ptr(),
                alt.data_ptr(), baseline_output.data_ptr(), None, 0,
                ctypes.byref(baseline_config), stream.cuda_stream, error, len(error),
            )
            if status != 0:
                raise RuntimeError(f"raw baseline returned {status}: {error.value.decode()}")
        else:
            status = baseline(
                x.data_ptr(), trellis.data_ptr(), levels.data_ptr(), banks.data_ptr(),
                alt.data_ptr(), baseline_output.data_ptr(), partials.data_ptr(),
                m, k, n, bits, 1, 2, 128, 3, 0, 1, 4, stream.cuda_stream,
            )
            if status != 0:
                raise RuntimeError(f"baseline returned {status}: {baseline_error().decode()}")

    def invoke_candidate() -> None:
        error = ctypes.create_string_buffer(4096)
        status = candidate(
            x.data_ptr(), trellis.data_ptr(), banks.data_ptr(), levels.data_ptr(),
            alt.data_ptr(), candidate_output.data_ptr(), None, 0,
            ctypes.byref(config), stream.cuda_stream, error, len(error),
        )
        if status != 0:
            raise RuntimeError(f"candidate returned {status}: {error.value.decode()}")

    arms = (invoke_baseline, invoke_candidate)
    if args.profile_arm is not None:
        chosen = 0 if args.profile_arm == "baseline" else 1
        for _ in range(20):
            arms[chosen]()
        stream.synchronize()
        arms[chosen]()
        stream.synchronize()
        print(json.dumps({
            "scope": "single_warmed_compressed_p32_profile_arm",
            "arm": args.profile_arm,
            "shape": {"m": m, "k": k, "n": n, "transition_bits": bits},
            "projection": args.projection,
            "m960_block_m": args.m960_block_m,
            "m960_block_n": args.m960_block_n,
            "baseline_block_m": baseline_config.block_m if args.baseline_raw else None,
            "baseline_block_n": baseline_config.block_n if args.baseline_raw else None,
            "snapshot_dir": str(args.snapshot_dir.resolve()) if args.snapshot_dir else None,
            "library_sha256": hashlib.sha256(
                (args.baseline if chosen == 0 else args.candidate).read_bytes()
            ).hexdigest(),
        }))
        return
    for _ in range(20):
        for arm in arms:
            arm()
    stream.synchronize()
    delta = (baseline_output - candidate_output).abs()
    mae = delta.double().mean().item()
    max_error = delta.max().item()
    exact = torch.equal(baseline_output, candidate_output)
    if not torch.isfinite(candidate_output).all():
        raise RuntimeError("candidate has non-finite outputs")

    timings = ([], [])
    for iteration in range(args.rounds):
        for arm in ((0, 1) if iteration % 2 == 0 else (1, 0)):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            arms[arm]()
            stop.record(stream)
            timings[arm].append((start, stop))
    stream.synchronize()
    measured = [[start.elapsed_time(stop) * 1000.0 for start, stop in arm]
                for arm in timings]
    medians = [sorted(arm)[len(arm) // 2] for arm in measured]
    print(json.dumps({
        "scope": "local_compressed_p32_gate_inner_projection",
        "gpu": props.name,
        "gpu_uuid": args.gpu_uuid,
        "shape": {"m": m, "k": k, "n": n, "transition_bits": bits},
        "projection": args.projection,
        "baseline_raw": args.baseline_raw,
        "m960_block_m": args.m960_block_m,
        "m960_block_n": args.m960_block_n,
        "baseline_block_m": baseline_config.block_m if args.baseline_raw else None,
        "baseline_block_n": baseline_config.block_n if args.baseline_raw else None,
        "seed": seed,
        "snapshot_dir": str(args.snapshot_dir.resolve()) if args.snapshot_dir else None,
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "bitwise_equal": exact,
        "mae": mae,
        "max_error": max_error,
        "baseline_median_us": medians[0],
        "candidate_median_us": medians[1],
        "speedup": medians[0] / medians[1],
        "baseline_samples_us": measured[0],
        "candidate_samples_us": measured[1],
    }))


if __name__ == "__main__":
    main()
