"""Screen a transient W3 decode+GEMM against the production M960 P32 core.

This is a GPU-local probe, not a qualified full-model throughput or quality gate.
The dense FP16 matrix is scratch rebuilt for each replay, never a weight cache.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from pathlib import Path


class RawConfig(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "abi_version", "struct_bytes", "m", "k", "n", "transition_bits",
        "split_count", "algorithm", "block_m", "block_n",
    )]


class RawDecodeConfig(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "abi_version", "struct_bytes", "k", "n", "transition_bits",
        "output_layout",
    )]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--projection", choices=("gate", "down"), required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--input-scale", type=float, default=0.02)
    parser.add_argument("--fp64-oracle-rows", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=100)
    args = parser.parse_args()

    import torch
    from safetensors import safe_open
    from gptqmodel.quantization.qvq_codecs import pgc16_levels_for_version

    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("this timing probe requires SM90")
    m = 960
    k, n = (2048, 8192) if args.projection == "gate" else (8192, 2048)
    prefix = f"model.layers.{args.layer}.mlp.{args.projection}_proj."
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())
    shard = args.snapshot / index["weight_map"][prefix + "trellis"]
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        trellis = tensors.get_tensor(prefix + "trellis").contiguous().cuda()
        banks = tensors.get_tensor(prefix + "bank_ids").contiguous().cuda()
        alt = tensors.get_tensor(prefix + "bank_alt_id").contiguous().cuda()
    levels = pgc16_levels_for_version("pgc16-v1").contiguous().cuda()
    generator = torch.Generator().manual_seed(20260924)
    x = (torch.randn((m, k), generator=generator) * args.input_scale).half().cuda()
    decoded = torch.empty((k, n), dtype=torch.float16, device="cuda")
    baseline_out = torch.empty((m, n), dtype=torch.float32, device="cuda")
    config = RawConfig(3, ctypes.sizeof(RawConfig), m, k, n, 6, 1, 5,
                       160 if args.projection == "gate" else 80,
                       64)

    library = ctypes.CDLL(str(args.library.resolve()), mode=ctypes.RTLD_LOCAL)
    launch = library.qvq_p32_wgmma_raw_launch
    launch.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_uint64,
        ctypes.POINTER(RawConfig), ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint64]
    launch.restype = ctypes.c_int
    decode = library.qvq_p32_w3_decode_raw_launch
    decode.argtypes = [ctypes.c_void_p] * 5 + [
        ctypes.POINTER(RawDecodeConfig), ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint64]
    decode.restype = ctypes.c_int
    library.qvq_p32_w3_decode_raw_abi_version.restype = ctypes.c_uint32
    if library.qvq_p32_w3_decode_raw_abi_version() != 1:
        raise RuntimeError("wrong W3 decoder raw ABI version")
    decode_config = RawDecodeConfig(1, ctypes.sizeof(RawDecodeConfig), k, n, 6, 0)
    stream = torch.cuda.current_stream()

    def baseline() -> None:
        error = ctypes.create_string_buffer(4096)
        result = launch(x.data_ptr(), trellis.data_ptr(), banks.data_ptr(),
                        levels.data_ptr(), alt.data_ptr(), baseline_out.data_ptr(),
                        None, 0, ctypes.byref(config),
                        torch.cuda.current_stream().cuda_stream,
                        error, len(error))
        if result:
            raise RuntimeError(error.value.decode())

    def candidate() -> torch.Tensor:
        error = ctypes.create_string_buffer(4096)
        result = decode(trellis.data_ptr(), banks.data_ptr(), levels.data_ptr(),
                        alt.data_ptr(), decoded.data_ptr(),
                        ctypes.byref(decode_config),
                        torch.cuda.current_stream().cuda_stream,
                        error, len(error))
        if result:
            raise RuntimeError(f"decode launch failed: {error.value.decode()}")
        return torch.mm(x, decoded, out_dtype=torch.float32)

    for _ in range(10):
        baseline()
        candidate()
    stream.synchronize()
    baseline()
    candidate_out = candidate()
    stream.synchronize()
    delta = (baseline_out - candidate_out).float()
    abs_delta = delta.abs()
    relative_l2 = float(torch.linalg.vector_norm(delta).item() /
                        torch.linalg.vector_norm(baseline_out).item())
    accuracy = {
        "fp32_bitwise_equal": bool(torch.equal(baseline_out, candidate_out)),
        "baseline_nonzero_values": int(torch.count_nonzero(baseline_out).item()),
        "baseline_output_abs_max": float(baseline_out.abs().max().item()),
        "decoded_weight_nonzero_values": int(torch.count_nonzero(decoded).item()),
        "mean_absolute_difference": float(abs_delta.mean().item()),
        "max_absolute_difference": float(abs_delta.max().item()),
        "relative_l2_difference": relative_l2,
    }
    if args.fp64_oracle_rows:
        if args.fp64_oracle_rows < 0 or args.fp64_oracle_rows > m:
            raise ValueError("FP64 oracle row count must be in [0, 960]")
        reference = torch.mm(
            x[:args.fp64_oracle_rows].double(), decoded.double())
        control_error = (baseline_out[:args.fp64_oracle_rows].double() - reference).abs()
        candidate_error = (candidate_out[:args.fp64_oracle_rows].double() - reference).abs()
        accuracy["fp64_oracle_rows"] = args.fp64_oracle_rows
        accuracy["fp64_control_max_abs_error"] = float(control_error.max().item())
        accuracy["fp64_candidate_max_abs_error"] = float(candidate_error.max().item())
        accuracy["fp64_control_mean_abs_error"] = float(control_error.mean().item())
        accuracy["fp64_candidate_mean_abs_error"] = float(candidate_error.mean().item())

    graphs = []
    graph_outputs = []
    for arm in (baseline, candidate):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs.append(arm())
        graphs.append(graph)
    for _ in range(20):
        graphs[0].replay()
        graphs[1].replay()
    stream.synchronize()
    x.add_(0.125)
    graphs[0].replay()
    graphs[1].replay()
    stream.synchronize()
    changed_input_delta = (baseline_out - graph_outputs[1]).abs()
    accuracy["changed_input_graph_replay_bitwise_equal"] = bool(
        torch.equal(baseline_out, graph_outputs[1]))
    accuracy["changed_input_graph_replay_max_abs_difference"] = float(
        changed_input_delta.max().item())
    x.sub_(0.125)
    stream.synchronize()

    # Paired A/B/B/A rounds keep thermal and clock drift balanced.
    latencies = [[], []]
    for round_index in range(args.rounds):
        order = (0, 1) if round_index % 2 == 0 else (1, 0)
        for arm_index in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            graphs[arm_index].replay()
            end.record(stream)
            end.synchronize()
            latencies[arm_index].append(start.elapsed_time(end) * 1000)
    print(json.dumps({
        "scope": "single_projection_transient_decode_probe_not_full_model",
        "shape": {"m": m, "k": k, "n": n, "transition_bits": 6},
        "projection": args.projection,
        "layer": args.layer,
        "input_scale": args.input_scale,
        "accuracy_vs_compressed_wgmma": accuracy,
        "median_us": {"compressed_wgmma": statistics.median(latencies[0]),
                      "transient_decode_and_gemm": statistics.median(latencies[1])},
        "candidate_scratch_bytes": decoded.numel() * decoded.element_size(),
        "candidate_weight_cache_bytes": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
