"""Compare production P32 Hadamard epilogues on identical GPU inputs."""

import argparse
import ctypes
import statistics

import torch


def load(path):
    library = ctypes.CDLL(path)
    entry = library.qvq_p32_hadamard_epilogue
    entry.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    entry.restype = ctypes.c_int
    return entry


def measure(entry, source, scale, output, loops, stream):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(loops):
        status = entry(
            source.data_ptr(), scale.data_ptr(), output.data_ptr(),
            source.shape[0], source.shape[1], 1, stream.cuda_stream,
        )
        if status != 0:
            raise RuntimeError(f"Hadamard epilogue returned {status}")
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) * 1000 / loops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--rows", type=int, default=960)
    parser.add_argument("--columns", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--loops", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--profile-only", action="store_true", help="launch one baseline kernel for NCU")
    args = parser.parse_args()
    baseline = load(args.baseline)
    candidate = load(args.candidate)
    # The native ABI rejects a null stream; PyTorch's default CUDA stream has
    # handle zero, so use an explicit stream and wait for tensor initialization.
    stream = torch.cuda.Stream()
    for columns in args.columns:
        generator = torch.Generator(device="cuda").manual_seed(20260924 + columns)
        source = torch.randn(
            (args.rows, columns), device="cuda", dtype=torch.float32,
            generator=generator,
        )
        scale = torch.ones((columns,), device="cuda", dtype=torch.float16)
        control = torch.empty_like(source, dtype=torch.float16)
        changed = torch.empty_like(source, dtype=torch.float16)
        stream.wait_stream(torch.cuda.current_stream())
        if args.profile_only:
            measure(baseline, source, scale, control, 1, stream)
            continue
        measure(baseline, source, scale, control, 10, stream)
        measure(candidate, source, scale, changed, 10, stream)
        exact = torch.equal(control.view(torch.int16), changed.view(torch.int16))
        if not exact:
            mismatches = int(torch.count_nonzero(control.view(torch.int16) != changed.view(torch.int16)))
            raise AssertionError(f"M={args.rows} N={columns}: {mismatches} FP16 bit mismatches")
        samples = {"baseline": [], "candidate": []}
        for repeat in range(args.repeats):
            arms = (("baseline", baseline, control), ("candidate", candidate, changed))
            if repeat % 2:
                arms = arms[::-1]
            for name, entry, output in arms:
                samples[name].append(measure(entry, source, scale, output, args.loops, stream))
        reference_us = statistics.median(samples["baseline"])
        candidate_us = statistics.median(samples["candidate"])
        print(
            f"M={args.rows} N={columns} exact_fp16={exact} "
            f"baseline_us={reference_us:.3f} candidate_us={candidate_us:.3f} "
            f"speedup={reference_us / candidate_us:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
