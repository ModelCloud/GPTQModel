"""Compare SM90 raw input-Hadamard implementations on identical FP16 inputs."""

import argparse
import ctypes
import statistics

import torch


class Config(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_bytes", ctypes.c_uint32),
        ("rows", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
    ]


def load(path):
    library = ctypes.CDLL(path)
    launch = library.qvq_hadamard_input_raw_launch
    launch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint64, ctypes.POINTER(Config), ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint64,
    ]
    launch.restype = ctypes.c_int
    return launch


def measure(launch, source, scale, output, workspace, config, stream, loops):
    error = ctypes.create_string_buffer(256)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record(stream)
    for _ in range(loops):
        status = launch(
            source.data_ptr(), scale.data_ptr(), output.data_ptr(), workspace.data_ptr(),
            workspace.numel() * workspace.element_size(), ctypes.byref(config),
            stream.cuda_stream, error, len(error),
        )
        if status:
            raise RuntimeError(error.value.decode())
    end.record(stream)
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / loops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--rows", type=int, nargs="+", default=[128, 960])
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--loops", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    baseline, candidate = load(args.baseline), load(args.candidate)
    stream = torch.cuda.Stream()
    for rows in args.rows:
        generator = torch.Generator(device="cuda").manual_seed(20260924 + rows + args.width)
        source = torch.randn((rows, args.width), device="cuda", dtype=torch.float16, generator=generator)
        scale = (0.75 + 0.25 * torch.rand(
            (args.width,), device="cuda", dtype=torch.float16, generator=generator,
        )).contiguous()
        control = torch.empty_like(source)
        changed = torch.empty_like(source)
        scratch = torch.empty_like(source)
        config = Config(2, ctypes.sizeof(Config), rows, args.width)
        stream.wait_stream(torch.cuda.current_stream())
        if args.profile_only:
            measure(baseline, source, scale, control, scratch, config, stream, 1)
            continue
        measure(baseline, source, scale, control, scratch, config, stream, 10)
        measure(candidate, source, scale, changed, scratch, config, stream, 10)
        exact = torch.equal(control.view(torch.int16), changed.view(torch.int16))
        if not exact:
            mismatches = int(torch.count_nonzero(control.view(torch.int16) != changed.view(torch.int16)))
            raise AssertionError(f"M={rows} K={args.width}: {mismatches} FP16 bit mismatches")
        samples = {"baseline": [], "candidate": []}
        for repeat in range(args.repeats):
            arms = (("baseline", baseline, control), ("candidate", candidate, changed))
            if repeat % 2:
                arms = arms[::-1]
            for name, launch, output in arms:
                samples[name].append(measure(launch, source, scale, output, scratch, config, stream, args.loops))
        reference_us = statistics.median(samples["baseline"])
        candidate_us = statistics.median(samples["candidate"])
        print(
            f"M={rows} K={args.width} exact_fp16={exact} "
            f"baseline_us={reference_us:.3f} candidate_us={candidate_us:.3f} "
            f"speedup={reference_us / candidate_us:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
