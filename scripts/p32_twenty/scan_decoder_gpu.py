"""Experiment 21 GPU state scan versus direct windows; no GEMM/model claim."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SNAPSHOT = Path(
    "/root/qvq-results/calibration-fisher-frontier-wave14-v1/llama32-1b-f6_yaqa125x_seed7"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--group-steps", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(SNAPSHOT.resolve()):
        parser.error("Output must be outside snapshot")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.uuid
    for _ in range(3):
        values = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--id=" + args.uuid,
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            .strip()
            .split(",")
        )
        procs = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        if int(values[0]) > 8 or int(values[1]) or args.uuid in procs:
            raise RuntimeError("GPU must be idle before scan study")
        time.sleep(1)
    import torch
    import triton
    import triton.language as tl
    from safetensors import safe_open

    from gptqmodel.quantization.qvq import (
        repack_p32_planar_to_window,
        unpack_p32_window_states,
    )

    @triton.jit
    def compose(a, b, c, d):
        return tl.minimum(a + c, 16), ((b << c) ^ d) & 65535

    @triton.jit
    def decode(W, Y, T: tl.constexpr, SCAN: tl.constexpr, GROUP: tl.constexpr):
        tile = tl.program_id(0)
        i = tl.arange(0, 128)
        bit = (127 - i) * T
        word, shift = bit // 32, bit % 32
        lo = tl.load(W + tile * (4 * T) + word).to(tl.uint32)
        hi = tl.load(W + tile * (4 * T) + (word + 1) % (4 * T)).to(tl.uint32)
        state = ((lo >> shift) | tl.where(shift > 0, hi << (32 - shift), 0)) & 65535
        if SCAN:
            if GROUP == 1:
                edge = state & ((1 << T) - 1)
                shifts, suffix = tl.associative_scan(
                    (tl.full((128,), T, tl.int32), edge), 0, compose
                )
                initial = tl.sum(tl.where(i == 127, suffix, 0), 0)
                state = ((initial << shifts) ^ suffix) & 65535
            else:
                # Each packed group is an affine transfer; no exponential LUT.
                group_shift: tl.constexpr = min(16, GROUP * T)
                ends = tl.reshape(
                    tl.where(i % GROUP == GROUP - 1, state, 0), (128 // GROUP, GROUP)
                )
                edges = tl.sum(ends, 1) & ((1 << group_shift) - 1)
                shifts, suffix = tl.associative_scan(
                    (tl.full((128 // GROUP,), group_shift, tl.int32), edges), 0, compose
                )
                g = tl.arange(0, 128 // GROUP)
                initial = tl.sum(tl.where(g == 128 // GROUP - 1, suffix, 0), 0)
                boundaries = ((initial << shifts) ^ suffix) & 65535
                starts = tl.gather(boundaries, (i // GROUP + 128 // GROUP - 1) % (128 // GROUP), 0)
                local_shift = tl.minimum(16, (i % GROUP + 1) * T)
                partial = state & ((1 << local_shift) - 1)
                state = ((starts << local_shift) ^ partial) & 65535
        tl.store(Y + tile * 128 + i, state)

    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    modules = sorted(k[:-12] for k in index if k.endswith(".bank_alt_id"))
    report = {
        "experiment": 21 if args.group_steps == 1 else 24,
        "group_steps": args.group_steps,
        "scope": "GPU state-only scan versus direct-window extraction; not linear-layer or model performance",
        "uuid": args.uuid,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for prefix in modules:
        key = prefix + ".trellis"
        with safe_open(str(SNAPSHOT / index[key]), framework="pt", device="cpu") as f:
            words = f.get_tensor(key).cuda()
        bits = words.shape[-1] / 8
        window = repack_p32_planar_to_window(words, bits=bits)
        reference = unpack_p32_window_states(window, bits=bits)
        output = torch.empty(reference.shape, device="cuda", dtype=torch.int32)
        timings = {}
        for scan in [False, True]:
            for warps in [4, 8]:

                def run(
                    window=window, output=output, bits=bits, scan=scan, warps=warps
                ):
                    decode[(window.numel() // window.shape[-1],)](
                        window,
                        output,
                        int(2 * bits),
                        scan,
                        args.group_steps,
                        num_warps=warps,
                    )

                run()
                if not torch.equal(output.long(), reference):
                    raise AssertionError((prefix, scan, warps, "state mismatch"))
                for _ in range(5):
                    run()
                torch.cuda.synchronize()
                procs = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=gpu_uuid,pid",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                for line in procs.splitlines():
                    gpu, pid = [x.strip() for x in line.split(",")]
                    if gpu == args.uuid and int(pid) != os.getpid():
                        raise RuntimeError("Foreign GPU process before timing")
                samples = []
                for _ in range(10):
                    a, b = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    a.record()
                    run()
                    b.record()
                    b.synchronize()
                    samples.append(a.elapsed_time(b))
                timings[f"{'scan' if scan else 'direct'}_warps{warps}"] = samples
        report["rows"].append(
            {
                "module": prefix,
                "bits": bits,
                "tiles": window.numel() // window.shape[-1],
                "exact_states": True,
                "samples_ms": timings,
                "checkpoint_payload_bytes": words.numel() * words.element_size(),
                "output_scratch_bytes": output.numel() * output.element_size(),
            }
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("PASS", prefix, flush=True)
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
