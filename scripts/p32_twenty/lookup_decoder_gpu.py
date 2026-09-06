"""Experiments 5/6: direct PGC arithmetic versus index/pair LUT decode.

State/pair materialization only; no GEMM or model-speed claim. Snapshot read only.
"""

import argparse
import json
import os
import struct
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uuid", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--packed-half", action="store_true")
    args = p.parse_args()
    if args.output.resolve().is_relative_to(SNAPSHOT.resolve()):
        p.error("Output must be outside snapshot")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.uuid
    for _ in range(3):
        fields = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=" + args.uuid,
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).split(",")
        procs = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        if int(fields[0]) > 8 or int(fields[1]) or args.uuid in procs:
            raise RuntimeError("GPU not idle")
        time.sleep(1)
    import torch
    import triton
    import triton.language as tl
    from safetensors import safe_open

    from gptqmodel.quantization.qvq import (
        decode_p32_window_tiles,
        repack_p32_planar_to_window,
    )
    from gptqmodel.quantization.qvq_codecs import pgc16_levels_for_version

    @triton.jit
    def decode(
        W,
        BANK,
        LEVELS,
        INDEX,
        PAIRS,
        Y,
        T: tl.constexpr,
        ALT: tl.constexpr,
        MODE: tl.constexpr,
        PACKED: tl.constexpr,
    ):
        tile = tl.program_id(0)
        pair = tl.arange(0, 128)
        position = (127 - pair) * T
        word, shift = position // 32, position % 32
        lo = tl.load(W + tile * (4 * T) + word).to(tl.uint32)
        hi = tl.load(W + tile * (4 * T) + (word + 1) % (4 * T)).to(tl.uint32)
        state = ((lo >> shift) | tl.where(shift > 0, hi << (32 - shift), 0)) & 65535
        bank = tl.load(BANK + tile).to(tl.uint32)
        state = state ^ (((bank >> (pair // 16)) & 1) * ALT)
        if MODE == 2:
            if PACKED:
                bits = tl.load(PAIRS + state)
                tl.store(Y + tile * 128 + pair, bits)
            else:
                component = tl.arange(0, 2)
                values = tl.load(PAIRS + state[:, None] * 2 + component[None, :])
                tl.store(
                    Y + tile * 256 + pair[:, None] * 2 + component[None, :], values
                )
        else:
            if MODE == 1:
                mixed = tl.load(INDEX + state).to(tl.uint32)
            else:
                mixed = state ^ (state >> 8)
                mixed = (mixed * 40503 + 17011) & 65535
                mixed = mixed ^ (mixed >> 7)
            if PACKED:
                first = (
                    tl.load(LEVELS + (mixed >> 8))
                    .to(tl.uint16, bitcast=True)
                    .to(tl.uint32)
                )
                second = (
                    tl.load(LEVELS + (mixed & 255))
                    .to(tl.uint16, bitcast=True)
                    .to(tl.uint32)
                )
                tl.store(Y + tile * 128 + pair, first | (second << 16))
            else:
                component = tl.arange(0, 2)
                indices = tl.where(
                    component[None, :] == 0, mixed[:, None] >> 8, mixed[:, None] & 255
                )
                values = tl.load(LEVELS + indices)
                tl.store(
                    Y + tile * 256 + pair[:, None] * 2 + component[None, :], values
                )

    cfg = json.loads((SNAPSHOT / "quantize_config.json").read_text())
    levels = pgc16_levels_for_version(cfg["codebook"]).cuda()
    if args.packed_half:
        levels = levels.half()
    state = torch.arange(65536, device="cuda", dtype=torch.int64)
    mixed = state ^ (state >> 8)
    mixed = (mixed * 40503 + 17011) & 65535
    mixed = mixed ^ (mixed >> 7)
    indices = mixed.to(torch.uint16)
    pairs = torch.stack([levels[mixed >> 8], levels[mixed & 255]], dim=1).contiguous()
    if args.packed_half:
        pairs = pairs.view(torch.int32).reshape(-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lut_files = []
    for name, tensor in [("indices", indices), ("pairs", pairs)]:
        metadata = json.dumps(
            {
                "format": "P32LUT1",
                "codebook": cfg["codebook"],
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        ).encode()
        payload = (
            struct.pack("<I", len(metadata)) + metadata + tensor.cpu().numpy().tobytes()
        )
        path = args.output.parent / (args.output.stem + "-" + name + ".bin")
        path.write_bytes(payload)
        lut_files.append({"name": name, "path": str(path), "bytes": len(payload)})
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    report = {
        "experiment": 6 if args.packed_half else 5,
        "packed_half": args.packed_half,
        "scope": "unfused decoded pairs; not linear-layer performance",
        "lut_files": lut_files,
        "uuid": args.uuid,
        "rows": [],
    }
    masks = {
        4: [0, 0x5A5A, 0x3C3C, 0xC3C3],
        5: [0, 0x9696, 0x3C3C, 0xC3C3],
        6: [0, 0x6969, 0x5A5A, 0x3C3C],
        7: [0, 0xC3C3, 0x9696, 0x5A5A],
    }
    for module in sorted(k[:-12] for k in index if k.endswith(".bank_alt_id")):

        def read(suffix, module=module):
            key = module + "." + suffix
            with safe_open(
                str(SNAPSHOT / index[key]), framework="pt", device="cpu"
            ) as f:
                return f.get_tensor(key).cuda()

        words, bank, alt = [read(k) for k in ["trellis", "bank_ids", "bank_alt_id"]]
        bits = words.shape[-1] / 8
        t = int(bits * 2)
        window = repack_p32_planar_to_window(words, bits=bits)
        expected = decode_p32_window_tiles(
            window,
            bits=bits,
            bank_ids=bank,
            bank_alt_id=alt,
            codebook_version=cfg["codebook"],
        )
        if args.packed_half:
            expected = expected.half().view(torch.int32)
        alt_mask = masks[t][int(alt.item())]
        output = torch.empty_like(expected)
        samples = {}
        for mode in [0, 1, 2]:
            for warps in [1, 4]:

                def run(
                    window=window,
                    bank=bank,
                    t=t,
                    alt_mask=alt_mask,
                    mode=mode,
                    warps=warps,
                    output=output,
                ):
                    decode[(window.numel() // window.shape[-1],)](
                        window,
                        bank,
                        levels,
                        indices,
                        pairs,
                        output,
                        t,
                        alt_mask,
                        mode,
                        args.packed_half,
                        num_warps=warps,
                    )

                run()
                if not torch.equal(output, expected):
                    raise AssertionError((module, mode, warps, "pair mismatch"))
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
                    gpu, pid = [v.strip() for v in line.split(",")]
                    if gpu == args.uuid and int(pid) != os.getpid():
                        raise RuntimeError("Foreign GPU process")
                times = []
                for _ in range(10):
                    a, b = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    a.record()
                    run()
                    b.record()
                    b.synchronize()
                    times.append(a.elapsed_time(b))
                samples[f"mode{mode}_warps{warps}"] = times
        report["rows"].append(
            {
                "module": module,
                "bits": bits,
                "exact_pairs": True,
                "samples_ms": samples,
                "output_bytes": output.numel() * output.element_size(),
            }
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("PASS", module, flush=True)
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
