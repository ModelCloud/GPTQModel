# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Full GPTQ quantization timing comparison: fuse_same_input_forward vs baseline.

Runs the same Llama 3.2 1B Instruct calibration twice, once with fusion
enabled and once without, and reports wall time, peak VRAM, and final loss.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CALIBRATION = [
    "The quick brown fox jumps over the lazy dog. " * 20,
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data. " * 10,
    "Quantization reduces the precision of model weights to lower memory and inference costs. " * 10,
    "Large language models have demonstrated remarkable capabilities in natural language understanding. " * 10,
    "The transformer architecture relies on self-attention mechanisms to process sequential data. " * 10,
] * 2

CHILD_SCRIPT = """
import gc, os, sys, time, torch
from gptqmodel import GPTQModel, QuantizeConfig, FusedForwardConfig, BACKEND

fuse = bool(int(os.environ["_FUSE"]))
splice = os.environ.get("_SPLICE", "view")

cfg = QuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
    act_group_aware=True,
    scale_search="activation",
    damp_percent=0.05,
    fused_forward=FusedForwardConfig(splice=splice) if fuse else None,
)

t0 = time.perf_counter()
model = GPTQModel.load(
    "/monster/data/model/Llama-3.2-1B-Instruct",
    quantize_config=cfg,
    trust_remote_code=False,
    dtype="auto",
    device_map="cuda:0",
    attn_implementation="eager",
)
load_s = time.perf_counter() - t0

cal = [
    "The quick brown fox jumps over the lazy dog. " * 20,
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data. " * 10,
    "Quantization reduces the precision of model weights to lower memory and inference costs. " * 10,
    "Large language models have demonstrated remarkable capabilities in natural language understanding. " * 10,
    "The transformer architecture relies on self-attention mechanisms to process sequential data. " * 10,
] * 2

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.perf_counter()
model.quantize(
    cal,
    calibration_concat_size=256,
    calibration_sort="desc",
    batch_size=1,
    backend=BACKEND.GPTQ_TORCH,
)
torch.cuda.synchronize()
quant_s = time.perf_counter() - t0
peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

print("BENCH_LOAD_TIME: %.3f" % load_s)
print("BENCH_QUANT_TIME: %.3f" % quant_s)
print("BENCH_PEAK_MEM_GB: %.3f" % peak_gb)
print("BENCH_FUSE: %d" % int(fuse))
"""


def run_variant(fuse: bool) -> dict:
    repo_root = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    env["_FUSE"] = "1" if fuse else "0"
    env["_SPLICE"] = os.environ.get("FUSED_SPLICE", "view")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTEST_CURRENT_TEST"] = "benchmark_fused_quant_full"
    env["LOGBAR_ANIMATION"] = "0"
    env["LOGBAR_PROGRESS_OUTPUT_INTERVAL"] = "1000"
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = "6"

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(CHILD_SCRIPT)
        child_path = f.name

    proc = subprocess.Popen(
        [sys.executable, child_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )

    load_time = None
    quant_time = None
    peak_mem = None
    loss_mean = None
    loss_max = None
    line_re = re.compile(r"BENCH_(\w+):\s*([\d.]+|True|False)")
    loss_re = re.compile(r"Quantization loss summary:.*?mean=([\d.eE+-]+).*?max=([\d.eE+-]+)")

    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        m = loss_re.search(line)
        if m:
            loss_mean = float(m.group(1))
            loss_max = float(m.group(2))
        m = line_re.search(line)
        if m:
            k, v = m.group(1), m.group(2)
            if k == "LOAD_TIME":
                load_time = float(v)
            elif k == "QUANT_TIME":
                quant_time = float(v)
            elif k == "PEAK_MEM_GB":
                peak_mem = float(v)

    proc.wait()
    os.unlink(child_path)

    if proc.returncode != 0:
        raise RuntimeError(f"Variant fuse={fuse} failed with code {proc.returncode}")

    return {
        "fuse": fuse,
        "load_s": load_time,
        "quant_s": quant_time,
        "peak_gb": peak_mem,
        "loss_mean": loss_mean,
        "loss_max": loss_max,
    }


def main():
    print("Running baseline (fuse=False) ...")
    base = run_variant(False)
    print("\nRunning fused (fuse=True) ...")
    fused = run_variant(True)

    print("\n" + "=" * 80)
    print(f"{'fuse':>6} | {'load_s':>8} | {'quant_s':>9} | {'peak_GB':>8} | {'loss_mean':>12} | {'loss_max':>12}")
    print("-" * 80)
    for r in (base, fused):
        print(f"{str(r['fuse']):>6} | {r['load_s']:>8.2f} | {r['quant_s']:>9.2f} | {r['peak_gb']:>8.2f} | {r['loss_mean']:>12.6e} | {r['loss_max']:>12.6e}")

    if base["quant_s"] and fused["quant_s"]:
        diff = fused["quant_s"] - base["quant_s"]
        pct = diff / base["quant_s"] * 100
        print("-" * 80)
        print(f"Fused vs baseline: {diff:+.2f}s ({pct:+.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
