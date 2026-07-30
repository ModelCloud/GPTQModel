#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""End-to-end `model.fuse()` validation on Laguna-S-2.1-GPTQ-FIXED.

Loads the real quantized checkpoint, optionally fuses QKV and gate/up groups,
runs forward/generate on the requested batch/sequence sizes, and reports
latency, throughput, and numerical parity.

GPU IDs are interpreted as physical PCI-bus-ordered indices from
`nvidia-smi` (`CUDA_DEVICE_ORDER=PCI_BUS_ID` is set at import time).  Use
`--skip-preflight` to bypass the idle-gate when nvidia-smi is unavailable.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


# Ensure we use the local repo sources, not any installed GPT-QModel egg.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def _query_nvidia_smi_gpus() -> list[dict]:
    """Return a list of GPU dicts from nvidia-smi with physical indices."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id,uuid,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
    except Exception:  # noqa: BLE001
        return []
    rows: list[dict] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append({
            "index": int(parts[0]),
            "pci": parts[1],
            "uuid": parts[2],
            "util": float(parts[3]) if parts[3] else 0.0,
            "mem": float(parts[4]) if parts[4] else 0.0,
        })
    return rows


def _parse_gpu_list(gpu_arg: str) -> list[int]:
    """Parse a comma-separated list of physical GPU indices."""
    return [int(x.strip()) for x in gpu_arg.split(",") if x.strip()]


def _set_cuda_visible_devices(args: argparse.Namespace) -> None:
    """Restrict this process to the requested GPU(s) before torch is imported.

    For single-GPU runs this is the single ``--gpu`` index.  For ``--device-map``
    runs the process is restricted to the full visible set (all physical GPUs)
    because ``device_map`` will decide how to distribute layers across them.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        # Honor an explicit external restriction.
        return

    gpu_indices = _parse_gpu_list(args.gpu)

    if args.device_map is None:
        if len(gpu_indices) != 1:
            raise ValueError(
                f"--gpu must be a single physical index for single-GPU load; got {args.gpu!r}. "
                "Use --device-map for multi-GPU."
            )
        target_uuids: list[str] = []
        for g in _query_nvidia_smi_gpus():
            if g["index"] == gpu_indices[0]:
                target_uuids.append(g["uuid"])
        if target_uuids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(target_uuids)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_indices[0])
    else:
        # Restrict to all visible GPUs; device_map will manage distribution.
        all_gpus = _query_nvidia_smi_gpus()
        if all_gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(g["uuid"] for g in all_gpus)


def _preflight(args: argparse.Namespace) -> None:
    """Verify target GPU(s) are idle before importing Torch."""
    if args.skip_preflight:
        print("Preflight: skipped (--skip-preflight).")
        return

    try:
        all_gpus = _query_nvidia_smi_gpus()
    except Exception as exc:  # noqa: BLE001
        print(f"Preflight: nvidia-smi not available ({exc}); continuing without idle check.")
        return

    if args.device_map is not None:
        targets = [g["index"] for g in all_gpus]
        label = f"all {len(targets)} visible GPUs (device_map={args.device_map})"
    else:
        targets = _parse_gpu_list(args.gpu)
        label = f"GPU(s) {targets}"

    if not targets:
        raise RuntimeError("Preflight: no target GPUs specified.")

    _check_idle(args, targets, label, "preflight")
    print("Preflight: passed.")
    last_rows = {g["index"]: g for g in _query_nvidia_smi_gpus()}
    for idx in targets:
        g = last_rows.get(idx)
        if g:
            print(
                f"  GPU {idx}: PCI {g['pci']}, UUID {g['uuid']}, "
                f"util={g['util']}%, mem={g['mem']}MiB"
            )


def _check_idle(
    args: argparse.Namespace,
    targets: list[int],
    label: str,
    stage: str,
    samples: int | None = None,
) -> None:
    """Run consecutive idle samples on the requested GPUs."""
    samples = samples if samples is not None else args.preflight_samples
    print(f"Preflight ({stage}): checking {label} for {samples} consecutive idle samples...")
    for sample in range(samples):
        rows = {g["index"]: g for g in _query_nvidia_smi_gpus()}
        for idx in targets:
            if idx not in rows:
                raise RuntimeError(f"Preflight ({stage}): requested GPU {idx} not found in nvidia-smi inventory.")
            g = rows[idx]
            if g["util"] > args.preflight_util_threshold:
                raise RuntimeError(
                    f"Preflight ({stage}): GPU {idx} has non-zero compute utilization "
                    f"({g['util']}%) at sample {sample + 1}/{samples}."
                )
            if g["mem"] > args.preflight_mem_mb:
                raise RuntimeError(
                    f"Preflight ({stage}): GPU {idx} has {g['mem']} MiB memory used "
                    f"(threshold {args.preflight_mem_mb} MiB) at sample "
                    f"{sample + 1}/{samples}."
                )
        if sample < samples - 1:
            time.sleep(1.0)


def _recheck_idle(args: argparse.Namespace, stage: str) -> None:
    """Re-run the idle check (e.g. after model load, before timing)."""
    if args.skip_preflight:
        return
    all_gpus = _query_nvidia_smi_gpus()
    if args.device_map is not None:
        targets = [g["index"] for g in all_gpus]
        label = f"all {len(targets)} visible GPUs"
    else:
        targets = _parse_gpu_list(args.gpu)
        label = f"GPU(s) {targets}"
    if not targets:
        return
    _check_idle(args, targets, label, stage, samples=1)
    print(f"Preflight ({stage}): passed.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/monster/data/model/Laguna-S-2.1-GPTQ-FIXED")
    p.add_argument("--backend", default="GPTQ_MARLIN")
    p.add_argument("--gpu", type=str, default="0",
                   help="Physical CUDA device index for single-GPU load (ignored when --device-map is set).")
    p.add_argument("--device-map", type=str, default=None,
                   help="torch device_map for load (e.g. 'auto'). When set, --gpu is ignored.")
    p.add_argument("--fuse", action="store_true", help="Apply model.fuse() before benchmark")
    p.add_argument("--fuse-qkv", action=argparse.BooleanOptionalAction, default=None,
                   help="Fuse QKV projections (default: same as --fuse; use --no-fuse-qkv to disable).")
    p.add_argument("--fuse-gate-up", action=argparse.BooleanOptionalAction, default=None,
                   help="Fuse gate/up projections (default: same as --fuse; use --no-fuse-gate-up to disable).")
    p.add_argument("--moe-grouped-dispatch", action="store_true",
                   help="Enable GPT-QModel grouped/batched MoE dispatch before benchmarking.")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--seq-len", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attn-implementation", type=str, default="flash_attention_2",
                   help="Attention implementation to request during load.")
    p.add_argument("--disable-speculative", action="store_true",
                   help="Disable any speculative decoding config from the checkpoint.")
    p.add_argument("--optimize", action="store_true",
                   help="Call model.optimize() after load/fuse to compile the model.")
    p.add_argument("--optimize-mode", type=str, default="reduce-overhead",
                   help="Torch compile mode passed to model.optimize().")
    p.add_argument("--optimize-backend", type=str, default="inductor",
                   help="Torch compile backend passed to model.optimize().")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip the nvidia-smi idle preflight.")
    p.add_argument("--preflight-samples", type=int, default=3,
                   help="Number of consecutive idle nvidia-smi samples to require.")
    p.add_argument("--preflight-util-threshold", type=float, default=0.0,
                   help="Maximum tolerated GPU compute utilization (%%) during preflight.")
    p.add_argument("--preflight-mem-mb", type=int, default=512,
                   help="Maximum tolerated GPU memory use (MiB) during preflight.")
    return p.parse_args()


# Parse args and restrict CUDA visibility before importing torch/CUDA.
args = _parse_args()
_set_cuda_visible_devices(args)
_preflight(args)

import torch  # noqa: E402


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _throughput(batch: int, seq: int, max_new_tokens: int | None, mean_ms: float) -> tuple[float, float]:
    total_tokens = batch * (seq + (max_new_tokens or 0))
    decode_tokens = batch * (max_new_tokens or 0)
    tok_per_sec = total_tokens / (mean_ms / 1000.0)
    decode_tok_per_sec = decode_tokens / (mean_ms / 1000.0) if decode_tokens > 0 else 0.0
    return tok_per_sec, decode_tok_per_sec


def _sync_all_devices() -> None:
    """Synchronize only the GPUs this process is allowed to see."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def _run_forward(
    model,
    input_ids: torch.Tensor,
    repeats: int,
    warmup: int,
    max_new_tokens: int | None,
) -> list[float]:
    # Warmup
    with torch.inference_mode():
        for _ in range(warmup):
            if max_new_tokens and max_new_tokens > 0:
                _ = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            else:
                _ = model(input_ids)
    _sync_all_devices()

    times = []
    with torch.inference_mode():
        for _ in range(repeats):
            _sync_all_devices()
            start = time.perf_counter()
            if max_new_tokens and max_new_tokens > 0:
                _ = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            else:
                _ = model(input_ids)
            _sync_all_devices()
            times.append((time.perf_counter() - start) * 1000.0)
    return times


def _vocab_size(model) -> int:
    return getattr(model, "config", None) and getattr(model.config, "vocab_size", None) or 32000


def main() -> None:
    torch.manual_seed(args.seed)

    if args.device_map is None:
        torch.cuda.set_device(0)

    from gptqmodel import BACKEND, GPTQModel

    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"GPU: {torch.cuda.current_device()} ({device_name})")
    print(f"Model: {args.model_path}")
    print(f"Backend: {args.backend}")
    print(f"Fuse: {args.fuse}")
    print(f"Batch sizes: {args.batch_sizes}, seq_len: {args.seq_len}, max_new_tokens: {args.max_new_tokens}")

    backend = getattr(BACKEND, args.backend.upper(), BACKEND.GPTQ_MARLIN)
    print(f"Loading model with attn_implementation={args.attn_implementation!r}...")
    load_kwargs = {
        "backend": backend,
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map
    else:
        load_kwargs["device"] = "cuda:0"
    model = GPTQModel.load(
        args.model_path,
        **load_kwargs,
    )
    print("Model loaded.")

    # Re-check idle state after the (possibly long) model load.
    _recheck_idle(args, "post-load")

    # Disable speculative decoding if requested; it can distort per-token
    # timing and is not relevant for the fused-quantization benchmarks.
    if args.disable_speculative:
        gen_cfg = getattr(model.model, "generation_config", None) or getattr(model, "generation_config", None)
        if gen_cfg is not None:
            if getattr(gen_cfg, "speculative_config", None) is not None:
                gen_cfg.speculative_config = None
            gen_cfg.do_sample = False
            print("Disabled speculative decoding / do_sample.")

    # Report the attention implementation that is actually in use.
    attn_impl = (
        getattr(model.config, "_attn_implementation", None)
        or getattr(model.config, "attn_implementation", None)
    )
    print(f"Attention implementation: {attn_impl}")

    input_device = next(model.model.parameters()).device if hasattr(model, "model") else torch.device("cuda:0")
    print(f"Input device: {input_device}")

    from gptqmodel.utils.moe_dispatch import (
        enable_grouped_dispatch_for_model,
        register_linear_loop_experts,
    )
    if args.moe_grouped_dispatch:
        registered = register_linear_loop_experts()
        moe_enabled = enable_grouped_dispatch_for_model(model.model)
        print(f"Grouped MoE dispatch: registered={registered}, enabled={moe_enabled} module(s)")

    vocab_size = _vocab_size(model)
    cases = [(b, args.seq_len) for b in args.batch_sizes]
    # Reuse the same input_ids for unfused and fused parity comparison.
    input_ids_by_case: dict[tuple[int, int], torch.Tensor] = {}
    for batch, seq in cases:
        input_ids_by_case[(batch, seq)] = torch.randint(
            0, vocab_size, (batch, seq), device=input_device
        )

    unfused_logits: dict[tuple[int, int], torch.Tensor] = {}
    fused_logits: dict[tuple[int, int], torch.Tensor] = {}
    results = []

    base_state = "grouped" if args.moe_grouped_dispatch else "unfused"
    for batch, seq in cases:
        input_ids = input_ids_by_case[(batch, seq)]
        print(f"\nBenchmarking {base_state} batch={batch} seq={seq}")
        with torch.inference_mode():
            out = model(input_ids)
            logits_unfused = out.logits if hasattr(out, "logits") else out
        unfused_logits[(batch, seq)] = logits_unfused.detach().cpu()

        times = _run_forward(
            model,
            input_ids,
            repeats=args.repeats,
            warmup=args.warmup,
            max_new_tokens=args.max_new_tokens,
        )
        mean_ms = _mean(times)
        tok_per_sec, decode_tok_per_sec = _throughput(batch, seq, args.max_new_tokens, mean_ms)
        results.append({
            "batch": batch,
            "seq": seq,
            "ms": mean_ms,
            "tok/s": tok_per_sec,
            "decode_tok/s": decode_tok_per_sec,
            "state": base_state,
        })
        print(f"  {base_state}: {mean_ms:.3f} ms, {tok_per_sec:.1f} tok/s ({decode_tok_per_sec:.1f} decode tok/s)")

    if args.fuse:
        fuse_qkv = args.fuse_qkv if args.fuse_qkv is not None else args.fuse
        fuse_gate_up = args.fuse_gate_up if args.fuse_gate_up is not None else args.fuse
        print("\nFusing model...")
        counts = model.fuse(
            qkv=fuse_qkv,
            gate_up=fuse_gate_up,
            gate_up_activation=fuse_gate_up,
            free_original_weights=True,
        )
        print(f"  Fused counts: {counts}")

    if args.optimize:
        print("\nCompiling model...")
        model.optimize(backend=args.optimize_backend, mode=args.optimize_mode, fullgraph=False)
        print("Model compiled.")

    if args.fuse or args.optimize:
        fused_suffix = "fused+compiled" if args.optimize else "fused"
        fused_label = f"{base_state}+{fused_suffix}"
        for batch, seq in cases:
            input_ids = input_ids_by_case[(batch, seq)]
            print(f"\nBenchmarking {fused_label} batch={batch} seq={seq}")
            with torch.inference_mode():
                out = model(input_ids)
                logits_fused = out.logits if hasattr(out, "logits") else out
            fused_logits[(batch, seq)] = logits_fused.detach().cpu()

            times = _run_forward(
                model,
                input_ids,
                repeats=args.repeats,
                warmup=args.warmup,
                max_new_tokens=args.max_new_tokens,
            )
            mean_ms = _mean(times)
            tok_per_sec, decode_tok_per_sec = _throughput(batch, seq, args.max_new_tokens, mean_ms)
            results.append({
                "batch": batch,
                "seq": seq,
                "ms": mean_ms,
                "tok/s": tok_per_sec,
                "decode_tok/s": decode_tok_per_sec,
                "state": fused_label,
            })
            print(
                f"  {fused_label}: {mean_ms:.3f} ms, {tok_per_sec:.1f} tok/s "
                f"({decode_tok_per_sec:.1f} decode tok/s)"
            )

    print("\n=== Summary ===")
    print(f"{'state':<10} {'batch':>6} {'seq':>5} {'ms':>10} {'tok/s':>12} {'decode_tok/s':>14}")
    for r in results:
        print(
            f"{r['state']:<10} {r['batch']:>6} {r['seq']:>5} {r['ms']:>10.3f} "
            f"{r['tok/s']:>12.1f} {r['decode_tok/s']:>14.1f}"
        )

    if args.fuse:
        print("\n=== Accuracy ===")
        for batch, seq in cases:
            lu = unfused_logits[(batch, seq)]
            lf = fused_logits[(batch, seq)]
            max_diff = (lu - lf).abs().max().item()
            mean_diff = (lu - lf).abs().mean().item()
            print(f"batch={batch} seq={seq}: max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f}")


if __name__ == "__main__":
    main()
