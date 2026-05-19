#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME

from gptqmodel import GPTQModel
from gptqmodel import extension
from gptqmodel.quantization.config import FORMAT, METHOD, ParoConfig
from gptqmodel.utils.paroquant import (
    apply_paroquant_rotation,
    build_identity_rotation_buffers,
    clear_paroquant_rotation_extension_cache,
)

CALIBRATION_TEXTS = [
    "Summarize the role of CUDA kernel compilation in PyTorch custom operators.",
    "Explain why a quantization fallback path can make model compression much slower.",
    "Qwen models are decoder-only transformers optimized for generation workloads.",
    "ParoQuant applies pairwise rotations before quantization to reduce approximation error.",
    "A small calibration set is enough for reproducing failures even when accuracy is not the goal.",
    "The purpose of this run is to reproduce the issue path, not to measure final model quality.",
    "When a JIT extension fails instantly, the root cause is often toolchain discovery rather than CUDA execution.",
    "Quantization logs should clearly distinguish compilation failures from runtime numerical problems.",
]

pytestmark = [pytest.mark.model, pytest.mark.slow]


#   python tests_ParoQuant.py --mode quantize \
#     --model /monster/data/model/Qwen3.5-27B \
#     --output-dir /tmp/paroquant_qwen3_0_6b_test \
#     --calibration-samples 8 \
#     --batch-size 1 \
#     --opt-rotation-epochs 1 \
#     --opt-finetune-epochs 1 \
#     --opt-train-samples 8 \
#     --opt-validation-samples 1 \
#     --opt-batch-size 4

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ParoQuant repro helper")
    parser.add_argument("--mode", choices=("all", "jit", "quantize"), default="all")
    parser.add_argument("--rebuild", action="store_true", help="clear ParoQuant JIT cache before probing")
    parser.add_argument(
        "--model",
        default="/monster/data/model/Qwen3-0.6B-Base",
        help="local model path for quantize mode",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/paroquant_qwen3_0_6b_test",
        help="save path for quantized model",
    )
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calibration-samples", type=int, default=8)
    parser.add_argument("--calibration-concat-size", type=int, default=0)
    parser.add_argument("--opt-rotation-epochs", type=int, default=1)
    parser.add_argument("--opt-finetune-epochs", type=int, default=1)
    parser.add_argument("--opt-train-samples", type=int, default=8)
    parser.add_argument("--opt-validation-samples", type=int, default=1)
    parser.add_argument("--opt-batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    return parser.parse_args()


def _default_args() -> argparse.Namespace:
    return argparse.Namespace(
        mode="all",
        rebuild=False,
        model="/monster/data/model/Qwen3-0.6B-Base",
        output_dir="/tmp/paroquant_qwen3_0_6b_test",
        bits=4,
        group_size=128,
        batch_size=1,
        calibration_samples=8,
        calibration_concat_size=0,
        opt_rotation_epochs=1,
        opt_finetune_epochs=1,
        opt_train_samples=8,
        opt_validation_samples=1,
        opt_batch_size=4,
        dtype="bfloat16",
    )


def print_environment() -> None:
    print("== Environment ==")
    print(f"python={sys.version}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_home={CUDA_HOME}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"device_count={torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        print(f"device[{idx}] capability={torch.cuda.get_device_capability(idx)}")


def run_jit_repro(*, rebuild: bool) -> int:
    print_environment()
    if not torch.cuda.is_available():
        print("CUDA is not available, skip ParoQuant repro.")
        return 2

    print("\n== JIT Repro ==")
    if rebuild:
        clear_paroquant_rotation_extension_cache()
    started = time.perf_counter()
    ok = extension.is_available("paroquant", use_cache=not rebuild)
    elapsed = time.perf_counter() - started
    print(f"is_available={ok}")
    print(f"elapsed={elapsed:.3f}s")
    print(f"error={extension.error('paroquant')}")
    if not ok:
        return 1

    print("\n== Fused Rotation Probe ==")
    device = torch.device("cuda:0")
    x = torch.randn(32, 128, device=device, dtype=torch.bfloat16)
    pairs, theta, scales = build_identity_rotation_buffers(
        in_features=128,
        group_size=128,
        krot=1,
        device=device,
        dtype=torch.bfloat16,
    )
    y = apply_paroquant_rotation(x, pairs, theta, scales, group_size=128)
    print(f"output_shape={tuple(y.shape)}")
    print(f"output_dtype={y.dtype}")
    print(f"max_abs_diff={(y - x).abs().max().item():.6f}")
    return 0


def _resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def run_quantize_repro(args: argparse.Namespace) -> int:
    print_environment()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"model path does not exist: {model_path}")
        return 2
    if not torch.cuda.is_available():
        print("CUDA is not available, skip quantize repro.")
        return 2

    calibration_dataset = CALIBRATION_TEXTS[: args.calibration_samples]
    print("\n== Quantize Setup ==")
    print(f"model={model_path}")
    print(f"output_dir={args.output_dir}")
    print(f"calibration_samples={len(calibration_dataset)}")
    print(f"batch_size={args.batch_size}")

    qcfg = ParoConfig(
        bits=args.bits,
        group_size=args.group_size,
        method=METHOD.PARO,
        format=FORMAT.PAROQUANT,
        opt_scope="module",
        opt_rotation_epochs=args.opt_rotation_epochs,
        opt_finetune_epochs=args.opt_finetune_epochs,
        opt_train_samples=args.opt_train_samples,
        opt_validation_samples=args.opt_validation_samples,
        opt_batch_size=args.opt_batch_size,
        opt_pair_impl="fast",
        opt_quantizer_impl="reference",
        opt_stage_impl="fast",
        offload_to_disk=True,
    )

    if args.rebuild:
        clear_paroquant_rotation_extension_cache()

    print("\n== Load Model ==")
    load_started = time.perf_counter()
    model = GPTQModel.load(
        str(model_path),
        quantize_config=qcfg,
        trust_remote_code=False,
        dtype=_resolve_dtype(args.dtype),
    )
    print(f"load_elapsed={time.perf_counter() - load_started:.3f}s")

    print("\n== Quantize ==")
    quant_started = time.perf_counter()
    quant_logs = model.quantize(
        calibration_dataset,
        batch_size=args.batch_size,
        calibration_concat_size=args.calibration_concat_size,
        calibration_sort="desc",
    )
    quant_elapsed = time.perf_counter() - quant_started
    print(f"quant_elapsed={quant_elapsed:.3f}s")
    print(f"quant_log_keys={sorted(quant_logs.keys()) if isinstance(quant_logs, dict) else type(quant_logs).__name__}")
    print(f"paroquant_extension_error={extension.error('paroquant')}")

    print("\n== Save ==")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_started = time.perf_counter()
    model.save(str(output_dir))
    print(f"save_elapsed={time.perf_counter() - save_started:.3f}s")
    print(f"saved_to={output_dir}")
    return 0


def main() -> int:
    os.environ.setdefault("GPTQMODEL_EXT_VERBOSE", "1")
    args = parse_args()
    if args.mode == "jit":
        return run_jit_repro(rebuild=args.rebuild)
    if args.mode == "quantize":
        return run_quantize_repro(args)

    jit_result = run_jit_repro(rebuild=args.rebuild)
    if jit_result != 0:
        print("\nJIT repro failed; skip quantize repro.")
        return jit_result

    print("\n== Proceed To Quantize Repro ==")
    return run_quantize_repro(args)


def test_paroquant_jit_then_quantize_qwen3_0_6b():
    os.environ.setdefault("GPTQMODEL_EXT_VERBOSE", "1")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available for ParoQuant repro.")

    args = _default_args()
    model_path = Path(args.model)
    if not model_path.exists():
        pytest.skip(f"Model path missing: {model_path}")

    jit_result = run_jit_repro(rebuild=args.rebuild)
    assert jit_result == 0, f"ParoQuant JIT repro failed with exit code {jit_result}"

    quant_result = run_quantize_repro(args)
    assert quant_result == 0, f"ParoQuant quantize repro failed with exit code {quant_result}"


if __name__ == "__main__":
    raise SystemExit(main())
