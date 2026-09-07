# SPDX-License-Identifier: Apache-2.0
"""Guarded subprocess driver shared by tiny and local-model recovery checks."""


def main():
    import argparse
    import faulthandler
    import os
    import signal
    import sys

    faulthandler.dump_traceback_later(60)

    import torch
    from transformers import AutoConfig

    from gptqmodel import (
        BACKEND,
        CheckpointConfig,
        GPTQModel,
        QuantizeConfig,
        TelemetryConfig,
    )
    from gptqmodel.looper.checkpoint import CheckpointExtension, CheckpointStopped
    from gptqmodel.quantization.config import ExpertsRoutingOverride, MoEConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-gpus", type=int, default=0)
    parser.add_argument("--audit-hessians", action="store_true")
    parser.add_argument("--eora", action="store_true")
    parser.add_argument("--format")
    parser.add_argument("--paro-no-cudagraph", action="store_true")
    parser.add_argument(
        "--method",
        choices=[
            "gptq",
            "awq",
            "rtn",
            "fp8",
            "gguf",
            "bitsandbytes",
            "qqq",
            "paro",
            "exl3",
        ],
        default="gptq",
    )
    parser.add_argument(
        "--mode",
        default="run",
        choices=[
            "run",
            "baseline",
            "term",
            "int",
            "kill-before",
            "kill-after",
            "error",
            "kill-hessian",
            "kill-hessian-early",
        ],
    )
    args = parser.parse_args()
    if os.environ.get("PYTHON_GIL") == "0":
        assert not sys._is_gil_enabled(), "GIL enabled after model-stack imports"
    torch.manual_seed(6789)
    import random

    import numpy as np

    random.seed(6789)
    np.random.seed(6789)
    config = AutoConfig.from_pretrained(args.model)
    calibration = [
        {
            "input_ids": torch.randint(3, min(config.vocab_size, 100), (1, 32)),
            "attention_mask": torch.ones(1, 32, dtype=torch.long),
        }
        for _ in range(4)
    ]
    from gptqmodel.adapter.adapter import Lora
    from gptqmodel.quantization import FORMAT, METHOD
    from gptqmodel.quantization.config import (
        AWQConfig,
        BitsAndBytesConfig,
        EXL3Config,
        FP8Config,
        GGUFConfig,
        ParoConfig,
        QQQConfig,
        RTNConfig,
    )

    config_class = {
        "awq": AWQConfig,
        "rtn": RTNConfig,
        "fp8": FP8Config,
        "gguf": GGUFConfig,
        "bitsandbytes": BitsAndBytesConfig,
        "qqq": QQQConfig,
        "paro": ParoConfig,
        "exl3": EXL3Config,
    }.get(args.method, QuantizeConfig)
    method_settings = {
        "gptq": {"method": METHOD.GPTQ, "format": FORMAT.GPTQ, "bits": 4},
        "awq": {"method": METHOD.AWQ, "format": FORMAT.GEMM, "bits": 4},
        "rtn": {"bits": 4},
        "fp8": {"bits": 8},
        "gguf": {"bits": "q4_0"},
        "bitsandbytes": {"bits": 4},
        "qqq": {"bits": 4},
        "paro": {
            "bits": 4,
            "opt_rotation_epochs": 1,
            "opt_finetune_epochs": 1,
            "opt_train_samples": 4,
            "opt_validation_samples": 1,
        },
        "exl3": {"bits": 3.0},
    }[args.method]
    if args.format:
        method_settings["format"] = args.format
        if args.method == "bitsandbytes" and args.format == "int8":
            method_settings["bits"] = 8
    if args.paro_no_cudagraph:
        method_settings["opt_stage_cudagraph"] = False
    if args.method != "gguf":
        method_settings.update(
            group_size=128
            if args.method == "qqq"
            else -1
            if args.method == "exl3"
            else 32,
            desc_act=False,
        )
    if args.format == "gemv_fast":
        method_settings.update(group_size=128, pack_dtype=torch.int16)
    backend = {
        "fp8": BACKEND.FP8_TORCH,
        "gguf": BACKEND.GGUF_TORCH,
        "bitsandbytes": BACKEND.BITSANDBYTES,
        "qqq": BACKEND.QQQ,
        "paro": BACKEND.PAROQUANT_TRITON,
        "exl3": BACKEND.EXL3_EXLLAMA_V3,
    }.get(args.method, BACKEND.TORCH)
    if args.format in {"gemv", "gemv_fast"}:
        backend = BACKEND.AWQ_GEMV if args.format == "gemv" else BACKEND.AWQ_GEMV_FAST
    elif args.format == "bitblas":
        backend = BACKEND.AWQ_BITBLAS if args.method == "awq" else BACKEND.GPTQ_BITBLAS
    elif args.format == "llm-awq":
        backend = BACKEND.AUTO

    qcfg = config_class(
        **method_settings,
        adapter=Lora(rank=4, path=args.checkpoint + "-adapter") if args.eora else None,
        # Enable diagnostics only on resume: this must not invalidate identity.
        telemetry=TelemetryConfig(
            device=bool(args.require_gpus) and args.mode == "run"
        ),
        device=args.device,
        offload_to_disk=True,
        offload_to_disk_path=args.output + "-offload",
        moe=MoEConfig(routing=ExpertsRoutingOverride())
        if config.model_type == "qwen3_moe"
        else None,
    )
    model = GPTQModel.load(args.model, quantize_config=qcfg, backend=backend)

    if args.require_gpus:
        import json
        import threading

        from gptqmodel.quantization.gptq import GPTQ

        assert torch.cuda.device_count() == args.require_gpus
        placement_lock = threading.Lock()
        observed_devices = set()
        original_finalize = GPTQ.finalize_hessian

        def record_placement(task, *positional, **keywords):
            hessian = original_finalize(task, *positional, **keywords)
            with placement_lock:
                observed_devices.add(str(hessian.device))
                print(
                    "QUANT_DEVICE "
                    + json.dumps(
                        {
                            "name": task._named_module.full_name,
                            "device": str(hessian.device),
                        }
                    ),
                    flush=True,
                )
            return hessian

        GPTQ.finalize_hessian = record_placement

    if args.audit_hessians:
        # Observe actual task construction, accumulated batches and the exact
        # finalized Hessian consumed by GPTQ. Do not change the math or force
        # materialization earlier than the normal quantization path does.
        import hashlib
        import json
        import threading

        from gptqmodel.quantization.gptq import GPTQ

        audit_lock = threading.Lock()

        def emit(event, task, **fields):
            name = task._named_module.full_name
            with audit_lock:
                sys.stdout.write(
                    "HESSIAN_AUDIT "
                    + json.dumps({"event": event, "name": name, **fields})
                    + "\n"
                )
                sys.stdout.flush()

        def digest(tensor):
            data = (
                tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            )
            return hashlib.sha256(data).hexdigest()

        initialize = GPTQ.__init__
        add_batch = GPTQ.add_batch
        finalize_hessian = GPTQ.finalize_hessian

        def fresh(task, *positional, **keywords):
            initialize(task, *positional, **keywords)
            assert task.nsamples == task.fwd_counter == task._hessian_total_samples == 0
            assert (
                task.H is None
                and not task._device_hessian_partials
                and not task._device_sample_counts
            )
            emit("fresh", task, nsamples=task.nsamples, fwd_counter=task.fwd_counter)

        def accumulate(task, inp, out, batch_index=None):
            add_batch(task, inp, out, batch_index=batch_index)
            emit("batch", task, batch_index=batch_index, input_hash=digest(inp))
            name = task._named_module.full_name
            late_target = name.endswith((".mlp.down_proj", ".mlp.experts.0.down_proj"))
            target = (
                late_target if args.mode == "kill-hessian" else ".self_attn." in name
            )
            if (
                args.mode in {"kill-hessian", "kill-hessian-early"}
                and ".layers.1." in name
                and target
            ):
                with task.lock:
                    if task.fwd_counter == 2:
                        assert task.nsamples > 0 and task._device_hessian_partials
                        emit(
                            "kill",
                            task,
                            nsamples=task.nsamples,
                            fwd_counter=task.fwd_counter,
                            partial_hashes=[
                                digest(value)
                                for value in task._device_hessian_partials.values()
                            ],
                        )
                        os.kill(os.getpid(), signal.SIGKILL)

        def finalized(task, *positional, **keywords):
            hessian = finalize_hessian(task, *positional, **keywords)
            emit(
                "final",
                task,
                nsamples=task.nsamples,
                fwd_counter=task.fwd_counter,
                shape=list(hessian.shape),
                dtype=str(hessian.dtype),
                hessian_hash=digest(hessian),
            )
            return hessian

        GPTQ.__init__ = fresh
        GPTQ.add_batch = accumulate
        GPTQ.finalize_hessian = finalized

    original = CheckpointExtension.on_boundary

    def boundary(extension, event):
        index = event.step.index
        print(f"CHECKPOINT_BOUNDARY {index}", flush=True)
        if index == 0 and args.mode in {"int", "term"}:
            os.kill(
                os.getpid(), signal.SIGINT if args.mode == "int" else signal.SIGTERM
            )
        if index == 1 and args.mode == "kill-before":
            os.kill(os.getpid(), signal.SIGKILL)
        if index == 1 and args.mode == "error":
            write = extension.store._atomic_write

            def fail(destination, data):
                if destination.name == "CURRENT":
                    raise OSError("injected publication failure")
                write(destination, data)

            extension.store._atomic_write = fail
        original(extension, event)
        if index == 0 and args.mode == "kill-after":
            os.kill(os.getpid(), signal.SIGKILL)

    CheckpointExtension.on_boundary = boundary
    try:
        model.quantize(
            calibration,
            batch_size=1,
            backend=backend,
            calibration_data_min_length=1,
            checkpoint=None
            if args.mode == "baseline"
            else CheckpointConfig(args.checkpoint),
        )
    except CheckpointStopped:
        return 75
    except OSError as exc:
        if args.mode != "error" or str(exc) != "injected publication failure":
            raise
        return 74
    if args.require_gpus:
        assert observed_devices == {f"cuda:{i}" for i in range(args.require_gpus)}
    if os.environ.get("PYTHON_GIL") == "0":
        assert not sys._is_gil_enabled(), "GIL enabled during quantization"
    if args.require_gpus:
        from gptqmodel.utils.device_telemetry import get_device_telemetry_records

        records = get_device_telemetry_records()
        if args.mode == "run":
            assert {
                record["target_device"]
                for record in records
                if record["event"] == "quant_prepare"
            } == {f"cuda:{index}" for index in range(args.require_gpus)}, (
                "configured telemetry must propagate to both GPU worker paths"
            )
        for record in records:
            if record["event"].startswith("checkpoint_"):
                print("CHECKPOINT_TELEMETRY " + json.dumps(record), flush=True)
    model.save(args.output)
    if args.eora:
        from pathlib import Path

        from safetensors.torch import load_file, save_file

        tensors = load_file(
            Path(args.checkpoint + "-adapter") / "adapter_model.safetensors"
        )
        save_file(tensors, Path(args.output) / "adapter_model.safetensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
