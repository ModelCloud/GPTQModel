# SPDX-License-Identifier: Apache-2.0
"""Guarded subprocess driver shared by tiny and local-model recovery checks."""


def main():
    import argparse
    import os
    import signal
    import sys

    import torch
    from transformers import AutoConfig

    from gptqmodel import BACKEND, CheckpointConfig, GPTQModel, QuantizeConfig
    from gptqmodel.looper.checkpoint import CheckpointExtension, CheckpointStopped
    from gptqmodel.quantization.config import ExpertsRoutingOverride, MoEConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--audit-hessians", action="store_true")
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
    config = AutoConfig.from_pretrained(args.model)
    calibration = [
        {
            "input_ids": torch.randint(3, min(config.vocab_size, 100), (1, 32)),
            "attention_mask": torch.ones(1, 32, dtype=torch.long),
        }
        for _ in range(4)
    ]
    qcfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device=args.device,
        offload_to_disk=True,
        offload_to_disk_path=args.output + "-offload",
        moe=MoEConfig(routing=ExpertsRoutingOverride())
        if config.model_type == "qwen3_moe"
        else None,
    )
    model = GPTQModel.load(args.model, quantize_config=qcfg, backend=BACKEND.TORCH)

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
            backend=BACKEND.TORCH,
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
    model.save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
