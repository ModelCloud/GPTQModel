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
