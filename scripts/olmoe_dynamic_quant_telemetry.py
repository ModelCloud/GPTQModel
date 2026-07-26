#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Full GPTQ 4-bit OLMoE quantization with dynamic group_size=32 on half MoE experts.

Run with PYTHON_GIL=0 on the desired GPU set. Layer lifecycle and region timings are
emitted by the built-in QuantizationRegionTimer and new per-layer telemetry added to
`stage_layer.py`.
"""

import os
import time

# ENVIRONMENT NOTE:
# This script must be launched with `PYTHON_GIL=0`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
# and the desired `CUDA_VISIBLE_DEVICES` already exported in the shell. Keeping those
# external to the script avoids accidentally overriding the user's topology choice.

from datasets import load_dataset

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.looper.module_looper import StopMainLoop
from gptqmodel.quantization.config import (
    FORMAT,
    METHOD,
    ExpertsRoutingOverride,
    MoEConfig,
    QuantizeConfig,
)


class StopAfterLayer:
    """Layer callback that raises StopMainLoop once a target layer index is reached."""

    def __init__(self, target: int):
        self._target = target

    def layer_complete(self, *, layer_idx: int, submodule_finalized: bool):
        if layer_idx >= self._target:
            return StopMainLoop(f"Requested stop after layer {layer_idx}")
        return None


def build_dynamic_half_experts() -> dict:
    """Return dynamic PCRE patterns that bump group_size to 32 for the first half of MoE experts.

    OLMoE has 64 experts per layer. This matches expert indices 0-31 across all layers
    and projection names (gate_proj, up_proj, down_proj) without matching whole layers.
    """
    # Match module names like: model.layers.0.mlp.experts.0.down_proj
    pattern = r"+:^.*\.mlp\.experts\.(3[0-1]|[12][0-9]|[0-9])\..*$"
    return {
        pattern: {
            "bits": 4,
            "group_size": 32,
            "desc_act": False,
            "sym": True,
        },
    }


def load_calibration(rows: int = 512, concat_size: int = 2048):
    ds = load_dataset("/monster/data/model/dataset/nm-calibration", name="LLM", split="train")
    ds = ds.select(range(min(rows, len(ds))))
    return ds


def main():
    model_id = os.environ.get("OLMOE_MODEL_ID", "/monster/data/model/OLMoE-1B-7B-0924")
    save_path = os.environ.get("OLMOE_SAVE_PATH", "/tmp/olmoe_gptq_4bit_gp128_dynamic")
    calibration_rows = int(os.environ.get("OLMOE_CALIBRATION_ROWS", "512"))
    concat_size = int(os.environ.get("OLMOE_CALIBRATION_CONCAT_SIZE", "2048"))
    batch_size = int(os.environ.get("OLMOE_BATCH_SIZE", "1"))
    backend = os.environ.get("OLMOE_BACKEND", "AUTO")
    stop_after_raw = os.environ.get("OLMOE_STOP_AFTER_LAYER")
    stop_after_layer = int(stop_after_raw) if stop_after_raw is not None else None

    print(f"PID={os.getpid()} PYTHON_GIL={os.environ.get('PYTHON_GIL')} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    dynamic = build_dynamic_half_experts()

    quantize_config = QuantizeConfig(
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        scale_search="activation",
        dynamic=dynamic,
        moe=MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok="all")),
        offload_to_disk=True,
    )

    print(f"QuantizeConfig: {quantize_config}")

    calibration = load_calibration(rows=calibration_rows, concat_size=concat_size)

    print(f"Loading model from {model_id}...")
    load_start = time.perf_counter()
    model = GPTQModel.load(
        model_id,
        quantize_config=quantize_config,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
    )
    print(f"Model loaded in {time.perf_counter() - load_start:.3f}s")

    if hasattr(model.config, "pad_token_id") and not model.config.pad_token_id:
        model.config.pad_token_id = model.tokenizer.pad_token_id or 0
    if hasattr(model.config, "eos_token_id") and not model.config.eos_token_id:
        model.config.eos_token_id = model.tokenizer.eos_token_id or 0

    if stop_after_layer is not None:
        model.layer_callback = StopAfterLayer(stop_after_layer)
        print(f"Stop-after-layer callback installed at layer {stop_after_layer}")

    backend_enum = BACKEND.AUTO
    if backend != "AUTO":
        backend_enum = BACKEND[backend]

    print("Starting quantization...")
    quant_start = time.perf_counter()
    model.quantize(
        calibration,
        calibration_concat_size=concat_size,
        calibration_sort="desc",
        backend=backend_enum,
        batch_size=batch_size,
    )
    print(f"Quantization completed in {time.perf_counter() - quant_start:.3f}s")

    print(f"Saving quantized model to {save_path}...")
    save_start = time.perf_counter()
    os.makedirs(save_path, exist_ok=True)
    model.save(save_path)
    print(f"Saved in {time.perf_counter() - save_start:.3f}s")

    print("Done.")


if __name__ == "__main__":
    main()
