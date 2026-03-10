# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import GGUFConfig


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")

SOURCE_MODEL = "/monster/data/model/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "./Llama-3.2-1B-Instruct-GGUF-Q4_K_M"


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL, use_fast=True)

    qconfig = GGUFConfig(
        bits=4,
        format="q_k_m",
        smoother=None,
        offload_to_disk=True,
        offload_to_disk_path="./gptqmodel_offload",
    )

    print("Resolved quantize config:")
    print(f"  type   = {type(qconfig).__name__}")
    print(f"  format = {qconfig.format}")
    print(f"  bits   = {qconfig.bits!r}")
    print(f"  bits_s = {str(qconfig.bits)}")

    model = GPTQModel.from_pretrained(
        model_id_or_path=SOURCE_MODEL,
        quantize_config=qconfig,
        trust_remote_code=False,
    )

    quant_log = model.quantize(
        calibration=None,
        tokenizer=tokenizer,
        backend=BACKEND.TORCH,
    )
    print("Quantize lifecycle keys:", list(quant_log.keys()))

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    quantized = GPTQModel.from_quantized(
        model_id_or_path=str(out_dir),
        backend=BACKEND.TORCH,
        device=device,
        trust_remote_code=False,
    )

    print("Inference kernel:", quantized.qlinear_kernel.__name__)

    prompt = "Which city is the capital city of France?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = quantized.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\nPrompt:")
    print(prompt)
    print("\nGeneration:")
    print(text)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
