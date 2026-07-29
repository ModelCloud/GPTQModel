#!/usr/bin/env python3
"""Download inference-optimization/Kimi-K3-0.40B and quantize it to MXFP4 for CPU."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel.nn_modules.qlinear.mxfp4_cpu import Mxfp4CpuLinear
from gptqmodel.quantization.config import MXFP4Config


def _replace_linears(module: torch.nn.Module, prefix: str = "") -> None:
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if name == "output_attn_res_proj":
            # This linear is outside the decoder layers and not in the KimiK3QModel module_tree.
            continue
        if isinstance(child, torch.nn.Linear):
            qlinear = Mxfp4CpuLinear(
                bits=4,
                group_size=-1,
                desc_act=False,
                sym=True,
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                use_vnni=False,  # prepack is done at load time
            )
            qlinear.pack_original(child, scales=None, zeros=None)
            setattr(module, name, qlinear)
        else:
            _replace_linears(child, full_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="inference-optimization/Kimi-K3-0.40B")
    parser.add_argument("--output_dir", default="/home/ubuntu/kimi-k3-0.4b-mxfp4-cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="cpu",
    )
    model.eval()

    print("Quantizing language_model decoder to MXFP4 ...")
    _replace_linears(model.language_model.model)

    print(f"Saving MXFP4 checkpoint to {output_dir} ...")
    model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    qcfg = MXFP4Config()
    with open(output_dir / "quantize_config.json", "w", encoding="utf-8") as f:
        json.dump(qcfg.to_dict(), f, indent=2)

    print("Saved.")
    print("Verify with:")
    print(f"  GPTQModel.load('{output_dir}', backend='mxfp4_cpu', device='cpu', trust_remote_code=True)")


if __name__ == "__main__":
    main()
