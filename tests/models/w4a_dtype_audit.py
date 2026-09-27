# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Assert encoded W4A activation handoffs across Llama decoder operators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .w4a_gb10_memory import require_w4a_test_headroom


def audit(checkpoint: Path, variant: str, *, require_full_coverage: bool = True) -> dict:
    require_w4a_test_headroom(require_scope=True)

    from transformers import AutoTokenizer

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.w4a_activation import W4AActivation
    from gptqmodel.nn_modules.qlinear.w4a_floatx import W4AFP8Linear

    backend = {
        "w4afp8": BACKEND.GPTQ_W4AFP8,
    }[variant]
    model = GPTQModel.load(str(checkpoint), backend=backend, device="cuda:0", dtype=torch.bfloat16)
    model.model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    input_ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"].to("cuda:0")

    samples: dict[str, dict] = {}
    handles = []

    def describe(value):
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, W4AActivation):
            return {"kind": "encoded", "mode": value.mode, "codes": str(value.codes.dtype),
                    "recipe": value.recipe,
                    "scales": str(value.scales.dtype), "global_scale":
                    str(value.global_scale.dtype) if value.global_scale is not None else None,
                    "shape": list(value.shape), "codes_ptr": value.codes.data_ptr(),
                    "scales_ptr": value.scales.data_ptr()}
        if isinstance(value, torch.Tensor):
            return {"kind": "tensor", "dtype": str(value.dtype), "shape": list(value.shape)}
        raise AssertionError(f"Unexpected activation type: {type(value)}")

    def before(name: str):
        def hook(_module, args, kwargs):
            if not args and "hidden_states" not in kwargs:
                raise AssertionError(f"{name} received no input")
            samples[name]["input"] = describe(args[0] if args else kwargs["hidden_states"])
        return hook

    def after(name: str):
        def hook(_module, _args, result):
            samples[name]["output"] = describe(result)
        return hook

    for name, module in model.model.named_modules():
        if isinstance(module, (W4AFP8Linear,)):
            kind = "w4a_linear"
            if not getattr(module, "_require_activation_stream", False):
                raise AssertionError(f"{name} permits a BF16/FP16 fallback inside a stream model")
        elif module.__class__.__name__ == "LlamaDecoderLayer":
            kind = "decoder_layer"
        elif module.__class__.__name__ == "LlamaRMSNorm":
            kind = "rms_norm"
        elif module.__class__.__name__ == "LlamaAttention":
            kind = "attention"
        elif module.__class__.__name__ == "LlamaMLP":
            kind = "mlp"
        else:
            continue
        selected = kind == "w4a_linear" or any(
            name == f"model.layers.{index}" or name.startswith(f"model.layers.{index}.")
            for index, layer in enumerate(model.model.model.layers)
            if getattr(layer, "_w4a_stream_mode", None) == variant
        )
        samples[name] = {"kind": kind, "selected": selected}
        handles.append(module.register_forward_pre_hook(before(name), with_kwargs=True))
        handles.append(module.register_forward_hook(after(name)))

    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids, use_cache=False)
            generated = model.model.generate(input_ids=input_ids, max_new_tokens=2, do_sample=False,
                                             pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()

    counts = {}
    expected_recipe = model.quantize_config.activation_recipe
    activation_version = model.quantize_config.activation_version
    if not bool(torch.isfinite(output.logits).all()) or generated.shape[1] != input_ids.shape[1] + 2:
        raise AssertionError("W4A stream forward or cached generation produced invalid output.")
    for name, entry in samples.items():
        kind = entry["kind"]
        if "input" not in entry or "output" not in entry:
            raise AssertionError(f"Hook did not run for {entry}")
        counts[kind] = counts.get(kind, 0) + 1
        if entry["selected"]:
            expected_code = "torch.float8_e4m3fn"
            output_encoded = activation_version == 2 or kind in {"decoder_layer", "rms_norm"}
            if output_encoded:
                if entry["output"]["kind"] != "encoded" or entry["output"]["codes"] != expected_code:
                    raise AssertionError(f"Selected W4A carrier boundary is not encoded: {name}: {entry}")
                if entry["output"]["recipe"] != expected_recipe:
                    raise AssertionError(f"Selected W4A boundary has the wrong scale recipe: {name}: {entry}")
            elif (entry["output"]["kind"] != "tensor"
                  or entry["output"]["dtype"] != "torch.bfloat16"):
                raise AssertionError(
                    f"A version-3 Linear/nonlinear branch did not emit model dtype: {name}: {entry}"
                )
            if kind != "decoder_layer" and (entry["input"]["kind"] != "encoded" or
                                               entry["input"]["codes"] != expected_code):
                raise AssertionError(f"Selected W4A consumer did not receive encoded input: {name}: {entry}")
            if kind != "decoder_layer" and entry["input"]["recipe"] != expected_recipe:
                raise AssertionError(f"Selected W4A consumer received the wrong scale recipe: {name}: {entry}")
        elif name == "model.norm" and getattr(model.model.model.layers[-1], "_w4a_stream_mode", None) == variant:
            if entry["input"]["kind"] != "encoded" or entry["output"].get("dtype") != "torch.bfloat16":
                raise AssertionError(f"Final norm did not consume the encoded stream: {entry}")
        elif entry["input"].get("dtype") != "torch.bfloat16" or entry["output"].get("dtype") != "torch.bfloat16":
            raise AssertionError(f"Unexpected dense boundary dtype: {name}: {entry}")
    if not counts.get("w4a_linear") or not counts.get("decoder_layer"):
        raise AssertionError(f"No W4A or decoder hooks ran: {counts}")

    selected_layers = [name for name, entry in samples.items()
                       if entry["kind"] == "decoder_layer" and entry["selected"]]
    if require_full_coverage and len(selected_layers) != len(model.model.model.layers):
        raise AssertionError(
            f"Full W4A coverage requires all {len(model.model.model.layers)} decoder layers; "
            f"found {len(selected_layers)}."
        )
    for prior, following in zip(selected_layers, selected_layers[1:]):
        if int(following.rsplit(".", 1)[1]) != int(prior.rsplit(".", 1)[1]) + 1:
            continue  # A dense layer between them is an explicit model-dtype boundary.
        if samples[following]["input"]["kind"] != "encoded":
            raise AssertionError(f"W4A stream was lost between {prior} and {following}.")
        for pointer in ("codes_ptr", "scales_ptr"):
            if samples[prior]["output"][pointer] != samples[following]["input"][pointer]:
                raise AssertionError(f"W4A {pointer} changed between {prior} and {following}.")
    for layer_name in selected_layers:
        for projections in (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                            ("mlp.gate_proj", "mlp.up_proj")):
            inputs = [samples[f"{layer_name}.{projection}"]["input"] for projection in projections]
            for pointer in ("codes_ptr", "scales_ptr"):
                if len({entry[pointer] for entry in inputs}) != 1:
                    raise AssertionError(f"Shared W4A input was re-packed for {layer_name}: {projections}.")

    return {
        "checkpoint": str(checkpoint.resolve()),
        "variant": variant,
        "activation_recipe": expected_recipe,
        "activation_version": activation_version,
        "model_dtype": "torch.bfloat16",
        "logits_dtype": str(output.logits.dtype),
        "input_tokens": input_ids.shape[1],
        "generated_tokens": generated.shape[1] - input_ids.shape[1],
        "counts": counts,
        "selected_layers": selected_layers,
        "full_coverage": len(selected_layers) == len(model.model.model.layers),
        "boundaries": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=("w4afp8",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-full-coverage", action=argparse.BooleanOptionalAction, default=True,
        help="Require every decoder layer (default); opt out only for partial-model diagnostics.",
    )
    args = parser.parse_args()
    report = audit(args.checkpoint, args.variant, require_full_coverage=args.require_full_coverage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "boundaries"}))


if __name__ == "__main__":
    main()
