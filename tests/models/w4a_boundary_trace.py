# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Profile reconstruction error at every end-to-end W4A carrier boundary."""

from __future__ import annotations

import argparse
import inspect
import json
from collections import defaultdict
from pathlib import Path

import torch

from .w4a_gb10_memory import require_w4a_test_headroom


def _category(caller: str, module_name: str | None) -> str:
    if caller == "_as_stream":
        return "decoder_input"
    if caller == "_norm_forward":
        return "rmsnorm_output"
    if caller == "_attention_forward":
        return "attention_to_o_proj"
    if caller == "_mlp_forward":
        return "mlp_product_to_down_proj"
    if caller == "_add_stream":
        return "residual_sum"
    if caller == "forward" and module_name:
        return f"linear_output:{module_name.rsplit('.', 1)[-1]}"
    return caller if not module_name else f"{caller}:{module_name}"


def _balanced_group_permutation(x: torch.Tensor) -> torch.Tensor:
    """Spread high-energy channels over block-16 scales within each GPTQ group."""
    width = x.shape[-1]
    if width % 128:
        raise ValueError("Balanced NVFP4 profiling requires K divisible by 128.")
    reduce_dims = tuple(range(x.ndim - 1))
    energy = x.float().square().mean(dim=reduce_dims)
    groups = []
    for start in range(0, width, 128):
        order = torch.argsort(energy[start:start + 128], descending=True) + start
        # Eight NVFP4 scale blocks each receive every eighth ranked channel.
        groups.extend(order[offset::8] for offset in range(8))
    return torch.cat(groups)


def profile(checkpoint: Path, prompt_result: Path, output: Path) -> dict:
    require_w4a_test_headroom(require_scope=True)

    from transformers import AutoTokenizer

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear import w4a_activation as activation_module
    from gptqmodel.nn_modules.qlinear import w4a_llama_stream as stream_module
    from gptqmodel.nn_modules.qlinear import w4a_boundary as boundary_module

    model = GPTQModel.load(
        str(checkpoint), backend=BACKEND.GPTQ_W4A_NVFP4,
        device="cuda:0", dtype=torch.bfloat16,
    )
    core = model.model
    core.eval()
    names = {id(module): name for name, module in core.named_modules()}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    prompt = json.loads(prompt_result.read_text())["tests"][0]["samples"][0]["prompt"]
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda:0")

    original = activation_module.pack_activation
    records: list[dict] = []

    def measured_pack(x, mode, **kwargs):
        frame = inspect.currentframe().f_back
        try:
            caller = frame.f_code.co_name
            # Residual packing lives in a free function. Find its enclosing
            # decoder so both residual boundaries retain their layer identity.
            module_name = None
            while frame is not None and module_name is None:
                owner = frame.f_locals.get("self")
                module_name = names.get(id(owner)) if owner is not None else None
                if isinstance(owner, boundary_module.NVFP4BoundaryQuantizer):
                    module_name = owner.key
                    if module_name.endswith("self_attn.o_proj.input"):
                        caller = "_attention_forward"
                    elif module_name.endswith("mlp.down_proj.input"):
                        caller = "_mlp_forward"
                    elif module_name.endswith(".input"):
                        caller = "_as_stream"
                    else:
                        caller = "_add_stream"
                frame = frame.f_back
        finally:
            del frame
        encoded = original(x, mode, **kwargs)
        if mode != "w4a_nvfp4" or x.numel() == 0:
            return encoded
        reference = x.float()
        reconstructed = encoded.decode(torch.float32)
        delta = reconstructed - reference
        reference_rms = reference.square().mean().sqrt()
        delta_rms = delta.square().mean().sqrt()
        reference_energy = reference.square().sum().clamp_min(1e-30)
        radial_error = (delta * reference).sum() / reference_energy
        energy_ratio = reconstructed.square().sum() / reference_energy
        permutation = _balanced_group_permutation(reference)
        balanced_encoded = original(reference[..., permutation], mode, **kwargs)
        balanced_decoded = balanced_encoded.decode(torch.float32)
        balanced_reconstructed = torch.empty_like(reference)
        balanced_reconstructed[..., permutation] = balanced_decoded
        balanced_delta_rms = (balanced_reconstructed - reference).square().mean().sqrt()
        balanced_relative_rmse = balanced_delta_rms / reference_rms.clamp_min(1e-30)
        relative_rmse = delta_rms / reference_rms.clamp_min(1e-30)
        flat_reference = reference.flatten()
        flat_reconstructed = reconstructed.flatten()
        records.append({
            "index": len(records),
            "category": _category(caller, module_name),
            "caller": caller,
            "module": module_name,
            "shape": list(x.shape),
            "relative_rmse": float(relative_rmse.item()),
            "radial_error_ratio": float(radial_error.item()),
            "reconstructed_energy_ratio": float(energy_ratio.item()),
            "balanced_relative_rmse": float(balanced_relative_rmse.item()),
            "balanced_error_ratio": float((balanced_relative_rmse / relative_rmse.clamp_min(1e-30)).item()),
            "cosine": float(torch.nn.functional.cosine_similarity(
                flat_reference, flat_reconstructed, dim=0
            ).item()),
            "reference_absmax": float(reference.abs().amax().item()),
            "error_absmax": float(delta.abs().amax().item()),
            "global_scale": float(encoded.global_scale.item()),
            "local_scale_min": float(encoded.scales.float().amin().item()),
            "local_scale_max": float(encoded.scales.float().amax().item()),
        })
        return encoded

    activation_module.pack_activation = measured_pack
    stream_module.pack_activation = measured_pack
    original_boundary_pack = boundary_module.pack_activation
    boundary_module.pack_activation = measured_pack
    try:
        with torch.inference_mode():
            result = core(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
    finally:
        activation_module.pack_activation = original
        stream_module.pack_activation = original
        boundary_module.pack_activation = original_boundary_pack

    if not records or not bool(torch.isfinite(result.logits).all()):
        raise AssertionError("Boundary profiling did not capture a finite W4A forward pass.")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)
    categories = {}
    for category, values in grouped.items():
        errors = torch.tensor([value["relative_rmse"] for value in values])
        balanced = torch.tensor([value["balanced_relative_rmse"] for value in values])
        categories[category] = {
            "count": len(values),
            "mean_relative_rmse": float(errors.mean()),
            "mean_balanced_relative_rmse": float(balanced.mean()),
            "balanced_error_ratio": float(balanced.mean() / errors.mean().clamp_min(1e-30)),
            "max_relative_rmse": float(errors.max()),
            "worst_boundary_index": values[int(errors.argmax())]["index"],
            "mean_radial_error_ratio": sum(value["radial_error_ratio"] for value in values) / len(values),
            "mean_reconstructed_energy_ratio": sum(value["reconstructed_energy_ratio"] for value in values) / len(values),
        }

    report = {
        "checkpoint": str(checkpoint.resolve()),
        "prompt_tokens": ids.shape[1],
        "next_token": int(result.logits[:, -1].argmax()),
        "boundary_count": len(records),
        "categories": categories,
        "boundaries": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = profile(args.checkpoint, args.prompt_result, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "boundaries"}, indent=2))


if __name__ == "__main__":
    main()
