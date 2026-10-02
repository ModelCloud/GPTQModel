# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Experimental same-weight layer reconstruction for NVFP4 producer scales."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch


def relative_token_error(actual: torch.Tensor, target: torch.Tensor) -> tuple[float, int]:
    """FP64 sum of relative squared errors, giving each nonzero token equal weight."""
    if actual.shape != target.shape or actual.ndim < 2 or not actual.numel():
        raise ValueError("Reconstruction needs matching nonempty token tensors")
    actual, target = actual.double(), target.double()
    if not bool(torch.isfinite(actual).all() and torch.isfinite(target).all()):
        raise ValueError("Nonfinite reconstruction target or output")
    energy = target.square().sum(-1)
    error = (actual - target).square().sum(-1)
    if bool((energy == 0).any()):
        raise ValueError("Relative reconstruction requires nonzero teacher tokens")
    return float((error / energy).sum()), energy.numel()


def search_scale(producer, objective, ratios: tuple[float, ...]) -> dict:
    """Keep the current scale on ties and restore it if any candidate fails."""
    if (not ratios or any(isinstance(r, bool) or not math.isfinite(r) or r <= 0 for r in ratios)):
        raise ValueError("Scale ratios must be finite positive numbers")
    initial = float(producer.global_scale)
    candidates = list(dict.fromkeys([initial] + [float(torch.tensor(initial * r, dtype=torch.float32))
                                                for r in ratios]))
    best_scale, best_loss, trials = initial, float("inf"), []
    try:
        for value in candidates:
            producer.set_scale(value)
            loss = float(objective())
            if not math.isfinite(loss) or loss < 0:
                raise ValueError("Scale search objective must be finite and nonnegative")
            trials.append({"scale": value, "loss": loss})
            if loss < best_loss:
                best_scale, best_loss = value, loss
        producer.set_scale(best_scale)
    except BaseException:
        producer.set_scale(initial)
        raise
    return {"initial_scale": initial, "selected_scale": best_scale,
            "initial_loss": trials[0]["loss"], "selected_loss": best_loss, "trials": trials}


@torch.inference_mode()
def reconstruct(core, teacher, samples, *, ratios=(.75, .875, 1., 1.125, 1.25, 1.5), progress=None) -> dict:
    from gptqmodel.nn_modules.qlinear.w4a_activation import W4AActivation
    from gptqmodel.nn_modules.qlinear.w4a_boundary import NVFP4BoundaryQuantizer, llama_nvfp4_boundaries
    from gptqmodel.quantization.activation_calibration import _capture_inputs, _move

    if (core.training or teacher.training or not samples
            or getattr(core, "_w4a_stream_version", None) != 4
            or getattr(core, "_w4a_stream_mode", None) != "w4a_nvfp4"
            or getattr(teacher, "_w4a_stream_mode", None) is not None):
        raise ValueError("Expected eval-mode version-4 A4 student, A16 teacher, and fitting samples")
    layers, teacher_layers = core.model.layers, teacher.model.layers
    if len(layers) != len(teacher_layers) or any(getattr(layer, "_w4a_stream_version", None) != 4 for layer in layers):
        raise ValueError("Reconstruction requires complete matching decoder coverage")
    specs = list(llama_nvfp4_boundaries(layers, [True] * len(layers)))
    producers = [(key, getattr(owner, f"_w4a_{name}_quantizer")) for owner, name, key in specs]
    if any(not isinstance(module, NVFP4BoundaryQuantizer) or not module.calibrated or module.observer is not None
           or module._forward_hooks for _, module in producers):
        raise ValueError("Reconstruction requires calibrated idle producers without experimental hooks")
    initial = [(module, module.scale_bits.clone()) for _, module in producers]
    old_policy = core._w4a_stream_global_scales
    records, layer_records = {}, []
    try:
        cached = _capture_inputs(core, samples)
        teacher_cached = _capture_inputs(teacher, samples)
        if len(cached) != len(teacher_cached) or any(
                not torch.equal(a.hidden, b.hidden) for a, b in zip(cached, teacher_cached, strict=True)):
            raise ValueError("Student and teacher embedding inputs differ")
        for index, (layer, teacher_layer) in enumerate(zip(layers, teacher_layers, strict=True)):
            device = layer.self_attn.q_proj.qweight.device
            teacher_device = next(teacher_layer.parameters()).device
            for sample in teacher_cached:
                target = teacher_layer(_move(sample.hidden, teacher_device), **_move(sample.kwargs, teacher_device))
                if not isinstance(target, torch.Tensor):
                    raise TypeError("A16 teacher must return dense decoder outputs")
                sample.hidden = _move(target, "cpu")

            def objective():
                total, count = 0., 0
                for sample, target in zip(cached, teacher_cached, strict=True):
                    actual = layer(_move(sample.hidden, device), **_move(sample.kwargs, device))
                    if not isinstance(actual, W4AActivation):
                        raise TypeError("Scale search must execute encoded student boundaries")
                    error, tokens = relative_token_error(actual.decode(torch.float32), _move(target.hidden, device))
                    total += error
                    count += tokens
                return total / count

            before = objective()
            for key, producer in producers:
                if not key.startswith(f"model.layers.{index}."):
                    continue
                row = search_scale(producer, objective, ratios)
                records[key] = row
                if progress is not None:
                    progress(key, row)
            after = objective()
            if after > before + 1e-12:
                raise AssertionError("Coordinate search increased the layer fitting objective")
            layer_records.append({"layer": index, "initial_loss": before, "selected_loss": after})
            for sample in cached:
                result = layer(_move(sample.hidden, device), **_move(sample.kwargs, device))
                if not isinstance(result, W4AActivation):
                    raise TypeError("Fitted layer must propagate an encoded carrier")
                sample.hidden = _move(result, "cpu")
        scales = {key: float(producer.global_scale) for key, producer in producers}
        core._w4a_stream_global_scales = scales
    except BaseException:
        for module, bits in initial:
            module.scale_bits.copy_(bits)
        core._w4a_stream_global_scales = old_policy
        raise
    return {"algorithm": "producer_layer_reconstruction", "recipe": core._w4a_stream_recipe,
            "objective": "mean_per_token_relative_squared_decoder_output_error",
            "candidate_ratios": list(ratios), "global_scales": scales, "boundaries": records,
            "layers": layer_records, "upstream_activations": "fitted_encoded_student_carriers",
            "weight_updates": 0}


def run(checkpoint: Path, reference: Path, calibration: Path, output: Path, *, rows=16, sequence_length=512) -> dict:
    from .w4a_calibration_data import file_digest, load_calibration_artifact
    from .w4a_gb10_memory import require_w4a_test_headroom
    from .w4a_nvfp4_calibrate import export_view
    from .w4a_nvfp4_norm_qat import _calibration_ids, _tensor_digest

    require_w4a_test_headroom(require_scope=True)
    if output.exists():
        raise FileExistsError(output)
    _, manifest = load_calibration_artifact(calibration)
    if not (checkpoint / "model.safetensors").samefile(reference / "model.safetensors"):
        raise ValueError("Reconstruction requires the identical native weight file")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (checkpoint / name).read_bytes() != (reference / name).read_bytes():
            raise ValueError(f"Teacher tokenizer differs: {name}")
    from transformers import AutoTokenizer
    from gptqmodel import BACKEND, GPTQModel

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    fit = _calibration_ids(tokenizer, calibration, rows, sequence_length, partition="fit")
    # The selection partition is used here only for a save/reload probe.
    probe = _calibration_ids(tokenizer, calibration, 1, 128, partition="selection")[0][None].cuda()
    student_wrapper = GPTQModel.load(str(checkpoint), backend=BACKEND.GPTQ_W4A_NVFP4,
                                     device="cuda:0", dtype=torch.bfloat16)
    teacher_wrapper = GPTQModel.load(str(reference), backend=BACKEND.GPTQ_TRITON,
                                     device="cuda:0", dtype=torch.bfloat16)
    core, teacher = student_wrapper.model.eval(), teacher_wrapper.model.eval()
    suffixes = ("qweight", "qzeros", "scales", "g_idx")
    native = _tensor_digest(core.state_dict(), suffixes)
    weight_hash = file_digest(checkpoint / "model.safetensors")
    report = reconstruct(core, teacher, fit, progress=lambda key, row:
                         print(json.dumps({"producer": key, **row}), flush=True))
    if _tensor_digest(core.state_dict(), suffixes) != native:
        raise AssertionError("Reconstruction changed a native GPTQ tensor")
    with torch.inference_mode():
        before = core(input_ids=probe, use_cache=False).logits.cpu()
    export_view(checkpoint, output, calibration, report["global_scales"], report["recipe"])
    del core, teacher, student_wrapper, teacher_wrapper
    gc.collect()
    torch.cuda.empty_cache()
    restored = GPTQModel.load(str(output), backend=BACKEND.GPTQ_W4A_NVFP4,
                             device="cuda:0", dtype=torch.bfloat16).model.eval()
    with torch.inference_mode():
        after = restored(input_ids=probe, use_cache=False).logits.cpu()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    if _tensor_digest(restored.state_dict(), suffixes) != native or file_digest(output / "model.safetensors") != weight_hash:
        raise AssertionError("Export changed native GPTQ weights")
    if file_digest(calibration / "manifest.json") != manifest["manifest_sha256"]:
        raise AssertionError("Calibration artifact changed during fitting")
    report.update({"checkpoint": str(checkpoint.resolve()), "reference": str(reference.resolve()),
                   "data_manifest_sha256": manifest["manifest_sha256"], "partition": "fit",
                   "fit_rows": rows, "fit_tokens": sum(len(ids) for ids in fit),
                   "sequence_length": sequence_length, "native_tensor_digest": native,
                   "weight_file_sha256": weight_hash, "reload_logits_exact": True,
                   "reload_native_tensors_exact": True})
    path = output / "w4a_producer_reconstruction_report.json"
    with path.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), "reload_logits_exact": True, "boundaries": len(report["global_scales"])}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "reference", "calibration", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=512)
    args = parser.parse_args()
    run(args.checkpoint, args.reference, args.calibration, args.output,
        rows=args.rows, sequence_length=args.sequence_length)


if __name__ == "__main__":
    main()
