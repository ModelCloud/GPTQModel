# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Fit producer activation scales on disjoint data, preserving native INT4 files."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from .w4a_calibration_data import (
    evaluated_dataset_configs, file_digest, load_calibration_artifact,
)
from .w4a_gb10_memory import require_w4a_test_headroom
from .w4a_nvfp4_norm_qat import _calibration_ids, _copy_checkpoint_shell, _tensor_digest


def export_view(source: Path, output: Path, calibration: Path, scales: dict, recipe: str) -> None:
    from gptqmodel.quantization.config import QuantizeConfig

    qconfig = json.loads((source / "quantize_config.json").read_text())
    qconfig["activation"] = {"version": 4, "mode": "w4a_nvfp4", "recipe": recipe, "global_scales": scales}
    QuantizeConfig.from_quant_config(qconfig)
    _copy_checkpoint_shell(source, output, calibration=calibration)
    (output / "model.safetensors").symlink_to((source / "model.safetensors").resolve())
    for filename in ("quantize_config.json", "config.json"):
        target = output / filename
        if target.is_symlink():
            target.unlink()
        data = qconfig if filename == "quantize_config.json" else json.loads((source / filename).read_text())
        if filename == "config.json":
            data["quantization_config"] = qconfig
        target.write_text(json.dumps(data, indent=2) + "\n")
    if not (output / "model.safetensors").samefile(source / "model.safetensors"):
        raise AssertionError("Calibration export changed the native weight file")


def run(checkpoint: Path, calibration: Path, output: Path, *, rows: int,
        selection_rows: int, sequence_length: int) -> dict:
    require_w4a_test_headroom(require_scope=True)
    if output.exists():
        raise FileExistsError(output)
    # Validate before allocating a GPU model or observing any activation. The
    # artifact must exclude every dataset the quality harness can score, so a
    # calibration pass can never fit on a benchmark it is later evaluated on.
    required_evaluations = evaluated_dataset_configs()
    _, data_manifest = load_calibration_artifact(calibration, required_evaluations=required_evaluations)
    from transformers import AutoTokenizer
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.quantization.activation_calibration import measure_nvfp4_producers

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    fit = _calibration_ids(tokenizer, calibration, rows, sequence_length)
    selection = _calibration_ids(tokenizer, calibration, selection_rows, sequence_length, partition="selection")
    wrapper = GPTQModel.load(str(checkpoint), backend=BACKEND.GPTQ_W4A_NVFP4,
                            device="cuda:0", dtype=torch.bfloat16)
    core = wrapper.model.eval()
    suffixes = ("qweight", "qzeros", "scales", "g_idx")
    original_digest = _tensor_digest(core.state_dict(), suffixes)
    file_hash = file_digest(checkpoint / "model.safetensors")
    print(json.dumps({"stage": "calibrate", "fit_rows": len(fit),
                      "fit_tokens": sum(ids.numel() for ids in fit), "selection_rows": len(selection)}), flush=True)
    report = wrapper.calibrate_activations(fit, progress=lambda key, row:
                                          print(json.dumps({"producer": key, **row}), flush=True))
    if wrapper.quantize_config.activation_global_scales != report["global_scales"]:
        raise AssertionError("Calibration did not update the checkpoint save configuration")
    if _tensor_digest(core.state_dict(), suffixes) != original_digest:
        raise AssertionError("Activation calibration modified a native GPTQ tensor")
    report["selection_reconstruction"] = measure_nvfp4_producers(core, selection)
    report.update({"checkpoint": str(checkpoint.resolve()), "calibration": str(calibration.resolve()),
                   "data_manifest_sha256": data_manifest["manifest_sha256"],
                   "evaluation_exclusions": sorted(required_evaluations),
                   "fit_rows": len(fit), "fit_tokens": sum(ids.numel() for ids in fit),
                   "selection_rows": len(selection), "selection_tokens": sum(ids.numel() for ids in selection),
                   "sequence_length": sequence_length, "native_tensor_digest": original_digest,
                   "weight_file_sha256": file_hash})
    probe = selection[0][:128].unsqueeze(0).cuda()
    with torch.inference_mode():
        before = core(input_ids=probe, use_cache=False).logits.cpu()
    export_view(checkpoint, output, calibration, report["global_scales"], report["recipe"])
    del core, wrapper
    gc.collect()
    torch.cuda.empty_cache()
    restored = GPTQModel.load(str(output), backend=BACKEND.GPTQ_W4A_NVFP4,
                             device="cuda:0", dtype=torch.bfloat16)
    restored.model.eval()
    if restored.quantize_config.activation_global_scales != report["global_scales"]:
        raise AssertionError("Producer scales did not survive save/reload")
    with torch.inference_mode():
        after = restored.model(input_ids=probe, use_cache=False).logits.cpu()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    if _tensor_digest(restored.model.state_dict(), suffixes) != original_digest:
        raise AssertionError("Reload changed a native GPTQ tensor")
    if file_digest(output / "model.safetensors") != file_hash:
        raise AssertionError("Native checkpoint file changed during calibration")
    report["reload_logits_exact"] = True
    report["reload_native_tensors_exact"] = True
    (output / "w4a_producer_calibration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "boundaries": len(report["global_scales"]),
                      "reload_logits_exact": True, "native_tensors_unchanged": True}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--selection-rows", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=2048)
    args = parser.parse_args()
    run(args.checkpoint, args.calibration, args.output, rows=args.rows,
        selection_rows=args.selection_rows, sequence_length=args.sequence_length)


if __name__ == "__main__":
    main()
