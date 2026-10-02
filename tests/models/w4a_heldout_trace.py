# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Disjoint-corpus diagnostics for same-weight W4A16 and encoded NVFP4."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .w4a_calibration_data import file_digest, load_calibration_artifact
from .w4a_gb10_memory import require_w4a_test_headroom


def probe_positions(length: int, count: int) -> torch.Tensor:
    """Fixed uniform positions with known next tokens; never choose by error."""
    if length < 2 or count < 1:
        raise ValueError("Held-out traces need at least two tokens and one probe")
    return torch.linspace(0, length - 2, min(length - 1, count), dtype=torch.float64).round().long()


def trace_heldout(checkpoint: Path, variant: str, calibration: Path, output: Path, *,
                  rows: int = 16, sequence_length: int = 2048, probes: int = 64,
                  token_energy: str = "none", token_global: bool = False,
                  training_replay: bool = False, hardware_forward: bool = False) -> dict:
    require_w4a_test_headroom(require_scope=True)
    if variant not in {"w4a16", "w4a_nvfp4"}:
        raise ValueError("Held-out diagnostics compare native W4A16 with NVFP4")
    if token_energy not in {"none", "all", "residual"} or (token_energy != "none" and variant != "w4a_nvfp4"):
        raise ValueError("Token-energy experiments require an NVFP4 selection trace")
    if token_global and (variant != "w4a_nvfp4" or token_energy != "none"):
        raise ValueError("Token-global experiments require a separate NVFP4 selection trace")
    if training_replay and (variant != "w4a_nvfp4" or token_energy != "none" or token_global):
        raise ValueError("Training replay requires an unmodified NVFP4 policy")
    if hardware_forward and not training_replay:
        raise ValueError("Hardware forward requires the training-replay diagnostic")
    if output.exists():
        raise FileExistsError(output)
    partitions, data = load_calibration_artifact(calibration)
    from transformers import AutoTokenizer
    import transformers
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.w4a_activation import W4AActivation
    from .w4a_nvfp4_norm_qat import _calibration_ids

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    samples = _calibration_ids(tokenizer, calibration, rows, sequence_length, partition="selection")
    positions = [probe_positions(len(ids), probes) for ids in samples]
    backend = BACKEND.GPTQ_TRITON if variant == "w4a16" else BACKEND.GPTQ_W4A_NVFP4
    contexts = ExitStack()
    tensors, handles, records = {}, [], []
    current_positions = None
    energy_stats = {}
    global_stats = {}

    def capture(index):
        def hook(_module, _args, result):
            value = result[0] if isinstance(result, tuple) else result
            if isinstance(value, W4AActivation):
                value = value.decode(torch.float32)
            tensors[f"layer_{index}"] = value[0, current_positions, :].float().cpu().contiguous()
        return hook

    try:
        if training_replay:
            from .w4a_weight_replay_audit import weight_replay_model

            core = contexts.enter_context(weight_replay_model(checkpoint, hardware_forward=hardware_forward))
        else:
            core = GPTQModel.load(str(checkpoint), backend=backend, device="cuda:0", dtype=torch.bfloat16).model.eval()
        output.mkdir(parents=True, exist_ok=False)
        if token_global:
            from .w4a_token_global import install_token_global_diagnostic

            global_handles, global_stats = install_token_global_diagnostic(core)
            handles.extend(global_handles)
        if token_energy != "none":
            from .w4a_token_energy import install_token_energy_diagnostic

            energy_handles, energy_stats = install_token_energy_diagnostic(core, token_energy)
            handles.extend(energy_handles)
        for index, layer in enumerate(core.model.layers):
            handles.append(layer.register_forward_hook(capture(index)))
        with torch.inference_mode():
            for index, (ids, probes_cpu) in enumerate(zip(samples, positions, strict=True)):
                current_positions = probes_cpu.cuda()
                tensors = {"input_ids": ids.cpu(), "probe_positions": probes_cpu,
                           "next_tokens": ids[probes_cpu + 1].cpu()}
                # Keep logits only for predeclared positions, bounding both GPU
                # head output and on-disk teacher data for the large vocabulary.
                device_ids = ids[None].cuda()
                hardware = getattr(core, "_w4a_training_hardware", None)
                frame = hardware.frame(device_ids, logits_to_keep=current_positions) if hardware is not None else nullcontext()
                with frame:
                    result = core(input_ids=device_ids, use_cache=False, logits_to_keep=current_positions)
                    if hardware is not None:
                        torch.testing.assert_close(result.logits, hardware.logits, rtol=0, atol=0)
                tensors["logits"] = result.logits[0].float().cpu().contiguous()
                if len(tensors) != len(core.model.layers) + 4 or any(
                        not bool(torch.isfinite(value).all()) for value in tensors.values()):
                    raise AssertionError("Incomplete or nonfinite held-out trace")
                name = f"sample_{index:04d}.safetensors"
                save_file(tensors, str(output / name))
                records.append({"article_id": partitions["selection"][index]["article_id"],
                                "file": name, "sha256": file_digest(output / name),
                                "tokens": len(ids), "probes": len(probes_cpu)})
                print(json.dumps({"variant": variant, "completed": index + 1, "rows": rows}), flush=True)
    finally:
        for handle in handles:
            handle.remove()
        contexts.close()
    if file_digest(calibration / "manifest.json") != data["manifest_sha256"]:
        raise ValueError("The calibration dataset changed during tracing")
    report = {"version": 1, "variant": variant, "checkpoint": str(checkpoint.resolve()),
              "weight_file_sha256": file_digest(checkpoint / "model.safetensors"),
              "quantize_config_sha256": file_digest(checkpoint / "quantize_config.json"),
              "runtime_versions": {"torch": str(torch.__version__), "transformers": transformers.__version__},
              "data_manifest_sha256": data["manifest_sha256"], "partition": "selection",
              "sequence_length": sequence_length, "probe_limit": probes,
              "probe_policy": "uniform_including_first_excluding_last_v1",
              "experimental_token_energy": token_energy, "token_energy_statistics": dict(energy_stats),
              "experimental_token_global": token_global, "token_global_statistics": dict(global_stats),
              "execution": ("weight_qad_hardware_forward" if hardware_forward
                            else "weight_qad_replay" if training_replay else "runtime"),
              "decoder_layers": len(core.model.layers), "samples": records}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def hidden_statistics(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    if reference.shape != actual.shape or reference.ndim != 2 or not reference.numel():
        raise ValueError("Hidden traces must be matching nonempty token-by-channel tensors")
    reference, actual = reference.double(), actual.double()
    if not bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()):
        raise ValueError("Nonfinite hidden trace")
    delta = actual - reference
    energy = reference.square().sum(-1)
    squared_error = delta.square().sum(-1)
    valid = energy > 1e-20
    denominator = energy[valid]
    cosine_denominator = (denominator * actual[valid].square().sum(-1)).sqrt().clamp_min(1e-30)
    return {"sse": float(squared_error.sum()), "energy": float(energy.sum()),
            "tokens": len(reference), "relative_tokens": int(valid.sum()),
            "token_relative_sse": float((squared_error[valid] / denominator).sum()),
            "cosine_sum": float((actual[valid] * reference[valid]).sum(-1).div(cosine_denominator).sum()),
            "radial_sum": float((delta[valid] * reference[valid]).sum(-1).div(denominator).sum())}


def logit_statistics(reference: torch.Tensor, actual: torch.Tensor, targets: torch.Tensor) -> dict:
    if (reference.shape != actual.shape or reference.ndim != 2 or not reference.numel()
            or targets.shape != (reference.shape[0],)):
        raise ValueError("Logits and next-token targets must have matching nonempty rows")
    if not bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()):
        raise ValueError("Nonfinite logit trace")
    if targets.dtype != torch.long or bool(((targets < 0) | (targets >= reference.shape[-1])).any()):
        raise ValueError("Next-token targets must be valid vocabulary indices")
    result = {"tokens": len(reference), "kl_sum": 0., "reference_nll_sum": 0., "actual_nll_sum": 0.,
              "argmax_agreement": int((reference.argmax(-1) == actual.argmax(-1)).sum())}
    # Full-vocabulary KL at sampled positions; no top-k truncation of the tail.
    for start in range(0, len(reference), 16):
        left = reference[start:start + 16].double().log_softmax(-1)
        right = actual[start:start + 16].double().log_softmax(-1)
        target = targets[start:start + 16, None]
        result["kl_sum"] += float((left.exp() * (left - right)).sum())
        result["reference_nll_sum"] -= float(left.gather(-1, target).sum())
        result["actual_nll_sum"] -= float(right.gather(-1, target).sum())
    return result


def compare_heldout(reference: Path, actual: Path, *, training_replay: bool = False) -> dict:
    left = json.loads((reference / "manifest.json").read_text())
    right = json.loads((actual / "manifest.json").read_text())
    for key in ("version", "weight_file_sha256", "data_manifest_sha256", "partition",
                "sequence_length", "probe_limit", "probe_policy", "decoder_layers", "runtime_versions"):
        if left[key] != right[key]:
            raise ValueError(f"Held-out trace protocol mismatch: {key}")
    if left["version"] != 1 or left["probe_policy"] != "uniform_including_first_excluding_last_v1":
        raise ValueError("Unsupported held-out trace protocol")
    reference_variant = "w4a_nvfp4" if training_replay else "w4a16"
    expected_execution = {"weight_qad_replay", "weight_qad_hardware_forward"} if training_replay else {"runtime"}
    if (left["partition"] != "selection" or left["variant"] != reference_variant
            or right["variant"] != "w4a_nvfp4"
            or left.get("execution", "runtime") != "runtime"
            or right.get("execution", "runtime") not in expected_execution):
        raise ValueError("Unexpected variants or execution modes for selection comparison")
    if training_replay:
        if left["quantize_config_sha256"] != right["quantize_config_sha256"]:
            raise ValueError("Replay audit requires identical activation policies")
        if any(row.get("experimental_token_global", False)
               or row.get("experimental_token_energy", "none") != "none" for row in (left, right)):
            raise ValueError("Replay audit requires unmodified runtime traces")
    if not left["samples"] or len(left["samples"]) != len(right["samples"]):
        raise ValueError("Held-out trace sample coverage differs")
    if len({sample["article_id"] for sample in left["samples"]}) != len(left["samples"]):
        raise ValueError("Duplicate held-out article IDs")
    layers, logits = defaultdict(lambda: defaultdict(float)), defaultdict(float)
    fidelity = {"rtol": 2e-3, "atol": 2e-3, "tensors": {}, "argmax_mismatches": 0}
    for a, b in zip(left["samples"], right["samples"], strict=True):
        for key in ("article_id", "tokens", "probes"):
            if a[key] != b[key]:
                raise ValueError(f"Held-out sample mismatch: {key}")
        paths = reference / a["file"], actual / b["file"]
        if file_digest(paths[0]) != a["sha256"] or file_digest(paths[1]) != b["sha256"]:
            raise ValueError("Held-out trace hash mismatch")
        base, quant = (load_file(str(path)) for path in paths)
        expected = {"input_ids", "probe_positions", "next_tokens", "logits"} | {
            f"layer_{i}" for i in range(left["decoder_layers"])}
        if set(base) != expected or set(quant) != expected:
            raise ValueError("Incomplete held-out layer or logit tensors")
        for key in ("input_ids", "probe_positions", "next_tokens"):
            if not torch.equal(base[key], quant[key]):
                raise ValueError(f"Held-out inputs differ: {key}")
        ids, positions = base["input_ids"], base["probe_positions"]
        if (ids.dtype != torch.long or ids.ndim != 1 or len(ids) != a["tokens"]
                or positions.dtype != torch.long or len(positions) != a["probes"]
                or not torch.equal(positions, probe_positions(len(ids), left["probe_limit"]))
                or not torch.equal(base["next_tokens"], ids[positions + 1])):
            raise ValueError("Held-out probes or targets do not follow the declared protocol")
        if any(value.ndim != 2 or len(value) != len(positions)
               for tensors in (base, quant) for key, value in tensors.items()
               if key == "logits" or key.startswith("layer_")):
            raise ValueError("Held-out tensor rows do not match declared probes")
        if training_replay:
            for key in sorted(expected - {"input_ids", "probe_positions", "next_tokens"}):
                reference_values, replay_values = base[key].double(), quant[key].double()
                if reference_values.shape != replay_values.shape or not bool(
                        torch.isfinite(reference_values).all() and torch.isfinite(replay_values).all()):
                    raise ValueError("Replay audit requires matching finite tensors")
                delta = (replay_values - reference_values).abs()
                row = fidelity["tensors"].setdefault(key, {"elements": 0, "outside_tolerance": 0,
                                                        "max_abs_error": 0.})
                row["elements"] += reference_values.numel()
                row["outside_tolerance"] += int((delta > fidelity["atol"] + fidelity["rtol"] * reference_values.abs()).sum())
                row["max_abs_error"] = max(row["max_abs_error"], float(delta.max()))
            fidelity["argmax_mismatches"] += int((base["logits"].argmax(-1) != quant["logits"].argmax(-1)).sum())
        for index in range(left["decoder_layers"]):
            key = f"layer_{index}"
            for metric, value in hidden_statistics(base[key], quant[key]).items():
                layers[key][metric] += value
        for metric, value in logit_statistics(base["logits"], quant["logits"], base["next_tokens"]).items():
            logits[metric] += value
    for row in layers.values():
        row["energy_weighted_relative_rmse"] = (row["sse"] / max(row["energy"], 1e-30)) ** .5
        row["equal_token_relative_rmse"] = (row["token_relative_sse"] / max(row["relative_tokens"], 1)) ** .5
        row["mean_cosine"] = row["cosine_sum"] / max(row["relative_tokens"], 1)
        row["mean_radial_error"] = row["radial_sum"] / max(row["relative_tokens"], 1)
    logits.update({"mean_kl": logits["kl_sum"] / logits["tokens"],
                   "sampled_reference_nll": logits["reference_nll_sum"] / logits["tokens"],
                   "sampled_actual_nll": logits["actual_nll_sum"] / logits["tokens"],
                   "argmax_agreement_fraction": logits["argmax_agreement"] / logits["tokens"]})
    fidelity["passed"] = (fidelity["argmax_mismatches"] == 0 and
                           all(row["outside_tolerance"] == 0 for row in fidelity["tensors"].values()))
    return {"reference": str(reference), "actual": str(actual), "rows": len(left["samples"]),
            "actual_execution": right.get("execution", "runtime"),
            "replay_validation": fidelity if training_replay else None,
            "experimental_token_energy": right.get("experimental_token_energy", "none"),
            "experimental_token_global": right.get("experimental_token_global", False),
            "data_manifest_sha256": left["data_manifest_sha256"], "layers": dict(layers), "logits": dict(logits),
            "scope": "uniform sampled token positions on disjoint held-out articles; not an e2e accuracy gate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-replay", action="store_true",
                        help="Compare encoded runtime against weight replay at rtol=atol=2e-3.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compare_heldout(args.reference, args.actual, training_replay=args.training_replay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": report["rows"], "logits": report["logits"]}))
    if args.training_replay and not report["replay_validation"]["passed"]:
        raise SystemExit("Weight replay differs from encoded runtime; see saved fidelity report")


if __name__ == "__main__":
    main()
