# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Norm-only teacher reconstruction for a true end-to-end NVFP4 stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from .w4a_gb10_memory import require_w4a_test_headroom
from .w4a_calibration_data import load_calibration_artifact


_ROTATION_FIELDS = (
    "online_full_had", "online_partial_had", "had_dim", "had_K", "K",
)


def _dequantize_for_replay(core, *, device, dtype):
    """Move/cast replaced Linears without rounding existing model buffers.

    In particular, Llama's RoPE frequencies must retain their loaded FP32
    values. A whole-model BF16 cast after dequantization changes attention's
    positional phases even when every INT4 weight is unchanged.
    """
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear, dequantize_model

    names = [name for name, module in core.named_modules() if isinstance(module, TorchLinear)]
    core = dequantize_model(core)
    for name in names:
        core.get_submodule(name).to(device=device, dtype=dtype)
    return core


def _activation_replay_config(source_checkpoint: Path, *, train_norms: bool = False):
    """Use the exported checkpoint's actual policy in all distillation tools."""
    from gptqmodel.quantization.activation_floatx import normalize_nvfp4_recipe

    config = json.loads((source_checkpoint / "quantize_config.json").read_text())
    activation = config.get("activation")
    if (not isinstance(activation, dict) or activation.get("mode") != "w4a_nvfp4"
            or activation.get("version") not in {2, 3, 4}):
        raise ValueError("QAD source must declare a supported NVFP4 activation policy.")
    version = activation["version"]
    recipe = normalize_nvfp4_recipe(activation.get("recipe", "four_six" if version == 2 else "least_squares"))
    if recipe not in {"nvidia", "four_six", "least_squares", "least_squares_grid"}:
        raise ValueError("QAD currently requires a dynamic NVFP4 recipe; frozen headroom replay is unsupported.")
    if version == 4:
        if config.get("rotation") not in {"hadamard", "random"}:
            raise ValueError("Version 4 QAD requires a rotated source with fused norms.")
        if train_norms:
            raise ValueError("Version 4 requires unit fused norms; norm-only QAT cannot preserve this contract.")
    return SimpleNamespace(
        activation_mode="w4a_nvfp4", activation_recipe=recipe,
        activation_version=version, dynamic_get=lambda **_kwargs: None,
        activation_global_scales=activation.get("global_scales"),
    )


def _tensor_digest(tensors: dict[str, torch.Tensor], suffixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in sorted(name for name in tensors if name.endswith(suffixes)):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        # Feed hashlib through NumPy's native contiguous conversion. Iterating
        # an UntypedStorage through Python takes minutes for a 1B checkpoint.
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _calibration_ids(
    tokenizer, calibration: Path, rows: int, sequence_length: int, *, partition: str = "fit",
    text_format: str = "plain",
) -> list[torch.Tensor]:
    partitions, _ = load_calibration_artifact(calibration)
    return _calibration_ids_from_verified_records(tokenizer, partitions, rows, sequence_length,
                                                 partition=partition, text_format=text_format)


def _calibration_ids_from_verified_records(
    tokenizer, partitions, rows: int, sequence_length: int, *, partition: str = "fit",
    text_format: str = "plain",
) -> list[torch.Tensor]:
    """Tokenize the already audited snapshot returned by the artifact loader."""
    if partition not in partitions or rows <= 0 or sequence_length <= 1:
        raise ValueError("Invalid calibration partition, row count, or sequence length")
    if text_format not in {"plain", "chat_worked_examples"}:
        raise ValueError("Unsupported calibration text format")
    if rows > len(partitions[partition]):
        raise ValueError(f"Requested {rows} {partition} rows but artifact has {len(partitions[partition])}")
    samples = []
    for record in partitions[partition][:rows]:
        text = record["text"]
        if text_format == "chat_worked_examples":
            if not record.get("problems"):
                raise ValueError("Chat calibration requires audited worked examples")
            messages = []
            for problem in record["problems"]:
                messages.extend((
                    {"role": "user", "content": problem["question"]},
                    {"role": "assistant", "content": f'{problem["reasoning"]} The result is {problem["answer"]}.'},
                ))
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            if not isinstance(text, str) or not text:
                raise ValueError("Calibration chat template returned empty or non-text output")
        ids = tokenizer(text, add_special_tokens=text_format == "plain",
                        truncation=True, max_length=sequence_length)["input_ids"]
        if not ids:
            raise ValueError(f"Empty tokenized calibration article: {record['article_id']}")
        samples.append(torch.tensor(ids[:sequence_length], dtype=torch.long))
    return samples


def _hidden_reconstruction_loss(
    student: tuple[torch.Tensor, ...], teacher: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if len(student) != len(teacher) or len(student) < 2:
        raise ValueError("Student and teacher must expose the same decoder hidden states.")
    losses = []
    # Skip the identical embedding output. The remaining entries cover every
    # decoder layer and the model's final norm in Transformers' LlamaModel.
    for current, target_cpu in zip(student[1:], teacher[1:], strict=True):
        target = target_cpu.to(device=current.device, dtype=torch.float32)
        current = current.float()
        denominator = target.square().mean().clamp_min(1e-6)
        losses.append((current - target).square().mean() / denominator)
    return torch.stack(losses).mean()


def _copy_checkpoint_shell(source: Path, output: Path, *, calibration: Path | None = None) -> None:
    partitions, manifest = load_calibration_artifact(calibration) if calibration is not None else ({}, None)
    if output.exists():
        raise FileExistsError(f"Norm QAT output already exists: {output}")
    output.mkdir(parents=True)
    for item in source.iterdir():
        # Reports belong to the run that wrote them. A derived run must not
        # inherit a report symlink and then overwrite its source's evidence.
        if item.name.startswith("w4a_") and item.name.endswith("_report.json"):
            continue
        if not item.is_file() or item.name in {
                "model.safetensors", "w4a_calibration_manifest.json"}:
            continue
        (output / item.name).symlink_to(item.resolve())
    if manifest is not None:
        provenance = {**manifest, "artifact": str(calibration.resolve()),
                      "selected_samples": [{key: value for key, value in row.items() if key != "text"}
                                           for records in partitions.values() for row in records]}
        (output / "w4a_calibration_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


def train_norms(
    checkpoint: Path,
    source_checkpoint: Path,
    calibration: Path,
    output: Path,
    *,
    rows: int,
    sequence_length: int,
    steps: int,
    learning_rate: float,
    seed: int,
    objective: str,
    validation_rows: int,
) -> dict:
    require_w4a_test_headroom(require_scope=True)
    qcfg = _activation_replay_config(source_checkpoint, train_norms=True)
    if (steps <= 0 or rows <= 0 or sequence_length <= 1 or learning_rate <= 0
            or validation_rows < 0):
        raise ValueError("rows, steps, sequence_length, and learning_rate must be positive.")
    if objective not in {"teacher_hidden", "causal_lm"}:
        raise ValueError(f"Unsupported norm QAT objective: {objective}.")

    from transformers import AutoTokenizer
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear import w4a_llama_replay as replay
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    torch.manual_seed(seed)
    random.seed(seed)
    wrapper = GPTQModel.load(
        str(checkpoint), backend=BACKEND.GPTQ_TORCH,
        device="cuda:0", dtype=torch.bfloat16,
    )
    core = wrapper.model
    rotation = {
        name: {field: getattr(module, field, None) for field in _ROTATION_FIELDS}
        for name, module in core.named_modules() if isinstance(module, TorchLinear)
    }
    core = _dequantize_for_replay(core, device="cuda:0", dtype=torch.bfloat16)
    modules = dict(core.named_modules())
    for name, fields in rotation.items():
        module = modules[name]
        for field, value in fields.items():
            setattr(module, field, value)
    core.config.use_cache = False

    for parameter in core.parameters():
        parameter.requires_grad_(False)
    norm_parameters = []
    initial_norms = {}

    def qat_norm_forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        value = hidden_states.float()
        variance = value.square().mean(dim=-1, keepdim=True)
        value = value * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * value).to(input_dtype)

    for name, module in core.named_modules():
        if isinstance(module, LlamaRMSNorm):
            module.weight.data = module.weight.data.float()
            module.weight.requires_grad_(True)
            module.forward = MethodType(qat_norm_forward, module)
            norm_parameters.append(module.weight)
            initial_norms[f"{name}.weight"] = module.weight.detach().cpu().clone()
    if len(norm_parameters) != 33:
        raise AssertionError(f"Expected 33 trainable Llama RMSNorm weights, found {len(norm_parameters)}.")

    original_round = replay._round

    def straight_through_round(x, mode, recipe=None, global_scale=None):
        quantized = replay.round_w4a_activation(x, mode, recipe, global_scale)
        return x + (quantized - x).detach()

    replay._round = straight_through_round
    replay.install_w4a_llama_replay(core, qcfg)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    samples = _calibration_ids(tokenizer, calibration, rows, sequence_length)
    validation = _calibration_ids(
        tokenizer, calibration, validation_rows, sequence_length, partition="selection",
    ) if validation_rows else []
    order = list(range(len(samples)))
    optimizer = torch.optim.AdamW(norm_parameters, lr=learning_rate, weight_decay=0.0)
    losses = []

    def model_hidden(ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        result = core.model(
            input_ids=ids.unsqueeze(0).to("cuda:0"), use_cache=False,
            output_hidden_states=True, return_dict=True,
        )
        return tuple(result.hidden_states)

    teacher_targets: list[tuple[torch.Tensor, ...]] = []
    validation_targets: list[tuple[torch.Tensor, ...]] = []
    validation_before = None

    if objective == "teacher_hidden":
        def identity_round(x, _mode, _recipe=None, _global_scale=None):
            return x

        replay._round = identity_round
        core.eval()
        with torch.inference_mode():
            for ids in (*samples, *validation):
                target = tuple(value.detach().cpu() for value in model_hidden(ids))
                if len(teacher_targets) < len(samples):
                    teacher_targets.append(target)
                else:
                    validation_targets.append(target)
        replay._round = straight_through_round

        if validation:
            core.eval()
            with torch.no_grad():
                initial = [
                    float(_hidden_reconstruction_loss(model_hidden(ids), target))
                    for ids, target in zip(validation, validation_targets, strict=True)
                ]
            validation_before = sum(initial) / len(initial)

    core.train()
    try:
        for step in range(steps):
            if step % len(order) == 0:
                random.shuffle(order)
            ids = samples[order[step % len(order)]].unsqueeze(0).to("cuda:0")
            optimizer.zero_grad(set_to_none=True)
            if objective == "teacher_hidden":
                sample_index = order[step % len(order)]
                hidden = core.model(
                    input_ids=ids, use_cache=False, output_hidden_states=True,
                    return_dict=True,
                ).hidden_states
                loss = _hidden_reconstruction_loss(hidden, teacher_targets[sample_index])
            else:
                result = core(input_ids=ids, labels=ids, use_cache=False)
                loss = result.loss.float()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Norm QAT produced nonfinite loss at step {step}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(norm_parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            print(json.dumps({"step": step + 1, "steps": steps, "loss": losses[-1]}), flush=True)
    finally:
        replay._round = original_round

    validation_after = None
    if objective == "teacher_hidden" and validation:
        replay._round = straight_through_round
        core.eval()
        try:
            with torch.no_grad():
                final = [
                    float(_hidden_reconstruction_loss(model_hidden(ids), target))
                    for ids, target in zip(validation, validation_targets, strict=True)
                ]
            validation_after = sum(final) / len(final)
        finally:
            replay._round = original_round

    learned_norms = {
        f"{name}.weight": module.weight.detach().cpu()
        for name, module in core.named_modules() if isinstance(module, LlamaRMSNorm)
    }
    max_norm_delta = max(
        float((learned_norms[name] - initial_norms[name]).abs().max())
        for name in learned_norms
    )
    mean_norm_delta = sum(
        float((learned_norms[name] - initial_norms[name]).abs().mean())
        for name in learned_norms
    ) / len(learned_norms)

    _copy_checkpoint_shell(source_checkpoint, output, calibration=calibration)
    source_tensors = load_file(str(source_checkpoint / "model.safetensors"), device="cpu")
    packed_digest = _tensor_digest(source_tensors, ("qweight", "qzeros", "scales", "g_idx"))
    for name, value in learned_norms.items():
        if name not in source_tensors:
            raise KeyError(f"Norm tensor missing from packed checkpoint: {name}")
        source_tensors[name] = value.to(source_tensors[name].dtype).contiguous()
    save_file(source_tensors, str(output / "model.safetensors"))
    output_tensors = load_file(str(output / "model.safetensors"), device="cpu")
    output_packed_digest = _tensor_digest(output_tensors, ("qweight", "qzeros", "scales", "g_idx"))
    if output_packed_digest != packed_digest:
        raise AssertionError("Norm QAT changed a packed GPTQ tensor.")

    report = {
        "checkpoint": str(checkpoint.resolve()),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "output": str(output.resolve()),
        "rows": rows,
        "sequence_length": sequence_length,
        "steps": steps,
        "learning_rate": learning_rate,
        "seed": seed,
        "objective": objective,
        "validation_rows": len(validation),
        "validation_loss_before": validation_before,
        "validation_loss_after": validation_after,
        "trainable_parameters": sum(parameter.numel() for parameter in norm_parameters),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "max_norm_delta": max_norm_delta,
        "mean_norm_delta": mean_norm_delta,
        "packed_gptq_digest": packed_digest,
    }
    (output / "w4a_norm_qat_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Matched W4A16 view used to reconstruct the GPTQ weights.")
    parser.add_argument("--source-checkpoint", type=Path, required=True,
                        help="NVFP4 checkpoint whose packed tensors and metadata are retained.")
    parser.add_argument("--calibration", type=Path, required=True,
                        help="Verified directory prepared by tests.models.w4a_calibration_data.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objective", choices=("teacher_hidden", "causal_lm"),
                        default="teacher_hidden")
    parser.add_argument("--validation-rows", type=int, default=4)
    args = parser.parse_args()
    train_norms(
        args.checkpoint, args.source_checkpoint, args.calibration, args.output,
        rows=args.rows, sequence_length=args.sequence_length, steps=args.steps,
        learning_rate=args.learning_rate, seed=args.seed, objective=args.objective,
        validation_rows=args.validation_rows,
    )


if __name__ == "__main__":
    main()
