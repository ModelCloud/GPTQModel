# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Distill true NVFP4 replay while preserving the native GPTQ W4A16 model.

The trainable state is limited to relative deltas on the existing GPTQ group
scales.  INT4 codes, zero points, group indices, and int32 packing never enter
the optimizer and are asserted byte-identical after export.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .w4a_gb10_memory import require_w4a_test_headroom
from .w4a_nvfp4_norm_qat import (
    _ROTATION_FIELDS,
    _activation_replay_config,
    _calibration_ids,
    _copy_checkpoint_shell,
    _dequantize_for_replay,
    _tensor_digest,
)


def _relative_hidden_loss(
    student: tuple[torch.Tensor, ...], teacher: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if len(student) != len(teacher) or len(student) < 2:
        raise ValueError("Student and teacher hidden-state layouts differ.")
    losses = []
    for current, target_cpu in zip(student[1:], teacher[1:], strict=True):
        target = target_cpu.to(device=current.device, dtype=torch.float32)
        current = current.float()
        losses.append(
            (current - target).square().mean()
            / target.square().mean().clamp_min(1e-6)
        )
    return torch.stack(losses).mean()


def _logit_distillation_loss(
    student: torch.Tensor, teacher_cpu: torch.Tensor, temperature: float,
) -> torch.Tensor:
    teacher = teacher_cpu.to(device=student.device, dtype=torch.float32)
    student_log = F.log_softmax(student.float() / temperature, dim=-1)
    teacher_log = F.log_softmax(teacher / temperature, dim=-1)
    # Average one categorical KL per token.  ``batchmean`` would divide only
    # by the batch dimension and make the learning rate depend on sequence
    # length.
    loss = F.kl_div(student_log, teacher_log, log_target=True, reduction="none")
    return loss.sum(dim=-1).mean() * temperature * temperature


def _student_loss(result, target: dict, *, temperature: float,
                  hidden_weight: float, logit_weight: float) -> tuple[torch.Tensor, dict]:
    hidden = _relative_hidden_loss(tuple(result.hidden_states), target["hidden_states"])
    logits = _logit_distillation_loss(result.logits, target["logits"], temperature)
    total = hidden_weight * hidden + logit_weight * logits
    return total, {
        "hidden": float(hidden.detach()),
        "logits": float(logits.detach()),
        "total": float(total.detach()),
    }


def _install_trainable_gptq_scales(
    core: torch.nn.Module,
    codes: dict[str, torch.Tensor],
    initial_scales: dict[str, torch.Tensor],
) -> tuple[list[torch.nn.Parameter], dict[str, torch.nn.Module]]:
    modules = dict(core.named_modules())
    parameters = []
    selected = {}
    for name, code in codes.items():
        module = modules[name]
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"Expected dense Linear after GPTQ dequantization: {name}")
        if module.in_features % 128:
            raise ValueError(f"GPTQ scale QAD requires group size 128: {name}")
        expected = (module.in_features // 128, module.out_features)
        if initial_scales[name].shape != expected or code.shape != (
            module.in_features, module.out_features
        ):
            raise ValueError(f"GPTQ scale or code shape mismatch for {name}.")

        # Reconstructing from the frozen centered code plane makes delta=0 the
        # exact native GPTQ function.  BF16 code storage replaces the dense
        # dequantized weight, so the training representation stays bounded.
        code_weight = code.T.contiguous().to(
            device=module.weight.device, dtype=module.weight.dtype,
        )
        scale = initial_scales[name].to(device=module.weight.device, dtype=torch.float32)
        module.register_buffer("_w4a_qad_code_weight", code_weight, persistent=False)
        module.register_buffer("_w4a_qad_initial_scale", scale, persistent=False)
        module.register_parameter(
            "_w4a_qad_log_scale_delta",
            torch.nn.Parameter(torch.zeros_like(scale)),
        )
        # The dense parameter is redundant once the exact code/scale forward
        # is installed.  Removing it also prevents accidental optimization.
        module.register_parameter("weight", None)

        def scale_forward(self, x):
            multiplier = self._w4a_qad_log_scale_delta.exp()
            current = self._w4a_qad_initial_scale * multiplier
            # Native GPTQ applies one scale to each 128-wide partial result
            # before summing K groups.  Keep that order here: folding scales
            # into a BF16 dense weight hides sub-ULP scale updates and gives
            # the optimizer a different numerical function from runtime.
            scale = current + (current.to(torch.float16).float() - current).detach()
            rows = x.numel() // self.in_features
            grouped_x = x.reshape(rows, self.in_features // 128, 128)
            grouped_codes = self._w4a_qad_code_weight.reshape(
                self.out_features, self.in_features // 128, 128
            )
            # Hardware accumulates each FP4 group in FP32. Returning a
            # BF16/FP16 partial here adds an extra rounding before GPTQ scale
            # application and teaches a different function from inference.
            partial = torch.einsum("tgi,ogi->tgo", grouped_x.float(), grouped_codes.float())
            result = (partial.float() * scale.unsqueeze(0)).sum(dim=1)
            if self.bias is not None:
                result = result + self.bias.float()
            return result.reshape(*x.shape[:-1], self.out_features).to(x.dtype)

        module.forward = MethodType(scale_forward, module)
        parameters.append(module._w4a_qad_log_scale_delta)
        selected[name] = module
    return parameters, selected


def _scale_regularizer(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    numerator = sum(value.square().sum() for value in parameters)
    denominator = sum(value.numel() for value in parameters)
    return numerator / denominator


@torch.no_grad()
def _clamp_scale_deltas(parameters: list[torch.nn.Parameter], max_scale_change: float) -> None:
    lower = math.log(1.0 - max_scale_change)
    upper = math.log(1.0 + max_scale_change)
    for value in parameters:
        value.clamp_(lower, upper)


def train_scales(
    checkpoint: Path,
    source_checkpoint: Path,
    calibration: Path,
    output: Path,
    *,
    rows: int,
    validation_rows: int,
    sequence_length: int,
    steps: int,
    learning_rate: float,
    preserve_weight: float,
    scale_regularization: float,
    max_scale_change: float,
    hidden_weight: float,
    logit_weight: float,
    temperature: float,
    seed: int,
    eval_interval: int,
    max_preserve_loss_increase: float,
) -> dict:
    require_w4a_test_headroom(require_scope=True)
    qcfg = _activation_replay_config(source_checkpoint)
    if (rows <= 0 or validation_rows <= 0 or sequence_length <= 1 or steps <= 0
            or learning_rate <= 0 or preserve_weight < 0 or scale_regularization < 0
            or not 0 < max_scale_change < 1 or hidden_weight < 0 or logit_weight < 0
            or hidden_weight + logit_weight <= 0 or temperature <= 0
            or eval_interval <= 0 or max_preserve_loss_increase < 0):
        raise ValueError("Invalid scale-QAD dimensions or optimization settings.")

    from transformers import AutoTokenizer

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear import w4a_llama_replay as replay
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.nn_modules.qlinear.w4a_floatx import W4AFP8Linear

    torch.manual_seed(seed)
    random.seed(seed)
    wrapper = GPTQModel.load(
        str(checkpoint), backend=BACKEND.GPTQ_TORCH,
        device="cuda:0", dtype=torch.bfloat16,
    )
    core = wrapper.model
    core.config.use_cache = False
    quantized = {
        name: module for name, module in core.named_modules() if isinstance(module, TorchLinear)
    }
    if len(quantized) != 112:
        raise AssertionError(f"Expected 112 GPTQ Linears, found {len(quantized)}.")
    codes = {
        name: W4AFP8Linear._centered_int4_codes(module).detach().cpu().to(torch.int8)
        for name, module in quantized.items()
    }
    initial_scales = {
        name: module.scales.detach().cpu().float().clone() for name, module in quantized.items()
    }
    rotation = {
        name: {field: getattr(module, field, None) for field in _ROTATION_FIELDS}
        for name, module in quantized.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    samples = _calibration_ids(tokenizer, calibration, rows, sequence_length)
    validation = _calibration_ids(
        tokenizer, calibration, validation_rows, sequence_length, partition="selection",
    )

    def teacher_target(ids: torch.Tensor) -> dict:
        result = core(
            input_ids=ids.unsqueeze(0).to("cuda:0"), use_cache=False,
            output_hidden_states=True, return_dict=True,
        )
        return {
            "logits": result.logits.detach().to(device="cpu", dtype=torch.bfloat16),
            "hidden_states": tuple(
                value.detach().to(device="cpu", dtype=torch.bfloat16)
                for value in result.hidden_states
            ),
        }

    core.eval()
    with torch.inference_mode():
        train_targets = [teacher_target(ids) for ids in samples]
        validation_targets = [teacher_target(ids) for ids in validation]

    core = _dequantize_for_replay(core, device="cuda:0", dtype=torch.bfloat16)
    modules = dict(core.named_modules())
    for name, fields in rotation.items():
        for field, value in fields.items():
            setattr(modules[name], field, value)
    core.config.use_cache = False
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    scale_parameters, scale_modules = _install_trainable_gptq_scales(
        core, codes, initial_scales,
    )

    replay.install_w4a_llama_replay(core, qcfg)
    original_round = replay._round

    def identity_round(x, _mode, _recipe=None, _global_scale=None):
        return x

    def straight_through_round(x, mode, recipe=None, global_scale=None):
        quantized = replay.round_w4a_activation(x, mode, recipe, global_scale)
        return x + (quantized - x).detach()

    optimizer = torch.optim.AdamW(
        scale_parameters, lr=learning_rate, weight_decay=0.0,
    )
    order = list(range(len(samples)))
    history = []
    evaluations = []

    def forward(ids: torch.Tensor):
        return core(
            input_ids=ids.unsqueeze(0).to("cuda:0"), use_cache=False,
            output_hidden_states=True, return_dict=True,
        )

    def evaluate(dataset, targets) -> dict:
        totals = {"a4": 0.0, "a16": 0.0, "a4_hidden": 0.0,
                  "a16_hidden": 0.0, "a4_logits": 0.0, "a16_logits": 0.0}
        core.eval()
        with torch.no_grad():
            for ids, target in zip(dataset, targets, strict=True):
                replay._round = straight_through_round
                replay.set_w4a_replay_enabled(core, True)
                a4, a4_parts = _student_loss(
                    forward(ids), target, temperature=temperature,
                    hidden_weight=hidden_weight, logit_weight=logit_weight,
                )
                replay._round = identity_round
                replay.set_w4a_replay_enabled(core, False)
                a16, a16_parts = _student_loss(
                    forward(ids), target, temperature=temperature,
                    hidden_weight=hidden_weight, logit_weight=logit_weight,
                )
                totals["a4"] += float(a4)
                totals["a16"] += float(a16)
                totals["a4_hidden"] += a4_parts["hidden"]
                totals["a16_hidden"] += a16_parts["hidden"]
                totals["a4_logits"] += a4_parts["logits"]
                totals["a16_logits"] += a16_parts["logits"]
        return {key: value / len(dataset) for key, value in totals.items()}

    try:
        validation_before = evaluate(validation, validation_targets)
        best_validation = dict(validation_before)
        best_step = 0
        best_deltas = [value.detach().cpu().clone() for value in scale_parameters]
        preserve_limit = validation_before["a16"] + max_preserve_loss_increase
        core.train()
        for step in range(steps):
            if step % len(order) == 0:
                random.shuffle(order)
            sample_index = order[step % len(order)]
            ids = samples[sample_index]
            target = train_targets[sample_index]
            optimizer.zero_grad(set_to_none=True)

            replay._round = straight_through_round
            replay.set_w4a_replay_enabled(core, True)
            a4_loss, a4_parts = _student_loss(
                forward(ids), target, temperature=temperature,
                hidden_weight=hidden_weight, logit_weight=logit_weight,
            )
            a4_loss.backward()

            replay._round = identity_round
            replay.set_w4a_replay_enabled(core, False)
            a16_loss, a16_parts = _student_loss(
                forward(ids), target, temperature=temperature,
                hidden_weight=hidden_weight, logit_weight=logit_weight,
            )
            preserve_loss = preserve_weight * a16_loss
            preserve_loss.backward()

            regularizer = _scale_regularizer(scale_parameters)
            if scale_regularization:
                (scale_regularization * regularizer).backward()
            torch.nn.utils.clip_grad_norm_(scale_parameters, 1.0)
            optimizer.step()
            _clamp_scale_deltas(scale_parameters, max_scale_change)
            entry = {
                "step": step + 1, "steps": steps,
                "a4": a4_parts, "a16": a16_parts,
                "regularizer": float(regularizer.detach()),
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)
            if (step + 1) % eval_interval == 0 or step + 1 == steps:
                current_validation = evaluate(validation, validation_targets)
                current_validation["step"] = step + 1
                current_validation["preserve_limit"] = preserve_limit
                accepted = (
                    current_validation["a16"] <= preserve_limit
                    and current_validation["a4"] < best_validation["a4"]
                )
                current_validation["accepted"] = accepted
                evaluations.append(current_validation)
                print(json.dumps({"validation": current_validation}), flush=True)
                if accepted:
                    best_validation = {
                        key: value for key, value in current_validation.items()
                        if key in validation_before
                    }
                    best_step = step + 1
                    best_deltas = [
                        value.detach().cpu().clone() for value in scale_parameters
                    ]
                core.train()
        with torch.no_grad():
            for parameter, best in zip(scale_parameters, best_deltas, strict=True):
                parameter.copy_(best.to(parameter.device))
        validation_after = evaluate(validation, validation_targets)
    finally:
        replay._round = original_round
        replay.set_w4a_replay_enabled(core, True)

    learned_multipliers = {
        name: module._w4a_qad_log_scale_delta.detach().exp().cpu()
        for name, module in scale_modules.items()
    }
    relative_changes = torch.cat([
        module._w4a_qad_log_scale_delta.detach().exp().flatten().cpu()
        for module in scale_modules.values()
    ])

    _copy_checkpoint_shell(source_checkpoint, output, calibration=calibration)
    source_tensors = load_file(str(source_checkpoint / "model.safetensors"), device="cpu")
    immutable_suffixes = ("qweight", "qzeros", "g_idx")
    packed_digest = _tensor_digest(source_tensors, immutable_suffixes)
    changed = 0
    for name, multiplier in learned_multipliers.items():
        key = f"{name}.scales"
        original = source_tensors[key]
        # Apply the relative update to the original serialized FP16 values.
        # Rebuilding from the loader's BF16 copy would rewrite most scales even
        # when the trust region selected step zero.
        converted = (original.float() * multiplier).to(original.dtype).contiguous()
        changed += int(torch.count_nonzero(converted != original))
        source_tensors[key] = converted
    save_file(source_tensors, str(output / "model.safetensors"))
    output_tensors = load_file(str(output / "model.safetensors"), device="cpu")
    if _tensor_digest(output_tensors, immutable_suffixes) != packed_digest:
        raise AssertionError("Scale QAD changed GPTQ codes, zero points, or group indices.")

    report = {
        "checkpoint": str(checkpoint.resolve()),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "activation_policy": {"version": qcfg.activation_version,
                              "mode": qcfg.activation_mode, "recipe": qcfg.activation_recipe},
        "output": str(output.resolve()),
        "rows": len(samples),
        "validation_rows": len(validation),
        "sequence_length": sequence_length,
        "steps": steps,
        "learning_rate": learning_rate,
        "preserve_weight": preserve_weight,
        "scale_regularization": scale_regularization,
        "max_scale_change": max_scale_change,
        "hidden_weight": hidden_weight,
        "logit_weight": logit_weight,
        "temperature": temperature,
        "seed": seed,
        "eval_interval": eval_interval,
        "max_preserve_loss_increase": max_preserve_loss_increase,
        "selected_step": best_step,
        "trainable_parameters": sum(value.numel() for value in scale_parameters),
        "changed_scale_values": changed,
        "relative_scale_min": float(relative_changes.min()),
        "relative_scale_max": float(relative_changes.max()),
        "relative_scale_mean": float(relative_changes.mean()),
        "validation_before": validation_before,
        "validation_after": validation_after,
        "first_step": history[0],
        "last_step": history[-1],
        "evaluations": evaluations,
        "immutable_gptq_digest": packed_digest,
    }
    (output / "w4a_scale_qad_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True,
                        help="Verified directory prepared by tests.models.w4a_calibration_data.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--validation-rows", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--preserve-weight", type=float, default=4.0)
    parser.add_argument("--scale-regularization", type=float, default=0.1)
    parser.add_argument("--max-scale-change", type=float, default=0.03)
    parser.add_argument("--hidden-weight", type=float, default=1.0)
    parser.add_argument("--logit-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--max-preserve-loss-increase", type=float, default=5e-4)
    args = parser.parse_args()
    train_scales(
        args.checkpoint, args.source_checkpoint, args.calibration, args.output,
        rows=args.rows, validation_rows=args.validation_rows,
        sequence_length=args.sequence_length, steps=args.steps,
        learning_rate=args.learning_rate, preserve_weight=args.preserve_weight,
        scale_regularization=args.scale_regularization,
        max_scale_change=args.max_scale_change, hidden_weight=args.hidden_weight,
        logit_weight=args.logit_weight, temperature=args.temperature, seed=args.seed,
        eval_interval=args.eval_interval,
        max_preserve_loss_increase=args.max_preserve_loss_increase,
    )


if __name__ == "__main__":
    main()
