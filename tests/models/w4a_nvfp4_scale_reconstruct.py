# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Refit native GPTQ group scales against a true NVFP4 activation stream.

The INT4 codes, zero points, group map, and int32 packing stay byte-identical.
Only the existing GPTQ scale tensor is solved, so the saved model remains a
native GPTQ checkpoint and the post-init two-plane E2M1 decomposition remains
exact for its updated dequantized weights.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from .w4a_gb10_memory import require_w4a_test_headroom
from .w4a_nvfp4_norm_qat import (
    _ROTATION_FIELDS,
    _calibration_ids,
    _copy_checkpoint_shell,
    _dequantize_for_replay,
    _tensor_digest,
)


def _fit_scales(
    x: torch.Tensor,
    teacher_x: torch.Tensor,
    target: torch.Tensor,
    codes: torch.Tensor,
    initial: torch.Tensor,
    *,
    ridge: float,
    chunk_size: int,
    preserve_weight: float,
    max_scale_change: float,
) -> tuple[torch.Tensor, dict]:
    """Solve one scale per GPTQ group and output channel."""
    if (x.ndim != 2 or teacher_x.ndim != 2 or target.ndim != 2
            or codes.ndim != 2 or initial.ndim != 2):
        raise ValueError("Scale reconstruction expects rank-two tensors.")
    tokens, width = x.shape
    groups, outputs = initial.shape
    if (teacher_x.shape != x.shape or codes.shape != (width, outputs)
            or target.shape != (tokens, outputs)
            or width != groups * 128):
        raise ValueError("Scale reconstruction tensor shapes do not match GPTQ group-128 layout.")
    if tokens < max(96, groups * 3):
        raise ValueError(f"Need at least {max(96, groups * 3)} tokens to fit {groups} scales.")

    train_end = max(groups + 1, int(tokens * 0.60))
    select_end = max(train_end + 1, int(tokens * 0.80))
    if select_end >= tokens:
        select_end = tokens - 1
    splits = {
        "train": slice(0, train_end),
        "select": slice(train_end, select_end),
        "holdout": slice(select_end, tokens),
    }
    fitted = initial.float().clone()
    accepted = 0
    before_holdout_sse = after_holdout_sse = holdout_energy = 0.0
    teacher_holdout_sse = 0.0

    x_device = x.to("cuda:0", dtype=torch.float32)
    teacher_x_device = teacher_x.to("cuda:0", dtype=torch.float32)
    target_device = target.to("cuda:0", dtype=torch.float32)
    codes_device = codes.to("cuda:0", dtype=torch.float32)
    initial_device = initial.to("cuda:0", dtype=torch.float32)
    for start in range(0, outputs, chunk_size):
        stop = min(start + chunk_size, outputs)
        contributions = []
        teacher_contributions = []
        for group in range(groups):
            left = group * 128
            contributions.append(
                x_device[:, left:left + 128] @ codes_device[left:left + 128, start:stop]
            )
            teacher_contributions.append(
                teacher_x_device[:, left:left + 128] @ codes_device[left:left + 128, start:stop]
            )
        design = torch.stack(contributions, dim=1)  # [tokens, groups, output_chunk]
        teacher_design = torch.stack(teacher_contributions, dim=1)
        student_train = design[splits["train"]].permute(2, 0, 1)
        teacher_train = teacher_design[splits["train"]].permute(2, 0, 1)
        train = torch.cat((student_train, preserve_weight ** 0.5 * teacher_train), dim=1)
        original_train_target = target_device[splits["train"], start:stop].T
        train_target = torch.cat(
            (original_train_target, preserve_weight ** 0.5 * original_train_target), dim=1,
        )
        gram = train.transpose(1, 2) @ train
        rhs = (train.transpose(1, 2) @ train_target.unsqueeze(-1)).squeeze(-1)
        base = initial_device[:, start:stop].T
        diagonal_mean = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-12)
        penalty = ridge * diagonal_mean
        identity = torch.eye(groups, device=gram.device, dtype=gram.dtype).unsqueeze(0)
        candidate = torch.linalg.solve(
            gram + penalty[:, None, None] * identity,
            rhs + penalty[:, None] * base,
        )
        candidate = candidate.clamp(
            min=base * (1.0 - max_scale_change),
            max=base * (1.0 + max_scale_change),
        )

        selection = design[splits["select"]].permute(2, 0, 1)
        teacher_selection = teacher_design[splits["select"]].permute(2, 0, 1)
        selection_target = target_device[splits["select"], start:stop].T
        base_selection = torch.bmm(selection, base.unsqueeze(-1)).squeeze(-1)
        candidate_selection = torch.bmm(selection, candidate.unsqueeze(-1)).squeeze(-1)
        base_teacher_selection = torch.bmm(teacher_selection, base.unsqueeze(-1)).squeeze(-1)
        candidate_teacher_selection = torch.bmm(
            teacher_selection, candidate.unsqueeze(-1)
        ).squeeze(-1)
        base_sse = (base_selection - selection_target).square().sum(dim=-1)
        candidate_sse = (candidate_selection - selection_target).square().sum(dim=-1)
        base_sse += preserve_weight * (base_teacher_selection - selection_target).square().sum(dim=-1)
        candidate_sse += preserve_weight * (
            candidate_teacher_selection - selection_target
        ).square().sum(dim=-1)
        use = candidate_sse < base_sse
        chosen = torch.where(use[:, None], candidate, base)
        accepted += int(use.sum().item())
        fitted[:, start:stop] = chosen.T.cpu()

        holdout = design[splits["holdout"]].permute(2, 0, 1)
        teacher_holdout = teacher_design[splits["holdout"]].permute(2, 0, 1)
        holdout_target = target_device[splits["holdout"], start:stop].T
        base_holdout = torch.bmm(holdout, base.unsqueeze(-1)).squeeze(-1)
        chosen_holdout = torch.bmm(holdout, chosen.unsqueeze(-1)).squeeze(-1)
        chosen_teacher_holdout = torch.bmm(
            teacher_holdout, chosen.unsqueeze(-1)
        ).squeeze(-1)
        before_holdout_sse += float((base_holdout - holdout_target).square().sum())
        after_holdout_sse += float((chosen_holdout - holdout_target).square().sum())
        teacher_holdout_sse += float((chosen_teacher_holdout - holdout_target).square().sum())
        holdout_energy += float(holdout_target.square().sum())

    keep_module = (
        after_holdout_sse + preserve_weight * teacher_holdout_sse < before_holdout_sse
    )
    if not keep_module:
        fitted.copy_(initial.float())
        after_holdout_sse = before_holdout_sse
        teacher_holdout_sse = 0.0
        accepted = 0

    report = {
        "tokens": tokens,
        "groups": groups,
        "outputs": outputs,
        "accepted_output_channels": accepted,
        "accepted_fraction": accepted / outputs,
        "module_update_kept": keep_module,
        "holdout_relative_rmse_before": (before_holdout_sse / max(holdout_energy, 1e-30)) ** 0.5,
        "holdout_relative_rmse_after": (after_holdout_sse / max(holdout_energy, 1e-30)) ** 0.5,
        "teacher_holdout_relative_rmse_after": (
            teacher_holdout_sse / max(holdout_energy, 1e-30)
        ) ** 0.5,
    }
    return fitted, report


def reconstruct_scales(
    checkpoint: Path,
    source_checkpoint: Path,
    calibration: Path,
    output: Path,
    *,
    rows: int,
    sequence_length: int,
    token_limit: int,
    ridge: float,
    chunk_size: int,
    preserve_weight: float,
    max_scale_change: float,
) -> dict:
    require_w4a_test_headroom(require_scope=True)
    if (rows <= 0 or sequence_length <= 1 or token_limit < 192 or ridge < 0
            or chunk_size <= 0 or preserve_weight < 0
            or not 0 < max_scale_change < 1):
        raise ValueError("Invalid scale reconstruction dimensions or regularization.")

    from transformers import AutoTokenizer

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear import w4a_llama_replay as replay
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.nn_modules.qlinear.w4a_floatx import W4AFP8Linear

    wrapper = GPTQModel.load(
        str(checkpoint), backend=BACKEND.GPTQ_TORCH,
        device="cuda:0", dtype=torch.bfloat16,
    )
    core = wrapper.model
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
    core = _dequantize_for_replay(core, device="cuda:0", dtype=torch.bfloat16)
    modules = dict(core.named_modules())
    for name, fields in rotation.items():
        for field, value in fields.items():
            setattr(modules[name], field, value)
    core.eval()
    core.config.use_cache = False

    qcfg = SimpleNamespace(
        activation_mode="w4a_nvfp4", activation_recipe="least_squares",
        dynamic_get=lambda **_kwargs: None,
    )
    replay.install_w4a_llama_replay(core, qcfg)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    samples = _calibration_ids(tokenizer, calibration, rows, sequence_length)

    teacher_outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    teacher_inputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    student_inputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    phase = "teacher"
    handles = []

    def capture_input(name):
        def hook(_module, args):
            destination = student_inputs[name] if phase == "student" else teacher_inputs[name]
            if sum(value.shape[0] for value in destination) < token_limit:
                value = args[0].detach().reshape(-1, args[0].shape[-1]).cpu()
                remaining = token_limit - sum(item.shape[0] for item in destination)
                destination.append(value[:remaining])
        return hook

    def capture_output(name):
        def hook(_module, _args, result):
            if phase == "teacher" and sum(value.shape[0] for value in teacher_outputs[name]) < token_limit:
                value = result.detach().reshape(-1, result.shape[-1]).cpu()
                remaining = token_limit - sum(item.shape[0] for item in teacher_outputs[name])
                teacher_outputs[name].append(value[:remaining])
        return hook

    for name in quantized:
        handles.append(modules[name].register_forward_pre_hook(capture_input(name)))
        handles.append(modules[name].register_forward_hook(capture_output(name)))

    original_round = replay._round

    def identity_round(x, _mode, _recipe=None, _global_scale=None):
        return x

    try:
        replay._round = identity_round
        replay.set_w4a_replay_enabled(core, False)
        with torch.inference_mode():
            for ids in samples:
                core(input_ids=ids.unsqueeze(0).to("cuda:0"), use_cache=False)
        phase = "student"
        replay._round = replay.round_w4a_activation
        replay.set_w4a_replay_enabled(core, True)
        with torch.inference_mode():
            for ids in samples:
                core(input_ids=ids.unsqueeze(0).to("cuda:0"), use_cache=False)
    finally:
        replay._round = original_round
        replay.set_w4a_replay_enabled(core, True)
        for handle in handles:
            handle.remove()

    learned_scales = {}
    module_reports = {}
    for index, name in enumerate(quantized, start=1):
        x = torch.cat(student_inputs[name], dim=0)
        teacher_x = torch.cat(teacher_inputs[name], dim=0)
        target = torch.cat(teacher_outputs[name], dim=0)
        if x.shape[0] != target.shape[0] or teacher_x.shape[0] != target.shape[0]:
            raise AssertionError(f"Teacher/student token count differs for {name}.")
        learned, module_report = _fit_scales(
            x, teacher_x, target, codes[name], initial_scales[name], ridge=ridge,
            chunk_size=chunk_size, preserve_weight=preserve_weight,
            max_scale_change=max_scale_change,
        )
        learned_scales[name] = learned
        module_reports[name] = module_report
        print(json.dumps({"module": index, "modules": len(quantized), "name": name, **module_report}), flush=True)

    _copy_checkpoint_shell(source_checkpoint, output, calibration=calibration)
    source_tensors = load_file(str(source_checkpoint / "model.safetensors"), device="cpu")
    immutable_suffixes = ("qweight", "qzeros", "g_idx")
    packed_digest = _tensor_digest(source_tensors, immutable_suffixes)
    changed = 0
    for name, learned in learned_scales.items():
        key = f"{name}.scales"
        original = source_tensors[key]
        converted = learned.to(original.dtype).contiguous()
        changed += int(torch.count_nonzero(converted != original))
        source_tensors[key] = converted
    save_file(source_tensors, str(output / "model.safetensors"))
    output_tensors = load_file(str(output / "model.safetensors"), device="cpu")
    if _tensor_digest(output_tensors, immutable_suffixes) != packed_digest:
        raise AssertionError("Scale reconstruction changed GPTQ codes, zeros, or group indices.")

    before = sum(value["holdout_relative_rmse_before"] for value in module_reports.values()) / len(module_reports)
    after = sum(value["holdout_relative_rmse_after"] for value in module_reports.values()) / len(module_reports)
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "output": str(output.resolve()),
        "rows": rows,
        "sequence_length": sequence_length,
        "token_limit": token_limit,
        "ridge": ridge,
        "preserve_weight": preserve_weight,
        "max_scale_change": max_scale_change,
        "modules": len(module_reports),
        "mean_holdout_relative_rmse_before": before,
        "mean_holdout_relative_rmse_after": after,
        "changed_scale_values": changed,
        "immutable_gptq_digest": packed_digest,
        "module_reports": module_reports,
    }
    (output / "w4a_scale_reconstruction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "module_reports"}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True,
                        help="Verified directory prepared by tests.models.w4a_calibration_data.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--token-limit", type=int, default=768)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--preserve-weight", type=float, default=1.0)
    parser.add_argument("--max-scale-change", type=float, default=0.1)
    args = parser.parse_args()
    reconstruct_scales(
        args.checkpoint, args.source_checkpoint, args.calibration, args.output,
        rows=args.rows, sequence_length=args.sequence_length, token_limit=args.token_limit,
        ridge=args.ridge, chunk_size=args.chunk_size,
        preserve_weight=args.preserve_weight, max_scale_change=args.max_scale_change,
    )


if __name__ == "__main__":
    main()
