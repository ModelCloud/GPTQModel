#!/usr/bin/env python3
"""Test QTIP-style end-to-end soft-target alignment without changing the QVQ format."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.looper.qvq_output_alignment import _FixedTrellisAlignmentLinear
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.utils.model import recurse_setattr

if __package__:
    from scripts.analyze_gptq_low_bit_grid import load_nm_evaluation_batch
else:
    from analyze_gptq_low_bit_grid import load_nm_evaluation_batch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("neuralmagic/calibration"))
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-rows", type=int, default=32)
    parser.add_argument("--validation-offset", type=int, default=96)
    parser.add_argument("--validation-rows", type=int, default=16)
    parser.add_argument("--evaluation-offset", type=int, default=128)
    parser.add_argument("--evaluation-rows", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--evaluation-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--gradient-accumulation-mode", choices=("sum", "mean"), default="sum")
    parser.add_argument("--forward-parameter-precision", choices=("qtip-fp32", "native"), default="qtip-fp32")
    parser.add_argument("--trainable-parameter-scope", choices=("su-sv", "all"), default="su-sv")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path)
    return parser


def _load_dense(path: Path, device: str):
    kwargs = {
        "dtype": torch.float16,
        "attn_implementation": "eager",
        "local_files_only": True,
    }
    if torch.device(device).type == "mps":
        # Direct sharded materialization onto MPS can terminate inside Metal.
        # One CPU materialization followed by one transfer is stable.
        return AutoModelForCausalLM.from_pretrained(
            path,
            low_cpu_mem_usage=True,
            **kwargs,
        ).to(device).eval()
    return AutoModelForCausalLM.from_pretrained(path, device_map={"": device}, **kwargs).eval()


def _encoded_rows(tokenizer, args, *, offset: int, rows: int) -> dict[str, torch.Tensor]:
    encoded, _ = load_nm_evaluation_batch(
        tokenizer,
        dataset_path=args.dataset,
        row_offset=offset,
        rows=rows,
        max_length=args.max_length,
    )
    return encoded


def _batches(encoded: dict[str, torch.Tensor], batch_size: int):
    rows = next(iter(encoded.values())).shape[0]
    for start in range(0, rows, batch_size):
        batch = {name: value[start : start + batch_size] for name, value in encoded.items()}
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None and attention_mask.ndim == 2:
            # Tokenization pads the complete row slice to one global width.
            # Remove only columns that are padding for every row in this batch
            # so batch-1/full-length training does not repeatedly forward the
            # slice's longest sequence. This preserves left/right padding,
            # internal masks, token order, and every valid next-token pair.
            active_columns = attention_mask.ne(0).any(dim=0).nonzero(as_tuple=False).flatten()
            if active_columns.numel() == 0:
                raise ValueError("QVQ end-to-end batch contains no valid tokens.")
            first = int(active_columns[0])
            last = int(active_columns[-1]) + 1
            for name, value in tuple(batch.items()):
                if value.ndim >= 2 and value.shape[1] == attention_mask.shape[1]:
                    batch[name] = value[:, first:last]
        yield batch


def _valid_next_token_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    keep = attention_mask[:, 1:].to(torch.bool)
    return logits[:, :-1][keep]


@torch.no_grad()
def _logit_digest(model, encoded, *, batch_size: int, device: torch.device) -> str:
    """Hash held-out non-padding logits without retaining vocabulary tensors."""

    digest = hashlib.sha256()
    for batch in _batches(encoded, batch_size):
        batch = {name: value.to(device) for name, value in batch.items()}
        logits = _valid_next_token_logits(
            model(**batch, use_cache=False).logits,
            batch["attention_mask"],
        ).contiguous().cpu()
        digest.update(str(logits.dtype).encode("ascii"))
        digest.update(str(tuple(logits.shape)).encode("ascii"))
        digest.update(logits.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _evaluate(teacher, student, encoded, *, batch_size: int, device: torch.device) -> dict[str, float]:
    sums = {"kld": 0.0, "jsd": 0.0, "top1": 0.0, "top5": 0.0}
    tokens = 0
    for batch in _batches(encoded, batch_size):
        batch = {name: value.to(device) for name, value in batch.items()}
        teacher_logits = _valid_next_token_logits(
            teacher(**batch, use_cache=False).logits,
            batch["attention_mask"],
        ).float()
        student_logits = _valid_next_token_logits(
            student(**batch, use_cache=False).logits,
            batch["attention_mask"],
        ).float()
        teacher_log_prob = F.log_softmax(teacher_logits, dim=-1)
        student_log_prob = F.log_softmax(student_logits, dim=-1)
        teacher_prob = teacher_log_prob.exp()
        student_prob = student_log_prob.exp()
        midpoint_log = ((teacher_prob + student_prob) * 0.5).clamp_min(1e-30).log()
        count = teacher_logits.shape[0]
        sums["kld"] += (teacher_prob * (teacher_log_prob - student_log_prob)).sum().item()
        sums["jsd"] += (
            0.5
            * (
                (teacher_prob * (teacher_log_prob - midpoint_log)).sum()
                + (student_prob * (student_log_prob - midpoint_log)).sum()
            ).item()
        )
        teacher_top5 = teacher_logits.topk(5, dim=-1).indices
        student_top5 = student_logits.topk(5, dim=-1).indices
        sums["top1"] += (teacher_top5[:, 0] == student_top5[:, 0]).sum().item()
        sums["top5"] += (
            (teacher_top5.unsqueeze(-1) == student_top5.unsqueeze(-2))
            .any(dim=-1)
            .float()
            .sum()
            .item()
            / 5.0
        )
        tokens += count
    return {
        "valid_tokens": float(tokens),
        "forward_kld": sums["kld"] / tokens,
        "jensen_shannon": sums["jsd"] / tokens,
        "top1_agreement": sums["top1"] / tokens,
        "top5_overlap": sums["top5"] / tokens,
    }


def _passes_accuracy_gate(baseline: dict[str, float], candidate: dict[str, float]) -> bool:
    """Apply one strict, same-population distributional acceptance gate."""

    required = ("valid_tokens", "forward_kld", "jensen_shannon", "top1_agreement", "top5_overlap")
    for label, metrics in (("baseline", baseline), ("candidate", candidate)):
        missing = set(required) - set(metrics)
        if missing:
            raise ValueError(f"QVQ {label} metrics are missing acceptance fields: {sorted(missing)}")
        nonfinite = [name for name in required if not math.isfinite(float(metrics[name]))]
        if nonfinite:
            raise ValueError(f"QVQ {label} metrics contain non-finite acceptance fields: {nonfinite}")
    if baseline["valid_tokens"] <= 0 or candidate["valid_tokens"] <= 0:
        raise ValueError("QVQ acceptance metrics require a positive valid-token population.")
    if candidate["valid_tokens"] != baseline["valid_tokens"]:
        raise ValueError(
            "QVQ acceptance metrics must use the same valid-token population: "
            f"baseline={baseline['valid_tokens']} candidate={candidate['valid_tokens']}"
        )
    return (
        candidate["forward_kld"] < baseline["forward_kld"]
        and candidate["jensen_shannon"] < baseline["jensen_shannon"]
        and candidate["top1_agreement"] >= baseline["top1_agreement"]
        and candidate["top5_overlap"] >= baseline["top5_overlap"]
    )


def _install_differentiable_qvq(student) -> tuple[dict[str, QVQLinear], dict[str, _FixedTrellisAlignmentLinear]]:
    originals = {
        name: module
        for name, module in student.named_modules()
        if name and isinstance(module, QVQLinear)
    }
    temporaries = {}
    for name, module in originals.items():
        temporary = _FixedTrellisAlignmentLinear(
            inner_weight=module.get_inner_weight_tensor(dtype=torch.float32),
            SU=module.SU,
            SV=module.SV,
            bias=module.bias,
            output_dtype=None,
        )
        recurse_setattr(student, name, temporary)
        temporaries[name] = temporary
    if not temporaries:
        raise RuntimeError("End-to-end QVQ alignment found no QVQLinear modules.")
    return originals, temporaries


def _restore_runtime_qvq(student, originals, temporaries, state) -> None:
    for (name, original), (best_SU, best_SV) in zip(originals.items(), state):
        temporary = temporaries[name]
        temporary.SU.data.copy_(best_SU.to(device=temporary.SU.device))
        temporary.SV.data.copy_(best_SV.to(device=temporary.SV.device))
        original.SU.copy_(temporary.SU.detach().to(device=original.SU.device, dtype=torch.float32))
        original.SV.copy_(temporary.SV.detach().to(device=original.SV.device, dtype=torch.float32))
        recurse_setattr(student, name, original)


def _state(temporaries) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (module.SU.detach().cpu().clone(), module.SV.detach().cpu().clone())
        for module in temporaries.values()
    ]


def _parameter_state(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in parameters]


def _restore_parameter_state(parameters: list[torch.nn.Parameter], state: list[torch.Tensor]) -> None:
    if len(parameters) != len(state):
        raise ValueError(f"QVQ parameter state count changed: parameters={len(parameters)} state={len(state)}")
    for parameter, value in zip(parameters, state):
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _floating_tensor_dtypes(model) -> list[tuple[torch.nn.Module, str, bool, torch.dtype]]:
    tensors = []
    for module in model.modules():
        tensors.extend(
            (module, name, True, parameter.dtype)
            for name, parameter in module._parameters.items()
            if parameter is not None and parameter.is_floating_point()
        )
        tensors.extend(
            (module, name, False, buffer.dtype)
            for name, buffer in module._buffers.items()
            if buffer is not None and buffer.is_floating_point()
        )
    return tensors


def _restore_floating_tensor_dtypes(
    tensors: list[tuple[torch.nn.Module, str, bool, torch.dtype]],
) -> None:
    for module, name, is_parameter, dtype in tensors:
        if is_parameter:
            current = module._parameters[name]
            if current.dtype != dtype:
                current.data = current.data.to(dtype=dtype, copy=True)
        else:
            current = module._buffers[name]
            if current.dtype != dtype:
                module._buffers[name] = current.to(dtype=dtype, copy=True)


def main() -> None:
    args = _parser().parse_args()
    integer_fields = (
        "train_rows",
        "validation_rows",
        "evaluation_rows",
        "max_length",
        "train_batch_size",
        "evaluation_batch_size",
        "gradient_accumulation",
        "epochs",
    )
    if any(getattr(args, name) < 1 for name in integer_fields):
        raise ValueError("QVQ end-to-end alignment row, batch, length, accumulation, and epoch counts must be positive.")
    if any(getattr(args, name) < 0 for name in ("train_offset", "validation_offset", "evaluation_offset")):
        raise ValueError("QVQ end-to-end alignment row offsets must be nonnegative.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("QVQ end-to-end alignment learning rate must be finite and positive.")
    if args.results.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {args.results}")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.dense_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train = _encoded_rows(tokenizer, args, offset=args.train_offset, rows=args.train_rows)
    validation = _encoded_rows(tokenizer, args, offset=args.validation_offset, rows=args.validation_rows)
    evaluation = _encoded_rows(tokenizer, args, offset=args.evaluation_offset, rows=args.evaluation_rows)

    started = time.perf_counter()
    teacher = _load_dense(args.dense_model, args.device)
    quantized = GPTQModel.load(
        str(args.checkpoint),
        backend=BACKEND.QVQ,
        dtype=torch.float16,
        device_map={"": args.device},
        attn_implementation="eager",
    )
    student = quantized.model.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(False)

    baseline_validation = _evaluate(teacher, student, validation, batch_size=args.evaluation_batch_size, device=device)
    baseline_evaluation = _evaluate(teacher, student, evaluation, batch_size=args.evaluation_batch_size, device=device)
    original_dtypes = _floating_tensor_dtypes(student)
    if args.forward_parameter_precision == "qtip-fp32":
        # Match QTIP's `quant_model.float()` plus FP16 autocast contract. Only
        # temporary SU/SV remain trainable in this partial-checkpoint
        # experiment; all live dtypes are restored exactly before save.
        student.float()
    originals, temporaries = _install_differentiable_qvq(student)
    if args.trainable_parameter_scope == "all":
        # QTIP's official E2E implementation passes `quant_model.parameters()`
        # to Adam after replacing SU/SV with Parameters. Keep that scope an
        # explicit research arm because it also changes the unquantized
        # embeddings, norms, biases, and language-model head.
        for parameter in student.parameters():
            parameter.requires_grad_(True)
        parameters = list(student.parameters())
    else:
        parameters = [parameter for module in temporaries.values() for parameter in (module.SU, module.SV)]
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_parameter_state = _parameter_state(parameters)
    best_validation = _evaluate(teacher, student, validation, batch_size=args.evaluation_batch_size, device=device)
    train_losses = []

    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(_batches(train, args.train_batch_size)):
            batch = {name: value.to(device) for name, value in batch.items()}
            with torch.no_grad():
                teacher_logits = _valid_next_token_logits(
                    teacher(**batch, use_cache=False).logits,
                    batch["attention_mask"],
                ).float()
                teacher_prob = F.softmax(teacher_logits, dim=-1)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                student_logits = _valid_next_token_logits(
                    student(**batch, use_cache=False).logits,
                    batch["attention_mask"],
                ).float()
                loss = -(teacher_prob * F.log_softmax(student_logits, dim=-1)).sum(dim=-1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "QVQ end-to-end alignment produced a non-finite training loss "
                    f"at epoch={epoch} batch={batch_index}."
                )
            # The authoritative QTIP implementation sums full per-batch
            # gradients across `ft_update_freq`; `mean` is an explicit W1
            # stability control rather than a claim about the paper recipe.
            backward_loss = (
                loss / args.gradient_accumulation
                if args.gradient_accumulation_mode == "mean"
                else loss
            )
            scaler.scale(backward_loss).backward()
            train_losses.append(float(loss.detach().item()))
            final_batch = batch_index + 1 == math.ceil(args.train_rows / args.train_batch_size)
            if (batch_index + 1) % args.gradient_accumulation == 0 or final_batch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        current = _evaluate(teacher, student, validation, batch_size=args.evaluation_batch_size, device=device)
        if current["forward_kld"] < best_validation["forward_kld"]:
            best_validation = current
            best_parameter_state = _parameter_state(parameters)

    _restore_parameter_state(parameters, best_parameter_state)
    _restore_runtime_qvq(student, originals, temporaries, _state(temporaries))
    _restore_floating_tensor_dtypes(original_dtypes)
    candidate_validation = _evaluate(teacher, student, validation, batch_size=args.evaluation_batch_size, device=device)
    candidate_evaluation = _evaluate(teacher, student, evaluation, batch_size=args.evaluation_batch_size, device=device)
    # Select and save only from the declared validation split. The evaluation
    # population is a report-only holdout and must not influence model choice.
    accepted = _passes_accuracy_gate(baseline_validation, candidate_validation)
    evaluation_passed = _passes_accuracy_gate(baseline_evaluation, candidate_evaluation)
    if accepted and args.output_checkpoint is not None:
        live_validation_logit_sha256 = _logit_digest(
            student,
            validation,
            batch_size=args.evaluation_batch_size,
            device=device,
        )
        if args.output_checkpoint.exists():
            raise FileExistsError(f"Refusing to overwrite existing checkpoint: {args.output_checkpoint}")
        quantized.save(str(args.output_checkpoint))
        reloaded = GPTQModel.load(
            str(args.output_checkpoint),
            backend=BACKEND.QVQ,
            dtype=torch.float16,
            device_map={"": args.device},
            attn_implementation="eager",
        )
        reloaded_evaluation = _evaluate(
            teacher,
            reloaded.model.eval(),
            evaluation,
            batch_size=args.evaluation_batch_size,
            device=device,
        )
        reloaded_validation_logit_sha256 = _logit_digest(
            reloaded.model.eval(),
            validation,
            batch_size=args.evaluation_batch_size,
            device=device,
        )
        if reloaded_validation_logit_sha256 != live_validation_logit_sha256:
            raise AssertionError(
                "QVQ end-to-end save/reload changed held-out FP16 logits: "
                f"live={live_validation_logit_sha256} reloaded={reloaded_validation_logit_sha256}"
            )
        for key, candidate_value in candidate_evaluation.items():
            if not math.isclose(reloaded_evaluation[key], candidate_value, rel_tol=0, abs_tol=1e-12):
                raise AssertionError(
                    f"QVQ end-to-end save/reload changed `{key}`: "
                    f"live={candidate_value} reloaded={reloaded_evaluation[key]}"
                )
    else:
        reloaded_evaluation = None
        live_validation_logit_sha256 = None
        reloaded_validation_logit_sha256 = None

    report = {
        "commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1]
        ).strip(),
        "python": platform.python_version(),
        "python_gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
        "torch": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "dense_model": str(args.dense_model),
        "checkpoint": str(args.checkpoint),
        "output_checkpoint": str(args.output_checkpoint) if accepted and args.output_checkpoint else None,
        "fixed_trellis_modules": len(originals),
        "trainable_parameter_scope": args.trainable_parameter_scope,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "gradient_accumulation_math": args.gradient_accumulation_mode,
        "forward_precision": args.forward_parameter_precision,
        "config": vars(args) | {"dense_model": str(args.dense_model), "checkpoint": str(args.checkpoint)},
        "baseline_validation": baseline_validation,
        "candidate_validation": candidate_validation,
        "baseline_evaluation": baseline_evaluation,
        "candidate_evaluation": candidate_evaluation,
        "reloaded_evaluation": reloaded_evaluation,
        "live_validation_logit_sha256": live_validation_logit_sha256,
        "reloaded_validation_logit_sha256": reloaded_validation_logit_sha256,
        "best_surrogate_validation": best_validation,
        "mean_train_cross_entropy": sum(train_losses) / len(train_losses),
        "accepted": accepted,
        "acceptance_split": "validation",
        "evaluation_passed": evaluation_passed,
        "seconds": time.perf_counter() - started,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
