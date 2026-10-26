# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Full-parameter QAD for NVFP4 activations and native GPTQ INT4 weights.

The optimization representation is a latent FP32 weight tensor with a native
GPTQ INT4 straight-through forward. Export rounds it back to [-8, 7], packs those
codes into the checkpoint's standard INT32 qweight, and leaves qzeros, g_idx,
and activation-scale metadata in their native GPTQ layout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType

import torch
from safetensors.torch import load_file, save_file

from .w4a_gb10_memory import require_w4a_test_headroom
from .w4a_nvfp4_norm_qat import (
    _ROTATION_FIELDS,
    _activation_replay_config,
    _calibration_ids_from_verified_records,
    _copy_checkpoint_shell,
    _dequantize_for_replay,
    _tensor_digest,
)
from .w4a_nvfp4_scale_qad import _student_loss


def weight_qad_preflight(checkpoint: Path, source_checkpoint: Path, calibration: Path,
                         output: Path, *, rows: int, validation_rows: int,
                         teacher_cache_dir: Path | None = None,
                         include_records: bool = False,
                         calibration_format: str = "plain") -> dict | tuple[dict, dict]:
    """Pin the data partition and unchanged initial weights before GPU work."""
    from .w4a_calibration_data import ARITHMETIC_CORPUS, file_digest, load_calibration_artifact

    if calibration_format not in {"plain", "chat_worked_examples"}:
        raise ValueError("Unsupported calibration text format")

    for path in (output, teacher_cache_dir):
        if path is not None and (path.exists() or path.is_symlink()):
            raise FileExistsError(path)
    if not (checkpoint / "model.safetensors").samefile(source_checkpoint / "model.safetensors"):
        raise ValueError("Weight adaptation teacher and source must share the exact native weight file")
    configs = [json.loads((path / "quantize_config.json").read_text())
               for path in (checkpoint, source_checkpoint)]
    for field in ("bits", "group_size", "desc_act", "sym", "quant_method", "pack_dtype", "rotation"):
        if configs[0].get(field) != configs[1].get(field):
            raise ValueError(f"Weight adaptation teacher/source quantization setting differs: {field}")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (checkpoint / name).read_bytes() != (source_checkpoint / name).read_bytes():
            raise ValueError(f"Weight adaptation teacher/source tokenizer differs: {name}")
    partitions, manifest = load_calibration_artifact(calibration)
    if calibration_format == "chat_worked_examples" and manifest["corpus"] != ARITHMETIC_CORPUS:
        raise ValueError("Chat calibration requires the audited arithmetic corpus")
    if not 0 < rows <= len(partitions["fit"]) or not 0 < validation_rows <= len(partitions["selection"]):
        raise ValueError("Requested fitting/selection rows exceed the audited artifact or are empty")
    report = {"artifact": str(calibration.resolve()), "manifest_sha256": manifest["manifest_sha256"],
            "corpus": manifest["corpus"],
            "calibration_format": calibration_format,
            "chat_template_sha256": file_digest(checkpoint / "chat_template.jinja"),
            "fit_article_ids": [record["article_id"] for record in partitions["fit"][:rows]],
            "selection_article_ids": [record["article_id"] for record in partitions["selection"][:validation_rows]]}
    return (report, partitions) if include_records else report


class DiskTeacherTargets(Sequence):
    """Lossless teacher tensors with only one sample resident per access.

    Each run owns a new directory; cached targets are never silently reused
    across changed checkpoints, prompts, or tokenizer settings.
    On Linux, release file-cache pages after writes and reads so a large
    teacher corpus does not consume the training cgroup's memory budget.
    """
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.paths: list[Path] = []
        self.bytes_on_disk = 0

    @staticmethod
    def _release_file_cache(path: Path, *, synchronize: bool = False) -> None:
        if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
            return
        with path.open("rb") as handle:
            if synchronize:
                # DONTNEED cannot discard dirty pages. Persist this newly
                # written private target file before asking for reclamation.
                os.fsync(handle.fileno())
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

    def append(self, target: dict) -> None:
        tensors = {"logits": target["logits"].detach().cpu().clone().contiguous()}
        tensors.update({f"hidden_{i}": value.detach().cpu().clone().contiguous()
                        for i, value in enumerate(target["hidden_states"])})
        path = self.directory / f"{len(self.paths):08d}.safetensors"
        save_file(tensors, str(path))
        self._release_file_cache(path, synchronize=True)
        self.paths.append(path)
        self.bytes_on_disk += path.stat().st_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        if not isinstance(index, int):
            raise TypeError("Teacher targets must be loaded one sample at a time.")
        path = self.paths[index]
        mapped = load_file(str(path), device="cpu")
        # Own only this sample's tensors, then close safetensors' mappings
        # before advising away its cached file pages. Returned tensors must
        # never alias storage whose lifetime depends on the mapped file.
        tensors = {name: value.clone() for name, value in mapped.items()}
        del mapped
        self._release_file_cache(path)
        return {"logits": tensors["logits"],
                "hidden_states": tuple(tensors[f"hidden_{i}"] for i in range(len(tensors) - 1))}


def _project_master_codes(master_codes: torch.Tensor,
                          base_codes: torch.Tensor, *, radius: float = 0.25) -> torch.Tensor:
    """Keep master direction within a configurable, strictly interior code cell.

    The 0.49 upper bound leaves an FP32-safe gap before a rounding boundary.
    A larger radius permits earlier discrete transitions during subsequent
    fitting; it must never change the original serialized codes at step zero.
    """
    _validate_master_cell_radius(radius)
    offset = master_codes.float() - base_codes.float()
    return base_codes.float() + radius * torch.tanh(offset / radius)


def _validate_master_cell_radius(radius: float) -> None:
    if isinstance(radius, bool) or not math.isfinite(radius) or not 0 < radius <= 0.49:
        raise ValueError("Master cell radius must be finite and in (0, 0.49]")


def _latent_codes(module: torch.nn.Module) -> torch.Tensor:
    if hasattr(module, "_w4a_qad_latent_codes"):
        return module._w4a_qad_latent_codes
    return module._w4a_qad_latent_weight / module._w4a_qad_scales.T.unsqueeze(-1)


class WeightGradientCoverage:
    """Reject a detached training branch even when another lane has gradients."""

    def __init__(self, named_parameters):
        self.expected = set(named_parameters)
        self.seen = set()
        self.active = None
        self.counts = {"a4": 0, "a16": 0}
        self.handles = []
        for name, parameter in named_parameters.items():
            def record(gradient, *, name=name):
                if self.active is not None:
                    self.seen.add(name)
                return gradient
            self.handles.append(parameter.register_hook(record))

    @contextmanager
    def lane(self, name):
        if self.active is not None or name not in self.counts:
            raise RuntimeError("Gradient coverage needs one A4 or A16 backward at a time")
        self.active = name
        self.seen.clear()
        try:
            yield
            missing = self.expected - self.seen
            if missing:
                raise RuntimeError(f"Missing {name} weight gradients: {sorted(missing)}")
            self.counts[name] += 1
        finally:
            self.active = None
            self.seen.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _token_length_summary(samples: Sequence[torch.Tensor]) -> dict[str, int]:
    lengths = [sample.numel() for sample in samples]
    if not lengths or min(lengths) <= 0:
        raise ValueError("Token accounting requires nonempty samples")
    return {"rows": len(lengths), "tokens": sum(lengths),
            "minimum": min(lengths), "maximum": max(lengths)}


def _install_trainable_gptq_codes(
    core: torch.nn.Module,
    codes: dict[str, torch.Tensor],
    scales: dict[str, torch.Tensor],
    master_weights: dict[str, torch.Tensor] | None = None,
    *, parameterization: str = "physical_weight", master_cell_radius: float = 0.25,
) -> tuple[list[torch.nn.Parameter], dict[str, torch.nn.Module]]:
    _validate_master_cell_radius(master_cell_radius)
    if parameterization not in {"physical_weight", "code_cell"}:
        raise ValueError("Unknown GPTQ weight parameterization")
    modules = dict(core.named_modules())
    parameters: list[torch.nn.Parameter] = []
    selected = {}
    for name, centered in codes.items():
        module = modules[name]
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"Expected dense Linear after GPTQ dequantization: {name}")
        if module.in_features % 128:
            raise ValueError(f"GPTQ weight QAD requires group size 128: {name}")
        expected_codes = (module.in_features, module.out_features)
        expected_scales = (module.in_features // 128, module.out_features)
        if centered.shape != expected_codes or scales[name].shape != expected_scales:
            raise ValueError(f"GPTQ code or scale shape mismatch for {name}.")

        base = centered.T.contiguous().to(
            device=module.weight.device, dtype=torch.int8,
        )
        scale = scales[name].to(device=module.weight.device, dtype=torch.float32)
        grouped_base = base.reshape(
            module.out_features, module.in_features // 128, 128
        )
        weight_scale = scale.T.unsqueeze(-1)
        if master_weights is None:
            latent_value = grouped_base.float() * weight_scale
        else:
            master = master_weights[name].to(
                device=module.weight.device, dtype=torch.float32,
            ).reshape_as(grouped_base)
            # Preserve the exact existing GPTQ code at step zero while
            # restoring the full-precision master's position and direction
            # inside that code's quantization cell.  tanh avoids piling values
            # on a shared clipping boundary. The configurable radius retains
            # the residual direction while preserving a strictly interior
            # starting position before any serialized INT4 code changes.
            initial_latent_codes = _project_master_codes(
                master / weight_scale, grouped_base, radius=master_cell_radius,
            )
            latent_value = initial_latent_codes * weight_scale
            initial_codes = (latent_value / weight_scale).round().clamp(-8, 7)
            if not torch.equal(initial_codes.to(torch.int8), grouped_base):
                raise AssertionError(f"Master initialization changed initial GPTQ codes for {name}.")
        latent = torch.nn.Parameter(
            latent_value / weight_scale if parameterization == "code_cell" else latent_value
        )
        module.register_buffer("_w4a_qad_base_codes", base, persistent=False)
        module.register_buffer("_w4a_qad_scales", scale, persistent=False)
        module.register_buffer(
            "_w4a_qad_initial_latent_codes",
            (latent_value / weight_scale).detach().clone(), persistent=False,
        )
        module.register_parameter(
            "_w4a_qad_latent_codes" if parameterization == "code_cell" else "_w4a_qad_latent_weight",
            latent,
        )
        module.register_parameter("weight", None)

        def weight_forward(self, x):
            scale = self._w4a_qad_scales
            latent_codes = _latent_codes(self)
            rounded = latent_codes.round().clamp(-8, 7)
            # Forward uses real INT4 values; backward follows the latent
            # selected parameterization. Code-cell updates avoid dividing a
            # common physical Adam step by vastly different GPTQ scales.
            ste_codes = latent_codes + (rounded - latent_codes).detach()
            rows = x.numel() // self.in_features
            grouped_x = x.reshape(rows, self.in_features // 128, 128)
            grouped_codes = ste_codes.to(x.dtype).reshape(
                self.out_features, self.in_features // 128, 128
            )
            # Hardware accumulates each FP4 group in FP32. Returning a
            # BF16/FP16 partial here adds an extra rounding before GPTQ scale
            # application and teaches a different function from inference.
            partial = torch.einsum("tgi,ogi->tgo", grouped_x.float(), grouped_codes.float())
            # Preserve GPTQ's group contract: scale each partial K reduction,
            # then sum groups. The FP16 cast matches serialized scale storage.
            scale = scale + (scale.to(torch.float16).float() - scale).detach()
            result = (partial.float() * scale.unsqueeze(0)).sum(dim=1)
            if self.bias is not None:
                result = result + self.bias.float()
            return result.reshape(*x.shape[:-1], self.out_features).to(x.dtype)

        module.forward = MethodType(weight_forward, module)
        parameters.append(latent)
        selected[name] = module
    return parameters, selected


def _recover_hadamard_rotation(
    native_embeddings: torch.Tensor, rotated_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Recover the saved D @ H rotation from unquantized embedding witnesses.

    Sampling a new random D is invalid even with the same run seed: model load
    and other operations may have advanced the RNG. Since H is orthogonal and
    symmetric, (E @ D @ H) @ H = E @ D. Column correlations recover D's signs.
    No GPTQ weights or evaluation data participate in the recovery.
    """
    if (native_embeddings.ndim != 2 or native_embeddings.shape != rotated_embeddings.shape
            or native_embeddings.shape[0] < 2):
        raise ValueError("Rotation recovery requires matching [tokens, hidden] embedding witnesses.")
    width = native_embeddings.shape[1]
    if width < 2 or width & (width - 1):
        raise ValueError("Rotation recovery requires a power-of-two hidden width.")
    native = native_embeddings.detach().to(device="cpu", dtype=torch.float64)
    saved = rotated_embeddings.detach().to(device="cpu", dtype=torch.float64)
    hadamard = torch.ones((1, 1), dtype=torch.float64)
    while hadamard.shape[0] < width:
        hadamard = torch.cat((torch.cat((hadamard, hadamard), dim=1),
                              torch.cat((hadamard, -hadamard), dim=1)), dim=0)
    hadamard /= math.sqrt(width)
    signed = saved @ hadamard.T
    correlations = (native * signed).sum(dim=0)
    if not bool(torch.isfinite(correlations).all()) or bool((correlations == 0).any()):
        raise ValueError("Embedding witnesses cannot identify every Hadamard sign.")
    recovered = correlations.sign()[:, None] * hadamard
    predicted = native @ recovered
    relative_error = (predicted - saved).norm() / saved.norm().clamp_min(1e-30)
    # This is an identity/provenance check on BF16 checkpoint witnesses, not a
    # relaxed kernel oracle tolerance. Correct transforms retain BF16 rounding
    # noise; unrelated rotations or checkpoints must fail before training.
    if float(relative_error) > 0.005:
        raise ValueError(f"Saved embeddings do not match a signed Hadamard rotation: {float(relative_error):.6g}")
    return recovered, {
        "witness_tokens": native.shape[0], "hidden_size": width,
        "relative_embedding_error": float(relative_error),
        "negative_signs": int((correlations < 0).sum()),
    }


@torch.inference_mode()
def _load_rotated_master_weights(
    checkpoint: Path,
    module_names: tuple[str, ...],
    rotation: str | None,
    rotated_embeddings: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Load BF16 master weights in the exact coordinate system used by GPTQ."""
    from transformers import AutoModelForCausalLM

    from gptqmodel.quantization.rotation.rotation import fuse_layer_norms, rotate_model
    from gptqmodel.utils.torch import torch_empty_cache

    if rotation != "hadamard":
        raise ValueError(
            "Full-precision master initialization currently requires Hadamard rotation."
        )
    master = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True,
    )
    witness_rows = min(64, rotated_embeddings.shape[0])
    recovered_rotation, recovery_report = _recover_hadamard_rotation(
        master.model.embed_tokens.weight[:witness_rows], rotated_embeddings[:witness_rows],
    )
    print(json.dumps({"master_rotation_recovery": recovery_report}), flush=True)
    if master.config.tie_word_embeddings:
        master.config.tie_word_embeddings = False
        master.lm_head.weight = torch.nn.Parameter(master.lm_head.weight.detach().clone())
    fuse_layer_norms(
        master, pre_lm_head_norm_module_name="model.norm",
        layers_node="model.layers", lm_head_name="lm_head",
    )
    master, _ = rotate_model(
        master, rotate_mode=rotation, device=torch.device("cuda:0"),
        layers_node="model.layers", lm_head_name="lm_head",
        Q=recovered_rotation.to("cuda:0"),
    )
    modules = dict(master.named_modules())
    result = {
        name: modules[name].weight.detach().to(device="cpu", dtype=torch.float32).clone()
        for name in module_names
    }
    del modules, master
    torch_empty_cache()
    return result


def _weight_regularizer(parameters: list[torch.nn.Parameter],
                        modules: dict[str, torch.nn.Module]) -> torch.Tensor:
    numerator = torch.zeros((), device=parameters[0].device)
    count = 0
    for module in modules.values():
        # Measure movement in quantizer-cell units. A physical-weight penalty
        # is scaled down twice by small GPTQ scales and does not prevent many
        # latent values from crossing INT4 decision boundaries together.
        delta = (
            _latent_codes(module)
            - module._w4a_qad_initial_latent_codes
        )
        numerator = numerator + delta.square().sum()
        count += delta.numel()
    return numerator / count


@torch.no_grad()
def _clamp_latent_codes(modules: dict[str, torch.nn.Module], max_code_change: float) -> None:
    for module in modules.values():
        codes = _latent_codes(module)
        base = module._w4a_qad_base_codes.reshape_as(codes).float()
        scale = module._w4a_qad_scales.T.unsqueeze(-1)
        if hasattr(module, "_w4a_qad_latent_codes"):
            lower = (base - max_code_change).clamp_min(-8)
            upper = (base + max_code_change).clamp_max(7)
            codes.copy_(torch.maximum(torch.minimum(codes, upper), lower))
            continue
        lower = torch.maximum(-8.0 * scale, (base - max_code_change) * scale)
        upper = torch.minimum(7.0 * scale, (base + max_code_change) * scale)
        module._w4a_qad_latent_weight.copy_(torch.maximum(
            torch.minimum(module._w4a_qad_latent_weight, upper), lower,
        ))


@torch.no_grad()
def _snapshot_codes(modules: dict[str, torch.nn.Module]) -> dict[str, torch.Tensor]:
    return {
        name: _latent_codes(module)
        .round().clamp(-8, 7).reshape(module.out_features, module.in_features)
        .to(device="cpu", dtype=torch.int8).T.contiguous()
        for name, module in modules.items()
    }


@torch.no_grad()
def _changed_codes_by_module(modules: dict[str, torch.nn.Module]) -> dict[str, int]:
    """Locate discrete changes, including rejected candidates before rollback."""
    changed = {}
    for name, module in modules.items():
        current = _latent_codes(module)
        current = current.round().clamp(-8, 7).reshape_as(module._w4a_qad_base_codes)
        changed[name] = int(torch.count_nonzero(current != module._w4a_qad_base_codes))
    return changed


def _changed_code_count(modules: dict[str, torch.nn.Module]) -> int:
    return sum(_changed_codes_by_module(modules).values())


class DiagnosticCandidate:
    """Keep an A4-loss candidate for measurement, without accepting its quality."""

    def __init__(self, initial_loss: float):
        if not math.isfinite(initial_loss):
            raise ValueError("Diagnostic selection requires finite losses")
        self.loss = initial_loss
        self.step = 0
        self.validation = None
        self.codes = None

    def consider(self, step: int, validation: dict, snapshot) -> bool:
        if not all(math.isfinite(validation[key]) for key in ("a4", "a16")):
            raise ValueError("Diagnostic selection requires finite losses")
        if validation["a4"] >= self.loss:
            return False
        # Complete the snapshot before replacing any previous candidate.
        codes = snapshot()
        self.loss = validation["a4"]
        self.step = step
        self.validation = dict(validation)
        self.codes = codes
        return True


def _pack_centered_int4(centered: torch.Tensor) -> torch.Tensor:
    if centered.ndim != 2 or centered.shape[0] % 8:
        raise ValueError("GPTQ centered codes must have [K, N] shape with K divisible by 8.")
    if bool(((centered < -8) | (centered > 7)).any()):
        raise ValueError("Centered GPTQ codes must be in [-8, 7].")
    raw = centered.to(torch.int32) + 8
    shifts = torch.arange(8, dtype=torch.int32, device=centered.device).mul_(4).view(1, 8, 1)
    return (raw.reshape(centered.shape[0] // 8, 8, centered.shape[1]) << shifts).sum(
        dim=1, dtype=torch.int32
    ).contiguous()


def _export_code_candidate(source_checkpoint: Path, output: Path, calibration: Path | None,
                           codes: dict[str, torch.Tensor]) -> tuple[int, str]:
    """Write a new native checkpoint, replacing only selected INT4 code tensors."""
    _copy_checkpoint_shell(source_checkpoint, output, calibration=calibration)
    source_tensors = load_file(str(source_checkpoint / "model.safetensors"), device="cpu")
    immutable_digest = _tensor_digest(source_tensors, ("qzeros", "g_idx"))
    changed_codes = 0
    for name, centered in codes.items():
        key = f"{name}.qweight"
        packed = _pack_centered_int4(centered)
        if packed.shape != source_tensors[key].shape or source_tensors[key].dtype != torch.int32:
            raise ValueError(f"Packed GPTQ qweight layout mismatch for {name}.")
        old = source_tensors[key]
        old_raw = torch.stack([
            (old >> (4 * index)) & 15 for index in range(8)
        ], dim=1).reshape_as(centered)
        changed_codes += int(torch.count_nonzero(old_raw.to(torch.int8) - 8 != centered))
        source_tensors[key] = packed
    save_file(source_tensors, str(output / "model.safetensors"))
    output_tensors = load_file(str(output / "model.safetensors"), device="cpu")
    if _tensor_digest(output_tensors, ("qzeros", "g_idx")) != immutable_digest:
        raise AssertionError("Weight QAD changed GPTQ zero points or group indices.")
    if any(tensor.dtype != torch.int32 for name, tensor in output_tensors.items()
           if name.endswith("qweight")):
        raise AssertionError("Weight QAD did not preserve INT32 GPTQ qweight storage.")
    return changed_codes, immutable_digest


def _weight_qad_loss(result, target: dict, *, objective: str,
                     first_trainable_layer: int, last_trainable_layer: int,
                     temperature: float, hidden_weight: float,
                     logit_weight: float) -> tuple[torch.Tensor, dict]:
    if objective == "global":
        if hidden_weight == 0:
            from .w4a_nvfp4_scale_qad import _logit_distillation_loss
            logits = _logit_distillation_loss(result.logits, target["logits"], temperature)
            total = logit_weight * logits
            return total, {
                "hidden": 0.0,
                "logits": float(logits.detach()),
                "total": float(total.detach()),
            }
        return _student_loss(
            result, target, temperature=temperature,
            hidden_weight=hidden_weight, logit_weight=logit_weight,
        )
    if objective != "block":
        raise ValueError(f"Unsupported weight-QAD objective: {objective}.")

    if hidden_weight:
        hidden_losses = []
        # hidden_states[0] is the embedding output; hidden_states[index + 1]
        # is the output of decoder layer `index`.
        for index in range(first_trainable_layer, last_trainable_layer + 1):
            current = result.hidden_states[index + 1].float()
            expected = target["hidden_states"][index + 1].to(
                device=current.device, dtype=torch.float32,
            )
            hidden_losses.append(
                (current - expected).square().mean()
                / expected.square().mean().clamp_min(1e-6)
            )
        hidden = torch.stack(hidden_losses).mean()
    else:
        hidden = result.logits.new_zeros(())
    if logit_weight:
        from .w4a_nvfp4_scale_qad import _logit_distillation_loss
        logits = _logit_distillation_loss(result.logits, target["logits"], temperature)
    else:
        logits = hidden.new_zeros(())
    total = hidden_weight * hidden + logit_weight * logits
    return total, {
        "hidden": float(hidden.detach()),
        "logits": float(logits.detach()),
        "total": float(total.detach()),
    }


def straight_through_activation_round(x, mode, recipe=None, global_scale=None):
    """The shared forward used by weight adaptation and its runtime audit."""
    from gptqmodel.nn_modules.qlinear import w4a_llama_replay as replay
    from gptqmodel.nn_modules.qlinear.w4a_activation import pack_activation

    if mode == "w4a_nvfp4":
        if global_scale is not None:
            # Checkpoint producer scales are Python FP32-compatible scalars;
            # the hardware packer accepts a device tensor. Do not route this
            # conversion through the model's FP16/BF16 compute dtype.
            global_scale = torch.as_tensor(global_scale, device=x.device, dtype=torch.float32)
        quantized_value = pack_activation(
            x.detach(), mode, global_scale=global_scale,
            model_dtype=(x.dtype if x.dtype in (torch.float16, torch.bfloat16)
                         else torch.bfloat16), recipe=recipe,
        ).decode(x.dtype)
    else:
        quantized_value = replay.round_w4a_activation(x, mode, recipe, global_scale)
    return x + (quantized_value - x).detach()


def _cuda_memory_usage() -> dict[str, int]:
    """Allocator counters complement the external GB10 cgroup monitor.

    These cover PyTorch's CUDA allocator, not all driver or host allocations.
    Reading them does not empty caches or change the allocator policy.
    """
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def train_weights(
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
    weight_regularization: float,
    max_code_change: float,
    hidden_weight: float,
    logit_weight: float,
    temperature: float,
    seed: int,
    eval_interval: int,
    max_preserve_loss_increase: float,
    first_trainable_layer: int,
    last_trainable_layer: int,
    objective: str,
    master_checkpoint: Path | None,
    gradient_accumulation: int,
    warmup_steps: int,
    minimum_lr_ratio: float,
    stop_on_preservation_breach: bool,
    teacher_cache_dir: Path | None = None,
    gradient_checkpointing: bool = False,
    hardware_forward: bool = False,
    weight_parameterization: str = "physical_weight",
    adam_epsilon: float | None = None,
    native_forward: bool = False,
    master_cell_radius: float = 0.25,
    diagnostic_candidate_output: Path | None = None,
    calibration_format: str = "plain",
) -> dict:
    _validate_master_cell_radius(master_cell_radius)
    if master_checkpoint is None and master_cell_radius != 0.25:
        raise ValueError("A nondefault master cell radius requires a master checkpoint")
    if diagnostic_candidate_output is not None:
        if diagnostic_candidate_output.resolve() == output.resolve():
            raise ValueError("Diagnostic candidate must have a separate output directory")
        if diagnostic_candidate_output.exists() or diagnostic_candidate_output.is_symlink():
            raise FileExistsError(diagnostic_candidate_output)
    require_w4a_test_headroom(require_scope=True)
    qcfg = _activation_replay_config(source_checkpoint)
    if (rows <= 0 or validation_rows <= 0 or sequence_length <= 1 or steps <= 0
            or learning_rate <= 0 or preserve_weight < 0 or weight_regularization < 0
            or not 0.5 < max_code_change <= 15 or hidden_weight < 0 or logit_weight < 0
            or hidden_weight + logit_weight <= 0 or temperature <= 0
            or eval_interval <= 0 or max_preserve_loss_increase < 0
            or gradient_accumulation <= 0 or not 0 <= warmup_steps < steps
            or not 0 < minimum_lr_ratio <= 1
            or not 0 <= first_trainable_layer <= last_trainable_layer < 16
            or objective not in {"block", "global"}
            or (adam_epsilon is not None and (not math.isfinite(adam_epsilon) or adam_epsilon <= 0))
            or weight_parameterization not in {"physical_weight", "code_cell"}):
        raise ValueError("Invalid weight-QAD dimensions or optimization settings.")

    data_provenance, verified_records = weight_qad_preflight(
        checkpoint, source_checkpoint, calibration, output, rows=rows,
        validation_rows=validation_rows, teacher_cache_dir=teacher_cache_dir, include_records=True,
        calibration_format=calibration_format,
    )
    print(json.dumps({"stage": "preflight_complete", "fit_rows": rows,
                      "selection_rows": validation_rows,
                      "data_manifest_sha256": data_provenance["manifest_sha256"]}), flush=True)
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
    selected_quantized = {
        name: module for name, module in quantized.items()
        if any(name.startswith(f"model.layers.{index}.")
               for index in range(first_trainable_layer, last_trainable_layer + 1))
    }
    expected_selected = (last_trainable_layer - first_trainable_layer + 1) * 7
    if len(selected_quantized) != expected_selected:
        raise AssertionError(
            f"Expected {expected_selected} selected GPTQ Linears, found {len(selected_quantized)}."
        )
    codes = {
        name: W4AFP8Linear._centered_int4_codes(module).detach().cpu().to(torch.int8)
        for name, module in selected_quantized.items()
    }
    scales = {
        name: module.scales.detach().cpu().float().clone()
        for name, module in selected_quantized.items()
    }
    rotation = {
        name: {field: getattr(module, field, None) for field in _ROTATION_FIELDS}
        for name, module in quantized.items()
    }
    master_weights = None
    if master_checkpoint is not None:
        master_weights = _load_rotated_master_weights(
            master_checkpoint, tuple(selected_quantized), wrapper.quantize_config.rotation,
            core.model.embed_tokens.weight,
        )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    samples = _calibration_ids_from_verified_records(tokenizer, verified_records, rows, sequence_length,
                                                      text_format=calibration_format)
    validation = _calibration_ids_from_verified_records(
        tokenizer, verified_records, validation_rows, sequence_length, partition="selection",
        text_format=calibration_format,
    )
    del verified_records

    def teacher_target(ids: torch.Tensor) -> dict:
        result = core(
            input_ids=ids.unsqueeze(0).to("cuda:0"), use_cache=False,
            output_hidden_states=hidden_weight > 0, return_dict=True,
        )
        return {
            "logits": result.logits.detach().to(device="cpu", dtype=torch.bfloat16),
            "hidden_states": tuple(
                value.detach().to(device="cpu", dtype=torch.bfloat16)
                for value in (result.hidden_states or ())
            ) if hidden_weight > 0 else (),
        }

    core.eval()
    if teacher_cache_dir is None:
        train_targets, validation_targets = [], []
    else:
        teacher_cache_dir.mkdir(parents=True, exist_ok=False)
        train_targets = DiskTeacherTargets(teacher_cache_dir / "train")
        validation_targets = DiskTeacherTargets(teacher_cache_dir / "validation")
        (teacher_cache_dir / "manifest.json").write_text(json.dumps({
            "checkpoint": str(checkpoint.resolve()),
            "source_checkpoint": str(source_checkpoint.resolve()),
            "calibration": str(calibration.resolve()),
            "data_provenance": data_provenance,
            "rows": len(samples), "validation_rows": len(validation),
            "sequence_length": sequence_length, "seed": seed,
        }, indent=2) + "\n")
    with torch.inference_mode():
        for partition, dataset, targets in (("fit", samples, train_targets),
                                             ("selection", validation, validation_targets)):
            for index, ids in enumerate(dataset):
                targets.append(teacher_target(ids))
                if (index + 1) % 16 == 0 or index + 1 == len(dataset):
                    print(json.dumps({"stage": "teacher_cache", "partition": partition,
                                      "completed": index + 1, "rows": len(dataset)}), flush=True)

    core = _dequantize_for_replay(core, device="cuda:0", dtype=torch.bfloat16)
    modules = dict(core.named_modules())
    for name, fields in rotation.items():
        for field, value in fields.items():
            setattr(modules[name], field, value)
    core.config.use_cache = False
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    weight_parameters, weight_modules = _install_trainable_gptq_codes(
        core, codes, scales, master_weights, parameterization=weight_parameterization,
        master_cell_radius=master_cell_radius,
    )
    del codes, scales, master_weights

    replay.install_w4a_llama_replay(core, qcfg)
    if gradient_checkpointing:
        # Non-reentrant checkpointing supports frozen embeddings and the
        # trainable latent INT4 parameters inside each decoder. Each lane's
        # backward completes before the replay policy is switched below.
        core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    original_round = replay._round
    hardware = None
    hardware_forward_checks = 0
    if hardware_forward:
        from .w4a_hardware_forward import HardwareForward

        runtime = GPTQModel.load(str(source_checkpoint), backend=BACKEND.GPTQ_W4A_NVFP4,
                                device="cuda:0", dtype=torch.bfloat16).model.eval()
        hardware = HardwareForward(core, runtime, weight_modules)
        if hardware.sync_weights() != 0:
            raise AssertionError("Hardware-forward initialization changed saved INT4 codes")
    native = None
    native_forward_checks = 0
    if native_forward:
        from .w4a_native_forward import NativeForward

        native_runtime = GPTQModel.load(str(checkpoint), backend=BACKEND.GPTQ_TORCH,
                                       device="cuda:0", dtype=torch.bfloat16).model.eval()
        native = NativeForward(core, native_runtime, weight_modules)
        if native.sync_weights() != 0:
            raise AssertionError("Native-forward initialization changed saved INT4 codes")

    def identity_round(x, _mode, _recipe=None, _global_scale=None):
        return x

    if adam_epsilon is None:
        # Code gradients include the small GPTQ group scale. Keep Adam's
        # denominator floor from suppressing those gradients by default.
        adam_epsilon = 1e-12 if weight_parameterization == "code_cell" else 1e-8
    optimizer = torch.optim.AdamW(weight_parameters, lr=learning_rate,
                                 eps=adam_epsilon, weight_decay=0.0)
    coverage = WeightGradientCoverage(dict(zip(weight_modules, weight_parameters, strict=True)))
    order = list(range(len(samples)))
    history = []
    evaluations = []
    stop_reason = None

    def forward(ids: torch.Tensor):
        nonlocal hardware_forward_checks, native_forward_checks
        result = core(
            input_ids=ids.unsqueeze(0).to("cuda:0"), use_cache=False,
            output_hidden_states=hidden_weight > 0, return_dict=True,
        )
        if hardware is not None and hardware.active:
            if not torch.equal(result.logits, hardware.logits):
                raise AssertionError("Adaptation logits differ from the encoded hardware forward")
            hardware_forward_checks += 1
        if native is not None and native.active:
            if not torch.equal(result.logits, native.logits):
                raise AssertionError("Preservation logits differ from the native GPTQ forward")
            native_forward_checks += 1
        return result

    def frame(ids):
        return hardware.frame(ids.unsqueeze(0).to("cuda:0")) if hardware is not None else nullcontext()

    def native_frame(ids):
        return native.frame(ids.unsqueeze(0).to("cuda:0")) if native is not None else nullcontext()

    def evaluate(dataset, targets) -> dict:
        totals = {"a4": 0.0, "a16": 0.0, "a4_hidden": 0.0,
                  "a16_hidden": 0.0, "a4_logits": 0.0, "a16_logits": 0.0}
        core.eval()
        with torch.no_grad():
            for ids, target in zip(dataset, targets, strict=True):
                replay._round = straight_through_activation_round
                replay.set_w4a_replay_enabled(core, True)
                with frame(ids):
                    a4, a4_parts = _weight_qad_loss(
                        forward(ids), target, objective=objective,
                        first_trainable_layer=first_trainable_layer,
                        last_trainable_layer=last_trainable_layer,
                        temperature=temperature, hidden_weight=hidden_weight,
                        logit_weight=logit_weight,
                    )
                replay._round = identity_round
                replay.set_w4a_replay_enabled(core, False)
                with native_frame(ids):
                    a16, a16_parts = _weight_qad_loss(
                        forward(ids), target, objective=objective,
                        first_trainable_layer=first_trainable_layer,
                        last_trainable_layer=last_trainable_layer,
                        temperature=temperature, hidden_weight=hidden_weight,
                        logit_weight=logit_weight,
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
        if native is not None and validation_before["a16"] != 0.0:
            raise AssertionError("Unchanged native preservation model must exactly reproduce its teacher")
        print(json.dumps({"validation_before": validation_before}), flush=True)
        best_validation = dict(validation_before)
        best_step = 0
        best_codes = _snapshot_codes(weight_modules)
        diagnostic = DiagnosticCandidate(validation_before["a4"]) if diagnostic_candidate_output is not None else None
        preserve_limit = validation_before["a16"] + max_preserve_loss_increase
        core.train()
        sample_cursor = 0
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            step_tokens = 0
            a4_sums = {"hidden": 0.0, "logits": 0.0, "total": 0.0}
            a16_sums = {"hidden": 0.0, "logits": 0.0, "total": 0.0}
            for _micro_step in range(gradient_accumulation):
                if sample_cursor % len(order) == 0:
                    random.shuffle(order)
                sample_index = order[sample_cursor % len(order)]
                sample_cursor += 1
                ids = samples[sample_index]
                step_tokens += ids.numel()
                target = train_targets[sample_index]

                replay._round = straight_through_activation_round
                replay.set_w4a_replay_enabled(core, True)
                with frame(ids):
                    a4_loss, micro_a4 = _weight_qad_loss(
                        forward(ids), target, objective=objective,
                        first_trainable_layer=first_trainable_layer,
                        last_trainable_layer=last_trainable_layer,
                        temperature=temperature, hidden_weight=hidden_weight,
                        logit_weight=logit_weight,
                    )
                    with coverage.lane("a4"):
                        (a4_loss / gradient_accumulation).backward()
                replay._round = identity_round
                replay.set_w4a_replay_enabled(core, False)
                with native_frame(ids):
                    a16_loss, micro_a16 = _weight_qad_loss(
                        forward(ids), target, objective=objective,
                        first_trainable_layer=first_trainable_layer,
                        last_trainable_layer=last_trainable_layer,
                        temperature=temperature, hidden_weight=hidden_weight,
                        logit_weight=logit_weight,
                    )
                    with coverage.lane("a16"):
                        (preserve_weight * a16_loss / gradient_accumulation).backward()
                for key in a4_sums:
                    a4_sums[key] += micro_a4[key] / gradient_accumulation
                    a16_sums[key] += micro_a16[key] / gradient_accumulation
            regularizer = _weight_regularizer(weight_parameters, weight_modules)
            if weight_regularization:
                (weight_regularization * regularizer).backward()
            torch.nn.utils.clip_grad_norm_(weight_parameters, 1.0, error_if_nonfinite=True)
            if step < warmup_steps:
                lr_ratio = (step + 1) / max(warmup_steps, 1)
            else:
                progress = (step - warmup_steps) / max(steps - warmup_steps - 1, 1)
                lr_ratio = minimum_lr_ratio + (1.0 - minimum_lr_ratio) * (
                    0.5 + 0.5 * math.cos(math.pi * progress)
                )
            current_lr = learning_rate * lr_ratio
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.step()
            _clamp_latent_codes(weight_modules, max_code_change)
            refreshed_modules = hardware.sync_weights() if hardware is not None else 0
            native_refreshed = native.sync_weights() if native is not None else 0
            entry = {
                "step": step + 1, "steps": steps,
                "tokens": step_tokens,
                "a4": a4_sums, "a16": a16_sums,
                "regularizer": float(regularizer.detach()),
                "learning_rate": current_lr,
                "hardware_refreshed_modules": refreshed_modules,
                "native_refreshed_modules": native_refreshed,
                "cuda_memory": _cuda_memory_usage(),
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)

            if (step + 1) % eval_interval == 0 or step + 1 == steps:
                current = evaluate(validation, validation_targets)
                current["step"] = step + 1
                current["preserve_limit"] = preserve_limit
                current["changed_codes_by_module"] = _changed_codes_by_module(weight_modules)
                current["changed_int4_values"] = sum(current["changed_codes_by_module"].values())
                accepted = current["a16"] <= preserve_limit and current["a4"] < best_validation["a4"]
                current["accepted"] = accepted
                if diagnostic is not None:
                    current["selected_for_diagnostic"] = diagnostic.consider(
                        step + 1, current, lambda: _snapshot_codes(weight_modules),
                    )
                evaluations.append(current)
                print(json.dumps({"validation": current}), flush=True)
                if accepted:
                    best_validation = {
                        key: value for key, value in current.items() if key in validation_before
                    }
                    best_step = step + 1
                    best_codes = _snapshot_codes(weight_modules)
                if current["a16"] > preserve_limit and stop_on_preservation_breach:
                    # Optional early stop exports the last accepted snapshot.
                    # Without it, later optimizer steps may recover native
                    # preservation; snapshot acceptance remains unchanged.
                    stop_reason = "w4a16_preservation_limit"
                    break
                core.train()
    finally:
        replay._round = original_round
        replay.set_w4a_replay_enabled(core, True)
        if hardware is not None:
            hardware.close()
        if native is not None:
            native.close()
        coverage.close()

    from .w4a_calibration_data import file_digest
    if file_digest(calibration / "manifest.json") != data_provenance["manifest_sha256"]:
        raise ValueError("Calibration manifest changed during weight adaptation")
    changed_codes, immutable_digest = _export_code_candidate(source_checkpoint, output, calibration, best_codes)

    report = {
        "checkpoint": str(checkpoint.resolve()),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "data_provenance": data_provenance,
        "activation_policy": {"fused_norms": bool(getattr(qcfg, "rotation", None)),
                              "mode": qcfg.activation_mode, "recipe": qcfg.activation_recipe},
        "master_checkpoint": (
            str(master_checkpoint.resolve()) if master_checkpoint is not None else None
        ),
        "output": str(output.resolve()),
        "export_role": "native_preservation_selected",
        "downstream_acceptance": "not_evaluated",
        "teacher_cache_dir": str(teacher_cache_dir.resolve()) if teacher_cache_dir else None,
        "teacher_cache_bytes": (train_targets.bytes_on_disk + validation_targets.bytes_on_disk
                                if teacher_cache_dir else None),
        "rows": len(samples),
        "validation_rows": len(validation),
        "sequence_length": sequence_length,
        "steps": steps,
        "gradient_accumulation": gradient_accumulation,
        "gradient_checkpointing": gradient_checkpointing,
        "hardware_forward": hardware_forward,
        "exact_hardware_forward_checks": hardware_forward_checks,
        "native_forward": native_forward,
        "exact_native_forward_checks": native_forward_checks,
        "gradient_coverage_checks": dict(coverage.counts),
        "maximum_tokens_per_step": gradient_accumulation * sequence_length,
        "processed_training_tokens": sum(entry["tokens"] for entry in history),
        "fit_token_lengths": _token_length_summary(samples),
        "selection_token_lengths": _token_length_summary(validation),
        "learning_rate": learning_rate,
        "weight_parameterization": weight_parameterization,
        "master_cell_radius": master_cell_radius if master_checkpoint is not None else None,
        "adam_epsilon": adam_epsilon,
        "warmup_steps": warmup_steps,
        "minimum_lr_ratio": minimum_lr_ratio,
        "stop_on_preservation_breach": stop_on_preservation_breach,
        "preserve_weight": preserve_weight,
        "weight_regularization": weight_regularization,
        "max_code_change": max_code_change,
        "hidden_weight": hidden_weight,
        "logit_weight": logit_weight,
        "temperature": temperature,
        "seed": seed,
        "eval_interval": eval_interval,
        "max_preserve_loss_increase": max_preserve_loss_increase,
        "native_selection_limit": preserve_limit,
        "first_trainable_layer": first_trainable_layer,
        "last_trainable_layer": last_trainable_layer,
        "objective": objective,
        "stop_reason": stop_reason,
        "selected_step": best_step,
        "trainable_parameters": sum(value.numel() for value in weight_parameters),
        "changed_int4_values": changed_codes,
        "validation_before": validation_before,
        "validation_after": best_validation,
        "first_step": history[0],
        "last_step": history[-1],
        "evaluations": evaluations,
        "immutable_zero_group_digest": immutable_digest,
    }
    if diagnostic is not None:
        report["diagnostic_candidate"] = {"requested_output": str(diagnostic_candidate_output.resolve()),
                                           "exported": False, "selected_step": diagnostic.step}
        if diagnostic.codes is not None:
            diagnostic_changes, diagnostic_digest = _export_code_candidate(
                source_checkpoint, diagnostic_candidate_output, calibration, diagnostic.codes,
            )
            diagnostic_report = {
                **report, "output": str(diagnostic_candidate_output.resolve()),
                "export_role": "diagnostic_only", "downstream_acceptance": "not_evaluated",
                "passes_native_selection_gate": diagnostic.validation["a16"] <= preserve_limit,
                "selected_step": diagnostic.step, "changed_int4_values": diagnostic_changes,
                "validation_after": {key: diagnostic.validation[key] for key in validation_before},
                "immutable_zero_group_digest": diagnostic_digest,
            }
            diagnostic_report.pop("diagnostic_candidate")
            (diagnostic_candidate_output / "w4a_weight_qad_report.json").write_text(
                json.dumps(diagnostic_report, indent=2) + "\n")
            report["diagnostic_candidate"].update({"exported": True,
                "passes_native_selection_gate": diagnostic_report["passes_native_selection_gate"],
                "changed_int4_values": diagnostic_changes})
    (output / "w4a_weight_qad_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--master-checkpoint", type=Path)
    parser.add_argument("--master-cell-radius", type=float, default=0.25,
                        help="Initial latent-code radius in (0, 0.49]; preserves original INT4 codes. Requires a master checkpoint when changed.")
    parser.add_argument("--teacher-cache-dir", type=Path,
                        help="Write lossless teacher targets to a new directory and load one sample at a time.")
    parser.add_argument("--calibration", type=Path, required=True,
                        help="Verified directory prepared by tests.models.w4a_calibration_data.")
    parser.add_argument("--calibration-format", choices=("plain", "chat_worked_examples"), default="plain",
                        help="Render audited arithmetic examples as question/answer turns using the checkpoint chat template.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-candidate-output", type=Path,
                        help="Separately export the best A4-loss candidate for downstream measurement, even when it fails native KL selection. Never implies acceptance.")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--validation-rows", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-parameterization", choices=("physical_weight", "code_cell"),
                        default="physical_weight",
                        help="Optimize physical weights or dimensionless GPTQ code cells; LR uses those units.")
    parser.add_argument("--adam-epsilon", type=float,
                        help="Adam denominator floor; defaults to 1e-8 for physical weights, 1e-12 for code cells.")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Recompute decoder activations during backward to reduce peak training memory.")
    parser.add_argument("--hardware-forward", action="store_true",
                        help="Use actual encoded NVFP4 forward values with straight-through surrogate gradients.")
    parser.add_argument("--native-forward", action="store_true",
                        help="Use exact native GPTQ W4A16 values for the preservation objective.")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--minimum-lr-ratio", type=float, default=1.0)
    parser.add_argument(
        "--stop-on-preservation-breach", action="store_true",
        help="Stop at the first held-out W4A16 trust-region breach.",
    )
    parser.add_argument("--preserve-weight", type=float, default=4.0)
    parser.add_argument("--weight-regularization", type=float, default=1e-4)
    parser.add_argument("--max-code-change", type=float, default=1.0)
    parser.add_argument("--hidden-weight", type=float, default=1.0)
    parser.add_argument("--logit-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--max-preserve-loss-increase", type=float, default=5e-4)
    parser.add_argument("--first-trainable-layer", type=int, default=0)
    parser.add_argument("--last-trainable-layer", type=int, default=15)
    parser.add_argument("--objective", choices=("block", "global"), default="block")
    args = parser.parse_args()
    train_weights(
        args.checkpoint, args.source_checkpoint, args.calibration, args.output,
        rows=args.rows, validation_rows=args.validation_rows,
        sequence_length=args.sequence_length, steps=args.steps,
        learning_rate=args.learning_rate, preserve_weight=args.preserve_weight,
        weight_regularization=args.weight_regularization,
        max_code_change=args.max_code_change, hidden_weight=args.hidden_weight,
        logit_weight=args.logit_weight, temperature=args.temperature, seed=args.seed,
        eval_interval=args.eval_interval,
        max_preserve_loss_increase=args.max_preserve_loss_increase,
        first_trainable_layer=args.first_trainable_layer,
        last_trainable_layer=args.last_trainable_layer,
        objective=args.objective,
        master_checkpoint=args.master_checkpoint,
        master_cell_radius=args.master_cell_radius,
        diagnostic_candidate_output=args.diagnostic_candidate_output,
        calibration_format=args.calibration_format,
        teacher_cache_dir=args.teacher_cache_dir,
        gradient_accumulation=args.gradient_accumulation,
        gradient_checkpointing=args.gradient_checkpointing,
        hardware_forward=args.hardware_forward,
        native_forward=args.native_forward,
        weight_parameterization=args.weight_parameterization,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        minimum_lr_ratio=args.minimum_lr_ratio,
        stop_on_preservation_breach=args.stop_on_preservation_breach,
    )


if __name__ == "__main__":
    main()
