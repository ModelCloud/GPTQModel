# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Assert encoded W4A activation handoffs across Llama decoder operators."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .w4a_gb10_memory import require_w4a_test_headroom


def _checkpoint_fingerprint(checkpoint: Path) -> dict[str, str | None]:
    """Bind this single-file Llama audit to the bytes loaded from disk.

    Record absent optional tokenizer/generation files too, so adding one later
    cannot silently reuse an audit of a different input configuration.
    """
    required = ("model.safetensors", "config.json", "quantize_config.json",
                "tokenizer.json", "tokenizer_config.json")
    optional = ("chat_template.jinja", "generation_config.json", "special_tokens_map.json",
                "added_tokens.json", "tokenizer.model")
    result = {}
    for name in required + optional:
        path = checkpoint / name
        if name in optional and not path.exists() and not path.is_symlink():
            result[name] = None
            continue
        with path.open("rb") as handle:
            result[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return result


def _verify_checkpoint_fingerprint(checkpoint: Path, expected: dict[str, str | None]) -> None:
    if _checkpoint_fingerprint(checkpoint) != expected:
        raise ValueError("Checkpoint files changed since the consumer audit fingerprint was captured")


def _nvfp4_decode_reference(carrier, *, apply_outer_scales: bool = True) -> torch.Tensor:
    """Independent FP64 E2M1 decode with a tensor inverse of scale swizzling."""
    rows, width = carrier.codes.shape[0], carrier.shape[-1]
    packed = carrier.codes.view(torch.uint8)
    indices = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(rows, width).long()
    codebook = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.,
                             -0., -.5, -1., -1.5, -2., -3., -4., -6.),
                            dtype=torch.float64, device=packed.device)
    padded_rows = (rows + 127) // 128 * 128
    # [K-group, row-tile, block-half, row-low, row-high, block-low]
    # -> [row-tile, row-high, row-low, K-group, block-half, block-low].
    local = (carrier.scales.double().reshape(width // 128, padded_rows // 128, 2, 32, 4, 4)
             .permute(1, 4, 3, 0, 2, 5).reshape(padded_rows, width // 16)[:rows])
    result = codebook[indices] * local.repeat_interleave(16, dim=-1)
    if apply_outer_scales:
        result = result * carrier.global_scale.double()
        if carrier.token_scale is not None:
            result = result * carrier.token_scale.double()[:, None]
    return result.reshape(carrier.shape)


def _nvfp4_linear_reference(module, carrier, row_indices: torch.Tensor, *,
                           accumulation_dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Independent native-INT4 oracle with explicit accumulation precision.

    FP64 measures mathematical reconstruction. FP32 models the documented
    hardware accumulation/epilogue sequence, including final BF16 tie rounding.
    Neither path reads prepared weight planes or invokes the production GEMM.
    """
    if accumulation_dtype not in (torch.float32, torch.float64):
        raise ValueError("The oracle supports FP32 or FP64 accumulation")
    if ((module.online_full_had or module.online_partial_had)
            and not carrier.rotation_applied):
        raise AssertionError("The audited GEMM operand must already carry its online rotation")
    width = carrier.shape[-1]
    finite_precision = accumulation_dtype == torch.float32
    x = _nvfp4_decode_reference(carrier, apply_outer_scales=not finite_precision).reshape(-1, width)[row_indices]
    shifts = torch.arange(8, device=x.device, dtype=torch.int32) * 4
    expected_groups = torch.arange(width, device=x.device, dtype=torch.int32) // 128
    if not torch.equal(module.g_idx, expected_groups):
        raise AssertionError("NVFP4 GEMM oracle requires contiguous GPTQ groups")
    result = torch.zeros((len(row_indices), module.out_features), device=x.device, dtype=accumulation_dtype)
    for group in range(width // 128):
        packed = module.qweight[group * 16:(group + 1) * 16]
        codes = ((packed[:, None, :] >> shifts[None, :, None]) & 15).reshape(128, module.out_features)
        zeros = ((module.qzeros[group, :, None] >> shifts[None, :]) & 15).reshape(-1)
        if module.qzero_format() == 1:
            zeros = (zeros + 1) & 15
        if not bool((zeros == 8).all()):
            raise AssertionError("NVFP4 GEMM oracle requires symmetric INT4 zero point 8")
        centered = codes.double() - zeros.double()[None, :]
        operand = x[:, group * 128:(group + 1) * 128]
        if finite_precision:
            # Construct exact plane values independently from integer codes;
            # model the FP32 rounding at each specified arithmetic stage.
            high = torch.floor((centered + 2) / 4)
            low = centered - 4 * high
            partial = (operand @ low).float() + 4 * (operand @ high).float()
            result = result + partial * module.scales[group].float()
        else:
            result += (operand @ centered) * module.scales[group].double()
    if finite_precision:
        result = result * carrier.global_scale.float()
        if carrier.token_scale is not None:
            result = result * carrier.token_scale[row_indices, None].float()
    if module.bias is not None:
        result = result + module.bias.to(accumulation_dtype)
    return result


def _scale_layout_probe_rows(rows: int) -> list[int]:
    if rows < 1:
        raise ValueError("A GEMM audit requires at least one token row")
    return sorted({index for index in (0, 1, 15, 16, 31, 32, 33, 63, 64, 95, 96, 127, 128, 129, rows - 1)
                   if index < rows})


def _boundary_policy(name: str, config) -> tuple[str, str | None]:
    """Resolve the expected carrier from saved policy, independently of hooks."""
    parts = name.split(".")
    if parts[0:2] != ["model", "layers"] or len(parts) < 3:
        raise ValueError(f"Not a decoder boundary: {name}")
    is_mlp = len(parts) > 3 and parts[3] in {"mlp", "post_attention_layernorm"}
    if is_mlp:
        if int(parts[2]) in (config.activation_mlp_fp8_layers or ()):
            return "w4afp8", None
        return config.activation_mode, config.activation_recipe
    mode = config.activation_attention_mode or config.activation_mode
    recipe = (config.activation_attention_recipe or config.activation_recipe) if mode == "w4a_nvfp4" else None
    return mode, recipe


def _capture_fp32_output(module, operand):
    """Rerun only a failing Linear and capture its final pre-cast accumulator."""
    from gptqmodel.nn_modules.qlinear import w4a_nvfp4_triton as kernels

    original = kernels.nvfp4_accumulate_group
    captured = []

    def capture(*args, **kwargs):
        if kwargs["last"]:
            output = torch.empty_like(args[5], dtype=torch.float32)
            original(*args[:5], output, *args[6:], **kwargs)
            captured.append(output)
        return original(*args, **kwargs)

    kernels.nvfp4_accumulate_group = capture
    try:
        module.forward(operand)
    finally:
        kernels.nvfp4_accumulate_group = original
    if len(captured) != 1:
        raise AssertionError("Expected one final accumulator from the failing Linear")
    return captured[0]


def _assert_carrier_transport(producer: dict, consumer: dict, boundary: str) -> None:
    """A handoff must preserve the complete carrier, including outer scales."""
    fields = ("kind", "mode", "recipe", "shape", "model_dtype", "rotation_applied",
              "codes", "scales", "global_scale", "token_scale",
              "codes_ptr", "scales_ptr", "global_scale_ptr", "token_scale_ptr",
              "codes_version", "scales_version", "global_scale_version", "token_scale_version")
    if producer.get("kind") != "encoded" or consumer.get("kind") != "encoded":
        raise AssertionError(f"Encoded carrier lost at {boundary}")
    for field in fields:
        if field not in producer or field not in consumer:
            raise AssertionError(f"Missing carrier field {field} at {boundary}")
        if producer[field] != consumer[field]:
            raise AssertionError(f"Carrier {field} changed at {boundary}")


def _audit_handoffs(samples: dict, selected_layers: list[str]) -> int:
    """Check each prefill/decode invocation, rather than only the last call."""
    checked = 0

    def edge(producer, consumer):
        nonlocal checked
        left, right = samples[producer]["calls"], samples[consumer]["calls"]
        if not left or len(left) != len(right):
            raise AssertionError(f"Different invocation counts at {producer} -> {consumer}")
        for index, (old, new) in enumerate(zip(left, right, strict=True)):
            _assert_carrier_transport(old["output"], new["input"],
                                      f"{producer} -> {consumer}, invocation {index}")
            checked += 1

    for prior, following in zip(selected_layers, selected_layers[1:]):
        if int(following.rsplit(".", 1)[1]) == int(prior.rsplit(".", 1)[1]) + 1:
            edge(prior, following)
    for layer in selected_layers:
        for norm, consumers in (
            ("input_layernorm", ("self_attn", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
            ("post_attention_layernorm", ("mlp", "mlp.gate_proj", "mlp.up_proj")),
        ):
            for consumer in consumers:
                edge(f"{layer}.{norm}", f"{layer}.{consumer}")
    return checked


def audit(checkpoint: Path, variant: str, *, require_full_coverage: bool = True,
          prompt_result: Path | None = None, failure_dump_dir: Path | None = None) -> dict:
    require_w4a_test_headroom(require_scope=True)
    checkpoint_sha256 = _checkpoint_fingerprint(checkpoint)
    audit_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    from transformers import AutoTokenizer

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.w4a_activation import W4AActivation
    from gptqmodel.nn_modules.qlinear.w4a_floatx import W4AFP8Linear
    from gptqmodel.nn_modules.qlinear.w4a_nvfp4 import W4ANVFP4Linear

    backend = {
        "w4afp8": BACKEND.GPTQ_W4AFP8,
        "w4a_nvfp4": BACKEND.GPTQ_W4A_NVFP4,
    }[variant]
    model = GPTQModel.load(str(checkpoint), backend=backend, device="cuda:0", dtype=torch.bfloat16)
    model.model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if prompt_result is None:
        input_ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"].to("cuda:0")
    else:
        prompt = json.loads(prompt_result.read_text())["tests"][0]["samples"][0]["prompt"]
        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda:0")

    samples: dict[str, dict] = {}
    handles = []
    decode_checks = []
    gemm_checks = []
    pending_operands = {}

    def describe(value):
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, W4AActivation):
            if value.mode == "w4a_nvfp4":
                reference = _nvfp4_decode_reference(value).float()
                actual = value.decode(torch.float32)
                torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
                decode_checks.append(float((actual - reference).abs().max()))
            info = {"kind": "encoded", "mode": value.mode, "codes": str(value.codes.dtype),
                    "model_dtype": str(value.model_dtype), "rotation_applied": value.rotation_applied,
                    "recipe": value.recipe,
                    "token_scale": str(value.token_scale.dtype) if value.token_scale is not None else None,
                    "scales": str(value.scales.dtype), "global_scale":
                    str(value.global_scale.dtype) if value.global_scale is not None else None,
                    "shape": list(value.shape), "codes_ptr": value.codes.data_ptr(),
                    "scales_ptr": value.scales.data_ptr()}
            for key in ("codes", "scales", "global_scale", "token_scale"):
                tensor = getattr(value, key)
                info[f"{key}_ptr"] = tensor.data_ptr() if tensor is not None else None
                # Inference tensors have no version counter. Ordinary tensors
                # retain theirs so in-place changes are detected when possible.
                info[f"{key}_version"] = (tensor._version if tensor is not None
                                          and not tensor.is_inference() else None)
            return info
        if isinstance(value, torch.Tensor):
            return {"kind": "tensor", "dtype": str(value.dtype), "shape": list(value.shape)}
        raise AssertionError(f"Unexpected activation type: {type(value)}")

    def before(name: str):
        def hook(_module, args, kwargs):
            if not args and "hidden_states" not in kwargs:
                raise AssertionError(f"{name} received no input")
            operand = args[0] if args else kwargs["hidden_states"]
            samples[name]["input"] = describe(operand)
            # Mixed-policy NVFP4 modules also consume FP8 carriers. The E2M1
            # oracle applies only to actual FP4 operands; FP8 handoffs still
            # undergo the same complete carrier/policy checks below.
            if (isinstance(_module, W4ANVFP4Linear) and isinstance(operand, W4AActivation)
                    and operand.mode == "w4a_nvfp4"):
                pending_operands[name] = operand
        return hook

    def after(name: str):
        def hook(_module, _args, result):
            if name in pending_operands:
                operand = pending_operands.pop(name)
                # Probe row-bit transitions in the hardware scale layout,
                # its next tile and masked tail, plus single-token decoding.
                rows = torch.tensor(_scale_layout_probe_rows(operand.codes.shape[0]), device=result.device)
                precise = _nvfp4_linear_reference(_module, operand, rows)
                finite = _nvfp4_linear_reference(_module, operand, rows, accumulation_dtype=torch.float32)
                torch.testing.assert_close(finite.double(), precise, rtol=2e-3, atol=2e-3)
                expected = finite.to(result.dtype)
                actual = result.reshape(-1, _module.out_features)[rows]
                try:
                    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3,
                                               msg=lambda message: f"{name}: {message}")
                except AssertionError:
                    raw = _capture_fp32_output(_module, operand)[rows]
                    failed = (~torch.isclose(actual, expected, rtol=2e-3, atol=2e-3)).nonzero()
                    details = []
                    for row, column in failed[:8].tolist():
                        details.append({"row": int(rows[row]), "column": column,
                                        "fp64": float(precise[row, column]), "fp32": float(raw[row, column]),
                                        "actual": float(actual[row, column]), "expected": float(expected[row, column])})
                    print(json.dumps({"consumer_failure": name, "mismatches": len(failed),
                                      "pre_cast_max_abs_error": float((raw.double() - precise).abs().max()),
                                      "output_matches_fp32_cast": torch.equal(actual, raw.to(actual.dtype)),
                                      "examples": details}), flush=True)
                    if failure_dump_dir is not None:
                        from safetensors.torch import save_file
                        failure_dump_dir.mkdir(parents=True, exist_ok=False)
                        tensors = {"codes": operand.codes.view(torch.uint8), "block_scales": operand.scales.view(torch.uint8),
                                   "global_scale": operand.global_scale, "qweight": _module.qweight,
                                   "qzeros": _module.qzeros, "weight_scales": _module.scales, "g_idx": _module.g_idx,
                                   "row_indices": rows, "actual": actual, "fp64_reference": precise, "fp32_output": raw}
                        if operand.token_scale is not None:
                            tensors["token_scale"] = operand.token_scale
                        if _module.bias is not None:
                            tensors["bias"] = _module.bias
                        save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()},
                                  str(failure_dump_dir / "consumer.safetensors"))
                        (failure_dump_dir / "metadata.json").write_text(json.dumps({
                            "checkpoint": str(checkpoint), "module": name, "shape": operand.shape,
                            "qzero_format": _module.qzero_format(), "recipe": operand.recipe,
                            "rotation_applied": operand.rotation_applied,
                        }, indent=2) + "\n")
                    raise
                gemm_checks.append({"module": name, "rows": len(rows),
                                    "max_abs_error": float((actual.float() - expected.float()).abs().max()),
                                    "fp32_oracle_vs_fp64_max_abs_error": float((finite.double() - precise).abs().max())})
            samples[name]["output"] = describe(result)
            samples[name]["calls"].append({"input": samples[name]["input"],
                                           "output": samples[name]["output"]})
        return hook

    for name, module in model.model.named_modules():
        if isinstance(module, (W4AFP8Linear, W4ANVFP4Linear)):
            kind = "w4a_linear"
            if not getattr(module, "_require_activation_stream", False):
                raise AssertionError(f"{name} permits a BF16/FP16 fallback inside a stream model")
            if variant == "w4afp8":
                assert isinstance(module, W4AFP8Linear) and not isinstance(module, W4ANVFP4Linear)
            else:
                assert isinstance(module, W4ANVFP4Linear)
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
        samples[name] = {"kind": kind, "selected": selected, "calls": []}
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
    fused_norms = bool(getattr(model.model, "_w4a_stream_fused_norms", False))
    if not bool(torch.isfinite(output.logits).all()) or generated.shape[1] != input_ids.shape[1] + 2:
        raise AssertionError("W4A stream forward or cached generation produced invalid output.")
    for name, entry in samples.items():
        kind = entry["kind"]
        if "input" not in entry or "output" not in entry:
            raise AssertionError(f"Hook did not run for {entry}")
        counts[kind] = counts.get(kind, 0) + 1
        if entry["selected"]:
            boundary_mode, boundary_recipe = _boundary_policy(name, model.quantize_config)
            expected_code = "torch.float8_e4m3fn" if boundary_mode == "w4afp8" else "torch.float4_e2m1fn_x2"
            output_encoded = kind in {"decoder_layer", "rms_norm"}
            if output_encoded:
                if (entry["output"]["kind"] != "encoded" or entry["output"]["codes"] != expected_code
                        or entry["output"]["mode"] != boundary_mode):
                    raise AssertionError(f"Selected W4A carrier boundary is not encoded: {name}: {entry}")
                if entry["output"]["recipe"] != boundary_recipe:
                    raise AssertionError(f"Selected W4A boundary has the wrong scale recipe: {name}: {entry}")
            elif (entry["output"]["kind"] != "tensor"
                  or entry["output"]["dtype"] != "torch.bfloat16"):
                raise AssertionError(
                    f"A Linear/nonlinear branch did not emit model dtype: {name}: {entry}"
                )
            if kind == "rms_norm" and getattr(model.model.get_submodule(name), "_w4a_preserve_norm_codes", False):
                if entry["output"]["token_scale"] != "torch.float32":
                    raise AssertionError(f"Fused-norm boundary lost its token multiplier: {name}")
                for pointer in ("codes_ptr", "scales_ptr"):
                    if entry["input"][pointer] != entry["output"][pointer]:
                        raise AssertionError(f"Fused-norm boundary repacked its FP4 operand: {name}")
            if kind != "decoder_layer" and (entry["input"]["kind"] != "encoded" or
                                               entry["input"]["codes"] != expected_code or
                                               entry["input"]["mode"] != boundary_mode):
                raise AssertionError(f"Selected W4A consumer did not receive encoded input: {name}: {entry}")
            if kind != "decoder_layer" and entry["input"]["recipe"] != boundary_recipe:
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
    handoffs_checked = _audit_handoffs(samples, selected_layers)
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

    _verify_checkpoint_fingerprint(checkpoint, checkpoint_sha256)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != audit_source_sha256:
        raise ValueError("Consumer audit source changed during execution")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "audit_source_sha256": audit_source_sha256,
        "variant": variant,
        "activation_recipe": expected_recipe,
        "fused_norms": fused_norms,
        "model_dtype": "torch.bfloat16",
        "logits_dtype": str(output.logits.dtype),
        "input_tokens": input_ids.shape[1],
        "prompt_result": str(prompt_result) if prompt_result is not None else None,
        "generated_tokens": generated.shape[1] - input_ids.shape[1],
        "counts": counts,
        "selected_layers": selected_layers,
        "full_coverage": len(selected_layers) == len(model.model.model.layers),
        "handoffs_checked": handoffs_checked,
        "independent_decode_checks": len(decode_checks),
        "decode_max_abs_error": max(decode_checks, default=None),
        "independent_gemm_checks": len(gemm_checks),
        "gemm_checks": gemm_checks,
        "boundaries": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=("w4afp8", "w4a_nvfp4"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-result", type=Path,
                        help="Use the first saved evaluation prompt, including its exact chat rendering.")
    parser.add_argument("--failure-dump-dir", type=Path,
                        help="Save native tensors and accumulator evidence to a new directory on GEMM failure.")
    parser.add_argument(
        "--require-full-coverage", action=argparse.BooleanOptionalAction, default=True,
        help="Require every decoder layer (default); opt out only for partial-model diagnostics.",
    )
    args = parser.parse_args()
    report = audit(args.checkpoint, args.variant, require_full_coverage=args.require_full_coverage,
                   prompt_result=args.prompt_result, failure_dump_dir=args.failure_dump_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"boundaries", "gemm_checks"}}))


if __name__ == "__main__":
    main()
