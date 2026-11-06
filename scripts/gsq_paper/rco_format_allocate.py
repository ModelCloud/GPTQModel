# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Run the RCO budget allocation for AWQ, EXL3 or GGUF.

Three stages, so the expensive measurement can run on one GPU per backend while
the allocation stays on the host:

    # 1. score every option of every projection (per-backend quantizer)
    python scripts/gsq_paper/rco_format_allocate.py measure \
        --method awq --model <dense model> --out scores.json \
        --options 2,3,4 --limit 0

    # 2. solve the exact-budget assignment
    python scripts/gsq_paper/rco_format_allocate.py allocate \
        --method awq --scores scores.json --target-bits 2.5 --out plan.json

    # 3. quantize with the per-module widths the plan selected
    python scripts/gsq_paper/rco_format_allocate.py quantize \
        --method awq --model <dense model> --plan plan.json --out <dir>

All three methods use reconstruction-error allocation. This runner does not
perform differentiable Gumbel/KL interpolation. The score metric records whether
an input Hessian was supplied. Budgets cover packed projection weights and their
metadata/padding, excluding biases and unquantized model state.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=("measure", "allocate", "quantize"))
    parser.add_argument("--method", required=True, choices=("awq", "exl3", "gguf"))
    parser.add_argument("--model", help="dense source model (measure/quantize)")
    parser.add_argument("--scores", help="measure output (allocate)")
    parser.add_argument("--plan", help="allocate output (quantize)")
    parser.add_argument("--out", required=True, help="output path for this stage")
    parser.add_argument("--options", help="comma-separated widths or GGUF aliases")
    parser.add_argument("--target-bits", type=float, default=2.5)
    parser.add_argument("--group-size", type=int, default=None, help="AWQ group size (measure default: 128; otherwise inherited)")
    parser.add_argument(
        "--sym", action=argparse.BooleanOptionalAction, default=None, help="AWQ symmetric codes"
    )
    parser.add_argument("--hessian", help=".pt file: {module name: input second moment}")
    parser.add_argument("--limit", type=int, default=0, help="measure only the first N modules")
    parser.add_argument("--device", default="cuda:0", help="one GPU per backend")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--resolution", type=int, default=0, help="DP budget buckets (0 = auto)")
    parser.add_argument("--calibration", help="text/json calibration corpus for `quantize`")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.resolution < 0:
        parser.error("--limit and --resolution must be nonnegative")
    return args


def parse_options(raw, method):
    from gptqmodel.quantization.rco_formats import (
        AWQ_STAGED_WIDTHS,
        EXL3_STAGED_WIDTHS,
        GGUF_STAGED_ALIASES,
    )

    if not raw:
        return {
            "awq": AWQ_STAGED_WIDTHS,
            "exl3": EXL3_STAGED_WIDTHS,
            "gguf": GGUF_STAGED_ALIASES,
        }[method]
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if method == "gguf":
        return tuple(values)
    return tuple(int(value) for value in values)


def _check_settings(args, settings):
    if args.method == "awq":
        for field in ("group_size", "sym"):
            value = getattr(args, field, None)
            if value is not None and value != settings[field]:
                raise ValueError(f"--{field.replace('_', '-')} disagrees with measured RCO settings")


def quant_config(method, plan, args):
    """Carry measured settings and validated widths into final quantization."""
    from gptqmodel.quantization.config import AWQConfig, EXL3Config, GGUFConfig
    from gptqmodel.utils.torch import resolve_torch_device

    if method != plan.method:
        raise ValueError("RCO plan method does not match --method")
    _check_settings(args, plan.settings)
    config_cls = {"awq": AWQConfig, "exl3": EXL3Config, "gguf": GGUFConfig}[method]
    base = next(iter(plan.dynamic.values()))["bits"]
    return config_cls(bits=base, dynamic=plan.dynamic, **plan.settings, offload_to_disk=False,
                      device=resolve_torch_device(args.device))


def awq_dequantize(weight, bits, group_size, sym):
    """Dequantize one matrix exactly as ``AWQProcessor.pseudo_quantize_tensor`` does."""
    org_shape = weight.shape
    grouped = weight
    if group_size > 0:
        if org_shape[-1] % group_size:
            raise ValueError(
                f"in_features {org_shape[-1]} is not a multiple of group_size {group_size}"
            )
        grouped = weight.reshape(-1, group_size)
    if sym:
        max_int = 2 ** (bits - 1) - 1
        min_int = -(2 ** (bits - 1))
        scale = grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / max_int
        codes = torch.clamp(torch.round(grouped / scale), min_int, max_int)
        return (codes * scale).reshape(org_shape)
    max_int = 2**bits - 1
    max_val = grouped.amax(dim=1, keepdim=True)
    min_val = grouped.amin(dim=1, keepdim=True)
    scale = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scale)).clamp_(0, max_int)
    codes = torch.clamp(torch.round(grouped / scale) + zeros, 0, max_int)
    return ((codes - zeros) * scale).reshape(org_shape)


def measure(args):
    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import AWQConfig, EXL3Config, GGUFConfig
    from gptqmodel.quantization.rco_formats import (
        adapter_for,
        discover_projections,
        relative_reconstruction_error,
    )
    from gptqmodel.utils.torch import resolve_torch_device

    options = parse_options(args.options, args.method)
    group_size = 128 if args.group_size is None else args.group_size
    sym = True if args.sym is None else args.sym
    config_cls = {"awq": AWQConfig, "exl3": EXL3Config, "gguf": GGUFConfig}[args.method]
    settings = {"group_size": group_size, "sym": sym} if args.method == "awq" else {}
    adapter = adapter_for(config_cls(bits=options[0], **settings, offload_to_disk=False), options=options)
    device = resolve_torch_device(args.device)
    dtype = getattr(torch, args.dtype)
    wrapper = GPTQModel.load(args.model, quantize_config=adapter.qcfg, device=device, dtype=dtype)
    model = wrapper.model.to(device).eval()
    names = list(discover_projections(wrapper))
    if args.limit:
        names = names[:args.limit]
    if not names:
        raise SystemExit("no declared quantisable projection found")
    shapes = {name: tuple(model.get_submodule(name).weight.shape) for name in names}
    # Validate packability before expensive quantizer calls.
    adapter.cost_matrix([shapes[name] for name in names])

    hessians = {}
    if args.hessian:
        hessians = torch.load(args.hessian, map_location="cpu", weights_only=True)
        if any(name not in hessians for name in names):
            raise ValueError("Hessian file must cover every measured projection")

    measured = {}
    params = {}
    with torch.inference_mode():
        for position, name in enumerate(names, 1):
            module = model.get_submodule(name)
            # AWQ and GGUF are scored on the host: their quantizers are plain
            # torch/numpy and keeping the matrices on the CPU leaves the GPU free
            # for the backend that needs it (EXL3).
            weight = module.weight.detach().float()
            if args.method != "exl3":
                weight = weight.cpu()
            hessian = hessians.get(name)
            if hessian is not None and args.method != "exl3":
                hessian = hessian.cpu()
            params[name] = int(weight.numel())
            errors = []
            for option in adapter.options():
                if args.method == "awq":
                    restored = awq_dequantize(
                        weight, int(option.payload["bits"]), group_size, sym
                    )
                    errors.append(relative_reconstruction_error(weight, restored, hessian))
                elif args.method == "exl3":
                    errors.append(exl3_error(weight, int(option.payload["bits"]), hessian, device, adapter.qcfg))
                else:
                    restored = gguf_dequantize(weight, option.label)
                    errors.append(relative_reconstruction_error(weight, restored, hessian))
            measured[name] = errors
            print(
                f"  [{position}/{len(names)}] {name} "
                + " ".join(
                    f"{option.label}={error:.3e}"
                    for option, error in zip(adapter.options(), errors)
                ),
                flush=True,
            )

    payload = {
        "schema_version": 1,
        "method": args.method,
        "settings": adapter.settings(),
        "shapes": shapes,
        "score_metric": "relative_hessian" if args.hessian else "relative_frobenius",
        "search": "reconstruction",
        "options": list(adapter.labels()),
        "costs": [float(cost) for cost in adapter.costs()],
        "errors": measured,
        "params": params,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out} ({len(measured)} modules)", flush=True)


def exl3_error(weight, bits, hessian, device, config=None):
    """Score the native reconstruction against the original weight and Hessian."""
    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    from gptqmodel.looper.exllamav3_processor import _EXL3_SIGMA_REG, _OUT_SCALES_TO_ARG
    from gptqmodel.quantization.config import EXL3Config
    from gptqmodel.quantization.rco_formats import relative_reconstruction_error

    config = EXL3Config(bits=bits) if config is None else config
    original_hessian = hessian
    in_features = weight.shape[1]
    if hessian is None:
        hessian = torch.eye(in_features, dtype=torch.float32, device=device)
    else:
        hessian = hessian.to(device=device, dtype=torch.float32)
    # `finalize_capture_H` mutates H in place, so every option needs its own copy.
    h_data = {"H": hessian.clone(), "count": 1, "finalized": False}
    # Mirror `EXL3Processor._build_quant_args` so a measured width scores exactly
    # the way the real quantizer will pack it.
    quant_args = {
        "K": int(bits),
        "devices": [device],
        "apply_out_scales": _OUT_SCALES_TO_ARG.get(config.out_scales, None),
        "sigma_reg": _EXL3_SIGMA_REG,
        "seed": 787,
    }
    if config.codebook in {"mcg", "mul1"}:
        quant_args[config.codebook] = True
    restored, _, _ = quantize_exl3(
        weight.to(device).t().contiguous().clone(), h_data, quant_args, True
    )
    # Native proxy can be -1 on OOM or 0 on fallback. Compare the returned
    # reconstruction with the original Hessian instead of treating sentinels as accuracy.
    return relative_reconstruction_error(weight, restored.t(), original_hessian)


def gguf_dequantize(weight, alias):
    """Round-trip one matrix through ggml's own quantizer."""
    from gptqmodel.nn_modules.qlinear.gguf import (
        _GGUF_TYPE_INFO,
        _dequantize_gguf_tensor_numpy,
        _normalize_gguf_bits,
        _quantize_gguf_tensor_numpy,
    )

    # Resolve the public alias (`q4_k`) to the canonical ggml type (`Q4_K`) and
    # use the same packer/dequantizer the loader uses, including its fallbacks.
    _, tensor_qtype = _normalize_gguf_bits(alias)
    block_size = _GGUF_TYPE_INFO[tensor_qtype]["block_size"]
    columns = weight.shape[1]
    padded = torch.nn.functional.pad(weight, (0, (-columns) % block_size))
    packed = _quantize_gguf_tensor_numpy(padded.cpu().numpy(), tensor_qtype)
    restored = _dequantize_gguf_tensor_numpy(packed, tensor_qtype)
    return torch.from_numpy(np.asarray(restored, dtype=np.float32))[:, :columns]


def _load_score_data(paths):
    """Read labelled score columns with matching settings, counts and shapes."""
    from gptqmodel.quantization.rco_formats import adapter_from_settings

    payloads = [json.loads(Path(item).read_text()) for item in paths.split(",")]
    first = payloads[0]
    required = {"schema_version", "method", "search", "options", "settings", "score_metric", "params", "shapes"}
    names, columns, labels = None, [], []
    for payload in payloads:
        if not required <= payload.keys() or payload["schema_version"] != 1:
            raise ValueError("RCO scores require versioned option labels, settings, parameter counts and shapes")
        if payload["search"] != "reconstruction" or payload["score_metric"] not in {"relative_frobenius", "relative_hessian"}:
            raise ValueError("RCO scores must identify their reconstruction metric")
        for key in ("method", "settings", "score_metric", "params", "shapes"):
            if payload[key] != first[key]:
                raise ValueError(f"RCO score files disagree on {key}")
        table = payload.get("errors", payload.get("losses"))
        if not isinstance(table, dict) or not table:
            raise ValueError("RCO score table must be nonempty")
        names = list(table) if names is None else names
        if set(table) != set(names) or set(payload["params"]) != set(names) or set(payload["shapes"]) != set(names):
            raise ValueError("RCO scores, counts and shapes must cover the same modules")
        option_labels = payload["options"]
        rows = [[table[name]] if isinstance(table[name], (int, float)) else table[name] for name in names]
        errors = torch.tensor(rows, dtype=torch.float64)
        if errors.shape != (len(names), len(option_labels)) or not torch.isfinite(errors).all() or (errors < 0).any():
            raise ValueError("RCO errors must be finite nonnegative values matching the option columns")
        columns.append(errors)
        labels.extend(option_labels)
    adapter = adapter_from_settings(first["method"], first["settings"], labels)
    if adapter.labels() != tuple(labels):
        raise ValueError("RCO option labels must be canonical and unique")
    weights = torch.tensor([first["params"][name] for name in names], dtype=torch.float64)
    shapes = {name: tuple(first["shapes"][name]) for name in names}
    adapter.cost_matrix([shapes[name] for name in names])
    if not torch.isfinite(weights).all() or any(weights[i] != math.prod(shapes[name]) for i, name in enumerate(names)):
        raise ValueError("RCO parameter counts disagree with matrix shapes")
    return names, torch.cat(columns, dim=1), adapter, weights, shapes, first["score_metric"]


def _load_score_matrix(paths):
    names, errors, *_ = _load_score_data(paths)
    return names, errors


def allocate(args):
    from gptqmodel.quantization.rco_formats import allocate as solve

    names, errors, adapter, weights, shapes, metric = _load_score_data(args.scores)
    if adapter.method.value != args.method:
        raise ValueError("RCO score method does not match --method")
    _check_settings(args, adapter.settings())
    if args.options:
        requested = parse_options(args.options, args.method)
        labels = tuple(requested if args.method == "gguf" else
                       [f'{"w" if args.method == "awq" else "k"}{width}' for width in requested])
        if labels != adapter.labels():
            raise ValueError("--options must match the measured score columns in order")
    plan = solve(adapter, -errors, weights, args.target_bits, names=names, shapes=shapes,
                 score_metric=metric, resolution=args.resolution or None)
    Path(args.out).write_text(json.dumps(plan.to_dict(), indent=2))
    print(f"wrote {args.out}: target={plan.target_bits} achieved={plan.achieved_bits:.6f} "
          f"counts={plan.counts} search={plan.search}", flush=True)
    if plan.runtime_note:
        print(f"RUNTIME: {plan.runtime_note}", flush=True)


def quantize(args):
    from gptqmodel import GPTQModel
    from gptqmodel.quantization.rco_formats import RCOPlan, discover_projections

    plan = RCOPlan.from_dict(json.loads(Path(args.plan).read_text()))
    dtype = getattr(torch, args.dtype)
    config = quant_config(args.method, plan, args)
    backend = None
    if args.method == "awq" and any(
        int(entry["bits"]) != 4 for entry in plan.dynamic.values()
    ):
        # The CUDA GEMM/GEMV kernels only unpack 4-bit codes; a non-W4 plan needs
        # the portable torch kernel. `AwqTorchLinear` advertises that backend, so
        # request it explicitly instead of relying on AUTO.
        from gptqmodel.utils.backend import BACKEND

        backend = BACKEND.AWQ_TORCH
        print(f"RUNTIME: {plan.runtime_note}", flush=True)
    wrapper = GPTQModel.load(
        args.model, quantize_config=config, dtype=dtype
    )
    names = discover_projections(wrapper)
    if set(names) != set(plan.names) or not plan.shapes:
        raise ValueError("RCO plan must cover the complete model definition; remeasure without --limit")
    if any(tuple(wrapper.model.get_submodule(name).weight.shape) != plan.shapes[name] for name in names):
        raise ValueError("RCO plan matrix shapes differ from the loaded model")
    wrapper.quantize(calibration=load_calibration(args.calibration), backend=backend)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wrapper.save_quantized(str(out))
    print(f"saved {out} achieved={plan.achieved_bits:.4f} counts={plan.counts}", flush=True)


def load_calibration(path):
    """Read a calibration corpus from a JSON list or a newline-delimited file."""
    if not path:
        return None
    raw = Path(path).read_text()
    if path.endswith(".json"):
        payload = json.loads(raw)
        if isinstance(payload, dict):
            payload = payload.get("texts") or payload.get("text") or []
        return [str(item) for item in payload]
    return [line for line in raw.splitlines() if line.strip()]


def main(argv=None):
    args = parse_args(argv)
    if args.stage in ("measure", "quantize") and not args.model:
        raise SystemExit(f"--model is required for `{args.stage}`")
    if args.stage == "allocate" and not args.scores:
        raise SystemExit("--scores is required for `allocate`")
    if args.stage == "quantize" and not args.plan:
        raise SystemExit("--plan is required for `quantize`")
    {"measure": measure, "allocate": allocate, "quantize": quantize}[args.stage](args)


if __name__ == "__main__":
    main(sys.argv[1:])
