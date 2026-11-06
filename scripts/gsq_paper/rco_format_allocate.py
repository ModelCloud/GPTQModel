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

Fidelity
--------
AWQ is scored with the affine quantizer the processor itself uses, and its
staged arms are dense, so the same scores feed the paper's interpolation search.
EXL3 and GGUF store a trellis and ggml blocks respectively; neither is a linear
function of a dense weight, so both are scored with the Hessian-weighted
reconstruction error their own quantizer reports. ``plan.json`` records which
search produced the assignment in its ``search`` field.

See ``gptqmodel/quantization/rco_formats.py`` for the adapters and the
GSQ / exact-budget references (arXiv 2604.18556 Sec. 4.2, arXiv 2605.00649
Secs. 2-3).
"""

import argparse
import json
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
    parser.add_argument("--group-size", type=int, default=128, help="AWQ group size")
    parser.add_argument(
        "--sym", action=argparse.BooleanOptionalAction, default=True, help="AWQ symmetric codes"
    )
    parser.add_argument("--hessian", help=".pt file: {module name: input second moment}")
    parser.add_argument("--limit", type=int, default=0, help="measure only the first N modules")
    parser.add_argument("--device", default="cuda:0", help="one GPU per backend")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--resolution", type=int, default=0, help="DP budget buckets (0 = auto)")
    parser.add_argument("--calibration", help="text/json calibration corpus for `quantize`")
    return parser.parse_args(argv)


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


def quant_config(method, plan, args):
    """Build the backend config that carries the plan's per-module widths."""
    from gptqmodel.quantization.config import AWQConfig, EXL3Config, GGUFConfig

    dynamic = {name: dict(entry) for name, entry in plan.dynamic.items()}
    if method == "awq":
        base = min(int(entry["bits"]) for entry in dynamic.values())
        return AWQConfig(bits=base, group_size=args.group_size, sym=args.sym, dynamic=dynamic)
    if method == "exl3":
        base = min(int(entry["bits"]) for entry in dynamic.values())
        return EXL3Config(bits=float(base), dynamic=dynamic)
    base = next(iter(dynamic.values()))["bits"]
    return GGUFConfig(bits=base, dynamic=dynamic)


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

    options = parse_options(args.options, args.method)
    if args.method == "awq":
        adapter = adapter_for(AWQConfig(bits=4, group_size=args.group_size, sym=args.sym))
    elif args.method == "exl3":
        adapter = adapter_for(EXL3Config(bits=float(min(options))))
    else:
        adapter = adapter_for(GGUFConfig(bits=options[0]))
        adapter.aliases = tuple(options)

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    wrapper = GPTQModel.load(args.model, device=device, dtype=dtype)
    model = wrapper.model.to(device).eval()

    names = list(discover_projections(model))
    if args.limit:
        names = names[: args.limit]
    if not names:
        raise SystemExit("no quantisable projection found")

    hessians = {}
    if args.hessian:
        hessians = torch.load(args.hessian, map_location="cpu", weights_only=True)

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
                        weight, int(option.payload["bits"]), args.group_size, args.sym
                    )
                    errors.append(relative_reconstruction_error(weight, restored, hessian))
                elif args.method == "exl3":
                    errors.append(exl3_error(weight, int(option.payload["bits"]), hessian, device))
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
        "method": args.method,
        "search": adapter.search,
        "options": list(adapter.labels()),
        "costs": [float(cost) for cost in adapter.costs()],
        "group_size": args.group_size,
        "sym": bool(args.sym),
        "errors": measured,
        "params": params,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out} ({len(measured)} modules)", flush=True)


def exl3_error(weight, bits, hessian, device):
    """Score one EXL3 width with the quantizer's own Hessian-weighted proxy."""
    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    from gptqmodel.looper.exllamav3_processor import _EXL3_SIGMA_REG, _OUT_SCALES_TO_ARG

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
        "apply_out_scales": _OUT_SCALES_TO_ARG.get("auto", None),
        "sigma_reg": _EXL3_SIGMA_REG,
        "seed": 787,
        "mcg": True,
    }
    _, proxy, _ = quantize_exl3(
        weight.to(device).t().contiguous(), h_data, quant_args, False
    )
    return float(proxy)


def gguf_dequantize(weight, alias):
    """Round-trip one matrix through ggml's own quantizer."""
    from gptqmodel.nn_modules.qlinear.gguf import (
        _dequantize_gguf_tensor_numpy,
        _normalize_gguf_bits,
        _quantize_gguf_tensor_numpy,
    )

    # Resolve the public alias (`q4_k`) to the canonical ggml type (`Q4_K`) and
    # use the same packer/dequantizer the loader uses, including its fallbacks.
    _, tensor_qtype = _normalize_gguf_bits(alias)
    packed = _quantize_gguf_tensor_numpy(weight.numpy(), tensor_qtype)
    restored = _dequantize_gguf_tensor_numpy(packed, tensor_qtype)
    return torch.from_numpy(np.asarray(restored, dtype=np.float32))


def _load_score_matrix(paths):
    """Return `(names, errors[n_modules, n_options])` from one or more score files.

    A single `measure` run writes the whole matrix. The EXL3 arm sweep instead
    writes one file per width, each holding the same module names under
    `losses`; passing every staged arm as a comma-separated list rebuilds the
    same matrix without re-measuring.
    """
    payloads = [json.loads(Path(item).read_text()) for item in paths.split(",")]
    if len(payloads) == 1:
        table = payloads[0].get("errors") or payloads[0]["losses"]
        names = list(table)
        return names, torch.tensor([table[name] for name in names], dtype=torch.float32)
    names = list(payloads[0].get("errors") or payloads[0]["losses"])
    columns = []
    for payload in payloads:
        table = payload.get("errors") or payload["losses"]
        if set(table) != set(names):
            raise SystemExit("staged arm score files cover different module sets")
        columns.append([float(table[name]) for name in names])
    return names, torch.tensor(columns, dtype=torch.float32).t().contiguous()


def allocate(args):
    from gptqmodel.quantization.config import AWQConfig, EXL3Config, GGUFConfig
    from gptqmodel.quantization.rco_formats import adapter_for
    from gptqmodel.quantization.rco_formats import allocate as solve

    # `measure` writes `errors`; the EXL3 arm sweep writes the same data as
    # `losses`. Accept both so a plan can be rebuilt from either source.
    names, errors = _load_score_matrix(args.scores)
    payload = json.loads(Path(args.scores.split(",")[0]).read_text())
    if args.method == "awq":
        adapter = adapter_for(
            AWQConfig(
                bits=4,
                group_size=int(payload.get("group_size", args.group_size)),
                sym=bool(payload.get("sym", args.sym)),
            )
        )
    elif args.method == "exl3":
        adapter = adapter_for(EXL3Config(bits=float(min(parse_options(args.options, "exl3")))))
    else:
        adapter = adapter_for(GGUFConfig(bits="q4_k"))
        adapter.aliases = tuple(payload["options"])

    # RCO maximises the score, so a lower reconstruction error is a higher score.
    scores = -errors
    # The budget is a ratio over the parameters that are actually quantized, so a
    # 5.2M k_proj must not weigh the same as a 209.7M mlp projection. Without
    # per-module counts the solver would think it can buy accuracy cheaply by
    # upgrading every small matrix and downgrading the MLPs, which is the
    # opposite of what the parameter budget allows.
    params = payload.get("params")
    if params:
        weights = torch.tensor([params[name] for name in names], dtype=torch.float32)
    else:
        print(
            "WARN: scores file has no 'params'; falling back to uniform module weights. "
            "Re-run the `measure` stage to get a parameter-weighted allocation.",
            file=sys.stderr,
            flush=True,
        )
        weights = torch.ones(len(names), dtype=torch.float32)
    plan = solve(
        adapter,
        scores,
        weights,
        args.target_bits,
        names=names,
        resolution=args.resolution or None,
    )
    Path(args.out).write_text(json.dumps(plan.to_dict(), indent=2))
    print(
        f"wrote {args.out}: target={plan.target_bits} achieved={plan.achieved_bits:.4f} "
        f"counts={plan.counts} search={plan.search}",
        flush=True,
    )
    if plan.runtime_note:
        print(f"RUNTIME: {plan.runtime_note}", flush=True)


def quantize(args):
    from gptqmodel import GPTQModel
    from gptqmodel.quantization.rco_formats import RCOPlan

    plan = RCOPlan.from_dict(json.loads(Path(args.plan).read_text()))
    device = torch.device(args.device)
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
        args.model, quantize_config=config, device=device, dtype=dtype
    )
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
