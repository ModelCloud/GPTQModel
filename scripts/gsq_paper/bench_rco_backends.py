# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Compare the differentiable RCO search on Torch MPS and native MLX.

The benchmark uses one dense-decoder MLP projection per layer, resident dense
teacher hidden states, shared token IDs, and a reproducible seed in each
backend's Gumbel generator. Candidate weights and teacher activations are dense
offline search scratch; the packed inference checkpoint is neither loaded nor
modified.

    python scripts/gsq_paper/bench_rco_backends.py \
        --model /path/to/SmolLM2-135M --json-out /tmp/rco-backends.json
"""

import argparse
import gc
import json
import platform
import statistics
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


def _revision(root):
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _source_path(distribution):
    """Return a local checkout path recorded by a direct pip installation."""
    try:
        direct_url = metadata.distribution(distribution).read_text("direct_url.json")
        if not direct_url:
            return None
        payload = json.loads(direct_url)
        url = payload.get("url", "")
        return Path(url.removeprefix("file://")) if url.startswith("file://") else None
    except (metadata.PackageNotFoundError, ValueError):
        return None


def _source_revision(distribution, override):
    if override:
        return _revision(override)
    source = _source_path(distribution)
    return _revision(source) if source else None


def _synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def _median(samples):
    return statistics.median(samples)


def _torch_trial(
    model_path, model_type, token_ids, *, device_name, tokens, steps, draws, warmup, repeats,
    include_probabilities,
):
    from transformers import AutoModelForCausalLM

    from gptqmodel.quantization.rco import (
        InterpolatedLlamaLayers,
        LlamaTeacherKL,
        projected_gumbel_search,
    )

    device = torch.device(device_name)
    torch.manual_seed(7)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True
    )
    model = model.to(device).eval().requires_grad_(False)
    names = [
        f"model.layers.{index}.mlp.down_proj"
        for index in range(model.config.num_hidden_layers)
    ]
    options = {}
    for name in names:
        reference = model.get_submodule(name).weight.detach()
        options[name] = [(reference * 0.75).clone(), reference.clone()]
    weights = torch.tensor(
        [model.get_submodule(name).weight.numel() for name in names], device=device
    )
    costs = torch.tensor([2.0, 3.0], device=device)
    ids = token_ids.to(device)
    objective = LlamaTeacherKL(
        model, [ids], "/tmp/rco-backend-teacher-torch", chunk_tokens=tokens,
        resident_cache=True,
    )
    interpolated = InterpolatedLlamaLayers(
        model, options, weights, costs, control_device=device
    ).setup()
    interpolated.retract(2.5)
    _synchronize(device)

    def run(search=interpolated, loss_fn=objective, token_ids=ids):
        search.alpha.data.zero_()
        search.retract(2.5)
        torch.manual_seed(11)
        started = time.perf_counter()
        probabilities, history = projected_gumbel_search(
            search,
            [0],
            2.5,
            steps=steps,
            gumbel_samples=draws,
            objective=loss_fn,
            batch_weights=[token_ids.numel()],
            log_every=0,
        )
        _synchronize(device)
        return time.perf_counter() - started, probabilities, history

    for _ in range(warmup):
        run()
    samples = []
    final = None
    for _ in range(repeats):
        elapsed, probabilities, history = run()
        samples.append(elapsed)
        final = (probabilities, history)
    probabilities, history = final
    result = {
        "backend": "torch",
        "model_type": model_type,
        "device": str(device),
        "dtype": "bfloat16",
        "layers": len(names),
        "resident_teacher": True,
        "samples_seconds": samples,
        "median_seconds": _median(samples),
        "seconds_per_step": _median(samples) / steps,
        "final_loss": history[-1]["loss"],
        "final_drift": history[-1]["drift"],
    }
    if include_probabilities:
        result["probabilities"] = probabilities.detach().cpu().tolist()
    interpolated.cleanup()
    del interpolated, objective, model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def _mlx_trial(
    model_path, model_type, token_ids, *, tokens, steps, draws, warmup, repeats,
    include_probabilities,
):
    import mlx.core as mx
    from mlx_lm.utils import load_model

    from gptqmodel.quantization.rco_mlx import (
        MLXInterpolatedLlamaLayers,
        MLXTeacherKL,
        _child,
        projected_gumbel_search_mlx,
    )

    model, config = load_model(Path(model_path))
    if config.get("model_type") != model_type:
        raise ValueError(f"Torch and MLX loaded different model types: {model_type!r}, {config.get('model_type')!r}")
    model.eval()
    names = [
        f"model.layers.{index}.mlp.down_proj"
        for index in range(config["num_hidden_layers"])
    ]
    options = {}
    for name in names:
        reference = torch.from_dlpack(_child(model, name).weight).detach().cpu()
        options[name] = [(reference * 0.75).clone(), reference.clone()]
    weights = torch.tensor([options[name][-1].numel() for name in names])
    costs = torch.tensor([2.0, 3.0])
    ids = mx.array(token_ids.numpy())
    objective = MLXTeacherKL(
        model, [ids], "/tmp/rco-backend-teacher-mlx", chunk_tokens=tokens,
        resident_cache=True,
    )
    interpolated = MLXInterpolatedLlamaLayers(model, options, weights, costs)
    interpolated.retract(2.5)
    mx.eval(interpolated.alpha)

    def run(search=interpolated, loss_fn=objective):
        mx.random.seed(11)
        search.alpha = mx.zeros_like(search.alpha)
        search.retract(2.5)
        started = time.perf_counter()
        probabilities, history = projected_gumbel_search_mlx(
            search,
            [0],
            2.5,
            steps=steps,
            gumbel_samples=draws,
            objective=loss_fn,
            batch_weights=[tokens],
            log_every=0,
        )
        mx.eval(probabilities, search.alpha)
        return time.perf_counter() - started, probabilities, history

    for _ in range(warmup):
        run()
    samples = []
    final = None
    for _ in range(repeats):
        elapsed, probabilities, history = run()
        samples.append(elapsed)
        final = (probabilities, history)
    probabilities, history = final
    result = {
        "backend": "mlx",
        "model_type": model_type,
        "device": "metal",
        "dtype": "model-native",
        "layers": len(names),
        "resident_teacher": True,
        "samples_seconds": samples,
        "median_seconds": _median(samples),
        "seconds_per_step": _median(samples) / steps,
        "final_loss": history[-1]["loss"],
        "final_drift": history[-1]["drift"],
    }
    if include_probabilities:
        result["probabilities"] = probabilities.tolist()
    interpolated.cleanup()
    objective.cleanup()
    del interpolated, objective, model
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--backends", default="torch,mlx",
        help="comma-separated torch,mlx; Torch uses MPS on Apple silicon",
    )
    parser.add_argument(
        "--mlx-source",
        help="MLX checkout used for provenance when the wheel has no direct URL",
    )
    parser.add_argument(
        "--mlx-lm-source",
        help="MLX-LM checkout used for provenance; inferred from a direct pip install",
    )
    parser.add_argument(
        "--include-probabilities",
        action="store_true",
        help="include full per-layer probability arrays in the JSON output",
    )
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()
    if any(value < 0 for value in (args.warmup,)):
        parser.error("--warmup must be nonnegative")
    if any(value < 1 for value in (args.tokens, args.steps, args.draws, args.repeats)):
        parser.error("--tokens, --steps, --draws and --repeats must be positive")

    backends = [name.strip() for name in args.backends.split(",") if name.strip()]
    if not backends or any(name not in ("torch", "mlx") for name in backends):
        raise SystemExit("--backends accepts torch and mlx")
    if not Path(args.model).exists():
        raise SystemExit(
            f"Model path does not exist: {args.model}; pass a cached local checkpoint"
        )
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    from gptqmodel.quantization.rco_mlx import rco_mlx_supports_model

    if not rco_mlx_supports_model(config.model_type):
        raise SystemExit(
            "RCO backend benchmark supports dense Llama and Qwen3 models, "
            f"got {config.model_type}"
        )
    torch.manual_seed(7)
    token_ids = torch.randint(1, config.vocab_size, (1, args.tokens), dtype=torch.long)
    results = []
    if "torch" in backends:
        if not torch.backends.mps.is_available():
            raise SystemExit("Torch MPS is unavailable")
        results.append(
            _torch_trial(
                args.model,
                config.model_type,
                token_ids,
                device_name="mps",
                tokens=args.tokens,
                steps=args.steps,
                draws=args.draws,
                warmup=args.warmup,
                repeats=args.repeats,
                include_probabilities=args.include_probabilities,
            )
        )
    if "mlx" in backends:
        results.append(
            _mlx_trial(
                args.model,
                config.model_type,
                token_ids,
                tokens=args.tokens,
                steps=args.steps,
                draws=args.draws,
                warmup=args.warmup,
                repeats=args.repeats,
                include_probabilities=args.include_probabilities,
            )
        )

    by_backend = {result["backend"]: result for result in results}
    if "torch" in by_backend and "mlx" in by_backend:
        by_backend["torch"]["speedup_from_mlx"] = (
            by_backend["torch"]["median_seconds"]
            / by_backend["mlx"]["median_seconds"]
        )
    payload = {
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mlx": _version("mlx"),
            "mlx_lm": _version("mlx-lm"),
            "mlx_revision": _source_revision("mlx", args.mlx_source),
            "mlx_lm_revision": _source_revision("mlx-lm", args.mlx_lm_source),
        },
        "configuration": {
            "model": str(args.model),
            "model_type": config.model_type,
            "tokens": args.tokens,
            "steps": args.steps,
            "gumbel_samples": args.draws,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "teacher_cache": "resident",
            "module": "mlp.down_proj per decoder layer",
            "tokens_fixed_across_backends": True,
            "probabilities_included": args.include_probabilities,
        },
        "results": results,
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
