# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare dense, QVQ W3.5A16, and QVQ W3.5A8 on held-out text.

Perplexity is teacher-forced next-token perplexity. KLD is
``KL(dense || candidate)`` in nats. Top-1 is exact dense/candidate argmax
agreement; Top-5 and Top-10 are mean set overlap fractions. All metrics use
the same shifted, non-padding next-token positions. Model compute defaults to
BF16; W3.5A8 quantizes each targeted linear input to FP8 E4M3 and dequantizes
back to that compute dtype at the kernel boundary. Every arm enables its runtime
KV cache, so W3.5A8 metrics also include its mandatory FP8 E4M3 K/V error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.nn_modules.qvq_fp8_cache import QVQFP8DynamicCache
from gptqmodel.quantization import FORMAT
from scripts.qvq_evaluate import _row_text, validate_evaluation_is_held_out
from scripts.qvq_quantize import DatasetSlice, load_dataset_slice

_A8_CONTRACT = {
    "bits": 8,
    "format": "float8_e4m3fn",
    "scale_method": "dynamic_per_token",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-model", required=True)
    parser.add_argument("--w35-a16-checkpoint", type=Path, required=True)
    parser.add_argument("--w35-a8-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--row-start", type=int, required=True)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--text-column")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="Dense and non-FP8 compute dtype (default: bfloat16)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_contract(path: Path, *, expect_a8: bool) -> dict[str, Any]:
    path = path.expanduser().resolve()
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config does not exist: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        raise TypeError(f"Checkpoint has no quantization_config: {path}")
    if quant.get("method", quant.get("quant_method")) != "qvq":
        raise ValueError(f"Checkpoint is not QVQ: {path}")
    if float(quant.get("bits", -1)) != 3.5:
        raise ValueError(f"Checkpoint is not W3.5: {path}")
    if quant.get("format") != FORMAT.QVQ_V2B2_P32.value:
        raise ValueError(f"Checkpoint is not QVQ P32: {path}")
    activation = quant.get("activation_quantization")
    if expect_a8 and activation != _A8_CONTRACT:
        raise ValueError(f"Checkpoint does not have the required A8 contract: {path}")
    if not expect_a8 and activation is not None:
        raise ValueError(
            f"A16 checkpoint unexpectedly enables activation quantization: {path}"
        )
    return {
        "path": str(path),
        "config_sha256": _sha256(config_path),
        "quantization_config": quant,
    }


def _prediction_rows(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logits and labels for valid teacher-forced next-token positions."""

    if logits.ndim != 3 or input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("Expected logits [B,S,V], input IDs [B,S], and mask [B,S]")
    if logits.shape[:2] != input_ids.shape or input_ids.shape != attention_mask.shape:
        raise ValueError("Logits, input IDs, and attention mask must share [B,S]")
    if logits.shape[1] < 2:
        return logits.new_empty((0, logits.shape[-1])), input_ids.new_empty((0,))
    keep = attention_mask[:, 1:].to(dtype=torch.bool)
    return logits[:, :-1][keep].float(), input_ids[:, 1:][keep]


def _topk_overlap(
    teacher_top10: torch.Tensor, candidate_top10: torch.Tensor, width: int
) -> torch.Tensor:
    teacher = teacher_top10[:, :width]
    candidate = candidate_top10[:, :width]
    return (
        (teacher.unsqueeze(-1) == candidate.unsqueeze(-2))
        .any(dim=-1)
        .float()
        .mean(dim=-1)
    )


def _output_logits(output) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, dict):
        logits = output.get("logits")
    if logits is None and isinstance(output, (tuple, list)) and output:
        logits = output[0]
    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            f"Model forward did not return tensor logits (output type: {type(output).__name__})"
        )
    return logits


class _ArmAccumulator:
    def __init__(self, *, dense_reference: bool = False) -> None:
        self.dense_reference = dense_reference
        self.tokens = 0
        self.nll_sum = 0.0
        self.kld_values: list[torch.Tensor] = []
        self.top1_sum = 0.0
        self.top5_sum = 0.0
        self.top10_sum = 0.0
        self.label_top1_sum = 0.0
        self.label_top5_sum = 0.0
        self.label_top10_sum = 0.0

    def add(
        self,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        dense_logp: torch.Tensor,
        dense_top10: torch.Tensor,
    ) -> None:
        if logits.shape != dense_logp.shape:
            raise ValueError(
                "Candidate and dense prediction rows must have equal shape"
            )
        count = logits.shape[0]
        if count < 1:
            return
        logp = F.log_softmax(logits, dim=-1)
        top10 = logits.topk(min(10, logits.shape[-1]), dim=-1).indices
        self.tokens += count
        self.nll_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
        if self.dense_reference:
            self.kld_values.append(torch.zeros(count, dtype=torch.float32))
        else:
            kld = (dense_logp.exp() * (dense_logp - logp)).sum(dim=-1)
            self.kld_values.append(kld.detach().cpu())
        self.top1_sum += float((dense_top10[:, 0] == top10[:, 0]).sum().item())
        self.top5_sum += float(_topk_overlap(dense_top10, top10, 5).sum().item())
        self.top10_sum += float(_topk_overlap(dense_top10, top10, 10).sum().item())
        self.label_top1_sum += float(
            (top10[:, :1] == labels[:, None]).any(-1).sum().item()
        )
        self.label_top5_sum += float(
            (top10[:, :5] == labels[:, None]).any(-1).sum().item()
        )
        self.label_top10_sum += float(
            (top10[:, :10] == labels[:, None]).any(-1).sum().item()
        )

    def result(self) -> dict[str, Any]:
        if self.tokens < 1:
            raise ValueError("No valid next-token positions were evaluated")
        kld = torch.cat(self.kld_values).double()
        mean_nll = self.nll_sum / self.tokens
        return {
            "tokens": self.tokens,
            "mean_nll": mean_nll,
            "perplexity": math.exp(mean_nll),
            "kld_dense_to_arm_nats": {
                "mean": float(kld.mean().item()),
                "median": float(kld.median().item()),
                "p95": float(torch.quantile(kld, 0.95).item()),
                "p99": float(torch.quantile(kld, 0.99).item()),
                "max": float(kld.max().item()),
            },
            "dense_top1_agreement": self.top1_sum / self.tokens,
            "dense_top5_overlap": self.top5_sum / self.tokens,
            "dense_top10_overlap": self.top10_sum / self.tokens,
            "next_token_top1_accuracy": self.label_top1_sum / self.tokens,
            "next_token_top5_accuracy": self.label_top5_sum / self.tokens,
            "next_token_top10_accuracy": self.label_top10_sum / self.tokens,
        }


def _load_dense(args: argparse.Namespace, dtype: torch.dtype):
    return AutoModelForCausalLM.from_pretrained(
        args.dense_model,
        dtype=dtype,
        device_map={"": args.device},
        attn_implementation="eager",
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    ).eval()


def _load_qvq(args: argparse.Namespace, checkpoint: Path, dtype: torch.dtype):
    return GPTQModel.load(
        str(checkpoint),
        backend=BACKEND.QVQ,
        dtype=dtype,
        device_map={"": args.device},
        attn_implementation="eager",
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.row_start < 0 or args.rows < 1 or args.max_length < 2:
        raise ValueError(
            "row start must be nonnegative, rows positive, and max length >= 2"
        )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {output}")

    a16_contract = _checkpoint_contract(args.w35_a16_checkpoint, expect_a8=False)
    a8_contract = _checkpoint_contract(args.w35_a8_checkpoint, expect_a8=True)
    evaluation = DatasetSlice(
        args.dataset,
        args.dataset_config,
        args.dataset_split,
        args.row_start,
        args.rows,
    )
    for checkpoint in (args.w35_a16_checkpoint, args.w35_a8_checkpoint):
        validate_evaluation_is_held_out(checkpoint, evaluation, allow_overlap=False)
    dataset = load_dataset_slice(evaluation)
    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.dense_model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_started = time.perf_counter()
    dense = _load_dense(args, model_dtype)
    dense_load_seconds = time.perf_counter() - load_started
    load_started = time.perf_counter()
    a16 = _load_qvq(args, args.w35_a16_checkpoint, model_dtype)
    a16_load_seconds = time.perf_counter() - load_started
    load_started = time.perf_counter()
    a8 = _load_qvq(args, args.w35_a8_checkpoint, model_dtype)
    a8_load_seconds = time.perf_counter() - load_started

    a16_layers = [
        module for module in a16.model.modules() if isinstance(module, QVQLinear)
    ]
    a8_layers = [
        module for module in a8.model.modules() if isinstance(module, QVQLinear)
    ]
    if not a16_layers or len(a16_layers) != len(a8_layers):
        raise RuntimeError("W3.5 checkpoints have inconsistent QVQ layer counts")
    if any(
        layer.bits != 3.5 or layer.activation_quantization is not None
        for layer in a16_layers
    ):
        raise RuntimeError("Loaded A16 modules do not satisfy W3.5A16")
    if any(
        layer.bits != 3.5 or layer.activation_quantization is None
        for layer in a8_layers
    ):
        raise RuntimeError("Loaded A8 modules do not satisfy W3.5A8")

    accumulators = {
        "dense": _ArmAccumulator(dense_reference=True),
        "qvq_w3.5_a16": _ArmAccumulator(),
        "qvq_w3.5_a8": _ArmAccumulator(),
    }
    forward_seconds = {name: 0.0 for name in accumulators}
    a8_cache_validation = {
        "rows_validated": 0,
        "cache_class": QVQFP8DynamicCache.__name__,
        "payload_dtypes": set(),
        "scale_dtypes": set(),
        "minimum_storage_ratio_vs_dense": 1.0,
        "maximum_storage_ratio_vs_dense": 0.0,
        "maximum_sequence_length": 0,
        "all_payloads_fp8": True,
        "no_full_precision_residual": True,
    }
    started_all = time.perf_counter()
    for row_index, row in enumerate(dataset):
        text = _row_text(dict(row), tokenizer, args.text_column)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        encoded = {name: value.to(args.device) for name, value in encoded.items()}
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(encoded["input_ids"])

        rows: dict[str, torch.Tensor] = {}
        labels = None
        for name, model in (
            ("dense", dense),
            ("qvq_w3.5_a16", a16),
            ("qvq_w3.5_a8", a8),
        ):
            forward_started = time.perf_counter()
            model_output = model(**encoded, use_cache=True)
            logits = _output_logits(model_output)
            torch.cuda.synchronize(args.device)
            forward_seconds[name] += time.perf_counter() - forward_started
            cache = getattr(model_output, "past_key_values", None)
            if name == "qvq_w3.5_a8":
                if not isinstance(cache, QVQFP8DynamicCache):
                    raise RuntimeError(
                        "W3.5A8 quality evaluation did not return QVQFP8DynamicCache"
                    )
                cache.assert_fp8_storage()
                telemetry = cache.telemetry()
                if (
                    not telemetry["all_payloads_fp8"]
                    or not telemetry["no_full_precision_residual"]
                ):
                    raise RuntimeError(
                        "W3.5A8 quality evaluation retained non-FP8 K/V payloads"
                    )
                ratios = telemetry["storage_ratio_vs_dense"]
                a8_cache_validation["rows_validated"] += 1
                a8_cache_validation["payload_dtypes"].update(
                    telemetry["payload_dtypes"]
                )
                a8_cache_validation["scale_dtypes"].update(telemetry["scale_dtypes"])
                a8_cache_validation["minimum_storage_ratio_vs_dense"] = min(
                    a8_cache_validation["minimum_storage_ratio_vs_dense"], ratios
                )
                a8_cache_validation["maximum_storage_ratio_vs_dense"] = max(
                    a8_cache_validation["maximum_storage_ratio_vs_dense"], ratios
                )
                a8_cache_validation["maximum_sequence_length"] = max(
                    a8_cache_validation["maximum_sequence_length"],
                    *telemetry["sequence_lengths"],
                )
            elif isinstance(cache, QVQFP8DynamicCache):
                raise RuntimeError(f"{name} unexpectedly used the QVQ FP8 cache")
            rows[name], arm_labels = _prediction_rows(
                logits, encoded["input_ids"], attention_mask
            )
            labels = arm_labels if labels is None else labels
            if not torch.equal(labels, arm_labels):
                raise RuntimeError("Evaluation labels differ between arms")
            del logits, model_output, cache

        if labels is None or labels.numel() < 1:
            continue
        dense_logp = F.log_softmax(rows["dense"], dim=-1)
        dense_top10 = (
            rows["dense"].topk(min(10, rows["dense"].shape[-1]), dim=-1).indices
        )
        for name, accumulator in accumulators.items():
            accumulator.add(
                logits=rows[name],
                labels=labels,
                dense_logp=dense_logp,
                dense_top10=dense_top10,
            )
        del rows, labels, dense_logp, dense_top10

        if row_index == 0 or (row_index + 1) % 16 == 0 or row_index + 1 == len(dataset):
            elapsed = time.perf_counter() - started_all
            print(
                f"[quality] rows={row_index + 1}/{len(dataset)} "
                f"tokens={accumulators['dense'].tokens} elapsed={elapsed:.1f}s",
                flush=True,
            )

    results = {name: accumulator.result() for name, accumulator in accumulators.items()}
    dense_ppl = results["dense"]["perplexity"]
    for result in results.values():
        result["perplexity_delta_vs_dense"] = result["perplexity"] - dense_ppl
        result["perplexity_ratio_vs_dense"] = result["perplexity"] / dense_ppl

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    report = {
        "schema": "qvq.activation-quality-comparison.v1",
        "metric_contract": {
            "positions": "teacher-forced shifted non-padding next-token positions",
            "cache": "enabled for every arm; W3.5A8 requires exclusive FP8 E4M3 K/V payload storage",
            "perplexity": "exp(sum next-token NLL / valid next-token count)",
            "kld": "KL(dense || arm) in nats over the full vocabulary",
            "top1": "exact argmax agreement with dense",
            "top5_top10": "mean fraction of the dense top-k set present in the arm top-k set",
            "next_token_accuracy": "ground-truth next token present in arm top-k",
        },
        "dense_model": args.dense_model,
        "checkpoints": {
            "qvq_w3.5_a16": a16_contract,
            "qvq_w3.5_a8": a8_contract,
        },
        "evaluation": {
            **asdict(evaluation),
            "max_length": args.max_length,
            "dataset_sha256": _sha256(Path(args.dataset).expanduser()),
        },
        "results": results,
        "a8_kv_cache_validation": {
            **a8_cache_validation,
            "payload_dtypes": sorted(a8_cache_validation["payload_dtypes"]),
            "scale_dtypes": sorted(a8_cache_validation["scale_dtypes"]),
        },
        "runtime": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "python": platform.python_version(),
            "python_gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "gptqmodel": package_version("gptqmodel"),
            "device": args.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "total_memory_bytes": props.total_memory,
            "dtype": str(model_dtype),
            "backend": BACKEND.QVQ.value,
            "qvq_linear_count_per_checkpoint": len(a16_layers),
            "seconds": {
                "dense_load": dense_load_seconds,
                "qvq_w3.5_a16_load": a16_load_seconds,
                "qvq_w3.5_a8_load": a8_load_seconds,
                "forward": forward_seconds,
                "total_evaluation": time.perf_counter() - started_all,
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
