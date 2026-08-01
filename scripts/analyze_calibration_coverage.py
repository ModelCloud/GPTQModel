#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Pre-quantization activation-coverage scanner for calibration dataset selection.

Loads a dense causal LM once and evaluates one or more calibration dataset
sources. For each candidate source/mix it reports:

- per-layer activation coverage (fraction of hidden dimensions whose maximum
  absolute value exceeds a relative threshold across the calibration data)
- sample redundancy (mean nearest-neighbor cosine similarity of per-row
  pooled hidden-state signatures)
- a greedy row-selection pass that recommends the smallest subset of rows
  which reaches near-maximum coverage, so users can avoid feeding near-
  duplicate calibration data to GPTQ

Use this to choose or combine calibration datasets before running full GPTQ,
especially on models like Qwen3-8B where a small or homogeneous calibration
set can leave large parts of the weight space inactive.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.request import urlretrieve

import torch
import torch.nn.functional as F

try:
    from tokenicer import Tokenicer
except Exception:  # pragma: no cover - Tokenicer is a dependency of gptqmodel
    Tokenicer = None  # type: ignore


DATASET_NAME_SEP = ":"


def _run_nvidia_smi(*arguments: str) -> str:
    result = subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _preflight_physical_gpu(physical_index: int) -> Dict[str, Any]:
    """Resolve a physical GPU index to a UUID and restrict the process to it."""

    output = _run_nvidia_smi(
        "--query-gpu=index,pci.bus_id,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    )
    target = None
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            continue
        index, bus_id, uuid, name, _ = fields
        if int(index) == physical_index:
            target = {
                "physical_index": int(index),
                "pci_bus_id": bus_id,
                "uuid": uuid,
                "name": name,
            }
            break
    if target is None:
        raise RuntimeError(f"Physical GPU {physical_index} not found.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = target["uuid"]
    return target


def _maybe_download(url: str, dest_dir: Path) -> str:
    """Download a raw text URL to dest_dir and return its local path."""

    dest_dir.mkdir(parents=True, exist_ok=True)
    base = Path(url).name or "downloaded_dataset"
    if not base.endswith(".txt"):
        base = base + ".txt"
    local_path = dest_dir / base
    if not local_path.exists():
        print(f"[data] Downloading {url} to {local_path} ...")
        urlretrieve(url, str(local_path))
    else:
        print(f"[data] Using cached {local_path}")
    return str(local_path)


def _parse_dataset_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Parse `path`, `path:name`, or a raw URL into (path, name)."""

    if spec.startswith(("http://", "https://")):
        return spec, None
    parts = spec.split(DATASET_NAME_SEP, 1)
    if len(parts) == 2 and parts[1]:
        return parts[0], parts[1]
    return spec, None


SampleType = Union[str, List[Dict[str, str]]]


def _load_raw_samples(
    dataset_path: str,
    dataset_name: Optional[str],
    text_separator: str,
    download_dir: Optional[Path] = None,
) -> List[SampleType]:
    """Load raw calibration samples from HF dataset, parquet, or raw text.

    A sample is either a raw text string or a list of chat-message dicts.
    """

    if dataset_path.startswith(("http://", "https://")) and download_dir is not None:
        dataset_path = _maybe_download(dataset_path, download_dir)

    path = Path(dataset_path)
    if path.suffix in (".parquet", ".parq"):
        try:
            from datasets import load_dataset
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("The `datasets` library is required") from exc
        ds = load_dataset("parquet", data_files=str(path), split="train")
    elif path.suffix == ".txt":
        content = path.read_text(encoding="utf-8")
        parts = [p.strip() for p in content.split(text_separator)]
        return [p for p in parts if p]
    else:
        try:
            from datasets import load_dataset
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("The `datasets` library is required") from exc
        ds = load_dataset(str(path), name=dataset_name, split="train")

    if "messages" in ds.column_names:
        return [list(row["messages"]) for row in ds]
    if "text" in ds.column_names:
        return [str(row["text"]) for row in ds]
    if "content" in ds.column_names:
        return [str(row["content"]) for row in ds]

    raise ValueError(
        f"Dataset {dataset_path} has unsupported columns {ds.column_names}; "
        "expected one of `text`, `messages`, `content`."
    )


def _tokenize_sample(
    tokenizer,
    sample: SampleType,
    concat_size: int,
    min_length: int,
    apply_chat_template: bool = False,
) -> List[Dict[str, Any]]:
    """Tokenize one calibration sample (raw text or chat messages) into chunks."""

    if isinstance(sample, list):
        # Chat-formatted calibration row; use the model's chat template.
        encoded = tokenizer.apply_chat_template(
            sample,
            tokenize=True,
            add_generation_prompt=False,
        )
        ids = encoded["input_ids"]
    elif apply_chat_template and getattr(tokenizer, "chat_template", None) is not None:
        # Wrap a raw text snippet as a single user turn.
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample}],
            tokenize=True,
            add_generation_prompt=False,
        )
        ids = encoded["input_ids"]
    else:
        encoded = tokenizer(
            sample,
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=False,
        )
        ids = encoded["input_ids"][0].tolist()

    chunks: List[Dict[str, Any]] = []
    for start in range(0, len(ids), concat_size):
        chunk_ids = ids[start : start + concat_size]
        if len(chunk_ids) < min_length:
            continue
        chunks.append({
            "input_ids": chunk_ids,
            "attention_mask": [1] * len(chunk_ids),
        })
    return chunks


class RowStatsAccumulator:
    """Accumulate per-layer max and per-row mean signature from one or more chunks."""

    def __init__(self, num_layers: int, hidden_size: int, signature_dims: int) -> None:
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.signature_dims = signature_dims
        self.layer_max: List[torch.Tensor] = [
            torch.zeros(hidden_size, dtype=torch.float32) for _ in range(num_layers)
        ]
        self.layer_sig_sum: List[torch.Tensor] = [
            torch.zeros(signature_dims, dtype=torch.float32) for _ in range(num_layers)
        ]
        self.token_count = 0

    def update(
        self,
        hidden_states: Sequence[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> None:
        """Update running per-row statistics with hidden states from one forward pass."""

        mask = attention_mask.bool()  # (batch, seq)
        for layer_idx, h in enumerate(hidden_states):
            h = h.float()
            h_masked = h * mask.unsqueeze(-1)

            # Per-channel maximum absolute value across batch and sequence.
            flat = h_masked.abs().reshape(-1, h.size(-1))
            chunk_max = flat.max(dim=0)[0].cpu()
            self.layer_max[layer_idx] = torch.maximum(
                self.layer_max[layer_idx], chunk_max
            )

            # Per-token mean hidden state, pooled to `signature_dims`.
            counts = mask.sum(dim=-1, keepdim=True).to(torch.float32)  # (batch, 1)
            sample_mean = h_masked.sum(dim=1) / counts.clamp(min=1.0)  # (batch, hidden)
            sample_mean = sample_mean.unsqueeze(1)  # (batch, 1, hidden)
            pooled = F.adaptive_avg_pool1d(
                sample_mean, self.signature_dims
            ).squeeze(1)  # (batch, signature_dims)
            weighted = (pooled * counts).sum(dim=0).cpu()
            self.layer_sig_sum[layer_idx] += weighted

        self.token_count += int(mask.sum().item())

    def finalize(self) -> Tuple[List[torch.Tensor], torch.Tensor, int]:
        """Return per-layer max vectors, signature tensor, and token count."""

        sig = torch.stack(
            [self.layer_sig_sum[layer] / max(1, self.token_count) for layer in range(self.num_layers)],
            dim=0,
        )
        return self.layer_max, sig, self.token_count


def _precompute_dataset(
    model,
    tokenizer,
    samples: Sequence[SampleType],
    num_layers: int,
    hidden_size: int,
    signature_dims: int,
    device: torch.device,
    concat_size: int,
    min_length: int,
    apply_chat_template: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run the dense model over each row of one dataset and collect per-row stats."""

    row_max_list: List[List[torch.Tensor]] = []
    row_sig_list: List[torch.Tensor] = []
    row_tokens: List[int] = []

    for row_idx, sample in enumerate(samples):
        chunks = _tokenize_sample(tokenizer, sample, concat_size, min_length, apply_chat_template)
        if not chunks:
            continue
        acc = RowStatsAccumulator(num_layers, hidden_size, signature_dims)
        for chunk in chunks:
            ids_t = torch.tensor([chunk["input_ids"]], dtype=torch.long, device=device)
            mask_t = torch.tensor([chunk["attention_mask"]], dtype=torch.long, device=device)
            with torch.no_grad():
                outputs = model(
                    input_ids=ids_t,
                    attention_mask=mask_t,
                    use_cache=False,
                    output_hidden_states=True,
                )
            acc.update(outputs.hidden_states, mask_t)
            del outputs

        maxes, sig, ntok = acc.finalize()
        row_max_list.append(maxes)
        row_sig_list.append(sig)
        row_tokens.append(ntok)

        if (row_idx + 1) % 50 == 0:
            print(f"[scan]   processed {row_idx + 1} rows ...")

    if not row_max_list:
        return None

    # Transpose from list-of-rows to per-layer tensors.
    num_rows = len(row_max_list)
    row_max_per_layer = [
        torch.stack([row_max_list[i][layer] for i in range(num_rows)], dim=0)
        for layer in range(num_layers)
    ]
    row_sig_per_layer = [
        torch.stack([row_sig_list[i][layer] for i in range(num_rows)], dim=0)
        for layer in range(num_layers)
    ]

    return {
        "row_max": row_max_per_layer,
        "row_sig": row_sig_per_layer,
        "row_tokens": row_tokens,
        "num_rows": num_rows,
    }


def _compute_thresholds(
    row_max: List[torch.Tensor],
    threshold_factor: float,
) -> List[torch.Tensor]:
    """Return per-layer fixed coverage thresholds."""

    return [row_max[layer].max() * threshold_factor for layer in range(len(row_max))]


def _coverage(
    row_max: List[torch.Tensor],
    indices: Sequence[int],
    thresholds: List[torch.Tensor],
) -> Tuple[float, List[float]]:
    """Compute coverage for a selected subset of rows."""

    layer_coverage: List[float] = []
    for layer, threshold in enumerate(thresholds):
        selected_max = row_max[layer][indices].max(dim=0)[0]
        covered = (selected_max > threshold).float().mean().item()
        layer_coverage.append(covered)
    overall = sum(layer_coverage) / len(layer_coverage) if layer_coverage else 0.0
    return overall, layer_coverage


def _redundancy(
    row_sig: List[torch.Tensor],
    indices: Sequence[int],
) -> float:
    """Mean nearest-neighbor cosine similarity of the selected row signatures."""

    redundancies: List[float] = []
    for layer in range(len(row_sig)):
        sig = row_sig[layer][indices]
        if sig.size(0) < 2:
            return 0.0
        sig = F.normalize(sig, dim=-1)
        sim = sig @ sig.T
        eye = torch.eye(sim.size(0), dtype=torch.bool)
        sim = sim.masked_fill(eye, 0.0)
        redundancies.append(float(sim.max(dim=-1)[0].mean()))
    return sum(redundancies) / len(redundancies) if redundancies else 0.0


def _metrics_for_indices(
    row_max: List[torch.Tensor],
    row_sig: List[torch.Tensor],
    row_tokens: List[int],
    indices: Sequence[int],
    thresholds: List[torch.Tensor],
) -> Dict[str, Any]:
    """Coverage, diversity, redundancy, and token count for a row subset."""

    overall, layer_coverage = _coverage(row_max, indices, thresholds)
    redundancy = _redundancy(row_sig, indices)
    diversity = max(0.0, 1.0 - redundancy)
    tokens = sum(row_tokens[i] for i in indices)
    return {
        "coverage": overall,
        "layer_coverage": layer_coverage,
        "diversity": diversity,
        "redundancy": redundancy,
        "score": overall * diversity,
        "total_tokens": tokens,
    }


def _greedy_select(
    row_max: List[torch.Tensor],
    row_sig: List[torch.Tensor],
    row_tokens: List[int],
    thresholds: List[torch.Tensor],
    min_gain: float,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Greedily pick rows that maximize per-layer activation coverage."""

    num_layers = len(row_max)
    num_rows = row_max[0].size(0)
    hidden_size = row_max[0].size(1)

    selected: List[int] = []
    current_max = [torch.zeros(hidden_size, dtype=torch.float32) for _ in range(num_layers)]
    current_coverage = 0.0
    curve: List[Dict[str, Any]] = []

    while len(selected) < num_rows:
        gains = torch.zeros(num_rows)
        for layer in range(num_layers):
            # candidate max if each row is added to the current selection.
            cand_max = torch.maximum(current_max[layer].unsqueeze(0), row_max[layer])  # (num_rows, hidden)
            covered = (cand_max > thresholds[layer]).float().mean(dim=1)  # (num_rows,)
            gains += covered
        gain = gains / num_layers - current_coverage
        if selected:
            gain[selected] = -1.0
        best_gain, best_idx = gain.max(dim=0)
        if best_gain.item() <= min_gain:
            break
        if max_rows is not None and len(selected) >= max_rows:
            break

        best_idx_i = best_idx.item()
        selected.append(int(best_idx_i))
        for layer in range(num_layers):
            current_max[layer] = torch.maximum(current_max[layer], row_max[layer][best_idx_i])
        current_coverage += best_gain.item()
        curve.append({"rows": len(selected), "coverage": round(current_coverage, 4)})

    selected_metrics = _metrics_for_indices(
        row_max, row_sig, row_tokens, selected, thresholds
    )
    return {
        "selected_indices": selected,
        "coverage_curve": curve,
        "final_coverage": round(selected_metrics["coverage"], 4),
        "diversity": round(selected_metrics["diversity"], 4),
        "redundancy": round(selected_metrics["redundancy"], 4),
        "score": round(selected_metrics["score"], 4),
        "total_tokens": selected_metrics["total_tokens"],
    }


def _score_candidate(
    dataset_stats: List[Dict[str, Any]],
    subset_indices: Tuple[int, ...],
    max_samples: int,
    thresholds: Optional[List[torch.Tensor]] = None,
    threshold_factor: float = 1e-3,
    min_gain: float = 1e-4,
) -> Tuple[str, Dict[str, Any]]:
    """Combine selected datasets and produce coverage + greedy row selection."""

    if max_samples == 0:
        per_dataset = None
    else:
        per_dataset = max(1, max_samples // len(subset_indices))

    def _cap(n: int) -> int:
        return min(per_dataset, n) if per_dataset is not None else n

    name_parts: List[str] = []
    for ds_idx in subset_indices:
        stats = dataset_stats[ds_idx]
        n = _cap(stats["num_rows"])
        if n == 0:
            continue
        name_parts.append(f"ds{ds_idx}({n})")

    num_layers = len(dataset_stats[0]["row_max"])
    combined_row_max = [
        torch.cat([stats["row_max"][layer][:_cap(stats["num_rows"])] for stats in [dataset_stats[i] for i in subset_indices]], dim=0)
        for layer in range(num_layers)
    ]
    combined_row_sig = [
        torch.cat([stats["row_sig"][layer][:_cap(stats["num_rows"])] for stats in [dataset_stats[i] for i in subset_indices]], dim=0)
        for layer in range(num_layers)
    ]
    combined_tokens: List[int] = []
    for ds_idx in subset_indices:
        stats = dataset_stats[ds_idx]
        n = _cap(stats["num_rows"])
        combined_tokens.extend(stats["row_tokens"][:n])

    name = " + ".join(name_parts)
    num_rows = combined_row_max[0].size(0)
    if thresholds is None:
        thresholds = _compute_thresholds(combined_row_max, threshold_factor)

    full_metrics = _metrics_for_indices(
        combined_row_max, combined_row_sig, combined_tokens, list(range(num_rows)), thresholds
    )
    greedy = _greedy_select(
        combined_row_max,
        combined_row_sig,
        combined_tokens,
        thresholds,
        min_gain=min_gain,
        max_rows=max_samples if max_samples > 0 else None,
    )

    return name, {
        "num_rows": num_rows,
        "coverage": round(full_metrics["coverage"], 4),
        "diversity": round(full_metrics["diversity"], 4),
        "redundancy": round(full_metrics["redundancy"], 4),
        "score": round(full_metrics["score"], 4),
        "total_tokens": full_metrics["total_tokens"],
        "layer_coverage": full_metrics["layer_coverage"],
        "row_selection": greedy,
    }


def _build_report(candidates: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Build a JSON-serializable report and markdown table."""

    rows = []
    selection_blocks = []
    for name, metrics in candidates:
        rows.append({
            "candidate": name,
            "num_rows": metrics["num_rows"],
            "coverage": metrics["coverage"],
            "diversity": metrics["diversity"],
            "redundancy": metrics["redundancy"],
            "score": metrics["score"],
            "total_tokens": metrics["total_tokens"],
        })
        greedy = metrics["row_selection"]
        curve = greedy["coverage_curve"]
        curve_str = ", ".join(f"({c['rows']}, {c['coverage']})" for c in curve[:10])
        if len(curve) > 10:
            curve_str += f", ... ({len(curve)} total)"
        selection_blocks.append(
            f"**{name}** — selected {len(greedy['selected_indices'])} rows: "
            f"coverage {greedy['final_coverage']}, diversity {greedy['diversity']}, "
            f"tokens {greedy['total_tokens']}\n"
            f"Coverage curve (rows, coverage): {curve_str}"
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    best = rows[0] if rows else None

    markdown_lines = [
        "# Calibration coverage report",
        "",
        "| candidate | rows | coverage | diversity | redundancy | score | tokens |",
        "|-----------|------|----------|-----------|------------|-------|--------|",
    ]
    for row in rows:
        markdown_lines.append(
            f"| {row['candidate']} | {row['num_rows']} | {row['coverage']:.4f} | {row['diversity']:.4f} "
            f"| {row['redundancy']:.4f} | {row['score']:.4f} | {row['total_tokens']} |"
        )
    if best:
        markdown_lines.extend(["", f"**Recommended candidate:** `{best['candidate']}`"])

    if selection_blocks:
        markdown_lines.extend(["", "## Greedy row selection", ""])
        markdown_lines.extend(selection_blocks)

    return {
        "candidates": rows,
        "best": best,
        "selection": [c[1]["row_selection"] for c in candidates],
        "markdown": "\n".join(markdown_lines),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare calibration dataset mixes by activation coverage and diversity."
    )
    parser.add_argument("--model", required=True, help="Dense Hugging Face model ID or local path.")
    parser.add_argument(
        "--dataset",
        required=True,
        action="append",
        help="Calibration dataset path, HF dataset `path:name`, or raw text URL/file.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for coverage report and JSON.")
    parser.add_argument("--physical-gpu", type=int, help="Physical nvidia-smi GPU index to use.")
    parser.add_argument("--max-samples", type=int, default=0, help="Max rows per dataset to scan and select (0 = use full dataset).")
    parser.add_argument("--concat-size", type=int, default=2048, help="Max tokens per forward chunk for each row.")
    parser.add_argument("--min-length", type=int, default=10, help="Drop chunks shorter than this.")
    parser.add_argument("--signature-dims", type=int, default=64, help="Dimension of per-row pooled signature.")
    parser.add_argument("--max-subset-size", type=int, default=3, help="Largest dataset combination to evaluate.")
    parser.add_argument("--coverage-threshold", type=float, default=1e-3, help="Coverage threshold as fraction of per-layer peak.")
    parser.add_argument("--coverage-gain-tolerance", type=float, default=1e-4, help="Stop greedy row selection when coverage gain falls below this.")
    parser.add_argument("--text-separator", default="===========", help="Separator for raw text files with multiple documents.")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--apply-chat-template", action="store_true", help="Wrap raw text samples as a user message and apply the tokenizer chat template.")
    return parser.parse_args()


def _load_tokenizer(model_path: str, *, trust_remote_code: bool = False):
    """Load the tokenizer through Tokenicer for GPT-QModel-compatible normalization."""

    if Tokenicer is None:
        raise RuntimeError(
            "Tokenicer is not installed. It is a dependency of gptqmodel; "
            "ensure the active environment is the GPT-QModel venv."
        )

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    tokenicer = Tokenicer.load(
        model_path,
        model_config=config,
        trust_remote_code=trust_remote_code,
    )
    return tokenicer.tokenizer


def main() -> int:
    args = _parse_args()

    if args.physical_gpu is not None:
        _preflight_physical_gpu(args.physical_gpu)

    from transformers import AutoConfig, AutoModelForCausalLM

    dtype = getattr(torch, args.torch_dtype)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[load] Loading tokenizer from {args.model} via Tokenicer ...")
    tokenizer = _load_tokenizer(args.model, trust_remote_code=args.trust_remote_code)

    print(f"[load] Loading model from {args.model} ...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    print("[load] Moving model to accelerator ...")
    model = model.to(device)
    model.eval()

    hidden_size = int(model.config.hidden_size)
    num_layers = int(model.config.num_hidden_layers) + 1  # embedding + all layer outputs

    # Pre-compute per-row statistics for every dataset.
    dataset_specs = [_parse_dataset_spec(spec) for spec in args.dataset]
    dataset_stats: List[Dict[str, Any]] = []
    output_dir = Path(args.output_dir)
    for ds_idx, (path, name) in enumerate(dataset_specs):
        print(f"[data] Loading dataset {ds_idx}: {path} (name={name}) ...")
        samples = _load_raw_samples(path, name, args.text_separator, download_dir=output_dir)
        if args.max_samples > 0:
            sample_set = samples[: args.max_samples]
            print(f"[data]   got {len(samples)} rows; scanning first {args.max_samples} ...")
        else:
            sample_set = samples
            print(f"[data]   got {len(samples)} rows; scanning all rows ...")
        stats = _precompute_dataset(
            model,
            tokenizer,
            sample_set,
            num_layers,
            hidden_size,
            args.signature_dims,
            device,
            args.concat_size,
            args.min_length,
            apply_chat_template=args.apply_chat_template,
        )
        if stats is None:
            print(f"[scan]   dataset {ds_idx} produced no valid rows")
            continue
        print(f"[scan]   dataset {ds_idx}: {stats['num_rows']} rows, {sum(stats['row_tokens'])} tokens")
        dataset_stats.append(stats)

    if not dataset_stats:
        raise RuntimeError("No usable calibration data found.")

    # Evaluate all requested dataset combinations.
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    subsets: List[Tuple[int, ...]] = []
    for size in range(1, min(args.max_subset_size, len(dataset_stats)) + 1):
        subsets.extend(itertools.combinations(range(len(dataset_stats)), size))

    for subset in subsets:
        name, metrics = _score_candidate(
            dataset_stats,
            subset,
            args.max_samples,
            threshold_factor=args.coverage_threshold,
            min_gain=args.coverage_gain_tolerance,
        )
        print(
            f"[scan] Candidate `{name}`: full coverage={metrics['coverage']:.4f}, "
            f"selected {len(metrics['row_selection']['selected_indices'])} rows -> "
            f"coverage={metrics['row_selection']['final_coverage']:.4f}"
        )
        candidates.append((name, metrics))

    report = _build_report(candidates)
    report["config"] = {
        "model": args.model,
        "datasets": [f"{p}:{n or ''}" for p, n in dataset_specs],
        "max_samples": args.max_samples,
        "concat_size": args.concat_size,
        "signature_dims": args.signature_dims,
        "coverage_threshold": args.coverage_threshold,
        "coverage_gain_tolerance": args.coverage_gain_tolerance,
        "dtype": args.torch_dtype,
        "device": str(device),
        "apply_chat_template": args.apply_chat_template,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "coverage_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "coverage_report.md").write_text(
        report["markdown"] + "\n",
        encoding="utf-8",
    )

    print(report["markdown"])
    print(f"\nWrote coverage_report.json/md to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
