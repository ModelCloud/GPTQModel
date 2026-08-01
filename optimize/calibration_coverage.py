#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Pre-quantization calibration dataset coverage scanner.

Loads a dense causal LM once and profiles activation statistics for target
linear modules (q/k/v_proj, gate/up_proj, down_proj) across one or more
calibration datasets and a held-out reference set.

For each dataset the profile stores:

* the additive per-input-channel Hessian diagonal ``sum(inp.reshape(-1, C).pow(2), dim=0)``
* the running per-channel ``max(abs(inp))``
* a mergeable per-channel percentile sketch exposing p50/p90/p99
* the token count reaching the module

A ``score(P, Ref)`` ranks calibration datasets/mixes by how much they reduce
importance-weighted tail under-coverage against the held-out reference. Unions
are computed for free by adding Hessian diagonals, max'ing per-channel maxima,
and merging percentile sketches.

GPU preflight runs with the standard library only and sets
``CUDA_VISIBLE_DEVICES`` before ``torch`` is imported.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import gc
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import urlretrieve

if TYPE_CHECKING:
    import torch

DATASET_NAME_SEP = ":"


# ---------------------------------------------------------------------------
# stdlib-only GPU preflight (must run before importing torch)
# ---------------------------------------------------------------------------


def _run_nvidia_smi(*arguments: str) -> str:
    result = subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _preflight_physical_gpu(
    physical_index: int,
    allow_busy: bool = False,
    idle_samples: int = 3,
    idle_interval_seconds: float = 1.0,
    max_driver_memory_mib: int = 16,
) -> dict[str, object]:
    """Resolve a physical GPU index to a UUID, verify idle/exclusivity, and set CUDA_VISIBLE_DEVICES."""

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    inventory = _run_nvidia_smi(
        "--query-gpu=index,pci.bus_id,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    )
    target: dict[str, object] | None = None
    for line in inventory.splitlines():
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
        raise RuntimeError(f"Physical GPU {physical_index} not found in nvidia-smi inventory.")

    target_uuid = str(target["uuid"])

    accepted_samples = []
    for sample_index in range(idle_samples):
        snapshot = _run_nvidia_smi(
            "-i", target_uuid,
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        )
        fields = [field.strip() for field in snapshot.split(",")]
        if len(fields) != 2:
            raise RuntimeError(f"Unexpected nvidia-smi snapshot for GPU {target_uuid}: {snapshot!r}")
        memory_used_mib = int(fields[0])
        utilization_pct = int(fields[1])

        try:
            process_output = _run_nvidia_smi(
                "-i", target_uuid,
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            )
        except subprocess.CalledProcessError:
            process_output = ""

        foreign_processes: list[dict[str, object]] = []
        for line in process_output.splitlines():
            proc_fields = [field.strip() for field in line.split(",", 3)]
            if len(proc_fields) != 4:
                continue
            gpu_uuid, pid_str, process_name, _ = proc_fields
            if gpu_uuid != target_uuid:
                continue
            pid = int(pid_str)
            if pid != os.getpid():
                foreign_processes.append({"pid": pid, "process_name": process_name})

        accepted_samples.append({
            "memory_used_mib": memory_used_mib,
            "utilization_pct": utilization_pct,
            "foreign_processes": foreign_processes,
        })

        if not allow_busy and (utilization_pct != 0 or memory_used_mib > max_driver_memory_mib or foreign_processes):
            raise RuntimeError(
                f"GPU idle preflight rejected physical_id={physical_index} uuid={target_uuid}: "
                f"utilization={utilization_pct}% memory={memory_used_mib}MiB "
                f"foreign_processes={foreign_processes} "
                f"(pass --allow-busy-gpu to skip this gate)"
            )

        if sample_index + 1 < idle_samples:
            time.sleep(idle_interval_seconds)

    os.environ["CUDA_VISIBLE_DEVICES"] = target_uuid
    final = accepted_samples[-1]
    print(
        f"[preflight] Accepted physical GPU {physical_index}: {target['name']} "
        f"pci={target['pci_bus_id']} uuid={target_uuid} "
        f"utilization={final['utilization_pct']}% memory={final['memory_used_mib']}MiB "
        f"foreign_processes={len(final['foreign_processes'])} "
        f"samples={idle_samples} max_driver_memory_mib={max_driver_memory_mib} "
        f"allow_busy={allow_busy}"
    )
    return target


def _cpu_supports_fp16() -> bool:
    """Detect AVX-512 FP16 support for native float16 compute on x86 CPUs."""
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        flags = text.split("flags\t\t: ", 1)[1].split("\n", 1)[0]
    except (OSError, IndexError):
        return False
    return "avx512_fp16" in flags


# ---------------------------------------------------------------------------
# CLI / data loading
# ---------------------------------------------------------------------------


def _parse_dataset_spec(spec: str) -> tuple[str, str | None]:
    """Parse ``path``, ``path:name``, or a raw URL into (path, name)."""

    if spec.startswith(("http://", "https://")):
        return spec, None
    parts = spec.split(DATASET_NAME_SEP, 1)
    if len(parts) == 2 and parts[1]:
        return parts[0], parts[1]
    return spec, None


def _maybe_download(url: str, dest_dir: Path) -> str:
    """Download a raw text URL to ``dest_dir`` and return its local path."""

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


def _load_raw_samples(
    dataset_path: str,
    dataset_name: str | None,
    text_separator: str,
    download_dir: Path | None = None,
) -> list[str] | list[list[dict[str, str]]]:
    """Load raw calibration samples from a HF dataset, parquet, or raw text file."""

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
    sample: str | list[dict[str, str]],
    concat_size: int,
    min_length: int,
    apply_chat_template: bool = False,
) -> list[dict[str, list[int]]]:
    """Tokenize one calibration sample into chunks of at most ``concat_size`` tokens."""

    if isinstance(sample, list):
        encoded = tokenizer.apply_chat_template(
            sample,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )
        ids = encoded["input_ids"]
    elif apply_chat_template and getattr(tokenizer, "chat_template", None) is not None:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample}],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
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

    if hasattr(ids, "tolist"):
        ids = ids.tolist()

    chunks: list[dict[str, list[int]]] = []
    for start in range(0, len(ids), concat_size):
        chunk_ids = ids[start:start + concat_size]
        if len(chunk_ids) < min_length:
            continue
        chunks.append({
            "input_ids": chunk_ids,
            "attention_mask": [1] * len(chunk_ids),
        })
    return chunks


def _load_tokenizer(model_path: str, *, trust_remote_code: bool = False):
    """Load the tokenizer, preferring Tokenicer for GPT-QModel-compatible normalization."""

    try:
        from tokenicer import Tokenicer
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        tokenicer = Tokenicer.load(
            model_path,
            model_config=config,
            trust_remote_code=trust_remote_code,
        )
        return tokenicer.tokenizer
    except (ImportError, RuntimeError, ValueError, OSError):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank calibration datasets/mixes by importance-weighted tail "
            "under-coverage vs a held-out reference."
        )
    )
    parser.add_argument("--model", required=True, help="Dense Hugging Face model ID or local path.")
    parser.add_argument(
        "--dataset",
        required=True,
        action="append",
        help="Calibration dataset path, HF dataset `path:name`, or raw text URL/file.",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Held-out reference prompt set (same formats as --dataset).",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for coverage report and JSON.")
    parser.add_argument("--physical-gpu", type=int, help="Physical nvidia-smi GPU index to use.")
    parser.add_argument("--allow-busy-gpu", action="store_true", help="Skip strict idle/exclusivity gate.")
    parser.add_argument("--max-samples", type=int, default=0, help="Max rows per dataset (0 = full).")
    parser.add_argument("--concat-size", type=int, default=2048, help="Max tokens per forward chunk.")
    parser.add_argument("--min-length", type=int, default=10, help="Drop chunks shorter than this.")
    parser.add_argument(
        "--sketch-samples",
        type=int,
        default=256,
        help="Per-channel reservoir size for the mergeable percentile sketch.",
    )
    parser.add_argument(
        "--min-conditional-gain",
        type=float,
        default=0.0,
        help="Stop greedy selection when gain <= this.",
    )
    parser.add_argument(
        "--fallback-threshold",
        default="0.5%",
        help="Fallback threshold (int/float count or 'N%%' of total tokens).",
    )
    parser.add_argument("--text-separator", default="===========", help="Separator for raw text files.")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--greedy-threads",
        type=int,
        default=0,
        help="Threads for the greedy candidate search (0 = auto). Requires PYTHON_GIL=0 for parallelism.",
    )
    parser.add_argument(
        "--target-gain",
        type=float,
        default=None,
        help=(
            "Minimum cumulative score reduction (gain) the selected mix must reach. "
            "Selection continues beyond the floor while marginal gains remain positive."
        ),
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help=(
            "Minimum total tokens the selected mix must reach. "
            "Selection continues beyond the floor while marginal gains remain positive."
        ),
    )
    parser.add_argument(
        "--target-tokens-mode",
        choices=("gain", "gain_per_token"),
        default="gain",
        help=(
            "When --target-tokens is set, pick the next dataset by raw conditional gain "
            "(gain) or by gain per token (gain_per_token)."
        ),
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Wrap raw text samples as a user message and apply the tokenizer chat template.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core data structures: target module grouping, accumulator, profile
# ---------------------------------------------------------------------------

TARGET_SUFFIXES = {
    "q_proj": "q",
    "k_proj": "k",
    "v_proj": "v",
    "gate_proj": "gate",
    "up_proj": "up",
    "down_proj": "down",
    # GPT-2 / OpenAI-GPT style Conv1D modules
    "c_attn": "qkv",
    "c_fc": "fc",
    "c_proj": "proj",
}


@dataclass
class TargetGroup:
    group_id: str
    kind: str
    columns: int
    representative: object | None
    members: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class ActivationAccumulator:
    """Additive, mergeable activation statistics for a single module or shared-input group."""

    columns: int
    max_samples: int
    diag: torch.Tensor = field(init=False)
    max_abs: torch.Tensor = field(init=False)
    sample: torch.Tensor | None = field(init=False, default=None)
    tokens: int = field(init=False, default=0)
    total_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        import torch
        self.diag = torch.zeros(self.columns, dtype=torch.float32)
        self.max_abs = torch.zeros(self.columns, dtype=torch.float32)

    def update(self, x: torch.Tensor) -> None:
        import torch

        if x is None or x.numel() == 0:
            return
        if x.shape[-1] != self.columns:
            raise ValueError(
                f"Expected last dim {self.columns} for {self}, got {x.shape[-1]}"
            )

        flat = x.reshape(-1, self.columns).float()
        n = flat.shape[0]
        if n == 0:
            return

        self.diag += flat.pow(2).sum(dim=0).cpu()

        abs_flat = flat.abs()
        batch_max = abs_flat.max(dim=0)[0].cpu()
        self.max_abs = torch.maximum(self.max_abs, batch_max)

        per_channel = abs_flat.transpose(0, 1).contiguous().cpu().float()
        self.sample = _update_sample(self.sample, per_channel, self.total_count, self.max_samples)
        self.tokens += n
        self.total_count += n

    def quantile(self, q: float) -> torch.Tensor:
        import torch
        if self.sample is None or self.sample.numel() == 0:
            return torch.zeros(self.columns, dtype=torch.float32)
        return torch.quantile(self.sample.to(torch.float32), q, dim=1)

    def quantiles(self, qs: Sequence[float]) -> torch.Tensor:
        import torch
        if self.sample is None or self.sample.numel() == 0:
            return torch.zeros(len(qs), self.columns, dtype=torch.float32)
        return torch.quantile(
            self.sample.to(torch.float32),
            torch.tensor(list(qs), dtype=torch.float32),
            dim=1,
        )

    def merge(self, other: ActivationAccumulator) -> ActivationAccumulator:
        import torch
        merged = ActivationAccumulator(self.columns, self.max_samples)
        merged.diag = self.diag + other.diag
        merged.max_abs = torch.maximum(self.max_abs, other.max_abs)
        merged.tokens = self.tokens + other.tokens
        merged.total_count = self.total_count + other.total_count
        merged.sample = _merge_samples(
            self.sample, self.total_count,
            other.sample, other.total_count,
            self.max_samples,
        )
        return merged


@dataclass
class ModuleProfile:
    module_name: str
    group_id: str
    role: str
    columns: int
    accum: ActivationAccumulator


@dataclass
class DatasetProfile:
    """Profile keyed by (dataset, target module). Groups share accumulators for same-input modules."""

    name: str
    groups: dict[str, ActivationAccumulator] = field(default_factory=dict)
    modules: dict[str, ModuleProfile] = field(default_factory=dict)
    total_tokens: int = 0

    @classmethod
    def from_groups(cls, name: str, groups: list[TargetGroup], max_samples: int) -> DatasetProfile:
        profile = cls(name=name)
        for group in groups:
            accum = ActivationAccumulator(group.columns, max_samples)
            profile.groups[group.group_id] = accum
            for member_name, role in group.members:
                profile.modules[member_name] = ModuleProfile(
                    member_name, group.group_id, role, group.columns, accum
                )
        return profile

    @classmethod
    def empty_like(cls, name: str, ref: DatasetProfile, max_samples: int) -> DatasetProfile:
        profile = cls(name=name, total_tokens=0)
        for group_id, ref_accum in ref.groups.items():
            profile.groups[group_id] = ActivationAccumulator(ref_accum.columns, max_samples)
        for module_name, ref_mod in ref.modules.items():
            profile.modules[module_name] = ModuleProfile(
                module_name,
                ref_mod.group_id,
                ref_mod.role,
                ref_mod.columns,
                profile.groups[ref_mod.group_id],
            )
        return profile

    def update(self, group_id: str, x: torch.Tensor) -> None:
        accum = self.groups.get(group_id)
        if accum is None:
            return
        accum.update(x)

    def merge(self, other: DatasetProfile) -> DatasetProfile:
        new = DatasetProfile(
            name=f"{self.name}+{other.name}",
            total_tokens=self.total_tokens + other.total_tokens,
        )
        for group_id in set(self.groups) | set(other.groups):
            a = self.groups.get(group_id)
            b = other.groups.get(group_id)
            if a is not None and b is not None:
                new.groups[group_id] = a.merge(b)
            elif a is not None:
                new.groups[group_id] = a.merge(ActivationAccumulator(a.columns, a.max_samples))
            else:
                new.groups[group_id] = b.merge(ActivationAccumulator(b.columns, b.max_samples))
        for module_name in set(self.modules) | set(other.modules):
            mod = self.modules.get(module_name) or other.modules[module_name]
            new.modules[module_name] = ModuleProfile(
                module_name,
                mod.group_id,
                mod.role,
                mod.columns,
                new.groups[mod.group_id],
            )
        return new


def _save_profile(profile: DatasetProfile, path: Path) -> None:
    """Persist a ``DatasetProfile`` to disk so we can free it from memory."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, path)


def _load_profile(path: Path) -> DatasetProfile:
    """Load a previously saved ``DatasetProfile``."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


@dataclass
class ScanContext:
    active_profile: DatasetProfile | None = None


# ---------------------------------------------------------------------------
# Reservoir / weighted sample sketch helpers
# ---------------------------------------------------------------------------


def _weighted_sample_topk(values: torch.Tensor, weights: torch.Tensor, k: int) -> torch.Tensor:
    """Weighted reservoir sampling via A-Res keys (``log(u) / w``); picks largest keys."""
    import torch

    u = torch.rand_like(values).clamp_min(1e-12)
    keys = torch.log(u) / weights
    idx = keys.topk(k, dim=1, largest=True, sorted=False).indices
    return torch.gather(values, 1, idx)


def _random_sample(values: torch.Tensor, max_samples: int) -> torch.Tensor | None:
    import torch

    if max_samples <= 0:
        return None
    c, m = values.shape
    if m <= max_samples:
        return values
    weights = torch.ones(c, m, dtype=torch.float32, device=values.device)
    return _weighted_sample_topk(values, weights, max_samples)


def _update_sample(
    sample: torch.Tensor | None,
    values: torch.Tensor,
    total_count: int,
    max_samples: int,
) -> torch.Tensor | None:
    import torch

    if max_samples <= 0:
        return sample
    if sample is None or sample.numel() == 0:
        return _random_sample(values, max_samples)

    c, s = sample.shape
    _, m = values.shape
    new_total = total_count + m
    combined = torch.cat([sample, values], dim=1)
    if new_total <= max_samples:
        return combined

    weights = torch.ones(c, s + m, dtype=torch.float32, device=sample.device)
    if total_count > 0 and s > 0:
        weights[:, :s] = total_count / s
    # new values each represent one observation -> weight 1

    return _weighted_sample_topk(combined, weights, max_samples)


def _merge_samples(
    a: torch.Tensor | None,
    total_a: int,
    b: torch.Tensor | None,
    total_b: int,
    max_samples: int,
) -> torch.Tensor | None:
    import torch

    if a is None or a.numel() == 0:
        if b is None or b.numel() == 0:
            return None
        return b.clone()
    if b is None or b.numel() == 0:
        return a.clone()

    sa = a.shape[1]
    sb = b.shape[1]
    if sa + sb <= max_samples:
        return torch.cat([a, b], dim=1)

    combined = torch.cat([a, b], dim=1)
    c = combined.shape[0]
    weights = torch.ones(c, sa + sb, dtype=torch.float32, device=a.device)
    if total_a > 0 and sa > 0:
        weights[:, :sa] = total_a / sa
    if total_b > 0 and sb > 0:
        weights[:, sa:] = total_b / sb

    return _weighted_sample_topk(combined, weights, max_samples)


# ---------------------------------------------------------------------------
# Model inspection and hooks
# ---------------------------------------------------------------------------


def find_target_groups(model: object) -> list[TargetGroup]:
    """Find target linear modules and group same-input siblings (q/k/v and gate/up, plus GPT-2 c_attn/c_fc)."""

    from torch import nn
    from transformers import Conv1D as TransformersConv1D

    groups: dict[str, TargetGroup] = {}
    for name, module in model.named_modules():
        matched_suffix = None
        for suffix in TARGET_SUFFIXES:
            if name.endswith(suffix):
                matched_suffix = suffix
                break
        if matched_suffix is None:
            continue

        if isinstance(module, nn.Linear):
            columns = module.in_features
        elif isinstance(module, TransformersConv1D):
            columns = module.weight.shape[0]
        else:
            continue

        role = TARGET_SUFFIXES[matched_suffix]
        parent = name.rsplit(".", 1)[0]
        if role in ("q", "k", "v", "qkv"):
            group_id = f"{parent}:attn"
            kind = "attn"
        elif role in ("gate", "up", "fc"):
            group_id = f"{parent}:mlp"
            kind = "mlp"
        else:
            group_id = name
            kind = "down"

        if group_id not in groups:
            groups[group_id] = TargetGroup(group_id, kind, columns, None, [])
        group = groups[group_id]
        group.members.append((name, role))
        if group.representative is None:
            group.representative = module
        if kind == "attn" and role in ("q", "qkv"):
            group.representative = module
        if kind == "mlp" and role in ("gate", "fc"):
            group.representative = module

    return sorted(groups.values(), key=lambda g: g.group_id)


def register_hooks(groups: list[TargetGroup], context: ScanContext) -> list:
    """Register one forward-pre hook per same-input group."""

    handles = []
    for group in groups:
        rep = group.representative
        if rep is None:
            continue

        def make_hook(group_id: str):
            def hook(module: object, inp: tuple, _group_id: str = group_id, _ctx: ScanContext = context) -> None:
                if _ctx.active_profile is None:
                    return
                x = inp[0] if isinstance(inp, tuple) else inp
                _ctx.active_profile.update(_group_id, x)
            return hook

        handle = rep.register_forward_pre_hook(make_hook(group.group_id))
        handles.append(handle)
    return handles


# ---------------------------------------------------------------------------
# Scan, score, merge, greedy
# ---------------------------------------------------------------------------


def scan_dataset(
    model: object,
    tokenizer: object,
    profile: DatasetProfile,
    samples: list[str] | list[list[dict[str, str]]],
    device: object,
    concat_size: int,
    min_length: int,
    apply_chat_template: bool,
) -> None:
    """Run one dataset through the model; hooks populate ``profile``."""

    import torch

    for row_idx, sample in enumerate(samples):
        chunks = _tokenize_sample(tokenizer, sample, concat_size, min_length, apply_chat_template)
        for chunk in chunks:
            input_ids = torch.tensor([chunk["input_ids"]], dtype=torch.long, device=device)
            attention_mask = torch.tensor([chunk["attention_mask"]], dtype=torch.long, device=device)
            profile.total_tokens += int(input_ids.numel())
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            del input_ids, attention_mask
        if (row_idx + 1) % 50 == 0:
            print(f"[scan] {profile.name}: processed {row_idx + 1} rows, {profile.total_tokens} tokens")
    print(f"[scan] {profile.name}: done, {profile.total_tokens} total tokens")


def _prepare_scale_search_importance(diag: torch.Tensor) -> torch.Tensor:
    """Mirror the ACTIVATION branch of ``Quantizer._prepare_scale_search_hessian``."""

    import torch

    importance = diag.detach().to(dtype=torch.float32)
    importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0)
    diagonal_mean = importance.mean()
    valid = torch.isfinite(diagonal_mean) & (diagonal_mean > 0)
    safe_mean = torch.where(valid, diagonal_mean, torch.ones_like(diagonal_mean))
    normalized = importance / safe_mean
    return torch.where(valid, normalized, torch.ones_like(normalized))


def score(profile: DatasetProfile, ref: DatasetProfile) -> float:
    """Sum of per-module importance-weighted tail under-coverage; lower is better.

    Importance is derived from the reference profile's Hessian diagonal so that
    channels with high held-out activation energy dominate the objective. The
    gap is ``relu(ref_p99 - calib_p99) / ref_p99`` per channel.
    """

    import torch

    total = 0.0
    for module_name, ref_mod in ref.modules.items():
        ref_accum = ref_mod.accum
        ref_diag = ref_accum.diag.float()
        ref_p99 = ref_accum.quantile(0.99).float().clamp_min(0.0)
        importance = _prepare_scale_search_importance(ref_diag)

        calib_mod = profile.modules.get(module_name)
        if calib_mod is None or calib_mod.accum.sample is None or calib_mod.accum.total_count == 0:
            calib_p99 = torch.zeros_like(ref_p99)
        else:
            calib_p99 = calib_mod.accum.quantile(0.99).float().clamp_min(0.0)

        denom = ref_p99.clamp_min(1e-12)
        gap = torch.relu(ref_p99 - calib_p99) / denom
        gap = torch.where(ref_p99 > 0, gap, torch.zeros_like(gap))
        total += float((importance * gap).sum())
    return total


def _has_gil_disabled() -> bool:
    try:
        return sys._is_gil_enabled() is False
    except AttributeError:
        return False


def _greedy_worker(
    name: str,
    selected: DatasetProfile,
    candidate: DatasetProfile,
    ref: DatasetProfile,
) -> tuple[str, float, int]:
    merged = selected.merge(candidate)
    s = score(merged, ref)
    return name, s, candidate.total_tokens


def greedy_select(
    profiles: dict[str, DatasetProfile],
    ref: DatasetProfile,
    min_gain: float,
    max_samples: int,
    greedy_threads: int = 0,
    target_gain: float | None = None,
    target_tokens: int | None = None,
    target_tokens_mode: str = "gain",
) -> tuple[DatasetProfile, list[dict[str, object]], float, float, list[str]]:
    """Greedily add the dataset with the largest conditional gain.

    Stopping rules (treats targets as floors, not ceilings):
    - If ``target_gain`` is set, the mix must reach at least that cumulative score
      reduction. Selection continues beyond the floor while the next marginal gain
      is greater than ``min_gain``.
    - If ``target_tokens`` is set, the mix must contain at least that many tokens.
      Selection continues beyond the floor while the next marginal gain is greater
      than ``min_gain``.
    - Otherwise stop when the next best conditional gain <= ``min_gain``.

    The inner candidate evaluation is parallelized with a thread pool so that, when
    PYTHON_GIL=0 and a free-threaded CPython/torch build is used, the CPU-bound
    ``merge``/``score`` work runs concurrently. Workers return only lightweight
    scores so that intermediate merged profiles are not kept alive across all
    candidates.
    """

    selected = DatasetProfile.empty_like("selected", ref, max_samples)
    current_score = score(selected, ref)
    order: list[dict[str, object]] = []
    remaining = list(profiles.keys())
    warnings: list[str] = []

    workers = greedy_threads
    if workers <= 0:
        workers = max(1, min(os.cpu_count() or 1, 16))

    if not _has_gil_disabled() and workers > 1:
        print(
            f"[warn] GIL is enabled; greedy search will use {workers} thread(s) but "
            "CPU-bound torch work will be serialized by the GIL. Run with PYTHON_GIL=0 for parallelism."
        )

    def _target_gain_met(cumulative: float) -> bool:
        return target_gain is None or cumulative >= target_gain

    def _target_tokens_met(total: int) -> bool:
        return target_tokens is None or total >= target_tokens

    cumulative_gain = 0.0
    while remaining:
        best_metric = -float("inf")
        best_name: str | None = None
        best_score = current_score
        best_tokens = 0
        best_gain = -float("inf")

        def _consider(
            name: str,
            s: float,
            cand_tokens: int,
            _score: float = current_score,
        ) -> None:
            nonlocal best_metric, best_name, best_score, best_tokens, best_gain
            gain = _score - s
            if target_tokens is not None and target_tokens_mode == "gain_per_token" and cand_tokens > 0:
                metric = gain / cand_tokens
            else:
                metric = gain
            if metric > best_metric:
                best_metric = metric
                best_name = name
                best_score = s
                best_tokens = cand_tokens
                best_gain = gain

        if len(remaining) <= 1 or workers <= 1:
            for name in remaining:
                merged = selected.merge(profiles[name])
                s = score(merged, ref)
                _consider(name, s, profiles[name].total_tokens)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_name = {
                    executor.submit(_greedy_worker, name, selected, profiles[name], ref): name
                    for name in remaining
                }
                for future in concurrent.futures.as_completed(future_to_name):
                    name, s, cand_tokens = future.result()
                    _consider(name, s, cand_tokens)

        if best_name is None:
            break

        # If no remaining candidate improves the score, the search is finished.
        if best_gain <= 0:
            if target_gain is not None and cumulative_gain < target_gain:
                warnings.append(
                    f"Target cumulative gain floor {target_gain} not reached "
                    f"(reached {cumulative_gain:.6f}); all remaining datasets are redundant."
                )
            if target_tokens is not None and selected.total_tokens < target_tokens:
                warnings.append(
                    f"Target token floor {target_tokens} not reached "
                    f"(reached {selected.total_tokens}); all remaining datasets are redundant."
                )
            break

        gain_met = _target_gain_met(cumulative_gain + best_gain)
        tokens_met = _target_tokens_met(selected.total_tokens + best_tokens)
        if gain_met and tokens_met and best_gain <= min_gain:
            break

        selected = selected.merge(profiles[best_name])
        current_score = best_score
        cumulative_gain += best_gain
        order.append({
            "name": best_name,
            "conditional_gain": best_gain,
            "score_after": current_score,
            "total_tokens": selected.total_tokens,
        })
        remaining.remove(best_name)

    return selected, order, current_score, cumulative_gain, warnings


# ---------------------------------------------------------------------------
# MoE fallback detection
# ---------------------------------------------------------------------------


def _resolve_fallback_threshold(
    threshold_setting: str,
    expected_total_tokens: int | None,
) -> tuple[float | None, bool]:
    """Resolve a ``Fallback.threshold`` value against total token count."""

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from gptqmodel.quantization.config import Fallback
    from gptqmodel.utils.fallback import resolve_threshold

    fb = Fallback(threshold=threshold_setting)
    return resolve_threshold(fb, expected_total_tokens)


def find_fallback_modules(profile: DatasetProfile, threshold_setting: str) -> list[dict[str, object]]:
    """List modules whose token count is below the configured fallback threshold."""

    expected_total = profile.total_tokens
    if expected_total <= 0:
        return []

    threshold_value, is_percent = _resolve_fallback_threshold(threshold_setting, expected_total)
    if threshold_value is None:
        return []

    flagged = []
    for module_name, mod in profile.modules.items():
        if mod.accum.tokens < threshold_value:
            flagged.append({
                "module": module_name,
                "group": mod.group_id,
                "role": mod.role,
                "tokens": mod.accum.tokens,
                "threshold": threshold_value,
                "is_percent": is_percent,
            })
    return sorted(flagged, key=lambda x: (x["group"], x["module"]))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_threshold(threshold: float, is_percent: bool, expected_total: int) -> str:
    if is_percent:
        pct = 100.0 * threshold / max(1, expected_total)
        return f"{pct:.4g}% of {expected_total}"
    return f"{threshold}"


def _build_report(
    config: dict[str, object],
    ref: DatasetProfile,
    profiles: dict[str, DatasetProfile],
    standalone_scores: dict[str, float],
    selected: DatasetProfile,
    greedy_order: list[dict[str, object]],
    final_score: float,
    cumulative_gain: float,
    fallback_by_dataset: dict[str, list[dict[str, object]]],
    fallback_selected: list[dict[str, object]],
    complementarity: list[dict[str, object]],
    timings: dict[str, float],
    warnings: list[str],
) -> dict[str, object]:

    per_dataset = []
    for name in sorted(profiles):
        p = profiles[name]
        fb = fallback_by_dataset[name]
        per_dataset.append({
            "name": name,
            "standalone_score": round(standalone_scores[name], 6),
            "total_tokens": p.total_tokens,
            "fallback_modules": [m["module"] for m in fb],
            "fallback_count": len(fb),
        })

    greedy_rows = []
    for step, item in enumerate(greedy_order, start=1):
        greedy_rows.append({
            "step": step,
            "dataset": item["name"],
            "conditional_gain": round(float(item["conditional_gain"]), 6),
            "score_after": round(float(item["score_after"]), 6),
            "total_tokens": item["total_tokens"],
        })

    score_start = round(final_score + cumulative_gain, 6)
    selected_mix = {
        "datasets": [item["name"] for item in greedy_order],
        "score_start": score_start,
        "score": round(final_score, 6),
        "cumulative_gain": round(cumulative_gain, 6),
        "total_tokens": selected.total_tokens,
        "fallback_modules": [m["module"] for m in fallback_selected],
        "fallback_count": len(fallback_selected),
        "target_gain": config.get("target_gain"),
        "target_tokens": config.get("target_tokens"),
        "target_tokens_mode": config.get("target_tokens_mode"),
    }

    complementarity_rows = [
        {
            "name": item["name"],
            "conditional_gain": round(float(item["conditional_gain"]), 6),
            "verdict": item["verdict"],
        }
        for item in complementarity
    ]

    target_gain = config.get("target_gain")
    target_tokens = config.get("target_tokens")
    target_tokens_mode = config.get("target_tokens_mode", "gain")
    target_lines = []
    if target_gain is not None:
        target_lines.append(f"- target cumulative gain (floor): {target_gain}")
    if target_tokens is not None:
        target_lines.append(f"- target tokens (floor): {target_tokens} (selection mode: {target_tokens_mode})")

    warning_lines: list[str] = []
    if warnings:
        warning_lines.extend(["", "## Warnings", ""])
        for w in warnings:
            warning_lines.append(f"- {w}")

    markdown_lines = [
        "# Calibration coverage report",
        "",
        "## Reference",
        f"- model: {config['model']}",
        f"- reference: {ref.name} ({ref.total_tokens} tokens, {len(ref.modules)} modules)",
    ] + target_lines + warning_lines + [
        "",
        "## Per-dataset standalone scores",
        "",
        "| dataset | tokens | standalone score | fallback modules |",
        "|---------|--------|------------------|------------------|",
    ]
    for row in per_dataset:
        markdown_lines.append(
            f"| {row['name']} | {row['total_tokens']} | {row['standalone_score']:.6f} | "
            f"{row['fallback_count']} |"
        )

    markdown_lines.extend([
        "",
        "## Greedy ranking (ranked by conditional gain)",
        "",
        "| step | dataset | conditional gain | score after | tokens |",
        "|------|---------|------------------|-------------|--------|",
    ])
    for row in greedy_rows:
        markdown_lines.append(
            f"| {row['step']} | {row['dataset']} | {row['conditional_gain']:.6f} | "
            f"{row['score_after']:.6f} | {row['total_tokens']} |"
        )

    markdown_lines.extend([
        "",
        f"## Selected mix: {' -> '.join(selected_mix['datasets']) or '(none)'}",
        f"- score at start: {selected_mix['score_start']:.6f}",
        f"- final score: {selected_mix['score']:.6f}",
        f"- cumulative gain: {selected_mix['cumulative_gain']:.6f}",
        f"- total tokens: {selected_mix['total_tokens']}",
        f"- fallback modules: {selected_mix['fallback_count']}",
    ])

    if fallback_selected:
        markdown_lines.extend([
            "",
            "### Would-fall-back modules/experts",
            "",
            "| module | role | tokens | threshold |",
            "|--------|------|--------|-----------|",
        ])
        for m in fallback_selected:
            threshold_str = _format_threshold(
                float(m["threshold"]), bool(m["is_percent"]), selected.total_tokens
            )
            markdown_lines.append(
                f"| {m['module']} | {m['role']} | {m['tokens']} | {threshold_str} |"
            )
        markdown_lines.extend([
            "",
            "Recommendation: for MoE models consider ``ExpertsRoutingBypass`` or ``ExpertsRoutingOverride`` "
            + "so starved experts receive calibration data, or add datasets that route to those experts.",
        ])

    markdown_lines.extend([
        "",
        "## Complementarity vs selected mix",
        "",
        "| dataset | conditional gain | verdict |",
        "|---------|------------------|---------|",
    ])
    for row in complementarity_rows:
        markdown_lines.append(
            f"| {row['name']} | {row['conditional_gain']:.6f} | {row['verdict']} |"
        )

    markdown_lines.extend([
        "",
        "## Timing",
        "",
        "| stage | seconds |",
        "|-------|---------|",
    ])
    for key, value in timings.items():
        markdown_lines.append(f"| {key} | {value:.3f} |")

    markdown_lines.extend([
        "",
        "## How to read this report",
        "",
        (
            "*Score* is the importance-weighted tail under-coverage versus the held-out reference. "
            "For every target module and input channel we compute:"
        ),
        "",
        "- ``importance = diag / mean(diag)`` (with NaNs clamped to 0) from the reference Hessian diagonal.",
        "- ``gap = relu(ref_p99 - calib_p99) / ref_p99`` per channel.",
        "- ``score = sum(importance * gap)`` over all modules/channels.",
        "",
        (
            "A lower score is better. A score of ``0`` means the calibration data covers every "
            "reference tail. A high score means important reference channels are not seen in the calibration set."
        ),
        "",
        "*Standalone score* is ``score(dataset, ref)`` for each dataset alone.",
        "",
        (
            "*Conditional gain* for a dataset is ``score(mix_before, ref) - score(mix_before + dataset, ref)``: "
            "how much adding that dataset to the current mix reduces the score. Positive values are complementary; "
            "zero or negative values are redundant. The greedy ranking always selects the next dataset by this gain. "
            "*Targets are floors, not ceilings*: once ``--target-gain`` or ``--target-tokens`` is reached, the search "
            "continues as long as the next conditional gain remains positive. The cumulative gain is "
            "``score_start - score_final`` for the selected mix."
        ),
    ])

    return {
        "config": config,
        "reference": {
            "name": ref.name,
            "total_tokens": ref.total_tokens,
            "modules": len(ref.modules),
        },
        "per_dataset": per_dataset,
        "greedy_ranking": greedy_rows,
        "selected_mix": selected_mix,
        "complementarity": complementarity_rows,
        "warnings": warnings,
        "fallback": {
            "threshold_setting": config["fallback_threshold"],
            "selected_mix": fallback_selected,
            "per_dataset": {name: fallback_by_dataset[name] for name in fallback_by_dataset},
        },
        "timings": {k: round(v, 6) for k, v in timings.items()},
        "markdown": "\n".join(markdown_lines),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    start_time = time.perf_counter()
    timings: dict[str, float] = {}

    if args.physical_gpu is not None:
        _preflight_physical_gpu(
            args.physical_gpu,
            allow_busy=args.allow_busy_gpu,
        )

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.set_num_threads(min(16, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)

    if args.physical_gpu is not None:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    dtype_name = args.torch_dtype
    # float16 needs AVX-512 FP16 support on x86 CPUs; bfloat16 works on x86 CPUs
    # with BF16 acceleration (and oneDNN falls back otherwise), so keep the user's dtype.
    if device.type == "cpu" and dtype_name == "float16" and not _cpu_supports_fp16():
        print("[warn] CPU lacks AVX-512 FP16; falling back from float16 to float32")
        dtype = torch.float32
    else:
        dtype = getattr(torch, dtype_name)

    print(f"[load] Loading tokenizer from {args.model} ...")
    tokenizer = _load_tokenizer(args.model, trust_remote_code=args.trust_remote_code)

    print(f"[load] Loading model from {args.model} ...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model_kwargs: dict[str, object] = {
        "config": config,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if device.type == "cpu":
        # Load directly on CPU and avoid a second copy during .to(device).
        model_kwargs["device_map"] = "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    print(f"[load] Moving model to {device} ...")
    model = model.to(device)
    model.eval()
    gc.collect()
    timings["load"] = time.perf_counter() - start_time

    target_groups = find_target_groups(model)
    module_count = sum(len(g.members) for g in target_groups)
    print(f"[model] Found {len(target_groups)} target groups ({module_count} modules)")
    if module_count == 0:
        print(
            "[warn] No target linear modules found. This script supports models with "
            "q/k/v_proj, gate/up/down_proj, or GPT-2-style c_attn/c_fc/c_proj modules."
        )

    context = ScanContext()
    handles = register_hooks(target_groups, context)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scan_start = time.perf_counter()

    profile_paths: list[tuple[str, Path]] = []
    profile_names: set[str] = set()
    for spec in args.dataset:
        dataset_path, dataset_name = _parse_dataset_spec(spec)
        profile_name = dataset_name or Path(dataset_path).name
        base_name = profile_name
        counter = 1
        while profile_name in profile_names:
            profile_name = f"{base_name}_{counter}"
            counter += 1
        profile_names.add(profile_name)
        print(f"[data] Loading calibration dataset `{profile_name}` from {dataset_path} ...")
        samples = _load_raw_samples(dataset_path, dataset_name, args.text_separator, output_dir)
        if args.max_samples > 0:
            samples = samples[: args.max_samples]
            print(f"[data]   using first {len(samples)} rows")
        else:
            print(f"[data]   using all {len(samples)} rows")

        profile = DatasetProfile.from_groups(profile_name, target_groups, args.sketch_samples)
        context.active_profile = profile
        try:
            scan_dataset(
                model,
                tokenizer,
                profile,
                samples,
                device,
                args.concat_size,
                args.min_length,
                args.apply_chat_template,
            )
        finally:
            context.active_profile = None
        profile_path = output_dir / f"{profile_name}.profile.pt"
        _save_profile(profile, profile_path)
        profile_paths.append((profile_name, profile_path))
        del profile, samples
        gc.collect()

    ref_path, ref_name = _parse_dataset_spec(args.reference)
    ref_profile_name = ref_name or f"ref:{Path(ref_path).name}"
    print(f"[data] Loading reference dataset `{ref_profile_name}` from {ref_path} ...")
    ref_samples = _load_raw_samples(ref_path, ref_name, args.text_separator, output_dir)
    if args.max_samples > 0:
        ref_samples = ref_samples[: args.max_samples]

    ref = DatasetProfile.from_groups(ref_profile_name, target_groups, args.sketch_samples)
    context.active_profile = ref
    try:
        scan_dataset(
            model,
            tokenizer,
            ref,
            ref_samples,
            device,
            args.concat_size,
            args.min_length,
            args.apply_chat_template,
        )
    finally:
        context.active_profile = None
    gc.collect()

    for handle in handles:
        handle.remove()
    timings["scan"] = time.perf_counter() - scan_start

    print("[mem] Releasing model before scoring ...")
    del model, tokenizer, ref_samples
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass

    print("[mem] Loading candidate profiles for scoring ...")
    profiles: dict[str, DatasetProfile] = {}
    for name, path in profile_paths:
        profiles[name] = _load_profile(path)

    t = time.perf_counter()
    print("[score] Computing standalone scores ...")
    standalone_scores = {name: score(p, ref) for name, p in profiles.items()}
    timings["standalone_scores"] = time.perf_counter() - t

    t = time.perf_counter()
    print("[score] Running greedy complementarity selection ...")
    selected, greedy_order, final_score, cumulative_gain, warnings = greedy_select(
        profiles,
        ref,
        args.min_conditional_gain,
        args.sketch_samples,
        args.greedy_threads,
        target_gain=args.target_gain,
        target_tokens=args.target_tokens,
        target_tokens_mode=args.target_tokens_mode,
    )
    for w in warnings:
        print(f"[warn] {w}")
    timings["greedy"] = time.perf_counter() - t

    t = time.perf_counter()
    print("[score] Computing per-dataset fallback lists ...")
    fallback_by_dataset = {name: find_fallback_modules(p, args.fallback_threshold) for name, p in profiles.items()}
    fallback_selected = find_fallback_modules(selected, args.fallback_threshold)
    timings["fallback"] = time.perf_counter() - t

    t = time.perf_counter()
    print("[score] Computing complementarity vs selected mix ...")
    greedy_gains = {item["name"]: float(item["conditional_gain"]) for item in greedy_order}
    selected_order = {name: idx for idx, name in enumerate(greedy_gains)}
    complementarity = []
    for name, p in profiles.items():
        if name in selected_order:
            gain = greedy_gains[name]
            verdict = "selected"
        else:
            merged = selected.merge(p)
            s = score(merged, ref)
            gain = final_score - s
            verdict = "complementary" if gain > 0 else "redundant"
        complementarity.append({
            "name": name,
            "conditional_gain": gain,
            "verdict": verdict,
        })
    def _sort_complementarity(x: dict[str, object]) -> tuple[int, float, str]:
        if x["name"] in selected_order:
            return (0, float(selected_order[x["name"]]), str(x["name"]))
        return (1, -float(x["conditional_gain"]), str(x["name"]))

    complementarity.sort(key=_sort_complementarity)
    timings["complementarity"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - start_time

    report_config = {
        "model": args.model,
        "datasets": [f"{path}:{name or ''}" for path, name in map(_parse_dataset_spec, args.dataset)],
        "reference": args.reference,
        "max_samples": args.max_samples,
        "concat_size": args.concat_size,
        "min_length": args.min_length,
        "sketch_samples": args.sketch_samples,
        "min_conditional_gain": args.min_conditional_gain,
        "fallback_threshold": args.fallback_threshold,
        "torch_dtype": args.torch_dtype,
        "device": str(device),
        "apply_chat_template": args.apply_chat_template,
        "target_gain": args.target_gain,
        "target_tokens": args.target_tokens,
        "target_tokens_mode": args.target_tokens_mode,
        "greedy_threads": args.greedy_threads,
    }

    report = _build_report(
        report_config,
        ref,
        profiles,
        standalone_scores,
        selected,
        greedy_order,
        final_score,
        cumulative_gain,
        fallback_by_dataset,
        fallback_selected,
        complementarity,
        timings,
        warnings,
    )

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
