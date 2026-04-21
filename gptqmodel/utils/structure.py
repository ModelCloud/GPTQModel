# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""
Usage:
  # Focus on mixers/MLP, show params/buffers, depth limit
  `filter` "self_attn|linear_attn|mlp" --show-params --max-depth 4

  # Control MoE collapsing:
  #   - collapse any ModuleList whose name matches the regex and whose children are numeric
  #   - show the first N experts and collapse the rest
  `experts-regex` "(^|\\.)experts($|\\.)" --experts-show 1
  `no-collapse-experts`  # disable collapsing

Flags semantics:
- show_params/show_buffers: print detailed lines for parameters and (optionally) buffers.
- show_all: print detailed lines for BOTH parameters AND buffers (names included), overriding the other two.
- filter_regex: planned hook (compiled but not applied; keep for future filtering of node names).

Notes:
- Detects shared submodules and avoids re-printing them.
- Collapsing is generic: any numeric-indexed ModuleList whose qualified name matches `experts-regex`.
- Large layer stacks are capped to the first 4 children by default.
"""

import atexit
import copy
import gc
import inspect
import json
import os
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import pcre
import torch
from safetensors import safe_open
from torch import nn

from ..utils.logger import setup_logger


# =========================
#   ANSI color helpers
# =========================
RESET = "\033[0m"
DIM = "\033[2m"
FG_GRAY = "\033[90m"
FG_CYAN = "\033[36m"
FG_YELLOW = "\033[33m"

log = setup_logger()

def _maybe(s: str, code: str, *, color: bool) -> str:
    return f"{code}{s}{RESET}" if color else s

# =========================
#   Dtype size registry
# =========================
_DTYPE_BYTES: Dict[object, float] = {
    torch.float32: 4, torch.float: 4,
    torch.bfloat16: 2,
    torch.float16: 2, torch.half: 2,
    torch.uint8: 1, torch.int8: 1,
    torch.int16: 2, torch.short: 2,
    torch.int32: 4, torch.int: 4,
    torch.bool: 1,
}
for _dtype_name in (
    *[
        name
        for name in (
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e4m3fnuz",
            "float8_e5m2fnuz",
            "float8_e8m0fnu",
        )
        if hasattr(torch, name)
    ],
    *[name for name in ("float4_e2m1fn_x2",) if hasattr(torch, name)],
):
    _DTYPE_BYTES[getattr(torch, _dtype_name)] = 1

class _FakeDType:
    """Sentinel dtype for experimental 4-bit formats."""
    def __init__(self, name: str): self.name = name
    def __repr__(self): return self.name
    def __str__(self): return self.name

MXFP4 = _FakeDType("MXFP4")
NVFP4 = _FakeDType("NVFP4")
_DTYPE_BYTES[MXFP4] = 0.5
_DTYPE_BYTES[NVFP4] = 0.5

def _elem_size(d) -> float:
    return _DTYPE_BYTES.get(d, 0.0)

# =========================
#   Formatting helpers
# =========================
def human_count(n: int) -> str:
    if n < 0: return str(n)
    if n < 1_000: return str(n)
    for label, scale in (("T", 1_000_000_000_000),
                         ("B", 1_000_000_000),
                         ("M", 1_000_000),
                         ("K", 1_000)):
        if n >= scale:
            return f"{n/scale:.2f}{label}"
    return str(n)

def _human_bytes(n: float) -> str:
    if n <= 0: return "0B"
    for unit in ["B","KB","MB","GB","TB","PB"]:
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"

# =========================
#   Counting & summaries
# =========================
def _param_summary(mod: nn.Module, *, recurse: bool = True) -> Tuple[int, int]:
    if recurse:
        p = sum(p.numel() for p in mod.parameters())
        b = sum(b.numel() for b in mod.buffers())
    else:
        p = sum(p.numel() for _, p in mod.named_parameters(recurse=False))
        b = sum(b.numel() for _, b in mod.named_buffers(recurse=False))
    return p, b

def _counts_for_module(mod: nn.Module) -> Tuple[int, int]:
    return _param_summary(mod, recurse=False)

def _summarize_module_tensors(mod: nn.Module, *, recurse: bool = False):
    dev_counts: Dict[str, int] = {}
    dtype_set: Set[object] = set()
    total_elems = 0
    alloc_bytes = 0.0
    est_bytes = 0.0

    def _iter_tensors() -> Iterable[torch.Tensor]:
        for _, t in mod.named_parameters(recurse=recurse): yield t
        for _, t in mod.named_buffers(recurse=recurse): yield t

    for t in _iter_tensors():
        if t is None: continue
        is_meta = bool(getattr(t, "is_meta", False) or (hasattr(t, "device") and t.device.type == "meta"))
        dev_key = "meta" if is_meta else (str(t.device) if hasattr(t, "device") else "-")
        n = t.numel()
        total_elems += n
        dev_counts[dev_key] = dev_counts.get(dev_key, 0) + n
        dt = getattr(t, "dtype", None)
        dtype_set.add(dt)
        esize = (t.element_size() if (not is_meta and hasattr(t, "element_size")) else _elem_size(dt)) or 0.0
        if not is_meta:
            alloc_bytes += n * esize
        est_bytes += n * esize

    if not dev_counts:
        device_str = "-"
    elif len(dev_counts) == 1:
        device_str = next(iter(dev_counts.keys()))
    else:
        top = sorted(dev_counts.items(), key=lambda kv: kv[1], reverse=True)[:2]
        device_str = "mixed[" + ", ".join(k for k, _ in top) + ("" if len(dev_counts) <= 2 else ", …") + "]"

    if not dtype_set:
        dtype_str = "-"
    elif len(dtype_set) == 1:
        d = next(iter(dtype_set))
        dtype_str = (str(d).replace("torch.", "")) if d is not None else "-"
    else:
        dnames = [(str(d).replace("torch.", "")) if d is not None else "-" for d in list(dtype_set)[:3]]
        dtype_str = "mixed[" + ", ".join(dnames) + ("" if len(dtype_set) <= 3 else ", …") + "]"

    return device_str, dtype_str, total_elems, alloc_bytes, est_bytes

def _annotate(mod: nn.Module, *, color: bool = True) -> str:
    device_str, dtype_str, _n, alloc_b, est_b = _summarize_module_tensors(mod, recurse=False)
    base = f"[{device_str} | {dtype_str} | ~{_human_bytes(alloc_b)}]"
    if est_b > alloc_b:
        base = base[:-1] + f" (est~{_human_bytes(est_b)})]"
    return _maybe(base, DIM, color=color)

# =========================
#   Printing functions
# =========================
def _format_line(prefix: str, trunk: str, qual_name: str, mod: nn.Module, show_counts: bool, color: bool) -> str:
    cls = mod.__class__.__name__
    left = _maybe(prefix + trunk, FG_GRAY, color=color)
    name = _maybe(qual_name, FG_CYAN, color=color)
    klass = _maybe(cls, DIM, color=color)
    if show_counts:
        p, b = _counts_for_module(mod)
        counts = _maybe(f"(P={human_count(p)} B={human_count(b)})", FG_YELLOW, color=color)
        return f"{left}{name}: {klass}  {counts}"
    else:
        return f"{left}{name}: {klass}"

def _print_params(indent: str, mod: nn.Module, *, include_buffers: bool, color: bool):
    def _line(kind: str, name: str, t: torch.Tensor) -> str:
        dev = "meta" if (getattr(t, "is_meta", False) or t.device.type == "meta") else str(t.device)
        dt = str(getattr(t, "dtype", None)).replace("torch.", "")
        esize = t.element_size() if (not getattr(t, "is_meta", False) and hasattr(t, "device") and t.device.type != "meta") else _elem_size(getattr(t, "dtype", None))
        sizeb = t.numel() * (esize or 0.0)
        kind_c = _maybe(kind, FG_CYAN, color=color)
        name_c = _maybe(name, FG_GRAY, color=color)
        size_y = _maybe(_human_bytes(sizeb), FG_YELLOW, color=color)
        return f"{indent}{kind_c}: {name_c}  shape={tuple(t.shape)} dtype={dt} device={dev} ~{size_y}"

    for n, p in mod.named_parameters(recurse=False): print(_line("param", n, p))
    if include_buffers:
        for n, b in mod.named_buffers(recurse=False): print(_line("buffer", n, b))

# =========================
#   Main tree printer
# =========================
def print_module_tree(
    model: nn.Module,
    *,
    root_name: str = "model",
    max_depth: Optional[int] = None,
    filter_regex: Optional[str] = None,
    show_params: bool = False,
    show_buffers: bool = False,
    show_all: bool = True,   # If True: show detailed lines for BOTH params and buffers (names included)
    color: bool = True,
    collapse_experts: bool = True,
    experts_regex: str = r"(^|\.)experts($|\.)",
    experts_show: int = 1,
    layers_regex: str = r"(^|\.)((model_)?layers|layer|h|blocks|block)($|\.)",
    layers_show: Optional[int] = 4,
):
    """
    Pretty-print a module tree with sizes, devices, dtypes, and optional param/buffer details.
    Visual/UI features:
      • Depth-based colors for module names (each level cycles its own color)
      • Distinct colors for dtype tokens and device tokens
      • Params/Buffers are indented one level deeper under each module for clarity
    """

    # ------------------------------------------------------------------
    # Depth color palette (portable 16-color ANSI for consistent display)
    # ------------------------------------------------------------------
    DEPTH_COLORS = [
        "\033[36m",  # cyan
        "\033[33m",  # yellow
        "\033[35m",  # magenta
        "\033[32m",  # green
        "\033[34m",  # blue
        "\033[31m",  # red
    ]
    def depth_color(depth: int) -> str:
        return DEPTH_COLORS[depth % len(DEPTH_COLORS)]

    # ------------------------------------------------------------------
    # Token color maps (dtype/device) — 16-color ANSI with clear labels
    # ------------------------------------------------------------------
    DTYPE_COLOR = {
        "float32": "\033[36m",        # cyan
        "float":   "\033[36m",        # cyan (alias)
        "bfloat16":"\033[35m",        # magenta
        "float16": "\033[33m",        # yellow
        "half":    "\033[33m",        # yellow (alias)
        "MXFP4":   "\033[36m",        # cyan (sentinel 4-bit)
        "NVFP4":   "\033[36m",        # cyan (sentinel 4-bit)
        "int8":    "\033[31m",        # red
        "uint8":   "\033[31m",        # red
        "int16":   "\033[31m",        # red
        "short":   "\033[31m",        # red
        "int32":   "\033[31m",        # red
        "int":     "\033[31m",        # red
        "bool":    "\033[37m",        # white/gray
        "-":       "\033[37m",        # white/gray (unknown)
    }
    for _dtype_name in (
        *[
            name
            for name in (
                "float8_e4m3fn",
                "float8_e5m2",
                "float8_e4m3fnuz",
                "float8_e5m2fnuz",
                "float8_e8m0fnu",
            )
            if hasattr(torch, name)
        ],
        *[name for name in ("float4_e2m1fn_x2",) if hasattr(torch, name)],
    ):
        DTYPE_COLOR[_dtype_name] = "\033[34m" if _dtype_name.startswith("float8_") else "\033[36m"
    DEVICE_COLOR = {
        "cpu":          "\033[37m",  # white/gray
        "cuda":         "\033[32m",  # green
        "xpu":          "\033[34m",  # blue
        "npu":          "\033[35m",  # magenta
        "mps":          "\033[33m",  # yellow
        "hip":          "\033[31m",  # red
        "privateuseone":"\033[36m",  # cyan
        "meta":         "\033[90m",  # dim gray
        "-":            "\033[37m",  # white/gray (unknown)
    }

    def color_dtype(dtype_name: str) -> str:
        code = DTYPE_COLOR.get(dtype_name, "")
        return f"{code}{dtype_name}{RESET}" if (color and code) else dtype_name

    def color_device(device_str: str) -> str:
        # Accept full device strings like "cuda:0" -> key "cuda"
        key = device_str.split(":")[0] if device_str else "-"
        code = DEVICE_COLOR.get(key, "")
        return f"{code}{device_str}{RESET}" if (color and code) else device_str

    # ------------------------------------------------------------------
    # Local helpers (annotation + param printing with colored tokens)
    # ------------------------------------------------------------------
    def colorize_annotation(annot: str) -> str:
        """
        _annotate(mod) returns strings like:
          "[cuda:0 | float16 | ~123MB]" or "[mixed[cuda:0, cpu] | mixed[float16, bfloat16] | ~...]"
        We color the device token(s) and dtype token(s) in-place.
        """
        if not color or "[" not in annot or "]" not in annot:
            return annot
        try:
            # extract inside [...] and split by ' | '
            left = annot[:annot.find("[")]
            inner = annot[annot.find("[")+1 : annot.rfind("]")]
            right = annot[annot.rfind("]")+1:]
            parts = [p.strip() for p in inner.split("|")]
            if len(parts) >= 2:
                # devices
                dev = parts[0]
                if dev.startswith("mixed[") and dev.endswith("]"):
                    items = dev[6:-1]
                    colored = ", ".join(color_device(s.strip()) for s in items.split(","))
                    parts[0] = f"mixed[{colored}]"
                else:
                    parts[0] = color_device(dev)

                # dtypes
                dt = parts[1]
                if dt.startswith("mixed[") and dt.endswith("]"):
                    items = dt[6:-1]
                    colored = ", ".join(color_dtype(s.strip()) for s in items.split(","))
                    parts[1] = f"mixed[{colored}]"
                else:
                    parts[1] = color_dtype(dt)

                return left + "[" + " | ".join(parts) + "]" + right
        except Exception:
            return annot
        return annot

    def print_params_with_colors(indent: str, mod: nn.Module, *, include_buffers: bool):
        """
        Local printer for params/buffers with colored dtype/device tokens and sizes.
        Mirrors _print_params but adds coloring.
        """
        def _line(kind: str, name: str, t: torch.Tensor) -> str:
            # device
            is_meta = bool(getattr(t, "is_meta", False) or (hasattr(t, "device") and t.device.type == "meta"))
            dev_str = "meta" if is_meta else (str(t.device) if hasattr(t, "device") else "-")
            dev_col = color_device(dev_str)

            # dtype
            dt_raw = getattr(t, "dtype", None)
            dt_name = (str(dt_raw).replace("torch.", "")) if dt_raw is not None else "-"
            dt_col = color_dtype(dt_name)

            # size
            if not is_meta and hasattr(t, "element_size"):
                esize = t.element_size()
            else:
                esize = _elem_size(dt_raw) or 0.0
            sizeb = t.numel() * (esize or 0.0)

            kind_c = _maybe(kind, FG_CYAN, color=color)  # "param"/"buffer" label (cyan)
            name_c = _maybe(name, FG_GRAY, color=color)  # parameter/buffer name (gray)
            size_y = _maybe(_human_bytes(sizeb), FG_YELLOW, color=color)  # size (yellow)
            return f"{indent}{kind_c}: {name_c}  shape={tuple(t.shape)} dtype={dt_col} device={dev_col} ~{size_y}"

        for n, p in mod.named_parameters(recurse=False):
            print(_line("param", n, p))
        if include_buffers:
            for n, b in mod.named_buffers(recurse=False):
                print(_line("buffer", n, b))

    # ------------------------------------------------------------------
    # Setup + utilities
    # ------------------------------------------------------------------
    _ = pcre.compile(filter_regex) if filter_regex else None  # reserved for future
    experts_path_re = pcre.compile(experts_regex)
    layers_name_re = pcre.compile(layers_regex) if layers_show is not None else None
    seen: Set[int] = set()

    total_p = sum(p.numel() for p in model.parameters())
    total_b = sum(b.numel() for b in model.buffers())  # fixed loop variable

    def numeric_children(container: nn.Module):
        if not isinstance(container, (nn.ModuleList, nn.Sequential)):
            return None
        children = list(container.named_children())
        if not children:
            return None
        if not all(name.isdigit() for name, _ in children):
            return None
        return children

    def collapse_spec(qual_name: str, container: nn.Module):
        children = numeric_children(container)
        if children is None:
            return None

        total_children = len(children)
        if experts_path_re.search(qual_name):
            if not collapse_experts:
                return None

            show_count = max(0, experts_show)
            if total_children > show_count:
                return children, show_count, "expert"
            return None

        if layers_name_re is None or not layers_name_re.search(qual_name):
            return None

        show_count = max(0, layers_show)
        if total_children > show_count:
            return children, show_count, "layer"
        return None

    def _format_line(prefix: str, trunk: str, qual_name: str, mod: nn.Module,
                     show_counts: bool, depth: int) -> str:
        cls = mod.__class__.__name__
        left = _maybe(prefix + trunk, FG_GRAY, color=color)          # tree trunk (gray)
        name = _maybe(qual_name, depth_color(depth), color=color)    # module name (depth-based color)
        klass = _maybe(cls, DIM, color=color)                        # class name (dim)
        if show_counts:
            p, b = _counts_for_module(mod)
            counts = _maybe(f"(P={human_count(p)} B={human_count(b)})", FG_YELLOW, color=color)  # counts (yellow)
            return f"{left}{name}: {klass}  {counts}"
        else:
            return f"{left}{name}: {klass}"

    # ------------------------------------------------------------------
    # Recursive printer
    # ------------------------------------------------------------------
    def rec(mod: nn.Module, name: str, depth: int, prefix: str, is_last: bool):
        if max_depth is not None and depth > max_depth:
            return
        mod_id = id(mod)
        shared = "" if mod_id not in seen else "  ↩ shared ref"
        seen.add(mod_id)

        trunk = "└─ " if is_last else "├─ "
        line = _format_line(prefix, trunk, name, mod, show_counts=True, depth=depth)
        annot = colorize_annotation(_annotate(mod, color=color))
        print(line + " " + annot + shared)
        if shared:
            return

        # child base indent (same depth as module trunk)
        indent = prefix + ("   " if is_last else "│  ")
        # params/buffers indent: one level deeper so they clearly nest under the module
        param_indent = indent + ("   " if is_last else "│  ")

        if show_all:
            print_params_with_colors(param_indent, mod, include_buffers=True)
        elif show_params or show_buffers:
            print_params_with_colors(param_indent, mod, include_buffers=show_buffers)

        collapse = collapse_spec(name, mod)
        if collapse is not None:
            children, show_count, item_label = collapse
            shown_children = children[:max(0, min(show_count, len(children)))]
            for i, (child_name, child) in enumerate(shown_children):
                child_is_last = (i == len(shown_children) - 1) and (len(shown_children) == len(children))
                rec(child, f"{name}.{child_name}", depth + 1, indent, child_is_last)

            if len(shown_children) < len(children) and len(children) > 0:
                p_one, b_one = _param_summary(children[0][1], recurse=True)
                collapsed = (
                    f"• … collapsed (repeats {len(shown_children)}..{len(children)-1}, "
                    f"per-{item_label} P={human_count(p_one)} B={human_count(b_one)})"
                )
                print(_maybe(indent + collapsed, DIM, color=color))
            return

        children = list(mod.named_children())
        n = len(children)
        for i, (child_name, child) in enumerate(children):
            last = (i == n - 1)
            rec(child, f"{name}.{child_name}" if name else child_name, depth + 1, indent, last)

    # ------------------------------------------------------------------
    # Root print + recursion
    # ------------------------------------------------------------------
    root_line = _format_line("", "", root_name, model, show_counts=True, depth=0)
    root_annot = colorize_annotation(_annotate(model, color=color))
    print(root_line + " " + root_annot)

    # Root params/buffers appear one level deeper than child trunks
    root_trunk_indent = "   "
    root_param_indent = root_trunk_indent + "   "

    if show_all:
        print_params_with_colors(root_param_indent, model, include_buffers=True)
    elif show_params or show_buffers:
        print_params_with_colors(root_param_indent, model, include_buffers=show_buffers)

    children_root = list(model.named_children())
    for i, (child_name, child) in enumerate(children_root):
        last = (i == len(children_root) - 1)
        rec(child, f"{root_name}.{child_name}", 1, "", last)

    # Footer totals
    print("\nTotal parameters:", human_count(total_p), " | Total buffers:", human_count(total_b))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total_p - trainable
    print("Trainable:", human_count(trainable), " | Frozen:", human_count(frozen))


def _get_qualified_name(root: torch.nn.Module, obj: torch.nn.Module) -> str:
    for name, mod in root.named_modules():
        if mod is obj:
            if not name:
                raise ValueError("Provided module is the root; expected a submodule.")
            return name
    raise ValueError("Could not find the given module within the target model.")

def _get_parent_and_leaf_by_path(root: torch.nn.Module, dotted: str):
    if "." in dotted:
        parent_path, leaf = dotted.rsplit(".", 1)
        parent = root.get_submodule(parent_path)
    else:
        parent, leaf = root, dotted
    return parent, leaf

def _ensure_target_storage_on_device_(param: torch.nn.Parameter, device: torch.device) -> torch.nn.Parameter:
    """Make sure `param`'s storage is on `device` without using set_ across devices."""
    # meta -> allocate fresh on device
    if getattr(param, "is_meta", False) or param.device.type == "meta":
        return torch.nn.Parameter(torch.empty_like(param, device=device), requires_grad=False)
    # already on device -> keep
    if param.device == device:
        return param
    # CPU or wrong GPU -> rebind data storage on target device
    param.data = param.data.to(device, copy=True)  # alloc new storage on device; keeps Parameter identity
    return param


@dataclass(frozen=True)
class _MoEAliasSpec:
    """MoE alias groups derived entirely from the model definition's `module_tree`."""

    runtime_root_path: tuple[str, ...]
    root_alias_paths: tuple[tuple[str, ...], ...]
    runtime_experts_path: tuple[str, ...]
    expert_alias_paths: tuple[tuple[str, ...], ...]
    runtime_leaf_groups: tuple[tuple[str, ...], ...]
    leaf_alias_groups: tuple[tuple[tuple[str, ...], ...], ...]


@dataclass
class _LazyWeightRenaming:
    """Lightweight 1:1 renaming rule that mirrors `transformers.WeightRenaming` matching semantics."""

    source_patterns: list[str] | str
    target_patterns: list[str] | str

    def __post_init__(self) -> None:
        self.source_patterns = self._coerce_patterns(self.source_patterns)
        self.target_patterns = self._coerce_patterns(self.target_patterns)
        if len(self.source_patterns) != 1 or len(self.target_patterns) != 1:
            raise ValueError("_LazyWeightRenaming expects exactly one source and one target pattern.")

    @staticmethod
    def _coerce_patterns(patterns: list[str] | tuple[str, ...] | str | None) -> list[str]:
        if isinstance(patterns, str):
            return [patterns]
        if isinstance(patterns, (list, tuple)):
            return [pattern for pattern in patterns if isinstance(pattern, str)]
        return []

    @staticmethod
    def _process_target_pattern(pattern: str) -> tuple[str, str | None]:
        # Mirror HF reverse-mapping prep: strip anchors/lookarounds, then turn
        # the first capturing group into a reusable `\1` placeholder.
        pattern = pattern.removeprefix("^")
        pattern = pattern.removesuffix("$")
        pattern = re.sub(r"\(\?.+?\)?\)", "", pattern)
        pattern = pattern.replace(r"\.", ".")

        capturing_group_match = re.search(r"\(.+?\)", pattern)
        captured_group = None
        if capturing_group_match:
            captured_group = capturing_group_match.group(0)
            pattern = pattern.replace(captured_group, r"\1", 1)

        return pattern, captured_group

    @staticmethod
    def _process_source_pattern(source_pattern: str, target_pattern: str) -> str:
        if target_pattern.startswith("^"):
            source_pattern = f"^{source_pattern}" if not source_pattern.startswith("^") else source_pattern
        if target_pattern.endswith("$"):
            source_pattern = f"{source_pattern}$" if not source_pattern.endswith("$") else source_pattern
        return source_pattern

    def _prepared_patterns(self) -> tuple[list[str], list[str], re.Pattern]:
        # Derive the regex lazily so the object only stores the original
        # source/target patterns, not cached match state.
        processed_targets: list[str] = []
        target_capturing_groups: list[str] = []
        for pattern in self.target_patterns:
            processed_pattern, captured_group = self._process_target_pattern(pattern)
            processed_targets.append(processed_pattern)
            if captured_group is not None:
                target_capturing_groups.append(captured_group)

        unique_capturing_groups = set(target_capturing_groups)
        if len(unique_capturing_groups) > 1:
            raise ValueError(
                f"Multiple different capturing groups found in target_patterns: {unique_capturing_groups}. "
                f"All target patterns must use the same capturing group pattern."
            )
        unique_capturing_group = unique_capturing_groups.pop() if unique_capturing_groups else None

        processed_sources: list[str] = []
        for i, pattern in enumerate(self.source_patterns):
            processed_pattern = pattern
            if r"\1" in processed_pattern:
                if unique_capturing_group is None:
                    raise ValueError(
                        f"Source pattern '{pattern}' contains \\1 backreference, but no capturing groups "
                        f"found in target_patterns."
                    )
                processed_pattern = processed_pattern.replace(r"\1", unique_capturing_group, 1)
            processed_pattern = self._process_source_pattern(processed_pattern, self.target_patterns[i])
            processed_sources.append(processed_pattern)

        branches = []
        for i, source_pattern in enumerate(processed_sources):
            group_name = f"g{i}"
            pattern = source_pattern.replace(".*.", r"\..*\.")
            branches.append(f"(?P<{group_name}>{pattern})")
        compiled_sources = re.compile("|".join(branches))

        return processed_sources, processed_targets, compiled_sources

    def rename_source_key(self, source_key: str) -> tuple[str, str | None]:
        _, processed_targets, compiled_sources = self._prepared_patterns()
        match_object = compiled_sources.search(source_key)
        if match_object is None:
            return source_key, None

        matching_group_name = next(name for name, val in match_object.groupdict().items() if val is not None)
        source_pattern_that_matched = self.source_patterns[int(matching_group_name[1:])]
        replacement = processed_targets[0]
        if r"\1" in replacement:
            replaced_group_idx = compiled_sources.groupindex[matching_group_name] + 1
            replacement = replacement.replace(r"\1", match_object.group(replaced_group_idx))
        renamed_key = source_key.replace(match_object.group(0), replacement, 1)
        return renamed_key, source_pattern_that_matched


class LazyTurtle:
    """Checkpoint-backed shell materializer for local dense checkpoints.

    The traditional offload path builds a meta shell model and then instantiates
    a full CPU "turtle" model from `from_pretrained()` so submodules can be
    copied over on demand. For very large local sharded checkpoints this upfront
    load is dominated by walking every shard.

    This source keeps only the checkpoint index in memory and materializes the
    requested shell submodule directly from the relevant safetensors shards. If
    a local checkpoint only provides PyTorch pickle weights (`.bin`, `.pt`,
    `.pth`, `.ckpt`), LazyTurtle converts them into temporary safetensors once
    and then reads from that converted directory.
    """

    supports_reload = False
    is_lazy_checkpoint_source = True
    _FALLBACK_CHECKPOINT_EXTENSIONS = (".bin", ".pt", ".pth", ".ckpt")
    _TEMPDIR_PREFIX = "lazyturtle-"

    def __init__(
        self,
        *,
        model_local_path: str,
        config: Any,
        model_init_kwargs: Optional[Dict[str, Any]] = None,
        module_tree: Optional[Any] = None,
        hf_conversion_map_reversed: Optional[Any] = None,
        target_model: Optional[nn.Module] = None,
    ) -> None:
        self.config = copy.deepcopy(config)
        self._model_init_kwargs = dict(model_init_kwargs or {})
        self.model_local_path, self._weight_map = self._resolve_checkpoint_source(
            model_local_path=model_local_path,
            target_model=target_model,
        )
        # Lazy checkpoint name resolution must come from model-definition truth.
        self._module_tree = copy.deepcopy(module_tree)
        self._module_tree_layer_prefix, self._moe_alias_specs = self._build_moe_alias_specs(self._module_tree)
        # Resolve runtime->checkpoint renamings once up front so per-tensor
        # lookups can apply the same renaming order as HF loading.
        alias_items = self._normalize_runtime_to_checkpoint_renamings(
            hf_conversion_map_reversed
            if hf_conversion_map_reversed is not None
            else self.infer_hf_conversion_map_reversed(target_model=target_model),
        )
        self._runtime_to_checkpoint_renamings = tuple(alias_items)
        self._lock = threading.RLock()

    @classmethod
    def maybe_create(
        cls,
        *,
        model_local_path: Optional[str],
        config: Any,
        model_init_kwargs: Optional[Dict[str, Any]] = None,
        module_tree: Optional[Any] = None,
        hf_conversion_map_reversed: Optional[Any] = None,
        target_model: Optional[nn.Module] = None,
    ) -> Optional["LazyTurtle"]:
        if not model_local_path or not os.path.isdir(model_local_path):
            return None

        try:
            return cls(
                model_local_path=model_local_path,
                config=config,
                model_init_kwargs=model_init_kwargs,
                module_tree=module_tree,
                hf_conversion_map_reversed=hf_conversion_map_reversed,
                target_model=target_model,
            )
        except Exception as exc:
            log.debug(
                "LazyTurtle: disabled for `%s`: %s",
                model_local_path,
                exc,
            )
            return None

    def eval(self) -> "LazyTurtle":
        return self

    def materialize_submodule(
        self,
        *,
        target_model: torch.nn.Module,
        target_submodule: torch.nn.Module,
        device: torch.device,
        non_blocking: bool = False,
    ) -> torch.nn.Module:
        path = _get_qualified_name(target_model, target_submodule)
        with self._lock:
            self._copy_checkpoint_tensors_into_submodule(
                target_model=target_model,
                target_submodule=target_submodule,
                module_path=path,
                device=device,
                recurse=True,
                non_blocking=non_blocking,
            )
        if hasattr(target_model, "tie_weights"):
            target_model.tie_weights()
        return target_submodule

    def _convert_checkpoint_source_to_safetensors(
        self,
        *,
        model_local_path: str,
        target_model: Optional[nn.Module],
    ) -> str:
        from .model import get_checkpoints

        _, resolved_archive_file, _ = get_checkpoints(
            model_local_path,
            extensions=list(self._FALLBACK_CHECKPOINT_EXTENSIONS),
            possible_model_basenames=["model", "pytorch_model"],
        )
        tempdir = tempfile.mkdtemp(prefix=self._TEMPDIR_PREFIX)
        # CI jobs only guarantee writes under a temp root, so register best-effort
        # cleanup for the converted checkpoint directory at interpreter exit.
        atexit.register(shutil.rmtree, tempdir, ignore_errors=True)

        log.info(
            "LazyTurtle: no safetensors found under `%s`; converting `%s` into temporary safetensors at `%s`.",
            model_local_path,
            resolved_archive_file,
            tempdir,
        )

        model_cls = type(target_model) if target_model is not None else None
        if model_cls is None or not callable(getattr(model_cls, "from_pretrained", None)):
            raise TypeError("LazyTurtle: transformers-based fallback requires a target model class with from_pretrained().")

        load_kwargs = dict(self._model_init_kwargs)
        load_kwargs.pop("device_map", None)
        # The temporary safetensors export needs a full CPU materialization pass,
        # not the shell/offload settings used by the lazy runtime itself.
        load_kwargs["low_cpu_mem_usage"] = False

        loaded_model = None
        try:
            try:
                loaded_model = model_cls.from_pretrained(
                    model_local_path,
                    config=copy.deepcopy(self.config),
                    **load_kwargs,
                )
            except TypeError:
                loaded_model = model_cls.from_pretrained(
                    model_local_path,
                    **load_kwargs,
                )

            if not callable(getattr(loaded_model, "save_pretrained", None)):
                raise TypeError(
                    f"LazyTurtle: `{model_cls.__name__}` does not provide save_pretrained(), "
                    "cannot convert checkpoint via transformers full-model load."
                )
            loaded_model.save_pretrained(tempdir, safe_serialization=True)
            return tempdir
        except Exception:
            # If transformers reload/export fails, do not leave behind a partial
            # temp directory because LazyTurtle will be disabled for this model.
            shutil.rmtree(tempdir, ignore_errors=True)
            raise
        finally:
            if loaded_model is not None:
                del loaded_model
            gc.collect()

    def _resolve_checkpoint_source(
        self,
        *,
        model_local_path: str,
        target_model: Optional[nn.Module],
    ) -> tuple[str, Dict[str, str]]:
        try:
            weight_map = self._load_weight_map(model_local_path)
            return model_local_path, weight_map
        except FileNotFoundError:
            # Keep the fast path untouched for native safetensors models and only
            # pay the conversion cost when the local checkpoint is pickle-based.
            tempdir = self._convert_checkpoint_source_to_safetensors(
                model_local_path=model_local_path,
                target_model=target_model,
            )
            weight_map = self._load_weight_map(tempdir)
            return tempdir, weight_map

    def checkpoint_tensors_for_submodule(
        self,
        *,
        target_model: nn.Module,
        target_submodule: nn.Module,
        recurse: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Load checkpoint tensors for one shell submodule without mutating it."""

        path = _get_qualified_name(target_model, target_submodule)
        with self._lock:
            return self._load_checkpoint_tensors_for_module_path(
                module_path=path,
                recurse=recurse,
            )

    def sync_all_meta(
        self,
        *,
        shell_model: nn.Module,
        require_class_match: bool = True,
        verify_shapes: bool = True,
        tie_after: bool = True,
    ) -> int:
        del require_class_match, verify_shapes

        materialized = 0
        param_cache: Dict[tuple[str, torch.dtype, bool], nn.Parameter] = {}
        buffer_cache: Dict[tuple[str, torch.dtype], torch.Tensor] = {}

        with self._lock, torch.inference_mode():
            for qname, shell_sub in list(shell_model.named_modules()):
                materialized += self._materialize_direct_meta_tensors(
                    shell_sub=shell_sub,
                    module_path=qname,
                    param_cache=param_cache,
                    buffer_cache=buffer_cache,
                )

        if tie_after and hasattr(shell_model, "tie_weights") and getattr(shell_model.config, "tie_word_embeddings", False):
            try:
                shell_model.tie_weights()
                log.info("Module: Re-tied embedding weights on shell model after lazy sync")
            except Exception as exc:
                log.info(f"Module: tie_weights failed: {exc}")

        log.info("Module: Total direct tensors materialized from lazy checkpoint source: %s", materialized)
        return materialized

    def _load_weight_map(self, model_local_path: str) -> Dict[str, str]:
        from .model import get_checkpoints

        is_sharded, resolved_archive_file, _ = get_checkpoints(
            model_local_path,
            extensions=[".safetensors"],
            possible_model_basenames=["model", "pytorch_model"],
        )

        if is_sharded:
            with open(resolved_archive_file, encoding="utf-8") as fp:
                index = json.load(fp)
            weight_map = index.get("weight_map", {})
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Invalid safetensors index: {resolved_archive_file}")
            return {str(name): str(filename) for name, filename in weight_map.items()}

        shard_name = os.path.basename(resolved_archive_file)
        with safe_open(resolved_archive_file, framework="pt", device="cpu") as handler:
            keys = list(handler.keys())
        if not keys:
            raise ValueError(f"No tensors found in safetensors file: {resolved_archive_file}")
        return {str(name): shard_name for name in keys}

    @staticmethod
    def _join_tensor_name(module_path: str, rel_name: str) -> str:
        if not module_path:
            return rel_name
        if not rel_name:
            return module_path
        return f"{module_path}.{rel_name}"

    @staticmethod
    def _coerce_patterns(patterns: Any) -> list[str]:
        if isinstance(patterns, str):
            return [patterns]
        if isinstance(patterns, (list, tuple)):
            return [pattern for pattern in patterns if isinstance(pattern, str)]
        return []

    @classmethod
    def _extract_weight_renaming_patterns(cls, entry: Any) -> Optional[tuple[str, str]]:
        operations = getattr(entry, "operations", None)
        if operations:
            # LazyTurtle only needs reversible key renames here, not tensor ops.
            return None

        source_patterns = getattr(entry, "_original_source_patterns", getattr(entry, "source_patterns", None))
        target_patterns = getattr(entry, "_original_target_patterns", getattr(entry, "target_patterns", None))
        source_patterns = cls._coerce_patterns(source_patterns)
        target_patterns = cls._coerce_patterns(target_patterns)
        if len(source_patterns) != 1 or len(target_patterns) != 1:
            return None
        return source_patterns[0], target_patterns[0]

    @classmethod
    def _normalize_runtime_to_checkpoint_renamings(cls, raw_aliases: Optional[Any]) -> tuple[_LazyWeightRenaming, ...]:
        renamings: list[_LazyWeightRenaming] = []
        if raw_aliases is None:
            return ()

        if isinstance(raw_aliases, dict):
            # Backward compatibility for older runtime->checkpoint prefix maps.
            for runtime_prefix, checkpoint_prefix in raw_aliases.items():
                if not isinstance(runtime_prefix, str) or not isinstance(checkpoint_prefix, str):
                    continue
                runtime_prefix = runtime_prefix.strip(".")
                checkpoint_prefix = checkpoint_prefix.strip(".")
                if not runtime_prefix:
                    continue
                renamings.append(_LazyWeightRenaming(runtime_prefix, checkpoint_prefix))
            return tuple(renamings)

        if not isinstance(raw_aliases, (list, tuple)):
            return ()

        for entry in raw_aliases:
            patterns = cls._extract_weight_renaming_patterns(entry)
            if patterns is None:
                continue
            # New path: consume reversed WeightRenaming-style entries directly.
            runtime_pattern, checkpoint_pattern = patterns
            renamings.append(_LazyWeightRenaming(runtime_pattern, checkpoint_pattern))

        return tuple(renamings)

    @classmethod
    def _iter_hf_conversion_pairs(cls, conversion_mapping: Optional[Any]) -> Iterable[tuple[str, str]]:
        if isinstance(conversion_mapping, dict):
            for checkpoint_pattern, runtime_prefix in conversion_mapping.items():
                if isinstance(checkpoint_pattern, str) and isinstance(runtime_prefix, str):
                    yield checkpoint_pattern, runtime_prefix
            return

        if not isinstance(conversion_mapping, (list, tuple)):
            return

        for entry in conversion_mapping:
            patterns = cls._extract_weight_renaming_patterns(entry)
            if patterns is not None:
                yield patterns

    @classmethod
    def reverse_hf_conversion_map(cls, conversion_mapping: Optional[Any]) -> Optional[list[_LazyWeightRenaming]]:
        """Invert simple HF checkpoint renames into reversed `WeightRenaming`-style rules."""
        reversed_map: list[_LazyWeightRenaming] = []
        for checkpoint_pattern, runtime_prefix in cls._iter_hf_conversion_pairs(conversion_mapping):
            reversed_map.append(_LazyWeightRenaming(runtime_prefix, checkpoint_pattern))

        return reversed_map or None

    @classmethod
    def infer_hf_conversion_map_reversed(cls, *, target_model: Optional[nn.Module] = None) -> Optional[Any]:
        if target_model is None:
            return None

        model_type = getattr(getattr(target_model, "config", None), "model_type", None)
        if isinstance(model_type, str):
            # Prefer the public transformers conversion registry and fall back to
            # older per-model mappings when needed.
            try:
                conversion_mapping_module = import_module("transformers.conversion_mapping")
            except Exception:
                conversion_mapping_module = None
            if conversion_mapping_module is not None:
                get_checkpoint_conversion_mapping = getattr(
                    conversion_mapping_module,
                    "get_checkpoint_conversion_mapping",
                    None,
                )
                if callable(get_checkpoint_conversion_mapping):
                    reversed_map = cls.reverse_hf_conversion_map(get_checkpoint_conversion_mapping(model_type))
                    if reversed_map is not None:
                        return reversed_map

        return cls.reverse_hf_conversion_map(getattr(target_model, "_checkpoint_conversion_mapping", None))

    @staticmethod
    def _parse_module_spec(module_spec: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Split one module-tree token into ordered aliases and `:flag` suffixes."""

        parts = module_spec.split(":") if isinstance(module_spec, str) else [str(module_spec)]
        aliases = tuple(alias for alias in parts[0].split("|") if alias) if parts else (str(module_spec),)
        if not aliases:
            aliases = (str(module_spec),)
        flags = tuple(part for part in parts[1:] if part)
        return aliases, flags

    @staticmethod
    def _expand_path_aliases(path_aliases: tuple[tuple[str, ...], ...]) -> tuple[tuple[str, ...], ...]:
        """Expand a sequence of aliased path segments into every concrete path variant."""

        paths: list[tuple[str, ...]] = [()]
        for segment_aliases in path_aliases:
            next_paths: list[tuple[str, ...]] = []
            for prefix in paths:
                for alias in segment_aliases:
                    candidate = prefix + (alias,)
                    if candidate not in next_paths:
                        next_paths.append(candidate)
            paths = next_paths
        return tuple(paths)

    @classmethod
    def _build_moe_alias_specs(cls, module_tree: Optional[Any]) -> tuple[tuple[str, ...], tuple[_MoEAliasSpec, ...]]:
        """Extract runtime/checkpoint MoE aliases directly from the model definition's `module_tree`."""

        if not isinstance(module_tree, list):
            return (), ()

        layer_prefix: list[str] = []
        specs: list[_MoEAliasSpec] = []
        seen_specs: set[
            tuple[
                tuple[str, ...],
                tuple[tuple[str, ...], ...],
                tuple[str, ...],
                tuple[tuple[str, ...], ...],
                tuple[tuple[tuple[str, ...], ...], ...],
            ]
        ] = set()

        for item in module_tree:
            if item == "#":
                break
            if isinstance(item, str):
                aliases, _flags = cls._parse_module_spec(item)
                layer_prefix.append(aliases[0])

        def walk(node: Any, path: tuple[Any, ...], moe_root: Optional[tuple[tuple[str, ...], ...]]) -> None:
            if isinstance(node, dict):
                for raw_key, value in node.items():
                    if raw_key == "#":
                        walk(value, path + ("#",), moe_root)
                        continue
                    if not isinstance(raw_key, str):
                        continue
                    aliases, flags = cls._parse_module_spec(raw_key)
                    next_path = path + (aliases,)
                    next_moe_root = next_path if "moe" in flags and moe_root is None else moe_root
                    walk(value, next_path, next_moe_root)
                return

            if isinstance(node, (tuple, list)) and all(isinstance(item, str) for item in node):
                if moe_root is None or "#" not in path:
                    return

                placeholder_index = path.index("#")
                experts_path_aliases = tuple(path[:placeholder_index])
                grouped: Dict[int, list[tuple[str, ...]]] = {}
                for raw_leaf in node:
                    leaf_aliases, flags = cls._parse_module_spec(raw_leaf)
                    group_index = 0
                    for flag in flags:
                        if flag.isdigit():
                            group_index = int(flag)
                            break
                    grouped.setdefault(group_index, []).append(leaf_aliases)

                if not grouped:
                    return

                leaf_alias_groups = tuple(tuple(grouped[group]) for group in sorted(grouped))
                runtime_root_path = tuple(segment[0] for segment in moe_root)
                runtime_experts_path = tuple(segment[0] for segment in experts_path_aliases)
                runtime_leaf_groups = tuple(
                    tuple(leaf_aliases[0] for leaf_aliases in group)
                    for group in leaf_alias_groups
                )
                root_alias_paths = cls._expand_path_aliases(moe_root)
                expert_alias_paths = cls._expand_path_aliases(experts_path_aliases)
                spec_key = (
                    runtime_root_path,
                    root_alias_paths,
                    runtime_experts_path,
                    expert_alias_paths,
                    leaf_alias_groups,
                )
                if spec_key in seen_specs:
                    return
                seen_specs.add(spec_key)
                specs.append(
                    _MoEAliasSpec(
                        runtime_root_path=runtime_root_path,
                        root_alias_paths=root_alias_paths,
                        runtime_experts_path=runtime_experts_path,
                        expert_alias_paths=expert_alias_paths,
                        runtime_leaf_groups=runtime_leaf_groups,
                        leaf_alias_groups=leaf_alias_groups,
                    )
                )
                return

            if isinstance(node, (tuple, list)):
                for item in node:
                    walk(item, path, moe_root)

        found_hash = False
        for item in module_tree:
            if item == "#":
                found_hash = True
                continue
            if not found_hash:
                continue
            walk(item, (), None)

        return tuple(layer_prefix), tuple(specs)

    def _split_layer_relative_path(self, name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return `(layer_prefix_with_index, relative_parts)` for a runtime or checkpoint tensor path."""

        parts = tuple(part for part in name.split(".") if part)
        prefix = self._module_tree_layer_prefix
        if prefix:
            max_start = len(parts) - len(prefix)
            for start in range(max_start + 1):
                end = start + len(prefix)
                if parts[start:end] != prefix:
                    continue
                if end >= len(parts) or not parts[end].isdigit():
                    continue
                return parts[start : end + 1], parts[end + 1 :]
        return (), parts

    def _module_tree_name_aliases(self, name: str) -> list[str]:
        """Generate checkpoint-name candidates from MoE aliases declared in `module_tree`."""

        if not self._moe_alias_specs or not name:
            return []

        layer_head, rel_parts = self._split_layer_relative_path(name)
        if not rel_parts:
            return []

        aliases: list[str] = []
        seen = {name}

        for spec in self._moe_alias_specs:
            if tuple(rel_parts[: len(spec.runtime_root_path)]) == spec.runtime_root_path:
                tail = rel_parts[len(spec.runtime_root_path) :]
                for root_alias in spec.root_alias_paths:
                    alias = ".".join(layer_head + root_alias + tail)
                    if alias not in seen:
                        seen.add(alias)
                        aliases.append(alias)

            if len(rel_parts) < len(spec.runtime_experts_path) + 2:
                continue
            if tuple(rel_parts[: len(spec.runtime_experts_path)]) != spec.runtime_experts_path:
                continue

            expert_index = rel_parts[len(spec.runtime_experts_path)]
            runtime_leaf = rel_parts[len(spec.runtime_experts_path) + 1]
            if not expert_index.isdigit():
                continue

            for group_index, runtime_group in enumerate(spec.runtime_leaf_groups):
                if runtime_leaf not in runtime_group:
                    continue
                leaf_index = runtime_group.index(runtime_leaf)
                tail = rel_parts[len(spec.runtime_experts_path) + 2 :]
                for expert_alias_path in spec.expert_alias_paths:
                    for leaf_alias in spec.leaf_alias_groups[group_index][leaf_index]:
                        alias = ".".join(layer_head + expert_alias_path + (expert_index, leaf_alias) + tail)
                        if alias not in seen:
                            seen.add(alias)
                            aliases.append(alias)
                break

        return aliases

    @staticmethod
    def _candidate_module_paths(module_path: str, *, allow_empty: bool = False) -> list[str]:
        """Return progressively stripped module path aliases for checkpoint lookup."""

        if not module_path:
            return [""]

        parts = module_path.split(".")
        candidates: list[str] = []
        for drop_count in range(len(parts) + 1):
            candidate = ".".join(parts[drop_count:])
            if not candidate and not allow_empty:
                continue
            if candidate in candidates:
                continue
            candidates.append(candidate)
        return candidates

    def _runtime_to_checkpoint_alias_candidates(self, name: str) -> list[str]:
        """Return `name` plus the result of applying runtime->checkpoint renamings once.

        This handles model families whose execution shell layout does not match
        the serialized checkpoint layout. For example, Qwen2-VL runs with
        `model.language_model.layers.0...` in memory while the checkpoint stores
        the same tensors as `model.layers.0...`.
        """

        if not name:
            return [name]

        candidates: list[str] = [name]
        renamed = name
        # Apply the reversed HF renaming chain once, in order, just like
        # `transformers.rename_source_key()` walks WeightRenaming rules.
        for renaming in self._runtime_to_checkpoint_renamings:
            renamed, _ = renaming.rename_source_key(renamed)
        if renamed and renamed not in candidates:
            candidates.append(renamed)
        return candidates

    def _resolve_checkpoint_module_path(self, module_path: str) -> str:
        """Resolve a shell module path to the checkpoint path when wrappers add extra roots."""

        candidates = self._runtime_to_checkpoint_alias_candidates(module_path)
        for aliased in tuple(candidates):
            # Apply declared runtime->checkpoint aliases before the generic
            # prefix-stripping fallback so nested shell paths such as
            # `model.language_model.layers.0` can correctly resolve to
            # checkpoint paths like `model.layers.0`.
            candidates.extend(self._module_tree_name_aliases(aliased))
            for candidate_path in self._candidate_module_paths(aliased):
                candidates.extend(self._runtime_to_checkpoint_alias_candidates(candidate_path))

        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            prefix = f"{candidate}."
            if any(full_name.startswith(prefix) for full_name in self._weight_map):
                return candidate
        return module_path

    def _resolve_checkpoint_tensor_name(self, module_path: str, rel_name: str) -> str:
        """Resolve a tensor name against checkpoint paths declared by `module_tree` and shell path prefixes."""

        full_name = self._join_tensor_name(module_path, rel_name)
        candidates: list[str] = []
        seen: set[str] = set()
        for candidate_path in self._candidate_module_paths(module_path, allow_empty=True):
            for aliased_path in self._runtime_to_checkpoint_alias_candidates(candidate_path):
                candidate_name = self._join_tensor_name(aliased_path, rel_name)
                if candidate_name not in seen:
                    seen.add(candidate_name)
                    candidates.append(candidate_name)
                for alias in self._module_tree_name_aliases(candidate_name):
                    if alias in seen:
                        continue
                    seen.add(alias)
                    candidates.append(alias)

        for candidate in candidates:
            if candidate in self._weight_map:
                return candidate
        return full_name

    def _resolve_split_gate_up_tensor_name(
        self,
        module_path: str,
        rel_name: str,
    ) -> tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
        """Resolve split gate/up projection tensors against fused `gate_up_proj` checkpoint entries."""

        parts = rel_name.split(".")
        if len(parts) < 2:
            return None, None, None, None

        proj_name = parts[-2]
        tensor_name = parts[-1]
        if proj_name not in {"gate_proj", "up_proj"} or tensor_name not in {"weight", "bias"}:
            return None, None, None, None

        fused_parts = list(parts)
        fused_parts[-2] = "gate_up_proj"
        fused_rel_name = ".".join(fused_parts)
        split_index = 0 if proj_name == "gate_proj" else 1

        for candidate_path in self._candidate_module_paths(module_path, allow_empty=True):
            candidate_name = self._join_tensor_name(candidate_path, fused_rel_name)
            if candidate_name in self._weight_map:
                return candidate_name, None, split_index, 0

        return None, None, None, None

    def _resolve_fused_expert_tensor_name(
        self,
        module_path: str,
        rel_name: str,
    ) -> tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
        """Resolve defused expert leaf tensors against fused per-expert checkpoint tensors."""

        parts = rel_name.split(".")
        for expert_pos, part in enumerate(parts):
            if part != "experts" or expert_pos + 3 >= len(parts):
                continue
            if not parts[expert_pos + 1].isdigit():
                continue

            expert_index = int(parts[expert_pos + 1])
            proj_name = parts[expert_pos + 2]
            tensor_name = parts[expert_pos + 3]

            fused_leaf = None
            split_index = None
            split_dim = None

            if proj_name in {"gate_proj", "up_proj"}:
                split_index = 0 if proj_name == "gate_proj" else 1
                if tensor_name == "weight":
                    fused_leaf = "gate_up_proj"
                    split_dim = 1
                elif tensor_name == "bias":
                    fused_leaf = "gate_up_proj_bias"
                    split_dim = 0
            elif proj_name == "down_proj":
                if tensor_name == "weight":
                    fused_leaf = "down_proj"
                elif tensor_name == "bias":
                    fused_leaf = "down_proj_bias"

            if fused_leaf is None:
                return None, None, None, None

            fused_parts = parts[: expert_pos + 1] + [fused_leaf]
            fused_rel_name = ".".join(fused_parts)
            for candidate_path in self._candidate_module_paths(module_path, allow_empty=True):
                candidate_name = self._join_tensor_name(candidate_path, fused_rel_name)
                if candidate_name in self._weight_map:
                    return candidate_name, expert_index, split_index, split_dim

            return None, None, None, None

        return None, None, None, None

    @staticmethod
    def _transform_checkpoint_tensor(
        tensor: torch.Tensor,
        *,
        expert_index: Optional[int],
        split_index: Optional[int],
        split_dim: Optional[int],
        expected_shape: Optional[tuple[int, ...]] = None,
        prefer_transposed: Optional[bool] = None,
    ) -> Optional[torch.Tensor]:
        """Slice fused checkpoint tensors into the tensor layout expected by the shell module."""

        if expert_index is not None:
            if tensor.shape[0] <= expert_index:
                return None
            # Fused expert checkpoints store the expert axis first; peel it off before
            # reasoning about split dimensions or transpose decisions.
            tensor = tensor[expert_index].contiguous()

        if expected_shape is None:
            if split_index is not None:
                if split_dim is None or tensor.shape[split_dim] % 2 != 0:
                    return None
                tensor = tensor.chunk(2, dim=split_dim)[split_index].contiguous()
            return tensor

        expected_shape = tuple(expected_shape)

        # Some checkpoints store expert projections as (out, in) while others store
        # them as (in, out). Keep both candidates and let the defused leaf shape be
        # the final arbiter instead of hard-coding one model family's layout.
        candidates: list[tuple[torch.Tensor, bool]] = [(tensor, False)]
        if tensor.ndim == 2:
            transposed = tensor.transpose(0, 1).contiguous()
            if prefer_transposed is True:
                candidates = [(transposed, True), (tensor, False)]
            elif prefer_transposed is None and transposed.shape != tensor.shape:
                candidates.append((transposed, True))
            elif prefer_transposed is False and transposed.shape != tensor.shape:
                candidates.append((transposed, True))

        for candidate, used_transpose in candidates:
            if split_index is None:
                if tuple(candidate.shape) == expected_shape:
                    return candidate.contiguous()
                continue

            preferred_dims: list[int] = []
            mapped_split_dim = split_dim
            if (
                used_transpose
                and candidate.ndim == 2
                and split_dim is not None
                and 0 <= split_dim < 2
            ):
                # The resolver hint is expressed in the checkpoint's native layout.
                # Once we transpose a 2D candidate, the split dimension flips too.
                mapped_split_dim = 1 - split_dim
            if mapped_split_dim is not None and 0 <= mapped_split_dim < candidate.ndim:
                preferred_dims.append(mapped_split_dim)
            preferred_dims.extend(dim for dim in range(candidate.ndim) if dim not in preferred_dims)

            for dim in preferred_dims:
                if candidate.shape[dim] % 2 != 0:
                    continue
                split_tensor = candidate.chunk(2, dim=dim)[split_index].contiguous()
                if tuple(split_tensor.shape) == expected_shape:
                    return split_tensor

        return None

    @staticmethod
    def _resolve_prefer_transposed_hint(
        *,
        target_model: nn.Module,
        module_path: str,
        rel_name: str,
        modules_by_name: Dict[str, nn.Module],
    ) -> Optional[bool]:
        rel_parent, _, _leaf = rel_name.rpartition(".")
        current_path = module_path
        if rel_parent:
            current_path = LazyTurtle._join_tensor_name(module_path, rel_parent)

        # Expert containers usually expose `is_transposed`; leaf Linear modules do not.
        # Walk upward until we find the nearest owner that carries the layout hint.
        while True:
            owner = target_model if not current_path else modules_by_name.get(current_path)
            if owner is not None and hasattr(owner, "is_transposed"):
                value = getattr(owner, "is_transposed")
                if isinstance(value, bool):
                    return value

            if not current_path:
                break
            current_path = current_path.rpartition(".")[0]

        return None

    def _resolve_checkpoint_tensor_source(
        self,
        module_path: str,
        rel_name: str,
    ) -> tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
        """Resolve a target tensor name to its checkpoint source and optional fused split index."""

        full_name = self._resolve_checkpoint_tensor_name(module_path, rel_name)
        if full_name in self._weight_map:
            return full_name, None, None, None

        resolved = self._resolve_split_gate_up_tensor_name(module_path, rel_name)
        if resolved[0] is not None:
            return resolved

        resolved = self._resolve_fused_expert_tensor_name(module_path, rel_name)
        if resolved[0] is not None:
            return resolved

        # Direct-meta rematerialization often visits a leaf Linear whose relative name is
        # just `weight` / `bias`. Retry resolution with the full module path so leaf-only
        # materialization can still map back to fused expert checkpoint tensors.
        combined_name = self._join_tensor_name(module_path, rel_name)
        resolved = self._resolve_split_gate_up_tensor_name("", combined_name)
        if resolved[0] is not None:
            return resolved

        return self._resolve_fused_expert_tensor_name("", combined_name)

    @staticmethod
    def _materialization_issue_message(
        *,
        phase: str,
        kind: str,
        module_path: str,
        rel_name: str,
        reason: str,
        full_name: Optional[str] = None,
        source_shape: Optional[tuple[int, ...]] = None,
        target_shape: Optional[tuple[int, ...]] = None,
        expert_index: Optional[int] = None,
        split_index: Optional[int] = None,
        split_dim: Optional[int] = None,
    ) -> str:
        """Build a consistent error message for checkpoint-backed materialization failures."""

        details = []
        if full_name is not None:
            details.append(f"checkpoint={full_name}")
        if source_shape is not None:
            details.append(f"source_shape={source_shape}")
        if target_shape is not None:
            details.append(f"target_shape={target_shape}")
        if expert_index is not None:
            details.append(f"expert_index={expert_index}")
        if split_index is not None:
            details.append(f"split_index={split_index}")
        if split_dim is not None:
            details.append(f"split_dim={split_dim}")

        suffix = f" ({', '.join(details)})" if details else ""
        return (
            f"LazyTurtle: {phase} {kind} `{rel_name}` under `{module_path or '<root>'}`: "
            f"{reason}{suffix}"
        )

    def _load_checkpoint_tensors_for_module_path(
        self,
        *,
        module_path: str,
        recurse: bool,
    ) -> Dict[str, torch.Tensor]:
        """Return raw checkpoint tensors keyed by submodule-relative names."""

        resolved_module_path = self._resolve_checkpoint_module_path(module_path)
        prefix = f"{resolved_module_path}."
        grouped_names: Dict[str, list[tuple[str, str]]] = {}
        for full_name, shard in self._weight_map.items():
            if not full_name.startswith(prefix):
                continue

            rel_name = full_name[len(prefix):]
            if not rel_name:
                continue
            if not recurse and "." in rel_name:
                continue

            grouped_names.setdefault(shard, []).append((rel_name, full_name))

        tensors: Dict[str, torch.Tensor] = {}
        for shard, names in grouped_names.items():
            shard_path = os.path.join(self.model_local_path, shard)
            with safe_open(shard_path, framework="pt", device="cpu") as handler:
                for rel_name, full_name in names:
                    tensors[rel_name] = handler.get_tensor(full_name)
        return tensors

    def _copy_checkpoint_tensors_into_submodule(
        self,
        *,
        target_model: nn.Module,
        target_submodule: nn.Module,
        module_path: str,
        device: torch.device,
        recurse: bool,
        non_blocking: bool,
    ) -> None:
        """Materialize checkpoint tensors into a shell submodule and rebuild missing init-only buffers."""

        t_params = dict(target_submodule.named_parameters(recurse=recurse))
        t_bufs = dict(target_submodule.named_buffers(recurse=recurse))
        modules_by_name = dict(target_model.named_modules())
        missing_nonpersistent_buffers: list[tuple[str, str]] = []

        grouped_names: Dict[str, list[tuple[str, str, str, Optional[int], Optional[int], Optional[int]]]] = {}
        for rel_name in t_params:
            full_name, expert_index, split_index, split_dim = self._resolve_checkpoint_tensor_source(module_path, rel_name)
            if full_name is None:
                continue
            shard = self._weight_map.get(full_name)
            if shard is None:
                raise RuntimeError(
                    self._materialization_issue_message(
                        phase="submodule materialization",
                        kind="param",
                        module_path=module_path,
                        rel_name=rel_name,
                        reason="checkpoint tensor mapping resolved to a missing shard",
                        full_name=full_name,
                        target_shape=tuple(t_params[rel_name].shape),
                        expert_index=expert_index,
                        split_index=split_index,
                        split_dim=split_dim,
                    )
                )
            grouped_names.setdefault(shard, []).append(("param", rel_name, full_name, expert_index, split_index, split_dim))

        for rel_name, target_buffer in list(t_bufs.items()):
            full_name, expert_index, split_index, split_dim = self._resolve_checkpoint_tensor_source(module_path, rel_name)
            if full_name is None:
                full_name = self._resolve_checkpoint_tensor_name(module_path, rel_name)
                expert_index = None
                split_index = None
                split_dim = None
            shard = self._weight_map.get(full_name)
            if shard is None:
                t_parent, leaf = _get_parent_and_leaf_by_path(target_submodule, rel_name)
                non_persistent = leaf in getattr(t_parent, "_non_persistent_buffers_set", set())
                if non_persistent:
                    if (
                        getattr(target_buffer, "is_meta", False)
                        or target_buffer.device.type == "meta"
                        or target_buffer.device != device
                    ):
                        missing_nonpersistent_buffers.append((rel_name, leaf))
                    continue
                if getattr(target_buffer, "is_meta", False) or target_buffer.device.type == "meta":
                    if leaf in getattr(t_parent, "_buffers", {}):
                        del t_parent._buffers[leaf]
                continue
            grouped_names.setdefault(shard, []).append(("buffer", rel_name, full_name, expert_index, split_index, split_dim))

        with torch.inference_mode():
            for shard, entries in grouped_names.items():
                shard_path = os.path.join(self.model_local_path, shard)
                with safe_open(shard_path, framework="pt", device="cpu") as handler:
                    for kind, rel_name, full_name, expert_index, split_index, split_dim in entries:
                        target_tensor = t_params.get(rel_name) if kind == "param" else t_bufs.get(rel_name)
                        expected_shape = tuple(target_tensor.shape) if target_tensor is not None else None
                        prefer_transposed = self._resolve_prefer_transposed_hint(
                            target_model=target_model,
                            module_path=module_path,
                            rel_name=rel_name,
                            modules_by_name=modules_by_name,
                        )
                        checkpoint_tensor = handler.get_tensor(full_name)
                        tensor = self._transform_checkpoint_tensor(
                            checkpoint_tensor,
                            expert_index=expert_index,
                            split_index=split_index,
                            split_dim=split_dim,
                            expected_shape=expected_shape,
                            prefer_transposed=prefer_transposed,
                        )
                        if tensor is None:
                            raise RuntimeError(self._materialization_issue_message(
                                phase="submodule materialization",
                                kind=kind,
                                module_path=module_path,
                                rel_name=rel_name,
                                reason="checkpoint tensor could not be reshaped into the target layout",
                                full_name=full_name,
                                source_shape=tuple(checkpoint_tensor.shape),
                                target_shape=expected_shape,
                                expert_index=expert_index,
                                split_index=split_index,
                                split_dim=split_dim,
                            ))
                        if kind == "param":
                            target_param = t_params.get(rel_name)
                            if target_param is None:
                                raise RuntimeError(self._materialization_issue_message(
                                    phase="submodule materialization",
                                    kind=kind,
                                    module_path=module_path,
                                    rel_name=rel_name,
                                    reason="target tensor disappeared before materialization",
                                    full_name=full_name,
                                    source_shape=tuple(tensor.shape),
                                    expert_index=expert_index,
                                    split_index=split_index,
                                    split_dim=split_dim,
                                ))
                            if target_param.shape != tensor.shape:
                                raise RuntimeError(self._materialization_issue_message(
                                    phase="submodule materialization",
                                    kind=kind,
                                    module_path=module_path,
                                    rel_name=rel_name,
                                    reason="target tensor shape does not match the transformed checkpoint tensor",
                                    full_name=full_name,
                                    source_shape=tuple(tensor.shape),
                                    target_shape=tuple(target_param.shape),
                                    expert_index=expert_index,
                                    split_index=split_index,
                                    split_dim=split_dim,
                                ))
                            target_param_new = _ensure_target_storage_on_device_(target_param, device)
                            if target_param_new is not target_param:
                                t_parent, leaf = _get_parent_and_leaf_by_path(target_submodule, rel_name)
                                setattr(t_parent, leaf, target_param_new)
                                target_param = target_param_new
                            source = tensor.detach()
                            if source.dtype != target_param.dtype:
                                source = source.to(dtype=target_param.dtype)
                            target_param.detach().copy_(source, non_blocking=(non_blocking and source.is_pinned()))
                            continue

                        target_buffer = t_bufs.get(rel_name)
                        t_parent, leaf = _get_parent_and_leaf_by_path(target_submodule, rel_name)
                        persistent = leaf not in getattr(t_parent, "_non_persistent_buffers_set", set())

                        source = tensor.detach()
                        if target_buffer is None:
                            new_buffer = source.to(device=device)
                            t_parent.register_buffer(leaf, new_buffer, persistent=persistent)
                            t_bufs[rel_name] = new_buffer
                            continue

                        if tuple(target_buffer.shape) != tuple(source.shape):
                            raise RuntimeError(self._materialization_issue_message(
                                phase="submodule materialization",
                                kind=kind,
                                module_path=module_path,
                                rel_name=rel_name,
                                reason="target tensor shape does not match the transformed checkpoint tensor",
                                full_name=full_name,
                                source_shape=tuple(source.shape),
                                target_shape=tuple(target_buffer.shape),
                                expert_index=expert_index,
                                split_index=split_index,
                                split_dim=split_dim,
                            ))

                        if getattr(target_buffer, "is_meta", False) or target_buffer.device.type == "meta":
                            new_buffer = torch.empty_like(target_buffer, device=device)
                            new_buffer.copy_(source.to(dtype=new_buffer.dtype), non_blocking=(non_blocking and source.is_pinned()))
                            t_parent.register_buffer(leaf, new_buffer, persistent=persistent)
                            t_bufs[rel_name] = new_buffer
                            continue

                        if target_buffer.device != device:
                            new_buffer = torch.empty_like(target_buffer, device=device)
                            new_buffer.copy_(source.to(dtype=new_buffer.dtype), non_blocking=(non_blocking and source.is_pinned()))
                            t_parent.register_buffer(leaf, new_buffer, persistent=persistent)
                            t_bufs[rel_name] = new_buffer
                        else:
                            if source.dtype != target_buffer.dtype:
                                source = source.to(dtype=target_buffer.dtype)
                            target_buffer.copy_(source, non_blocking=(non_blocking and source.is_pinned()))

        self._restore_missing_nonpersistent_buffers(
            target_model=target_model,
            target_submodule=target_submodule,
            t_bufs=t_bufs,
            missing_nonpersistent_buffers=missing_nonpersistent_buffers,
            device=device,
        )

    def _build_nonpersistent_buffer_template(
        self,
        *,
        owner_module: nn.Module,
        target_model: nn.Module,
    ) -> Optional[nn.Module]:
        """Construct a CPU template module for init-only buffers missing from checkpoint shards."""

        config_source = getattr(owner_module, "config", None)
        if config_source is None:
            config_source = getattr(target_model, "config", None)
        if config_source is None:
            config_source = self.config

        module_type = type(owner_module)
        try:
            signature = inspect.signature(module_type)
        except (TypeError, ValueError):
            return None

        params = list(signature.parameters.values())
        if not params:
            return None

        args = []
        kwargs = {}

        for param in params:
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            if param.name == "config":
                if config_source is None:
                    return None
                value = copy.deepcopy(config_source)
            elif param.name == "device":
                value = torch.device("cpu")
            elif hasattr(owner_module, param.name):
                # Some remote-code modules rebuild buffers from constructor attributes instead of config.
                raw_value = getattr(owner_module, param.name)
                if isinstance(raw_value, torch.Tensor) and raw_value.device.type == "meta":
                    scalar_attr_name = f"scalar_{param.name}"
                    if hasattr(owner_module, scalar_attr_name):
                        raw_value = getattr(owner_module, scalar_attr_name)
                    elif param.default is not inspect.Parameter.empty:
                        continue
                    else:
                        return None
                value = copy.deepcopy(raw_value)
            elif param.default is not inspect.Parameter.empty:
                continue
            else:
                return None

            if param.kind is inspect.Parameter.POSITIONAL_ONLY:
                args.append(value)
            else:
                kwargs[param.name] = value

        try:
            return module_type(*args, **kwargs)
        except Exception as exc:
            log.debug(
                "LazyTurtle: failed to build template for `%s`: %s",
                module_type.__name__,
                exc,
            )
            return None

    def _restore_missing_nonpersistent_buffers(
        self,
        *,
        target_model: nn.Module,
        target_submodule: nn.Module,
        t_bufs: Dict[str, torch.Tensor],
        missing_nonpersistent_buffers: list[tuple[str, str]],
        device: torch.device,
    ) -> None:
        """Restore constructor-owned buffers that are intentionally absent from checkpoints."""

        owner_templates: Dict[str, Optional[nn.Module]] = {}
        for rel_name, leaf in missing_nonpersistent_buffers:
            parent_rel_path, _, _ = rel_name.rpartition(".")
            owner_module = target_submodule if not parent_rel_path else dict(target_submodule.named_modules()).get(parent_rel_path)
            if owner_module is None:
                continue

            current_buffer = t_bufs.get(rel_name)
            if (
                current_buffer is not None
                and not getattr(current_buffer, "is_meta", False)
                and current_buffer.device.type != "meta"
            ):
                source_buffer = current_buffer.detach()
            else:
                if parent_rel_path not in owner_templates:
                    owner_templates[parent_rel_path] = self._build_nonpersistent_buffer_template(
                        owner_module=owner_module,
                        target_model=target_model,
                    )
                template = owner_templates[parent_rel_path]
                if template is None:
                    continue
                source_buffer = dict(template.named_buffers(recurse=False)).get(leaf)
                if source_buffer is None:
                    continue
                source_buffer = source_buffer.detach()

            target_dtype = source_buffer.dtype if current_buffer is None else current_buffer.dtype
            materialized = source_buffer.to(device=device, dtype=target_dtype)
            owner_module.register_buffer(leaf, materialized, persistent=False)
            t_bufs[rel_name] = materialized

    def _materialize_direct_meta_tensors(
        self,
        *,
        shell_sub: nn.Module,
        module_path: str,
        param_cache: Dict[tuple[str, Optional[int], Optional[int], Optional[int], torch.dtype, bool], nn.Parameter],
        buffer_cache: Dict[tuple[str, Optional[int], Optional[int], Optional[int], torch.dtype], torch.Tensor],
    ) -> int:
        synced = 0

        with torch.inference_mode():
            for name, shell_param in dict(shell_sub.named_parameters(recurse=False)).items():
                if not _is_meta_tensor(shell_param):
                    continue

                full_name, expert_index, split_index, split_dim = self._resolve_checkpoint_tensor_source(module_path, name)
                if full_name is None:
                    continue
                shard = self._weight_map.get(full_name)
                if shard is None:
                    raise RuntimeError(self._materialization_issue_message(
                        phase="direct-meta sync",
                        kind="param",
                        module_path=module_path,
                        rel_name=name,
                        reason="checkpoint tensor mapping resolved to a missing shard",
                        full_name=full_name,
                        target_shape=tuple(shell_param.shape),
                        expert_index=expert_index,
                        split_index=split_index,
                        split_dim=split_dim,
                    ))

                source_path = os.path.join(self.model_local_path, shard)
                with safe_open(source_path, framework="pt", device="cpu") as handler:
                    checkpoint_param = handler.get_tensor(full_name)
                source_param = self._transform_checkpoint_tensor(
                    checkpoint_param,
                    expert_index=expert_index,
                    split_index=split_index,
                    split_dim=split_dim,
                    expected_shape=tuple(shell_param.shape),
                    prefer_transposed=getattr(shell_sub, "is_transposed", None),
                )
                if source_param is None:
                    raise RuntimeError(self._materialization_issue_message(
                        phase="direct-meta sync",
                        kind="param",
                        module_path=module_path,
                        rel_name=name,
                        reason="checkpoint tensor could not be reshaped into the target layout",
                        full_name=full_name,
                        source_shape=tuple(checkpoint_param.shape),
                        target_shape=tuple(shell_param.shape),
                        expert_index=expert_index,
                        split_index=split_index,
                        split_dim=split_dim,
                    ))

                if shell_param.shape != source_param.shape:
                    raise RuntimeError(self._materialization_issue_message(
                        phase="direct-meta sync",
                        kind="param",
                        module_path=module_path,
                        rel_name=name,
                        reason="target tensor shape does not match the transformed checkpoint tensor",
                        full_name=full_name,
                        source_shape=tuple(source_param.shape),
                        target_shape=tuple(shell_param.shape),
                        expert_index=expert_index,
                        split_index=split_index,
                        split_dim=split_dim,
                    ))

                cache_key = (full_name, expert_index, split_index, split_dim, shell_param.dtype, shell_param.requires_grad)
                new_param = param_cache.get(cache_key)
                if new_param is None:
                    if source_param.dtype != shell_param.dtype:
                        source_param = source_param.to(dtype=shell_param.dtype)
                    new_param = nn.Parameter(
                        source_param.clone(),
                        requires_grad=shell_param.requires_grad,
                    )
                    param_cache[cache_key] = new_param

                shell_sub.register_parameter(name, new_param)
                synced += 1

            for name, shell_buffer in list(dict(shell_sub.named_buffers(recurse=False)).items()):
                if not _is_meta_tensor(shell_buffer):
                    continue

                full_name, expert_index, split_index, split_dim = self._resolve_checkpoint_tensor_source(module_path, name)
                if full_name is None:
                    continue
                shard = self._weight_map.get(full_name)
                if shard is None:
                    raise RuntimeError(self._materialization_issue_message(
                        phase="direct-meta sync",
                        kind="buffer",
                        module_path=module_path,
                        rel_name=name,
                        reason="checkpoint tensor mapping resolved to a missing shard",
                        full_name=full_name,
                        target_shape=tuple(shell_buffer.shape),
                        expert_index=expert_index,
                        split_index=split_index,
                        split_dim=split_dim,
                    ))

                source_path = os.path.join(self.model_local_path, shard)
                with safe_open(source_path, framework="pt", device="cpu") as handler:
                    checkpoint_buffer = handler.get_tensor(full_name)
                source_buffer = self._transform_checkpoint_tensor(
                    checkpoint_buffer,
                    expert_index=expert_index,
                    split_index=split_index,
                    split_dim=split_dim,
                    expected_shape=tuple(shell_buffer.shape),
                    prefer_transposed=getattr(shell_sub, "is_transposed", None),
                )
                if source_buffer is None:
                    raise RuntimeError(self._materialization_issue_message(
                        phase="direct-meta sync",
                        kind="buffer",
                        module_path=module_path,
                        rel_name=name,
                        reason="checkpoint tensor could not be reshaped into the target layout",
                        full_name=full_name,
                        source_shape=tuple(checkpoint_buffer.shape),
                        target_shape=tuple(shell_buffer.shape),
                        expert_index=expert_index,
                        split_index=split_index,
                        split_dim=split_dim,
                    ))

                if shell_buffer.shape != source_buffer.shape:
                    raise RuntimeError(self._materialization_issue_message(
                        phase="direct-meta sync",
                        kind="buffer",
                        module_path=module_path,
                        rel_name=name,
                        reason="target tensor shape does not match the transformed checkpoint tensor",
                        full_name=full_name,
                        source_shape=tuple(source_buffer.shape),
                        target_shape=tuple(shell_buffer.shape),
                        expert_index=expert_index,
                        split_index=split_index,
                        split_dim=split_dim,
                    ))

                persistent = name not in getattr(shell_sub, "_non_persistent_buffers_set", set())
                cache_key = (full_name, expert_index, split_index, split_dim, shell_buffer.dtype)
                new_buffer = buffer_cache.get(cache_key)
                if new_buffer is None:
                    if source_buffer.dtype != shell_buffer.dtype:
                        source_buffer = source_buffer.to(dtype=shell_buffer.dtype)
                    new_buffer = source_buffer.clone()
                    buffer_cache[cache_key] = new_buffer

                shell_sub.register_buffer(name, new_buffer, persistent=persistent)
                synced += 1

        return synced

def alias_from_turtle_for_submodule(
    target_model: torch.nn.Module,
    turtle_model: "LazyTurtle",
    target_submodule: torch.nn.Module,
    device: torch.device,
    non_blocking: bool = False,
) -> torch.nn.Module:
    # Lazy turtle supports materialization from checkpoint storage into CPU or accelerator devices.
    assert device not in [None, torch.device("meta")]
    if not hasattr(turtle_model, "materialize_submodule"):
        raise TypeError(
            f"Expected LazyTurtle-compatible source, got `{type(turtle_model).__name__}`."
        )

    return turtle_model.materialize_submodule(
        target_model=target_model,
        target_submodule=target_submodule,
        device=device,
        non_blocking=non_blocking,
    )

def _is_meta_tensor(t: torch.Tensor) -> bool:
    return bool(getattr(t, "is_meta", False)) or (hasattr(t, "device") and t.device.type == "meta")

def alias_all_from_turtle_if_meta(
    shell_model: nn.Module,
    turtle_model: Optional["LazyTurtle"],
    *,
    require_class_match: bool = True,
    verify_shapes: bool = True,
    tie_after: bool = True,
) -> int:
    """
    Materialize any remaining direct meta tensors in `shell_model` from the lazy turtle source.
    """
    if turtle_model is None:
        return 0

    if not hasattr(turtle_model, "sync_all_meta"):
        raise TypeError(
            f"Expected LazyTurtle-compatible source, got `{type(turtle_model).__name__}`."
        )
    return turtle_model.sync_all_meta(
        shell_model=shell_model,
        require_class_match=require_class_match,
        verify_shapes=verify_shapes,
        tie_after=tie_after,
    )
