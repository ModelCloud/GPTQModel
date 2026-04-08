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

import copy
import inspect
import json
import os
import threading
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import pcre as re
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
if hasattr(torch, "float8_e4m3fn"):
    _DTYPE_BYTES[torch.float8_e4m3fn] = 1
if hasattr(torch, "float8_e5m2"):
    _DTYPE_BYTES[torch.float8_e5m2] = 1

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
        "float8_e4m3fn": "\033[34m",  # blue
        "float8_e5m2":   "\033[34m",  # blue
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
    _ = re.compile(filter_regex) if filter_regex else None  # reserved for future
    experts_path_re = re.compile(experts_regex)
    layers_name_re = re.compile(layers_regex) if layers_show is not None else None
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


class LazyTurtle:
    """Checkpoint-backed shell materializer for local safetensors models.

    The traditional offload path builds a meta shell model and then instantiates
    a full CPU "turtle" model from `from_pretrained()` so submodules can be
    copied over on demand. For very large local sharded checkpoints this upfront
    load is dominated by walking every shard.

    This source keeps only the checkpoint index in memory and materializes the
    requested shell submodule directly from the relevant safetensors shards.
    """

    supports_reload = False
    is_lazy_checkpoint_source = True

    def __init__(
        self,
        *,
        model_local_path: str,
        config: Any,
        model_init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_local_path = model_local_path
        self.config = copy.deepcopy(config)
        self._model_init_kwargs = dict(model_init_kwargs or {})
        self._weight_map = self._load_weight_map(model_local_path)
        self._lock = threading.RLock()

    @classmethod
    def maybe_create(
        cls,
        *,
        model_local_path: Optional[str],
        config: Any,
        model_init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional["LazyTurtle"]:
        if not model_local_path or not os.path.isdir(model_local_path):
            return None

        try:
            return cls(
                model_local_path=model_local_path,
                config=config,
                model_init_kwargs=model_init_kwargs,
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
        t_params = dict(target_submodule.named_parameters(recurse=recurse))
        t_bufs = dict(target_submodule.named_buffers(recurse=recurse))
        missing_nonpersistent_buffers: list[tuple[str, str]] = []

        grouped_names: Dict[str, list[tuple[str, str, str]]] = {}
        for rel_name in t_params:
            full_name = self._join_tensor_name(module_path, rel_name)
            shard = self._weight_map.get(full_name)
            if shard is None:
                continue
            grouped_names.setdefault(shard, []).append(("param", rel_name, full_name))

        for rel_name, target_buffer in list(t_bufs.items()):
            full_name = self._join_tensor_name(module_path, rel_name)
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
            grouped_names.setdefault(shard, []).append(("buffer", rel_name, full_name))

        with torch.inference_mode():
            for shard, entries in grouped_names.items():
                shard_path = os.path.join(self.model_local_path, shard)
                with safe_open(shard_path, framework="pt", device="cpu") as handler:
                    for kind, rel_name, full_name in entries:
                        tensor = handler.get_tensor(full_name)
                        if kind == "param":
                            target_param = t_params.get(rel_name)
                            if target_param is None or target_param.shape != tensor.shape:
                                continue
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
        """Construct a CPU module template when missing buffers must come from init logic."""

        config_source = getattr(owner_module, "config", None)
        if config_source is None:
            config_source = getattr(target_model, "config", None)
        if config_source is None:
            config_source = self.config
        if config_source is None:
            return None

        module_type = type(owner_module)
        try:
            signature = inspect.signature(module_type)
        except (TypeError, ValueError):
            return None

        params = list(signature.parameters.values())
        if not params or params[0].name != "config":
            return None

        args = []
        kwargs = {}
        if params[0].kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(copy.deepcopy(config_source))
        else:
            kwargs["config"] = copy.deepcopy(config_source)

        device_param = signature.parameters.get("device")
        if device_param is not None:
            if device_param.kind is inspect.Parameter.POSITIONAL_ONLY:
                args.append(torch.device("cpu"))
            else:
                kwargs["device"] = torch.device("cpu")

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
        param_cache: Dict[tuple[str, torch.dtype, bool], nn.Parameter],
        buffer_cache: Dict[tuple[str, torch.dtype], torch.Tensor],
    ) -> int:
        synced = 0

        with torch.inference_mode():
            for name, shell_param in dict(shell_sub.named_parameters(recurse=False)).items():
                if not _is_meta_tensor(shell_param):
                    continue

                full_name = self._join_tensor_name(module_path, name)
                shard = self._weight_map.get(full_name)
                if shard is None:
                    continue

                source_path = os.path.join(self.model_local_path, shard)
                with safe_open(source_path, framework="pt", device="cpu") as handler:
                    source_param = handler.get_tensor(full_name)

                if shell_param.shape != source_param.shape:
                    continue

                cache_key = (full_name, shell_param.dtype, shell_param.requires_grad)
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

                full_name = self._join_tensor_name(module_path, name)
                shard = self._weight_map.get(full_name)
                if shard is None:
                    continue

                source_path = os.path.join(self.model_local_path, shard)
                with safe_open(source_path, framework="pt", device="cpu") as handler:
                    source_buffer = handler.get_tensor(full_name)

                if shell_buffer.shape != source_buffer.shape:
                    continue

                persistent = name not in getattr(shell_sub, "_non_persistent_buffers_set", set())
                cache_key = (full_name, shell_buffer.dtype)
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
