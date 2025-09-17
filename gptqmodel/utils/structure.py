"""
Usage:
  # Focus on mixers/MLP, show params/buffers, depth limit
  `filter` "self_attn|linear_attn|mlp" --show-params --max-depth 4

  # Control MoE collapsing:
  #   - collapse any ModuleList whose name matches the regex and whose children are numeric
  #   - show the first N experts and collapse the rest
  `experts-regex` "(^|\\.)experts($|\\.)" --experts-show 1
  `no-collapse-experts`  # disable collapsing

Notes:
- Detects shared submodules and avoids re-printing them.
- Collapsing is generic: any numeric-indexed ModuleList whose qualified name matches `experts-regex`.
"""

import re
from typing import Optional, Set, Tuple

from torch import nn


def human_count(n: int) -> str:
    if n < 1_000:
        return str(n)
    units = ["K", "M", "B", "T"]
    k = 0
    x = float(n)
    while x >= 1000 and k < len(units) - 1:
        x /= 1000.0
        k += 1
    return f"{x:.2f}{units[k]}"


def _param_summary(mod: nn.Module, recurse: bool = False) -> Tuple[int, int]:
    p = sum(p.numel() for p in mod.parameters(recurse=recurse))
    b = sum(bf.numel() for bf in mod.buffers(recurse=recurse))
    return p, b


def _format_line(prefix: str, trunk: str, name: str, mod: nn.Module, show_counts: bool, color: bool) -> str:
    cls = mod.__class__.__name__
    p, b = _param_summary(mod, recurse=False)
    counts = f"  (P={human_count(p)} B={human_count(b)})" if show_counts else ""
    if color:
        BLUE, CYAN, DIM, RESET = "\033[34m", "\033[36m", "\033[2m", "\033[0m"
        return f"{prefix}{trunk}{BLUE}{name}{RESET}: {CYAN}{cls}{RESET}{DIM}{counts}{RESET}"
    return f"{prefix}{trunk}{name}: {cls}{counts}"


def _print_params(prefix: str, mod: nn.Module, include_buffers: bool, color: bool, max_items: int = 100):
    items = []
    for n, p in mod.named_parameters(recurse=False):
        items.append(("param", n, tuple(p.shape), p.dtype))
    if include_buffers:
        for n, b in mod.named_buffers(recurse=False):
            items.append(("buffer", n, tuple(b.shape), b.dtype))
    if not items:
        return
    if color:
        DIM, RESET = "\033[2m", "\033[0m"
    else:
        DIM = RESET = ""
    for kind, n, shape, dt in items[:max_items]:
        print(f"{prefix}{DIM}• {kind}: {n:32s} shape={shape!s:>20s} dtype={str(dt)}{RESET}")
    if len(items) > max_items:
        print(f"{prefix}{DIM}• … {len(items)-max_items} more{RESET}")


from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn

# =========================
#   ANSI color helpers
# =========================
RESET = "\033[0m"
DIM = "\033[2m"
FG_GRAY = "\033[90m"
FG_CYAN = "\033[36m"
FG_YELLOW = "\033[33m"

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
        if n < 1024: return f"{n:.0f}{unit}" if unit == "B" else f"{n:.2f}{unit}"
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
        if not is_meta: alloc_bytes += n * esize
        est_bytes += n * esize

    # summarize
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
    left = _maybe(prefix + trunk, FG_GRAY, color=color)   # tree lines
    name = _maybe(qual_name, FG_CYAN, color=color)        # softened path name
    klass = _maybe(cls, DIM, color=color)                 # class name dimmed
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

    for n, p in mod.named_parameters(recurse=False):
        print(_line("param", n, p))
    if include_buffers:
        for n, b in mod.named_buffers(recurse=False):
            print(_line("buffer", n, b))

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
    color: bool = True,
    collapse_experts: bool = True,
    experts_regex: str = r"(^|\.)experts($|\.)",
    experts_show: int = 1,
):
    _ = re.compile(filter_regex) if filter_regex else None
    experts_name_re = re.compile(experts_regex) if collapse_experts else None
    seen: Set[int] = set()

    total_p = sum(p.numel() for p in model.parameters())
    total_b = sum(b.numel() for b in model.buffers())

    def should_collapse(qual_name: str, container: nn.Module) -> bool:
        if not experts_name_re: return False
        if not experts_name_re.search(qual_name): return False
        if not isinstance(container, (nn.ModuleList, nn.Sequential)): return False
        names = [n for n, _ in container.named_children()]
        if not names: return False
        return all(n.isdigit() for n in names) and len(names) > max(0, experts_show)

    def rec(mod: nn.Module, name: str, depth: int, prefix: str, is_last: bool):
        if max_depth is not None and depth > max_depth: return
        mod_id = id(mod)
        shared = "" if mod_id not in seen else "  ↩ shared ref"
        seen.add(mod_id)
        trunk = "└─ " if is_last else "├─ "
        line = _format_line(prefix, trunk, name, mod, show_counts=True, color=color)
        print(line + " " + _annotate(mod, color=color) + shared)
        if shared: return
        if show_params or show_buffers:
            _print_params(prefix + ("   " if is_last else "│  "), mod, include_buffers=show_buffers, color=color)
        children = list(mod.named_children())
        n = len(children)
        for i, (child_name, child) in enumerate(children):
            last = (i == n - 1)
            child_prefix = prefix + ("   " if is_last else "│  ")
            display_name = f"{name}.{child_name}" if name else child_name
            if should_collapse(display_name, child):
                line2 = _format_line(child_prefix, "└─ " if last else "├─ ", display_name, child, True, color)
                print(line2 + " " + _annotate(child, color=color))
                sub_children = list(child.named_children())
                total_k = len(sub_children)
                k_show = max(0, min(experts_show, total_k))
                for j, (sub_name, sub_mod) in enumerate(sub_children[:k_show]):
                    sub_last = (j == k_show - 1) and (k_show == total_k)
                    sub_prefix = child_prefix + ("   " if last else "│  ")
                    sub_trunk = "└─ " if sub_last else "├─ "
                    line3 = _format_line(sub_prefix, sub_trunk, f"{display_name}.{sub_name}", sub_mod, True, color)
                    print(line3 + " " + _annotate(sub_mod, color=color))
                    rec(sub_mod, f"{display_name}.{sub_name}", depth + 2, child_prefix + ("   " if last else "│  "), sub_last)
                if k_show < total_k:
                    p_one, b_one = _param_summary(sub_children[0][1], recurse=True)
                    collapsed = f"• … collapsed (repeats {k_show}..{total_k-1}, per-expert P={human_count(p_one)} B={human_count(b_one)})"
                    print(_maybe(child_prefix + ("   " if last else "│  ") + collapsed, DIM, color=color))
                continue
            rec(child, display_name, depth + 1, child_prefix, last)

    print(_format_line("", "", root_name, model, show_counts=True, color=color) + " " + _annotate(model, color=color))
    if show_params or show_buffers:
        _print_params("   ", model, include_buffers=show_buffers, color=color)
    children_root = list(model.named_children())
    for i, (child_name, child) in enumerate(children_root):
        last = (i == len(children_root) - 1)
        rec(child, f"{root_name}.{child_name}", 1, "", last)
    print("\nTotal parameters:", human_count(total_p), " | Total buffers:", human_count(total_b))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total_p - trainable
    print("Trainable:", human_count(trainable), " | Frozen:", human_count(frozen))
