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
"""

from typing import Dict, Iterable, Optional, Set, Tuple

import pcre as re
import torch
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
    experts_name_re = re.compile(experts_regex) if collapse_experts else None
    seen: Set[int] = set()

    total_p = sum(p.numel() for p in model.parameters())
    total_b = sum(b.numel() for b in model.buffers())  # fixed loop variable

    def should_collapse(qual_name: str, container: nn.Module) -> bool:
        if not experts_name_re:
            return False
        if not experts_name_re.search(qual_name):
            return False
        if not isinstance(container, (nn.ModuleList, nn.Sequential)):
            return False
        names = [n for n, _ in container.named_children()]
        if not names:
            return False
        return all(n.isdigit() for n in names) and len(names) > max(0, experts_show)

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

        children = list(mod.named_children())
        n = len(children)
        for i, (child_name, child) in enumerate(children):
            last = (i == n - 1)
            child_prefix = prefix + ("   " if is_last else "│  ")
            display_name = f"{name}.{child_name}" if name else child_name

            if should_collapse(display_name, child):
                line2 = _format_line(child_prefix, "└─ " if last else "├─ ",
                                     display_name, child, True, depth+1)
                annot2 = colorize_annotation(_annotate(child, color=color))
                print(line2 + " " + annot2)

                sub_children = list(child.named_children())
                total_k = len(sub_children)
                k_show = max(0, min(experts_show, total_k))

                for j, (sub_name, sub_mod) in enumerate(sub_children[:k_show]):
                    sub_last = (j == k_show - 1) and (k_show == total_k)
                    sub_prefix = child_prefix + ("   " if last else "│  ")
                    sub_trunk = "└─ " if sub_last else "├─ "
                    line3 = _format_line(sub_prefix, sub_trunk,
                                         f"{display_name}.{sub_name}",
                                         sub_mod, True, depth+2)
                    annot3 = colorize_annotation(_annotate(sub_mod, color=color))
                    print(line3 + " " + annot3)
                    rec(
                        sub_mod,
                        f"{display_name}.{sub_name}",
                        depth + 2,
                        child_prefix + ("   " if last else "│  "),
                        sub_last,
                    )

                if k_show < total_k and total_k > 0:
                    p_one, b_one = _param_summary(sub_children[0][1], recurse=True)
                    collapsed = (
                        f"• … collapsed (repeats {k_show}..{total_k-1}, "
                        f"per-expert P={human_count(p_one)} B={human_count(b_one)})"
                    )
                    print(_maybe(child_prefix + ("   " if last else "│  ") + collapsed, DIM, color=color))
                continue

            rec(child, display_name, depth + 1, child_prefix, last)

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

def alias_from_turtle_for_submodule(
    target_model: torch.nn.Module,
    turtle_model: torch.nn.Module,
    target_submodule: torch.nn.Module,
    device: torch.device,
    non_blocking: bool = False,
) -> torch.nn.Module:
    assert device not in [None, torch.device("cpu"), torch.device("meta")]
    # print(f"alias device = {device}")

    # Resolve path & source submodule (on CPU/mmap)
    path = _get_qualified_name(target_model, target_submodule)
    src_map = dict(turtle_model.named_modules())
    if path not in src_map:
        raise KeyError(f"Path '{path}' not found in turtle_model.")
    src_sub = src_map[path]

    # ---- copy params/buffers CPU->GPU into target_submodule (your existing code) ----
    t_params = dict(target_submodule.named_parameters(recurse=True))
    s_params = dict(src_sub.named_parameters(recurse=True))
    with torch.inference_mode():
        for name, s_p in s_params.items():
            t_p = t_params.get(name)
            if t_p is None or t_p.shape != s_p.shape:
                continue
            t_p_new = _ensure_target_storage_on_device_(t_p, device)
            if t_p_new is not t_p:
                parent, leaf = _get_parent_and_leaf_by_path(target_submodule, name)
                setattr(parent, leaf, t_p_new)
                t_p = t_p_new
            t_p.detach().copy_(s_p.detach(), non_blocking=(non_blocking and s_p.is_pinned()))

    t_bufs = dict(target_submodule.named_buffers(recurse=True))
    s_bufs = dict(src_sub.named_buffers(recurse=True))
    for name, s_b in s_bufs.items():
        tb = t_bufs.get(name)
        parent, leaf = _get_parent_and_leaf_by_path(target_submodule, name)
        if tb is None or getattr(tb, "is_meta", False) or tb.device.type == "meta":
            new_b = torch.empty_like(s_b, device=device)
            new_b.copy_(s_b.detach(), non_blocking=(non_blocking and s_b.is_pinned()))
            parent.register_buffer(leaf, new_b, persistent=True)
        else:
            if tb.device != device:
                new_tb = torch.empty_like(s_b, device=device)
                new_tb.copy_(s_b.detach(), non_blocking=(non_blocking and s_b.is_pinned()))
                parent.register_buffer(leaf, new_tb, persistent=True)
            else:
                tb.copy_(s_b.detach(), non_blocking=(non_blocking and s_b.is_pinned()))

    if hasattr(target_model, "tie_weights"):
        target_model.tie_weights()

    #print("Post alias: target_submodule device summary:")
    # for n, p in target_submodule.named_parameters(recurse=True):
        # print(f"  {n}: {p.device}")

    # return the *target* submodule, which is the injected result
    return target_submodule

def _is_meta_tensor(t: torch.Tensor) -> bool:
    return bool(getattr(t, "is_meta", False)) or (hasattr(t, "device") and t.device.type == "meta")

def _module_all_meta(mod: nn.Module) -> bool:
    """True if the module has at least one tensor and *all* its params/buffers are meta."""
    saw_any = False
    for _, p in mod.named_parameters(recurse=False):
        saw_any = True
        if not _is_meta_tensor(p):
            return False
    for _, b in mod.named_buffers(recurse=False):
        saw_any = True
        if not _is_meta_tensor(b):
            return False
    return saw_any  # modules with no tensors aren't considered 'meta' targets

def _is_leaf(mod: nn.Module) -> bool:
    return next(mod.named_children(), None) is None

def alias_all_from_turtle_if_meta(
    shell_model: nn.Module,
    turtle_model: nn.Module,
    *,
    require_class_match: bool = True,
    verify_shapes: bool = True,
    tie_after: bool = True,
) -> int:
    """
    Replace (alias) leaf submodules in `shell_model` with the corresponding submodules
    from `turtle_model` when the shell submodule's tensors are on meta.

    Logs each swap via log.info().
    """
    if turtle_model is None:
        return 0

    turtle_map = dict(turtle_model.named_modules())
    swapped = 0

    for qname, shell_sub in list(shell_model.named_modules()):
        if not qname:  # skip root
            continue
        if not _is_leaf(shell_sub):
            continue
        if not _module_all_meta(shell_sub):
            continue

        turtle_sub = turtle_map.get(qname, None)
        if turtle_sub is None:
            # log.info(f"Module: Skipped {qname}: not found in turtle model")
            continue

        if require_class_match and (shell_sub.__class__ is not turtle_sub.__class__):
            # log.info(
            #     f"Module: Skipped {qname}: class mismatch "
            #     f"(shell={shell_sub.__class__.__name__}, turtle={turtle_sub.__class__.__name__})"
            # )
            continue

        if verify_shapes:
            shell_ps = dict(shell_sub.named_parameters(recurse=False))
            turtle_ps = dict(turtle_sub.named_parameters(recurse=False))
            for n in set(shell_ps.keys()) & set(turtle_ps.keys()):
                if shell_ps[n].shape != turtle_ps[n].shape:
                    # log.info(
                    #     f"Module: Skipped {qname}: parameter shape mismatch at '{n}' "
                    #     f"(shell={tuple(shell_ps[n].shape)}, turtle={tuple(turtle_ps[n].shape)})"
                    # )
                    break
            else:
                shell_bs = dict(shell_sub.named_buffers(recurse=False))
                turtle_bs = dict(turtle_sub.named_buffers(recurse=False))
                for n in set(shell_bs.keys()) & set(turtle_bs.keys()):
                    if shell_bs[n].shape != turtle_bs[n].shape:
                        # log.info(
                        #     f"Module: Skipped {qname}: buffer shape mismatch at '{n}' "
                        #     f"(shell={tuple(shell_bs[n].shape)}, turtle={tuple(turtle_bs[n].shape)})"
                        # )
                        break
                else:
                    parent, leaf = _get_parent_and_leaf_by_path(shell_model, qname)
                    setattr(parent, leaf, turtle_sub)
                    swapped += 1
                    log.info(f"Module: Sync {qname} <- from turtle ({turtle_sub.__class__.__name__})")
                    continue
            continue

        parent, leaf = _get_parent_and_leaf_by_path(shell_model, qname)
        setattr(parent, leaf, turtle_sub)
        swapped += 1
        log.info(f"Module:: Sync {qname} <- from turtle ({turtle_sub.__class__.__name__})")

    if tie_after and hasattr(shell_model, "tie_weights") and getattr(shell_model.config, "tie_word_embeddings", False):
        try:
            shell_model.tie_weights()
            log.info("Module: Re-tied embedding weights on shell model after full sync")
        except Exception as e:
            log.info(f"Module: tie_weights failed: {e}")

    log.info(f"Module: Total synced modules: {swapped}")
    return swapped
