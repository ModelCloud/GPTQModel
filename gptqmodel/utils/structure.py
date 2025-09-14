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


def print_module_tree(
    model: nn.Module,
    *,
    root_name: str = "model",
    max_depth: Optional[int] = None,
    filter_regex: Optional[str] = None,
    show_params: bool = False,
    show_buffers: bool = False,
    color: bool = True,
    # MoE collapsing controls:
    collapse_experts: bool = True,
    experts_regex: str = r"(^|\.)experts($|\.)",
    experts_show: int = 1,
):
    re.compile(filter_regex) if filter_regex else None
    experts_name_re = re.compile(experts_regex) if collapse_experts else None
    seen: Set[int] = set()

    total_p = sum(p.numel() for p in model.parameters())
    total_b = sum(b.numel() for b in model.buffers())

    def should_collapse(qual_name: str, container: nn.Module) -> bool:
        if not experts_name_re:
            return False
        if not experts_name_re.search(qual_name):
            return False
        if not isinstance(container, (nn.ModuleList, nn.Sequential)):
            return False
        # must have numerically indexed children
        names = [n for n, _ in container.named_children()]
        if not names:
            return False
        return all(n.isdigit() for n in names) and len(names) > max(0, experts_show)

    def rec(mod: nn.Module, name: str, depth: int, prefix: str, is_last: bool):
        full_name = name
        if max_depth is not None and depth > max_depth:
            return

        mod_id = id(mod)
        shared = ""
        if mod_id in seen:
            shared = "  ↩ shared ref"
        else:
            seen.add(mod_id)

        trunk = "└─ " if is_last else "├─ "
        print(_format_line(prefix, trunk, full_name, mod, show_counts=True, color=color) + shared)
        if shared:
            return

        if show_params or show_buffers:
            _print_params(prefix + ("   " if is_last else "│  "), mod, include_buffers=show_buffers, color=color)

        # Children
        children = list(mod.named_children())
        n = len(children)
        for i, (child_name, child) in enumerate(children):
            last = (i == n - 1)
            child_prefix = prefix + ("   " if is_last else "│  ")
            display_name = f"{full_name}.{child_name}" if full_name else child_name

            # Collapsing logic for MoE-like ModuleLists
            if should_collapse(display_name, child):
                # Print the container node itself
                print(_format_line(child_prefix, "└─ " if last else "├─ ", display_name, child, True, color))

                # Materialize its numeric children
                sub_children = list(child.named_children())
                total_k = len(sub_children)
                k_show = max(0, min(experts_show, total_k))

                # Show first K experts fully
                for j, (sub_name, sub_mod) in enumerate(sub_children[:k_show]):
                    sub_last = (j == k_show - 1) and (k_show == total_k)  # if showing all, it's last
                    sub_prefix = child_prefix + ("   " if last else "│  ")
                    sub_trunk = "└─ " if sub_last else "├─ "
                    print(_format_line(sub_prefix, sub_trunk, f"{display_name}.{sub_name}", sub_mod, True, color))
                    # descend into the exemplar(s)
                    rec(
                        sub_mod,
                        f"{display_name}.{sub_name}",
                        depth + 2,
                        child_prefix + ("   " if last else "│  "),
                        sub_last,
                    )

                # Collapsed summary for the remainder
                if k_show < total_k:
                    # Estimate per-expert params/buffers from the first exemplar
                    p_one, b_one = _param_summary(sub_children[0][1], recurse=True)
                    p_rest = (total_k - k_show) * p_one
                    b_rest = (total_k - k_show) * b_one

                    if color:
                        DIM, RESET = "\033[2m", "\033[0m"
                    else:
                        DIM = RESET = ""

                    summary_prefix = child_prefix + ("   " if last else "│  ")
                    # Use a single summary line
                    print(
                        f"{summary_prefix}{DIM}• … collapsed (repeats {k_show}..{total_k-1}, "
                        f"per-expert P={human_count(p_one)} B={human_count(b_one)}, "
                        f"collapsed total P≈{human_count(p_rest)} B≈{human_count(b_rest)}){RESET}"
                    )
                continue  # handled this container; do not regular-rec descend

            # Normal recursion
            rec(child, display_name, depth + 1, child_prefix, last)

    # Root
    root_prefix = ""
    print(_format_line(root_prefix, "", root_name, model, show_counts=True, color=color))
    if show_params or show_buffers:
        _print_params("   ", model, include_buffers=show_buffers, color=color)

    # Descend from root
    children = list(model.named_children())
    for i, (child_name, child) in enumerate(children):
        last = (i == len(children) - 1)
        rec(child, f"{root_name}.{child_name}", 1, "", last)

    # Footer
    print("\nTotal parameters:", human_count(total_p), " | Total buffers:", human_count(total_b))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total_p - trainable
    print("Trainable:", human_count(trainable), " | Frozen:", human_count(frozen))
