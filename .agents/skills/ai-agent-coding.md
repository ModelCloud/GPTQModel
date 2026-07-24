---
name: ai-agent-coding
description: Code documentation conventions for agents editing GPT-QModel. Use when adding or modifying classes, dataclasses, public methods, module-level globals, static variables, or function parameters, so the next agent can understand the code quickly without noisy prose.
---

# AI Agent Coding Conventions

Agents iterate code quickly. Short, targeted comments on *what* a symbol represents, not *how* it is implemented, save future sessions from reading the implementation just to infer the contract.

## 1. New classes and dataclasses

Add a one-line docstring or inline comment describing the abstraction and its invariants. Keep it shorter than one paragraph.

```python
class FusedGroupForward:
    """Coordinate one GEMM for same-input linear groups and slice outputs per member."""
```

For dataclasses, document fields whose meaning is not obvious from the name or type:

```python
@dataclass
class FusedForwardConfig:
    # How to return per-member output slices: "view" avoids copies, "contiguous_copy" keeps downstream .view() safe.
    splice: str = "view"
```

## 2. Module-level globals and constants

Every module-level `ALL_CAPS` or mutable default should have a brief comment:

```python
_NOT_QUANTIZE_FLAG = ":!"  # Suffix marker in module_tree names for capture-only layers.
```

Group related globals together with a short section header when there are several.

## 3. Static and class variables

Document class-level state that controls behavior or is shared across instances:

```python
class GPTQProcessor(LoopProcessor):
    # Minimum samples required before the Hessian solve is considered stable.
    MIN_HESSIAN_SAMPLES: int = 32
```

## 4. Method/function parameters

For non-trivial public methods, comment only parameters whose name is ambiguous or whose default carries a design decision:

```python
def install_fused_group_forward(
    layer_module: nn.Module,
    layer_modules_blocks: List[List[str]],
    enabled: bool = True,
    splice: str = "view",  # output slicing strategy; default view avoids contiguous() copies
    logger=None,
) -> int:
    """Attach FusedGroupForward helpers to same-input module groups in a decoder layer."""
```

Do not add a docstring just to repeat the type hints.

## 5. New attributes on instances

When a class gains a new field (especially one mutated across lifecycle stages), add a short comment at assignment or declaration:

```python
self.fused_weight_storage: Optional[torch.Tensor] = None  # shared buffer lazily allocated on first forward
```

## 6. What not to comment

- Do not narrate the diff ("now we check X", "added for bug Y").
- Do not restate the code in English unless the intent is non-obvious.
- Avoid inline comments on every line; prefer clear variable names.

## 7. Quick checklist before committing agent-authored code

- [ ] New class/dataclass has a one-line docstring.
- [ ] New module globals/constants have a brief comment.
- [ ] New class/static variables have a brief comment.
- [ ] New method parameters with non-obvious defaults are annotated inline or in the docstring.
- [ ] New instance attributes set in `__init__` are documented.
- [ ] No diff-only or verbose prose comments were added.
