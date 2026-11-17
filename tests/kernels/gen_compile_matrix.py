# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regenerate docs/kernel-compile-matrix.md from the kernel declarations."""

from pathlib import Path

from gptqmodel.nn_modules.qlinear._compile_support import describe_kernel
from gptqmodel.utils import importer


HEADER = """<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Kernel compile and graph-capture matrix

Every quantized kernel declares two capabilities in its own class body, so
callers never guess from a class name:

- `SUPPORTS_COMPILE_LEVEL` - the highest `torch.compile` level its `forward`
  reaches, measured with `tests/kernels/compile_probe.py`:

  | Level | Probe |
  | --- | --- |
  | `NOT_APPLICABLE` | not a torch module (MLX/TTNN/XLA); its own compiler owns the graph |
  | `EAGER` | eager forward only; dynamo breaks the graph (or compile is a measured pessimization) |
  | `DYNAMO` | `torch.compile(fullgraph=True, backend="eager")` traces with zero graph breaks |
  | `AOT_EAGER` | `backend="aot_eager"` (meta/fake tensors and `aot_autograd`) |
  | `INDUCTOR` | `backend="inductor"` lowers the forward |
  | `CUDAGRAPH` | `mode="reduce-overhead"` captures the compiled forward as a CUDA graph |

- `SUPPORTS_GRAPH_CAPTURE` - whether the **eager** `forward` is safe inside an
  explicit `torch.cuda.graph(...)` capture: no host synchronisation, no
  tensor-to-host transfer, no Python-side allocation or device decision, no
  data-dependent branch. Independent of dynamo: a custom op without a fake
  implementation can be capture-safe yet untraceable.

The levels are the `torch.compile` API surface of torch 2.13+ (the project's
floor is 2.13 for Python < 3.11 and 2.15 otherwise): `backend="eager"`,
`backend="aot_eager"`, `backend="inductor"`, and `mode="reduce-overhead"`.
`max-autotune` is a tuning mode of `INDUCTOR`, not a separate capability, so it
is not a level. A declaration records the torch version it was measured on.

`SUPPORTS_COMPILE` remains the separate #4669 policy answer to "is compiling
this forward not a pessimization for static-cache `generate()`". A traceable
kernel can still keep it `False` when eager is faster.

Two MLX kernels (`AwqMlxQuantLinear`, `AWQMarlinMlxQuantLinear`) inherit
`SUPPORTS_COMPILE = True` from `AwqTorchLinear`. That flag is not consulted on
the MLX runtime (the model is not a `torch.nn.Module`), so their
`torch.compile` level is still `NOT_APPLICABLE`; the column shows the inherited
policy flag for completeness.

`tests/kernels/test_kernel_compile_protocol.py` enforces that every registered
kernel declares both attributes, that a capture claim has no host-sync
construct in its own `forward`, and that this table matches the code.
Regenerate with:

```bash
python -m tests.kernels.gen_compile_matrix
```

| Kernel | Module | Level | Graph capture | Compiled generation |
| --- | --- | --- | --- | --- |
"""


def render() -> str:
    rows = []
    for cls in sorted(importer._registry_kernel_classes(), key=lambda cls: cls.__name__):
        row = describe_kernel(cls)
        module = row["module"].replace("gptqmodel.nn_modules.qlinear.", "")
        rows.append(
            f"| `{row['kernel']}` | `{module}` | `{row['compile_level']}` | "
            f"{'yes' if row['graph_capture'] else 'no'} | "
            f"{'yes' if row['compiled_generation'] else 'no'} |"
        )
    return HEADER + "\n".join(rows) + "\n"


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[2] / "docs" / "kernel-compile-matrix.md"
    target.write_text(render(), encoding="utf-8")
    print(f"wrote {target}")
