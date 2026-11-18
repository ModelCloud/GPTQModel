# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regenerate docs/kernel-compile-matrix.md from the kernel declarations."""

from pathlib import Path

from gptqmodel.nn_modules.qlinear._compile_support import describe_kernel
from tests.kernels.compile_probe import protocol_kernels


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

- `SUPPORTS_COMPILE_LEVEL_BY_DEVICE` - optional per-device override, keyed by
  `torch.device.type`. The "Level" column is the default-device declaration;
  the "CPU level" column is what a CPU-resident instance declares. A kernel
  that runs on both devices records the CPU ceiling here when it differs (for
  example `TorchLinear`: CUDA `EAGER` because its streaming path allocates a
  `torch.cuda.Stream`, CPU `INDUCTOR`). Only the dict in the kernel's own class
  body applies, so an MLX or Triton subclass never inherits its torch parent's
  CPU answer.

A class declaration answers for the kernel's default runtime state. Where one
kernel class serves several stored formats, an instance may refine it through
`supports_graph_capture()`, and `graph_capture_safe(instance)` returns that
answer. `GGUFTorchLinear` is the one case today: its device-side dequantizers
(`Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K` - including the default `q_k_m`) are
capturable, while `Q2_K`, `Q3_K` and the remaining codebook types dequantize
through numpy on the host and cannot be captured (#5014).

The levels are the `torch.compile` API surface of torch 2.13+ (the project's
floor is 2.13 for Python < 3.11 and 2.15 otherwise): `backend="eager"`,
`backend="aot_eager"`, `backend="inductor"`, and `mode="reduce-overhead"`.
`max-autotune` is a tuning mode of `INDUCTOR`, not a separate capability, so it
is not a level. A declaration records the torch version it was measured on.
The CPU levels in this table were re-measured on
`torch 2.16.0.dev20261021+cpu`; the default-device levels keep their recorded
GPU measurements.

`SUPPORTS_COMPILE` remains the separate #4669 policy answer to "is compiling
this forward not a pessimization for static-cache `generate()`". A traceable
kernel can still keep it `False` when eager is faster, but then it has to
record the measured loss in `SUPPORTS_COMPILE_EAGER_REASON` (the "Policy
reason" column): a level at or above `INDUCTOR` is the documented bar for a
compiled win, so the two declarations cannot drift apart silently (#4984).

Two MLX kernels (`AwqMlxQuantLinear`, `AWQMarlinMlxQuantLinear`) inherit
`SUPPORTS_COMPILE = True` from `AwqTorchLinear`. That flag is not consulted on
the MLX runtime (the model is not a `torch.nn.Module`), so their
`torch.compile` level is still `NOT_APPLICABLE`; the column shows the inherited
policy flag for completeness.

`tests/kernels/test_kernel_compile_protocol.py` enforces that every registered
kernel declares both attributes, that a capture claim has no host-sync
construct in its own `forward`, that a compile-capable opt-out records its
reason, and that this table matches the code.
Regenerate with:

```bash
python -m tests.kernels.gen_compile_matrix
```

| Kernel | Module | Level | CPU level | Graph capture | Compiled generation | Policy reason |
| --- | --- | --- | --- | --- | --- | --- |
"""


def render() -> str:
    rows = []
    for cls in protocol_kernels():
        row = describe_kernel(cls)
        module = row["module"].replace("gptqmodel.nn_modules.qlinear.", "")
        reason = (row["compiled_generation_reason"] or "-").replace("|", "\\|")
        rows.append(
            f"| `{row['kernel']}` | `{module}` | `{row['compile_level']}` | "
            f"`{row['compile_level_cpu']}` | "
            f"{'yes' if row['graph_capture'] else 'no'} | "
            f"{'yes' if row['compiled_generation'] else 'no'} | {reason} |"
        )
    return HEADER + "\n".join(rows) + "\n"


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[2] / "docs" / "kernel-compile-matrix.md"
    target.write_text(render(), encoding="utf-8")
    print(f"wrote {target}")
