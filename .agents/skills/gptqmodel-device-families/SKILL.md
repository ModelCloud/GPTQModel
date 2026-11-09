---
name: gptqmodel-device-families
description: Keep NVIDIA and AMD support explicit when editing GPTQModel device selection, model loading, placement, hardware checks, or quant-linear SUPPORTS declarations.
---

# Explicit GPU families

GPTQModel's support-family enum is the hardware contract. PyTorch's device
namespace is the execution API. Keep those concepts separate.

| Context | NVIDIA | AMD ROCm |
| --- | --- | --- |
| GPTQModel support family | `DEVICE.CUDA` | `DEVICE.ROCM` |
| Torch runtime device | `torch.device("cuda:N")` | `torch.device("cuda:N")` under HIP |
| Runtime identification | CUDA build, no HIP | `torch.version.hip is not None` / centralized `IS_ROCM` |

`torch.cuda.is_available()`, `tensor.is_cuda`, and `device.type == "cuda"`
identify the runtime API, not NVIDIA hardware. Do not use them alone to choose
an NVIDIA kernel. Fold the HIP distinction into the family resolver and use
explicit families in GPTQModel logic.

## Declarations and selection

- Each concrete kernel owns its `SUPPORTS_DEVICES` hardware declaration.
  `[DEVICE.CUDA]` declares NVIDIA only; `[DEVICE.CUDA, DEVICE.ROCM]` declares
  both. `[DEVICE.ALL]` is appropriate for genuinely portable implementations.
  Keep declarations static; do not rewrite `DEVICE.CUDA` to ROCm in a support
  list based on the host. Consumers normalize requests, not declarations.
- Preserve `DEVICE.ROCM` in family-valued configuration such as
  `quantize_config.device`. Do not store `DEVICE.CUDA` just to satisfy a Torch
  API that rejects the string `"rocm"`.
- Use `models._const.normalize_device()` for a family-only decision. On HIP it
  maps runtime `"cuda"`, `"cuda:N"`, ordinal and Torch-device inputs to ROCm.
  For backward compatibility it also normalizes a `DEVICE.CUDA` *request* on
  HIP; new AMD-facing callers should request `DEVICE.ROCM` explicitly.
- Use the selector helpers in `utils/importer.py` when a concrete device or a
  device map is needed. Preserve ordinals for placement, capability checks and
  validation-cache keys. Do not replace `cuda:3` with a family and reconstruct
  it later as device 0. Never infer a family with `DEVICE(tensor.device.type)`.
- Keep `SUPPORTS_BITS`, group sizes, dtypes, formats, shape constraints and
  platform declarations consistent with each supported implementation. A family
  match is necessary but insufficient: validate architecture restrictions on
  every selected device. For example, AllSpark HIP supports RDNA4 gfx1200/1201;
  a supported current GPU does not make a selected gfx942 target supported.
- Derive AMD architecture restrictions from `gcnArchName`/the architecture
  helpers. HIP's `get_device_capability()` tuple is not an NVIDIA SM version.
- Do not expand ROCm support solely because code uses Torch or Triton. Check
  native extensions, inline PTX/CUTLASS, architecture-specific Triton paths,
  dependencies and numerical evidence. An unvalidated backend may remain
  NVIDIA-only even if some primitives appear portable.

## Torch boundaries and mixed GPU systems

Use `utils.torch.as_torch_device()` when passing a family enum to Torch tensor
creation, moves, streams, or other execution APIs. It maps `DEVICE.ROCM` to the
HIP runtime's `cuda` spelling and leaves families indexless so the current
device context remains effective. Explicit Torch devices retain their ordinal.
`DEVICE.to_torch_device()` intentionally defaults accelerator families to
ordinal 0; use it only when that default is explicitly desired. Family-wide
thread-pool locks must cover all devices of that family, not just GPU 0.

A machine may contain NVIDIA and AMD GPUs, but its installed Torch build and
visibility settings determine the devices addressed by that process's
`cuda:N`. Do not infer the active vendor from the physical host or treat the
same ordinal as identifying the same physical GPU across CUDA and HIP builds.
`DEVICE.ROCM` does not create a Torch `rocm:N` namespace or let a CUDA-only
Torch build execute on AMD. Keep the intended family explicit, resolve actual
placement against the active runtime, and reject unsupported placement instead
of silently treating one vendor as another. Static kernel-support inspection
may inspect either family without executing on it.

For hardware decisions, prefer:

```python
family = normalize_device(tensor.device)
if family == DEVICE.ROCM:
    # AMD-specific implementation or architecture validation
    ...
```

For placement, preserve the target separately:

```python
target = torch.device("cuda:1")  # runtime ordinal, not a vendor declaration
family = normalize_device(target)
# Select a kernel that declares family; retain target for placement.
module = move_to(module, target)
```

## Validation

Update `tests/test_rocm_kernel_contracts.py` when adding or changing a kernel's
hardware contract. Its audited inventory covers every registered quant-linear
and the role-only embedding kernel. Exercise HIP aliases, explicit ROCm,
NVIDIA behavior, concrete ordinals, and architecture rejection where relevant.
Mocks establish dispatch behavior; they do not establish kernel correctness.

For newly enabled inference paths, follow
`../gptqmodel-torch-oracle-numerics/SKILL.md`, run on actual target hardware,
and report exact hardware/runtime and any remaining gaps. Preserve packed
weight storage across repeated forwards and validate supported activation
dtypes against an independent oracle.
