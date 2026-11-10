# Tenstorrent (TT-Metal / PJRT-XLA) support

Status: **investigation + design (validated on hardware 2026-11-10)**.
Inference path proof-of-concept works; no upstream code changes are required for
the first milestone. Quantization-on-device is a later milestone with one
blocking bug identified below.

## Target hardware

| Item | Value |
| --- | --- |
| Host | `tt-quietbox`, ASRock B850M-C |
| Accelerators | 4x Tenstorrent Blackhole **p300c** (`1e52:b140`), PCIe Gen4 x4 |
| Firmware | 19.17.1.0 (cm 0.39.1.0, eth 1.13.1, dm_app 0.33.1.0, gddr 2.17) |
| Driver | tt-kmd 2.12.0 |
| OS / kernel | Ubuntu 26.04.1, `7.0.0-43-generic` |
| Stack | `pjrt-plugin-tt` 1.5.0.dev20260831000501, torch 2.11.0+cpu, torch-xla 2.9.0, jax 0.7.1 |
| Python | CPython 3.12.16 (pyenv) |
| SFPI | 7.69.0 (must match the plugin's bundled tt-metal) |

## How PyTorch sees the device

TT cards appear through **PJRT** as the PyTorch/XLA `xla` device namespace:

```python
from torch_xla import device as xd
d = xd()                      # torch.device("xla:0")
t = torch.ones(4, 4, device=d)
t.device.type                 # "xla"   <-- not "cuda", not "xpu", not "npu"
t.is_xla                      # True
```

There is no `torch.xla` module and no `xla` entry in `gptqmodel.models._const.DEVICE`.
This is the central integration gap: **GPTQModel has no `DEVICE.XLA` family**, so
every selector, support declaration and family resolver misses the device.

### Mapping onto GPTQModel concepts

| Context | Value on TT |
| --- | --- |
| GPTQModel support family | *does not exist yet* — needs `DEVICE.XLA` |
| Torch runtime device | `torch.device("xla:N")` |
| Runtime identification | `tensor.device.type == "xla"` |
| Multi-device | `xla:0..xla:3` are the 4 p300c cards |

Following `.agents/skills/gptqmodel-device-families/SKILL.md`, TT must be its own
explicit family. It must **not** be folded into `DEVICE.CUDA` or `DEVICE.ALL`
projections, and kernels must declare it explicitly.

## What already works (measured)

`TorchLinear` declares `SUPPORTS_DEVICES = [DEVICE.ALL]`, which is the correct
declaration for a portable implementation, so the kernel gate already admits TT.
Verified on hardware:

- `TorchLinear` constructed, packed 4-bit GPTQ weights uploaded, `.to(xla)` ok
- `forward()` returns a correctly shaped `xla` tensor: `(4, 128)` fp16, no error
- Dequantized-weight matmul path lowered and executed
- Primitives that work on TT: `bitwise_and`, integer→float casts, `matmul`,
  `mm`, `bmm`, `addmm` (uniform dtype), `einsum`, `gather`, `index_select`,
  `argmax`, `topk`, `scatter`, `where`, `amax`, `reshape`/`permute`/`cat`
- `matmul` fp16/bfloat16/fp32 all correct against CPU oracle

So **inference of a pre-quantized GPTQ model is reachable today** with an explicit
`xla` device selection; the quantized weights stay packed in the module.

## Blockers found (all reproduced)

### B1 — int32 bitwise shift is numerically wrong on TT  *[blocking, inference]*

TT lowers int32/int64 right-shift through **f32**, so shifts are exact only for
values below 2^24. GPTQ packs 8 nibbles per int32, so the high nibbles are
corrupted:

```
q = 0x12345678
CPU  q >> 0  = 305419896
TT   q >> 0  = 305419904     <-- 8 off; low 3 bits lost
```

int64 does not help — it is lowered identically. This corrupts
`_right_shift_unpack()` in `gptqmodel/nn_modules/qlinear/torch.py` and therefore
`dequantize_weight()` (which returns a graph that then fails with
`RuntimeError: Error code: 13`).

**Workaround (validated):** split the int32 word on the host into `uint8` bytes,
move those to the device, then extract nibbles with `uint8` mask/shift. `uint8`
fits in f32 exactly, so this is bit-exact:

```
xla low nibble : [8, 6, 4, 2, 15, 15, 15, 15]
cpu low nibble : [8, 6, 4, 2, 15, 15, 15, 15]   # exact match
```

### B2 — `out_dtype=` ATen overloads are unimplemented on TT

```python
torch.mm(x, w, out_dtype=torch.float32)
# NotImplementedError: Could not run 'aten::mm.dtype' with arguments from the 'CPU' backend
```

`matmul_fp32_reduce()` uses this for the wide-reduction path. It is currently
gated to `_WIDE_REDUCE_DEVICES = ("cuda",)`, so xla is **not** affected today —
but any future code that adds `"xla"` there will break. The portable equivalent
`torch.mm(x.float(), w.float()).to(dtype)` works.

### B3 — `addmm` with mixed bias dtype fails to lower

```python
torch.addmm(bias_fp32, x_fp16, w_fp16)
# RuntimeError: Check failed: status.ok(): HLO -> MHLO conversion failed.
```

Uniform-dtype `addmm` works. Use `mm(...) + bias` instead of promoting only the
bias.

### B4 — host readback needs explicit sync

Reading an `xla` tensor that came from a long op chain intermittently fails with
`Error code: 13`. `torch_xla.sync(wait=True)` before `.cpu()`, or `.item()`,
returns cleanly. This is a readback/ordering issue, not a numerics issue.

### B5 — fp16 is silently rounded to bf16 on transfer  *[critical, numerics]*

TT-Metal has **no native fp16**; the PJRT plugin rounds fp16 operands to bf16:

```python
v = torch.tensor([0.501953125], dtype=torch.float16)
v.to(dev).cpu()          # 0.5                     <-- 1 bf16 ULP lost
v.to(torch.bfloat16)     # 0.5                     <-- same rounding
```

Confirmed separately: an fp16 round-trip on device always equals
`v.to(bfloat16).to(float16)`. This affects scales, biases and any fp16 weight.

**Consequence:** every TT path must compute and store in **bfloat16**, not
float16. `TorchLinear.SUPPORTS_DTYPES` must additionally admit bf16 for TT
(it already lists `torch.bfloat16`, so the gate is satisfied), and any fp16
checkpoint must be declared unsupported for TT.

### B6 — `torch.stack` over more than ~2 inputs fails to lower  *[blocking, unpack]*

Building the 8-nibble unpack with a single `torch.stack([...8 slices...])`
raises `Error code: 13`. Stacking **two** tensors and using
`reshape(...).permute(...).reshape(...)` works. The validated sequence is:

```python
pair = torch.stack([lo, hi], dim=-1)      # (K/8, N, 4, 2)
nibbles = pair.reshape(K//8, N, 8).permute(0, 2, 1).reshape(K, N)
```

## Design

### Milestone 1 — inference of pre-quantized models  *[VALIDATED]*

Validated on hardware 2026-11-10 with a synthetic 4-bit GPTQ group-128 layer
(K=256, N=128), bfloat16, against an independent CPU oracle:

```
DEQUANT exact: True                      # bit-identical to CPU unpack+dequant
M1 bf16 matmul pcc=1.000000 maxdiff=0.125
```

Method that works today, with no upstream changes:

1. Load the model with the portable `TorchLinear` kernel
   (`SUPPORTS_DEVICES = [DEVICE.ALL]`, so the selector already admits TT).
2. Place it on the `xla` device explicitly.
3. Use **bfloat16**, never float16 (B5).
4. Replace the nibble unpack with the exact uint8 sequence (B1) and avoid
   `TorchLinear._dequantize_weight_cached_248`, whose
   `_right_shift_unpack` is wrong on TT.
5. `torch_xla.sync(wait=True)` before any host readback (B4).

### Milestone 2 — first-class `DEVICE.XLA` family (upstream changes)

1. `gptqmodel/models/_const.py`
   - add `XLA = "xla"` to `DEVICE`
   - `type` property already returns `.value`, so `"xla"` maps correctly
   - add `XLA` to `index` so it resolves to `xla:0` by default
2. `gptqmodel/utils/torch.py`
   - add `HAS_XLA` detection (`import torch_xla` succeeded and
     `torch_xla.device()` works)
3. `gptqmodel/utils/importer.py`
   - `auto_select_device()`: add `elif HAS_XLA: device = DEVICE.XLA`
   - `expand_selector_device_family()`: include `"xla"` in the indexless-type
     set and expand to `xla:0..N`
4. `gptqmodel/utils/arch.py`
   - TT has no CUDA-style compute capability; return `None` and make callers
     fall back to portable behavior
5. Tests
   - extend `tests/test_rocm_kernel_contracts.py` inventory with the XLA family
   - add a device-family resolution test (mocked) **plus** an actual-hardware
     numerical test, per `gptqmodel-torch-oracle-numerics`

### Milestone 3 — fix the unpack for TT

Scope: `_right_shift_unpack()` in `qlinear/torch.py`, which already has an NPU
special case, so the pattern is established:

```python
def _right_shift_unpack(values, shifts, dtype):
    if values.device.type == "xla":
        # int32>>N is lowered via f32 on TT and loses low bits.
        # Split to uint8 on host, move, then mask/shift (exact in f32).
        ...
    if values.device.type != "npu":
        return torch.bitwise_right_shift(values, shifts).to(dtype)
    ...
```

Must preserve packed storage: the split bytes are a *host-side* unpack of the
already-packed int32; no expanded weight cache may be retained across forwards
(see AGENTS.md "Quantized inference weight storage").

### Milestone 4 — on-device quantization (calibration/gradient)

Lower priority. Requires several ops TT does not yet lower (mixed-dtype `addmm`,
`out_dtype` overloads). Start by running the GPTQ calibration loop on CPU with
the quantized model on TT, then measure.

### Milestone 5 — native TT-NN kernel (performance)

`DEVICE.ALL` portable path is correct but slow (it dequantizes to fp16 and does
a full GEMM). A native GPTQ kernel would dequantize into bounded tiles and use
TT-NN matmul with sharding across the 4 cards. Must satisfy the persistent
packed-storage rule. This is the real performance milestone and should only be
attempted after M1-M3 are numerically locked.

## Validation plan

For every admitted path, per `gptqmodel-torch-oracle-numerics`:

- independent CPU oracle on identical packed weights
- report PCC and max-abs-diff for fp16 and bf16 activations
- repeat forwards to prove packed weights are unchanged
- record exact hardware: 4x p300c, fw 19.17.1, kmd 2.12.0, plugin 1.5.0.dev20260831000501
- assert no expanded weight cache is retained after the forward

## Environment notes

- Do **not** set `PYTHONPATH` at the tt-metal source tree. The
  `pjrt-plugin-tt` wheel ships a self-consistent tt-metal/ttnn; shadowing it
  breaks kernel JIT (`hostdev/profiler_common.h` missing) and produces a
  SFPI ABI mismatch.
- SFPI must match the plugin's bundled tt-metal (7.69.0 for this plugin build).
  A newer SFPI (7.96) fails to compile LLK kernels.
- `click` must stay at the plugin's pin (`8.3.1`); installing other packages that
  bump it breaks the plugin.
