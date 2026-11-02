<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Multi-GPU quantization

`CUDA_VISIBLE_DEVICES` controls which GPUs a process can see. It does not by
itself enable parallel quantization. For a MoE model such as DeepSeek, select a
balanced MoE device pool in `QuantizeConfig`. On a free-threaded Python build,
set `PYTHON_GIL=0` before starting Python to enable parallel expert
quantization.

For example, to use physical GPUs 0 and 1:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHON_GIL=0 python quantize_deepseek.py
```

Inside `quantize_deepseek.py`, configure the pool using the logical CUDA
indices visible to that process (`cuda:0` and `cuda:1` in this example):

```python
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization.config import VramStrategy

model_id = "MODEL_REPO_OR_LOCAL_PATH"
quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    device="cuda:0",
    moe_vram_strategy=VramStrategy.BALANCED,
    moe_vram_strategy_devices=["cuda:0", "cuda:1"],
)

model = GPTQModel.load(model_id, quant_config, dtype="bfloat16")
model.quantize(calibration_dataset)
model.save("DeepSeek-GPTQ-4bit")
```

Provide `calibration_dataset` using the normal `model.quantize()` input format.
The dense model modules remain on the default exclusive strategy; balanced
MoE placement distributes expert work across the selected devices. The Python
runtime must be a free-threaded build and the GIL must actually be disabled;
`PYTHON_GIL=0` cannot make a regular Python build free-threaded. GPT-QModel logs
the visible dense/MoE device pools and whether parallel MoE quantization is
eligible when quantization starts.

For a dense model, `moe_vram_strategy` does not apply. To distribute dense
module work, configure `dense_vram_strategy` and
`dense_vram_strategy_devices` instead. Loading an already-quantized model
across GPUs is a separate operation and can use `device_map="auto"`; that
controls inference placement, not quantization parallelism.

## GPU peer access (P2P)

Multi-GPU quantization moves quantized tensors between devices. GPT-QModel
probes and enables peer access automatically, at most once per device set, before
`clone_module_for_devices()`, from the weight-only looper, and for multi-device
`device_map` loads. Enablement goes through `hipDeviceEnablePeerAccess` on ROCm
and `cuCtxEnablePeerAccess` on CUDA; PyTorch has no wrapper for it.

Each directed pair is classified after a measured copy:

| State | Meaning | Transfer path |
|---|---|---|
| `enabled` | capability true and bandwidth above the gate | direct device-to-device copy |
| `unsupported` | capability false or enablement failed | host staging |
| `degraded` | capability true but bandwidth below the gate | forced host staging |

Configuration:

| Variable | Default | Meaning |
|---|---|---|
| `GPTQMODEL_P2P` | `auto` | `0` disables P2P, `1` forces the direct path and bypasses the relative gate |
| `GPTQMODEL_P2P_MIN_GBPS` | `5.0` | absolute floor; below it the link is `degraded` |
| `GPTQMODEL_P2P_MIN_RATIO` | `0.9` | direct copy must retain this fraction of pinned host staging, else `degraded` |
| `GPTQMODEL_P2P_PROBE_MIB` | `64` | preferred probe copy size (steps down to 32/16/4 under memory pressure) |

The ratio is deliberately below 1.0. On a plain PCIe topology the traffic crosses
the same root complex whether it is a peer copy or a pinned-host staged copy, so
a healthy direct link measures at parity: 53.1 GB/s direct vs 55.1 GB/s staged
on a 4x R9700 (`gfx1201`) pair and 26.2 vs 26.2 GB/s on a 2x RTX 4090 pair, i.e.
0.96x and 1.00x (64 MiB payload, double-synchronized timer, best of two
samples). A 1.0 gate classified every one of those links as `degraded` and forced
all traffic through host staging, which is the same PCIe path plus an extra
pinned-buffer hop. A 0.9 floor still catches a genuinely broken or oversubscribed
link (the tests pin a 0.5x link as `degraded`) while keeping a working peer link
on the direct path.

Measurements on this host with the default `GPTQMODEL_P2P=auto` (both directions
reported `enabled`): R9700 2<->3 at 52.5-53.1 GB/s (0.962-0.970 of staging),
RTX 4090 0<->1 at 26.2 GB/s (0.999-1.000). Enabling peer access is what makes
PyTorch's cross-device `copy_`/`.to()` use the direct path; `copy_tensor_to`
always issues a single `dst.copy_(src)` and never inserts a host relay (the
driver stages internally when a link is off).

A `strict=False` failure never raises; one informational summary line is
logged. Measure with `scripts/benchmark_multi_gpu_p2p.py`.

### RTX 4090 (consumer Ada) note

Consumer cards frequently report `can_device_access_peer == True` while direct
throughput is limited by PCIe. They are not special-cased by `sm_89`: every pair
is measured. A pair that clears the 0.9 ratio keeps the direct path; one that
does not is marked `degraded` and falls back to host staging.
`can_device_access_peer` alone is never treated as proof that P2P is in effect.

P2P and NCCL are different mechanisms. The `gloo`/`nccl` distributed path used by
GSQ (`gptqmodel/quantization/gsq_distributed.py`) is independent; verify it with
`tests/test_gsq_distributed.py` (gated on `is_nccl_available()`) in addition to
the P2P tests in `tests/test_p2p.py`.
