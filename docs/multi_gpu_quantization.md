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
| `GPTQMODEL_P2P` | `auto` | `0` disables P2P, `1` forces an enablement attempt |
| `GPTQMODEL_P2P_MIN_GBPS` | `5.0` | absolute degraded threshold |
| `GPTQMODEL_P2P_MIN_RATIO` | `1.0` | direct copy must be this multiple of pinned host staging, else `degraded` |
| `GPTQMODEL_P2P_PROBE_MIB` | `16` | probe copy size |

The ratio gate matters in practice: measured on 2x RTX 4090 and 4x R9700
(gfx1201) with a 128 MiB payload, `can_device_access_peer` was `True` in both
directions while the direct copy ran at 0.954x-0.998x of pinned host staging
(about 47 GB/s direct vs 55 GB/s staged on gfx1201, 26 GB/s vs 26 GB/s on the
4090s). An absolute floor alone (and certainly the capability bit) would have
declared those links usable; they are `degraded` and host-stage instead. P2P
on/off throughput is then effectively identical (54.25 vs 54.29 GB/s on gfx1201,
26.2568 vs 26.2549 GB/s on the 4090s), which is the expected result of the
fallback.

A `strict=False` failure never raises; one informational summary line is
logged. Measure with `scripts/benchmark_multi_gpu_p2p.py`.

### RTX 4090 (consumer Ada) note

Consumer cards frequently report `can_device_access_peer == True` while direct
throughput is poor. They are therefore not special-cased by `sm_89`: every pair
is measured and slow or asymmetric links are marked `degraded` and fall back to
host staging. `can_device_access_peer` alone is never treated as proof that P2P
is in effect.

P2P and NCCL are different mechanisms. The `gloo`/`nccl` distributed path used by
GSQ (`gptqmodel/quantization/gsq_distributed.py`) is independent; verify it with
`tests/test_gsq_distributed.py` (gated on `is_nccl_available()`) in addition to
the P2P tests in `tests/test_p2p.py`.
