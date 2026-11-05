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

PyTorch already enables peer access lazily inside its own copy path
(`c10/cuda/PeerToPeerAccess.cpp` -> `CUDACachingAllocator::enablePeerAccess`), so
cross-device `copy_`/`.to()` stays on the peer link without any help from
GPT-QModel. GPT-QModel additionally pre-enables the links before the first
cross-device transfer, so the first copy does not pay the setup cost, at most
once per device set: before `clone_module_for_devices()`, from the weight-only
looper, and for multi-device `device_map` loads. Enablement goes through
`hipDeviceEnablePeerAccess` on ROCm and `cuCtxEnablePeerAccess` on CUDA; PyTorch
has no enable wrapper of its own.

Peer links are **directed**. GPT-QModel checks `src -> dst` for every
cross-device move it issues itself; when that direction is unsupported (or
measured `degraded`), the move is staged through pinned host memory instead of
calling the peer copy.
On some ROCm topologies the matrix is asymmetric (for example `0 -> 1`
unsupported while `1 -> 0` works), and a direct `tensor.to("cuda:1")` segfaults
inside `hipMemcpyPeerAsync` rather than raising. That case is logged, and
NCCL/torch replicate is skipped in favor of deepcopy plus host staging. A
healthy symmetric link still uses the direct device-to-device path.

An enablement call that reports success for a direction the runtime capability
query denies is reported as `unverified` and is never used for a direct copy or
measured by the probe (the probe itself performs a direct copy). Enablement is
still kept for PyTorch's own paths; GPT-QModel moves in that direction are
staged through host memory.

Inference is **not** covered by this routing. Loading the quantized model with
`device_map="auto"` installs accelerate's `AlignDevicesHook`, whose
`send_to_device` moves activations with a plain `Tensor.to`; on an asymmetric
ROCm topology that still segfaults (exit 139) inside `hipMemcpyPeerAsync`.
Measured on 2x R9700 (`gfx1201`, ROCm 7.16.26385): quantization completes, then
`generate()` on the same two devices crashes from
`accelerate.hooks.AlignDevicesHook.pre_forward -> send_to_device`. Until that
move is routed through `safe_move_to`, run inference on a single device on such
a topology.

Configuration:

| Variable | Default | Meaning |
|---|---|---|
| `GPTQMODEL_P2P` | `auto` | `0` skips GPT-QModel's pre-enablement (PyTorch still peers lazily); `1` also attempts pairs the capability bit denies, but a denied direction stays host-staged (`unverified`) even if enablement reports success |
| `GPTQMODEL_P2P_PROBE` | unset | `1` measures every directed pair; `degraded` directions are host-staged |
| `GPTQMODEL_P2P_MIN_GBPS` | `5.0` | absolute floor; below it a direction is `degraded` (host-staged) |
| `GPTQMODEL_P2P_MIN_RATIO` | `1.0` | direct copy must reach this fraction of ordered pinned-host staging, else `degraded` |
| `GPTQMODEL_P2P_PROBE_MIB` | `64` | preferred probe copy size (steps down to 32/16/4 under memory pressure) |

### Bandwidth diagnostics (`GPTQMODEL_P2P_PROBE=1`)

Probing moves real payload across every directed pair and synchronizes each
sample, so it is **opt-in** and never runs during normal quantization. Its
verdict selects the transfer path: a `degraded` direction is staged through
pinned host memory, exactly like an unsupported one.

The reference is an explicit, correctly ordered `src -> pinned CPU -> dst`
pipeline. It pays for two transfers (D2H then H2D), so it is strictly slower
than a direct copy at the same link rate; the legs must be ordered, because
issuing them on two device streams without a dependency measures the staged path
at roughly twice its real speed. Measured with ordered legs (64 MiB,
double-synchronized timer, median of 7 samples):

| pair | direct GB/s | ordered staged GB/s | ratio |
|---|---|---|---|
| 2x R9700 (`gfx1201`) | 52.99 | 27.35 | 1.94 |
| 2x RTX 4090 (`sm_89`) | 26.23 | 13.27 | 1.98 |

A `1.0` gate therefore keeps every healthy PCIe peer link direct while still
flagging a link that is genuinely slower than the fallback. Lower
`GPTQMODEL_P2P_MIN_RATIO` if a topology really does measure at parity.

A `strict=False` failure never raises; one informational summary line is
logged. Measure with `scripts/benchmark_multi_gpu_p2p.py` (its `host` column is
the non-peer reference; `pre_enable` is already peer-enabled by PyTorch after
the warm-up, so it is not an "off" baseline).

### RTX 4090 (consumer Ada) note

Consumer cards frequently report `can_device_access_peer == True` while direct
throughput is limited by PCIe. They are not special-cased by `sm_89`: with the
probe enabled every pair is measured, and the measurement is the verdict.
`can_device_access_peer` alone is never treated as proof that P2P is effective.

P2P and NCCL are different mechanisms. The `gloo`/`nccl` distributed path used by
GSQ (`gptqmodel/quantization/gsq_distributed.py`) is independent; verify it with
`tests/test_gsq_distributed.py` (gated on `is_nccl_available()`) in addition to
the P2P tests in `tests/test_p2p.py`.

## Allocator backend for very wide layers

GPTQ builds a dense `columns x columns` fp32 Hessian per projection, and inverts
it with a Cholesky factor plus a cuSOLVER workspace. A 40960-wide projection
(`mlp.down_proj` on a 65B model) therefore needs roughly **three 6.25 GiB
buffers at once**. Data-parallel accumulation makes it worse: each participating
device holds a full partial Hessian, and `materialize_global_hessian` allocates a
separate accumulator on the target device.

The native CUDA caching allocator can fail that single multi-GiB request even
when the driver reports >13 GiB free, because it cannot assemble one contiguous
block next to the resident partials. The failure surfaces either as a plain OOM
or, with `expandable_segments`, as `CUDA driver error: out of memory ...
from cuMemSetAccess`. GPTQModel logs a hint when it falls back to CPU for this
reason.

Use the async allocator for these runs:

```bash
PYTORCH_ALLOC_CONF=backend:cudaMallocAsync python quantize.py
```

Measured on 4x RTX 4090 with a 65B model, `offload_to_disk=True` and
`auto_forward_data_parallel=True`:

| allocator | result |
| --- | --- |
| `native` (default) | OOM on the first wide `mlp.down_proj` |
| `native` + `expandable_segments` | still fails, now in `cuMemSetAccess` |
| `backend:cudaMallocAsync` | 80/80 layers, no OOM, `reserved` flat across layers |

`cudaMallocAsync` does not change numerics: quantizing the same checkpoint on
four devices produced **237/237 bit-identical tensors** compared with the native
backend.

Caveats:

* The backend is chosen when the process first touches CUDA. Set the variable in
  the environment, before Python starts; it cannot be switched at runtime.
* `torch.cuda.memory_snapshot()` and `torch.cuda.memory._record_memory_history()`
  are unsupported on this backend and raise. `memory_allocated`,
  `memory_reserved`, `memory_stats`, `memory_summary`, `mem_get_info` and
  `empty_cache()` all keep working.
* `expandable_segments` is a native-allocator feature and is ignored here.
