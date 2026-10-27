# Stable `hf_` packing API

External integrators (transformers/optimum, auto-round, ...) need a stable way
to ask which kernels can pack a quantization contract, which packing format to
target, and how to pack or repack a single layer. GPT-QModel exposes those
entry points in `gptqmodel.utils.packing`; every function carries the `hf_`
prefix and is covered by `tests/test_hf_packing_api.py`.

Anything without the `hf_` prefix (for example `pack_module`,
`select_quant_linear`) is internal and may change without notice.

## API surface

| Function | Purpose |
| --- | --- |
| `hf_check_packing_feasibility(...)` | Does a kernel exist that can pack (or repack) this contract on the target device? |
| `hf_check_best_packing_format(...)` | Which packing format has the fastest feasible kernel? |
| `hf_pack_layer(module, linear, scales, zeros, g_idx, ...)` | Pack one layer into an already-created quantized kernel. |
| `module.hf_pack(linear, scales, zeros, g_idx, ...)` | Method-level wrapper around `hf_pack_layer` for callers holding a kernel instance. |
| `hf_post_init(module)` | Run layer-level `post_init()` on a quantized layer or every quantized child of a container, including layers accelerate has offloaded. |
| `hf_repack_layer(module, backend, ...)` | Repack an already-quantized layer for another kernel. |

## Checking feasibility and picking a format

```python
from gptqmodel.utils.packing import (
    hf_check_best_packing_format,
    hf_check_packing_feasibility,
)

feasible = hf_check_packing_feasibility(
    bits=4,
    group_size=128,
    desc_act=False,
    sym=False,
    backend="marlin",
    device="cuda:0",
)

best_format = hf_check_best_packing_format(
    bits=4,
    group_size=128,
    desc_act=False,
    sym=False,
    device="cuda:0",
)
```

`hf_check_packing_feasibility` accepts `in_features`/`out_features` to include
layer shape constraints, and `dynamic` to mirror per-module overrides. Invalid
argument values (unknown format/method/backend) raise `ValueError`, while an
unsupported contract simply returns `False`.

Pass `repack=True` to probe repack targets instead of raw-weight packers.
Kernels such as Marlin and ExllamaV2 have no `pack()` implementation — they
receive GPTQ-packed tensors and convert them in `post_init()` — so they answer
`True` for `repack=True` and `False` for the default packing probe.

## Packing a layer

`hf_pack_layer` follows the internal model packing path. `scales` and `zeros`
use the quantizer-native layout `[out_features, num_groups]`. For symmetric
kernels `zeros` may be omitted and `2 ** (bits - 1)` is synthesized, while
asymmetric (`sym=False`) packing requires explicit zero points.

```python
import torch
from gptqmodel.utils.importer import hf_select_quant_linear_v2
from gptqmodel.utils.packing import hf_pack_layer

kernel_cls = hf_select_quant_linear_v2(
    bits=4,
    group_size=128,
    desc_act=False,
    sym=True,
    format="gptq_v2",
    quant_method="gptq",
    backend="auto",
    device_map="cpu",
)
module = kernel_cls(
    bits=4,
    group_size=128,
    desc_act=False,
    sym=True,
    in_features=linear.in_features,
    out_features=linear.out_features,
    bias=linear.bias is not None,
)

hf_pack_layer(
    module, linear, scales, zeros=None, g_idx=g_idx,
    pack_impl="original", checkpoint_format="gptq_v2",
)
```

`pack_impl` accepts `"original"`, `"block"`/`"cpu"` and `"gpu"`; block/GPU
requests fall back to the original packer when the kernel or device does not
support them. Pass `post_init=True` to run the kernel's `post_init()` after
packing, and `post_init_kwargs={...}` to forward arguments such as the
ExllamaV2 scratch space.

Packing is a CPU-storage operation: even `pack_impl="gpu"` registers the packed
buffers on CPU (the GPU is used only as a compute accelerator), matching the
internal model packing path. Move the module to its runtime device afterwards.

## Repacking a layer

`hf_repack_layer` converts an already-packed module to another backend. Kernels
with a native `repack_from_gptq`/`repack_from_awq` hook (BitBLAS) use it;
otherwise the packed tensors are copied into a target kernel that accepts the
same checkpoint format and `post_init()` performs the layout conversion.

```python
from gptqmodel.utils.packing import hf_repack_layer

marlin_module = hf_repack_layer(gptq_module, backend="marlin", device="cuda:0")
```

### Post-initializing offloaded layers

`transformers` dispatches the model with accelerate *before* the quantizer runs
`post_init()`. For a dict `device_map` that places layers on `cpu`/`disk`, the
packed tensors are on the meta device at that point and accelerate has already
recorded a pre-init snapshot in the layer's `weights_map`. Kernels that resize
or repack their buffers in `post_init()` (Marlin clears `g_idx`/`qzeros` and
rewrites `qweight`/`scales`) would otherwise leave a stale snapshot, and the
first forward fails with a shape error such as:

```text
ValueError: Trying to set a tensor of shape torch.Size([4096]) in "g_idx"
(which has shape torch.Size([0])), this looks incorrect.
```

`gptqmodel_post_init`, `hf_post_init` and `hf_repack_layer` therefore
materialize an offloaded layer for the duration of its `post_init()`, then
refresh the accelerate offload map and the on-disk `index.json` with the
resulting packed tensors before restoring the meta placeholders. Only packed
storage is written; no dense or dequantized copy is retained. This happens
automatically — no extra call is needed — and applies to both `cpu` and `disk`
entries in a dict `device_map`, including hooks attached to an ancestor module.

If the offload map has no writable place to store a refreshed tensor — for
example a `safetensors` index that points straight at the checkpoint, or an
entry for a tensor the layer no longer owns — the call raises instead of
returning stale weights, because continuing would fail on the next forward or
silently compute with pre-`post_init` data.

Kernels whose `post_init()` needs arguments (ExllamaV2 scratch space) are
repacked with `post_init=False` and initialized through `hf_post_init`:

```python
from gptqmodel.utils.exllamav2 import ScratchSpace
from gptqmodel.utils.packing import hf_post_init, hf_repack_layer

target = hf_repack_layer(source, backend="exllama_v2", device="cuda:1", post_init=False)
hf_post_init(target, scratch_space=ScratchSpace(target.temp_dq_size(), dev="cuda:1"))
```

When the requested backend is already in use, the source module is returned
unchanged. Targets that cannot consume the source layout raise
`NotImplementedError`, and unsupported backends raise `ValueError`.

Two guarantees keep copy-based repacking numerically correct:

* GPTQ v1/v2 zero-point flavors are aligned automatically. Kernels with
  `REQUIRES_FORMAT_V2` keep qzeros in the corrected v2 domain, so a copy into a
  v1-native target (Marlin, BitBLAS) runs the same conversion the loader/saver
  uses instead of copying offset zero points.
* Kernels that repack their buffers into a private runtime ABI inside
  `post_init()` (Marlin, Machete, Swordfish, BitBLAS) are valid repack targets
  but are rejected as repack *sources*, because their buffers no longer hold
  the checkpoint layout. Repack from the kernel holding the checkpoint layout
  (for example a Torch kernel) instead.
