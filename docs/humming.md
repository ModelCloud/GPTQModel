# Humming W2/W4/W8 CUDA GEMM backend

## Summary

`BACKEND.HUMMING` serves GPTQ and AWQ checkpoints with
[Humming](https://github.com/vllm-project/humming), the JIT-compiled weight-only
CUDA GEMM family (W1–W8, dense and MoE) that
[vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/humming.py)
and [SGLang](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/humming.py)
already consume through the optional `humming-kernels` PyPI package.

The port is deliberately thin, because Humming owns the packed layouts. GPT-QModel
keeps the checkpoint tensors registered under their usual GPTQ/AWQ names,
`post_init()` converts them once through Humming's own schema objects, and
`forward()` calls `humming_forward()`. No CUDA source is vendored into
`gptqmodel_ext`, and `humming` is only imported inside
`gptqmodel/utils/humming.py`.

## Where the code lives

| Path | Role |
| --- | --- |
| `gptqmodel/utils/backend.py` | `BACKEND.HUMMING` plus the `gptq_humming` / `awq_humming` legacy aliases |
| `gptqmodel/utils/humming.py` | Import guard, device/capability gates, schema conversion, locks, compute config |
| `gptqmodel/nn_modules/qlinear/humming.py` | `HummingLinear` (GPTQ) and `AwqHummingLinear` (AWQ) sub-kernels |
| `tests/kernels/test_humming.py` | Registration plus numeric checks against the dequantized reference |

## Opt-in, never AUTO

Both classes declare `SUPPORTS_FORMATS` with priority `0`, so
`importer.build_kernel_support_maps()` registers them as explicit candidates
while leaving `BACKEND.AUTO` on the existing default kernels. Humming
JIT-compiles one kernel image per (shape, dtype) pair on first use, which is a
poor default for a first model load:

```python
from gptqmodel import GPTQModel

model = GPTQModel.load("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", backend="humming")
```

`"gptq_humming"` and `"awq_humming"` still normalize to `BACKEND.HUMMING`, and
the method/format of the checkpoint selects the sub-kernel.

## Supported checkpoints

| Method | Format | Bits | Group sizes | `sym` | `desc_act` |
| --- | --- | --- | --- | --- | --- |
| GPTQ | `gptq`, `gptq_v2` | 2, 4, 8 | -1, 32, 64, 128 | both | must be `False` |
| AWQ | `gemm` | 4 | -1, 32, 64, 128 | both | accepted and ignored |

The bit ranges are layout limits, not preferences. GPT-QModel stores 3-bit GPTQ
with the `10-1-10-1-10` word scheme and 5/6/7-bit GPTQ in split planes; Humming
reads one continuous little-endian packing, so those checkpoints are rejected at
construction. AWQ only keeps the classic interleaved INT4 layout, which is the
single layout Humming's AWQ converter undoes.

`in_features % group_size` must be `0` for grouped checkpoints. Any `in_features`
/ `out_features` works otherwise: the layer keeps its logical shape and the
packed operands are padded to Humming's 256-wide N and 128-wide K tiles during
conversion.

## Runtime requirements

* NVIDIA SM75 (Turing) or newer; there is no HIP port, so ROCm hosts reject the
  backend even when the Python package imports.
* Linux, CUDA 12.x/13.x, `torch.float16` or `torch.bfloat16` compute.
* `pip install humming-kernels` (Apache-2.0).

When the package is missing, `validate_once()` returns the import error instead
of failing the whole `gptqmodel` import, so kernel selection reports a normal
"unsupported backend" rejection with the install hint.

## Conversion details

GPTQ (`HummingLinear.post_init`):

* `qweight.transpose(-1, -2)` becomes Humming's `weight`.
* `scales.transpose(-1, -2)` becomes `weight_scale`.
* `sym=False` forwards `qzeros.transpose(-1, -2)` as `zero_point`; the logical
  (v2) zero points are used, so a v1 checkpoint is shifted field-by-field first.
* `sym=True` drops `qzeros`, matching Humming's symmetric schema.

AWQ (`AwqHummingLinear.post_init`): the classic interleaved `qweight`, `scales`
and `qzeros` are passed through the `awq` schema, and `qzeros` is forwarded
whenever the checkpoint has one. Without it Humming switches to offset-binary
codes, which only matches checkpoints whose codes are already signed.

Both classes register a `locks` buffer (2048 `int32`, `persistent=False`): the
Stream-K scratch that Humming synchronizes on and that must not be shared
between layers.

The per-call compute policy mirrors the vLLM/SGLang compute-config JSON.
`humming_compute_config()` returns the dense/fp32 default that layers use, so
inference numerics never depend on ambient process state:

```python
humming_compute_config()                                  # dense, fp32 accum
humming_compute_config(batch_invariant=True)              # pinned split-K order
humming_compute_config(f16_accum=True)                    # fp16 MMA accum
```

With the default `use_batch_invariant=False`, humming's Stream-K reduction may
sum partial tiles in a different order on each call, so two identical forwards
can differ by a few 1e-2 on logits. Pass `batch_invariant=True` (and forward the
returned JSON through `humming_apply_linear(..., compute_config=...)`) when you
need bit-reproducible runs, for example when diffing kernels; it maps to vLLM's
`VLLM_BATCH_INVARIANT` and SGLang's fixed dense compute config.

## Limits

* Dense linear layers only. Humming's MoE/indexed GEMM is not wired into the
  GPT-QModel module tree yet.
* `post_init()` replaces the checkpoint tensors with Humming's transformed
  operands, so a loaded layer is inference-only, like Marlin and Swordfish.
  Reload from the checkpoint (or repack with a checkpoint-format backend) before
  saving; `gptqmodel.utils.packing` rejects Humming as a repack source for the
  same reason.

## Verification

```bash
pytest tests/kernels/test_humming.py -q          # numeric + registration
pytest tests/test_backend_naming.py -q           # aliases / method-agnostic naming
pytest tests/kernels/test_selection.py -q        # AUTO stays on the default kernels
```

The numeric cases compare `HummingLinear`/`AwqHummingLinear` against an fp32
dequantized matmul for every supported bit width, both group-size extremes,
symmetric and asymmetric GPTQ zero points, tile-misaligned shapes, empty
activations and dtype casts. A separate case pins the batch-invariant policy to
bit-exact repeated forwards.
